# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Physical-page PDF Sources for offline financial-report evidence retrieval.

A finance report only stays checkable while retrieval can point back into the
original document. This module therefore splits a *complete* PDF at its physical
page boundary: page 1 is the first physical page of the file, a page without
extractable text is still a page, and every page keeps the identity of the whole
document it came from.

Three views stay separate, matching the other captured Sources:

* the PDF bytes are the evidence. The digest of those bytes names the revision,
  so a revised PDF produces fresh page identities while citations captured from
  the old revision keep resolving to the revision they were read from;
* one page keeps exactly this page's text, so a table digit never drifts away
  from the page it was printed on and no other page's text is mixed in;
* retrieval ranks the captured pages of one document collection through the
  public Memory API and returns only citations the Memory store can resolve: a
  citation is never manufactured here.

``search_pdf_pages`` reads the scoped Memory, ranks its pages with an offline
lexical model, and returns the best distinct pages. The model is BM25 over
stemmed terms with prefix conflation and finance vocabulary; adjacent
content-word pairs, the share of the question a page covers, and the store's
own full-text ranking add independent signals, and the strongest pages feed
their agreed wording back as extra terms. Layout priors then apply the way a
finance report actually reads: a quantified question is answered on a primary
statement page, and statements and notes sit next to each other, so the
neighbours of a strong page are themselves plausible evidence.

The lexical index of the captured pages is derived from the pages alone, never
from the question, so it is cached for the life of the process: one collection
pays for its tokenisation once and every later question only rebinds the current
citations and the current full-text priors onto the cached index.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from pydantic import BaseModel

from powercontext.errors import InvalidSourceReferenceError
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

if TYPE_CHECKING:
    from collections.abc import Sequence

    from powercontext.artifacts import MemoryCitation

PDF_PAGE_SOURCE_NAME = "pdf-page"
PDF_PAGE_SOURCE_VERSION = "1"
PDF_PAGE_SOURCE_ID_PREFIX = "pdf"
PDF_PAGE_DIGEST_ID_PREFIX = "pdf-digest"
PDF_PAGE_DEFAULT_SEARCH_LIMIT = 10

# Lexical retrieval is deliberately offline and dependency free: the shared
# environment locks pypdf only, so ranking must not need any other library.
_TOKEN_PATTERN = re.compile(r"[a-z]+|\d+")
_PAGE_MARKER = "#p"
_EOF_MARKER = b"%%EOF"
_EOF_WINDOW = 8192
# A conforming writer ends the file with the marker, optionally followed by
# whitespace or the zero padding some tools append, and nothing else.
_INERT_BYTES = b"\x00\t\n\f\r "
_XREF_REFERENCE = re.compile(rb"startxref\s+(\d+)")
_XREF_SECTION_HEAD = re.compile(rb"xref\b|\d+\s+\d+\s*obj\b")
_TRUNCATED_MESSAGE = "PDF document is truncated and cannot be parsed completely"
_PAGE_SHORTFALL_MESSAGE = "PDF document declares {declared} pages but only {readable} can be read"
_PREFIX_LENGTH = 6
_PREFIX_WEIGHT = 0.75
_PAIR_SHIFT = 20
_PAIR_WEIGHT = 0.4
_PAIR_CAP = 3
_BM25_K1 = 1.2
_BM25_B = 0.75
_COVERAGE_WEIGHT = 0.6
_DOCUMENT_AFFINITY = 0.25
_EXPANSION_WEIGHT = 0.7
_FEEDBACK_DOCUMENTS = 4
_FEEDBACK_TERMS = 8
_FEEDBACK_MIN_SUPPORT = 2
_FEEDBACK_WEIGHT = 0.3
_FEEDBACK_MAX_SHARE = 3
_STATEMENT_BOOST = 0.3
_SEED_PAGES = 6
_NEIGHBOUR_DECAY = {1: 0.6, 2: 0.4}
_MAX_SEARCH_LIMIT = 1_000
_FTS_CANDIDATES = 50
_FTS_PRIOR_WEIGHT = 0.3
# Page indexes kept for reuse. One 503-page report costs about 15 MB of index, so
# the bound is small: a scope is searched through one revision set at a time, and
# a second slot covers an ingest landing between two questions.
_PAGE_INDEX_CACHE_SIZE = 2

_STOPWORD_TEXT = """
    a about above after again against all also am an and any are as at be because been before being below between both
    but by can could did do does doing done down during each few for from further had has have having he her here hers
    him his how i if in into is it its itself just me more most my no nor not now of off on once only or other our out
    over own same she should so some such than that the their theirs them then there these they this those through to
    too under until up very was we were what when where which while who whom why will with would you your
    annual company fiscal form including may report shall table year years contents
"""

_STOPWORDS = frozenset(_STOPWORD_TEXT.split())

# Words that make a question ask for a number. A financial report answers those
# on its primary statements far more often than on a narrative page.
_NUMERIC_TEXT = """
    amount capex capital cash cost costs debt depreciation ebitda expense expenses figure figures growth income increase
    many margin much number per percent percentage price quantity rate ratio revenue sales share total turnover value
"""

_NUMERIC_INTENT = frozenset(_NUMERIC_TEXT.split())

# Generic finance-report vocabulary. A key is a lowercase term prefix that a
# question token reaches through ``str.startswith``, so a question word and the
# report's own wording meet without a document-specific dictionary: nothing here
# names a company, a document or an answer. Matching on a fixed-width slice of
# the token instead would silently make every key shorter than that width
# unreachable, which is a dead rule nobody notices.
_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "geogra": ("geographic", "region", "country", "international", "division", "segment"),
    "region": ("geographic", "country"),
    "countr": ("geographic", "region", "international"),
    "legal": ("litigation", "proceeding", "lawsuit", "claim", "regulatory"),
    "litiga": ("legal", "proceeding", "lawsuit", "claim"),
    "battl": ("litigation", "proceeding", "lawsuit", "claim"),
    "restru": ("restructuring", "impairment", "severance", "productivity"),
    "ebitda": ("operating", "income", "depreciation", "amortization"),
    "operat": ("operating", "income", "result"),
    "amorti": ("depreciation",),
    "deprec": ("amortization",),
    "revenu": ("sales", "net"),
    "employ": ("employee", "personnel", "headcount"),
    "divid": ("dividend", "share", "shareholder"),
    "borrow": ("debt", "note", "loan", "commercial"),
    "pensio": ("pension", "benefit", "retirement"),
    "acquis": ("acquisition", "acquire", "business"),
    "shareh": ("shareholder", "share", "stock"),
    "govern": ("director", "executive", "board", "committee"),
    "risk": ("risk", "uncertainty", "exposure"),
}

# Longest key first, so the most specific vocabulary wins when several prefixes
# of one token are keys and the answer never depends on table order.
_EXPANSION_KEYS: tuple[str, ...] = tuple(sorted(_EXPANSIONS, key=lambda key: (-len(key), key)))


class PDFDocumentInput(BaseModel):
    """One complete offline PDF document captured as bytes."""

    document_id: str
    content: bytes


class PDFPageInput(BaseModel):
    """One physical page identified by its document, revision and page number."""

    document_id: str
    document_sha256: str
    page_number: int
    text: str


class PDFPageSource(Source):
    """Captured single-page PDF evidence that can be cited and re-read."""

    source_id: str
    document_id: str
    document_sha256: str
    page_number: int
    text: str


def pdf_page_source_id(document_id: str, document_sha256: str, page_number: int) -> str:
    """Build the delimiter-safe, bounded identity of one physical PDF page.

    Identity is exactly ``(document_id, document_sha256, page_number)``: two
    pages of one revision never share a page number, and two revisions keep
    disjoint identity spaces because the digest of the whole file is part of
    every name. Components are percent-encoded independently, so neither a
    document identifier nor a digest can introduce the ``:`` that separates the
    segments.

    An unusually long document identifier cannot fit into the shared identity
    limit; its own digest then stands in for it under a different prefix, so the
    two forms can never collide.
    """

    document = _require_document_id(document_id)
    digest = _require_document_sha256(document_sha256)
    page = _require_page_number(page_number)
    encoded = quote(digest, safe="")
    identity = f"{PDF_PAGE_SOURCE_ID_PREFIX}:{quote(document, safe='')}:{encoded}{_PAGE_MARKER}{page}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        hashed = hashlib.sha256(document.encode("utf-8")).hexdigest()
        identity = f"{PDF_PAGE_DIGEST_ID_PREFIX}:{hashed}:{encoded}{_PAGE_MARKER}{page}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise InvalidSourceReferenceError(
            "source_id",
            f"must not exceed {MAX_SOURCE_ID_LENGTH} characters",
        )
    return identity


def parse_pdf_document(value: PDFDocumentInput) -> tuple[PDFPageInput, ...]:
    """Parse every physical page of one PDF document in file order.

    The whole document is read before a single page is returned, so an empty,
    non-PDF, corrupt or partially readable file raises :class:`ValueError`
    instead of handing back a silently truncated report. A page with no
    extractable text stays in the result with empty text: it is still a page.
    """

    document_id = _require_document_id(value.document_id)
    content = value.content
    if not isinstance(content, (bytes, bytearray)):
        raise ValueError("PDF document content must be bytes")
    raw = bytes(content)
    if not raw:
        raise ValueError("PDF document must not be empty")
    document_sha256 = hashlib.sha256(raw).hexdigest()
    return tuple(
        PDFPageInput(
            document_id=document_id,
            document_sha256=document_sha256,
            page_number=page_number,
            text=text,
        )
        for page_number, text in enumerate(_extract_page_texts(raw), start=1)
    )


class PDFPageSourceAdapter:
    """Resolve and read the built-in physical-page PDF Source."""

    input_class = PDFPageInput
    name = PDF_PAGE_SOURCE_NAME
    source_class = PDFPageSource

    async def resolve(self, value: PDFPageInput, /) -> PDFPageSource:
        source_id = pdf_page_source_id(value.document_id, value.document_sha256, value.page_number)
        return PDFPageSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            document_sha256=value.document_sha256,
            page_number=value.page_number,
            text=value.text,
        )

    async def read(self, source: PDFPageSource, /) -> PDFPageInput:
        return PDFPageInput(
            document_id=source.document_id,
            document_sha256=source.document_sha256,
            page_number=source.page_number,
            text=source.text,
        )


class PDFPageTextEvidenceProjection:
    """Expose one physical page as canonical textual evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PDFPageSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PDFPageSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PDF_PAGE_SOURCE_NAME,
            source_id=source.source_id,
            content=f"{source.document_id}\n{source.text}",
            metadata={
                "document_id": source.document_id,
                "document_sha256": source.document_sha256,
                "page_number": source.page_number,
            },
        )


PDF_PAGE_SOURCE_ADAPTER = PDFPageSourceAdapter()
PDF_PAGE_SOURCE_DEFINITION: SourceDefinition[PDFPageInput, PDFPageSource, PDFPageInput] = AdapterSourceDefinition(
    PDF_PAGE_SOURCE_ADAPTER,
    version=PDF_PAGE_SOURCE_VERSION,
    projections=(PDFPageTextEvidenceProjection(),),
)


async def search_pdf_pages(
    memory: Any, question: str, limit: int = PDF_PAGE_DEFAULT_SEARCH_LIMIT
) -> tuple[MemoryCitation, ...]:
    """Return up to ``limit`` distinct evidence pages for one original question.

    Only the already-open scoped Memory is read: the caller restricts the data
    range by handing over that Memory. Every returned citation is an entry the
    store itself produced, so a caller can resolve it with ``memory.get`` and
    reach the very page text, document digest and physical page number it was
    built from. Query processing is offline and deterministic.
    """

    if not isinstance(question, str) or not question.strip():
        return ()
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        return ()
    entries = await _captured_page_entries(memory)
    if not entries:
        return ()
    prior = await _fts_prior(memory, question, entries)
    ranked = _rank_pages(question, entries, prior)
    return tuple(entry.citation for entry in ranked[: min(limit, _MAX_SEARCH_LIMIT)])


def _extract_page_texts(raw: bytes) -> tuple[str, ...]:
    """Read every page of one document, refusing to return a partial result."""

    _require_complete_file(raw)
    reader = _open_pdf_reader(raw)
    try:
        pages = reader.pages
        count = len(pages)
    except Exception as error:
        raise ValueError(f"PDF document cannot be parsed: {error}") from error
    if count < 1:
        raise ValueError("PDF document declares no physical page")
    declared = _declared_page_count(reader)
    if declared is not None and count < declared:
        raise ValueError(_PAGE_SHORTFALL_MESSAGE.format(declared=declared, readable=count))
    return tuple(_read_page_text(pages, index) for index in range(count))


def _require_complete_file(raw: bytes) -> None:
    """Reject a file whose end is gone: a truncated report has no last page.

    Every conforming PDF terminates with its end-of-file marker, and the last
    cross-reference section of the file announces its own offset just before that
    marker. A file cut short still carries the marker of its previous revision, so
    finding *a* marker proves nothing: a revision appended and then truncated
    leaves its own bytes after the last marker, and the surviving marker of the
    older revision would otherwise let a reader recover a prefix of the pages,
    which is exactly the partial report this contract must never return. The final
    marker must therefore be the end of the file, and it must close the
    cross-reference section the file's own trailer announces; the number of pages
    the document declares is then checked against the pages actually readable.
    """

    end = raw.rfind(_EOF_MARKER)
    if end < 0:
        raise ValueError(_TRUNCATED_MESSAGE)
    if raw[end + len(_EOF_MARKER) :].strip(_INERT_BYTES):
        raise ValueError(_TRUNCATED_MESSAGE)
    trailers = tuple(_XREF_REFERENCE.finditer(raw, max(0, end - _EOF_WINDOW), end))
    if not trailers:
        raise ValueError(_TRUNCATED_MESSAGE)
    trailer = trailers[-1]
    if raw[trailer.end() : end].strip(_INERT_BYTES):
        raise ValueError(_TRUNCATED_MESSAGE)
    if not _starts_cross_reference(raw, int(trailer.group(1))):
        raise ValueError(_TRUNCATED_MESSAGE)


def _starts_cross_reference(raw: bytes, offset: int) -> bool:
    """Tell whether ``offset`` really begins a cross-reference section.

    A cross-reference section is either the classic ``xref`` table or the object
    that carries a cross-reference stream, so the announced offset has to land on
    one of those two heads and inside the file.
    """

    if offset <= 0 or offset >= len(raw):
        return False
    head = raw[offset : offset + 64].lstrip(_INERT_BYTES)
    return _XREF_SECTION_HEAD.match(head) is not None


def _open_pdf_reader(raw: bytes) -> Any:
    """Open the file, accepting a permission marker with an empty password."""

    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(raw))
    except Exception as error:
        raise ValueError(f"PDF document cannot be parsed: {error}") from error
    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception as error:
            raise ValueError("PDF document cannot be read without a password") from error
        if not unlocked:
            raise ValueError("PDF document cannot be read without a password")
    return reader


def _declared_page_count(reader: Any) -> int | None:
    """Read how many pages the document's own page tree promises, if it says."""

    try:
        count = _resolved(_resolved(reader.trailer["/Root"])["/Pages"])["/Count"]
    except Exception:
        return None
    count = _resolved(count)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        return None
    return count


def _resolved(value: Any) -> Any:
    """Dereference an indirect object; a direct one already is the value."""

    return value.get_object() if hasattr(value, "get_object") else value


def _read_page_text(pages: Any, index: int) -> str:
    try:
        page = pages[index]
    except Exception as error:
        raise ValueError(f"PDF page {index + 1} cannot be read: {error}") from error
    text = _extract_page_text(page)
    if text is None:
        raise ValueError(f"PDF page {index + 1} text cannot be extracted")
    return text


def _extract_page_text(page: Any) -> str | None:
    """Render one page with the plain-text extraction, layout only as a rescue.

    The default mode is authoritative whenever it yields text, so a page that
    already extracts never changes and the second pass is paid only where it can
    matter. A page whose default extraction raises or comes back empty is retried
    with the layout mode: selectable text that only the layout mode exposes
    (vector glyphs, column order, unusual encodings) would otherwise be lost as a
    blank page. A page that stays empty in both modes is still blank.
    """

    try:
        rendered = page.extract_text()
    except Exception:
        rendered = None
    if isinstance(rendered, str) and rendered.strip():
        return rendered.strip()
    try:
        fallback = page.extract_text(extraction_mode="layout")
    except Exception:
        fallback = None
    if isinstance(fallback, str) and fallback.strip():
        return fallback.strip()
    if isinstance(rendered, str):
        return rendered.strip()
    if isinstance(fallback, str):
        return fallback.strip()
    return None


async def _captured_page_entries(memory: Any) -> tuple[_PageEntry, ...]:
    """Read the scoped Memory entries once, one entry per distinct page."""

    page = await memory.list()
    entries: list[_PageEntry] = []
    seen: set[str] = set()
    for record in getattr(page, "entries", ()):
        entry = _page_entry(record)
        if entry is None or entry.key in seen:
            continue
        seen.add(entry.key)
        entries.append(entry)
    return tuple(entries)


def _page_entry(record: Any) -> _PageEntry | None:
    """Describe one Memory entry as a page, or ``None`` for non-page evidence."""

    version = getattr(record, "entry", None)
    memory_ref = getattr(record, "memory_ref", None)
    text = getattr(version, "text", "")
    if version is None or memory_ref is None or not isinstance(text, str) or not text.strip():
        return None
    page_source = next(
        (
            source
            for source in getattr(version, "sources", ()) or ()
            if getattr(source, "source_type", "") == PDF_PAGE_SOURCE_NAME
        ),
        None,
    )
    if page_source is None:
        return None
    # Imported here because the Artifact package itself imports this package.
    from powercontext.artifacts import MemoryCitation

    document, page_number = _page_locator(page_source.source_id)
    return _PageEntry(
        key=page_source.source_id,
        document=document,
        page_number=page_number,
        document_id=text.partition("\n")[0],
        text=text,
        citation=MemoryCitation(
            memory_ref=memory_ref,
            entry_id=version.entry_id,
            entry_version_id=version.entry_version_id,
        ),
    )


def _page_locator(source_id: str) -> tuple[str, int | None]:
    """Recover the document and physical page from a page Source identity."""

    _, marker, tail = source_id.rpartition(_PAGE_MARKER)
    if not marker or not tail.isdigit():
        return source_id, None
    return source_id[: len(source_id) - len(marker) - len(tail)], int(tail)


async def _fts_prior(memory: Any, question: str, entries: tuple[_PageEntry, ...]) -> dict[str, float]:
    """Read the store's own full-text ranking as a small prior over the pages."""

    try:
        from powercontext.builtin.runtime import SearchMemoryRequest

        page = await memory.search(SearchMemoryRequest(query=question, mode="fts", limit=_FTS_CANDIDATES))
    except Exception:
        return {}
    hits = tuple(getattr(page, "hits", ()) or ())
    by_entry = {entry.citation.entry_id: entry.key for entry in entries}
    prior: dict[str, float] = {}
    for rank, hit in enumerate(hits):
        key = by_entry.get(getattr(hit, "entry_id", ""))
        if key is None:
            continue
        prior[key] = max(prior.get(key, 0.0), 1.0 - rank / len(hits))
    return prior


def _rank_pages(question: str, entries: tuple[_PageEntry, ...], prior: dict[str, float]) -> tuple[_PageEntry, ...]:
    """Rank the captured pages of one collection against one question."""

    indexes, vocabulary = _index_pages(entries, prior)
    terms = _query_terms(question)
    if not terms:
        return entries
    keywords = {term.key for term in terms}
    boost = _document_boost(indexes, keywords)
    pairs = _query_pairs(question, vocabulary)
    scored = _score_pages(indexes, terms, boost, keywords, pairs)
    feedback = _feedback_terms(indexes, tuple(index for _, index in scored), terms)
    if feedback:
        terms = terms + tuple(_QueryTerm(term, _FEEDBACK_WEIGHT) for term in feedback)
        scored = _score_pages(indexes, terms, boost, keywords, pairs)
    return tuple(index.entry for index in _spread_neighbours(scored))


# Tokenising a whole collection costs more than ranking it, and the tokens of a
# page never depend on the question, so the index of an unchanged set of captured
# pages is kept and reused by every later question.
_PAGE_INDEX_CACHE: OrderedDict[
    tuple[tuple[str, str], ...], tuple[tuple[_PageIndex, ...], dict[str, int]]
] = OrderedDict()
_PAGE_INDEX_CACHE_LOCK = threading.Lock()
_PAGE_INDEX_CACHE_HITS = 0
_PAGE_INDEX_CACHE_MISSES = 0


def _index_pages(
    entries: tuple[_PageEntry, ...],
    prior: dict[str, float],
) -> tuple[tuple[_PageIndex, ...], dict[str, int]]:
    """Index the captured pages, rebinding a cached index to this Memory state.

    Only the question-dependent parts are applied here, the citation of the
    current Memory entry of each page and the store's own full-text prior, so a
    second question over the same captured revisions never tokenises again.
    """

    cached, vocabulary = _cached_page_indexes(entries)
    current = {entry.key: entry for entry in entries}
    indexes = tuple(
        replace(index, entry=current[index.entry.key], prior=prior.get(index.entry.key, 0.0)) for index in cached
    )
    return indexes, vocabulary


def _cached_page_indexes(
    entries: tuple[_PageEntry, ...],
) -> tuple[tuple[_PageIndex, ...], dict[str, int]]:
    """Return the index of exactly this set of captured page revisions.

    The key is the sorted identity of the pages, and a page identity already
    carries its document digest and physical page number, so a revised document,
    an added page or a removed page all miss while re-asking over an unchanged
    Memory hits. A duplicate build cannot publish a wrong index, because the build
    happens outside the lock and the cache entry is written whole.
    """

    global _PAGE_INDEX_CACHE_HITS, _PAGE_INDEX_CACHE_MISSES
    key = tuple(sorted((entry.key, entry.citation.entry_version_id) for entry in entries))
    with _PAGE_INDEX_CACHE_LOCK:
        cached = _PAGE_INDEX_CACHE.get(key)
        if cached is not None:
            _PAGE_INDEX_CACHE.move_to_end(key)
            _PAGE_INDEX_CACHE_HITS += 1
            return cached
        _PAGE_INDEX_CACHE_MISSES += 1
    built = _build_page_indexes(entries)
    with _PAGE_INDEX_CACHE_LOCK:
        _PAGE_INDEX_CACHE[key] = built
        _PAGE_INDEX_CACHE.move_to_end(key)
        while len(_PAGE_INDEX_CACHE) > _PAGE_INDEX_CACHE_SIZE:
            _PAGE_INDEX_CACHE.popitem(last=False)
    return built


def _build_page_indexes(entries: tuple[_PageEntry, ...]) -> tuple[tuple[_PageIndex, ...], dict[str, int]]:
    """Tokenise every captured page once, from the pages alone."""

    vocabulary: dict[str, int] = {}
    indexes: list[_PageIndex] = []
    for entry in entries:
        body = _page_body(entry.text)
        tokens = _tokens(body)
        term_freq: dict[str, int] = {}
        prefix_freq: dict[str, int] = {}
        for token in tokens:
            term_freq[token] = term_freq.get(token, 0) + 1
            if len(token) > _PREFIX_LENGTH and not token.isdigit():
                key = token[:_PREFIX_LENGTH]
                prefix_freq[key] = prefix_freq.get(key, 0) + 1
        indexes.append(
            _PageIndex(
                entry=entry,
                tokens=len(tokens),
                term_freq=term_freq,
                prefix_freq=prefix_freq,
                pairs=_pairs(tokens, vocabulary),
                heading_terms=_statement_heading(body),
                prior=0.0,
            )
        )
    return tuple(indexes), vocabulary


def _clear_page_index_cache() -> None:
    """Drop the cached page index, so a caller can observe a cold build again."""

    global _PAGE_INDEX_CACHE_HITS, _PAGE_INDEX_CACHE_MISSES
    with _PAGE_INDEX_CACHE_LOCK:
        _PAGE_INDEX_CACHE.clear()
        _PAGE_INDEX_CACHE_HITS = 0
        _PAGE_INDEX_CACHE_MISSES = 0


def _page_index_cache_stats() -> tuple[int, int, int]:
    """Return ``(hits, misses, cached collections)`` of the page index cache."""

    with _PAGE_INDEX_CACHE_LOCK:
        return _PAGE_INDEX_CACHE_HITS, _PAGE_INDEX_CACHE_MISSES, len(_PAGE_INDEX_CACHE)


def _page_body(text: str) -> str:
    """Drop the document identifier line and a repeated running header."""

    _, _, body = text.partition("\n")
    lines = body.split("\n")
    if lines and lines[0].strip().lower().startswith("table of contents"):
        lines = lines[1:]
    return "\n".join(lines)


def _pairs(tokens: Sequence[str], vocabulary: dict[str, int]) -> frozenset[int]:
    """Encode the adjacent content-word pairs of one page, order independent."""

    identifiers = [vocabulary.setdefault(token, len(vocabulary)) for token in tokens if token not in _STOPWORDS]
    encoded: set[int] = set()
    for left, right in pairwise(identifiers):
        low, high = (left, right) if left <= right else (right, left)
        encoded.add((low << _PAIR_SHIFT) | high)
    return frozenset(encoded)


def _statement_heading(body: str) -> frozenset[str]:
    """Detect a primary-statement heading at the top of the page."""

    for line in body.split("\n")[:4]:
        stripped = line.strip().lower()
        if stripped.startswith(("consolidated statement", "consolidated balance sheet")):
            return frozenset(token for token in _tokens(stripped) if token not in _STOPWORDS)
    return frozenset()


def _expansions_for(token: str) -> tuple[str, ...]:
    """Return the vocabulary one question token opens, or nothing.

    The longest key that prefixes the token wins, so ``battl`` reaches a report
    that only ever writes ``battle`` and ``geogra`` still reaches every
    ``geographic``/``geography`` wording of the same question.
    """

    for key in _EXPANSION_KEYS:
        if token.startswith(key):
            return _EXPANSIONS[key]
    return ()


def _query_terms(question: str) -> tuple[_QueryTerm, ...]:
    """Analyse the question into weighted terms, including finance expansions."""

    weights: dict[str, float] = {}
    for token in _tokens(question):
        if token in _STOPWORDS:
            continue
        weights[token] = max(weights.get(token, 0.0), _term_weight(token))
    for token in tuple(weights):
        for extra in _expansions_for(token):
            normalized = _normalize(extra)
            if normalized and normalized not in weights:
                weights[normalized] = _EXPANSION_WEIGHT
    return tuple(_QueryTerm(term, weight) for term, weight in weights.items())


def _query_pairs(question: str, vocabulary: dict[str, int]) -> frozenset[int]:
    identifiers = [
        vocabulary[token]
        for token in _tokens(question)
        if token not in _STOPWORDS and token in vocabulary
    ]
    encoded: set[int] = set()
    for left, right in pairwise(identifiers):
        low, high = (left, right) if left <= right else (right, left)
        encoded.add((low << _PAIR_SHIFT) | high)
    return frozenset(encoded)


def _term_weight(token: str) -> float:
    if token.isdigit():
        return 0.7 if len(token) == 4 else 1.0
    return 0.5 if len(token) < 3 else 1.0


def _document_boost(indexes: tuple[_PageIndex, ...], query: set[str]) -> dict[str, float]:
    """Boost the document whose own identifier the question names."""

    documents = {index.entry.document_id for index in indexes}
    if len(documents) < 2:
        return {}
    boost: dict[str, float] = {}
    for document in documents:
        shared = set(_tokens(document)) & query
        if shared:
            boost[document] = 1.0 + _DOCUMENT_AFFINITY * len(shared)
    return boost


def _score_pages(
    indexes: tuple[_PageIndex, ...],
    terms: tuple[_QueryTerm, ...],
    boost: dict[str, float],
    keywords: set[str],
    pairs: frozenset[int],
) -> list[tuple[float, _PageIndex]]:
    """Score every page with BM25 and the priors of a finance report."""

    total = len(indexes)
    average = sum(index.tokens for index in indexes) / total
    average = average if average > 0 else 1.0
    numeric = bool(keywords & _NUMERIC_INTENT)
    document_frequency = {term.key: 0 for term in terms}
    for index in indexes:
        for term in terms:
            if _matched_count(index, term.key) > 0:
                document_frequency[term.key] += 1
    inverse = {key: _inverse_frequency(count, total) for key, count in document_frequency.items()}
    scored: list[tuple[float, _PageIndex]] = []
    for index in indexes:
        score = _bm25(index, terms, inverse, average) if index.tokens else 0.0
        if score > 0:
            matched = sum(1 for term in terms if _matched_count(index, term.key) > 0)
            score *= 1 + _COVERAGE_WEIGHT * matched / len(terms)
            score *= 1 + _FTS_PRIOR_WEIGHT * index.prior
            score *= boost.get(index.entry.document_id, 1.0)
            score *= 1 + _PAIR_WEIGHT * min(len(index.pairs & pairs), _PAIR_CAP) / _PAIR_CAP
            if numeric and index.heading_terms:
                score *= 1 + _STATEMENT_BOOST * min(len(index.heading_terms & keywords), 3)
        scored.append((score, index))
    scored.sort(key=lambda item: (-item[0], -item[1].prior, item[1].entry.key))
    return scored


def _bm25(index: _PageIndex, terms: tuple[_QueryTerm, ...], inverse: dict[str, float], average: float) -> float:
    score = 0.0
    for term in terms:
        count = _matched_count(index, term.key)
        if count < 1:
            continue
        saturation = count * (_BM25_K1 + 1) / (
            count + _BM25_K1 * (1 - _BM25_B + _BM25_B * index.tokens / average)
        )
        score += term.weight * inverse[term.key] * saturation
    return score


def _spread_neighbours(scored: list[tuple[float, _PageIndex]]) -> tuple[_PageIndex, ...]:
    """Let a strong page carry its immediate neighbours, as a report reads."""

    scores = {index.entry.key: score for score, index in scored}
    lookup = {
        (index.entry.document, index.entry.page_number): index
        for _, index in scored
        if index.entry.page_number is not None
    }
    for score, index in scored[:_SEED_PAGES]:
        number = index.entry.page_number
        if number is None:
            continue
        for distance, decay in _NEIGHBOUR_DECAY.items():
            for neighbour_number in (number - distance, number + distance):
                neighbour = lookup.get((index.entry.document, neighbour_number))
                if neighbour is None:
                    continue
                scores[neighbour.entry.key] = max(scores[neighbour.entry.key], score * decay)
    ordered = sorted(scored, key=lambda item: (-scores[item[1].entry.key], -item[1].prior, item[1].entry.key))
    return tuple(index for _, index in ordered)


def _matched_count(index: _PageIndex, term: str) -> int:
    """Count exact occurrences plus the longer words sharing the term prefix."""

    exact = index.term_freq.get(term, 0)
    if term.isdigit():
        return exact
    bucket = index.prefix_freq.get(term[:_PREFIX_LENGTH], 0)
    if len(term) > _PREFIX_LENGTH:
        bucket = max(0, bucket - exact)
    return exact + int(_PREFIX_WEIGHT * bucket)


def _feedback_terms(
    indexes: tuple[_PageIndex, ...],
    ranked: tuple[_PageIndex, ...],
    terms: tuple[_QueryTerm, ...],
) -> tuple[str, ...]:
    """Collect the salient wording of the best pages as extra query terms."""

    known = {term.key for term in terms}
    total = len(indexes)
    counts: dict[str, int] = {}
    support: dict[str, int] = {}
    for index in ranked[:_FEEDBACK_DOCUMENTS]:
        for term, count in index.term_freq.items():
            if term in known or term in _STOPWORDS or term.isdigit() or len(term) < 4:
                continue
            counts[term] = counts.get(term, 0) + count
            support[term] = support.get(term, 0) + 1
    share = dict.fromkeys(counts, 0)
    for index in indexes:
        for term in index.term_freq:
            if term in share:
                share[term] += 1
    candidates: list[tuple[float, str]] = []
    for term, count in counts.items():
        if support[term] < _FEEDBACK_MIN_SUPPORT or share[term] * _FEEDBACK_MAX_SHARE > total:
            continue
        candidates.append((count * _inverse_frequency(share[term], total), term))
    candidates.sort(reverse=True)
    return tuple(term for _, term in candidates[:_FEEDBACK_TERMS])


def _inverse_frequency(document_frequency: int, total: int) -> float:
    return math.log(1 + (total - document_frequency + 0.5) / (document_frequency + 0.5))


def _tokens(text: str) -> list[str]:
    return [_normalize(token) for token in _TOKEN_PATTERN.findall(text.lower())]


def _normalize(token: str) -> str:
    return _stem(token)


def _stem(token: str) -> str:
    """Fold plurals only; longer words meet through their shared prefix."""

    if token.isdigit() or len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith(("sses", "shes", "ches", "xes")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def _require_document_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidSourceReferenceError("document_id", "must be a non-empty, non-blank string")
    return value


def _require_document_sha256(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidSourceReferenceError("document_sha256", "must be a non-empty string")
    return value


def _require_page_number(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InvalidSourceReferenceError("page_number", "must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class _PageEntry:
    """One captured page reached through one Memory entry."""

    key: str
    document: str
    page_number: int | None
    document_id: str
    text: str
    citation: MemoryCitation


@dataclass(frozen=True, slots=True)
class _PageIndex:
    """The offline lexical index of one captured page."""

    entry: _PageEntry
    tokens: int
    term_freq: dict[str, int]
    prefix_freq: dict[str, int]
    pairs: frozenset[int]
    heading_terms: frozenset[str]
    prior: float


@dataclass(frozen=True, slots=True)
class _QueryTerm:
    """One weighted query term."""

    key: str
    weight: float


__all__ = [
    "PDF_PAGE_SOURCE_ADAPTER",
    "PDF_PAGE_SOURCE_DEFINITION",
    "PDF_PAGE_SOURCE_NAME",
    "PDF_PAGE_SOURCE_VERSION",
    "PDFDocumentInput",
    "PDFPageInput",
    "PDFPageSource",
    "PDFPageSourceAdapter",
    "PDFPageTextEvidenceProjection",
    "parse_pdf_document",
    "pdf_page_source_id",
    "search_pdf_pages",
]
