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

"""Rank the literal sentences of recalled Memory entries against one query.

Pure text analysis: no I/O, no model call, no persistence, and nothing that could rewrite,
translate or summarise evidence. The Builder keeps the byte budget and the citation identity;
this module answers only "which literal stretches of the already recalled text speak to this
query", and it answers with half-open character ranges of the entry itself, so every answer can
be sliced out of the original text and checked against it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

# Letters and digits in any script, without the underscore: the shape the acceptance tokenizer
# uses for English words and numbers, plus the CJK runs a query may carry.
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_BOUNDARY = re.compile(r"(?<=[.!?\u3002\uff01\uff1f])\s+|\n+")
_CLAUSE = re.compile(r"[;:,]\s+")
_MIN_PIECE_CHARS = 96
_MAX_PIECE_CHARS = 480
# A sentence that matches the query through one common word must not outrank a sentence
# anchored on a rare one; identifier frequency is what separates those two cases.
_LENGTH_PENALTY_CHARS = 1200.0
_BIGRAM_BONUS = 0.5


@dataclass(frozen=True)
class EvidenceSentence:
    """One literal character range of one entry together with its query relevance."""

    entry: int
    index: int
    start: int
    end: int
    score: float = 0.0


def tokenize(text: str) -> tuple[str, ...]:
    """Lowercase word and number tokens, script agnostic and order preserving."""

    return tuple(_TOKEN.findall(text.lower()))


def sentence_spans(text: str) -> tuple[tuple[int, int], ...]:
    """Split one entry into literal, non-overlapping character ranges.

    Ranges are half-open, trimmed of surrounding whitespace and never empty, so a caller can
    slice ``text[start:end]`` and get the sentence back verbatim. A run that carries no sentence
    break is cut at clause separators once it grows past ``_MAX_PIECE_CHARS``, which keeps one
    oversized range from swallowing a whole byte budget.
    """

    pieces: list[tuple[int, int]] = []
    cursor = 0
    for boundary in _BOUNDARY.finditer(text):
        pieces.extend(_bounded_pieces(text, cursor, boundary.start()))
        cursor = boundary.end()
    pieces.extend(_bounded_pieces(text, cursor, len(text)))
    return tuple(pieces)


def rank_evidence(query: str, texts: Sequence[str]) -> tuple[tuple[EvidenceSentence, ...], ...]:
    """Score every literal sentence of every entry against ``query``.

    Returns one tuple per entry, in document order, so a caller can both rank globally and grow
    a contiguous window inside a single entry. Ties keep the earlier entry -- the better ranked
    retrieval hit -- and the earlier sentence, which keeps the whole selection deterministic.
    """

    spans = tuple(sentence_spans(text) for text in texts)
    query_tokens = tokenize(query)
    terms = frozenset(query_tokens)
    bigrams = frozenset(pairwise(query_tokens))
    bodies: list[tuple[int, int, int, tuple[str, ...]]] = []
    frequency: dict[str, int] = {}
    for entry, entry_spans in enumerate(spans):
        text = texts[entry]
        for start, end in entry_spans:
            tokens = tokenize(text[start:end])
            bodies.append((entry, start, end, tokens))
            for term in terms.intersection(tokens):
                frequency[term] = frequency.get(term, 0) + 1
    total = len(bodies)
    weight = {term: math.log((total + 1.0) / (frequency.get(term, 0) + 0.5)) for term in terms}
    scored: list[list[EvidenceSentence]] = [[] for _ in texts]
    for entry, start, end, tokens in bodies:
        matched = terms.intersection(tokens)
        raw = math.fsum(weight[term] for term in sorted(matched))
        paired = bigrams.intersection(pairwise(tokens))
        if paired:
            raw += _BIGRAM_BONUS * math.fsum(weight[left] + weight[right] for left, right in sorted(paired))
        score = raw / (1.0 + (end - start) / _LENGTH_PENALTY_CHARS)
        scored[entry].append(EvidenceSentence(entry=entry, index=len(scored[entry]), start=start, end=end, score=score))
    return tuple(tuple(group) for group in scored)


def _bounded_pieces(text: str, start: int, end: int) -> tuple[tuple[int, int], ...]:
    """Trim one boundary-delimited range and cap its length at clause separators."""

    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end <= start:
        return ()
    if end - start <= _MAX_PIECE_CHARS:
        return ((start, end),)
    pieces: list[tuple[int, int]] = []
    cursor = start
    for separator in _CLAUSE.finditer(text, start, end):
        if separator.end() - cursor < _MIN_PIECE_CHARS:
            continue
        if separator.start() - cursor > _MAX_PIECE_CHARS:
            continue
        pieces.append((cursor, separator.start()))
        cursor = separator.end()
    pieces.append((cursor, end))
    return _capped(text, pieces)


def _capped(text: str, pieces: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Cut any remaining oversized piece at a space so no range dwarfs the budget."""

    capped: list[tuple[int, int]] = []
    for start, end in pieces:
        while end - start > _MAX_PIECE_CHARS:
            cut = text.rfind(" ", start + _MIN_PIECE_CHARS, start + _MAX_PIECE_CHARS)
            if cut <= start:
                break
            capped.append((start, cut))
            start = cut + 1
        if end > start:
            capped.append((start, end))
    return tuple(capped)
