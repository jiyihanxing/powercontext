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

"""Prepare final, bounded context from approved Artifact evidence."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import chain
from typing import TypeVar

from powercontext.artifacts import ArtifactAddress, ArtifactRef
from powercontext.builtin.artifacts.experience import Experience, ExperienceSearchHit, render_experience
from powercontext.builtin.artifacts.memory.models import MemoryCitation, MemoryHit
from powercontext.builtin.artifacts.profile.models import Profile
from powercontext.builtin.artifacts.topic_memory import TopicMemory, TopicMemorySearchHit
from powercontext.builtin.runtime.errors import PreparedContextInvariantError
from powercontext.builtin.runtime.evidence_ranking import EvidenceSentence, rank_evidence
from powercontext.builtin.runtime.models import PrepareContextRequest, PreparedContext
from powercontext.builtin.runtime.prepared_code import (
    CodeEvidenceRef,
    PreparedCodeCandidate,
    assemble_code,
)
from powercontext.builtin.runtime.prepared_text import (
    TRUST_POLICY,
    ContextTextItem,
    fit_context_text_item,
    render_context_text,
)
from powercontext.builtin.runtime.recall_sufficiency import RecallBudgetView

_MIN_TRUNCATED_CONTENT_BYTES = 64
_ELLIPSIS = "…"
_BEGIN_MARKER = "BEGIN_POWERCONTEXT_PREPARED_CONTEXT_V1"
_END_MARKER = "END_POWERCONTEXT_PREPARED_CONTEXT_V1"
_Item = TypeVar("_Item")
# An excerpt of one entry costs one citation, so a second slice of the same entry costs another
# citation instead of one more copy of the text. Neighbouring slices are therefore joined while
# the gap between them is cheaper than the citation block that joining removes.
_EVIDENCE_MERGE_GAP = 24
# A cut shorter than this is not worth the citation it has to pay for: leave the bytes unused
# instead of delivering a stub nobody can read.
_EVIDENCE_MIN_PARTIAL_CHARS = 32
_EVIDENCE_FRAGMENT_LIMIT = 16


@dataclass(frozen=True)
class _PreparedContextEntry:
    origin: PreparedContextOrigin
    kind: str
    citation: dict[str, object]
    content: str
    truncated: bool


@dataclass(frozen=True)
class _EntryFit:
    """One fit attempt on the non-assembly path: the entry that fitted, or why none did.

    The two drop reasons are disjoint and mirror the assembly path's
    :class:`~powercontext.builtin.runtime.prepared_text.FitOutcome`. An entry is dropped
    *below the minimum truncated content* when a shorter rendering exists and fits but is too
    short to be an honest delivery; it is dropped *with no fitting truncation* when no
    shortened rendering fits at all.
    """

    entry: _PreparedContextEntry | None = None
    dropped_below_min_bytes: bool = False
    dropped_no_fitting_truncation: bool = False


@dataclass(frozen=True)
class PreparedContextOmissions:
    """Aggregate counts of the items the byte budget omitted in one build.

    Every count is a per-call aggregate, never per-entry attribution and never a verdict about
    an entry: an entry losing to the budget is not a negative result about that entry.

    ``dropped_items`` is the sum of its two sub-counts, so "the budget could not fit this
    item" can be told apart from "this item was too short to truncate into the remaining
    space". A single merged counter would misreport the second cause as the first.
    """

    truncated_items: int = 0
    dropped_items: int = 0
    dropped_below_min_bytes: int = 0
    dropped_no_fitting_truncation: int = 0


@dataclass(frozen=True)
class PreparedContextBuild:
    """Final public context and the exact origins selected to produce it.

    ``fragments`` is filled by evidence selection and stays empty otherwise, and only then does
    ``origins`` describe excerpts rather than whole entries: one origin per fragment, in the same
    order, so a caller can pair the two by position and re-read every slice in the original entry.

    ``omissions`` is always filled by the Builder. The RFC 1560 recall trace is deliberately
    **not** a field here: ``ScopedContextApplication._prepare`` returns ``build.context`` and
    discards the rest, so such a field would have no production observer. The trace is
    delivered through the Runtime's optional ``RecallEffortSink`` instead.
    """

    context: PreparedContext
    origins: tuple[PreparedContextOrigin, ...]
    omissions: PreparedContextOmissions = PreparedContextOmissions()
    code_origins: tuple[CodeEvidenceRef, ...] = ()
    fragments: tuple[PreparedFragment, ...] = ()


@dataclass(frozen=True)
class MemoryEntryAddress:
    """Identify one exact Memory entry version across Scope boundaries."""

    memory: ArtifactAddress
    entry_id: str
    entry_version_id: str


@dataclass(frozen=True)
class PreparedFragment:
    """One literal excerpt of one Memory entry version.

    ``start`` and ``end`` are half-open Python character offsets into the entry text and
    ``text`` is exactly ``entry.text[start:end]``, so a fragment never rewrites, translates or
    summarises its source. That is what makes the citation an assertion a caller can re-check
    against the original text instead of a label it has to trust.
    """

    address: MemoryEntryAddress
    start: int
    end: int
    text: str


PreparedContextOrigin = MemoryCitation | ArtifactRef | MemoryEntryAddress | ArtifactAddress


@dataclass(frozen=True)
class PreparedMemoryCandidates:
    """Memory candidates read from one Scope."""

    scope_id: str
    memory_ref: ArtifactRef | None = None
    hits: tuple[MemoryHit, ...] = ()


@dataclass(frozen=True)
class PreparedExperienceCandidates:
    """Experience candidates read from one Scope."""

    scope_id: str
    hits: tuple[ExperienceSearchHit, ...] = ()


@dataclass(frozen=True)
class PreparedProfileCandidate:
    """The latest committed Profile snapshot read from one Scope."""

    scope_id: str
    profile: Profile


class PreparedContextBuilder:
    """Select and render final context without I/O, persistence, or reranking."""

    memory_candidate_limit = 16
    topic_memory_candidate_limit = 8
    experience_candidate_limit = 8
    candidate_limit = memory_candidate_limit
    entry_limit = 8
    topic_memory_entry_limit = 8
    experience_entry_limit = 2
    max_entry_content_bytes = 2000
    _text_entry_limit: int | None = None

    def empty(self) -> PreparedContext:
        return PreparedContext(status="empty", content=None, content_bytes=0)

    def build(
        self,
        *,
        request: PrepareContextRequest,
        scope_id: str | None = None,
        memory_ref: ArtifactRef | None = None,
        hits: Sequence[MemoryHit] = (),
        topic_memory_hits: Sequence[TopicMemorySearchHit] = (),
        experience_hits: Sequence[ExperienceSearchHit] = (),
    ) -> PreparedContext:
        return self.build_result(
            request=request,
            scope_id=scope_id,
            memory_ref=memory_ref,
            hits=hits,
            topic_memory_hits=topic_memory_hits,
            experience_hits=experience_hits,
        ).context

    def build_result(
        self,
        *,
        request: PrepareContextRequest,
        scope_id: str | None = None,
        memory_ref: ArtifactRef | None = None,
        hits: Sequence[MemoryHit] = (),
        topic_memory_hits: Sequence[TopicMemorySearchHit] = (),
        experience_hits: Sequence[ExperienceSearchHit] = (),
    ) -> PreparedContextBuild:
        return self.build_scopes_result(
            request=request,
            current_scope_id=scope_id,
            memory_candidates=(
                PreparedMemoryCandidates(scope_id=scope_id or "", memory_ref=memory_ref, hits=tuple(hits)),
            ),
            experience_candidates=(PreparedExperienceCandidates(scope_id=scope_id or "", hits=tuple(experience_hits)),),
            topic_memory_hits=topic_memory_hits,
        )

    def build_scopes_result(
        self,
        *,
        request: PrepareContextRequest,
        current_scope_id: str | None,
        memory_candidates: Sequence[PreparedMemoryCandidates] = (),
        topic_memory_hits: Sequence[TopicMemorySearchHit] = (),
        experience_candidates: Sequence[PreparedExperienceCandidates] = (),
        profile_candidates: Sequence[PreparedProfileCandidate] = (),
        code_candidates: Sequence[PreparedCodeCandidate] = (),
    ) -> PreparedContextBuild:
        if request.include_code:

            def historical(history_request: PrepareContextRequest, entries: int) -> PreparedContextBuild:
                builder = PreparedContextBuilder()
                builder.entry_limit = entries
                builder._text_entry_limit = entries if entries < self.entry_limit else None
                return builder.build_scopes_result(
                    request=history_request.model_copy(update={"include_code": False}),
                    current_scope_id=current_scope_id,
                    memory_candidates=memory_candidates,
                    topic_memory_hits=topic_memory_hits,
                    experience_candidates=experience_candidates,
                    profile_candidates=profile_candidates,
                )

            return assemble_code(request, code_candidates, self.entry_limit, historical)

        if sum(len(candidates.hits) for candidates in memory_candidates) > self.memory_candidate_limit:
            raise PreparedContextInvariantError("memory-candidate-limit")
        if len(topic_memory_hits) > self.topic_memory_candidate_limit:
            raise PreparedContextInvariantError("topic-memory-candidate-limit")
        if sum(len(candidates.hits) for candidates in experience_candidates) > self.experience_candidate_limit:
            raise PreparedContextInvariantError("experience-candidate-limit")

        # Evidence selection owns the plain path only: when the caller also supplied a text
        # assembly, that explicit section/limit contract keeps rendering the output.
        if request.evidence_selection and request.assembly is None:
            fragments = self._evidence_fragments(request, current_scope_id, memory_candidates)
            if not fragments:
                return PreparedContextBuild(context=self.empty(), origins=())
            content = _render_fragments(fragments)
            return PreparedContextBuild(
                context=PreparedContext(
                    status="ready",
                    content=content,
                    content_bytes=len(content.encode("utf-8")),
                ),
                origins=tuple(fragment.address for fragment in fragments),
                fragments=fragments,
            )

        content, origins, omissions = self._select_entries(
            request=request,
            current_scope_id=current_scope_id,
            memory_candidates=memory_candidates,
            experience_candidates=experience_candidates,
            profile_candidates=profile_candidates,
            topic_memory_hits=topic_memory_hits,
        )
        if content is None:
            return PreparedContextBuild(context=self.empty(), origins=(), omissions=omissions)
        content_bytes = len(content.encode("utf-8"))
        # The assembly path is bounded by construction (every fitted item is measured against
        # `max_bytes`), so the ceiling check belongs to the non-assembly renderer only.
        if request.assembly is None and content_bytes > request.max_bytes:
            raise PreparedContextInvariantError("output-budget")
        return PreparedContextBuild(
            context=PreparedContext(status="ready", content=content, content_bytes=content_bytes),
            origins=origins,
            omissions=omissions,
        )

    def probe_budget(
        self,
        *,
        request: PrepareContextRequest,
        current_scope_id: str | None,
        memory_candidates: Sequence[PreparedMemoryCandidates] = (),
        topic_memory_hits: Sequence[TopicMemorySearchHit] = (),
        experience_candidates: Sequence[PreparedExperienceCandidates] = (),
        profile_candidates: Sequence[PreparedProfileCandidate] = (),
    ) -> RecallBudgetView:
        """Report what the byte budget does to one candidate set, without delivering it.

        This is the RFC 1560 *budget probe*: one pass of this Builder's own pure selection and
        rendering code, keeping only the counters and discarding the rendered output. It lets
        the gate tell budget-limited thinness from recall-limited thinness.

        Pure and side-effect free: no I/O, no persistence, nothing reported to any sink, and
        the same inputs as :meth:`build_scopes_result` are guaranteed to yield the same
        counters because both run the same selection pass for the same request.
        """

        if request.evidence_selection and request.assembly is None:
            fragments = self._evidence_fragments(request, current_scope_id, memory_candidates)
            content = _render_fragments(fragments) if fragments else None
            content_bytes = 0 if content is None else len(content.encode("utf-8"))
            return RecallBudgetView(
                max_bytes=request.max_bytes,
                delivered_items=len(fragments),
                truncated_items=0,
                dropped_items=0,
                unused_bytes=max(0, request.max_bytes - content_bytes),
            )

        content, origins, omissions = self._select_entries(
            request=request,
            current_scope_id=current_scope_id,
            memory_candidates=memory_candidates,
            experience_candidates=experience_candidates,
            profile_candidates=profile_candidates,
            topic_memory_hits=topic_memory_hits,
        )
        content_bytes = 0 if content is None else len(content.encode("utf-8"))
        return RecallBudgetView(
            max_bytes=request.max_bytes,
            delivered_items=len(origins),
            truncated_items=omissions.truncated_items,
            dropped_items=omissions.dropped_items,
            unused_bytes=max(0, request.max_bytes - content_bytes),
        )

    def _select_entries(
        self,
        *,
        request: PrepareContextRequest,
        current_scope_id: str | None,
        memory_candidates: Sequence[PreparedMemoryCandidates],
        topic_memory_hits: Sequence[TopicMemorySearchHit],
        experience_candidates: Sequence[PreparedExperienceCandidates],
        profile_candidates: Sequence[PreparedProfileCandidate],
    ) -> tuple[str | None, tuple[PreparedContextOrigin, ...], PreparedContextOmissions]:
        """Run one complete pure selection pass and render it.

        Returns ``(content, origins, omissions)``; ``content`` is ``None`` exactly when nothing
        was selected, which is the empty-result case for both paths. This is the single
        selection path shared by :meth:`build_scopes_result` and :meth:`probe_budget`, so the
        probe's counters always describe the selection the build would actually perform.
        """

        if request.assembly is not None:
            return self._select_text(
                request,
                current_scope_id=current_scope_id,
                memory_candidates=memory_candidates,
                experience_candidates=experience_candidates,
                profile_candidates=profile_candidates,
                topic_memory_hits=topic_memory_hits,
            )

        memory_entries = _interleave_groups(
            tuple(
                self._memory_entries(
                    candidates.memory_ref,
                    candidates.hits,
                    scope_id=None if candidates.scope_id == current_scope_id else candidates.scope_id or None,
                )
                for candidates in memory_candidates
            )
        )
        topic_memory_entries = self._topic_memory_entries(topic_memory_hits)
        experience_entries = _interleave_groups(
            tuple(
                self._experience_entries(
                    candidates.hits,
                    scope_id=None if candidates.scope_id == current_scope_id else candidates.scope_id or None,
                )
                for candidates in experience_candidates
            )
        )[: self.experience_entry_limit]
        entries, omissions = self._fit_entries(request, memory_entries, topic_memory_entries, experience_entries)
        if not entries:
            return None, (), omissions
        return _render(entries), tuple(entry.origin for entry in entries), omissions

    def _select_text(
        self,
        request: PrepareContextRequest,
        *,
        current_scope_id: str | None,
        memory_candidates: Sequence[PreparedMemoryCandidates],
        experience_candidates: Sequence[PreparedExperienceCandidates],
        profile_candidates: Sequence[PreparedProfileCandidate],
        topic_memory_hits: Sequence[TopicMemorySearchHit],
    ) -> tuple[str | None, tuple[PreparedContextOrigin, ...], PreparedContextOmissions]:
        assembly = request.assembly
        if assembly is None:
            raise PreparedContextInvariantError("text-assembly-missing")
        included: list[ContextTextItem] = []
        origins: list[PreparedContextOrigin] = []
        truncated_items = 0
        dropped_below_min_bytes = 0
        dropped_no_fitting_truncation = 0
        for section in assembly.sections:
            # Unfilled earlier sections leave their capacity available to later ones.
            limit = (
                section.limit
                if self._text_entry_limit is None
                else min(section.limit, self._text_entry_limit - len(included))
            )
            if section.family == "profile":
                entries = self._profile_entries(profile_candidates)
            elif section.family == "topic-memory":
                entries = self._topic_memory_entries(topic_memory_hits, scope_id=current_scope_id)
            else:
                groups = (
                    tuple(
                        tuple(
                            self._memory_entries(group.memory_ref, (hit,), scope_id=group.scope_id)
                            for hit in group.hits
                        )
                        for group in memory_candidates
                    )
                    if section.family == "memory"
                    else tuple(
                        tuple(self._experience_entries((hit,), scope_id=group.scope_id) for hit in group.hits)
                        for group in experience_candidates
                    )
                )
                entries = tuple(chain.from_iterable(_interleave_groups(groups)))
            seen: set[tuple[str, str, str, int, str | None, str | None]] = set()
            rank = 0
            selected_count = 0
            for entry in entries:
                if selected_count >= limit:
                    break
                item = _text_item(entry)
                artifact = item.artifact
                identity = (
                    artifact.scope_id,
                    artifact.artifact.family,
                    artifact.artifact.artifact_id,
                    artifact.artifact.revision,
                    item.entry_id,
                    item.entry_version_id,
                )
                if identity in seen or not item.content.strip():
                    continue
                seen.add(identity)
                rank += 1
                outcome = fit_context_text_item(included, replace(item, recall_rank=rank), assembly, request.max_bytes)
                if outcome.item is None:
                    dropped_below_min_bytes += int(outcome.dropped_below_min_bytes)
                    dropped_no_fitting_truncation += int(outcome.dropped_no_fitting_truncation)
                    continue
                fitted = outcome.item
                included.append(fitted)
                origins.append(entry.origin)
                selected_count += 1
                truncated_items += int(fitted.truncated)
        omissions = PreparedContextOmissions(
            truncated_items=truncated_items,
            dropped_items=dropped_below_min_bytes + dropped_no_fitting_truncation,
            dropped_below_min_bytes=dropped_below_min_bytes,
            dropped_no_fitting_truncation=dropped_no_fitting_truncation,
        )
        if not included:
            return None, (), omissions
        content = render_context_text(included, assembly)
        return content, tuple(origins), omissions

    def _profile_entries(self, candidates: Sequence[PreparedProfileCandidate]) -> tuple[_PreparedContextEntry, ...]:
        entries = []
        for candidate in candidates:
            origin = ArtifactAddress(scope_id=candidate.scope_id, artifact=candidate.profile.as_ref())
            entries.append(
                _PreparedContextEntry(
                    origin=origin,
                    kind="profile",
                    citation={"artifact": origin.model_dump(mode="json")},
                    content=candidate.profile.content.content,
                    truncated=False,
                )
            )
        return tuple(entries)

    def _memory_entries(
        self,
        memory_ref: ArtifactRef | None,
        hits: Sequence[MemoryHit],
        *,
        scope_id: str | None = None,
    ) -> tuple[_PreparedContextEntry, ...]:
        if hits and memory_ref is None:
            raise PreparedContextInvariantError("memory-ref-missing")
        memory_entries: list[_PreparedContextEntry] = []
        seen: set[tuple[str, str]] = set()
        for hit in hits:
            if hit.memory_ref != memory_ref:
                raise PreparedContextInvariantError("memory-ref-mismatch")

            citation_key = (hit.entry_id, hit.entry_version_id)
            if citation_key in seen:
                continue
            seen.add(citation_key)
            if not hit.entry_id.strip() or not hit.entry_version_id.strip() or not hit.text.strip():
                continue
            citation = MemoryCitation(
                memory_ref=hit.memory_ref,
                entry_id=hit.entry_id,
                entry_version_id=hit.entry_version_id,
            )
            origin: PreparedContextOrigin = citation
            rendered_citation = citation.model_dump(mode="json")
            if scope_id is not None:
                memory = ArtifactAddress(scope_id=scope_id, artifact=hit.memory_ref)
                origin = MemoryEntryAddress(
                    memory=memory,
                    entry_id=hit.entry_id,
                    entry_version_id=hit.entry_version_id,
                )
                rendered_citation = {
                    "memory": memory.model_dump(mode="json"),
                    "entry_id": hit.entry_id,
                    "entry_version_id": hit.entry_version_id,
                }
            memory_entries.append(
                _PreparedContextEntry(
                    origin=origin,
                    kind="memory",
                    citation=rendered_citation,
                    content=hit.text,
                    truncated=False,
                )
            )
        return tuple(memory_entries)

    def _evidence_fragments(
        self,
        request: PrepareContextRequest,
        current_scope_id: str | None,
        memory_candidates: Sequence[PreparedMemoryCandidates],
    ) -> tuple[PreparedFragment, ...]:
        """Select literal excerpts of the recalled Memory entries for ``request.query``.

        Evidence selection is defined for the family that carries literal text, so only Memory
        takes part and Code, TopicMemory and Experience keep their own paths. A candidate group
        without a Scope identity is skipped, because a citation that cannot name its Scope cannot
        be checked against the original entry.
        """

        addresses: list[MemoryEntryAddress] = []
        texts: list[str] = []
        for candidates in memory_candidates:
            scope_id = candidates.scope_id or current_scope_id
            if not scope_id:
                continue
            for entry in self._memory_entries(candidates.memory_ref, candidates.hits, scope_id=scope_id):
                if not isinstance(entry.origin, MemoryEntryAddress):
                    continue
                addresses.append(entry.origin)
                texts.append(entry.content)
        if not texts:
            return ()
        return _EvidenceSelector(
            query=request.query,
            addresses=tuple(addresses),
            texts=tuple(texts),
            max_bytes=request.max_bytes,
        ).run()

    def _topic_memory_entries(
        self,
        hits: Sequence[TopicMemorySearchHit],
        *,
        scope_id: str | None = None,
    ) -> tuple[_PreparedContextEntry, ...]:
        topic_entries: list[_PreparedContextEntry] = []
        seen_topics: set[tuple[str, int]] = set()
        for hit in hits:
            if hit.artifact_ref.family != TopicMemory.family:
                raise PreparedContextInvariantError("topic-memory-family-mismatch")
            identity = (hit.artifact_ref.artifact_id, hit.artifact_ref.revision)
            if identity in seen_topics:
                continue
            seen_topics.add(identity)
            if len(topic_entries) >= self.topic_memory_entry_limit:
                break
            content = {"title": hit.title, "summary": hit.summary}
            if hit.snippet is not None:
                content["snippet"] = hit.snippet
            origin: PreparedContextOrigin = hit.artifact_ref
            citation = {"artifact_ref": hit.artifact_ref.model_dump(mode="json")}
            body = json.dumps(content, ensure_ascii=False, separators=(",", ":"))
            if scope_id is not None:
                origin = ArtifactAddress(scope_id=scope_id, artifact=hit.artifact_ref)
                citation = {"artifact": origin.model_dump(mode="json")}
                body = "\n\n".join(f"{label.capitalize()}: {value}" for label, value in content.items())
            topic_entries.append(
                _PreparedContextEntry(
                    origin=origin,
                    kind="topic-memory",
                    citation=citation,
                    content=body,
                    truncated=False,
                )
            )
        return tuple(topic_entries)

    def _experience_entries(
        self,
        hits: Sequence[ExperienceSearchHit],
        *,
        scope_id: str | None = None,
    ) -> tuple[_PreparedContextEntry, ...]:
        experience_entries: list[_PreparedContextEntry] = []
        seen_experiences: set[tuple[str, int]] = set()
        for hit in hits:
            if hit.artifact_ref.family != Experience.family:
                raise PreparedContextInvariantError("experience-family-mismatch")
            identity = (hit.artifact_ref.artifact_id, hit.artifact_ref.revision)
            if identity in seen_experiences:
                continue
            seen_experiences.add(identity)
            origin: PreparedContextOrigin = hit.artifact_ref
            rendered_citation: dict[str, object] = {"artifact_ref": hit.artifact_ref.model_dump(mode="json")}
            if scope_id is not None:
                origin = ArtifactAddress(scope_id=scope_id, artifact=hit.artifact_ref)
                rendered_citation = {"artifact": origin.model_dump(mode="json")}
            experience_entries.append(
                _PreparedContextEntry(
                    origin=origin,
                    kind="experience",
                    citation=rendered_citation,
                    content=render_experience(hit.content),
                    truncated=False,
                )
            )
        return tuple(experience_entries)

    def _fit_entries(
        self,
        request: PrepareContextRequest,
        memory_entries: Sequence[_PreparedContextEntry],
        topic_memory_entries: Sequence[_PreparedContextEntry],
        experience_entries: Sequence[_PreparedContextEntry],
    ) -> tuple[tuple[_PreparedContextEntry, ...], PreparedContextOmissions]:
        entries: list[_PreparedContextEntry] = []
        truncated_items = 0
        dropped_below_min_bytes = 0
        dropped_no_fitting_truncation = 0
        for candidate in _interleave(memory_entries, topic_memory_entries, experience_entries):
            if len(entries) >= self.entry_limit:
                break
            fit = self._fit_entry(
                entries,
                origin=candidate.origin,
                kind=candidate.kind,
                citation=candidate.citation,
                text=candidate.content,
                max_bytes=request.max_bytes,
            )
            if fit.entry is None:
                dropped_below_min_bytes += int(fit.dropped_below_min_bytes)
                dropped_no_fitting_truncation += int(fit.dropped_no_fitting_truncation)
                continue
            entries.append(fit.entry)
            if fit.entry.truncated:
                truncated_items += 1
        return tuple(entries), PreparedContextOmissions(
            truncated_items=truncated_items,
            dropped_items=dropped_below_min_bytes + dropped_no_fitting_truncation,
            dropped_below_min_bytes=dropped_below_min_bytes,
            dropped_no_fitting_truncation=dropped_no_fitting_truncation,
        )

    def _fit_entry(
        self,
        entries: Sequence[_PreparedContextEntry],
        *,
        origin: PreparedContextOrigin,
        kind: str,
        citation: dict[str, object],
        text: str,
        max_bytes: int,
    ) -> _EntryFit:
        """Fit one entry into the remaining budget, reporting why it was dropped if it was.

        The two ``None`` paths of the original code are kept apart: ``below-min-bytes`` means a
        shorter rendering exists and fits but is too short to deliver honestly, while
        ``no-fitting-truncation`` means no shortened rendering fitted at all.
        """

        source_bytes = len(text.encode("utf-8"))
        entry_budget = min(source_bytes, self.max_entry_content_bytes)
        candidate = _PreparedContextEntry(
            origin=origin,
            kind=kind,
            citation=citation,
            content=text if source_bytes <= entry_budget else _truncate_utf8(text, entry_budget),
            truncated=source_bytes > entry_budget,
        )
        if _rendered_bytes((*entries, candidate)) <= max_bytes:
            return _EntryFit(entry=candidate)
        if source_bytes < _MIN_TRUNCATED_CONTENT_BYTES:
            return _EntryFit(dropped_below_min_bytes=True)

        lower = _MIN_TRUNCATED_CONTENT_BYTES
        upper = min(entry_budget, source_bytes - 1)
        best: _PreparedContextEntry | None = None
        while lower <= upper:
            byte_budget = (lower + upper) // 2
            candidate = _PreparedContextEntry(
                origin=origin,
                kind=kind,
                citation=citation,
                content=_truncate_utf8(text, byte_budget),
                truncated=True,
            )
            if len(candidate.content.encode("utf-8")) < _MIN_TRUNCATED_CONTENT_BYTES:
                lower = byte_budget + 1
                continue
            if _rendered_bytes((*entries, candidate)) <= max_bytes:
                best = candidate
                lower = byte_budget + 1
            else:
                upper = byte_budget - 1
        if best is None:
            return _EntryFit(dropped_no_fitting_truncation=True)
        return _EntryFit(entry=best)


def _render_fragments(fragments: Sequence[PreparedFragment]) -> str:
    envelope = {"trust": "untrusted_history", "items": [_render_fragment(fragment) for fragment in fragments]}
    encoded = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
    return "\n\n".join((TRUST_POLICY, f"{_BEGIN_MARKER}\n{encoded}\n{_END_MARKER}"))


def _render_fragment(fragment: PreparedFragment) -> dict[str, object]:
    """Render one excerpt: the exact citation of its entry version, its text and its range."""

    address = fragment.address
    return {
        "citation": {
            "scope_id": address.memory.scope_id,
            "memory_ref": address.memory.artifact.model_dump(mode="json"),
            "entry_id": address.entry_id,
            "entry_version_id": address.entry_version_id,
        },
        "content": fragment.text,
        "start": fragment.start,
        "end": fragment.end,
    }


def _fragment_bytes(fragment: PreparedFragment) -> int:
    return len(json.dumps(_render_fragment(fragment), ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


# Everything the output costs before the first excerpt: the trust policy, both envelope markers
# and the JSON shell around the item list. Measured from the renderer itself, never from a literal.
_EVIDENCE_FIXED_BYTES = len(_render_fragments(()).encode("utf-8"))


def _merge_spans(spans: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Join overlapping or nearly touching slices so one citation covers both of them."""

    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start - merged[-1][1] <= _EVIDENCE_MERGE_GAP:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple((start, end) for start, end in merged)


class _EvidenceSelector:
    """Grow literal excerpts around the sentences that answer one query, inside one budget.

    Rank every literal sentence once, then fill the budget in four passes: whole relevant
    sentences, relevant cuts, then the same two for whatever is left. Within a pass the leading
    document is served first and grows towards its neighbours, so the text arrives as long
    contiguous slices instead of one citation per sentence. Every range is a plain character
    interval of the original entry: nothing here rewrites, reorders or summarises.
    """

    def __init__(
        self,
        *,
        query: str,
        addresses: tuple[MemoryEntryAddress, ...],
        texts: tuple[str, ...],
        max_bytes: int,
    ) -> None:
        self.addresses = addresses
        self.texts = texts
        self.max_bytes = max_bytes
        self.sentences = rank_evidence(query, texts)
        # The excerpt, not the entry, carries the fixed citation cost, so a budget this size
        # buys roughly one document's worth of text. That text is spent on the document whose
        # sentences answer the query best, because a thinner slice of a worse document still
        # cannot reconstruct evidence the leader's window already covers.
        self.leaders = tuple(
            entry
            for _, entry in sorted(
                (-max((sentence.score for sentence in group), default=0.0), entry)
                for entry, group in enumerate(self.sentences)
            )
        )
        self.spans: dict[int, list[tuple[int, int]]] = {}
        self.measured: dict[tuple[int, int, int], int] = {}

    def run(self) -> tuple[PreparedFragment, ...]:
        # Relevant whole sentences first, then relevant cuts, and only after relevance is
        # exhausted the same two passes for the rest: a cheap irrelevant sentence must never
        # take bytes that a relevant one could still use, while nothing that does fit is left
        # over at the end.
        while self._grow(relevant_only=True) and self._count() < _EVIDENCE_FRAGMENT_LIMIT:
            pass
        while self._fill(relevant_only=True) and self._count() < _EVIDENCE_FRAGMENT_LIMIT:
            pass
        while self._grow(relevant_only=False) and self._count() < _EVIDENCE_FRAGMENT_LIMIT:
            pass
        while self._fill(relevant_only=False) and self._count() < _EVIDENCE_FRAGMENT_LIMIT:
            pass
        return self.fragments()

    def fragments(self) -> tuple[PreparedFragment, ...]:
        """Every selected range, in entry order, as a literal slice of the original text."""

        fragments: list[PreparedFragment] = []
        for entry in sorted(self.spans):
            for start, end in _merge_spans(self.spans[entry]):
                fragments.append(
                    PreparedFragment(
                        address=self.addresses[entry],
                        start=start,
                        end=end,
                        text=self.texts[entry][start:end],
                    )
                )
        return tuple(fragments)

    def _grow(self, *, relevant_only: bool) -> bool:
        """Append one whole sentence of the leading document that still fits."""

        for entry in self.leaders:
            sentence = self._next_sentence(entry, relevant_only=relevant_only)
            if sentence is None:
                continue
            if self._measure(self._extend(entry, sentence.start, sentence.end)) > self.max_bytes:
                continue
            self.spans.setdefault(entry, []).append((sentence.start, sentence.end))
            return True
        return False

    def _fill(self, *, relevant_only: bool) -> bool:
        """Append the longest word-aligned cut of the best remaining sentence that still fits."""

        pending = [
            sentence
            for sentence in (
                self._next_sentence(entry, relevant_only=relevant_only) for entry in range(len(self.sentences))
            )
            if sentence
        ]
        for sentence in sorted(pending, key=lambda item: (-item.score, item.entry, item.index)):
            grown = self._cut(sentence)
            if grown is None:
                continue
            self.spans.setdefault(sentence.entry, []).append(grown)
            return True
        return False

    def _next_sentence(self, entry: int, *, relevant_only: bool) -> EvidenceSentence | None:
        """The best sentence of one entry that no slice covers yet.

        A sentence that touches a slice is preferred because it merges into that slice instead of
        paying a second citation, but only among sentences that are still worth their bytes: when
        the neighbour says nothing about the query, a relevant sentence elsewhere wins.
        """

        uncovered = [sentence for sentence in self.sentences[entry] if not self._covered(entry, sentence)]
        if relevant_only:
            uncovered = [sentence for sentence in uncovered if sentence.score > 0.0]
        if not uncovered:
            return None
        touching = [sentence for sentence in uncovered if self._touching(entry, sentence)]
        return max(touching or uncovered, key=lambda sentence: (sentence.score, -sentence.index))

    def _covered(self, entry: int, sentence: EvidenceSentence) -> bool:
        return any(start <= sentence.start and sentence.end <= end for start, end in self.spans.get(entry, ()))

    def _touching(self, entry: int, sentence: EvidenceSentence) -> bool:
        for start, end in self.spans.get(entry, ()):
            if sentence.start - end <= _EVIDENCE_MERGE_GAP and start - sentence.end <= _EVIDENCE_MERGE_GAP:
                return True
        return False

    def _reach(self, entry: int, sentence: EvidenceSentence) -> int:
        """How far the slices of this entry already reach into ``sentence`` from the left."""

        reach = 0
        for start, end in self.spans.get(entry, ()):
            if start < sentence.end and end > sentence.start:
                reach = max(reach, end)
        return reach

    def _edge(self, entry: int, sentence: EvidenceSentence) -> int:
        """Where the slices of this entry first touch ``sentence``, seen from the right."""

        edge = sentence.end
        for start, end in self.spans.get(entry, ()):
            if start < sentence.end and end > sentence.start:
                edge = min(edge, start)
        return edge

    def _cut(self, sentence: EvidenceSentence) -> tuple[int, int] | None:
        """The longest word-aligned piece of one sentence that still fits the budget.

        Both ends are tried because the budget usually runs out in the middle of a sentence, and
        which side of it is still free depends on where the neighbouring slices ended up.
        """

        right = self._cut_right(sentence)
        left = self._cut_left(sentence)
        if right is None or left is None:
            return right if left is None else left
        return right if right[1] - right[0] >= left[1] - left[0] else left

    def _cut_right(self, sentence: EvidenceSentence) -> tuple[int, int] | None:
        entry = sentence.entry
        floor = max(sentence.start + _EVIDENCE_MIN_PARTIAL_CHARS, self._reach(entry, sentence) + 1)
        if floor >= sentence.end:
            return None
        low, high, best = floor, sentence.end, 0
        while low <= high:
            middle = (low + high) // 2
            if self._measure(self._extend(entry, sentence.start, middle)) <= self.max_bytes:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if not best:
            return None
        return (sentence.start, self._snap_forward(entry, best - 1, floor))

    def _cut_left(self, sentence: EvidenceSentence) -> tuple[int, int] | None:
        entry = sentence.entry
        ceiling = min(sentence.end, self._edge(entry, sentence)) - _EVIDENCE_MIN_PARTIAL_CHARS
        if ceiling <= sentence.start:
            return None
        low, high, best = sentence.start, ceiling, 0
        while low <= high:
            middle = (low + high) // 2
            if self._measure(self._extend(entry, middle, sentence.end)) <= self.max_bytes:
                best = middle
                high = middle - 1
            else:
                low = middle + 1
        if not best:
            return None
        return (self._snap_back(entry, best, ceiling), sentence.end)

    def _snap_forward(self, entry: int, end: int, floor: int) -> int:
        """Move a right edge back to the enclosing word boundary, never below ``floor``."""

        text = self.texts[entry]
        for index in range(end, floor - 1, -1):
            if text[index].isspace():
                return index
        return end + 1

    def _snap_back(self, entry: int, start: int, ceiling: int) -> int:
        """Move a left edge forward to the enclosing word boundary, never above ``ceiling``."""

        text = self.texts[entry]
        for index in range(start, ceiling + 1):
            if text[index].isspace():
                return index + 1
        return start

    def _extend(self, entry: int, start: int, end: int) -> dict[int, list[tuple[int, int]]]:
        trial = dict(self.spans)
        trial[entry] = [*trial.get(entry, ()), (start, end)]
        return trial

    def _count(self) -> int:
        return sum(len(_merge_spans(spans)) for spans in self.spans.values())

    def _measure(self, spans: dict[int, list[tuple[int, int]]]) -> int:
        """Exact bytes of the rendering these spans would produce, not of the excerpts alone.

        The trust policy, the envelope markers, the JSON around the items and the commas between
        them all spend the caller's budget, so each item is measured with the same ``json.dumps``
        call the final render uses and the fixed part is measured once at import time.
        """

        total = 0
        count = 0
        for entry in sorted(spans):
            for start, end in _merge_spans(spans[entry]):
                total += self._item_bytes(entry, start, end)
                count += 1
        if not count:
            return _EVIDENCE_FIXED_BYTES
        return _EVIDENCE_FIXED_BYTES + total + count - 1

    def _item_bytes(self, entry: int, start: int, end: int) -> int:
        key = (entry, start, end)
        cached = self.measured.get(key)
        if cached is None:
            excerpt = PreparedFragment(
                address=self.addresses[entry],
                start=start,
                end=end,
                text=self.texts[entry][start:end],
            )
            cached = _fragment_bytes(excerpt)
            self.measured[key] = cached
        return cached


def _interleave(
    memory: Sequence[_PreparedContextEntry],
    topics: Sequence[_PreparedContextEntry],
    experiences: Sequence[_PreparedContextEntry],
) -> tuple[_PreparedContextEntry, ...]:
    """Preserve each family's rank while alternating Memory, Topic, Experience."""

    ordered: list[_PreparedContextEntry] = []
    for index in range(max(len(memory), len(topics), len(experiences))):
        if index < len(memory):
            ordered.append(memory[index])
        if index < len(topics):
            ordered.append(topics[index])
        if index < len(experiences):
            ordered.append(experiences[index])
    return tuple(ordered)


def _interleave_groups(groups: Sequence[Sequence[_Item]]) -> tuple[_Item, ...]:
    ordered: list[_Item] = []
    for index in range(max((len(group) for group in groups), default=0)):
        ordered.extend(group[index] for group in groups if index < len(group))
    return tuple(ordered)


def _render(entries: Sequence[_PreparedContextEntry]) -> str:
    envelope = {
        "trust": "untrusted_history",
        "items": [_render_entry(entry) for entry in entries],
    }
    encoded = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
    return "\n\n".join((TRUST_POLICY, f"{_BEGIN_MARKER}\n{encoded}\n{_END_MARKER}"))


def _text_item(entry: _PreparedContextEntry) -> ContextTextItem:
    origin = entry.origin
    if isinstance(origin, MemoryEntryAddress):
        return ContextTextItem(
            artifact=origin.memory,
            content=entry.content,
            recall_rank=0,
            entry_id=origin.entry_id,
            entry_version_id=origin.entry_version_id,
        )
    if isinstance(origin, ArtifactAddress):
        return ContextTextItem(artifact=origin, content=entry.content, recall_rank=0)
    raise PreparedContextInvariantError("text-scope-missing")


def _render_entry(entry: _PreparedContextEntry) -> dict[str, object]:
    rendered: dict[str, object] = {
        "citation": entry.citation,
        "content": entry.content,
        "truncated": entry.truncated,
    }
    if entry.kind != "memory":
        rendered["kind"] = entry.kind
    return rendered


def _rendered_bytes(entries: Sequence[_PreparedContextEntry]) -> int:
    return len(_render(entries).encode("utf-8"))


def _truncate_utf8(text: str, byte_budget: int) -> str:
    prefix_budget = byte_budget - len(_ELLIPSIS.encode("utf-8"))
    encoded_prefix = text.encode("utf-8")[:prefix_budget]
    return f"{encoded_prefix.decode('utf-8', errors='ignore')}{_ELLIPSIS}"
