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

"""Evidence-dependency lifecycle: withdraw and restore registered content.

A caller declares, per Scope, which Sources support which Memory entries and
which Topic Memory revisions. The graph is a DAG: registration order is the
topological order, because a node may only name parents that are already
registered. A caller-owned event withdraws or restores one Source, and the
``all``/``any`` policy of every descendant decides how far that reaches.

Availability is the only thing this layer changes. A withdrawn Memory entry
keeps its body and history (native deactivation, not deletion) and a withdrawn
Topic Memory Revision keeps its exact bytes; both simply stop being eligible for
generation, search, and Context assembly until support returns.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.persistence.errors import PersistenceError
from powercontext.builtin.persistence.source_lifecycle import (
    LifecycleFamily,
    LifecyclePolicy,
    LifecycleReference,
    SourceLifecycleRepository,
    StoredLifecycleNode,
)
from powercontext.builtin.records import BaseAccessError
from powercontext.builtin.runtime.models import GetMemoryEntryRequest
from powercontext.builtin.sources import validate_scope_id
from powercontext.errors import PowerContextError
from powercontext.limits import MAX_SOURCE_LIFECYCLE_NODE_ID_LENGTH
from powercontext.sources import SourceRef

if TYPE_CHECKING:
    from powercontext.builtin.runtime.application import BuiltinRuntime

RegisteredFamily: TypeAlias = Literal["memory", "topic"]

GENERATION_PURPOSE = "generation"
AUDIT_PURPOSE = "audit"


class SourceLifecycleError(ValueError):
    """Reject an event, a registration, or a read that the contract forbids.

    ``code`` is the stable token a caller may branch on; ``subject`` names the
    node, purpose, or reference the refusal is about.
    """

    def __init__(self, code: str, subject: object | None = None) -> None:
        self.code = code
        self.subject = subject
        super().__init__(code if subject is None else f"{code}: {subject}")


class SourceLifecycleEvent(BaseModel):
    """One caller-owned availability change for one registered Source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    revision: int = Field(ge=1)
    sequence: int = Field(ge=1)
    available: bool
    reason: str

    @field_validator("node_id", "reason")
    @classmethod
    def require_non_blank_text(cls, value: str, info) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be blank")  # noqa: TRY003
        return value


class SourceLifecycleNode(BaseModel):
    """One registered node and the availability the last committed event left it with."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    node_id: str
    family: LifecycleFamily
    reference: LifecycleReference
    parents: tuple[str, ...] = ()
    policy: LifecyclePolicy | None = None
    revision: int
    sequence: int
    available: bool


class SourceLifecycleApplication:
    """Expose the evidence graph of each Scope; it owns no graph of its own."""

    def __init__(self, runtime: BuiltinRuntime, repository: SourceLifecycleRepository) -> None:
        self._runtime = runtime
        self._repository = repository

    def for_scope(self, scope_id: str, /) -> ScopedSourceLifecycle:
        return ScopedSourceLifecycle(self._runtime, self._repository, scope_id)

    async def withdrawn_topics(self, scope_id: str, /) -> frozenset[str]:
        """Return the Topic Memory Artifact ids currently without support in one Scope."""

        nodes = await self._repository.load(validate_scope_id(scope_id))
        withdrawn: set[str] = set()
        for node in nodes:
            if node.family == "topic" and not node.available and isinstance(node.reference, ArtifactRef):
                withdrawn.add(node.reference.artifact_id)
        return frozenset(withdrawn)


class ScopedSourceLifecycle:
    """Register evidence, apply Source events, and resolve references in one Scope."""

    def __init__(
        self,
        runtime: BuiltinRuntime,
        repository: SourceLifecycleRepository,
        scope_id: str,
    ) -> None:
        self._runtime = runtime
        self._repository = repository
        self.scope_id = validate_scope_id(scope_id)

    async def register_source(
        self,
        node_id: str,
        reference: SourceRef,
        /,
        *,
        revision: int,
    ) -> SourceLifecycleNode:
        """Register one captured Source as a lifecycle root at an exact revision."""

        validated_id = _require_node_id(node_id)
        validated_revision = _require_positive_int("revision", revision)
        validated = _require_reference("source", reference)
        graph = await self._graph()
        registered = graph.get(validated_id)
        if registered is not None:
            _require_same_binding(
                registered,
                reference=validated,
                parents=(),
                policy=None,
                revision=validated_revision,
            )
            return _public(registered)
        await self._verify_reference(validated)
        node = StoredLifecycleNode(
            node_id=validated_id,
            family="source",
            reference=validated,
            revision=validated_revision,
            sequence=0,
            available=True,
            ordinal=_next_ordinal(graph),
        )
        await self._repository.insert(self.scope_id, node)
        return _public(node)

    async def register_memory(
        self,
        node_id: str,
        citation: MemoryCitation,
        /,
        *,
        parents: Sequence[str],
        policy: LifecyclePolicy,
    ) -> SourceLifecycleNode:
        """Register one exact Memory entry version as a node derived from ``parents``."""

        return await self._register_derived(
            _require_node_id(node_id),
            _require_reference("memory", citation),
            family="memory",
            parents=parents,
            policy=policy,
        )

    async def register_topic(
        self,
        node_id: str,
        reference: ArtifactRef,
        /,
        *,
        parents: Sequence[str],
        policy: LifecyclePolicy,
    ) -> SourceLifecycleNode:
        """Register one exact Topic Memory Revision as a node derived from ``parents``."""

        return await self._register_derived(
            _require_node_id(node_id),
            _require_reference("topic", reference),
            family="topic",
            parents=parents,
            policy=policy,
        )

    async def apply(self, event: SourceLifecycleEvent, /) -> list[str]:
        """Apply one Source event and return every available node id, sorted."""

        validated = _require_event(event)
        graph = await self._graph()
        ordered = tuple(graph.values())
        source = graph.get(validated.node_id)
        if source is None or source.family != "source":
            raise SourceLifecycleError("unknown-source", validated.node_id)
        if validated.revision != source.revision:
            raise SourceLifecycleError("revision-mismatch", validated.node_id)
        if validated.sequence < source.sequence:
            raise SourceLifecycleError("sequence-regression", validated.node_id)
        if validated.sequence == source.sequence:
            if validated.available != source.available:
                raise SourceLifecycleError("sequence-conflict", validated.node_id)
            return _available_ids(_evaluate(ordered))

        candidate = source.model_copy(update={"available": validated.available, "sequence": validated.sequence})
        availability = _evaluate(_replace(ordered, candidate))
        # The event Source is written once, through ``candidate``: it is the only node
        # whose committed row also carries a new ``sequence``.
        changed = tuple(
            node.model_copy(update={"available": availability[node.node_id]})
            for node in ordered
            if node.family != "source" and availability[node.node_id] != node.available
        )
        # Native state is applied before the graph commits. A failed write therefore leaves
        # the committed event untouched, the retry re-derives the same propagation, and the
        # writes that already landed are idempotent no-ops that add no Memory Revision.
        for node in changed:
            if node.family == "memory" and isinstance(node.reference, MemoryCitation):
                await self._runtime.memory.for_scope(self.scope_id).set_available(
                    node.reference,
                    available=availability[node.node_id],
                    reason=validated.reason,
                )
        await self._repository.save(self.scope_id, (candidate, *changed))
        return _available_ids(availability)

    async def resolve(self, node_id: str, /, *, purpose: str) -> SourceRef | MemoryCitation | ArtifactRef:
        """Return the registered reference under ``generation`` or ``audit`` eligibility."""

        graph = await self._graph()
        node = graph.get(node_id)
        if node is None:
            raise SourceLifecycleError("unknown-node", node_id)
        if purpose == GENERATION_PURPOSE:
            if not node.available:
                raise SourceLifecycleError("unavailable-for-generation", node_id)
        elif purpose != AUDIT_PURPOSE:
            raise SourceLifecycleError("unknown-purpose", purpose)
        return node.reference

    async def _register_derived(
        self,
        node_id: str,
        reference: LifecycleReference,
        *,
        family: RegisteredFamily,
        parents: Sequence[str],
        policy: LifecyclePolicy,
    ) -> SourceLifecycleNode:
        ordered_parents = _require_parents(node_id=node_id, parents=parents)
        validated_policy = _require_policy(policy)
        graph = await self._graph()
        registered = graph.get(node_id)
        if registered is not None:
            _require_same_binding(
                registered,
                reference=reference,
                parents=ordered_parents,
                policy=validated_policy,
                revision=None,
            )
            return _public(registered)
        for parent in ordered_parents:
            if parent not in graph:
                raise SourceLifecycleError("parent-not-registered", f"{parent} (child {node_id})")
        await self._verify_reference(reference)
        available = _evaluate_policy(validated_policy, [graph[parent].available for parent in ordered_parents])
        node = StoredLifecycleNode(
            node_id=node_id,
            family=family,
            reference=reference,
            parents=ordered_parents,
            policy=validated_policy,
            revision=0,
            sequence=0,
            available=available,
            ordinal=_next_ordinal(graph),
        )
        if family == "memory" and isinstance(reference, MemoryCitation):
            # Registration publishes the derived state immediately: the entry of a node
            # whose parents cannot support it is deactivated inside this call.
            await self._runtime.memory.for_scope(self.scope_id).set_available(
                reference,
                available=available,
                reason=f"source lifecycle registration of {node_id}",
            )
        await self._repository.insert(self.scope_id, node)
        return _public(node)

    async def _verify_reference(self, reference: LifecycleReference) -> None:
        """Reject a reference the Scope cannot read at registration time."""

        try:
            if isinstance(reference, SourceRef):
                await self._runtime.records.for_scope(self.scope_id).get_source(
                    reference.source_type,
                    reference.source_id,
                )
            elif isinstance(reference, MemoryCitation):
                await self._runtime.memory.for_scope(self.scope_id).get(
                    GetMemoryEntryRequest(citation=reference),
                )
            else:
                await self._runtime.records.for_scope(self.scope_id).get_artifact_revision(
                    reference.family,
                    reference.artifact_id,
                    reference.revision,
                )
        except (PowerContextError, PersistenceError, BaseAccessError, ValueError) as error:
            raise SourceLifecycleError("unreadable-reference", error) from error

    async def _graph(self) -> dict[str, StoredLifecycleNode]:
        ordered = await self._repository.load(self.scope_id)
        return {node.node_id: node for node in ordered}


def _evaluate(ordered: Sequence[StoredLifecycleNode]) -> dict[str, bool]:
    """Fold availability from Sources to descendants in registration (topological) order."""

    availability: dict[str, bool] = {}
    for node in ordered:
        if node.family == "source":
            availability[node.node_id] = node.available
            continue
        availability[node.node_id] = _evaluate_policy(
            _registered_policy(node),
            [availability[parent] for parent in node.parents],
        )
    return availability


def _evaluate_policy(policy: LifecyclePolicy, support: Sequence[bool]) -> bool:
    return all(support) if policy == "all" else any(support)


def _registered_policy(node: StoredLifecycleNode) -> LifecyclePolicy:
    if node.policy is None:
        raise SourceLifecycleError("derived-node-without-policy", node.node_id)
    return node.policy


def _available_ids(availability: dict[str, bool]) -> list[str]:
    return sorted(node_id for node_id, available in availability.items() if available)


def _replace(
    ordered: Sequence[StoredLifecycleNode],
    replacement: StoredLifecycleNode,
) -> tuple[StoredLifecycleNode, ...]:
    return tuple(replacement if node.node_id == replacement.node_id else node for node in ordered)


def _next_ordinal(graph: dict[str, StoredLifecycleNode]) -> int:
    return max((node.ordinal for node in graph.values()), default=-1) + 1


def _public(node: StoredLifecycleNode) -> SourceLifecycleNode:
    return SourceLifecycleNode(
        node_id=node.node_id,
        family=node.family,
        reference=node.reference,
        parents=node.parents,
        policy=node.policy,
        revision=node.revision,
        sequence=node.sequence,
        available=node.available,
    )


def _require_event(event: object) -> SourceLifecycleEvent:
    if isinstance(event, SourceLifecycleEvent):
        return event
    try:
        return SourceLifecycleEvent.model_validate(event)
    except ValueError as error:
        raise SourceLifecycleError("invalid-event", error) from error


def _require_node_id(node_id: object) -> str:
    if not isinstance(node_id, str) or not node_id.strip():
        raise SourceLifecycleError("invalid-node-id", node_id)
    if len(node_id) > MAX_SOURCE_LIFECYCLE_NODE_ID_LENGTH:
        raise SourceLifecycleError("node-id-too-long", node_id)
    return node_id


def _require_positive_int(field: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise SourceLifecycleError(f"invalid-{field}", value)
    return value


def _require_reference(family: LifecycleFamily, reference: object) -> LifecycleReference:
    expected = {"source": SourceRef, "memory": MemoryCitation, "topic": ArtifactRef}[family]
    if not isinstance(reference, expected):
        raise SourceLifecycleError(f"invalid-{family}-reference", type(reference).__name__)
    return reference


def _require_parents(*, node_id: str, parents: Sequence[str]) -> tuple[str, ...]:
    if isinstance(parents, str) or not isinstance(parents, Sequence):
        raise SourceLifecycleError("invalid-parents", node_id)
    ordered: list[str] = []
    for parent in parents:
        validated = _require_node_id(parent)
        if validated == node_id:
            raise SourceLifecycleError("self-parent", node_id)
        if validated in ordered:
            raise SourceLifecycleError("duplicate-parent", f"{validated} (child {node_id})")
        ordered.append(validated)
    if not ordered:
        raise SourceLifecycleError("missing-parents", node_id)
    return tuple(ordered)


def _require_policy(policy: object) -> LifecyclePolicy:
    if policy not in {"all", "any"}:
        raise SourceLifecycleError("invalid-policy", policy)
    return policy  # type: ignore[return-value]  # narrowed by the membership check above


def _require_same_binding(
    registered: StoredLifecycleNode,
    *,
    reference: LifecycleReference,
    parents: Sequence[str],
    policy: LifecyclePolicy | None,
    revision: int | None,
) -> None:
    if (
        registered.reference != reference
        or registered.parents != tuple(parents)
        or registered.policy != policy
        or (revision is not None and registered.revision != revision)
    ):
        raise SourceLifecycleError("duplicate-binding", registered.node_id)
