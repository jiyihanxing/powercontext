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

"""Evidence-dependency lifecycle persistence.

One row holds one registered node: what it refers to, which parents support it,
and the availability the last committed event left it with. The graph is
immutable after registration (registration order is the topological order), so
the only mutable columns are ``available`` and the Source event ``sequence``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, TypeAdapter
from sqlalchemy import insert, select, update
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.artifacts import ArtifactRef, MemoryCitation
from powercontext.builtin.persistence.codec import dump_model, load_model, stored_bytes
from powercontext.builtin.persistence.database import AsyncDatabase
from powercontext.builtin.persistence.errors import InvalidRepositoryArgumentError, RepositoryNotFoundError
from powercontext.builtin.persistence.tables import SOURCE_LIFECYCLE_NODES_TABLE
from powercontext.limits import MAX_SCOPE_ID_LENGTH, MAX_SOURCE_LIFECYCLE_NODE_ID_LENGTH
from powercontext.sources import SourceRef

LifecycleFamily: TypeAlias = Literal["source", "memory", "topic"]
LifecyclePolicy: TypeAlias = Literal["all", "any"]
LifecycleReference: TypeAlias = SourceRef | MemoryCitation | ArtifactRef

_REFERENCE_MODELS: dict[str, type[BaseModel]] = {
    "source": SourceRef,
    "memory": MemoryCitation,
    "topic": ArtifactRef,
}
_PARENTS_ADAPTER: TypeAdapter[tuple[str, ...]] = TypeAdapter(tuple[str, ...])


class StoredLifecycleNode(BaseModel):
    """One registered node together with its committed availability."""

    node_id: str
    family: LifecycleFamily
    reference: LifecycleReference
    parents: tuple[str, ...] = ()
    policy: LifecyclePolicy | None = None
    revision: int
    sequence: int
    available: bool
    ordinal: int


class SourceLifecycleRepository:
    """Read and write one Scope's evidence graph through explicit transactions."""

    def __init__(self, database: AsyncDatabase) -> None:
        self._database = database

    async def load(self, scope_id: str, /) -> tuple[StoredLifecycleNode, ...]:
        """Return every registered node in registration (topological) order."""

        _require_scope(scope_id)
        async with self._database.transaction() as connection:
            return await self.load_in(connection, scope_id)

    async def load_in(self, connection: AsyncConnection, scope_id: str, /) -> tuple[StoredLifecycleNode, ...]:
        _require_scope(scope_id)
        rows = (
            await connection.execute(
                select(SOURCE_LIFECYCLE_NODES_TABLE)
                .where(SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id)
                .order_by(SOURCE_LIFECYCLE_NODES_TABLE.c.ordinal)
            )
        ).mappings()
        return tuple(_decode_row(row) for row in rows)

    async def insert(self, scope_id: str, node: StoredLifecycleNode, /) -> None:
        """Append one registered node; registration never rewrites an existing row."""

        _require_scope(scope_id)
        values = _encode_row(scope_id, node)
        async with self._database.transaction() as connection:
            await connection.execute(insert(SOURCE_LIFECYCLE_NODES_TABLE).values(**values))

    async def save(self, scope_id: str, nodes: Sequence[StoredLifecycleNode], /) -> None:
        """Commit new availability (and Source sequence) for changed nodes at once."""

        _require_scope(scope_id)
        if not nodes:
            return
        async with self._database.transaction() as connection:
            for node in nodes:
                result = await connection.execute(
                    update(SOURCE_LIFECYCLE_NODES_TABLE)
                    .where(
                        SOURCE_LIFECYCLE_NODES_TABLE.c.scope_id == scope_id,
                        SOURCE_LIFECYCLE_NODES_TABLE.c.node_id == node.node_id,
                    )
                    .values(available=node.available, sequence=node.sequence)
                )
                if result.rowcount != 1:
                    raise RepositoryNotFoundError("source-lifecycle-node", (scope_id, node.node_id))


def _encode_row(scope_id: str, node: StoredLifecycleNode) -> dict[str, Any]:
    _require_node(node)
    model = _REFERENCE_MODELS[node.family]
    if not isinstance(node.reference, model):
        raise InvalidRepositoryArgumentError(
            "reference",
            f"{node.family!r} nodes require a {model.__name__} reference",
        )
    return {
        "scope_id": scope_id,
        "node_id": node.node_id,
        "family": node.family,
        "policy": node.policy,
        "revision": node.revision,
        "sequence": node.sequence,
        "available": node.available,
        "ordinal": node.ordinal,
        "reference": dump_model(node.reference, kind="source-lifecycle-reference", name=node.node_id),
        "parents": _PARENTS_ADAPTER.dump_json(node.parents),
    }


def _decode_row(row: Mapping[Any, Any]) -> StoredLifecycleNode:
    node_id = str(row["node_id"])
    family = str(row["family"])
    model = _REFERENCE_MODELS.get(family)
    if model is None:
        raise InvalidRepositoryArgumentError("family", f"{family!r} is not a lifecycle family")
    return StoredLifecycleNode(
        node_id=node_id,
        family=family,  # type: ignore[arg-type]  # constrained by _REFERENCE_MODELS above
        reference=load_model(
            model,  # type: ignore[arg-type]  # the family model constrains the stored reference
            stored_bytes(row["reference"], column="reference"),
            kind="source-lifecycle-reference",
            name=node_id,
        ),
        parents=_PARENTS_ADAPTER.validate_json(stored_bytes(row["parents"], column="parents"), strict=True),
        policy=None if row["policy"] is None else str(row["policy"]),  # type: ignore[arg-type]
        revision=int(row["revision"]),
        sequence=int(row["sequence"]),
        available=bool(row["available"]),
        ordinal=int(row["ordinal"]),
    )


def _require_scope(scope_id: object) -> None:
    _require_identifier("scope_id", scope_id, MAX_SCOPE_ID_LENGTH)


def _require_node(node: StoredLifecycleNode) -> None:
    _require_identifier("node_id", node.node_id, MAX_SOURCE_LIFECYCLE_NODE_ID_LENGTH)
    if node.family == "source" and node.revision < 1:
        raise InvalidRepositoryArgumentError("revision", "Source nodes require a positive revision")
    if node.family != "source" and node.revision != 0:
        raise InvalidRepositoryArgumentError("revision", "derived nodes carry no lifecycle revision")
    if node.sequence < 0:
        raise InvalidRepositoryArgumentError("sequence", "must not be negative")
    if node.ordinal < 0:
        raise InvalidRepositoryArgumentError("ordinal", "must not be negative")
    if node.family == "source":
        if node.parents or node.policy is not None:
            raise InvalidRepositoryArgumentError("parents", "Source nodes have no parents or policy")
        return
    if not node.parents:
        raise InvalidRepositoryArgumentError("parents", "derived nodes require at least one parent")
    if node.policy not in {"all", "any"}:
        raise InvalidRepositoryArgumentError("policy", "derived nodes require an 'all' or 'any' policy")
    for parent in node.parents:
        _require_identifier("parents", parent, MAX_SOURCE_LIFECYCLE_NODE_ID_LENGTH)


def _require_identifier(field: str, value: object, maximum: int) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InvalidRepositoryArgumentError(field, "must be a non-empty string")
    if len(value) > maximum:
        raise InvalidRepositoryArgumentError(field, f"must not exceed {maximum} characters")
