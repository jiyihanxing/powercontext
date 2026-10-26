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

"""seekdb Memory candidate retrieval from one native FULLTEXT and vector statement.

seekdb keeps the FULLTEXT index on ``pc_memory_entry_heads`` and the HNSW vector index on
``pc_memory_vector_entries``, and reads both in a single statement when each candidate branch is
its own native index scan. A hybrid candidate query therefore issues exactly one statement: the two
ordered branches are combined with ``UNION ALL`` and each branch keeps its own ``ORDER BY``/``LIMIT``,
so the database still chooses the candidates and the application only fuses the two returned
rankings. Selecting both branches in one statement is enough here; no second candidate query, and no
application-side scoring, is needed to reach the same per-channel rankings.

Scope, Memory identity, tag filter and the candidate limit stay inside the statement, and every
projection row is joined to its exact ``pc_memory_entry_versions`` body. The stored ``head_revision``
is not a filter: an unchanged projection may still carry the revision it was written with, while the
caller-validated current head remains the authority for the returned references.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pyobvector import VECTOR
from sqlalchemy import TextClause, bindparam, text
from sqlalchemy.ext.asyncio import AsyncConnection

from powercontext.builtin.artifacts.memory import (
    CapabilityNotSupportedError,
    MemorySearchChannels,
    MemorySearchRequest,
)
from powercontext.builtin.artifacts.memory.canonical import validate_embedding
from powercontext.builtin.persistence.memory_index import CompositeMemoryIndex, memory_channel_hits
from powercontext.builtin.persistence.oceanbase.memory_index import (
    OceanBaseMemoryFTSIndex,
    OceanBaseMemoryVectorIndex,
)
from powercontext.builtin.persistence.tags import memory_tag_parameters, memory_tag_sql

_FTS_CHANNEL = "fts"
_VECTOR_CHANNEL = "vector"

# Both branches return the same columns: the channel that selected the row, the projection identity,
# the exact entry body, and the ranking value of that channel.
_LEXICAL_CANDIDATES = """
SELECT
    :channel_fts AS channel,
    m.memory_artifact_id,
    m.entry_id,
    m.entry_version_id,
    v.text,
    MATCH (m.searchable_text) AGAINST (:analyzed_query) AS score,
    NULL AS distance
FROM pc_memory_entry_heads AS m
JOIN pc_memory_entry_versions AS v
  ON v.scope_id = m.scope_id
 AND v.memory_artifact_id = m.memory_artifact_id
 AND v.entry_version_id = m.entry_version_id
WHERE m.scope_id = :scope_id
  AND m.memory_artifact_id IN :memory_artifact_ids
  AND MATCH (m.searchable_text) AGAINST (:analyzed_query){tag_filter}
ORDER BY score DESC, m.memory_artifact_id, m.entry_id, m.entry_version_id
LIMIT :candidate_limit
"""

# APPROXIMATE must stay the last ordering token, so this ranking carries no tie-breakers; the caller
# restores a deterministic order over the returned distances. A tag filter keeps the ANN shortcut out
# of the plan, because a top-k approximation cannot honor a correlated filter, exactly as the
# vector-only candidate query already does.
_SEMANTIC_CANDIDATES = """
SELECT
    :channel_vector AS channel,
    m.memory_artifact_id,
    m.entry_id,
    m.entry_version_id,
    v.text,
    NULL AS score,
    l2_distance(m.embedding, :query_vector) AS distance
FROM pc_memory_vector_entries AS m
JOIN pc_memory_entry_versions AS v
  ON v.scope_id = m.scope_id
 AND v.memory_artifact_id = m.memory_artifact_id
 AND v.entry_version_id = m.entry_version_id
WHERE m.scope_id = :scope_id
  AND m.memory_artifact_id IN :memory_artifact_ids{tag_filter}
ORDER BY l2_distance(m.embedding, :query_vector){approximate}
LIMIT :candidate_limit
"""


class SeekDBMemoryIndex(CompositeMemoryIndex):
    """Compose seekdb FULLTEXT and vector candidates into one hybrid candidate statement."""

    def __init__(
        self,
        fts: OceanBaseMemoryFTSIndex,
        vector: OceanBaseMemoryVectorIndex | None = None,
    ) -> None:
        super().__init__(fts, *(() if vector is None else (vector,)))
        self._vector = vector
        self._plain = None if vector is None else _candidate_statement(dimension=vector.profile.dimension, tagged=False)
        self._tagged = None if vector is None else _candidate_statement(dimension=vector.profile.dimension, tagged=True)

    async def search(
        self,
        connection: AsyncConnection,
        scope_id: str,
        request: MemorySearchRequest,
        /,
    ) -> MemorySearchChannels:
        vector = self._vector
        # A tag filter changes both branches, so it selects a separately rendered statement.
        statement = self._tagged if request.tag_filter is not None else self._plain
        if request.mode != "hybrid" or vector is None or statement is None:
            return await super().search(connection, scope_id, request)
        if request.embedding_profile != vector.profile or request.query_vector is None:
            raise CapabilityNotSupportedError("embedding-profile")
        if not await vector.vector_complete(connection, scope_id, request.memories, vector.profile):
            raise CapabilityNotSupportedError("vector")
        rows = (
            (
                await connection.execute(
                    statement,
                    {
                        "channel_fts": _FTS_CHANNEL,
                        "channel_vector": _VECTOR_CHANNEL,
                        "analyzed_query": request.analyzed_query,
                        "query_vector": validate_embedding(
                            request.query_vector,
                            dimension=vector.profile.dimension,
                        ),
                        "scope_id": scope_id,
                        "memory_artifact_ids": tuple(memory.artifact_id for memory in request.memories),
                        "candidate_limit": request.candidate_limit,
                        **memory_tag_parameters(request.tag_filter),
                    },
                )
            )
            .mappings()
            .all()
        )
        return MemorySearchChannels(
            fts=memory_channel_hits(
                _lexical_ranking(tuple(row for row in rows if str(row["channel"]) == _FTS_CHANNEL)),
                request.memories,
            ),
            vector=memory_channel_hits(
                _semantic_ranking(tuple(row for row in rows if str(row["channel"]) == _VECTOR_CHANNEL)),
                request.memories,
            ),
        )


def _candidate_statement(*, dimension: int, tagged: bool) -> TextClause:
    """Render the fused candidate statement for one filter shape and embedding width."""

    tag_filter = memory_tag_sql("m") if tagged else ""
    statement = (
        "("
        + _LEXICAL_CANDIDATES.format(tag_filter=tag_filter)
        + ") UNION ALL ("
        + _SEMANTIC_CANDIDATES.format(
            tag_filter=tag_filter,
            approximate="" if tagged else " APPROXIMATE",
        )
        + ")"
    )
    candidates = text(statement).bindparams(
        bindparam("memory_artifact_ids", expanding=True),
        bindparam("query_vector", type_=VECTOR(dimension)),
    )
    if not tagged:
        return candidates
    return candidates.bindparams(
        bindparam("tag_keys", expanding=True),
        bindparam("tag_hashes", expanding=True),
    )


def _lexical_ranking(rows: Sequence[Mapping[Any, Any]]) -> tuple[Mapping[Any, Any], ...]:
    """Restore the lexical branch order the statement requested."""

    return tuple(
        sorted(
            rows,
            key=lambda row: (
                -float(row["score"]),
                str(row["memory_artifact_id"]),
                str(row["entry_id"]),
                str(row["entry_version_id"]),
            ),
        )
    )


def _semantic_ranking(rows: Sequence[Mapping[Any, Any]]) -> tuple[Mapping[Any, Any], ...]:
    """Break distance ties deterministically after the approximate top-k scan."""

    return tuple(
        sorted(
            rows,
            key=lambda row: (
                float(row["distance"]),
                str(row["memory_artifact_id"]),
                str(row["entry_id"]),
                str(row["entry_version_id"]),
            ),
        )
    )
