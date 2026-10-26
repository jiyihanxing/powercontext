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

"""Cross-encoder reranking through a caller-supplied HTTP scoring service.

The service caps how many candidates one request may carry, so the client splits one
coarse candidate pool into bounded batches. Batches keep the *global* one-based ranks
of the full pool: a batch never renumbers its candidates, which is what lets the
response scores be mapped back onto the original hits after the fact. Selection then
happens once, over the union of every batch, so a high-scoring candidate that was sent
in a later batch can still win the top slot.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Any

import httpx

from powercontext.builtin.artifacts.memory.models import MemoryHit
from powercontext.builtin.artifacts.memory.reranking import (
    MemoryRerankDecision,
    _validate_bounds,
)
from powercontext.builtin.inference.models import InferenceUsage

HTTP_RERANK_POLICY_ID = "http-cross-encoder-v1"


class HTTPMemoryReranker:
    """Score one coarse candidate pool with a batched HTTP cross-encoder.

    The reranker never mutates candidate identity: it only reports original global ranks.
    Transport failures and non-2xx responses are *not* translated, retried, or swallowed —
    httpx exceptions reach the caller unchanged so an outage stays visible instead of
    silently degrading into a different ordering. A 2xx body that does not honour the
    scoring schema is a ValueError, because that is a protocol violation rather than a
    transport condition.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        client: httpx.AsyncClient,
        batch_size: int = 32,
        policy_id: str = HTTP_RERANK_POLICY_ID,
    ) -> None:
        if not endpoint.strip():
            raise ValueError("rerank endpoint must be a non-empty URL")  # noqa: TRY003
        if batch_size < 1:
            raise ValueError("rerank batch_size must be positive")  # noqa: TRY003
        self.policy_id = policy_id
        self._endpoint = endpoint
        self._client = client
        self._batch_size = batch_size

    async def rerank(
        self,
        query: str,
        candidates: tuple[MemoryHit, ...],
        limit: int,
        /,
    ) -> MemoryRerankDecision:
        """Return the globally best limit original ranks by service score."""

        _validate_bounds(len(candidates), limit)
        scores: dict[int, float] = {}
        requests = 0
        for start in range(0, len(candidates), self._batch_size):
            batch = candidates[start : start + self._batch_size]
            expected = tuple(range(start + 1, start + 1 + len(batch)))
            response = await self._client.post(self._endpoint, json=_request_body(query, batch, expected))
            requests += 1
            response.raise_for_status()
            scores.update(_validated_scores(response, expected))
        # One global selection over the whole pool; ties break on the original coarse rank
        # so equal service scores keep the retrieval order instead of the batch order.
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return MemoryRerankDecision(
            selected_ranks=tuple(rank for rank, _score in ordered[:limit]),
            usage=InferenceUsage(requests=requests),
        )


def _request_body(query: str, batch: Sequence[MemoryHit], ranks: tuple[int, ...]) -> dict[str, Any]:
    """Build one request whose candidate ranks stay global to the full pool."""

    return {
        "query": query,
        "candidates": [{"rank": rank, "text": candidate.text} for rank, candidate in zip(ranks, batch, strict=True)],
    }


def _validated_scores(response: httpx.Response, expected_ranks: tuple[int, ...]) -> dict[int, float]:
    """Validate one response body and return exactly the ranks this batch sent."""

    body = response.json()
    if not isinstance(body, Mapping) or set(body) != {"scores"}:
        raise ValueError("rerank response must be a JSON object whose only field is scores")  # noqa: TRY003
    rows = body["scores"]
    if not isinstance(rows, list):
        raise ValueError("rerank response scores must be a JSON array")  # noqa: TRY003, TRY004
    expected = set(expected_ranks)
    scores: dict[int, float] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"rank", "score"}:
            raise ValueError("rerank score records must contain exactly rank and score")  # noqa: TRY003
        rank = row["rank"]
        if type(rank) is not int or rank not in expected:
            # A rank outside this batch means the service renumbered or crossed batches.
            raise ValueError(f"rerank score rank {rank!r} is not a candidate of this batch")  # noqa: TRY003
        if rank in scores:
            raise ValueError(f"rerank score rank {rank} was reported twice")  # noqa: TRY003
        score = row["score"]
        if isinstance(score, bool) or not isinstance(score, int | float) or not isfinite(score):
            raise ValueError(f"rerank score for rank {rank} must be a finite number")  # noqa: TRY003
        scores[rank] = float(score)
    if scores.keys() != expected:
        missing = sorted(expected - scores.keys())
        raise ValueError(f"rerank response did not cover these ranks: {missing}")  # noqa: TRY003
    return scores


__all__ = ["HTTP_RERANK_POLICY_ID", "HTTPMemoryReranker"]
