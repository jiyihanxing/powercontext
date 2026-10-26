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

"""Cross-call request coalescing and a bounded LRU vector cache for one embedding model.

Memory writes and vector searches reach an embedding provider from independent scopes, and
the same text is routinely requested again while an earlier request for it is still
outstanding. This wrapper turns those independent calls into shared provider work: every
distinct text owns at most one *flight*, a handle with its own event, and flights are
grouped into provider batches of at most ``max_batch_size`` texts.

How the pieces fit together:

* **Deadline window.** A caller never calls the provider itself. It registers a flight for
  each text it needs and returns to the event loop. The first flight of a window arms one
  loop timer; when it fires, every flight queued meanwhile — from one call or from several
  — is drained into batches, so requests arriving inside the same window share a batch.
  ``coalesce_ms=0`` closes the window on the next loop turn, which still batches everything
  one call needs. The drain is a plain loop callback: it holds no ``await``, so no caller
  cancellation can interrupt it and no queued flight can be left behind by a dead task.
* **One flight per text.** The flight is published in the in-flight map synchronously, so a
  caller that arrives while the provider call is running joins it instead of issuing a
  duplicate. Provider work is therefore bounded by *distinct texts*, not by callers.
* **Cache key.** The key is the text alone. A wrapper instance is bound to exactly one
  immutable :class:`~powercontext.builtin.artifacts.memory.models.EmbeddingProfile`, so the
  model, dimension, distance and normalization cannot change underneath a cached vector and
  do not need to be part of the key. The cache is a real LRU: a hit refreshes recency and
  the oldest successful vector is evicted past ``cache_size``.
* **Failure semantics.** A batch either succeeds as a whole or fails as a whole: the raised
  value reaches every waiter that depended on that batch, its flights leave the in-flight
  map, and nothing is cached, so the next caller retries the provider. Batches are
  independent tasks, so a failure in one leaves the others running. A vector count that
  disagrees with the batch is a provider contract violation, reported as
  :class:`InvalidInferenceOutputError` (a ``ValueError``).
* **Cancellation isolation.** Waiters await their own event, never a shared future, and the
  batch task is owned by the wrapper rather than by any caller. Cancelling a caller cancels
  only that caller's await: the provider call keeps running and every other waiter still
  receives its vectors.
* **Usage.** A caller is charged once per *batch it waited on* — duplicates inside one call
  collapse onto their flight, and two callers of the same batch are each charged that
  batch's provider usage, because both of them waited for it. A call served entirely from
  cache reports zero requests and no tokens.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from collections.abc import Sequence
from typing import TYPE_CHECKING

from powercontext.builtin.inference.errors import InvalidInferenceOutputError
from powercontext.builtin.inference.models import EmbeddingResult, EmbeddingVector, InferenceUsage
from powercontext.builtin.inference.protocols import EmbeddingModel

if TYPE_CHECKING:
    from powercontext.builtin.artifacts.memory.models import EmbeddingProfile


class _Flight:
    """The one outstanding computation of a single distinct text.

    ``event`` is per-flight rather than a shared future on purpose: each waiter awaits its
    own event, so cancelling a waiter cannot cancel the value the others are waiting for.
    """

    __slots__ = ("batch", "error", "event", "text", "vector")

    def __init__(self, text: str) -> None:
        self.text = text
        self.batch: _Batch | None = None
        self.event = asyncio.Event()
        self.error: Exception | None = None
        self.vector: EmbeddingVector | None = None


class _Batch:
    """One underlying provider request together with the flights it answers."""

    __slots__ = ("flights", "texts", "usage")

    def __init__(self, flights: Sequence[_Flight]) -> None:
        self.flights = tuple(flights)
        self.texts = tuple(flight.text for flight in self.flights)
        self.usage = InferenceUsage(requests=0)


class CoalescingEmbeddingModel:
    """Share one embedding provider call between every caller that needs its texts.

    The wrapper is safe to inject as ``open_builtin_contexts(..., embedding_model=model)``:
    it exposes the underlying ``profile`` and an asynchronous ``embed(texts)`` that returns
    vectors in the caller's original order, duplicates included. Callers are expected to
    share one event loop, which is what makes coalescing observable; the coordination state
    is rebuilt for a new loop while the vector cache survives, because a cached vector does
    not depend on the loop that produced it.
    """

    def __init__(
        self,
        model: EmbeddingModel,
        *,
        max_batch_size: int = 64,
        coalesce_ms: float = 5.0,
        cache_size: int = 1024,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")  # noqa: TRY003
        if coalesce_ms < 0:
            raise ValueError("coalesce_ms must not be negative")  # noqa: TRY003
        if cache_size < 0:
            raise ValueError("cache_size must not be negative")  # noqa: TRY003
        self._model = model
        self.profile: EmbeddingProfile = model.profile
        self._max_batch_size = max_batch_size
        self._coalesce_seconds = coalesce_ms / 1000.0
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EmbeddingVector] = OrderedDict()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._inflight: dict[str, _Flight] = {}
        self._pending: deque[_Flight] = deque()
        self._timer: asyncio.TimerHandle | None = None
        self._providers: set[asyncio.Task[None]] = set()

    async def embed(self, texts: tuple[str, ...], /) -> EmbeddingResult:
        """Return one vector per input text, sharing and caching provider work."""

        self._bind_loop()
        requested = tuple(texts)
        if not requested:
            return EmbeddingResult(vectors=(), usage=InferenceUsage(requests=0))

        vectors: dict[str, EmbeddingVector] = {}
        flights: list[_Flight] = []
        queued = 0
        for text in dict.fromkeys(requested):
            cached = self._cached(text)
            if cached is not None:
                vectors[text] = cached
                continue
            flight = self._inflight.get(text)
            if flight is None:
                flight = _Flight(text)
                self._inflight[text] = flight
                self._pending.append(flight)
                queued += 1
            flights.append(flight)
        if queued:
            self._arm()

        batches: list[_Batch] = []
        for flight in flights:
            await flight.event.wait()
            vectors[flight.text] = _settle(flight)
            batch = flight.batch
            if batch is not None and not any(batch is seen for seen in batches):
                batches.append(batch)
        return EmbeddingResult(
            vectors=tuple(vectors[text] for text in requested),
            usage=_sum_usage(batches),
        )

    def _bind_loop(self) -> None:
        """Drop coordination state that belongs to another event loop, keeping the cache."""

        loop = asyncio.get_running_loop()
        if self._loop is loop:
            return
        if self._timer is not None:
            self._timer.cancel()
        self._loop = loop
        self._inflight.clear()
        self._pending.clear()
        self._providers.clear()
        self._timer = None

    def _cached(self, text: str) -> EmbeddingVector | None:
        if self._cache_size <= 0:
            return None
        vector = self._cache.get(text)
        if vector is None:
            return None
        self._cache.move_to_end(text)
        return vector

    def _remember(self, text: str, vector: EmbeddingVector) -> None:
        if self._cache_size <= 0:
            return
        self._cache[text] = vector
        self._cache.move_to_end(text)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _arm(self) -> None:
        """Schedule the single drain that closes the current coalescing window."""

        if self._timer is not None:
            return
        loop = asyncio.get_running_loop()
        self._timer = loop.call_later(self._coalesce_seconds, self._drain)

    def _drain(self) -> None:
        """Move every queued flight into bounded provider batches.

        Runs as a loop callback rather than as a task, so it cannot be cancelled halfway and
        cannot yield between taking a flight off the queue and handing it to a batch.
        """

        self._timer = None
        while self._pending:
            flights = [self._pending.popleft() for _ in range(min(self._max_batch_size, len(self._pending)))]
            self._dispatch(_Batch(flights))

    def _dispatch(self, batch: _Batch) -> None:
        for flight in batch.flights:
            flight.batch = batch
        provider = asyncio.create_task(self._run_batch(batch), name="powercontext-embedding-batch")
        self._providers.add(provider)
        provider.add_done_callback(self._providers.discard)

    async def _run_batch(self, batch: _Batch) -> None:
        """Send one bounded batch to the provider and settle exactly its own flights.

        A provider failure belongs to the whole batch: every waiter observes the very same
        error instead of a rewritten one. Every other batch is a separate task, so an
        independent batch still completes while this one fails.
        """

        try:
            result = await self._model.embed(batch.texts)
            vectors = _ordered_vectors(batch.texts, result.vectors)
        except asyncio.CancelledError as cancelled:
            self._fail_batch(batch, cancelled)
            raise
        except Exception as error:
            self._fail_batch(batch, error)
            return
        self._complete_batch(batch, vectors, result.usage)

    def _complete_batch(self, batch: _Batch, vectors: tuple[EmbeddingVector, ...], usage: InferenceUsage) -> None:
        batch.usage = usage
        for flight, vector in zip(batch.flights, vectors, strict=True):
            self._remember(flight.text, vector)
            flight.vector = vector
            self._inflight.pop(flight.text, None)
        for flight in batch.flights:
            flight.event.set()

    def _fail_batch(self, batch: _Batch, error: Exception) -> None:
        for flight in batch.flights:
            flight.error = error
            self._inflight.pop(flight.text, None)
        for flight in batch.flights:
            flight.event.set()


def _settle(flight: _Flight) -> EmbeddingVector:
    """Read one settled flight, re-raising the failure every waiter of its batch shares."""

    if flight.error is not None:
        raise flight.error
    vector = flight.vector
    if vector is None:
        raise InvalidInferenceOutputError("embed", "coalesced batch produced no vector")
    return vector


def _ordered_vectors(texts: tuple[str, ...], vectors: Sequence[EmbeddingVector]) -> tuple[EmbeddingVector, ...]:
    """Reject a batch whose provider output does not match the texts that were sent."""

    if len(vectors) != len(texts):
        raise InvalidInferenceOutputError("embed", "provider returned the wrong vector count")
    return tuple(vectors)


def _sum_usage(batches: Sequence[_Batch]) -> InferenceUsage:
    """Sum one caller's distinct batches, keeping provider-reported counts additive."""

    return InferenceUsage(
        requests=sum(batch.usage.requests for batch in batches),
        input_tokens=_known_sum([batch.usage.input_tokens for batch in batches]),
        output_tokens=_known_sum([batch.usage.output_tokens for batch in batches]),
    )


def _known_sum(values: Sequence[int | None]) -> int | None:
    """Total a token field only when every batch reported it, mirroring usage aggregation."""

    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)


__all__ = ["CoalescingEmbeddingModel"]
