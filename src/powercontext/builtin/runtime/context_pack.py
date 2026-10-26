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

"""Lossless, prefix-reusable wire packs for one prepared context.

A :class:`ContextPack` is prepared text cut into ordered, independently decodable chunks
together with a digest over those chunks and the size of its own wire form. A
:class:`ContextPackDelta` names how many leading chunks of an already delivered pack stay
valid and carries only the remaining suffix. A receiver holding the previous pack rebuilds
this round's text byte for byte from the delta alone, so the protocol keeps no sender-side
state: nothing is cached in the preparing process and no Python object is shared with the
receiver.

Chunk boundaries are chosen from the text prefix only, so cutting the same text always
yields the same chunks and appending a suffix never moves the boundaries before it -- the
property prefix reuse depends on.

The wire form is the whole interop contract: canonical UTF-8 JSON (sorted keys, compact
separators, no extra whitespace), Base64 for every chunk, and SHA-256 over the
schema-plus-chunks JSON.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final

from powercontext.errors import PowerContextError

PACK_SCHEMA: Final[str] = "powercontext.context-pack.v1"
DELTA_SCHEMA: Final[str] = "powercontext.context-pack-delta.v1"
DEFAULT_CHUNK_BYTES: Final[int] = 512

_DIGEST_PREFIX: Final[str] = "sha256:"
_HEX_DIGITS: Final[frozenset[str]] = frozenset("0123456789abcdef")
_PACK_FIELDS: Final[frozenset[str]] = frozenset({"schema", "items", "digest", "byte_count"})
_DELTA_FIELDS: Final[frozenset[str]] = frozenset(
    {"schema", "base_digest", "target_digest", "prefix_items", "appended"}
)


class ContextPackError(PowerContextError, ValueError):
    """A context pack, delta, or chunking request violates the transfer protocol."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"Context Pack protocol failed: {code}")


def _canonical(payload: dict[str, Any]) -> bytes:
    """Encode one wire payload exactly as the receiver re-encodes it for hashing."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _encode_items(items: Sequence[bytes]) -> list[str]:
    return [base64.b64encode(item).decode("ascii") for item in items]


def _decode_item(value: Any) -> bytes:
    if not isinstance(value, str):
        raise ContextPackError("chunk-not-base64")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError) as error:
        raise ContextPackError("chunk-not-base64") from error


def _as_items(items: Iterable[Any], code: str) -> tuple[bytes, ...]:
    normalized: list[bytes] = []
    for item in items:
        if not isinstance(item, (bytes, bytearray, memoryview)):
            raise ContextPackError(code)
        normalized.append(bytes(item))
    return tuple(normalized)


def _digest_of(items: Sequence[bytes]) -> str:
    unsigned = {"schema": PACK_SCHEMA, "items": _encode_items(items)}
    return _DIGEST_PREFIX + hashlib.sha256(_canonical(unsigned)).hexdigest()


def _pack_wire(items: Sequence[bytes], digest: str, byte_count: int) -> bytes:
    return _canonical(
        {
            "schema": PACK_SCHEMA,
            "items": _encode_items(items),
            "digest": digest,
            "byte_count": byte_count,
        }
    )


def _pack_size(items: Sequence[bytes], digest: str) -> int:
    """Resolve ``byte_count``, which counts the field that carries it.

    The size grows with the digit count of the reported value, so iterating up from zero
    reaches the only self-consistent answer in a couple of steps.
    """

    size = 0
    for _ in range(8):
        measured = len(_pack_wire(items, digest, size))
        if measured == size:
            return size
        size = measured
    raise ContextPackError("byte-count-not-converged")


def _is_digest(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith(_DIGEST_PREFIX):
        return False
    hexadecimal = value[len(_DIGEST_PREFIX) :]
    return len(hexadecimal) == 64 and set(hexadecimal) <= _HEX_DIGITS


def _require_digest(value: Any) -> str:
    if not _is_digest(value):
        raise ContextPackError("digest-malformed")
    return value


def _require_prefix(value: Any) -> int:
    """Reject ``bool``, ``float``, and negative counts for ``prefix_items``."""

    if type(value) is not int or value < 0:
        raise ContextPackError("prefix-items-not-a-count")
    return value


def _wire_object(payload: Any, code: str) -> dict[str, Any]:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ContextPackError(code)
    try:
        decoded = json.loads(bytes(payload).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContextPackError(code) from error
    if not isinstance(decoded, dict):
        raise ContextPackError(code)
    return decoded


def _content_text(content: Any) -> str:
    """Accept text or UTF-8 bytes, and fail the protocol on anything else."""

    if isinstance(content, str):
        try:
            content.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ContextPackError("content-not-utf8") from error
        return content
    if isinstance(content, (bytes, bytearray, memoryview)):
        try:
            return bytes(content).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ContextPackError("content-not-utf8") from error
    raise ContextPackError("content-not-text")


def _chunk_text(text: str, chunk_bytes: Any) -> tuple[bytes, ...]:
    """Cut text into deterministic chunks that are valid UTF-8 and within the limit.

    A line is the preferred boundary because a rendered context changes one entry at a
    time, so keeping each line whole concentrates a change into the few chunks it touches
    instead of shifting every later boundary. A line that cannot fit is split at character
    boundaries; a limit that cannot hold one character is a protocol failure rather than a
    silently oversized chunk.
    """

    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ContextPackError("chunk-bytes-not-a-positive-count")
    chunks: list[bytes] = []
    pending: list[str] = []
    pending_bytes = 0

    def flush() -> None:
        nonlocal pending, pending_bytes
        if pending:
            chunks.append("".join(pending).encode("utf-8"))
            pending = []
            pending_bytes = 0

    def append(piece: str, piece_bytes: int) -> None:
        nonlocal pending_bytes
        if piece_bytes > chunk_bytes:
            raise ContextPackError("chunk-bytes-cannot-hold-one-character")
        if pending_bytes + piece_bytes > chunk_bytes:
            flush()
        pending.append(piece)
        pending_bytes += piece_bytes

    for line in text.splitlines(keepends=True):
        line_bytes = len(line.encode("utf-8"))
        if line_bytes <= chunk_bytes:
            append(line, line_bytes)
            continue
        flush()
        for character in line:
            append(character, len(character.encode("utf-8")))
    flush()
    return tuple(chunks)


@dataclass(frozen=True, slots=True)
class ContextPack:
    """One complete prepared context: ordered chunks, their digest, and its own size."""

    items: tuple[bytes, ...]
    digest: str
    byte_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple):
            raise ContextPackError("items-not-bytes")
        _as_items(self.items, "items-not-bytes")
        if self.digest != _digest_of(self.items):
            raise ContextPackError("pack-digest-mismatch")
        if self.byte_count != _pack_size(self.items, self.digest):
            raise ContextPackError("pack-byte-count-mismatch")

    @classmethod
    def build(cls, items: Iterable[Any], /) -> ContextPack:
        """Build the complete pack for already-cut chunks."""

        chunks = _as_items(items, "items-not-bytes")
        digest = _digest_of(chunks)
        return cls(items=chunks, digest=digest, byte_count=_pack_size(chunks, digest))

    @classmethod
    def from_content(cls, content: Any, /, *, chunk_bytes: int = DEFAULT_CHUNK_BYTES) -> ContextPack:
        """Cut prepared text into a deterministic complete pack."""

        return cls.build(_chunk_text(_content_text(content), chunk_bytes))

    @classmethod
    def from_bytes(cls, payload: Any, /) -> ContextPack:
        """Decode one received complete pack, validating its digest and byte count."""

        wire = _wire_object(payload, "pack-not-json")
        if set(wire) != _PACK_FIELDS:
            raise ContextPackError("pack-fields")
        if wire["schema"] != PACK_SCHEMA:
            raise ContextPackError("pack-schema")
        digest = _require_digest(wire["digest"])
        byte_count = wire["byte_count"]
        if type(byte_count) is not int or byte_count < 0:
            raise ContextPackError("pack-byte-count")
        encoded = wire["items"]
        if not isinstance(encoded, list):
            raise ContextPackError("pack-items")
        items = tuple(_decode_item(value) for value in encoded)
        if _digest_of(items) != digest:
            raise ContextPackError("pack-digest-mismatch")
        if _pack_size(items, digest) != byte_count:
            raise ContextPackError("pack-byte-count-mismatch")
        return cls(items=items, digest=digest, byte_count=byte_count)

    def to_bytes(self) -> bytes:
        """Serialize this pack in its canonical wire form."""

        return _pack_wire(self.items, self.digest, self.byte_count)


@dataclass(frozen=True, slots=True)
class ContextPackDelta:
    """The chunks a receiver still needs after reusing the base pack's prefix."""

    base_digest: str
    target_digest: str
    prefix_items: int
    appended: tuple[bytes, ...]

    @classmethod
    def between(cls, base: ContextPack, target: ContextPack, /) -> ContextPackDelta:
        """Carry a common chunk prefix plus the target suffix.

        Any valid prefix decodes correctly, so the choice is a pure transfer-cost one: the
        longest shared prefix leaves only the chunks that actually changed on the wire.
        """

        if not isinstance(base, ContextPack) or not isinstance(target, ContextPack):
            raise ContextPackError("pack-not-a-context-pack")
        prefix = 0
        shared = min(len(base.items), len(target.items))
        while prefix < shared and base.items[prefix] == target.items[prefix]:
            prefix += 1
        return cls(
            base_digest=base.digest,
            target_digest=target.digest,
            prefix_items=prefix,
            appended=target.items[prefix:],
        )

    @property
    def byte_count(self) -> int:
        """Wire size of this delta; measured, never written as a field of its own."""

        return len(self.to_bytes())

    def to_bytes(self) -> bytes:
        """Serialize this delta in its canonical wire form."""

        return _canonical(
            {
                "schema": DELTA_SCHEMA,
                "base_digest": _require_digest(self.base_digest),
                "target_digest": _require_digest(self.target_digest),
                "prefix_items": _require_prefix(self.prefix_items),
                "appended": _encode_items(_as_items(self.appended, "appended-not-bytes")),
            }
        )

    @classmethod
    def from_bytes(cls, payload: Any, /) -> ContextPackDelta:
        """Decode one received delta, validating every protocol field."""

        wire = _wire_object(payload, "delta-not-json")
        if set(wire) != _DELTA_FIELDS:
            raise ContextPackError("delta-fields")
        if wire["schema"] != DELTA_SCHEMA:
            raise ContextPackError("delta-schema")
        appended = wire["appended"]
        if not isinstance(appended, list):
            raise ContextPackError("delta-appended")
        return cls(
            base_digest=_require_digest(wire["base_digest"]),
            target_digest=_require_digest(wire["target_digest"]),
            prefix_items=_require_prefix(wire["prefix_items"]),
            appended=tuple(_decode_item(value) for value in appended),
        )


def apply_context_pack_delta(base: ContextPack, delta: ContextPackDelta, /) -> ContextPack:
    """Rebuild the target pack from a base pack and one received delta.

    The receiver needs nothing but these two values: the base digest pins the pack the
    sender encoded against, the prefix must stay inside that pack, and the rebuilt digest
    proves the result is exactly the pack the sender targeted. A stale base, an
    out-of-range prefix, or a target digest that does not match is a protocol failure
    instead of a silently different context.
    """

    if not isinstance(base, ContextPack) or not isinstance(delta, ContextPackDelta):
        raise ContextPackError("pack-not-a-context-pack")
    if delta.base_digest != base.digest:
        raise ContextPackError("delta-base-mismatch")
    prefix = _require_prefix(delta.prefix_items)
    if prefix > len(base.items):
        raise ContextPackError("delta-prefix-out-of-range")
    appended = _as_items(delta.appended, "appended-not-bytes")
    target = ContextPack.build((*base.items[:prefix], *appended))
    if target.digest != delta.target_digest:
        raise ContextPackError("delta-target-mismatch")
    return target


__all__ = [
    "DEFAULT_CHUNK_BYTES",
    "DELTA_SCHEMA",
    "PACK_SCHEMA",
    "ContextPack",
    "ContextPackDelta",
    "ContextPackError",
    "apply_context_pack_delta",
]
