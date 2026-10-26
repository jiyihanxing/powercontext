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

"""Pi session branch history as immutable per-event Sources.

A Pi v3 session file is a JSONL append log whose records form a tree through
``parentId``. The selected leaf plus its ancestors is one branch: the history a
user would resume from. This module captures every *visible* record on that
branch as its own ``CAPTURED`` Source, so a citation resolves back to the exact
record -- including a tool result far larger than any single Memory entry.

Three views deliberately stay separate:

* the branch history is the raw evidence and is never rewritten by compaction;
* compaction and branch-summary records are evidence too, kept with their own
  text instead of replacing the records they summarize;
* control records (model or thinking level changes, and every other type with no
  visible body) keep their place in the tree -- children still point at them --
  but they produce no text evidence.

``PiSessionConnector`` re-parses the whole selected branch on every run and only
submits records that are not imported yet. Its durable checkpoint carries one
digest per imported event identity, which is what lets a restart deduplicate and
what makes a rewritten history fail instead of silently replacing evidence.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, JsonValue

from powercontext.errors import InvalidSourceReferenceError
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.connectors import (
    ConnectorRunCompletion,
    ConnectorRunSession,
    ConnectorRunStatus,
    ConnectorSubmissionStatus,
)
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

PI_EVENT_SOURCE_NAME = "pi-event"
PI_EVENT_SOURCE_VERSION = "1"
PI_SESSION_CONNECTOR_NAME = "pi-session-history"
PI_SESSION_CONNECTOR_VERSION = "1"
PI_SESSION_CHECKPOINT_VERSION = 1
PI_SESSION_VERSION = 3
PI_EVENT_SOURCE_ID_PREFIX = "pi"
UNKNOWN_TOOL_LABEL = "unknown"

_SUMMARY_EVENT_TYPES = frozenset({"compaction", "branch_summary"})


class PiSessionInput(BaseModel):
    """One complete Pi v3 session archive plus the branch leaf to capture."""

    content: bytes
    leaf_id: str | None = None


class PiEventInput(BaseModel):
    """One visible record on the selected branch, with its original payload."""

    session_id: str
    event_id: str
    parent_id: str | None = None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


class PiEventSource(Source):
    """Captured single-event session evidence that can be cited and re-read."""

    source_id: str
    session_id: str
    event_id: str
    parent_id: str | None = None
    event_type: str
    role: str
    content: str
    record: dict[str, JsonValue]


def pi_event_source_id(session_id: str, event_id: str) -> str:
    """Build the delimiter-safe, bounded identity of one session event.

    Identity is exactly the pair (session, event): two events of one session
    never share a record id, and two sessions keep disjoint identity spaces.
    Components are percent-encoded independently, so neither an encoded session
    id nor an event id can introduce the ``:`` that separates the segments.
    """

    session = _require_identity_text(session_id, "session_id")
    event = _require_identity_text(event_id, "event_id")
    identity = f"{PI_EVENT_SOURCE_ID_PREFIX}:{quote(session, safe='')}:{quote(event, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise InvalidSourceReferenceError(
            "source_id",
            f"must not exceed {MAX_SOURCE_ID_LENGTH} characters",
        )
    return identity


def parse_pi_session(value: PiSessionInput, /) -> tuple[PiEventInput, ...]:
    """Parse one archive and return the visible events from root to leaf.

    The whole archive is validated before any event is returned, so an illegal
    encoding, malformed JSON record, duplicate id, dangling parent, cycle or
    missing leaf raises :class:`ValueError` instead of yielding a partial branch.
    """

    records = _read_records(_decode_content(value.content))
    session_id, nodes = _validate_archive(records)
    leaf_id = _select_leaf(nodes, value.leaf_id)
    events: list[PiEventInput] = []
    for event_id in _ancestor_chain(nodes, leaf_id):
        record = nodes[event_id]
        rendered = _render_visible_event(record)
        if rendered is None:
            # A control record keeps its place in the tree but carries no evidence.
            continue
        role, body = rendered
        events.append(
            PiEventInput(
                session_id=session_id,
                event_id=event_id,
                parent_id=_parent_of(record),
                event_type=_record_type(record),
                role=role,
                content=f"{role}: {body}",
                record=record,
            )
        )
    return tuple(events)


class PiEventSourceAdapter:
    """Resolve and read the built-in per-event Pi session Source."""

    input_class = PiEventInput
    name = PI_EVENT_SOURCE_NAME
    source_class = PiEventSource

    async def resolve(self, value: PiEventInput, /) -> PiEventSource:
        source_id = pi_event_source_id(value.session_id, value.event_id)
        return PiEventSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            session_id=value.session_id,
            event_id=value.event_id,
            parent_id=value.parent_id,
            event_type=value.event_type,
            role=value.role,
            content=value.content,
            record=value.record,
        )

    async def read(self, source: PiEventSource, /) -> PiEventInput:
        return PiEventInput(
            session_id=source.session_id,
            event_id=source.event_id,
            parent_id=source.parent_id,
            event_type=source.event_type,
            role=source.role,
            content=source.content,
            record=source.record,
        )


class PiEventTextEvidenceProjection:
    """Expose one branch event as canonical textual evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = PiEventSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: PiEventSource, /) -> TextEvidence:
        return TextEvidence(
            source_type=PI_EVENT_SOURCE_NAME,
            source_id=source.source_id,
            content=source.content,
            metadata={
                "session_id": source.session_id,
                "event_id": source.event_id,
                "parent_id": source.parent_id,
                "event_type": source.event_type,
                "role": source.role,
            },
        )


PI_EVENT_SOURCE_ADAPTER = PiEventSourceAdapter()
PI_EVENT_SOURCE_DEFINITION: SourceDefinition[PiEventInput, PiEventSource, PiEventInput] = AdapterSourceDefinition(
    PI_EVENT_SOURCE_ADAPTER,
    version=PI_EVENT_SOURCE_VERSION,
    projections=(PiEventTextEvidenceProjection(),),
)


class PiSessionConnector:
    """Import the selected Pi branch as immutable per-event Sources, resumably.

    Every run fully parses the archive. Records already present in the durable
    checkpoint are re-checked against their digest and skipped; only new records
    are submitted. A rewritten history raises instead of replacing evidence, and
    an unreadable archive raises before the lifecycle may advance anything.
    """

    name = PI_SESSION_CONNECTOR_NAME
    version = PI_SESSION_CONNECTOR_VERSION
    source_definitions = frozenset({PI_EVENT_SOURCE_NAME})

    def __init__(self, path: str | Path, leaf_id: str | None = None) -> None:
        self._path = Path(path)
        self._leaf_id = leaf_id

    @property
    def path(self) -> Path:
        return self._path

    @property
    def leaf_id(self) -> str | None:
        return self._leaf_id

    async def run(self, session: ConnectorRunSession, /) -> ConnectorRunCompletion:
        raw = await asyncio.to_thread(self._path.read_bytes)
        events = parse_pi_session(PiSessionInput(content=raw, leaf_id=self._leaf_id))

        imported = _imported_events(session.checkpoint)
        identities = tuple(pi_event_source_id(event.session_id, event.event_id) for event in events)
        digests = tuple(_event_digest(event) for event in events)

        # Refuse a rewritten history before submitting anything, so the refused
        # run leaves both the imported evidence and the checkpoint untouched.
        for identity, digest in zip(identities, digests, strict=True):
            known = imported.get(identity)
            if known is not None and known != digest:
                raise ValueError(f"Pi event {identity!r} changed after it was captured")

        for identity, event in zip(identities, events, strict=True):
            if identity in imported:
                continue
            outcome = await session.submit(identity, PI_EVENT_SOURCE_NAME, event)
            if outcome.status is not ConnectorSubmissionStatus.ACCEPTED:
                # A rejected item must never be recorded as imported.
                return ConnectorRunCompletion(status=ConnectorRunStatus.INCOMPLETE, checkpoint=session.checkpoint)

        checkpoint: dict[str, JsonValue] = {
            "version": PI_SESSION_CHECKPOINT_VERSION,
            "events": {**imported, **dict(zip(identities, digests, strict=True))},
        }
        return ConnectorRunCompletion(status=ConnectorRunStatus.COMPLETE, checkpoint=checkpoint)


def _decode_content(content: object) -> str:
    if not isinstance(content, (bytes, bytearray)):
        raise ValueError("Pi session content must be bytes")
    try:
        # ``utf-8-sig`` accepts one leading BOM and rejects every other byte.
        return bytes(content).decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Pi session archive must be valid UTF-8") from error


def _read_records(text: str) -> list[Any]:
    """Decode every physical line of the archive, rejecting any malformed one."""

    records: list[Any] = []
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"Pi session record {number} is not valid JSON: {error.msg}") from error
    return records


def _validate_archive(records: list[Any]) -> tuple[str, dict[str, dict[str, JsonValue]]]:
    """Validate the header and every parent link of one complete archive."""

    if not records:
        raise ValueError("Pi session archive must not be empty")
    header = records[0]
    if not isinstance(header, dict):
        raise ValueError("Pi session header must be a JSON object")
    if header.get("type") != "session":
        raise ValueError("Pi session archive must start with a session header")
    version = header.get("version")
    if isinstance(version, bool) or version != PI_SESSION_VERSION:
        raise ValueError(f"Pi session version must be {PI_SESSION_VERSION}")
    session_id = header.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise ValueError("Pi session header must declare a non-empty id")

    nodes: dict[str, dict[str, JsonValue]] = {}
    for record in records[1:]:
        if not isinstance(record, dict):
            raise ValueError("Pi session records must be JSON objects")
        event_id = record.get("id")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("Pi session record must declare a non-empty id")
        if event_id in nodes:
            raise ValueError(f"duplicate Pi session record id {event_id!r}")
        parent_id = record.get("parentId")
        if parent_id is not None:
            if not isinstance(parent_id, str) or not parent_id:
                raise ValueError(f"Pi session record {event_id!r} has an invalid parentId")
            # Requiring an already-materialized parent rejects a dangling parent
            # and every cycle, including a self reference.
            if parent_id not in nodes:
                raise ValueError(f"Pi session record {event_id!r} references unknown parent {parent_id!r}")
        nodes[event_id] = record
    return session_id, nodes


def _select_leaf(nodes: dict[str, dict[str, JsonValue]], leaf_id: object) -> str:
    if leaf_id is None:
        if not nodes:
            raise ValueError("Pi session archive has no record to select as leaf")
        return next(reversed(nodes))
    if not isinstance(leaf_id, str) or leaf_id not in nodes:
        raise ValueError(f"Pi session leaf {leaf_id!r} is not a record of this archive")
    return leaf_id


def _ancestor_chain(nodes: dict[str, dict[str, JsonValue]], leaf_id: str) -> tuple[str, ...]:
    chain: list[str] = []
    current: str | None = leaf_id
    while current is not None:
        chain.append(current)
        current = _parent_of(nodes[current])
    chain.reverse()
    return tuple(chain)


def _parent_of(record: dict[str, JsonValue]) -> str | None:
    parent_id = record.get("parentId")
    return parent_id if isinstance(parent_id, str) and parent_id else None


def _record_type(record: dict[str, JsonValue]) -> str:
    event_type = record.get("type")
    return event_type if isinstance(event_type, str) and event_type else "unknown"


def _render_visible_event(record: dict[str, JsonValue]) -> tuple[str, str] | None:
    """Render one record as ``(role, body)``, or ``None`` when it carries no text."""

    event_type = _record_type(record)
    if event_type == "message":
        message = record.get("message")
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        if not isinstance(role, str) or not role:
            return None
        if role == "toolResult":
            body = _tool_result_body(message)
        elif role == "bashExecution":
            body = _bash_body(message)
        else:
            body = _content_body(message.get("content"))
        return (role, body) if body else None
    if event_type in _SUMMARY_EVENT_TYPES:
        body = _content_body(record.get("summary"))
        return (event_type, body) if body else None
    if event_type == "custom_message":
        body = _content_body(record.get("content"))
        return ("custom", body) if body else None
    return None


def _content_body(value: object) -> str:
    """Render a string or a block list; thinking and image blocks never count."""

    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [part for part in (_block_body(block) for block in value) if part]
        return "\n".join(parts)
    return ""


def _block_body(block: object) -> str:
    if not isinstance(block, dict):
        return ""
    block_type = block.get("type")
    if block_type == "text":
        text = block.get("text")
        return text if isinstance(text, str) else ""
    if block_type == "toolCall":
        arguments = block.get("arguments")
        payload = _compact_json({} if arguments is None else arguments)
        return f"[tool_call {_label(block.get('name'))} {_label(block.get('id'))}] {payload}"
    # ``thinking``, ``image`` and unknown blocks are not visible evidence.
    return ""


def _tool_result_body(message: dict[str, JsonValue]) -> str:
    prefix = f"[tool_result {_label(message.get('toolName'))} {_label(message.get('toolCallId'))}]"
    text = _content_body(message.get("content"))
    return f"{prefix} {text}" if text else prefix


def _bash_body(message: dict[str, JsonValue]) -> str:
    command = message.get("command")
    output = message.get("output")
    text = f"$ {command}" if isinstance(command, str) and command else ""
    if isinstance(output, str) and output:
        return f"{text}\n{output}" if text else output
    return text


def _label(value: object) -> str:
    return value if isinstance(value, str) and value else UNKNOWN_TOOL_LABEL


def _compact_json(value: object) -> str:
    """Serialize tool arguments with sorted keys, no spaces and raw Unicode."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _event_digest(event: PiEventInput) -> str:
    """Digest everything one event Source stores, including its raw record."""

    canonical = json.dumps(event.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _imported_events(checkpoint: JsonValue | None) -> dict[str, str]:
    """Read the durable identity-to-digest map of already imported events."""

    if checkpoint is None:
        return {}
    if not isinstance(checkpoint, dict):
        raise ValueError("pi-session-history checkpoint must be a JSON object")
    if checkpoint.get("version") != PI_SESSION_CHECKPOINT_VERSION:
        raise ValueError("pi-session-history checkpoint has an unsupported version")
    events = checkpoint.get("events")
    if not isinstance(events, dict):
        raise ValueError("pi-session-history checkpoint must record imported events")
    imported: dict[str, str] = {}
    for identity, digest in events.items():
        if not isinstance(identity, str) or not isinstance(digest, str):
            raise ValueError("pi-session-history checkpoint entries must be identity and digest strings")
        imported[identity] = digest
    return imported


def _require_identity_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidSourceReferenceError(field, "must be a non-empty string")
    return value


__all__ = [
    "PI_EVENT_SOURCE_ADAPTER",
    "PI_EVENT_SOURCE_DEFINITION",
    "PI_EVENT_SOURCE_NAME",
    "PI_EVENT_SOURCE_VERSION",
    "PI_SESSION_CHECKPOINT_VERSION",
    "PI_SESSION_CONNECTOR_NAME",
    "PI_SESSION_CONNECTOR_VERSION",
    "PI_SESSION_VERSION",
    "PiEventInput",
    "PiEventSource",
    "PiEventSourceAdapter",
    "PiEventTextEvidenceProjection",
    "PiSessionConnector",
    "PiSessionInput",
    "parse_pi_session",
    "pi_event_source_id",
]
