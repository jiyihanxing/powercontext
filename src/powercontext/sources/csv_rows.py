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

"""Row-level CSV Sources for offline table evidence.

One CSV document is captured as one ``CAPTURED`` Source per data record. Each
row keeps its original header order and raw field strings, so a retrieval
citation can be resolved back to the exact record it came from even after the
document is reordered or the process restarts.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterator
from urllib.parse import quote

from pydantic import BaseModel, JsonValue, model_validator

from powercontext.errors import InvalidSourceReferenceError
from powercontext.limits import MAX_SOURCE_ID_LENGTH
from powercontext.sources.definitions import AdapterSourceDefinition, SourceDefinition
from powercontext.sources.models import Source, SourceMaterialization
from powercontext.sources.projections import TEXT_EVIDENCE_PROJECTION_KEY, TextEvidence

CSV_ROW_SOURCE_NAME = "csv-row"
CSV_ROW_SOURCE_VERSION = "1"
CSV_ROW_SOURCE_ID_PREFIX = "csv"


class CSVDocumentInput(BaseModel):
    """One complete offline CSV document captured as bytes."""

    document_id: str
    primary_key: str
    content: bytes


class CSVRowInput(BaseModel):
    """One data record identified by its document and primary key value."""

    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]

    @model_validator(mode="after")
    def require_aligned_record(self) -> CSVRowInput:
        if len(self.columns) != len(self.values):
            raise ValueError("row columns and values must have the same width")
        return self


class CSVRowSource(Source):
    """Captured single-record table evidence that can be cited and re-read."""

    source_id: str
    document_id: str
    primary_key: str
    primary_key_value: str
    columns: tuple[str, ...]
    values: tuple[str, ...]


def csv_row_source_id(document_id: str, primary_key_value: str) -> str:
    """Build the delimiter-safe, bounded identity of one CSV row.

    Components are percent-encoded independently, so neither an encoded
    ``document_id`` nor a ``primary_key_value`` can introduce the ``:`` that
    separates the three identity segments.
    """

    document = _require_trimmed(document_id, "document_id")
    if not isinstance(primary_key_value, str):
        raise InvalidSourceReferenceError("primary_key_value", "must be a string")
    identity = f"{CSV_ROW_SOURCE_ID_PREFIX}:{quote(document, safe='')}:{quote(primary_key_value, safe='')}"
    if len(identity) > MAX_SOURCE_ID_LENGTH:
        raise InvalidSourceReferenceError(
            "source_id",
            f"must not exceed {MAX_SOURCE_ID_LENGTH} characters",
        )
    return identity


def parse_csv_document(value: CSVDocumentInput) -> tuple[CSVRowInput, ...]:
    """Parse and fully validate one document before returning any row."""

    document_id = _require_trimmed(value.document_id, "document_id")
    # The primary key is compared exactly: a header column may legitimately
    # carry surrounding spaces and only an exact match may select it.
    primary_key = _require_text(value.primary_key, "primary_key")

    records = _read_records(value.content)
    try:
        header = next(records)
    except StopIteration:
        raise ValueError("CSV document must not be empty") from None
    columns = _validate_header(header)
    try:
        key_index = columns.index(primary_key)
    except ValueError:
        raise ValueError(f"primary key {primary_key!r} is not a header column") from None

    rows: list[CSVRowInput] = []
    seen: set[str] = set()
    for record in records:
        if not record:
            # A fully empty physical record carries no column values.
            continue
        if len(record) != len(columns):
            raise ValueError(f"CSV record has {len(record)} fields, expected {len(columns)}")
        primary_key_value = record[key_index]
        if not primary_key_value.strip():
            raise ValueError("CSV record primary key value must not be blank")
        if primary_key_value in seen:
            raise ValueError(f"duplicate primary key value {primary_key_value!r}")
        seen.add(primary_key_value)
        # Refuse the whole document if this record cannot carry a legal identity.
        csv_row_source_id(document_id, primary_key_value)
        rows.append(
            CSVRowInput(
                document_id=document_id,
                primary_key=primary_key,
                primary_key_value=primary_key_value,
                columns=columns,
                values=tuple(record),
            )
        )
    return tuple(rows)


class CSVRowSourceAdapter:
    """Resolve and read the built-in row-level CSV Source."""

    input_class = CSVRowInput
    name = CSV_ROW_SOURCE_NAME
    source_class = CSVRowSource

    async def resolve(self, value: CSVRowInput, /) -> CSVRowSource:
        source_id = csv_row_source_id(value.document_id, value.primary_key_value)
        return CSVRowSource(
            name=source_id,
            source_id=source_id,
            materialization=SourceMaterialization.CAPTURED,
            document_id=value.document_id,
            primary_key=value.primary_key,
            primary_key_value=value.primary_key_value,
            columns=value.columns,
            values=value.values,
        )

    async def read(self, source: CSVRowSource, /) -> CSVRowInput:
        return CSVRowInput(
            document_id=source.document_id,
            primary_key=source.primary_key,
            primary_key_value=source.primary_key_value,
            columns=source.columns,
            values=source.values,
        )


class CSVRowTextEvidenceProjection:
    """Expose one CSV row as canonical textual evidence."""

    name = TEXT_EVIDENCE_PROJECTION_KEY.name
    version = TEXT_EVIDENCE_PROJECTION_KEY.version
    source_class = CSVRowSource
    output_class: type[BaseModel] = TextEvidence

    def project(self, source: CSVRowSource, /) -> TextEvidence:
        row: dict[str, JsonValue] = dict(zip(source.columns, source.values, strict=True))
        return TextEvidence(
            source_type=CSV_ROW_SOURCE_NAME,
            source_id=source.source_id,
            content=json.dumps(row, ensure_ascii=False, separators=(",", ":")),
            metadata={
                "document_id": source.document_id,
                "primary_key": source.primary_key,
                "primary_key_value": source.primary_key_value,
                "columns": list(source.columns),
                "row": row,
            },
        )


CSV_ROW_SOURCE_ADAPTER = CSVRowSourceAdapter()
CSV_ROW_SOURCE_DEFINITION: SourceDefinition[CSVRowInput, CSVRowSource, CSVRowInput] = AdapterSourceDefinition(
    CSV_ROW_SOURCE_ADAPTER,
    version=CSV_ROW_SOURCE_VERSION,
    projections=(CSVRowTextEvidenceProjection(),),
)


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidSourceReferenceError(field, "must be a non-empty string")
    return value


def _require_trimmed(value: object, field: str) -> str:
    text = _require_text(value, field)
    if not text.strip():
        raise InvalidSourceReferenceError(field, "must not be blank")
    if text != text.strip():
        raise InvalidSourceReferenceError(field, "must not contain leading or trailing whitespace")
    return text


def _read_records(content: bytes) -> Iterator[list[str]]:
    """Yield records with the fixed :mod:`csv` semantics of this contract."""

    if not isinstance(content, (bytes, bytearray)):
        raise ValueError("CSV document content must be bytes")
    try:
        # ``utf-8-sig`` accepts one leading BOM and rejects every other byte.
        text = bytes(content).decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("CSV document must be valid UTF-8") from error
    # ``newline=""`` keeps a ``\r\n`` inside a quoted field byte-for-byte.
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        yield from reader
    except csv.Error as error:
        raise ValueError(f"malformed CSV document: {error}") from error


def _validate_header(header: list[str]) -> tuple[str, ...]:
    if not header:
        raise ValueError("CSV document must declare at least one header column")
    columns = tuple(header)
    if any(not column.strip() for column in columns):
        raise ValueError("CSV header columns must not be blank")
    if len(set(columns)) != len(columns):
        raise ValueError("CSV header columns must be unique")
    return columns


__all__ = [
    "CSV_ROW_SOURCE_ADAPTER",
    "CSV_ROW_SOURCE_DEFINITION",
    "CSV_ROW_SOURCE_NAME",
    "CSV_ROW_SOURCE_VERSION",
    "CSVDocumentInput",
    "CSVRowInput",
    "CSVRowSource",
    "CSVRowSourceAdapter",
    "CSVRowTextEvidenceProjection",
    "csv_row_source_id",
    "parse_csv_document",
]
