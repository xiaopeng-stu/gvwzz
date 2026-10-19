"""Bounded-memory, atomic JSON output for complete mathematical records.

Large arrays are emitted one record at a time. Each ordinary record uses the
C JSON encoder without per-exponent indentation. ``JsonSequence`` defers record
creation, so a full saved basis does not need a second copy in Python objects.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field as dataclass_field
import gc
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Iterable


@dataclass(frozen=True)
class JsonSequence:
    """A reusable JSON array backed by a factory of exact record iterables."""

    factory: Callable[[], Iterable]
    source_identity: object | None = dataclass_field(default=None, compare=False, repr=False)

    def __iter__(self):
        return iter(self.factory())


@dataclass(frozen=True)
class JsonWriteState:
    """Receipt for reusing exact bytes emitted by this writer, not an input file."""

    path: Path
    closing_offset: int
    keys: tuple[str, ...]
    size: int
    mtime_ns: int
    basis_polynomial_spans: tuple[tuple[int, int] | None, ...] = ()
    basis_source_identity: object | None = dataclass_field(default=None, compare=False, repr=False)

    def has_saved_polynomial(self, index):
        return (type(index) is int and 0 <= index < len(self.basis_polynomial_spans)
                and self.basis_polynomial_spans[index] is not None)


@dataclass(frozen=True)
class JsonSavedPolynomial:
    """An unchanged polynomial from a writer receipt, copied without re-encoding."""

    source: JsonWriteState
    index: int


@contextmanager
def output_gc_paused():
    """Only output records are temporary; preserve the caller's GC policy."""
    enabled = gc.isenabled()
    if enabled:
        gc.disable()
    try:
        yield
    finally:
        (gc.enable if enabled else gc.disable)()


def _check_source(state, stat=None):
    if stat is None:
        stat = state.path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (state.size, state.mtime_ns):
        raise ValueError("JSON file changed since it was saved")


class _WriteContext:
    """One open source handle per receipt for a whole atomic output operation."""

    def __init__(self, stack):
        self.stack = stack
        self.sources = {}
        self.basis_polynomial_spans = []
        self.basis_source_identity = None

    @staticmethod
    def _key(state):
        return state.path, state.size, state.mtime_ns

    def bind_source(self, state, source):
        _check_source(state, os.fstat(source.fileno()))
        self.sources[self._key(state)] = state, source

    def source(self, state):
        key = self._key(state)
        if key not in self.sources:
            _check_source(state)
            source = self.stack.enter_context(state.path.open("rb"))
            self.bind_source(state, source)
        return self.sources[key][1]

    def validate_sources(self):
        for state, _ in self.sources.values():
            _check_source(state)

    def copy_polynomial(self, sink, value, *, progress=None, field=""):
        state = value.source
        if not isinstance(state, JsonWriteState) or not state.has_saved_polynomial(value.index):
            raise ValueError("No saved polynomial span is available for this record")
        offset, length = state.basis_polynomial_spans[value.index]
        if (type(offset) is not int or type(length) is not int or offset < 1 or length < 2
                or offset + length > state.closing_offset):
            raise ValueError("Saved polynomial span is outside the recorded JSON source")
        source = self.source(state)
        source.seek(offset)
        remaining = length
        while remaining:
            block = source.read(min(4 * 1024 * 1024, remaining))
            if not block:
                raise ValueError("Saved polynomial ended before its recorded boundary")
            sink.write(block)
            remaining -= len(block)
            if progress is not None and remaining:
                progress(field, 0, sink.tell())


def _compact(value):
    # Nested lazy sequences beyond a streamed record remain correct. The result
    # model places its large sequences at explicit streaming boundaries.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      default=lambda item: list(item) if isinstance(item, JsonSequence)
                      else _unsupported(item)).encode("utf-8")


def _unsupported(value):
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _basis_record(sink, record, context):
    """Write and remember a polynomial's exact bytes without parsing JSON text.

    Only the known label layout is eligible. Its encoder-produced prefix fixes
    the start, and the outer label's closing brace fixes the end. Other values
    are written normally with no reusable span.
    """
    if (isinstance(record, dict) and tuple(record) == ("signature", "polynomial")
            and isinstance(record["polynomial"], dict)
            and tuple(record["polynomial"]) == ("expression", "terms")):
        encoded = _compact(record)
        prefix = b'{"signature":' + _compact(record["signature"]) + b',"polynomial":'
        if encoded.startswith(prefix) and encoded.endswith(b"}"):
            context.basis_polynomial_spans.append(
                (sink.tell() + len(prefix), len(encoded) - len(prefix) - 1))
        else:
            context.basis_polynomial_spans.append(None)
        sink.write(encoded)
        return True
    context.basis_polynomial_spans.append(None)
    return False


def _write_value(sink, value, *, context, progress=None, field="", depth=1):
    if isinstance(value, JsonSavedPolynomial):
        context.copy_polynomial(sink, value, progress=progress, field=field)
    elif isinstance(value, JsonSequence) or (depth == 1 and isinstance(value, (list, tuple))):
        capture_basis = depth == 1 and field == "basis"
        if capture_basis:
            context.basis_source_identity = getattr(value, "source_identity", None)
        sink.write(b"[")
        first = True
        for count, record in enumerate(value, 1):
            sink.write(b"\n" if first else b",\n")
            sink.write(b"  " * (depth + 1))
            captured = capture_basis and _basis_record(sink, record, context)
            if not captured:
                _write_value(sink, record, context=context, progress=progress,
                             field=field, depth=depth + 1)
            first = False
            if progress is not None and count % 128 == 0:
                progress(field, count, sink.tell())
        if not first:
            sink.write(b"\n" + b"  " * depth)
        sink.write(b"]")
    elif isinstance(value, dict) and any(isinstance(item, (JsonSequence, JsonSavedPolynomial))
                                        for item in value.values()):
        sink.write(b"{")
        first = True
        for key, item in value.items():
            sink.write(b"\n" if first else b",\n")
            sink.write(b"  " * (depth + 1) + _compact(key) + b": ")
            _write_value(sink, item, context=context, progress=progress,
                         field=field, depth=depth + 1)
            first = False
        if not first:
            sink.write(b"\n" + b"  " * depth)
        sink.write(b"}")
    else:
        sink.write(_compact(value))


def _write_fields(sink, fields, *, context, first=True, progress=None):
    for key, value in fields.items():
        if not isinstance(key, str):
            raise TypeError("Saved result field names must be strings")
        if progress is not None:
            progress(key, 0, sink.tell())
        sink.write(b"\n" if first else b",\n")
        sink.write(b"  " + _compact(key) + b": ")
        _write_value(sink, value, context=context, progress=progress, field=key)
        first = False
    return first


def _receipt(path, closing_offset, keys, spans=(), source_identity=None):
    stat = path.stat()
    return JsonWriteState(path.resolve(), closing_offset, tuple(keys),
                          stat.st_size, stat.st_mtime_ns, tuple(spans), source_identity)


def write_json(path, data, *, progress=None):
    """Write all fields atomically; the previous file survives every failure.

    A result object returns a receipt accepted by :func:`append_json_fields`.
    Other JSON root values remain supported and return ``None``.
    ``progress(field, records, bytes_written)`` never affects the file data.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with output_gc_paused(), ExitStack() as stack, tempfile.NamedTemporaryFile(
                mode="wb", prefix="." + path.name + ".", suffix=".tmp",
                dir=path.parent, delete=False, buffering=1024 * 1024) as sink:
            temporary = Path(sink.name)
            context = _WriteContext(stack)
            if isinstance(data, dict):
                sink.write(b"{")
                _write_fields(sink, data, context=context, progress=progress)
                closing_offset = sink.tell()
                sink.write(b"\n}\n")
            else:
                _write_value(sink, data, context=context, progress=progress)
                sink.write(b"\n")
                closing_offset = None
            context.validate_sources()
        # Windows requires all source and destination handles to be closed.
        os.replace(temporary, path)
        temporary = None
        return (_receipt(path, closing_offset, data, context.basis_polynomial_spans,
                         context.basis_source_identity) if closing_offset is not None else None)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def append_json_fields(path, fields, *, previous: JsonWriteState, progress=None):
    """Atomically add new top-level fields to a file written by ``write_json``.

    Copy already encoded source bytes, rather than serializing millions of
    polynomial terms again. An unchanged positive minimal-basis polynomial can
    also refer to its exact source span. No incomplete JSON file is published.
    """
    path = Path(path)
    if path.resolve() != previous.path:
        raise ValueError("JSON append receipt belongs to another path")
    _check_source(previous)
    if not isinstance(fields, dict):
        raise TypeError("Additional JSON fields must be a dictionary")
    if set(fields).intersection(previous.keys):
        raise ValueError("Additional JSON fields must not replace existing fields")
    temporary = None
    try:
        with output_gc_paused(), ExitStack() as stack, path.open("rb") as source, tempfile.NamedTemporaryFile(
                mode="wb", prefix="." + path.name + ".", suffix=".tmp",
                dir=path.parent, delete=False, buffering=1024 * 1024) as sink:
            temporary = Path(sink.name)
            context = _WriteContext(stack)
            context.bind_source(previous, source)
            remaining = previous.closing_offset
            while remaining:
                block = source.read(min(4 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError("JSON source ended before its recorded final field")
                sink.write(block)
                remaining -= len(block)
                if progress is not None:
                    progress("__copy__", 0, sink.tell())
            if source.read() != b"\n}\n":
                raise ValueError("JSON source no longer has the recorded closing delimiter")
            _write_fields(sink, fields, context=context, first=not previous.keys, progress=progress)
            closing_offset = sink.tell()
            sink.write(b"\n}\n")
            context.validate_sources()
        os.replace(temporary, path)
        temporary = None
        spans = (previous.basis_polynomial_spans if "basis" in previous.keys
                 else tuple(context.basis_polynomial_spans))
        identity = (previous.basis_source_identity if "basis" in previous.keys
                    else context.basis_source_identity)
        return _receipt(path, closing_offset, (*previous.keys, *fields), spans, identity)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
