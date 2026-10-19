"""Exact output, lazy arrays, and failure-safe saves across large integer boundaries."""
from contextlib import contextmanager
from dataclasses import replace
import gc
import json
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import weakref

from gvwzz.core import DECIMAL_BOUND
from gvwzz.output import JsonSavedPolynomial, JsonSequence, append_json_fields, write_json
from gvwzz.polynomial import IntegerPolynomialRing
from tests.postprocessing_checks import known_basis, solve


class PolynomialOutputChecks(unittest.TestCase):
    def test_cached_expression_matches_reference_in_all_orders(self):
        rng = random.Random(29)
        for order in ("lex", "grlex", "grevlex"):
            b = IntegerPolynomialRing(("x1", "x2", "u_3"), order)
            polynomials = [b.zero, b.one, -b.one]
            for coefficient in (-1, 1, -2, 2, 1 << 4097, -(1 << 4097),
                                DECIMAL_BOUND - 1, DECIMAL_BOUND, -DECIMAL_BOUND):
                polynomials.append(b.ring.from_dict({(1, 0, 0): coefficient,
                                                     (0, 0, 0): -coefficient}))
            for _ in range(70):
                terms = {tuple(rng.randrange(7) for _ in range(3)):
                         rng.choice((-13, -2, -1, 1, 2, 31))
                         for _ in range(rng.randrange(1, 20))}
                polynomials.append(b.ring.from_dict(terms))
            for p in polynomials:
                with self.subTest(order=order, terms=len(p)):
                    self.assertEqual(b.serialize(p), b._serialize_reference(p))

    def test_unusual_names_arbitrary_variables_and_decimal_guard(self):
        b = IntegerPolynomialRing(("α", "\u53d8\u91cf"), "lex")
        p = b.parse("3*α^2-\u53d8\u91cf+7")
        self.assertEqual(b.serialize(p), b._serialize_reference(p))
        b = IntegerPolynomialRing(tuple("v" + str(i) for i in range(129)))
        exponent = [0] * 129
        exponent[-1] = 257
        p = b.term(-(1 << 4097), tuple(exponent)) + b.one
        original = sys.get_int_max_str_digits()
        try:
            sys.set_int_max_str_digits(640)
            record = b.serialize(p)
            self.assertEqual(record, b._serialize_reference(p))
            self.assertIsNone(record["expression"])
            self.assertEqual(b.deserialize(record), p)
        finally:
            sys.set_int_max_str_digits(original)

    def test_serializer_keeps_public_validation_and_no_backend_cycle(self):
        for names in (("x",), ("α",)):
            b = IntegerPolynomialRing(names)
            b.serialize(b.one)
            foreign = IntegerPolynomialRing(("y",)).one
            with self.assertRaisesRegex(ValueError, "different ring"):
                b.serialize(foreign)
            ref = weakref.ref(b)
            del b
            # Refcount release must work even when output has cyclic GC paused.
            self.assertIsNone(ref())


class JsonOutputChecks(unittest.TestCase):
    def setUp(self):
        self.initial_gc = gc.isenabled()
        self.thresholds = gc.get_threshold()

    def tearDown(self):
        (gc.enable if self.initial_gc else gc.disable)()
        self.assertEqual(gc.get_threshold(), self.thresholds)

    def test_lazy_full_result_matches_eager_schema_and_loads(self):
        from gvwzz.cli import load_result
        from gvwzz.postprocessing import postprocess_basis
        # Representations, generation paths, insertion history and postprocessing
        # exercise every optional full-result branch without a long benchmark.
        result = solve(["2*x^2", "3*x*y+y", "y^2"], record_paths=True)
        result.polynomial_postprocessing = postprocess_basis(result)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "full.json"
            expected = result.to_dict()
            actual = result.to_json_dict()
            self.assertIsInstance(actual["basis"], JsonSequence)
            write_json(path, actual)
            self.assertEqual(json.loads(path.read_text()), expected)
            restored = load_result(path)
            self.assertEqual([p.polynomial for p in restored.basis],
                             [p.polynomial for p in result.basis])
            self.assertEqual(restored.syzygies, result.syzygies)
            write_json(path, result.to_json_dict(include_basis=False))
            self.assertEqual(json.loads(path.read_text()), result.to_dict(include_basis=False))

    def test_nested_vector_sequences_are_arrays(self):
        result = known_basis(["2*x", "3*y"])
        result.basis_representations = [[result.backend.one, result.backend.zero],
                                        [result.backend.zero, result.backend.one]]
        result.syzygy_representations = []
        with TemporaryDirectory() as folder:
            path = Path(folder) / "vectors.json"
            write_json(path, result.to_json_dict())
            self.assertEqual(json.loads(path.read_text()), result.to_dict())

    def test_arrays_created_only_during_write_and_gc_policy_restored(self):
        seen = []
        def records():
            self.assertFalse(gc.isenabled())
            for value in range(257):
                seen.append(value)
                yield {"value": value, "unicode": "α\u53d8\u91cf", "exponents": [0] * 129}
        with TemporaryDirectory() as folder:
            path = Path(folder) / "nested" / "result.json"
            data = {"basis": JsonSequence(records)}
            self.assertEqual(seen, [])
            for enabled in (False, True):
                (gc.enable if enabled else gc.disable)()
                seen.clear()
                progress = []
                write_json(path, data, progress=lambda *args: progress.append(args))
                self.assertEqual(seen, list(range(257)))
                self.assertEqual(len(json.loads(path.read_text())["basis"]), 257)
                self.assertEqual(gc.isenabled(), enabled)
                self.assertEqual([entry[1] for entry in progress], [0, 128, 256])
                self.assertGreater(progress[-1][2], progress[0][2])

    def test_atomic_failure_preserves_old_file_and_cleans_temp(self):
        def broken():
            yield {"ok": 1}
            raise RuntimeError("record failed")
        with TemporaryDirectory() as folder:
            path = Path(folder) / "result.json"
            original = b'{"original":true}\n'
            path.write_bytes(original)
            for enabled in (False, True):
                (gc.enable if enabled else gc.disable)()
                for failure, message in ((JsonSequence(broken), "record failed"),
                                         (JsonSequence(lambda: [object()]), "not JSON serializable")):
                    with self.assertRaisesRegex((RuntimeError, TypeError), message):
                        write_json(path, {"basis": failure})
                    self.assertEqual(path.read_bytes(), original)
                    self.assertEqual(list(Path(folder).iterdir()), [path])
                    self.assertEqual(gc.isenabled(), enabled)
            with patch("gvwzz.output.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    write_json(path, {"new": True})
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_append_copies_encoded_basis_without_serializing_again(self):
        calls = []
        def records():
            calls.append(1)
            yield {"coefficient": "-0x" + "f" * 1100, "exponents": [0] * 128 + [257]}
        with TemporaryDirectory() as folder:
            path = Path(folder) / "result.json"
            state = write_json(path, {"basis": JsonSequence(records)})
            prefix = path.read_bytes()[:state.closing_offset]
            state = append_json_fields(path, {"post": {"status": "completed"}}, previous=state)
            self.assertTrue(path.read_bytes().startswith(prefix))
            self.assertEqual(calls, [1])
            state = append_json_fields(path, {"timing": 1.25}, previous=state)
            actual = json.loads(path.read_text())
            self.assertEqual(actual["post"], {"status": "completed"})
            self.assertEqual(actual["timing"], 1.25)
            self.assertEqual(calls, [1])
            with self.assertRaisesRegex(ValueError, "replace existing"):
                append_json_fields(path, {"basis": []}, previous=state)
            before = path.read_bytes()
            with self.assertRaises(TypeError):
                append_json_fields(path, {"bad": object()}, previous=state)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(folder).iterdir()), [path])
            path.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, "changed"):
                append_json_fields(path, {"extra": True}, previous=state)

    def test_saved_spans_are_exact_polynomial_objects_including_unicode_and_braces(self):
        records = [
            {"signature": {"coefficient": "-0x" + "f" * 1100,
                           "index": 1, "exponents": [0] * 128 + [257]},
             "polynomial": {"expression": 'α + "quoted" + \\ + {braces}', "terms": [
                 {"coefficient": "0x" + "f" * 1100, "exponents": [0] * 128 + [257]}]}},
            {"signature": {"coefficient": "1", "index": 1, "exponents": [0]},
             "polynomial": {"expression": "0", "terms": []}},
        ]
        identity = object()
        with TemporaryDirectory() as folder:
            source = Path(folder) / "source.json"
            state = write_json(source, {"basis": JsonSequence(lambda: iter(records),
                                                               source_identity=identity)})
            self.assertIs(state.basis_source_identity, identity)
            self.assertEqual(len(state.basis_polynomial_spans), 2)
            raw = source.read_bytes()
            for index, (start, length) in enumerate(state.basis_polynomial_spans):
                self.assertTrue(state.has_saved_polynomial(index))
                self.assertEqual(json.loads(raw[start:start + length]), records[index]["polynomial"])
            extra = {"polynomial_postprocessing": {"minimal_basis": JsonSequence(
                lambda: (JsonSavedPolynomial(state, index) for index in (1, 0, 1)))}}
            original_open = Path.open
            opened = []
            def counting_open(path, *args, **kwargs):
                if path == source and args and args[0] == "rb":
                    opened.append(path)
                return original_open(path, *args, **kwargs)
            with patch.object(Path, "open", counting_open):
                final_state = append_json_fields(source, extra, previous=state)
            self.assertEqual(len(opened), 1)
            self.assertEqual(final_state.basis_polynomial_spans, state.basis_polynomial_spans)
            self.assertIs(final_state.basis_source_identity, identity)
            self.assertEqual(json.loads(source.read_text())["polynomial_postprocessing"]["minimal_basis"],
                             [records[index]["polynomial"] for index in (1, 0, 1)])
            # The new receipt can reference unchanged spans after the atomic replacement.
            copy = Path(folder) / "copy.json"
            write_json(copy, {"records": JsonSequence(lambda: (JsonSavedPolynomial(final_state, i)
                                                               for i in range(2)))})
            self.assertEqual(json.loads(copy.read_text())["records"],
                             [record["polynomial"] for record in records])

    def test_saved_span_failures_are_atomic_and_nonstandard_records_fall_back(self):
        records = [{"unexpected": []},
                   {"polynomial": {"expression": "x", "terms": []}, "signature": {}},
                   {"signature": {}, "polynomial": {"expression": "0", "terms": []}}]
        with TemporaryDirectory() as folder:
            source, output = Path(folder) / "source.json", Path(folder) / "output.json"
            state = write_json(source, {"basis": records})
            self.assertEqual(state.basis_polynomial_spans[:2], (None, None))
            self.assertTrue(state.has_saved_polynomial(2))
            self.assertFalse(state.has_saved_polynomial(True))
            self.assertFalse(state.has_saved_polynomial(-1))
            self.assertFalse(state.has_saved_polynomial(3))
            output.write_text('{"old":true}')
            original = output.read_bytes()
            invalid_states = (state, replace(state, basis_polynomial_spans=((1, state.size + 1),)))
            for invalid in invalid_states:
                with self.assertRaisesRegex(ValueError, "span"):
                    write_json(output, {"values": JsonSequence(lambda: [JsonSavedPolynomial(invalid, 0)])})
                self.assertEqual(output.read_bytes(), original)
            source.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, "changed"):
                write_json(output, {"values": JsonSequence(lambda: [JsonSavedPolynomial(state, 2)])})
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(set(Path(folder).iterdir()), {source, output})

    def test_empty_objects_arrays_and_general_json_roots(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "result.json"
            for data in ({}, [], None, {"none": None, "array": JsonSequence(lambda: [])}):
                state = write_json(path, data)
                actual = json.loads(path.read_text())
                if data == {}:
                    append_json_fields(path, {"first": []}, previous=state)
                    self.assertEqual(json.loads(path.read_text()), {"first": []})
                elif isinstance(data, dict):
                    self.assertEqual(actual, {"none": None, "array": []})
                else:
                    self.assertEqual(actual, data)


if __name__ == "__main__":
    unittest.main()
