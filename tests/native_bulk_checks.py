"""Independent exact oracles for the optional complete native reduction loop.

Eligible tests deliberately have no abort budgets or certificate tracking and
use G2/grevlex. Every such run must increment the bulk workspace counter;
testing only the older native per-step path is not sufficient here.
"""
from contextlib import ExitStack
from dataclasses import replace
import unittest
from unittest.mock import patch

from gvwzz import Options, compute, verify_generation_paths
from gvwzz.core import Label, Signature
from gvwzz.orders import ModuleOrder
from gvwzz.polynomial import IntegerPolynomialRing
from gvwzz.solver import GVWInteger
from gvwzz.verification import verify_result
import gvwzz.solver as solver_module

try:
    from gvwzz import _native_reduction as native_module
except ImportError:
    native_module = None


COUNTERS = ("reducer_queries", "reducer_candidates", "euclidean_sigdrop_skipped",
    "regular_steps", "singular_steps", "euclidean_steps", "euclidean_regular_steps",
    "euclidean_singular_steps", "tail_steps", "tail_exact_steps", "tail_euclidean_steps",
    "tail_cap_reached")


def records(polynomials):
    return tuple(tuple((m, int(c)) for m, c in sorted(p.items())) for p in polynomials)


def fixture(options=None, module="g2", order="grevlex", dimension=2):
    backend = IntegerPolynomialRing(["x", "y"] + [f"z{i}" for i in range(dimension-2)], order)
    inputs = [backend.one, backend.one, backend.one]
    engine = GVWInteger(backend, inputs, ModuleOrder(backend, inputs, module), options or Options())
    rows = [
        ("2*x^2+3*x*y-5*y+1", 1, (2, 0), 2),
        ("-3*x^2+2*y^2+7", -2, (2, 0), 2),
        ("5*x*y-3*y^2+x", 3, (1, 1), 0),
        ("-7*y^2+2*y-1", 2, (0, 2), 1),
        ("x^2-x*y+2", 1, (3, 0), 2),
        ("3*y+2", 1, (0, 1), 0),
        ("-2*x+y-1", 1, (1, 0), 1),
    ]
    for polynomial, coefficient, monomial, index in rows:
        engine._insert_basis(Label(Signature(coefficient, monomial + (0,)*(dimension-2), index),
                                   backend.parse(polynomial)))
    polynomial = backend.parse("14*x^4-11*x^3+6*x^2*y+13*x*y^2-9*y^3+5*x^2+7*y^2-17*x+19*y+23")
    return engine, Label(Signature(17, (4, 0) + (0,)*(dimension-2), 2), polynomial)


def reduce(engine, label, *, native, tail=False, first_strict=False):
    with ExitStack() as stack:
        if not native:
            stack.enter_context(patch.object(solver_module, "_native_reduce_working", None))
            stack.enter_context(patch.object(solver_module, "_native_find_reducer", None))
        if first_strict:
            if native:
                result = native_module.reduce_working(engine, label, False, True)
            else:
                chosen = engine._find_reducer(label.signature, engine.backend.leading(label.polynomial), "strict")
                if chosen is None:
                    raise AssertionError("test case must admit an exact strict first reducer")
                result = engine._reduce_working(engine._apply_reduction(label, chosen))
        else:
            result = engine._reduce_working(label, tail=tail)
    return result, tuple(engine.stats.counters[key] for key in COUNTERS)


@unittest.skipUnless(native_module is not None
    and getattr(native_module, "BULK_REDUCTION_AVAILABLE", False)
    and solver_module._native_reduce_working is not None,
    "optional GMP bulk reduction is unavailable or disabled")
class NativeBulkChecks(unittest.TestCase):
    def assert_pair(self, make, *, tail=False, first_strict=False, expected_bulk=True):
        answers = []
        for native in (True, False):
            engine, label = make()
            before = records(p.polynomial for p in engine.P)
            original = label.polynomial.copy()
            result, counters = reduce(engine, label, native=native, tail=tail, first_strict=first_strict)
            self.assertEqual(records(p.polynomial for p in engine.P), before)
            self.assertEqual(label.polynomial, original)
            if tail:
                self.assertEqual(engine.backend.leading(result.polynomial), engine.backend.leading(original))
            self.assertEqual(bool(engine.stats.counters["native_bulk_reduction_workspaces"]),
                             native and expected_bulk)
            answers.append((result.signature, records([result.polynomial]), counters))
        self.assertEqual(answers[0], answers[1])
        return answers[0]

    def test_policy_matrix_head_tail_singular_euclidean_and_huge_coefficients(self):
        singular = euclidean = tail_steps = signature_changes = 0
        huge = (1 << 4096) + 27
        for selection in ("oldest", "sparse", "unit", "height", "balanced", "newest"):
            for policy in ("regular-first", "singular-first", "first"):
                for priority in ("signature-first", "exact-first"):
                    for enabled in (False, True):
                        options = Options(reducer_selection=selection, reducer_policy=policy,
                            euclidean_priority=priority, euclidean=enabled)
                        for factor in (1, -huge):
                            def make():
                                engine, label = fixture(options)
                                return engine, Label(Signature(17*factor, label.signature.monomial, 2),
                                                     factor*label.polynomial)
                            for tail in (False, True):
                                with self.subTest(selection=selection, policy=policy, priority=priority,
                                                  euclidean=enabled, huge=factor != 1, tail=tail):
                                    signature, _, values = self.assert_pair(make, tail=tail)
                                    counters = dict(zip(COUNTERS, values))
                                    singular += counters["singular_steps"] + counters["euclidean_singular_steps"]
                                    euclidean += counters["euclidean_steps"] + counters["tail_euclidean_steps"]
                                    tail_steps += counters["tail_steps"]
                                    signature_changes += signature.coefficient != 17*factor
        self.assertGreater(singular, 0)
        self.assertGreater(euclidean, 0)
        self.assertGreater(tail_steps, 0)
        self.assertGreater(signature_changes, 0)

    def test_strict_first_empty_constant_and_no_reducer_workspaces(self):
        self.assert_pair(fixture, first_strict=True)
        for polynomial in ("0", "6"):
            def make():
                engine, label = fixture()
                return engine, Label(label.signature, engine.backend.parse(polynomial))
            self.assert_pair(make)

        def constant_case():
            backend = IntegerPolynomialRing(["x", "y"], "grevlex")
            inputs = [backend.one]*3
            engine = GVWInteger(backend, inputs, ModuleOrder(backend, inputs, "g2"), Options())
            engine._insert_basis(Label(Signature(1, (0, 0), 0), backend.parse("2")))
            return engine, Label(Signature(7, (0, 0), 2), backend.parse("6"))
        self.assert_pair(constant_case, first_strict=True)

        def no_reducer():
            engine, label = fixture()
            backend = engine.backend
            empty = GVWInteger(backend, engine.inputs, engine.order, Options())
            return empty, label
        self.assert_pair(no_reducer)
        self.assert_pair(no_reducer, tail=True)

    def test_budget_debug_order_and_width_guards_keep_exact_fallback(self):
        switches = [dict(max_seconds=3600), dict(max_basis=1000),
            dict(max_candidates=100000), dict(max_reduction_steps=100000),
            dict(max_polynomial_terms=1000), dict(max_coefficient_bits=10000),
            dict(debug=True), dict(tail_step_cap=2)]
        for options in switches:
            with self.subTest(options=options):
                self.assert_pair(lambda: fixture(Options(**options)), tail=True, expected_bulk=False)
        for module, order, dimension in (("pot", "grevlex", 2), ("g2", "grlex", 2),
                                          ("g2", "grevlex", 80)):
            self.assert_pair(lambda: fixture(module=module, order=order, dimension=dimension),
                             tail=True, expected_bulk=False)
        def deadline():
            engine, label = fixture()
            engine.deadline = 10**30
            return engine, label
        self.assert_pair(deadline, tail=True, expected_bulk=False)

    def test_path_and_representation_tracking_fall_back_with_verified_certificates(self):
        for switches in (dict(record_paths=True), dict(track_representations=True)):
            answers = []
            for native in (True, False):
                with ExitStack() as stack:
                    if not native:
                        stack.enter_context(patch.object(solver_module, "_native_reduce_working", None))
                        stack.enter_context(patch.object(solver_module, "_native_find_reducer", None))
                    result = compute(["-4*x^2+2*y+3", "6*x*y-5", "9*y^2+x"], ["x", "y"],
                        "grevlex", module_order="g2", options=Options(**switches),
                        show_basis=False, show_u=False)
                self.assertEqual(result.status, "completed")
                self.assertEqual(result.stats.counters["native_bulk_reduction_workspaces"], 0)
                if switches.get("record_paths"):
                    report = verify_generation_paths(result, strong=True)
                else:
                    report = verify_result(result, strong=True, mode="static", require_certificates=True)
                self.assertTrue(report["passed"])
                self.assertTrue(report["signature_basis_certified"])
                raw = result.to_dict()
                answers.append(({key: raw.get(key) for key in
                    ("basis", "syzygy_signatures", "representation_certificate", "generation_paths")},
                    tuple(result.stats.counters[key] for key in COUNTERS)))
            self.assertEqual(answers[0], answers[1])

    @staticmethod
    def long_chain(exponent=2500):
        backend = IntegerPolynomialRing(["x"], "grevlex")
        x = backend.gens[0]
        inputs = [backend.one]*3
        engine = GVWInteger(backend, inputs, ModuleOrder(backend, inputs, "g2"), Options())
        engine._insert_basis(Label(Signature(1, (0,), 0), x - 1))
        return engine, Label(Signature(17, (exponent,), 2), x**exponent)

    def test_progress_observes_flushed_bounded_chunks(self):
        engine, label = self.long_chain()
        engine.options = replace(engine.options, progress_interval=1)
        snapshots = []
        def callback():
            snapshots.append((engine.stats.counters["reducer_queries"], engine.stats.counters["regular_steps"]))
        with patch.object(engine, "_check_budget", callback):
            result, _ = reduce(engine, label, native=True)
        self.assertEqual(result.polynomial, engine.backend.one)
        self.assertEqual(engine.stats.counters["native_bulk_reduction_workspaces"], 1)
        self.assertEqual(snapshots[-1], (2501, 2500))
        self.assertGreaterEqual(len(snapshots), 3)
        previous = 0
        for queries, steps in snapshots:
            self.assertGreater(queries, previous)
            self.assertLessEqual(queries - previous, 1024)
            self.assertIn(queries - steps, (0, 1))
            previous = queries

    def test_progress_exception_and_keyboard_interrupt_preserve_counters_and_reuse(self):
        class CallbackFailure(RuntimeError):
            pass
        for exception in (CallbackFailure, KeyboardInterrupt):
            engine, label = self.long_chain()
            original = records(p.polynomial for p in engine.P)
            engine.options = replace(engine.options, progress_interval=1)
            snapshots = []
            def callback():
                snapshots.append((engine.stats.counters["reducer_queries"], engine.stats.counters["regular_steps"]))
                raise exception("deterministic callback interruption")
            with patch.object(engine, "_check_budget", callback), self.assertRaises(exception):
                reduce(engine, label, native=True)
            self.assertEqual(snapshots, [(1024, 1024)])
            self.assertEqual(records(p.polynomial for p in engine.P), original)
            engine.options = replace(engine.options, progress_interval=0)
            result, _ = reduce(engine, Label(Signature(17, (17,), 2), engine.backend.gens[0]**17), native=True)
            self.assertEqual(result.polynomial, engine.backend.one)
            self.assertEqual(engine.stats.counters["regular_steps"], 1024 + 17)
            self.assertEqual(engine.stats.counters["native_bulk_reduction_workspaces"], 2)

    def test_cached_reducer_owners_survive_multiple_contexts_and_repeated_updates(self):
        engine, _ = self.long_chain(5)
        before = records(p.polynomial for p in engine.P)
        huge = (1 << 4096) + 57
        for exponent in (5, 9, 17, 33, 65, 5):
            x = engine.backend.gens[0]
            label = Label(Signature(17, (exponent,), 2), huge*x**exponent - (huge - 3)*x)
            actual, _ = reduce(engine, label, native=True)
            oracle, _ = self.long_chain(exponent)
            ox = oracle.backend.gens[0]
            expected, _ = reduce(oracle, Label(label.signature,
                huge*ox**exponent - (huge - 3)*ox), native=False)
            self.assertEqual(actual.signature, expected.signature)
            self.assertEqual(records([actual.polynomial]), records([expected.polynomial]))
            self.assertEqual(records(p.polynomial for p in engine.P), before)
            self.assertEqual(actual.polynomial, engine.backend.parse("3"))
        self.assertEqual(engine.stats.counters["native_bulk_reduction_workspaces"], 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
