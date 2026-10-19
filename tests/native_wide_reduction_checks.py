"""Independent reduction oracles for the portable wide monomial selector.

The same small reduction problem is embedded into distant coordinates. A
common signature monomial changes the encoding width without changing any
signature comparison, which makes packing boundaries observable without an
expensive high-degree polynomial computation.
"""
from dataclasses import replace
import unittest
from unittest.mock import patch

from gvwzz import Options
from gvwzz.core import Label, Signature
from gvwzz.orders import ModuleOrder
from gvwzz.polynomial import IntegerPolynomialRing
from gvwzz.solver import GVWInteger
import gvwzz.solver as solver_module
from tests.native_bulk_checks import COUNTERS, fixture, native_module, records, reduce


def lifted_fixture(options=None, *, offset=8, factor=1, weighted=False):
    """Keep polynomial degree four while lifting ratios beyond 64 bits."""
    original, label = fixture(options)
    backend = IntegerPolynomialRing([f"x{i}" for i in range(12)], "grevlex")
    x, y = backend.gens[0], backend.gens[-1]
    inputs = [x**3 + y, y**2 + 1, x*y + 1] if weighted else [backend.one]*3
    order = ModuleOrder(backend, inputs, "g2", positions=[2, 0, 1] if weighted else None)
    engine = GVWInteger(backend, inputs, order, options or Options())

    def monomial(value, signature=False):
        lifted = [0]*12
        lifted[0], lifted[-1] = value
        if signature:
            lifted[6] = offset
        return tuple(lifted)

    def lift(row, multiplier=1):
        signature = Signature(multiplier*row.signature.coefficient,
                              monomial(row.signature.monomial, True), row.signature.index)
        polynomial = backend.ring.from_dict({monomial(m): multiplier*c
                                             for m, c in row.polynomial.items()})
        return Label(signature, polynomial)

    for row in original.P:
        engine._insert_basis(lift(row))
    return engine, lift(label, factor)


@unittest.skipUnless(native_module is not None
    and getattr(native_module, "BULK_REDUCTION_AVAILABLE", False)
    and solver_module._native_reduce_working is not None,
    "optional GMP bulk reduction is unavailable or disabled")
class NativeWideReductionChecks(unittest.TestCase):
    def assert_pair(self, make, *, tail=False, first_strict=False, expected_bulk=True,
                    expected_wide=True):
        answers = []
        for native in (True, False):
            engine, label = make()
            before = records(row.polynomial for row in engine.P)
            original = label.polynomial.copy()
            result, counters = reduce(engine, label, native=native, tail=tail,
                                      first_strict=first_strict)
            self.assertEqual(records(row.polynomial for row in engine.P), before)
            self.assertEqual(label.polynomial, original)
            if tail:
                self.assertEqual(engine.backend.leading(result.polynomial),
                                 engine.backend.leading(original))
            self.assertEqual(bool(engine.stats.counters["native_bulk_reduction_workspaces"]),
                             native and expected_bulk)
            self.assertEqual(bool(engine.stats.counters["native_wide_bulk_reduction_workspaces"]),
                             native and expected_bulk and expected_wide)
            answers.append((result.signature, records([result.polynomial]), counters))
        self.assertEqual(answers[0], answers[1])
        return answers[0]

    def test_wide_head_tail_policy_matrix_and_signed_huge_integers(self):
        huge = (1 << 4096) + 27
        observed = {"singular": 0, "euclidean": 0, "tail": 0}
        for selection in ("oldest", "sparse", "unit", "height", "balanced", "newest"):
            for policy in ("regular-first", "singular-first", "first"):
                for priority in ("signature-first", "exact-first"):
                    for enabled in (False, True):
                        options = Options(reducer_selection=selection, reducer_policy=policy,
                                          euclidean_priority=priority, euclidean=enabled)
                        for factor in (1, -huge):
                            for tail in (False, True):
                                with self.subTest(selection=selection, policy=policy,
                                                  priority=priority, euclidean=enabled,
                                                  huge=factor != 1, tail=tail):
                                    _, _, values = self.assert_pair(
                                        lambda: lifted_fixture(options, factor=factor), tail=tail)
                                    counters = dict(zip(COUNTERS, values))
                                    observed["singular"] += counters["singular_steps"]
                                    observed["euclidean"] += (counters["euclidean_steps"]
                                                               + counters["tail_euclidean_steps"])
                                    observed["tail"] += counters["tail_steps"]
        self.assertTrue(all(observed.values()), observed)

    def test_common_signature_translation_crosses_width_boundaries(self):
        # The ratio bound equals offset + 8. At 12 variables, nine-bit
        # lanes fit the wide selector; ten-bit lanes exceed its exact bound.
        reference = None
        for offset in (0, 7, 8, 23, 24, 503, 504, 8):
            for tail in (False, True):
                with self.subTest(offset=offset, tail=tail):
                    signature, polynomial, counters = self.assert_pair(
                        lambda: lifted_fixture(offset=offset), tail=tail,
                        expected_bulk=offset < 504, expected_wide=8 <= offset < 504)
                    normalized = (signature.coefficient,
                                  signature.monomial[:6] + (0,) + signature.monomial[7:],
                                  signature.index, polynomial, counters)
                    if reference is None:
                        reference = {}
                    if tail in reference:
                        self.assertEqual(normalized, reference[tail])
                    else:
                        reference[tail] = normalized

    def test_weighted_positions_strict_first_and_budget_fallback(self):
        for offset in (8, 24):
            self.assert_pair(lambda: lifted_fixture(offset=offset, weighted=True))
            self.assert_pair(lambda: lifted_fixture(offset=offset, weighted=True), tail=True)
            self.assert_pair(lambda: lifted_fixture(offset=offset), first_strict=True)
        for switches in (dict(tail_step_cap=2), dict(max_reduction_steps=100000),
                         dict(max_polynomial_terms=1000), dict(max_coefficient_bits=10000),
                         dict(debug=True)):
            with self.subTest(switches=switches):
                self.assert_pair(lambda: lifted_fixture(Options(**switches)),
                                 tail=True, expected_bulk=False)

    def test_wide_progress_interrupt_flushes_counters_and_keeps_basis(self):
        # Sparse terms keep the accumulator compact while the common
        # signature offset forces the portable wide selector.
        def chain(exponent):
            backend = IntegerPolynomialRing([f"x{i}" for i in range(12)], "grevlex")
            x = backend.gens[-1]
            inputs = [backend.one]*3
            engine = GVWInteger(backend, inputs, ModuleOrder(backend, inputs, "g2"),
                                Options(progress_interval=1))
            base = (0,)*6 + (480,) + (0,)*5
            engine._insert_basis(Label(Signature(1, base, 0), x - 1))
            monomial = base[:-1] + (exponent,)
            return engine, Label(Signature(17, monomial, 2), x**exponent)

        # Fifteen steps are enough to exercise the completion callback; the
        # existing long-chain tests cover interruption at the chunk boundary.
        engine, label = chain(15)
        frozen = records(row.polynomial for row in engine.P)
        snapshots = []
        def callback():
            snapshots.append((engine.stats.counters["reducer_queries"],
                              engine.stats.counters["regular_steps"]))
            raise KeyboardInterrupt("wide selector callback")
        with patch.object(engine, "_check_budget", callback), self.assertRaises(KeyboardInterrupt):
            reduce(engine, label, native=True)
        self.assertEqual(snapshots, [(16, 15)])
        self.assertEqual(records(row.polynomial for row in engine.P), frozen)
        self.assertEqual(engine.stats.counters["native_bulk_reduction_workspaces"], 1)
        self.assertEqual(engine.stats.counters["native_wide_bulk_reduction_workspaces"], 1)
        engine.options = replace(engine.options, progress_interval=0)
        result, _ = reduce(engine, label, native=True)
        self.assertEqual(result.polynomial, engine.backend.one)
        self.assertEqual(engine.stats.counters["regular_steps"], 30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
