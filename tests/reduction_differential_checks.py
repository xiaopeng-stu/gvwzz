"""Bounded reduction comparisons against a bundled, exact reference snapshot.

The default checks native and pure-Python execution in fresh interpreters and
needs no previous release::

    python -m tests.reduction_differential_checks

An explicit snapshot can also be selected::

    python -m tests.reduction_differential_checks --compare-snapshot path/to/reference.json

To create another snapshot from a separately built source tree::

    python -m tests.reduction_differential_checks --reference ../reference --snapshot reference.json

The 32 integration cases are separate from the unittest boundary checks in this
module. No wall-time limit is treated as a successful comparison. Deterministic
budget stops must preserve the ordered state and work counters.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

# Workers execute this file by absolute path while importing the selected
# source tree later. Keep the test helpers importable without installing them.
ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from tests.differential_checks import LOGICAL_COUNTERS, RESULT_FIELDS, first_difference


COUNTERS = LOGICAL_COUNTERS + ["tail_cap_reached", "tail_selective_skipped"]


class NativeCoefficientBoundaryChecks(unittest.TestCase):
    """An independent SymPy oracle across machine-integer boundaries."""

    @staticmethod
    def workspace_type():
        from gvwzz import solver
        if solver._native_reduce_working is None:
            raise unittest.SkipTest("native reduction extension is unavailable/disabled")
        from gvwzz._native_reduction import NativeWorkspace
        return NativeWorkspace

    def check_updates(self, backend, polynomial, operations):
        workspace = self.workspace_type()(backend, polynomial)
        expected = polynomial.copy()
        initial = polynomial.copy()
        for other, coefficient, shift in operations:
            old_other = other.copy()
            workspace.subtract_multiple(other, coefficient, shift)
            expected -= backend.multiply_term(other, coefficient, shift)
            self.assertEqual(workspace.result(), expected)
            self.assertEqual(workspace.leading(), backend.leading(expected) if expected else None)
            self.assertEqual(other, old_other, "cached reducer polynomial was mutated")
            self.assertEqual(polynomial, initial, "retained input polynomial was mutated")
        # Removed/reintroduced terms must not leave a stale heap entry or make
        # the same monomial appear twice while finishing the remainder.
        remaining = expected.copy()
        while remaining:
            leading = backend.leading(remaining)
            self.assertEqual(workspace.leading(), leading)
            workspace.finish_term()
            del remaining[leading[0]]
        self.assertIsNone(workspace.leading())
        self.assertEqual(workspace.result(), expected)

    def test_signed_64bit_boundaries_and_negation(self):
        from gvwzz.polynomial import IntegerPolynomialRing
        edge = 1 << 63
        values = (-edge - 1, -edge, -edge + 1, -1, 0, 1,
                  edge - 1, edge, edge + 1)
        for order in ("grlex", "grevlex"):
            backend = IntegerPolynomialRing(["x", "y"], order)
            x, y = backend.gens
            for a in values:
                for b in (-edge, -1, 1, edge - 1, edge):
                    with self.subTest(order=order, a=a, b=b):
                        p = x**4 + a*x**2 + b*y + 7
                        q = b*x**2 + a*y + 1
                        self.check_updates(backend, p, [(q, -1, (0, 0)),
                            (q, 1, (0, 0)), (x**2 + y, b, (0, 0)),
                            (x**2 + y, -b, (0, 0))])

    def test_product_and_addition_overflow_then_exact_cancellation(self):
        from gvwzz.polynomial import IntegerPolynomialRing
        edge = 1 << 63
        for order in ("grlex", "grevlex"):
            backend = IntegerPolynomialRing(["x", "y"], order)
            x, y = backend.gens
            p = x**4 + (edge - 2)*x**2 + (-edge + 2)*y**2 + 1
            q = ((1 << 62) + 3)*x**2 + (edge - 1)*y**2 + edge - 1
            self.check_updates(backend, p, [
                (x**2 - y**2, -3, (0, 0)),  # addition crosses both signs
                (q, 3, (0, 0)),             # product crosses int64
                (q, -3, (0, 0)),
                (x**2 - y**2, 3, (0, 0)),
                (p, 1, (0, 0)),             # all pending coefficients cancel
                (p, -1, (0, 0)),            # reinsert the original support
            ])

    def test_arbitrary_precision_multiplier_and_small_result_cancellation(self):
        from gvwzz.polynomial import IntegerPolynomialRing
        for order in ("grlex", "grevlex"):
            backend = IntegerPolynomialRing(["x", "y"], order)
            x, y = backend.gens
            p = x**5 + 7*x**2 + 2*y + 3
            q = 3*x**2 - 5*y + 1
            for huge in ((1 << 128) + 13, -((1 << 4096) + 53)):
                self.check_updates(backend, p, [(q, huge, (1, 0)),
                    (q, -huge, (1, 0)), (x, 2, (0, 1))])

    def test_public_integer_values_and_shared_reducer_cache(self):
        from gvwzz.polynomial import IntegerPolynomialRing
        backend = IntegerPolynomialRing(["x", "y"], "grevlex")
        x, y = backend.gens
        huge = (1 << 2048) + 31
        p = x**4 + 7*y + 1
        q = huge*x**2 - (huge + 3)*y + 2
        original_q = q.copy()
        workspace = self.workspace_type()
        first, second = workspace(backend, p), workspace(backend, p)
        first.subtract_multiple(q, huge, (0, 0))
        frozen_result = first.result()
        second.subtract_multiple(q, -huge, (0, 0))
        second.subtract_multiple(q, 3*huge, (0, 0))
        self.assertEqual(first.result(), frozen_result)
        self.assertEqual(q, original_q, "a shared cached reducer coefficient changed")
        third = workspace(backend, p)
        third.subtract_multiple(q, huge, (0, 0))
        self.assertEqual(third.result(), frozen_result)
        for work in (first, second, third):
            self.assertTrue(all(type(c) is int for c in work.polynomial.values()))
            self.assertIs(type(work.leading()[1]), int)

    def test_large_packed_keys_keep_exact_fallback(self):
        from gvwzz.polynomial import IntegerPolynomialRing
        for order in ("grlex", "grevlex"):
            backend = IntegerPolynomialRing([f"x{i}" for i in range(80)], order)
            x, y = backend.gens[0], backend.gens[-1]
            huge = (1 << 1024) + 7
            p = x**5 + huge*y**2 + 3
            q = huge*x**2 - 2*y + 1
            shift = (1,) + (0,)*79
            self.check_updates(backend, p, [(q, huge, shift), (q, -huge, shift)])


class PackedRatioBoundaryChecks(unittest.TestCase):
    """Compare packed admission with the independent tuple-key Python finder."""

    @staticmethod
    def native_api():
        from gvwzz import solver
        if solver._native_find_reducer is None:
            raise unittest.SkipTest("native reduction extension is unavailable/disabled")
        try:
            from gvwzz._native_reduction import _PackedRatioContext, find_reducer
        except ImportError:
            raise unittest.SkipTest("packed ratio context is not part of this build")
        return _PackedRatioContext, find_reducer

    @staticmethod
    def engine(scale, selection="newest"):
        from gvwzz import Options
        from gvwzz.core import Label, Signature
        from gvwzz.orders import ModuleOrder
        from gvwzz.polynomial import IntegerPolynomialRing
        from gvwzz.solver import GVWInteger
        backend = IntegerPolynomialRing(["x", "y"], "grevlex")
        weights = ((2*scale, 0), (0, 2*scale), (scale, scale))
        inputs = [backend.ring.from_dict({m: 1}) for m in weights]
        order = ModuleOrder(backend, inputs, "g2", positions=[2, 0, 1])
        engine = GVWInteger(backend, inputs, order, Options(reducer_selection=selection))
        exponents = ((0, 2*scale), (2*scale, 0), (scale, scale))
        for k in range(6):
            # All weighted signatures equal (2s,2s), but position ranks differ;
            # both x formal ratios are negative and some total degrees are too.
            m = ((3 if k < 3 else 5)*scale, 0)
            p = backend.ring.from_dict({m: (2, -3, 5)[k % 3], (0, 0): 7})
            sig = Signature((1, -2, 3)[k % 3], exponents[k % 3], k % 3)
            engine._insert_basis(Label(sig, p))
        return engine, exponents

    def compare_query(self, engine, signature, leading, bound, mode, euclidean,
                      expected_geometry=None):
        from gvwzz import solver
        context_type, native = self.native_api()
        context = context_type(engine, signature, bound)
        native_geometry = bool(getattr(context, "native_geometry_enabled", False))
        if expected_geometry is not None:
            self.assertEqual(native_geometry, expected_geometry)
        counter_names = ("reducer_candidates", "euclidean_sigdrop_skipped")
        before = {key: engine.stats.counters[key] for key in counter_names}
        actual = native(engine, signature, leading, mode, euclidean, False, context)
        actual_counts = {key: engine.stats.counters[key] - before[key] for key in counter_names}
        # Preserve native admission caches between queries, including when
        # only the signature or polynomial coefficient changes. A fresh
        # Python oracle query must neither consume nor overwrite that cache.
        packed = native_geometry or (bool(engine.reducer_query_cache)
            and len(next(reversed(engine.reducer_query_cache))) >= 4)
        native_cache = engine.reducer_query_cache.copy()
        engine.reducer_query_cache.clear()
        old_native = solver._native_find_reducer
        before = {key: engine.stats.counters[key] for key in counter_names}
        try:
            solver._native_find_reducer = None
            expected = engine._find_reducer(signature, leading, mode, euclidean)
        finally:
            solver._native_find_reducer = old_native
            engine.reducer_query_cache.clear()
            engine.reducer_query_cache.update(native_cache)
        expected_counts = {key: engine.stats.counters[key] - before[key] for key in counter_names}
        self.assertEqual(actual, expected, (signature, leading, bound, mode, euclidean))
        self.assertEqual(actual_counts, expected_counts,
                         (signature, leading, bound, mode, euclidean))
        return packed

    def test_negative_ratios_position_ties_and_signature_coefficients(self):
        from dataclasses import replace
        from gvwzz.core import Signature
        self.native_api()
        for selection in ("oldest", "newest", "sparse"):
            engine, exponents = self.engine(7, selection)
            for policy in ("regular-first", "singular-first", "first"):
                for priority in ("signature-first", "exact-first"):
                    engine.options = replace(engine.options, reducer_policy=policy,
                                             euclidean_priority=priority)
                    for index in range(3):
                        for sig_coefficient in (1, -2, (1 << 190) + 1):
                            signature = Signature(sig_coefficient, exponents[index], index)
                            for m in ((21, 0), (35, 0)):
                                for coefficient in (6, -7, (1 << 200) + 3):
                                    for mode, euclidean in (("strict", False), ("mixed", True)):
                                        self.assertTrue(self.compare_query(engine, signature,
                                            (m, coefficient), 56, mode, euclidean))

    def test_lane_boundary_basis_append_and_larger_workspace(self):
        from gvwzz.core import Label, Signature
        self.native_api()
        # 12*s straddles 2**42; with two variables, 42-bit packing occupies
        # 126 bits and the next lane width must use tuple-key fallback.
        for scale in (((1 << 42) - 1)//12, ((1 << 42) - 1)//12 + 1):
            engine, exponents = self.engine(scale)
            signature = Signature(1, exponents[0], 0)
            leading = ((3*scale, 0), 6)
            before = self.compare_query(engine, signature, leading, 8*scale, "mixed", True)
            self.assertEqual(before, 12*scale < (1 << 42))
            p = engine.backend.ring.from_dict({(3*scale, 0): 1, (0, 0): -9})
            engine._insert_basis(Label(Signature(1, (0, 0), 0), p))
            for bound in (8*scale, 16*scale, 8*scale):
                packed = self.compare_query(engine, signature, leading, bound, "mixed", True)
                self.assertEqual(packed, 4*scale + bound < (1 << 42))

    def test_context_reuse_after_singular_signature_coefficient_changes(self):
        from dataclasses import replace
        from gvwzz import solver
        from gvwzz.core import Signature
        context_type, native = self.native_api()
        engine, exponents = self.engine(7)
        counter_names = ("reducer_candidates", "euclidean_sigdrop_skipped")
        huge = (1 << 190) + 23
        for policy in ("regular-first", "singular-first", "first"):
            for priority in ("signature-first", "exact-first"):
                engine.options = replace(engine.options, reducer_policy=policy,
                                         euclidean_priority=priority)
                initial = Signature(4, exponents[1], 1)
                context = context_type(engine, initial, 56)
                # Only the coefficient changes, exactly as after a singular
                # step during one fixed native workspace lifetime. ±4 makes
                # an equal-signature row super for selected exact/Euclidean
                # scalars, so stale coefficient caches change the answer.
                for coefficient in (4, -4, huge, -huge, huge + 1, 1, 4):
                    signature = Signature(coefficient, initial.monomial, initial.index)
                    for leading_coefficient in (6, -7, (1 << 900) + 15, -((1 << 901) + 17)):
                        leading = ((21, 0), leading_coefficient)
                        for mode, euclidean in (("strict", False), ("mixed", True)):
                            old = {key: engine.stats.counters[key] for key in counter_names}
                            actual = native(engine, signature, leading, mode, euclidean,
                                            False, context)
                            actual_counts = {key: engine.stats.counters[key] - old[key]
                                             for key in counter_names}
                            native_cache = engine.reducer_query_cache.copy()
                            engine.reducer_query_cache.clear()
                            old = {key: engine.stats.counters[key] for key in counter_names}
                            old_native = solver._native_find_reducer
                            try:
                                solver._native_find_reducer = None
                                expected = engine._find_reducer(signature, leading, mode, euclidean)
                            finally:
                                solver._native_find_reducer = old_native
                                engine.reducer_query_cache.clear()
                                engine.reducer_query_cache.update(native_cache)
                            expected_counts = {key: engine.stats.counters[key] - old[key]
                                               for key in counter_names}
                            self.assertEqual(actual, expected, (policy, priority, signature, leading))
                            self.assertEqual(actual_counts, expected_counts,
                                             (policy, priority, signature, leading))

    def test_divisibility_guard_lanes_zero_and_high_bit_with_fallbacks(self):
        from dataclasses import replace
        from gvwzz import Options
        from gvwzz.core import Label, Signature
        from gvwzz.orders import ModuleOrder
        from gvwzz.polynomial import IntegerPolynomialRing
        from gvwzz.solver import GVWInteger
        self.native_api()
        from gvwzz._native_reduction import COEFFICIENT_BACKEND
        # Retain the old 64-bit boundaries and check wide guards at bits
        # 63/127. (33,7) exceeds geometry width; (32,8) exceeds ratio width.
        for dimension, bound in ((16, 7), (17, 7), (16, 8), (21, 3),
                                  (32, 7), (33, 7), (32, 8), (42, 3), (43, 3)):
            backend = IntegerPolynomialRing([f"x{i}" for i in range(dimension)], "grevlex")
            zero = (0,)*dimension
            largest = min(bound, 7)

            def exponent(first=0, last=0, middle=0):
                values = [0]*dimension
                values[0], values[-1], values[dimension//2] = first, last, middle
                return tuple(values)

            monomials = [zero, exponent(largest), exponent(last=largest),
                exponent(largest - 1, 1), exponent(1, largest - 1),
                exponent(middle=largest), exponent(1)]
            inputs = [backend.one, backend.one, backend.one]
            for selection in ("oldest", "newest", "sparse"):
                order = ModuleOrder(backend, inputs, "g2", positions=[2, 0, 1])
                engine = GVWInteger(backend, inputs, order, Options(reducer_selection=selection))
                for k, monomial in enumerate(monomials):
                    terms = {monomial: (11, 2, -3, 5, -7, 1, -1)[k]}
                    if monomial != zero:
                        terms[zero] = 13
                    engine._insert_basis(Label(Signature((1, -2, 3)[k % 3], zero, k % 3),
                                               backend.ring.from_dict(terms)))
                for appended in (False, True):
                    if appended:
                        p = backend.ring.from_dict({exponent(last=largest): 1, zero: -17})
                        engine._insert_basis(Label(Signature(1, zero, 2), p))
                    for policy in ("regular-first", "singular-first", "first"):
                        engine.options = replace(engine.options, reducer_policy=policy)
                        for monomial in monomials:
                            for coefficient in (6, -7, (1 << 200) + 3):
                                signature = Signature(coefficient, zero, 1)
                                for mode, euclidean in (("strict", False), ("mixed", True)):
                                    packed = self.compare_query(engine, signature,
                                        (monomial, coefficient), bound, mode, euclidean,
                                        expected_geometry=(COEFFICIENT_BACKEND == "gmp"
                                            and (dimension + 1)*bound.bit_length() <= 127
                                            and dimension*(bound.bit_length() + 1) <= 128))
                                    self.assertEqual(packed, (dimension + 1)*bound.bit_length() <= 127)

    def test_signed_ratio_127_bit_boundary_without_guarded_geometry(self):
        from gvwzz import Options
        from gvwzz.core import Label, Signature
        from gvwzz.orders import ModuleOrder
        from gvwzz.polynomial import IntegerPolynomialRing
        from gvwzz.solver import GVWInteger
        self.native_api()
        huge = (1 << 4096) + 31
        for dimension in (126, 127):
            backend = IntegerPolynomialRing([f"x{i}" for i in range(dimension)], "grevlex")
            zero = (0,)*dimension
            first = (1,) + (0,)*(dimension - 1)
            last = (0,)*(dimension - 1) + (1,)
            inputs = [backend.one]*3
            for negative in (False, True):
                engine = GVWInteger(backend, inputs,
                                    ModuleOrder(backend, inputs, "g2", positions=[2, 0, 1]),
                                    Options())
                for index, monomial in enumerate((first, last)):
                    # Keep max_signature + max_leading == 1. Combining
                    # nonzero signatures and leading monomials would instead
                    # need two-bit lanes and miss the exact 127/128 boundary.
                    signature = Signature((-2, 3)[index], zero if negative else monomial, index)
                    polynomial = backend.ring.from_dict({monomial if negative else zero:
                                                        (2, -3)[index]})
                    engine._insert_basis(Label(signature, polynomial))
                for monomial in (first, last):
                    for coefficient in (6, -7, huge, -huge):
                        signature = Signature(coefficient, zero if negative else monomial, 2)
                        leading = (monomial if negative else zero, coefficient)
                        for mode, euclidean in (("strict", False), ("mixed", True)):
                            packed = self.compare_query(engine, signature, leading,
                                1 if negative else 0, mode, euclidean, expected_geometry=False)
                            self.assertEqual(packed, dimension == 126)

    def test_geometry_cache_eviction_then_basis_append(self):
        from gvwzz.core import Label, Signature
        context_type, native = self.native_api()
        engine, _ = self.engine(1, "newest")
        signature = Signature(1, (200000, 0), 1)
        bound = 70000
        context = context_type(engine, signature, bound)
        self.compare_query(engine, signature, ((5, 0), 6), bound, "mixed", True)
        # Cross the native geometry entry limit with cheap scalar queries.
        # Inspect independent oracle answers before/after eviction and after
        # appending a new highest-ranked unit reducer.
        for exponent in range(65545):
            native(engine, signature, ((exponent, 0), 6), "mixed", True, False, context)
        if context.native_geometry_enabled:
            self.assert_bounded_geometry(context, evicted=True)
        for appended in (False, True):
            if appended:
                p = engine.backend.ring.from_dict({(1, 0): 1, (0, 0): -17})
                engine._insert_basis(Label(Signature(1, (0, 0), 2), p))
            for exponent in (0, 1, 3, 5, 4095, 4096, 8191, 8192, 65535, 65536, 65544):
                for coefficient in (6, -7, (1 << 200) + 3):
                    self.compare_query(engine, signature, ((exponent, 0), coefficient),
                                       bound, "mixed", True)
            if context.native_geometry_enabled:
                self.assert_bounded_geometry(context, evicted=True)

    def assert_bounded_geometry(self, context, *, evicted=False):
        entries, accounted_bytes, entry_limit, byte_limit, evictions = context.geometry_cache_info
        self.assertGreater(entries, 0)
        self.assertLessEqual(entries, entry_limit)
        self.assertGreater(accounted_bytes, 0)
        self.assertLessEqual(accounted_bytes, byte_limit)
        if evicted:
            self.assertGreater(evictions, 0)

    def test_wide_geometry_eviction_append_and_policy_changes_match_oracle(self):
        from dataclasses import replace
        from gvwzz import Options
        from gvwzz.core import Label, Signature
        from tests.native_wide_reduction_checks import lifted_fixture
        context_type, native = self.native_api()
        from gvwzz._native_reduction import COEFFICIENT_BACKEND
        if COEFFICIENT_BACKEND != "gmp":
            self.skipTest("native geometry cache requires GMP")
        engine, label = lifted_fixture(Options(reducer_selection="newest"))
        signature, bound = label.signature, 31
        context = context_type(engine, signature, bound)
        self.assertTrue(context.native_geometry_enabled)
        active = (0, 1, 2, 3, 4, 5, 7, 10, 11)

        def exponent(value):
            # Nine base-four digits yield distinct 12D monomials of degree
            # at most 25 here, including coordinates above the limb split.
            monomial = [0]*12
            for digit, coordinate in enumerate(active):
                monomial[coordinate] = (value >> (2*digit)) & 3
            return tuple(monomial)

        for value in range(65545):
            native(engine, signature, (exponent(value), 6), "mixed", True, False, context)
        self.assert_bounded_geometry(context, evicted=True)
        huge = (1 << 4096) + 23
        selected = (0, 1, 5, 255, 4095, 8192, 32768, 65535, 65536, 65544)
        for appended in (False, True):
            if appended:
                x, y = engine.backend.gens[0], engine.backend.gens[-1]
                common = (0,)*6 + (8,) + (0,)*5
                engine._insert_basis(Label(Signature(-2, common, 0), x + 5*y - 17))
            for policy in ("regular-first", "singular-first", "first"):
                for priority in ("signature-first", "exact-first"):
                    engine.options = replace(engine.options, reducer_policy=policy,
                                             euclidean_priority=priority)
                    for value in selected:
                        for coefficient in (6, -7, huge, -huge):
                            current = Signature(coefficient, signature.monomial, signature.index)
                            for mode, euclidean in (("strict", False), ("mixed", True)):
                                self.compare_query(engine, current, (exponent(value), coefficient),
                                                   bound, mode, euclidean, expected_geometry=True)
            self.assert_bounded_geometry(context, evicted=True)

    def test_geometry_byte_budget_evicts_before_entry_limit(self):
        from gvwzz import Options
        from gvwzz.core import Label, Signature
        from gvwzz.orders import ModuleOrder
        from gvwzz.polynomial import IntegerPolynomialRing
        from gvwzz.solver import GVWInteger
        context_type, native = self.native_api()
        from gvwzz._native_reduction import COEFFICIENT_BACKEND
        if COEFFICIENT_BACKEND != "gmp":
            self.skipTest("native geometry cache requires GMP")
        backend = IntegerPolynomialRing(["x", "y"], "grevlex")
        inputs = [backend.one]*3
        engine = GVWInteger(backend, inputs, ModuleOrder(backend, inputs, "g2"),
                            Options(reducer_selection="newest"))
        # Every leading monomial divides every query. Each entry therefore
        # owns at least 1000 reducer IDs; the byte limit is reached long
        # before the entry limit without a large polynomial computation.
        for index in range(1000):
            coefficient = (index % 13) + 2
            polynomial = backend.ring.from_dict({(0, 0): coefficient})
            engine._insert_basis(Label(Signature(1, (0, 0), index % 3), polynomial))
        signature, bound = Signature(7, (10000, 0), 2), 5000
        context = context_type(engine, signature, bound)
        for exponent in range(4200):
            native(engine, signature, ((exponent, 0), 6), "mixed", True, False, context)
        self.assert_bounded_geometry(context, evicted=True)
        entries, _, entry_limit, _, _ = context.geometry_cache_info
        self.assertLess(entries, entry_limit)
        for appended in (False, True):
            if appended:
                engine._insert_basis(Label(Signature(-3, (0, 0), 0), -backend.one))
            for exponent in (0, 1, 2048, 4199):
                for coefficient in (6, -7, (1 << 4096) + 31):
                    self.compare_query(engine, signature, ((exponent, 0), coefficient),
                                       bound, "mixed", True, expected_geometry=True)
            self.assert_bounded_geometry(context, evicted=True)


def katsura4() -> dict:
    return dict(variables=["u0", "u1", "u2", "u3"], polynomials=[
        "u0+2*u1+2*u2+2*u3-1",
        "u0^2+2*u1^2+2*u2^2+2*u3^2-u0",
        "2*u0*u1+2*u1*u2+2*u2*u3-u1",
        "u1^2+2*u0*u2+2*u1*u3-u2",
    ])


def make_cases() -> list[dict]:
    cases = []
    signed = ["-4*x^2+2*y+3", "6*x*y-5", "9*y^2+x"]
    singular = ["-2*x*y-3-6*y^2", "y+4*x-6", "3*x^2+4*y-2*x*y"]
    # Coefficients are deliberately much larger than any machine integer.
    huge = (1 << 2048) + 15
    big = [f"({-huge})*(4*x^2-2*y-3)",
           f"({huge})*(6*x*y-5)", f"({huge})*(9*y^2+x)"]
    for module in ("g2", "pot"):
        for number, order in enumerate(("grevlex", "grlex", "lex")):
            profiles = [
                ("default", signed, {}),
                ("singular-path", singular,
                 dict(reducer_policy="singular-first", euclidean_priority="exact-first",
                      record_paths=True, queue_tie="fifo",
                      reducer_selection=("oldest", "unit", "height")[number])),
                ("capped-vector", signed,
                 dict(reducer_policy="first", euclidean=False,
                      track_representations=True, tail_step_cap=2,
                      reduction_kernel="heap" if order == "lex" else "packed",
                      reducer_selection=("balanced", "newest", "sparse")[number])),
                ("bigint", big, {}),
            ]
            for suffix, polynomials, options in profiles:
                cases.append(dict(name=f"{module}-{order}-{suffix}",
                    variables=["x", "y"], polynomials=polynomials,
                    module_order=module, polynomial_order=order,
                    options=options, expected_status="completed"))
    for module, switches in (("g2", {}), ("pot", dict(
            record_paths=True, tail_policy="selective", g_tail_reduce=False))):
        cases.append(dict(name=f"katsura4-{module}", **katsura4(),
            module_order=module, polynomial_order="grevlex", options=switches,
            expected_status="completed"))
    for name, switches in (
            ("one-step", dict(max_reduction_steps=1)),
            ("seventeen-steps", dict(max_reduction_steps=17)),
            ("candidates", dict(max_candidates=20)),
            ("basis", dict(max_basis=5)),
            ("terms", dict(max_polynomial_terms=5)),
            ("coefficient-height", dict(max_coefficient_bits=5))):
        cases.append(dict(name=f"limit-{name}", **katsura4(),
            module_order="g2", polynomial_order="grevlex", options=switches,
            expected_status="limit"))
    return cases


def run_worker(source: str, mode: str) -> None:
    sys.path.insert(0, str(Path(source).resolve()))
    from gvwzz import Options, compute, verify_generation_paths
    from gvwzz.native_status import native_status
    from gvwzz.verification import verify_result

    from sympy.external.gmpy import GROUND_TYPES
    status = native_status()
    native = status["native_accelerators"]
    if mode == "native" and not all(native.values()):
        raise AssertionError(f"native reference/candidate is not fully enabled: {native}")
    if mode == "python" and any(native.values()):
        raise AssertionError(f"fallback process loaded an extension: {native}")
    case = json.load(sys.stdin)
    switches = dict(max_seconds=15, max_basis=180, max_candidates=25000,
                    max_reduction_steps=100000)
    switches.update(case["options"])
    options = Options(**switches)
    result = compute(case["polynomials"], case["variables"],
        polynomial_order=case["polynomial_order"], module_order=case["module_order"],
        options=options, show_basis=False, show_u=False)
    if result.status != case["expected_status"]:
        raise AssertionError(f'{case["name"]}: unexpected {result.status}: {result.message}')
    if result.status == "limit" and "max_seconds" in result.message:
        raise AssertionError("time-limited run must not pass")
    verification = None
    if result.status == "completed":
        if options.record_paths:
            report = verify_generation_paths(result, strong=True, timeout=15,
                                             max_reduction_steps=100000)
        else:
            report = verify_result(result, strong=True, mode="static", timeout=15,
                max_reduction_steps=100000,
                require_certificates=options.track_representations)
        if not report["passed"]:
            raise AssertionError(f'{case["name"]}: independent verification failed: {report}')
        if (options.record_paths or options.track_representations) and not report["signature_basis_certified"]:
            raise AssertionError(f'{case["name"]}: certificate not verified: {report}')
        verification = {key: report[key] for key in
            ("passed", "membership_verified", "signature_basis_certified")}
    raw = result.to_dict()
    answer = {key: raw[key] for key in RESULT_FIELDS if key in raw}
    answer["logical_counters"] = {key: raw["counters"].get(key, 0) for key in COUNTERS}
    answer["independent_verification"] = verification
    output = dict(answer=answer, execution=dict(native=native,
        coefficient_backend=status.get("native_reduction_coefficient_backend", "python"),
        sympy_ground_type=GROUND_TYPES,
        native_reduction_workspaces=raw["counters"].get("native_reduction_workspaces", 0)))
    json.dump(output, sys.stdout, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")


def evaluate(source: Path, case: dict, mode: str, timeout: float) -> dict:
    env = dict(os.environ, PYTHONHASHSEED="0")
    env.pop("GVWZZ_NO_NATIVE", None)
    if mode == "python":
        env["GVWZZ_NO_NATIVE"] = "1"
    process = subprocess.run([sys.executable, str(Path(__file__).resolve()),
        "--worker", str(source), "--worker-mode", mode], input=json.dumps(case),
        text=True, capture_output=True, timeout=timeout, cwd=source, env=env,
        check=False)
    if process.returncode:
        raise RuntimeError(f'{case["name"]}: {source.name}/{mode} failed\n{process.stderr}')
    return json.loads(process.stdout)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    here = ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path,
                        help="Optional separately built reference source tree.")
    parser.add_argument("--candidate", type=Path, default=here)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--snapshot", type=Path)
    group.add_argument("--compare-snapshot", type=Path)
    parser.add_argument("--case", action="append")
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--worker-mode", choices=("native", "python"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_worker(args.worker, args.worker_mode)
        return
    if args.snapshot and args.reference is None:
        parser.error("--snapshot requires an explicit --reference source tree")
    if not any((args.reference, args.snapshot, args.compare_snapshot)):
        args.compare_snapshot = here / "tests" / "fixtures" / "reduction_reference.json"
    if args.compare_snapshot and not args.compare_snapshot.is_file():
        parser.error(f"Reference snapshot not found: {args.compare_snapshot}")
    sources = [] if args.snapshot else [args.candidate]
    if not args.compare_snapshot:
        sources.append(args.reference)
    for source in sources:
        if not (source / "gvwzz" / "solver.py").is_file():
            parser.error(f"No gvwzz source tree at {source}")
    cases = make_cases()
    if args.case:
        unknown = set(args.case) - {case["name"] for case in cases}
        if unknown:
            parser.error(f"unknown case names: {sorted(unknown)}")
        cases = [case for case in cases if case["name"] in args.case]
    records = {}
    if args.compare_snapshot:
        saved = json.loads(args.compare_snapshot.read_text())
        records = saved["records"]
    native_workspaces = 0
    execution_backends = set()
    answers = []
    for case in cases:
        name = case["name"]
        if args.compare_snapshot:
            reference = records[name]
            if reference["case"] != case:
                raise AssertionError(f"{name}: snapshot input/options differ")
        else:
            output = evaluate(args.reference.resolve(), case, "native", args.timeout)
            reference = records[name] = dict(case=case, **output)
        if args.snapshot:
            print(f'REFERENCE {name}: {reference["answer"]["status"]} '
                  f'P={reference["answer"]["basis_size"]}', flush=True)
        else:
            for mode in ("native", "python"):
                actual = evaluate(args.candidate.resolve(), case, mode, args.timeout)
                difference = first_difference(reference["answer"], actual["answer"])
                if difference:
                    raise AssertionError(f"{name}/{mode}: {difference}")
                native_workspaces += actual["execution"]["native_reduction_workspaces"]
                execution_backends.add((mode, actual["execution"]["coefficient_backend"],
                                        actual["execution"]["sympy_ground_type"]))
            print(f'PASS {name}: {reference["answer"]["status"]} '
                  f'P={reference["answer"]["basis_size"]} (native + Python)', flush=True)
        answers.append([case, reference["answer"]])
    counters = {key: sum(answer[1]["logical_counters"][key] for answer in answers)
                for key in ("regular_steps", "singular_steps", "euclidean_steps",
                            "tail_steps", "tail_cap_reached")}
    max_bits = max(answer[1]["logical_counters"]["max_coefficient_bits"] for answer in answers)
    if not args.case:
        assert all(counters.values()), counters
        assert max_bits > 2048, max_bits
        if not args.snapshot:
            assert native_workspaces > 0, "no native reduction workspace exercised"
    summary = dict(cases=len(cases), sha256=digest(answers), coverage=counters,
                   max_coefficient_bits=max_bits, native_workspaces=native_workspaces,
                   execution_backends=sorted(execution_backends))
    if args.snapshot:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        args.snapshot.write_text(json.dumps(dict(schema=1, reference=str(args.reference.resolve()),
            summary=summary, records=records), indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
