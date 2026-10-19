"""Independent exact oracles for non-squarefree frozen multiplier pruning.

The oracle uses only unbounded integer tuple arithmetic and coordinatewise
comparison. It deliberately shares neither a packed representation nor a
divisor index with the implementation under review.
"""
from itertools import product
from random import Random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gvwzz import Options, compute, verify_generation_paths
from gvwzz.core import Label, Signature
from gvwzz.frozen import FrozenSeedIndex
from gvwzz.incremental import _POTStage
from gvwzz.orders import ModuleOrder
from gvwzz.polynomial import IntegerPolynomialRing
from gvwzz.solver import GVWInteger
from gvwzz.verification import verify_result


def oracle(rows, current, selected):
    selected = sorted(set(selected))
    shifts = {i: tuple(max(a - b, 0) for a, b in zip(rows[i], current))
              for i in selected}
    return tuple(i for i in selected if not any(
        all(a <= b for a, b in zip(shifts[j], shifts[i]))
        and (shifts[j] != shifts[i] or j < i)
        for j in selected if i != j))


def records(result):
    return [(p.signature.coefficient, p.signature.monomial, p.signature.index,
             tuple(sorted(p.polynomial.items()))) for p in result.basis]


class GeneralAntichainReviewChecks(unittest.TestCase):
    def check_rows(self, rows, current, selected):
        index = FrozenSeedIndex(len(current), rows)
        expected = oracle(rows, current, selected)
        self.assertEqual(index.minimal_parents(current, selected), expected)
        if hasattr(index, 'minimal_parents_mask'):
            mask = sum(1 << i for i in set(selected))
            self.assertEqual(tuple(index.minimal_parents_mask(current, mask)), expected)
        return expected

    def test_exhaustive_small_general_exponents(self):
        rows = list(product(range(4), repeat=2))
        # Restrict the selection so a zero shift is not always available.
        selections = [list(range(k, len(rows), 3)) for k in range(3)]
        for current in product(range(6), repeat=2):
            for selected in selections:
                self.check_rows(rows, current, selected)

    def test_arbitrary_precision_and_lane_boundaries(self):
        rng = Random(220054)
        boundaries = (0, 1, 2, 3, 7, 8, 31, 32, 63, 64, 127, 128,
                      255, 256, 65535, 65536, 2**32, 2**64, 10**80)
        for nvars in (1, 2, 3, 8, 65):
            for _ in range(18):
                rows = [tuple(rng.choice(boundaries) if rng.randrange(4) == 0
                              else 0 for _ in range(nvars)) for _ in range(36)]
                rows += rows[3:7]
                current = tuple(rng.choice(boundaries + (10**90,))
                                for _ in range(nvars))
                selected = rng.sample(range(len(rows)), 29)
                selected += selected[:4]
                self.check_rows(rows, current, selected)

    def test_empty_zero_and_earliest_selected_row(self):
        self.check_rows([], (), [])
        self.check_rows([(), (), ()], (), [2, 1, 2])
        self.check_rows([(2, 0), (2, 0), (2, 0), (3, 0)], (1, 0), [3, 2, 1, 2])
        self.check_rows([(2, 0), (0, 2)], (10**100, 10**100), [1, 0])
        self.check_rows([(3, 0), (0, 3)], (1, 1), [])

    def test_support_does_not_imply_general_monomial_divisibility(self):
        rows = [(4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (4, 1)]
        self.assertEqual(self.check_rows(rows, (1, 0), range(6)), (0, 1, 2, 3))

    def test_survivor_mask_then_minimal_selection(self):
        rng = Random(220055)
        for _ in range(80):
            rows = [tuple(rng.randrange(8) for _ in range(5)) for _ in range(40)]
            current = tuple(rng.randrange(8) for _ in range(5))
            signature = tuple(rng.randrange(8) for _ in range(5))
            selected = [i for i, row in enumerate(rows)
                        if any(min(a, b) > s for a, b, s
                               in zip(row, current, signature))]
            index = FrozenSeedIndex(5, rows)
            mask = index.survivor_mask(current, signature)
            self.assertEqual(mask, sum(1 << i for i in selected))
            self.check_rows(rows, current, selected)

    def test_native_list_and_mask_match_independent_oracle(self):
        try:
            from gvwzz._native_pairs import minimal_frozen_parents, minimal_frozen_mask
        except ImportError:
            self.skipTest('Optional native pair accelerator is not built')
        rng = Random(220056)
        for nvars in (0, 1, 3, 48, 64, 65, 129):
            for bound in (2, 63, 64, 255, 256, 2**64, 10**80):
                rows = [tuple(rng.randrange(bound + 1) if rng.randrange(5) == 0
                              else 0 for _ in range(nvars)) for _ in range(32)]
                rows += rows[:3]
                current = tuple(rng.randrange(bound * 2 + 1) for _ in range(nvars))
                selected = rng.sample(range(len(rows)), 27)
                expected = oracle(rows, current, selected)
                engine = SimpleNamespace(backend=SimpleNamespace(names=range(nvars)),
                    frozen_count=len(rows), leading=[(a, 1) for a in rows] + [(current, 1)],
                    _native_antichain=None)
                result = minimal_frozen_parents(engine, len(rows), selected)
                if result is not None:
                    self.assertEqual(tuple(result), expected, (nvars, bound, 'rows'))
                mask_result = minimal_frozen_mask(engine, len(rows), sum(1 << i for i in selected))
                self.assertEqual(result is None, mask_result is None, (nvars, bound))
                if mask_result is not None:
                    self.assertEqual(tuple(mask_result), expected, (nvars, bound, 'mask'))


class GeneralCurrentCoverReviewChecks(unittest.TestCase):
    def test_cache_requires_exponent_divisibility_and_commit(self):
        try:
            from gvwzz._native_pairs import (minimal_frozen_mask, commit_frozen_parents,
                                             emit_unit_current_pairs)
        except ImportError:
            self.skipTest('Optional native pair accelerator is not built')
        used_general_native = False
        # y^2 does not divide y, despite identical nonzero-coordinate support.
        for exponent, committed, upper in product((1, 2, 4), (False, True), (1, 2)):
            backend = IntegerPolynomialRing(('x', 'y'), 'grevlex')
            inputs = [backend.parse(f) for f in ('x^2*y^2', 'x^2', f'x^2*y^{exponent}')]
            if upper == 2:
                inputs[1], inputs[2] = inputs[2], inputs[1]
            order = ModuleOrder(backend, inputs, 'pot')
            native, original = (GVWInteger(backend, inputs, order, Options()) for _ in range(2))
            for engine in (native, original):
                engine.frozen_count = 1
                engine._skip_principal_pair = lambda i, j: i == 0 or j == 0
                for i, polynomial in enumerate(inputs):
                    engine._insert_basis(Label(Signature(1, (0, 0), min(i, 1)), polynomial))
                engine.S.insert_and_complete(Signature(1, (2, 2), 1))
                engine._emit_pair(0, upper)
            chosen = minimal_frozen_mask(native, upper, 1)
            if chosen is None:
                continue
            used_general_native = True
            self.assertEqual(chosen, [0])
            if committed:
                commit_frozen_parents(native, upper)
            emit_unit_current_pairs(native, 2, 1)
            original._emit_pair(1, 2)
            expected_rejection = int(committed and exponent >= 2)
            self.assertEqual(native.stats.counters['j_current_dominance_rejected'],
                             expected_rejection, (exponent, committed, upper))
            self.assertEqual(native.S.all_signatures(), original.S.all_signatures())
            native._process_queue()
            original._process_queue()
            self.assertEqual(native.P, original.P)
            self.assertEqual(native.S.all_signatures(), original.S.all_signatures())
        if not used_general_native:
            self.skipTest('Native antichain intentionally falls back on general exponents')

    def test_non_squarefree_complete_system_and_certificate_modes(self):
        cases = [
            (['x^2*z^2-w^2', 'x^2*y^2*z^2-y^2',
              'y^2*z^2*w^2-x^2', 'y^2*z^2*w^2-w^2'], ('x', 'y', 'z', 'w')),
            (['-x^3*y^2+z^2', 'x^3*z^2-y^2', '-y^2*z^2+x^3'], ('x', 'y', 'z')),
        ]
        for polynomials, variables in cases:
            for options in ({}, {'record_paths': True}, {'track_representations': True},
                            {'j_generation_filter': False}, {'max_candidates': 5}):
                opt = Options(incremental_pot=True, max_seconds=20, **options)
                actual = compute(polynomials, variables, 'grevlex', options=opt,
                                 show_basis=False, show_u=False)
                with patch.object(_POTStage, '_emit_pairs', GVWInteger._emit_pairs):
                    original = compute(polynomials, variables, 'grevlex', options=opt,
                                       show_basis=False, show_u=False)
                self.assertEqual(actual.status, original.status, options)
                self.assertEqual(records(actual), records(original), options)
                self.assertEqual(actual.to_dict()['syzygy_signatures'],
                                 original.to_dict()['syzygy_signatures'])
                if options.get('max_candidates'):
                    self.assertEqual(actual.stats.counters['j_dominance_rejected'], 0)
                else:
                    self.assertEqual(actual.status, 'completed')
                if options.get('record_paths'):
                    self.assertTrue(verify_generation_paths(actual, timeout=0,
                        max_reduction_steps=0)['passed'])
                if options.get('track_representations'):
                    self.assertTrue(verify_result(actual, timeout=0, mode='static',
                        require_certificates=True, max_reduction_steps=0)['passed'])


if __name__ == '__main__':
    unittest.main()
