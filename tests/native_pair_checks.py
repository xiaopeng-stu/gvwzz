"""Differential checks for frozen-seed batch omission and optional native rows."""
import unittest
import random
from itertools import permutations
from unittest.mock import patch

from gvwzz import Options, compute, verify_generation_paths
from gvwzz.incremental import _POTStage
from gvwzz.solver import GVWInteger
from gvwzz.verification import verify_result
from gvwzz.core import Label, Signature
from gvwzz.orders import ModuleOrder
from gvwzz.polynomial import IntegerPolynomialRing


def records(result):
    return [(p.signature.coefficient, p.signature.monomial, p.signature.index,
             tuple(sorted(p.polynomial.items()))) for p in result.basis]


class NativePairChecks(unittest.TestCase):
    def compare(self, polynomials, *, variables=('x', 'y', 'z'), **options):
        opt = Options(incremental_pot=True, max_seconds=10, **options)
        args = (polynomials, variables, 'grevlex')
        actual = compute(*args, options=opt, show_basis=False, show_u=False)
        with patch.object(_POTStage, '_emit_pairs', GVWInteger._emit_pairs):
            reference = compute(*args, options=opt, show_basis=False, show_u=False)
        self.assertEqual(actual.status, reference.status)
        self.assertEqual(records(actual), records(reference))
        # All logical pairs and algebraic work remain identical. Only the
        # permanent-covered J candidates' rejection point may have moved.
        for key in ('parent_pairs', 'j_generated', 'g_generated', 'g_discarded',
                    'j_inserted', 'g_inserted', 'zero_reductions',
                    'regular_steps', 'singular_steps', 'euclidean_steps',
                    'tail_steps', 'j_materialized'):
            self.assertEqual(actual.stats.counters[key], reference.stats.counters[key], key)
        return actual

    def test_unit_signed_and_general_coefficients(self):
        cases = [
            ['x*y-z', 'x*z-y', 'y*z-x'],
            ['-x*y+z', '-x*z+y', '-y*z+x'],
            ['2*x', 'y+x', 'x*y+1'],
            ['x', '2*y+x', 'y*z+1'],
            ['0', 'x*y-z', 'x*z-y', 'y*z-x'],
        ]
        for case in cases:
            with self.subTest(case=case):
                self.assertEqual(self.compare(case).status, 'completed')

    def test_arbitrary_exponent_fallback(self):
        result = self.compare(['x^70-y', 'x^69*z-1', 'z-y'])
        self.assertEqual(result.status, 'completed')

    def test_option_paths(self):
        case = ['x*y-z', 'x*z-y', 'y*z-x']
        for options in ({'lazy_candidates': False}, {'j_generation_filter': False},
                        {'j_deduplicate': False}, {'cover_mode': 'naive'},
                        {'queue_tie': 'fifo'}, {'queue_tie': 'small-lm'}):
            with self.subTest(options=options):
                self.compare(case, **options)

    def test_candidate_budgets_keep_original_cutoff(self):
        case = ['x*y-z', 'x*z-y', 'y*z-x']
        for budget in (1, 4, 10):
            with self.subTest(budget=budget):
                result = self.compare(case, max_candidates=budget)
                self.assertEqual(result.stats.counters['j_frozen_seed_rejected'], 0)
                self.assertEqual(result.stats.counters['j_dominance_rejected'], 0)

    def test_active_antichain_branch_against_original_parent_rows(self):
        for case in permutations(['x*y-z', 'x*z-y', 'y*z-x']):
            for signs in (0, 5):
                polys = ['-(' + f + ')' if signs & (1 << i) else f
                         for i, f in enumerate(case)]
                with self.subTest(polynomials=polys):
                    result = self.compare(polys)
                    self.assertEqual(result.status, 'completed')
                    self.assertGreater(result.stats.counters['j_dominance_rejected'], 0)

    def test_generation_paths(self):
        result = self.compare(['x*y-z', 'x*z-y', 'y*z-x'], record_paths=True)
        self.assertGreater(result.stats.counters['j_dominance_rejected'], 0)
        self.assertTrue(verify_generation_paths(result, timeout=0,
                                               max_reduction_steps=0)['passed'])

    def test_explicit_representations(self):
        result = self.compare(['x*y-z', 'x*z-y', 'y*z-x'], track_representations=True)
        self.assertGreater(result.stats.counters['j_dominance_rejected'], 0)
        self.assertTrue(verify_result(result, timeout=0, mode='static',
                                     require_certificates=True,
                                     max_reduction_steps=0)['passed'])

    def test_native_unit_rows_match_original_all_module_orders(self):
        try:
            from gvwzz._native_pairs import emit_unit_current_pairs
        except ImportError:
            self.skipTest('optional native emitter unavailable')
        rng = random.Random(210031)
        semantic = ('parent_pairs', 'j_generated', 'g_generated', 'g_discarded',
                    'g_omitted_comparable', 'j_enqueued', 'g_enqueued',
                    'j_generation_tests', 'j_generation_syzygy_rejected',
                    'j_dedup_discarded', 'j_dedup_replaced', 'j_dedup_settled',
                    'principal_syzygies_detected', 'principal_syzygies_unknown',
                    'syzygy_seed_rejected', 'syzygy_insert_calls')
        for dim in (2, 48):
            backend = IntegerPolynomialRing(['x' + str(k) for k in range(dim)], 'grevlex')
            for module in ('pot', 'top', 'g1', 'g2', 'schreyer', 'block'):
                inputs, labels = [], []
                for i in range(7):
                    lm = tuple(rng.randrange(4) if k < 2 else 0 for k in range(dim))
                    # Exercise an exact unpacked-parent fallback as well.
                    if i == 2:
                        lm = (70,) + lm[1:]
                    sig = tuple(rng.randrange(4) if k < 2 else 0 for k in range(dim))
                    lc = 2 if i == 3 else rng.choice((-1, 1))
                    polynomial = backend.ring.from_dict({lm: lc})
                    inputs.append(polynomial)
                    labels.append(Label(Signature(rng.choice((-3, 1, 2**200)), sig, i), polynomial))
                order = ModuleOrder(backend, inputs, module, positions=tuple(reversed(range(7))),
                                    block_sizes=(3, 4) if module == 'block' else None)
                native, original = (GVWInteger(backend, inputs, order, Options()) for _ in range(2))
                for label in labels:
                    native._insert_basis(label)
                    original._insert_basis(label)
                for j in range(1, len(labels)):
                    emit_unit_current_pairs(native, j, 0)
                    for i in range(j):
                        original._emit_pair(i, j)
                with self.subTest(dim=dim, module=module):
                    self.assertEqual(native.S.all_signatures(), original.S.all_signatures())
                    self.assertEqual(native.queue.heap, original.queue.heap)
                    for key in semantic:
                        self.assertEqual(native.stats.counters[key], original.stats.counters[key], key)
                    while native.queue:
                        a, b = native.queue.pop(), original.queue.pop()
                        self.assertEqual(a.label, b.label)

    def test_current_dominance_active_real_system(self):
        import gvwzz.incremental as incremental
        if incremental._native_emit_unit_current_pairs is None:
            self.skipTest('optional current-row native emitter unavailable')
        result = self.compare(['x*z-w', 'x*y*z-y', 'y*z*w-x', 'y*z*w-w'],
                              variables=('x', 'y', 'z', 'w'))
        self.assertEqual(result.status, 'completed')
        self.assertGreater(result.stats.counters['j_current_dominance_rejected'], 0)

    def test_current_dominance_both_upper_parent_branches(self):
        try:
            from gvwzz._native_pairs import (minimal_frozen_mask, commit_frozen_parents,
                                             emit_unit_current_pairs)
        except ImportError:
            self.skipTest('optional current-row native emitter unavailable')
        for upper in (1, 2):
            with self.subTest(upper_parent=upper):
                backend = IntegerPolynomialRing(('x', 'y'), 'grevlex')
                inputs = [backend.parse(f) for f in ('x*y', 'x', 'x*y^2')]
                if upper == 2:
                    inputs[1], inputs[2] = inputs[2], inputs[1]
                order = ModuleOrder(backend, inputs, 'pot')
                native, original = (GVWInteger(backend, inputs, order, Options()) for _ in range(2))
                for engine in (native, original):
                    engine.frozen_count = 1
                    engine._skip_principal_pair = lambda i, j: i == 0 or j == 0
                    for i, polynomial in enumerate(inputs):
                        engine._insert_basis(Label(Signature(1, (0, 0), min(i, 1)), polynomial))
                    engine.S.insert_and_complete(Signature(1, (1, 1), 1))
                    engine._emit_pair(0, upper)
                self.assertEqual(minimal_frozen_mask(native, upper, 1), [0])
                commit_frozen_parents(native, upper)
                emit_unit_current_pairs(native, 2, 1)
                original._emit_pair(1, 2)
                self.assertEqual(native.stats.counters['j_current_dominance_rejected'], 1)
                self.assertEqual(native.S.all_signatures(), original.S.all_signatures())
                for key in ('parent_pairs', 'j_generated', 'g_generated',
                            'principal_syzygies_detected', 'syzygy_insert_calls'):
                    self.assertEqual(native.stats.counters[key], original.stats.counters[key], key)
                native._process_queue()
                original._process_queue()
                self.assertEqual(native.P, original.P)
                self.assertEqual(native.S.all_signatures(), original.S.all_signatures())


if __name__ == '__main__':
    unittest.main()
