"""Regression checks for exact syzygy shortcuts and incremental divisor indices."""
from collections import OrderedDict
from contextlib import redirect_stderr
from io import StringIO
from itertools import combinations, product
from math import gcd
import random
import unittest
from unittest.mock import patch

from gvwzz import Options, compute, GVWInteger, reconstruct_syzygies, verify_generation_paths
from gvwzz.core import Signature
from gvwzz.orders import ModuleOrder
from gvwzz.polynomial import IntegerPolynomialRing
from gvwzz.structures import DivisorIndex, Stats, SyzygyStore
from gvwzz.verification import verify_result


def divides(a, b):
    return (a.index == b.index and b.coefficient % a.coefficient == 0
            and all(x <= y for x, y in zip(a.monomial, b.monomial)))


def naive_closure(values):
    """Independent repeated closure/minimization, used on small test inputs."""
    values = {Signature(abs(s.coefficient), s.monomial, s.index) for s in values}
    while True:
        new = set(values)
        for a, b in combinations(values, 2):
            if a.index == b.index:
                new.add(Signature(gcd(a.coefficient, b.coefficient),
                                  tuple(max(x, y) for x, y in zip(a.monomial, b.monomial)),
                                  a.index))
        new = {s for s in new if not any(t != s and divides(t, s) for t in new)}
        if new == values:
            return new
        values = new


def S(c, m, i=0):
    return Signature(c, tuple(m), i)


def engine(**settings):
    ring = IntegerPolynomialRing(['x'], 'lex')
    inputs = [ring.parse('2*x'), ring.parse('3*x')]
    return GVWInteger(ring, inputs, ModuleOrder(ring, inputs, 'pot'), Options(**settings))


class SyzygyChecks(unittest.TestCase):
    def store(self, n=2, k=2, debug=True):
        return SyzygyStore(n, k, Stats(), debug)

    def test_01_unit_component_avoids_all_g_allocations(self):
        st = self.store()
        st.insert_and_complete(S(1, (2, 0)))
        with patch('gvwzz.structures.gcd_signature', side_effect=RuntimeError('allocated')):
            st.debug = False
            st.insert_and_complete(S(1, (0, 2)))
        self.assertEqual(st.stats.counters['syzygy_g_shortcuts'], 1)
        self.assertEqual(st.stats.counters['syzygy_g_materialized'], 0)

    def test_02_unit_old_and_nonunit_seed(self):
        st = self.store()
        st.insert_and_complete(S(1, (2, 0)))
        st.insert_and_complete(S(-12, (0, 2)))
        self.assertEqual(st.all_signatures(), [S(1, (2, 0)), S(12, (0, 2))])
        self.assertEqual(st.stats.counters['syzygy_g_skipped_old'], 1)

    def test_03_old_coefficient_divides_seed(self):
        st = self.store(debug=False)
        st.insert_and_complete(S(6, (1, 0)))
        with patch.object(st, 'has_divisor', wraps=st.has_divisor) as check:
            with patch('gvwzz.structures.gcd_signature', side_effect=RuntimeError('allocated')):
                st.insert_and_complete(S(18, (0, 1)))
            self.assertEqual(check.call_count, 1)  # The seed query only.
        self.assertEqual(st.stats.counters['syzygy_g_skipped_old'], 1)

    def test_04_seed_divides_old_coefficient(self):
        st = self.store()
        st.insert_and_complete(S(6, (0, 1)))
        st.insert_and_complete(S(2, (1, 0)))
        self.assertEqual(st.stats.counters['syzygy_g_skipped_seed'], 1)
        self.assertEqual(st.stats.counters['syzygy_g_shortcuts'], 1)

    def test_05_old_first_classification_even_when_seed_divides(self):
        st = self.store()
        st.insert_and_complete(S(4, (0, 1)))
        st.insert_and_complete(S(2, (1, 1)))
        before = st.stats.counters.copy()
        st.insert_and_complete(S(2, (1, 0)))
        self.assertEqual(st.stats.counters['syzygy_g_skipped_old'] - before['syzygy_g_skipped_old'], 2)
        self.assertEqual(st.stats.counters['syzygy_g_skipped_seed'], before['syzygy_g_skipped_seed'])
        self.assertEqual(st.all_signatures(), [S(4, (0, 1)), S(2, (1, 0))])

    def test_06_incomparable_coefficients_really_complete(self):
        st = self.store()
        st.insert_and_complete(S(4, (1, 0)))
        st.insert_and_complete(S(6, (0, 1)))
        self.assertEqual(set(st.all_signatures()), {S(4, (1, 0)), S(6, (0, 1)), S(2, (1, 1))})
        self.assertEqual(st.stats.counters['syzygy_g_materialized'], 1)

    def test_07_same_monomial_completion_removes_both_parents(self):
        st = self.store()
        st.insert_and_complete(S(4, (1, 0)))
        old_index = st.tries[0]
        st.insert_and_complete(S(6, (1, 0)))
        self.assertEqual(st.all_signatures(), [S(2, (1, 0))])
        self.assertIs(st.tries[0], old_index)  # Post16 deletes buckets in place.
        self.assertEqual(st.divisor_coefficients((1, 0), 0), (2,))

    def test_08_no_deletion_preserves_index_object(self):
        st = self.store()
        st.insert_and_complete(S(4, (1, 0)))
        old_index = st.tries[0]
        st.insert_and_complete(S(6, (0, 1)))
        self.assertIs(st.tries[0], old_index)
        self.assertEqual(st.stats.counters['syzygy_index_records_written'], 3)
        self.assertEqual(st.stats.counters['syzygy_index_appends'], 2)

    def test_09_deletion_drops_stale_payloads(self):
        st = self.store()
        st.insert_and_complete(S(6, (2, 0)))
        self.assertEqual(st.divisor_coefficients((2, 0), 0), (6,))
        old_index = st.tries[0]
        st.insert_and_complete(S(3, (1, 0)))
        self.assertIs(st.tries[0], old_index)  # Post16 deletes buckets in place.
        self.assertEqual(st.divisor_coefficients((2, 0), 0), (3,))
        self.assertEqual(st.stats.counters['syzygy_index_rebuilds'], 0)
        self.assertEqual(st.stats.counters['syzygy_index_deletions'], 1)

    def test_10_negative_seed_and_rejected_associate(self):
        st = self.store()
        st.insert_and_complete(S(-3, (1, 0)))
        version, index = st.version, st.tries[0]
        self.assertFalse(st.insert_and_complete(S(6, (2, 0))))
        self.assertEqual(st.version, version)
        self.assertIs(st.tries[0], index)
        self.assertEqual(st.all_signatures(), [S(3, (1, 0))])

    def test_11_component_caches_and_indices_isolated(self):
        st = self.store()
        st.insert_and_complete(S(2, (1, 0), 1))
        st.divisor_coefficients((2, 2), 1)
        other_index, other_cache = st.tries[1], st.divisor_cache[1].copy()
        self.assertEqual(st.divisor_coefficients((0, 1), 0), ())
        st.insert_and_complete(S(1, (0, 1), 0))
        self.assertEqual(st.divisor_coefficients((0, 1), 0), (1,))
        self.assertIs(st.tries[1], other_index)
        self.assertEqual(st.divisor_cache[1], other_cache)

    def test_12_complete_signature_not_just_monomial(self):
        st = self.store()
        st.insert_and_complete(S(6, (1, 0)))
        self.assertFalse(st.has_divisor(S(5, (2, 0))))
        self.assertTrue(st.has_divisor(S(12, (2, 0))))

    def test_13_query_order_matches_full_rebuild(self):
        st = self.store()
        for sig in [S(6, (3, 0)), S(10, (0, 3)), S(15, (2, 2)), S(2, (1, 1))]:
            st.insert_and_complete(sig)
            fresh = DivisorIndex(2)
            groups = {}
            for s in st.components[0]:
                groups.setdefault(s.monomial, []).append(s.coefficient)
            for m, c in groups.items():
                fresh.set(m, c)
            for m in product(range(5), repeat=2):
                expected = tuple(c for group in fresh.divisors(m) for c in group)
                self.assertEqual(st.divisor_coefficients(m, 0), expected)

    def test_14_randomized_against_independent_closure(self):
        rng = random.Random(20261015)
        for trial in range(36):
            dims = (1, 2, 3, 8, 48, 2)[trial % 6]
            st = self.store(dims, 3, False)
            expected = set()
            for step in range(30):
                m = tuple(rng.randrange(3) if k < min(4, dims) else 0 for k in range(dims))
                z = S(rng.choice([-15, -6, -1, 1, 2, 4, 6, 10]), m, rng.randrange(3))
                expected = naive_closure(expected | {z})
                st.insert_and_complete(z)
                self.assertEqual(set(st.all_signatures()), expected, (trial, step))
                for query in (z, S(30, m, z.index)):
                    self.assertEqual(st.has_divisor(query), any(divides(s, query) for s in expected))
            counts = st.stats.counters
            self.assertEqual(counts['syzygy_g_signatures'],
                             counts['syzygy_g_shortcuts'] + counts['syzygy_g_materialized'])

    def test_15_large_integer_coefficients(self):
        a = 2**220 * 3
        b = 2**180 * 5
        st = self.store()
        st.insert_and_complete(S(a, (1, 0)))
        st.insert_and_complete(S(-b, (0, 1)))
        self.assertIn(S(2**180, (1, 1)), st.all_signatures())
        self.assertEqual(set(st.all_signatures()), naive_closure({S(a, (1, 0)), S(-b, (0, 1))}))

    def test_16_unit_antichain_writes_each_record_once(self):
        st = self.store(16, 1, False)
        for i in range(16):
            st.insert_and_complete(S(1, [int(i == j) for j in range(16)]))
        self.assertEqual(st.stats.counters['syzygy_index_records_written'], 16)
        self.assertEqual(st.stats.counters['syzygy_g_shortcuts'], 120)
        self.assertEqual(st.stats.counters['syzygy_g_materialized'], 0)

    def test_17_progress_marks_syzygy_work_and_restores_phase(self):
        sol = engine(progress_interval=20)
        sol.phase = 'candidate generation'
        buf = StringIO()
        with redirect_stderr(buf), patch.object(sol, '_check_budget') as check:
            sol._insert_syzygy(S(1, (1,)), check_budget=False)
            check.assert_not_called()
        self.assertIn('phase=syzygy completion', buf.getvalue())
        self.assertEqual(sol.phase, 'candidate generation')

    def test_18_silent_mode_does_not_print_or_add_budget_checks(self):
        sol = engine()
        with patch.object(sol, '_report_progress') as progress, patch.object(sol, '_check_budget') as check:
            sol._insert_syzygy(S(1, (1,)), check_budget=False)
            progress.assert_not_called()
            check.assert_not_called()

    def test_19_exception_restores_phase(self):
        sol = engine()
        sol.phase = 'initialization'
        with patch.object(sol.S, '_complete_uncovered', side_effect=ValueError('test')):
            with self.assertRaises(ValueError):
                sol._insert_syzygy(S(1, (1,)))
        self.assertEqual(sol.phase, 'initialization')

    def test_20_existing_budget_boundary_preserved(self):
        sol = engine()
        with patch.object(sol, '_check_budget') as check:
            sol._insert_syzygy(S(1, (1,)), check_budget=True)
            check.assert_called_once()

    def test_21_online_paths_replay_and_criteria(self):
        f = ['4*x^2', '9*y^4', '6*x*y^2+y']
        results = []
        for flags in ({'track_representations': True}, {'record_paths': True}):
            r = compute(f, ['x', 'y'], 'lex', 'pot',
                        options=Options(g_first=False, **flags), show_basis=False, show_u=False)
            self.assertEqual(r.status, 'completed')
            results.append(r)
        online, paths = results
        self.assertEqual(online.basis, paths.basis)
        self.assertEqual(online.syzygies, paths.syzygies)
        self.assertTrue(verify_result(online, mode='static', strong=True,
                                     require_certificates=True)['signature_basis_certified'])
        self.assertTrue(verify_generation_paths(paths)['signature_basis_certified'])
        self.assertEqual(reconstruct_syzygies(paths, verify=True).syzygies,
                         online.syzygy_representations)

    def test_22_zero_inputs_and_no_syzygies(self):
        for f in (['0', '0'], ['2*x']):
            r = compute(f, ['x'], 'lex', 'pot', options=Options(track_representations=True),
                        show_basis=False, show_u=False)
            self.assertEqual(r.status, 'completed')
            self.assertTrue(verify_result(r, require_certificates=True)['passed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
