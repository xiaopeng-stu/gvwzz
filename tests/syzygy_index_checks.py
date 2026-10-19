"""Post16: sparse two-way syzygy index, exact coefficients, and stable ordering."""
import random
import unittest
from itertools import product
from unittest.mock import patch

from gvwzz.core import Signature
from gvwzz.structures import DivisorIndex, SyzygyIndex, SyzygyStore, Stats
from tests.syzygy_checks import naive_closure


def divides(a, b):
    return all(x <= y for x, y in zip(a, b))


def sig(c, m, i=0):
    return Signature(c, tuple(m), i)


class SyzygyIndexChecks(unittest.TestCase):
    def test_01_empty_index(self):
        idx = SyzygyIndex(3)
        self.assertEqual(list(idx.divisors((0, 0, 0))), [])
        self.assertEqual(list(idx.multiples((0, 0, 0))), [])

    def test_02_constant(self):
        idx = SyzygyIndex(2)
        idx.set((0, 0), [7])
        self.assertEqual(list(idx.divisors((3, 5))), [[7]])
        self.assertEqual(list(idx.multiples((0, 0))), [((0, 0), [7])])
        self.assertEqual(list(idx.multiples((0, 1))), [])
        self.assertEqual(idx.values, [[], []])

    def test_03_new_thresholds_between_existing_values(self):
        idx = SyzygyIndex(1)
        values = (8, 3, 5, 1, 12, 0, 10, 6)
        for e in values:
            idx.set((e,), e)
        for q in range(15):
            self.assertEqual(list(idx.divisors((q,))), [e for e in values if e <= q])
            self.assertEqual(list(idx.multiples((q,))), [((e,), e) for e in values if e >= q])

    def test_04_update_keeps_insertion_id(self):
        idx = SyzygyIndex(2)
        idx.set((3, 0), [6])
        idx.set((0, 3), [10])
        before = (dict(idx.positions), idx.all_mask, [dict(x) for x in idx.masks])
        idx.set((3, 0), [6, 4])
        self.assertEqual((idx.positions, idx.all_mask, idx.masks), before)
        self.assertEqual(list(idx.divisors((3, 3))), [[6, 4], [10]])

    def test_05_none_payload_is_not_enumerated(self):
        idx = SyzygyIndex(1)
        idx.set((2,), None)
        idx.set((3,), [5])
        self.assertEqual(list(idx.divisors((4,))), [[5]])
        self.assertEqual(list(idx.multiples((1,))), [((3,), [5])])
        idx.set((2,), [7])
        self.assertEqual(list(idx.divisors((4,))), [[7], [5]])

    def test_06_more_than_one_machine_word(self):
        idx = SyzygyIndex(2)
        for i in range(1025):
            idx.set((i, 1024 - i), i)
        self.assertEqual(list(idx.divisors((1024, 1024))), list(range(1025)))
        self.assertEqual(list(idx.multiples((1000, 0))), [((i, 1024-i), i) for i in range(1000, 1025)])

    def test_07_large_exponents_use_observed_thresholds(self):
        idx = SyzygyIndex(1)
        big = 10**20
        idx.set((big,), 'large')
        idx.set((1,), 'small')
        self.assertEqual(len(idx.values[0]), 2)
        self.assertEqual(list(idx.divisors((big-1,))), ['small'])
        self.assertEqual(list(idx.multiples((2,))), [((big,), 'large')])

    def test_08_sparse_48_dimensions(self):
        idx = SyzygyIndex(48)
        for i in range(48):
            m = tuple(int(i == j) for j in range(48))
            idx.set(m, i)
        m = tuple(int(j in (2, 35, 47)) for j in range(48))
        self.assertEqual(list(idx.divisors(m)), [2, 35, 47])
        self.assertEqual(list(idx.multiples(m)), [])
        self.assertEqual(sum(len(v) for v in idx.values), 48)

    def test_09_wrong_dimension_rejected(self):
        idx = SyzygyIndex(2)
        for method in (idx.divisors, idx.multiples):
            with self.assertRaises(ValueError):
                list(method((1,)))
        with self.assertRaises(ValueError):
            idx.set((1,), [])
        self.assertEqual(idx.all_mask, 0)

    def test_10_zero_dimensional_monomial(self):
        idx = SyzygyIndex(0)
        idx.set((), 'one')
        self.assertEqual(list(idx.divisors(())), ['one'])
        self.assertEqual(list(idx.multiples(())), [((), 'one')])

    def test_11_randomized_queries_match_scan_and_original_index(self):
        rng = random.Random(160721)
        for dim in (1, 2, 3, 8, 48):
            idx, ref = SyzygyIndex(dim), DivisorIndex(dim)
            data = {}
            for j in range(160):
                m = tuple(rng.randrange(5) if k < 6 and rng.random() < 0.5 else 0 for k in range(dim))
                data[m] = j
                idx.set(m, j)
                ref.set(m, j)
                for _ in range(3):
                    q = tuple(rng.randrange(6) if k < 6 else 0 for k in range(dim))
                    self.assertEqual(list(idx.divisors(q)), list(ref.divisors(q)))
                    self.assertEqual(list(idx.divisors(q)), [v for m, v in data.items() if divides(m, q)])
                    self.assertEqual(list(idx.multiples(q)), [(m, v) for m, v in data.items() if divides(q, m)])

    def test_12_append_reuses_component_list(self):
        st = SyzygyStore(2, 1, Stats())
        st.insert_and_complete(sig(1, (1, 0)))
        old, idx = st.components[0], st.tries[0]
        st.insert_and_complete(sig(1, (0, 1)))
        self.assertIs(st.components[0], old)
        self.assertIs(st.tries[0], idx)
        self.assertEqual(st.components[0], [sig(1, (1, 0)), sig(1, (0, 1))])

    def test_13_unit_metadata_nonunit_to_unit(self):
        st = SyzygyStore(2, 1, Stats(), debug=True)
        st.insert_and_complete(sig(6, (1, 0)))
        self.assertEqual(st.nonunit_counts, [1])
        st.insert_and_complete(sig(1, (0, 0)))
        self.assertEqual(st.nonunit_counts, [0])
        self.assertEqual(st.components[0], [sig(1, (0, 0))])

    def test_14_unit_metadata_mixed_components(self):
        st = SyzygyStore(2, 2, Stats(), debug=True)
        st.insert_and_complete(sig(1, (1, 0)))
        st.insert_and_complete(sig(12, (0, 1)))
        st.insert_and_complete(sig(7, (1, 0), 1))
        self.assertEqual(st.nonunit_counts, [1, 1])
        st.insert_and_complete(sig(1, (0, 0)))
        self.assertEqual(st.nonunit_counts, [0, 1])

    def test_15_reverse_query_still_checks_coefficients(self):
        st = SyzygyStore(2, 1, Stats(), debug=True)
        st.insert_and_complete(sig(2, (2, 0)))
        st.insert_and_complete(sig(6, (1, 0)))
        self.assertEqual(st.components[0], [sig(2, (2, 0)), sig(6, (1, 0))])
        self.assertEqual(st.stats.counters['syzygy_multiple_candidates'], 1)
        self.assertEqual(st.stats.counters['syzygy_index_rebuilds'], 0)

    def test_16_no_signature_scan_for_unit_antichain(self):
        st = SyzygyStore(48, 1, Stats())
        for i in range(32):
            st.insert_and_complete(sig(1, (int(i == j) for j in range(48))))
        with patch.object(Signature, 'divides', side_effect=RuntimeError('linear signature scan')):
            st.insert_and_complete(sig(1, (int(40 == j) for j in range(48))))
        self.assertEqual(len(st), 33)
        self.assertEqual(st.stats.counters['syzygy_multiple_candidates'], 0)

    def test_17_deletion_has_no_stale_reverse_rows(self):
        st = SyzygyStore(2, 1, Stats())
        st.insert_and_complete(sig(6, (3, 0)))
        st.insert_and_complete(sig(6, (0, 3)))
        st.insert_and_complete(sig(3, (1, 0)))
        self.assertEqual(list(st.tries[0].multiples((2, 0))), [])
        self.assertEqual(st.divisor_coefficients((3, 0), 0), (3,))

    def test_18_debug_snapshot_preserved(self):
        st = SyzygyStore(2, 1, Stats(), debug=True)
        for c, m in ((1,(3,0)), (1,(0,3)), (2,(1,1)), (1,(0,0))):
            st.insert_and_complete(sig(c, m))
        self.assertEqual(st.components[0], [sig(1, (0, 0))])

    def test_19_large_integer_coefficient_not_lost(self):
        st = SyzygyStore(2, 1, Stats(), debug=True)
        a, b = 2**300 * 3, 2**250 * 5
        inputs = {sig(a, (2, 0)), sig(-b, (0, 2)), sig(2**251, (1, 1))}
        for s in sorted(inputs, key=lambda s: s.monomial):
            st.insert_and_complete(s)
        self.assertEqual(set(st.all_signatures()), naive_closure(inputs))

    def test_20_repeated_closure_and_queries_mixed_coefficients(self):
        rng = random.Random(117015)
        for dim in (1, 2, 3, 8, 48):
            for trial in range(10):
                st = SyzygyStore(dim, 3, Stats())
                expected = set()
                for step in range(30):
                    z = sig(rng.choice([-15, -6, -1, 1, 2, 3, 6, 10]),
                            (rng.randrange(4) if k < 3 else 0 for k in range(dim)), rng.randrange(3))
                    expected = naive_closure(expected | {z})
                    st.insert_and_complete(z)
                    self.assertEqual(set(st.all_signatures()), expected)
                    self.assertEqual(st.nonunit_counts, [sum(s.index==i and s.coefficient != 1 for s in expected) for i in range(3)])
                    self.assertEqual(len(st), len(expected))

    def test_21_delete_and_reinsert_moves_bucket_to_end(self):
        idx = SyzygyIndex(1)
        idx.set((5,), 'old'); idx.set((3,), 'survivor')
        self.assertTrue(idx.remove((5,)))
        self.assertFalse(idx.remove((5,)))
        idx.set((5,), 'new')
        self.assertEqual(list(idx.divisors((8,))), ['survivor', 'new'])
        self.assertEqual(list(idx.multiples((1,))), [((3,), 'survivor'), ((5,), 'new')])
        self.assertIsNone(idx.monomials[0])
        self.assertIsNone(idx.payloads[0])

    def test_22_deleted_thresholds_cannot_leak(self):
        idx = SyzygyIndex(1)
        idx.set((8,), 'deleted'); idx.remove((8,)); idx.set((5,), 'live')
        self.assertEqual(list(idx.multiples((6,))), [])
        self.assertEqual(list(idx.divisors((9,))), ['live'])
        idx.set((6,), 'middle')
        self.assertEqual(list(idx.multiples((6,))), [((6,), 'middle')])
        self.assertEqual(list(idx.divisors((7,))), ['live', 'middle'])

    def test_23_compaction_threshold(self):
        idx = SyzygyIndex(2)
        for i in range(130): idx.set((i, 130-i), i)
        for i in range(63): idx.remove((i, 130-i))
        self.assertFalse(idx.needs_compaction())
        for i in range(63, 66): idx.remove((i, 130-i))
        self.assertTrue(idx.needs_compaction())

    def test_24_store_compaction_keeps_order_and_exact_queries(self):
        st = SyzygyStore(2, 1, Stats())
        for i in range(80): st.insert_and_complete(sig(1, (i, 100-i)))
        old = st.tries[0]
        st.insert_and_complete(sig(1, (0, 0)))
        self.assertEqual(st.all_signatures(), [sig(1, (0, 0))])
        self.assertIsNot(st.tries[0], old)
        self.assertEqual(st.stats.counters['syzygy_index_compactions'], 1)
        self.assertEqual(st.stats.counters['syzygy_index_rebuilds'], 1)
        self.assertEqual(len(st.tries[0].payloads), 1)
        self.assertEqual(st.divisor_coefficients((200, 200), 0), (1,))
        self.assertEqual(list(st.tries[0].multiples((1, 1))), [])

    def test_25_dynamic_insert_delete_queries_against_ordered_dict(self):
        rng = random.Random(660117)
        for dim in (1, 3, 48):
            idx, data = SyzygyIndex(dim), {}
            for step in range(500):
                if data and rng.randrange(3) == 0:
                    m = rng.choice(list(data)); del data[m]; idx.remove(m)
                else:
                    m = tuple(rng.randrange(6) if k < 4 else 0 for k in range(dim))
                    data[m] = step; idx.set(m, step)
                q = tuple(rng.randrange(7) if k < 4 else 0 for k in range(dim))
                self.assertEqual(list(idx.divisors(q)), [v for m, v in data.items() if divides(m, q)])
                self.assertEqual(list(idx.multiples(q)), [(m, v) for m, v in data.items() if divides(q, m)])
                self.assertEqual(idx.all_mask.bit_count(), len(data))

    def test_26_ordered_component_keyed_deletion(self):
        st = SyzygyStore(2, 1, Stats())
        for m in ((4, 0), (0, 4), (2, 2)):
            st.insert_and_complete(sig(1, m))
        component = st.components[0]
        old_snapshot = tuple(component)
        self.assertEqual(component[0], sig(1, (4, 0)))
        st.insert_and_complete(sig(1, (1, 1)))
        self.assertIs(st.components[0], component)
        self.assertEqual(list(component), [sig(1, (4, 0)), sig(1, (0, 4)), sig(1, (1, 1))])
        self.assertEqual(old_snapshot[-1], sig(1, (2, 2)))
        self.assertEqual(component[-1], sig(1, (1, 1)))
        self.assertEqual(component[:2], old_snapshot[:2])

    def test_27_survivor_order_after_many_deletions(self):
        st = SyzygyStore(2, 1, Stats())
        for i in range(100): st.insert_and_complete(sig(1, (i, 200-i)))
        old = list(st.components[0])
        st.insert_and_complete(sig(1, (20, 100)))
        expected = [s for s in old if not divides((20, 100), s.monomial)] + [sig(1, (20, 100))]
        self.assertEqual(list(st.components[0]), expected)
        self.assertEqual(st.stats.counters['syzygy_index_compactions'], 1)
        fresh = DivisorIndex(2)
        for s in expected: fresh.set(s.monomial, [s.coefficient])
        for m in product((0, 20, 100, 200), repeat=2):
            self.assertEqual(st.divisor_coefficients(m, 0), tuple(c for g in fresh.divisors(m) for c in g))

    def test_28_component_and_index_deletions_release_records(self):
        st = SyzygyStore(2, 1, Stats())
        st.insert_and_complete(sig(7, (2, 0)))
        st.insert_and_complete(sig(7, (0, 2)))
        st.insert_and_complete(sig(1, (1, 0)))
        self.assertNotIn(((2, 0), 7), st.components[0]._entries)
        self.assertNotIn((2, 0), st.tries[0].positions)
        self.assertEqual(st.nonunit_counts, [1])
        self.assertEqual(st.all_signatures(), [sig(7, (0, 2)), sig(1, (1, 0))])


if __name__ == '__main__':
    unittest.main(verbosity=2)
