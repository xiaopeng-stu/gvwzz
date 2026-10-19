"""Exactness checks for the native row emitter's batched syzygy oracle."""
import os
import random
import unittest

from gvwzz.core import Signature
from gvwzz.structures import Stats, SyzygyStore

try:
    if os.environ.get('GVWZZ_NO_NATIVE'):
        raise ImportError('native acceleration disabled')
    from gvwzz._native_indices import PackedSyzygyQueries
except ImportError:
    PackedSyzygyQueries = None


def pack(monomial):
    return int.from_bytes(bytes(monomial), 'little')


@unittest.skipIf(PackedSyzygyQueries is None, 'optional native oracle is not built')
class NativeSyzygyOracleChecks(unittest.TestCase):
    def test_queries_match_single_exact_divisor_across_mutations(self):
        rng = random.Random(251031)
        coefficients = (1, -1, 2, 3, -6, 10, 2**64 - 1, 2**64,
                        -(2**4096 + 17), 3 * (2**4096 + 17))
        for nvars in (0, 1, 7, 8, 9):
            store = SyzygyStore(nvars, 3, Stats())
            oracle = PackedSyzygyQueries(store)
            for step in range(60):
                monomial = tuple(rng.choice((0, 1, 2, 127, 128, 2**100))
                                 if k < 3 else 0 for k in range(nvars))
                store.insert_and_complete(Signature(
                    rng.choice(coefficients), monomial, rng.randrange(3)))
                oracle.refresh()
                query_count = store.stats.counters['syzygy_existence_queries']
                for _ in range(15):
                    monomial = tuple(rng.choice((0, 1, 2, 3, 126, 127, 128, 255))
                                     if k < 3 else 0 for k in range(nvars))
                    coefficient = rng.choice(coefficients)
                    index = rng.randrange(3)
                    target = Signature(coefficient, monomial, index)
                    expected = any(s.divides(target) for s in store.components[index])
                    self.assertEqual(oracle.query(coefficient, pack(monomial), index), expected)
                    self.assertEqual(oracle.query(coefficient, pack(monomial), index), expected)
                oracle.flush()
                self.assertEqual(store.stats.counters['syzygy_existence_queries'], query_count + 30)

    def test_negative_cache_is_invalidated_and_old_positive_witness_remains_valid(self):
        store = SyzygyStore(2, 1, Stats())
        oracle = PackedSyzygyQueries(store)
        store.insert_and_complete(Signature(6, (2, 1), 0))
        oracle.refresh(0)
        self.assertFalse(oracle.query(2, pack((2, 1)), 0))
        self.assertTrue(oracle.query(6, pack((2, 1)), 0))
        store.insert_and_complete(Signature(2, (1, 1), 0))
        oracle.refresh(0)
        self.assertTrue(oracle.query(2, pack((2, 1)), 0))
        self.assertTrue(oracle.query(6, pack((2, 1)), 0))
        self.assertFalse(oracle.query(3, pack((2, 1)), 0))
        oracle.flush()

    def test_high_lanes_and_arbitrary_precision_coefficients_are_not_truncated(self):
        huge = 2**4096 + 17
        store = SyzygyStore(8, 1, Stats())
        store.insert_and_complete(Signature(huge, (127,) * 8, 0))
        store.insert_and_complete(Signature(5, (128, 0, 0, 0, 0, 0, 0, 0), 0))
        oracle = PackedSyzygyQueries(store)
        for coefficient in (huge, -huge, huge * 7, huge + 1, 5, 2**64 + 5):
            for monomial in ((127,) * 8, (128,) * 8, (0,) * 8):
                target = Signature(coefficient, monomial, 0)
                expected = any(s.divides(target) for s in store.components[0])
                self.assertEqual(oracle.query(coefficient, pack(monomial), 0), expected)
        oracle.flush()

    def test_rejects_invalid_packed_length_and_component(self):
        oracle = PackedSyzygyQueries(SyzygyStore(2, 1, Stats()))
        for packed in (-1, 1 << 16):
            with self.assertRaises(ValueError):
                oracle.query(1, packed, 0)
        for index in (-1, 1):
            with self.assertRaises(IndexError):
                oracle.query(1, 0, index)


if __name__ == '__main__':
    unittest.main(verbosity=2)
