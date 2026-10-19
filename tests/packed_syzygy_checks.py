"""Exactness checks for the packed, deferred syzygy generation gate."""
import random
import unittest

from gvwzz.core import Signature
from gvwzz.structures import Stats, SyzygyIndex, SyzygyStore


def pack(monomial):
    return int.from_bytes(bytes(monomial), 'little')


class PackedSyzygyChecks(unittest.TestCase):
    def test_block_masks_against_scan_after_updates_and_deletions(self):
        rng = random.Random(20261015)
        for dimension in (0, 1, 7, 8, 9, 48):
            trie = SyzygyIndex(dimension)
            data = {}
            for step in range(150):
                if data and rng.randrange(4) == 0:
                    monomial = rng.choice(list(data))
                    del data[monomial]
                    trie.remove(monomial)
                else:
                    monomial = tuple(rng.choice((0, 0, 1, 3, 127, 128, 256, 10**20))
                                     if rng.randrange(4) == 0 else 0
                                     for _ in range(dimension))
                    data[monomial] = step
                    trie.set(monomial, step)
                for _ in range(3):
                    query = tuple(rng.choice((0, 1, 3, 127)) for _ in range(dimension))
                    expected = sum(1 << row for m, row in trie.positions.items()
                                   if all(a <= b for a, b in zip(m, query)))
                    self.assertEqual(trie._divisor_mask_packed(pack(query)), expected)
                    # A second call uses the block caches, including zero masks.
                    self.assertEqual(trie._divisor_mask_packed(pack(query)), expected)

    def test_mixed_coefficients_and_versions_match_actual_witnesses(self):
        rng = random.Random(91844)
        for dimension in (1, 3, 9, 48):
            store = SyzygyStore(dimension, 2, Stats())
            for step in range(65):
                monomial = tuple(rng.randrange(4) if k < 3 else 0
                                 for k in range(dimension))
                store.insert_and_complete(Signature(rng.choice((1, 2, 3, 6, 10)),
                                                    monomial, rng.randrange(2)))
                for _ in range(12):
                    monomial = tuple(rng.randrange(5) if k < 3 else 0
                                     for k in range(dimension))
                    query = Signature(rng.choice((-30, -7, 1, 2, 5, 6, 10, 12)),
                                      monomial, rng.randrange(2))
                    expected = any(s.divides(query) for s in store.all_signatures())
                    self.assertEqual(store._has_divisor_packed(
                        query.coefficient, pack(monomial), query.index), expected)
                    self.assertEqual(store.has_divisor(query), expected)

    def test_guard_borrow_and_unbounded_stored_exponents(self):
        store = SyzygyStore(9, 1, Stats())
        a = (127, 0, 0, 0, 0, 0, 0, 1, 0)
        b = (0, 0, 0, 0, 0, 0, 0, 0, 10**20)
        store.insert_and_complete(Signature(2, a, 0))
        store.insert_and_complete(Signature(3, b, 0))
        # Establish a packed recent witness, then force a borrow in its first
        # lane and a zero in another lane. Neither may match across lanes.
        self.assertTrue(store._has_divisor_packed(2, pack(a), 0))
        for q in ((126, 1, 0, 0, 0, 0, 0, 1, 0),
                  (127, 0, 0, 0, 0, 0, 0, 0, 127),
                  (127,) * 9):
            for c in (1, 2, 3, 6):
                expected = any(s.divides(Signature(c, q, 0))
                               for s in store.all_signatures())
                self.assertEqual(store._has_divisor_packed(c, pack(q), 0), expected)

    def test_negative_cache_invalidated_and_positive_witness_survives_compaction(self):
        store = SyzygyStore(2, 1, Stats())
        for i in range(80):
            store.insert_and_complete(Signature(2, (i, 100 - i), 0))
        self.assertFalse(store._has_divisor_packed(1, pack((100, 100)), 0))
        self.assertTrue(store._has_divisor_packed(2, pack((100, 100)), 0))
        store.insert_and_complete(Signature(1, (0, 0), 0))
        self.assertGreater(store.stats.counters['syzygy_index_compactions'], 0)
        for c in (1, 2, 17):
            self.assertTrue(store._has_divisor_packed(c, pack((100, 100)), 0))
            self.assertTrue(store._has_divisor_packed(c, pack((0, 0)), 0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
