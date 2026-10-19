"""Exact differential checks for the optional packed syzygy lookup kernel."""
import random
import unittest

from gvwzz.core import Signature
from gvwzz.structures import Stats, SyzygyIndex, SyzygyStore

try:
    from gvwzz._native_indices import divisor_mask_packed, has_divisor_packed
except ImportError:
    divisor_mask_packed = has_divisor_packed = None


def pack(m):
    return int.from_bytes(bytes(m), 'little')


@unittest.skipIf(divisor_mask_packed is None, 'optional native index is not built')
class NativeIndexChecks(unittest.TestCase):
    def test_native_mask_matches_scan_with_unbounded_thresholds_and_deleted_rows(self):
        rng = random.Random(21551)
        for n in (0, 1, 7, 8, 9, 48, 137):
            trie = SyzygyIndex(n)
            for step in range(230):
                if trie.positions and step % 5 == 0:
                    trie.remove(rng.choice(list(trie.positions)))
                else:
                    m = [0] * n
                    for _ in range(min(n, 5)):
                        m[rng.randrange(n)] = rng.choice((1, 2, 127, 128, 255, 256, 2**170))
                    trie.set(tuple(m), step)
                for _ in range(3):
                    m = tuple(rng.choice((0, 1, 2, 127)) for _ in range(n))
                    expected = sum(1 << row for stored, row in trie.positions.items()
                                   if all(a <= b for a, b in zip(stored, m)))
                    self.assertEqual(divisor_mask_packed(trie, pack(m)), expected)
                    self.assertEqual(divisor_mask_packed(trie, pack(m)), expected)

    def test_native_and_python_stores_preserve_answers_counts_and_versions(self):
        rng = random.Random(21552)
        for n in (1, 9, 48):
            for unit_only in (True, False):
                native = SyzygyStore(n, 2, Stats())
                portable = SyzygyStore(n, 2, Stats())
                portable._has_divisor_packed = portable._has_divisor_packed_python
                for step in range(85):
                    m = tuple(rng.randrange(8) if k < 3 else 0 for k in range(n))
                    c = 1 if unit_only else rng.choice((2, -3, 6, 10, 2**200 + 1))
                    sig = Signature(c, m, rng.randrange(2))
                    self.assertEqual(native.insert_and_complete(sig), portable.insert_and_complete(sig))
                    for _ in range(10):
                        m = tuple(rng.randrange(9) if k < 3 else 0 for k in range(n))
                        c = rng.choice((-30, -1, 1, 6, 2**300, 2**200 + 1))
                        i = rng.randrange(2)
                        self.assertEqual(native._has_divisor_packed(c, pack(m), i),
                                         portable._has_divisor_packed(c, pack(m), i))
                    self.assertEqual(native.all_signatures(), portable.all_signatures())
                    self.assertEqual(native.stats.counters, portable.stats.counters)


if __name__ == '__main__':
    unittest.main(verbosity=2)
