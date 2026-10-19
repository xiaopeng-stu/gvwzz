"""Exact exhaustive and large-exponent checks for frozen-prefix pruning."""
from itertools import product
from math import gcd, lcm
from random import Random
from types import SimpleNamespace
import unittest

from gvwzz.frozen import FrozenSeedIndex


class FrozenSeedChecks(unittest.TestCase):
    def test_exhaustive_intersection_equivalence_and_parent_order(self):
        monomials = list(product(range(3), repeat=3))
        parents = monomials + [monomials[4], monomials[4]]
        index = FrozenSeedIndex(3, parents)
        for b in monomials:
            for s in monomials:
                expected = tuple(i for i, a in enumerate(parents)
                    if not all(a[k] <= max(a[k], b[k]) - b[k] + s[k]
                               for k in range(3)))
                found, omitted = index.surviving_parents(b, s)
                self.assertEqual(found, expected)
                self.assertEqual(omitted, len(parents) - len(expected))

    def test_arbitrary_exponents_and_sparse_48_variable_rows(self):
        rng = Random(20261015)
        parents = []
        for _ in range(300):
            a = [0] * 48
            for k in rng.sample(range(48), 5):
                a[k] = rng.choice((1, 2, 127, 128, 255, 256, 10**40))
            parents.append(tuple(a))
        index = FrozenSeedIndex(48, parents)
        for _ in range(100):
            b = tuple(rng.choice((0, 0, 1, 128, 256, 10**41)) for _ in range(48))
            s = tuple(rng.choice((0, 0, 1, 127, 255, 10**40)) for _ in range(48))
            expected = sum(1 << i for i, a in enumerate(parents)
                if any(min(x, y) > z for x, y, z in zip(a, b, s)))
            self.assertEqual(index.survivor_mask(b, s), expected)

    def test_general_coefficient_gate_is_necessary(self):
        for a, b, c in product(range(-7, 8), repeat=3):
            if not a or not b or not c:
                continue
            coefficient = (lcm(abs(a), abs(b)) // b) * c
            self.assertEqual(coefficient % a == 0, c % gcd(abs(a), abs(b)) == 0)
        # This pair is rejected geometrically but not over the integers.
        self.assertEqual(FrozenSeedIndex(2, [(1, 0)]).survivor_mask((0, 1), (0, 0)), 0)
        self.assertNotEqual((lcm(6, 4) // 4) % 6, 0)

    def test_empty_rows_and_empty_ring(self):
        self.assertEqual(FrozenSeedIndex(3, []).surviving_parents((0, 0, 0), (0, 0, 0)), ((), 0))
        self.assertEqual(FrozenSeedIndex(0, [(), ()]).surviving_parents((), ()), ((), 2))

    def test_minimal_multiplier_antichain_and_earliest_rows(self):
        rng = Random(20261016)
        for largest in (1, 3):
            for _ in range(150):
                parents = [tuple(rng.randrange(largest + 1) for _ in range(4))
                           for _ in range(25)]
                parents += parents[:3]
                b = tuple(rng.randrange(largest + 1) for _ in range(4))
                index = FrozenSeedIndex(4, parents)
                selected = sorted(rng.sample(range(len(parents)), 20))
                u = {i: tuple(max(a - c, 0) for a, c in zip(parents[i], b))
                     for i in selected}
                expected = tuple(i for i in selected if not any(
                    all(x <= y for x, y in zip(u[j], u[i]))
                    and (u[j] != u[i] or j < i) for j in selected if j != i))
                found = index.minimal_parents(b, selected)
                self.assertEqual(found, expected)
                for i in selected:
                    self.assertTrue(any(all(x <= y for x, y in zip(u[j], u[i])) for j in found))

    def test_antichain_empty_and_arbitrary_exponents(self):
        index = FrozenSeedIndex(2, [(10**50, 0), (10**50 + 1, 1), (0, 10**50)])
        self.assertEqual(index.minimal_parents((10**50 - 1, 0), range(3)), (0, 2))
        self.assertEqual(index.minimal_parents((0, 0), ()), ())

    def test_optional_native_antichain_matches_unbounded_oracle(self):
        try:
            from gvwzz._native_pairs import minimal_frozen_parents
        except ImportError:
            self.skipTest('Optional native pair accelerator is not built')
        rng = Random(20261017)
        for nvars in (1, 3, 8, 48, 64):
            for _ in range(20):
                rows = []
                for _ in range(100):
                    a = [0] * nvars
                    for k in rng.sample(range(nvars), rng.randrange(min(7, nvars) + 1)):
                        a[k] = 1
                    rows.append(tuple(a))
                index = FrozenSeedIndex(nvars, rows)
                b = tuple(rng.randrange(2) for _ in range(nvars))
                engine = SimpleNamespace(backend=SimpleNamespace(names=range(nvars)),
                    frozen_count=len(rows), leading=[(a, 1) for a in rows] + [(b, 1)],
                    _native_antichain=None)
                selected = sorted(rng.sample(range(len(rows)), 80))
                self.assertEqual(tuple(minimal_frozen_parents(engine, len(rows), selected)),
                                 index.minimal_parents(b, selected))
        for nvars, rows, current in ((2, [(2**32, 0)], (1, 0)),
                                     (2, [(1, 0)], (2**32, 0))):
            engine = SimpleNamespace(backend=SimpleNamespace(names=range(nvars)),
                frozen_count=len(rows), leading=[(a, 1) for a in rows] + [(current, 1)],
                _native_antichain=None)
            self.assertIsNone(minimal_frozen_parents(engine, len(rows), [0]))


if __name__ == '__main__':
    unittest.main()
