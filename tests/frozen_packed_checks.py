"""Exact independent checks for unbounded packed frozen multipliers."""
import random
import unittest

from gvwzz.frozen import FrozenSeedIndex


def minimal_oracle(rows, leading, selected):
    selected = sorted(set(selected))
    shifts = {i: tuple(max(a - b, 0) for a, b in zip(rows[i], leading))
              for i in selected}
    return tuple(i for i in selected if not any(
        (shifts[j] != shifts[i] or j < i) and
        all(a <= b for a, b in zip(shifts[j], shifts[i]))
        for j in selected if i != j))


class FrozenPackedChecks(unittest.TestCase):
    def check_selection(self, rows, leading, selected):
        index = FrozenSeedIndex(len(leading), rows)
        expected = minimal_oracle(rows, leading, selected)
        self.assertEqual(index.minimal_parents(leading, iter(selected)), expected)
        mask = sum(1 << i for i in set(selected))
        self.assertEqual(index.minimal_parents_mask(leading, mask), expected)

    def test_lane_borrows_and_large_current_coordinates(self):
        for power in (2, 7, 8, 127, 128, 255, 256, 2**64, 10**80):
            rows = [(power, 0, 1), (power - 1, 1, 0), (0, power, 1),
                    (1, power - 1, 0), (power + 1, 1, 1)]
            for leading in ((0, 0, 0), (power - 1, 0, 0), (power, power, 0),
                            (10 * power, 0, 0), (0, 10 * power, 10 * power)):
                self.check_selection(rows, leading, list(reversed(range(len(rows)))))

    def test_powers_of_two_are_not_degree_one_variables(self):
        rows = [(2, 0, 0), (0, 2, 0), (1, 1, 0), (0, 0, 4),
                (1, 0, 1), (0, 1, 1), (0, 0, 2), (1, 0, 2)]
        self.check_selection(rows, (0, 0, 0), list(range(len(rows))))

    def test_earliest_original_row_with_repeated_unordered_parents(self):
        rows = [(3, 1), (3, 2), (4, 1), (2, 3), (2, 4), (3, 1)]
        selected = [5, 3, 2, 1, 5, 0, 4, 2]
        self.check_selection(rows, (2, 2), selected)
        self.check_selection(rows, (10, 10), selected)

    def test_sparse_high_dimensions_and_arbitrary_exponents(self):
        rng = random.Random(22001)
        for nvars in (1, 3, 51, 130):
            for _ in range(15):
                rows = []
                for _ in range(45):
                    row = [0] * nvars
                    for k in rng.sample(range(nvars), min(5, nvars)):
                        row[k] = rng.choice((0, 1, 2, 3, 256, 10**60))
                    rows.append(tuple(row))
                leading = tuple(rng.choice((0, 0, 1, 2, 257, 10**70))
                                for _ in range(nvars))
                selected = [rng.randrange(len(rows)) for _ in range(40)]
                self.check_selection(rows, leading, selected)

    def test_dense_antichains_with_mixed_degrees(self):
        rng = random.Random(22002)
        rows = [tuple(rng.randrange(1, 5) for _ in range(20)) for _ in range(300)]
        for leading in ((0,) * 20, (1,) * 20, (3,) * 20):
            self.check_selection(rows, leading, list(range(len(rows))))

    def test_empty_parent_selection_and_zero_variable_ring(self):
        self.check_selection([], (0, 0), [])
        self.check_selection([(2, 3)], (0, 0), [])
        self.check_selection([(), (), ()], (), [2, 0, 2, 1])


if __name__ == '__main__':
    unittest.main()
