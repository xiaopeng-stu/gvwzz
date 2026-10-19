"""Independent complete-LT oracle for stage-compaction fast paths."""
from random import Random
from types import SimpleNamespace
import unittest

from gvwzz.core import divides_monomial
from gvwzz.incremental import _minimal_records
from gvwzz.polynomial import IntegerPolynomialRing


def reference(leads, backend):
    rows = [(m, abs(c), i, 1 if c > 0 else -1) for i, (m, c) in enumerate(leads)]
    rows.sort(key=lambda row: (backend.order_key(row[0]), row[1], row[2]))
    keep = []
    for row in rows:
        if not any(row[1] % old[1] == 0 and divides_monomial(old[0], row[0]) for old in keep):
            keep.append(row)
    return [(i, sign) for _, _, i, sign in keep]


class IncrementalMinimizationChecks(unittest.TestCase):
    def compare(self, leads, backend):
        checks = []
        engine = SimpleNamespace(backend=backend, leading=leads,
            options=SimpleNamespace(debug=True), _check_budget=lambda: checks.append(None))
        self.assertEqual(_minimal_records(engine), reference(leads, backend))
        self.assertEqual(len(checks), len(leads))

    def test_squarefree_units_exact_order_and_duplicate_signs(self):
        rng = Random(20261018)
        for order in ('lex', 'grlex', 'grevlex'):
            backend = IntegerPolynomialRing(['x' + str(i) for i in range(8)], order)
            for _ in range(50):
                leads = [(tuple(rng.randrange(2) for _ in range(8)), rng.choice((-1, 1)))
                         for _ in range(30)]
                leads += leads[:5]
                rng.shuffle(leads)
                self.compare(leads, backend)

    def test_general_coefficient_and_large_exponent_fallbacks(self):
        rng = Random(20261019)
        for order in ('lex', 'grlex', 'grevlex'):
            backend = IntegerPolynomialRing(['x', 'y', 'z'], order)
            for _ in range(30):
                leads = [(tuple(rng.randrange(4) for _ in range(3)), rng.choice((-6, -3, -1, 1, 2, 3, 6)))
                         for _ in range(20)]
                self.compare(leads, backend)
            self.compare([((10**30, 0, 0), 1), ((10**30 + 1, 0, 0), -1), ((0, 1, 0), 2)], backend)

    def test_zero_monomial_and_empty_input(self):
        backend = IntegerPolynomialRing(['x', 'y'], 'grevlex')
        self.compare([], backend)
        self.compare([((1, 0), 1), ((0, 0), -1), ((0, 0), 1), ((0, 1), 1)], backend)

    def test_degree_limit_falls_back(self):
        backend = IntegerPolynomialRing(['x' + str(i) for i in range(10)], 'grevlex')
        self.compare([((1,) * 10, 1), ((1,) * 9 + (0,), -1), ((1,) * 10, -1)], backend)


if __name__ == '__main__':
    unittest.main()
