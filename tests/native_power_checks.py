"""Exact general-exponent native multiplier selection and committed covers."""
from random import Random
from types import SimpleNamespace
import unittest

from gvwzz import Options
from gvwzz.core import Label, Signature
from gvwzz.orders import ModuleOrder
from gvwzz.polynomial import IntegerPolynomialRing
from gvwzz.solver import GVWInteger
try:
    from gvwzz._native_pairs import (minimal_frozen_parents, minimal_frozen_mask,
                                     commit_frozen_parents, emit_unit_current_pairs)
except ImportError:
    minimal_frozen_parents = None


def brute(rows, current, parents):
    shifts = {i: tuple(max(a - b, 0) for a, b in zip(rows[i], current)) for i in parents}
    return sorted(i for i, u in shifts.items() if not any(
        all(x <= y for x, y in zip(v, u)) and (v != u or j < i)
        for j, v in shifts.items() if j != i))


def engine_for(rows, current):
    return SimpleNamespace(backend=SimpleNamespace(names=range(len(current))),
        frozen_count=len(rows), leading=[(a, 1) for a in rows] + [(current, 1)])


@unittest.skipIf(minimal_frozen_parents is None, 'optional native pair accelerator unavailable')
class NativePowerChecks(unittest.TestCase):
    def test_random_powers_dimensions_masks_and_earliest_duplicates(self):
        rng = Random(220151)
        for nvars in (0, 1, 3, 51, 65, 129):
            for _ in range(20):
                def monomial():
                    out = [0] * nvars
                    for k in rng.sample(range(nvars), rng.randrange(min(nvars, 7) + 1)):
                        out[k] = rng.choice((1, 2, 7, 64, 255, 256, 65536, 2**32 - 1))
                    return tuple(out)
                rows = [monomial() for _ in range(70)]
                rows.extend(rows[:10])
                current = monomial()
                parents = rng.sample(range(len(rows)), 60)
                engine = engine_for(rows, current)
                expected = brute(rows, current, parents)
                with self.subTest(nvars=nvars, current=current):
                    self.assertEqual(minimal_frozen_parents(engine, len(rows), parents), expected)
                    self.assertEqual(minimal_frozen_mask(engine, len(rows),
                        sum(1 << i for i in parents)), expected)

    def test_nontrivial_sparse_antichains_and_single_variable_dominance(self):
        rng = Random(220152)
        for nvars in (3, 51, 129):
            for _ in range(15):
                current = tuple(rng.randrange(3) for _ in range(nvars))
                rows = []
                for _ in range(100):
                    shift = [0] * nvars
                    for k in rng.sample(range(nvars), rng.randint(2, min(nvars, 5))):
                        shift[k] = rng.randrange(1, 8)
                    rows.append(tuple(a + b for a, b in zip(current, shift)))
                # Single-variable exponent witnesses remove some, not all,
                # nonsingletons; shuffled duplicates check earliest tie IDs.
                for k in range(min(4, nvars)):
                    shift = [0] * nvars
                    shift[k] = 3
                    rows.append(tuple(a + b for a, b in zip(current, shift)))
                    shift[k] = 5
                    rows.append(tuple(a + b for a, b in zip(current, shift)))
                rows += rows[-4:]
                parents = list(range(len(rows)))
                rng.shuffle(parents)
                engine = engine_for(rows, current)
                expected = brute(rows, current, parents)
                self.assertEqual(minimal_frozen_parents(engine, len(rows), parents), expected)
                self.assertEqual(minimal_frozen_mask(engine, len(rows),
                    (1 << len(rows)) - 1), expected)

    def test_unbounded_fallback_and_invalid_parent_indices(self):
        for rows, current in (([(2**32, 0)], (1, 0)), ([(1, 0)], (10**100, 0))):
            engine = engine_for(rows, current)
            self.assertIsNone(minimal_frozen_parents(engine, len(rows), [0]))
            self.assertIsNone(minimal_frozen_mask(engine, len(rows), 1))
        engine = engine_for([(1, 0)], (0, 2))
        for parents in ([-1], [1], [10**100]):
            with self.assertRaises(IndexError):
                minimal_frozen_parents(engine, 1, parents)
        for mask in (-1, 2, 1 << 100):
            with self.assertRaises(ValueError):
                minimal_frozen_mask(engine, 1, mask)
        self.assertEqual(minimal_frozen_parents(engine, 1, []), [])
        self.assertEqual(minimal_frozen_mask(engine, 1, 0), [])

    def test_general_power_covers_both_upper_parents_and_exponent_boundaries(self):
        for upper in (1, 2):
            for y_power in (2, 3, 4):
                for commit in (False, True):
                    backend = IntegerPolynomialRing(('x', 'y'), 'grevlex')
                    inputs = [backend.parse(f) for f in ('x^2*y^3', 'x^2', f'x^2*y^{y_power}')]
                    if upper == 2:
                        inputs[1], inputs[2] = inputs[2], inputs[1]
                    order = ModuleOrder(backend, inputs, 'pot')
                    native, original = (GVWInteger(backend, inputs, order, Options()) for _ in range(2))
                    for engine in (native, original):
                        engine.frozen_count = 1
                        engine._skip_principal_pair = lambda i, j: i == 0 or j == 0
                        for i, polynomial in enumerate(inputs):
                            engine._insert_basis(Label(Signature(1, (0, 0), min(i, 1)), polynomial))
                        engine.S.insert_and_complete(Signature(1, (2, 3), 1))
                        engine._emit_pair(0, upper)
                    self.assertEqual(minimal_frozen_mask(native, upper, 1), [0])
                    if commit:
                        commit_frozen_parents(native, upper)
                    emit_unit_current_pairs(native, 2, 1)
                    original._emit_pair(1, 2)
                    with self.subTest(upper=upper, y_power=y_power, commit=commit):
                        self.assertEqual(native.stats.counters['j_current_dominance_rejected'],
                                         int(commit and y_power >= 3))
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
