"""Exact comparisons for the optional native sparse reduction kernel."""
import random
import unittest

from gvwzz import Options
from gvwzz.core import Label, Signature
from gvwzz.orders import ModuleOrder
from gvwzz.polynomial import IntegerPolynomialRing, PackedContext
from gvwzz.solver import GVWInteger
import gvwzz.solver as solver_module

try:
    from gvwzz._native_reduction import _Context, NativeWorkspace
except ImportError:
    _Context = None


@unittest.skipIf(_Context is None or solver_module._native_reduce_working is None,
                 'optional native extension is disabled or not built')
class NativeReductionChecks(unittest.TestCase):
    def test_exact_codec_across_byte_boundaries_and_large_exponents(self):
        rng = random.Random(20261015)
        for order in ('grlex', 'grevlex'):
            for dimensions in (1, 2, 7, 8, 9, 48, 49):
                backend = IntegerPolynomialRing(['x' + str(i) for i in range(dimensions)], order)
                for bits in (1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 80):
                    native = _Context(backend, bits)
                    reference = PackedContext(backend, bits)
                    for _ in range(10):
                        monomial = [0] * dimensions
                        remaining = (1 << bits) - 1
                        for _ in range(5):
                            exponent = rng.randrange(remaining + 1)
                            monomial[rng.randrange(dimensions)] += exponent
                            remaining -= exponent
                        monomial = tuple(monomial)
                        encoded = native.encode(monomial)
                        self.assertEqual(encoded, reference.encode(monomial))
                        self.assertEqual(native.decode(encoded), monomial)

    def test_workspace_keeps_unbounded_signed_integer_coefficients(self):
        for order in ('grlex', 'grevlex'):
            backend = IntegerPolynomialRing(['x', 'y'], order)
            x, y = backend.gens
            huge = (1 << 4000) + 51
            p = huge * x**3 + (1 - huge) * x * y + 9
            q = x**2 + 3 * y
            work = NativeWorkspace(backend, p)
            work.subtract_multiple(q, huge, (1, 0))
            expected = p - backend.multiply_term(q, huge, (1, 0))
            self.assertEqual(work.result(), expected)
            while work.leading() is not None:
                work.finish_term()
            self.assertEqual(work.result(), expected)

    def test_all_reducer_policies_match_python_without_tracking(self):
        native_find = solver_module._native_find_reducer
        native_work = solver_module._native_reduce_working
        try:
            for selection in ('oldest', 'sparse', 'unit', 'height', 'balanced', 'newest'):
                for policy in ('regular-first', 'singular-first', 'first'):
                    for priority in ('exact-first', 'signature-first'):
                        options = Options(reducer_selection=selection,
                            reducer_policy=policy, euclidean_priority=priority,
                            max_seconds=10, max_reduction_steps=100000)
                        outputs = []
                        for native in (True, False):
                            solver_module._native_find_reducer = native_find if native else None
                            solver_module._native_reduce_working = native_work if native else None
                            backend = IntegerPolynomialRing(['x', 'y'], 'grevlex')
                            inputs = [backend.parse(p) for p in
                                      ('2*x^2+y', '3*x*y+1', '5*y^2+x')]
                            order = ModuleOrder(backend, inputs, 'pot', positions=[1, 0, 2])
                            result = GVWInteger(backend, inputs, order, options).run()
                            self.assertEqual(result.status, 'completed')
                            outputs.append(([(x.signature, x.polynomial) for x in result.basis],
                                            result.syzygies))
                        self.assertEqual(outputs[0], outputs[1], (selection, policy, priority))
        finally:
            solver_module._native_find_reducer = native_find
            solver_module._native_reduce_working = native_work

    def test_packed_pot_shortcut_preserves_reduction_choices(self):
        native_find = solver_module._native_find_reducer
        native_work = solver_module._native_reduce_working
        shortcuts = 0
        try:
            for selection in ('oldest', 'sparse', 'unit', 'height', 'balanced', 'newest'):
                for policy in ('regular-first', 'singular-first', 'first'):
                    for coefficient in (1, 6, -(1 << 300)):
                        outputs = []
                        for native in (True, False):
                            solver_module._native_find_reducer = native_find if native else None
                            solver_module._native_reduce_working = native_work if native else None
                            backend = IntegerPolynomialRing(['x', 'y'], 'grevlex')
                            inputs = [backend.parse(p) for p in ('x+y', '3*y+1', 'x^4')]
                            engine = GVWInteger(backend, inputs, ModuleOrder(backend, inputs, 'pot'),
                                Options(reducer_selection=selection, reducer_policy=policy))
                            for i in (0, 1):
                                engine._insert_basis(Label(Signature(1, (0, 0), i), inputs[i]))
                            polynomial = coefficient * backend.parse('x^4+x^2*y+2')
                            result = engine._reduce_working(Label(Signature(1, (0, 0), 2), polynomial))
                            counters = engine.stats.counters
                            shortcuts += counters['native_frozen_reducer_shortcuts']
                            outputs.append((result.signature, result.polynomial,
                                tuple(counters[name] for name in
                                      ('regular_steps', 'singular_steps', 'euclidean_steps'))))
                        self.assertEqual(outputs[0], outputs[1], (selection, policy, coefficient))
            self.assertGreater(shortcuts, 0)
        finally:
            solver_module._native_find_reducer = native_find
            solver_module._native_reduce_working = native_work


if __name__ == '__main__':
    unittest.main()
