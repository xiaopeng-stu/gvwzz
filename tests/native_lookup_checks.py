"""Exact policy-oracle checks for packed signature keys and cached divisor lookup."""
from dataclasses import replace
import random
import unittest
from gvwzz import Options
from gvwzz.core import Label, Signature
from gvwzz.orders import ModuleOrder
from gvwzz.polynomial import IntegerPolynomialRing
from gvwzz.solver import GVWInteger
import gvwzz.solver as solver_module
try:
    from gvwzz._native_reduction import _PackedRatioContext
except ImportError:
    _PackedRatioContext = None

@unittest.skipIf(solver_module._native_find_reducer is None or _PackedRatioContext is None, 'native kernel unavailable')
class NativeLookupChecks(unittest.TestCase):
    def test_all_orders_policies_and_append_boundaries(self):
        rng = random.Random(20261016)
        native = solver_module._native_find_reducer
        checked = 0
        try:
            for poly_order in ('grevlex', 'grlex', 'lex'):
                for module in ('pot', 'top', 'g1', 'g2', 'schreyer', 'block'):
                    if poly_order == 'lex' and module == 'g1':
                        continue
                    for selection in ('oldest', 'sparse', 'unit', 'height', 'balanced', 'newest'):
                        backend = IntegerPolynomialRing(['x', 'y'], poly_order)
                        inputs = [backend.parse(x) for x in ('x^2+y', '3*x*y+1', '5*y^2+x')]
                        order = ModuleOrder(backend, inputs, module, positions=[2, 0, 1],
                            block_sizes=[2, 1] if module == 'block' else None)
                        engine = GVWInteger(backend, inputs, order, Options(reducer_selection=selection))
                        for row in range(81):
                            monomial = (rng.randrange(6), rng.randrange(6))
                            coefficient = rng.choice([1, -1, 2, -3, 5, (1 << 150) + 1])
                            polynomial = backend.ring.from_dict({monomial: coefficient, (0, 0): 7})
                            sig = Signature(rng.choice([1, -2, 3, -(1 << 160)]),
                                (rng.randrange(5), rng.randrange(5)), rng.randrange(3))
                            engine._insert_basis(Label(sig, polynomial))
                            if row not in (0, 7, 31, 63, 80):
                                continue
                            for policy in ('regular-first', 'singular-first', 'first'):
                                engine.options = replace(engine.options, reducer_policy=policy)
                                for priority in ('exact-first', 'signature-first'):
                                    engine.options = replace(engine.options, euclidean_priority=priority)
                                    for mode, euclidean in (('strict', False), ('mixed', False), ('mixed', True)):
                                        sig = Signature(rng.choice([1, -2, 3, (1 << 220) + 3]),
                                            (rng.randrange(9), rng.randrange(9)), rng.randrange(3))
                                        leading = ((rng.randrange(8), rng.randrange(8)),
                                            rng.choice([0, 1, -1, 2, -5, (1 << 1000) + 12]))
                                        engine.reducer_query_cache.clear()
                                        context = _PackedRatioContext(engine, sig, sum(leading[0]))
                                        actual = native(engine, sig, leading, mode, euclidean, False, context)
                                        engine.reducer_query_cache.clear()
                                        solver_module._native_find_reducer = None
                                        expected = engine._find_reducer(sig, leading, mode, euclidean)
                                        solver_module._native_find_reducer = native
                                        self.assertEqual(actual, expected, (poly_order, module, selection, row, policy, priority, mode))
                                        checked += 1
        finally:
            solver_module._native_find_reducer = native
        self.assertEqual(checked, 9180)

if __name__ == '__main__':
    unittest.main()
