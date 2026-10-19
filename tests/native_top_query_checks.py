"""Native generation-top existence against an independent complete-basis scan."""
from collections import Counter
import os
from random import Random
from types import SimpleNamespace
import unittest

from gvwzz.core import Label, Signature
from gvwzz.monomial_packing import pack_monomial
from gvwzz.orders import ModuleOrder
from gvwzz.polynomial import IntegerPolynomialRing

try:
    if os.environ.get("GVWZZ_NO_NATIVE"):
        raise ImportError("native acceleration disabled")
    from gvwzz._native_pairs import NativeTopQuery
except ImportError:
    NativeTopQuery = None


def engine_for(weights, positions=None, module_order="g2", poly_order="grevlex"):
    backend = IntegerPolynomialRing(tuple(f"x{k}" for k in range(len(weights[0]))), poly_order)
    inputs = [backend.ring.from_dict({weight: 1}) for weight in weights]
    return SimpleNamespace(
        backend=backend, order=ModuleOrder(backend, inputs, module_order, positions),
        P=[], leading=[], stats=SimpleNamespace(counters=Counter()),
    )


def append(engine, leading, coefficient, signature, component=0):
    engine.P.append(Label(Signature(1, signature, component), None))
    engine.leading.append((leading, coefficient))


def ratio_key(engine, parent):
    signature = engine.P[parent].signature
    return engine.order.key_monomial(
        tuple(a - b for a, b in zip(signature.monomial, engine.leading[parent][0])),
        signature.index,
    )


def oracle(engine, target, coefficient, parent):
    target_ratio = ratio_key(engine, parent)
    for label, (monomial, scalar) in zip(engine.P, engine.leading):
        if not all(a <= b for a, b in zip(monomial, target)) or coefficient % scalar:
            continue
        ratio = engine.order.key_monomial(
            tuple(a - b for a, b in zip(label.signature.monomial, monomial)),
            label.signature.index,
        )
        if ratio <= target_ratio:
            return True
    return False


@unittest.skipIf(NativeTopQuery is None, "native pairs extension is unavailable")
class NativeTopQueryChecks(unittest.TestCase):
    def test_incremental_random_queries_match_all_basis_witnesses(self):
        rng = Random(250071)
        large = (1 << 4096) + 7
        for nvars in (1, 2, 7, 8, 9, 12, 16, 17, 32, 64, 129):
            with self.subTest(nvars=nvars):
                weights = [tuple(rng.randrange(11) for _ in range(nvars)) for _ in range(3)]
                engine = engine_for(weights, positions=(2, 0, 1))
                index = NativeTopQuery(engine)
                for size in (10, 30, 100):
                    while len(engine.P) < size:
                        append(engine,
                               tuple(rng.randrange(13) for _ in range(nvars)),
                               rng.choice((1, -1, 2, -3, 6, large, -large)),
                               tuple(rng.randrange(16) for _ in range(nvars)),
                               rng.randrange(3))
                    index.refresh(engine)
                    self.assertTrue(index.enabled)
                    for _ in range(120):
                        target = tuple(rng.randrange(20) for _ in range(nvars))
                        coefficient = rng.choice((1, -1, 2, -6, 12, large, -large * 6, large + 1))
                        parent = rng.randrange(size)
                        other = rng.randrange(size)
                        left, right = ratio_key(engine, parent), ratio_key(engine, other)
                        self.assertEqual(index.compare(parent, other), (left > right) - (left < right))
                        self.assertEqual(index.has_top(pack_monomial(target), coefficient, parent),
                                         oracle(engine, target, coefficient, parent))
                index.flush()
                self.assertEqual(engine.stats.counters["native_generation_top_queries"], 360)

    def test_unit_dominance_preserves_queries_after_witness_replacement(self):
        engine = engine_for([(0, 0)])
        append(engine, (4, 0), 1, (8, 0))
        index = NativeTopQuery(engine)
        append(engine, (2, 0), -1, (5, 0))
        index.refresh(engine)
        self.assertEqual(index.sizes()["unit_witnesses"], 1)
        append(engine, (1, 0), 1, (5, 0))
        index.refresh(engine)
        self.assertEqual(index.sizes()["unit_witnesses"], 2)
        for x in range(8):
            for parent in range(3):
                target = (x, 0)
                self.assertEqual(index.has_top(pack_monomial(target), 1, parent),
                                 oracle(engine, target, 1, parent))

    def test_parent_comparison_equal_weights_uses_component_rank(self):
        engine = engine_for([(3, 0), (0, 3)], positions=(1, 0))
        append(engine, (4, 0), 1, (1, 0), 0)
        append(engine, (0, 5), 7, (0, 2), 1)
        append(engine, (8, 0), -3, (5, 0), 0)
        index = NativeTopQuery(engine)
        self.assertEqual(index.compare(0, 2), 0)
        self.assertEqual(index.compare(0, 1), 1)
        self.assertEqual(index.compare(1, 2), -1)
        for i in range(3):
            for j in range(3):
                left, right = ratio_key(engine, i), ratio_key(engine, j)
                self.assertEqual(index.compare(i, j), (left > right) - (left < right))

    def test_nonunit_coefficients_keep_all_4097_bits(self):
        engine = engine_for([(0, 0)])
        large = (1 << 4096) + 3
        append(engine, (1, 0), large, (1, 0))
        append(engine, (1, 0), -large, (0, 0))
        index = NativeTopQuery(engine)
        self.assertEqual(index.sizes()["nonunit_witnesses"], 1)
        for coefficient in (large, -large * 5, large + 1, 1, -1):
            self.assertEqual(index.has_top(pack_monomial((1, 0)), coefficient, 0),
                             oracle(engine, (1, 0), coefficient, 0))

    def test_large_leading_coordinate_cannot_divide_packed_target(self):
        engine = engine_for([(127, 0), (128, 0)])
        append(engine, (128, 0), 1, (0, 0), 0)
        append(engine, (127, 0), 3, (126, 0), 1)
        index = NativeTopQuery(engine)
        self.assertTrue(index.enabled)
        for parent in (0, 1):
            for coefficient in (1, 3, -3):
                self.assertEqual(index.has_top(pack_monomial((127, 0)), coefficient, parent),
                                 oracle(engine, (127, 0), coefficient, parent))

    def test_unsupported_order_and_ratio_range_decline_without_narrowing(self):
        for order, nvars in (("pot", 2), ("pot", 17)):
            engine = engine_for([(0,) * nvars], module_order=order)
            self.assertFalse(NativeTopQuery(engine).enabled)
        engine = engine_for([(0, 0)])
        append(engine, (0, 0), 1, (1 << 200, 0))
        index = NativeTopQuery(engine)
        self.assertFalse(index.enabled)
        with self.assertRaises(ValueError):
            index.has_top(0, 1, 0)

    def test_dynamic_word_tails_control_divisibility_and_nonunit_identity(self):
        huge = (1 << 4097) + 3
        for nvars in (17, 32, 64, 129):
            zero = (0,) * nvars
            engine = engine_for([zero])
            for coordinate, coefficient in ((0, 1), (16, 3), (nvars - 1, huge)):
                leading = [0] * nvars
                leading[coordinate] = 2
                signature = leading.copy()
                signature[-1] += 1
                append(engine, tuple(leading), coefficient, tuple(signature))
            index = NativeTopQuery(engine)
            self.assertTrue(index.enabled)
            self.assertEqual(index.sizes()["nonunit_witnesses"], 2)
            for coordinate in (0, 8, 16, nvars - 1):
                for exponent in (0, 1, 2, 127):
                    target = [0] * nvars
                    target[coordinate] = exponent
                    target = tuple(target)
                    for coefficient in (1, 3, huge, -3 * huge, huge + 1):
                        for parent in range(3):
                            self.assertEqual(
                                index.has_top(pack_monomial(target), coefficient, parent),
                                oracle(engine, target, coefficient, parent))

    def test_checked_query_rejects_overflow_and_reserved_lanes(self):
        engine = engine_for([(0, 0)])
        append(engine, (0, 0), 1, (0, 0))
        index = NativeTopQuery(engine)
        for target in (-1, 1 << 64, 128, 1 << 16):
            with self.assertRaises(ValueError):
                index.has_top(target, 1, 0)
        with self.assertRaises(IndexError):
            index.has_top(0, 1, 1)


if __name__ == "__main__":
    unittest.main()
