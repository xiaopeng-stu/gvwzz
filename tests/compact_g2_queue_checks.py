"""Exact weighted G2 keys, packed admission, and unbounded queue transitions."""
import random
import unittest

from gvwzz.core import Signature
from gvwzz.monomial_packing import pack_monomial
from gvwzz.orders import ModuleOrder
from gvwzz.polynomial import IntegerPolynomialRing
from gvwzz.structures import CandidateQueue, Stats
from tests import queue_scalar_checks


class CompactG2QueueChecks(queue_scalar_checks.ScalarQueueChecks):
    """Run every original scalar-queue oracle with nontrivial G2 weights."""

    def make_pair(self, tie='fifo', dedup=True, nvars=5):
        b = IntegerPolynomialRing([f'x{i}' for i in range(nvars)], 'grevlex')
        weights = []
        for k, exponent in enumerate((127, 2, 4)):
            m = [0] * nvars
            m[k % nvars] = exponent
            weights.append(b.ring.from_dict({tuple(m): 1}))
        mo = ModuleOrder(b, weights, 'g2', positions=[2, 0, 1])
        fast = CandidateQueue(b, mo, Stats(), tie_policy=tie, deduplicate=dedup)
        plain = CandidateQueue(b, mo, Stats(), tie_policy=tie, deduplicate=dedup)
        plain._disable_compact_order()
        self.assertTrue(fast._compact_order)
        return b, fast, plain

    def test_weighted_signature_scalar_order_and_reconstruction(self):
        rng = random.Random(250341)
        for nvars in (1, 3, 9, 137):
            _, queue, _ = self.make_pair(nvars=nvars)
            signatures = [Signature(rng.choice((-1, 1, 2**4097)),
                                    tuple(rng.randrange(128) for _ in range(nvars)),
                                    rng.randrange(3)) for _ in range(300)]
            compact = lambda sig: queue._packed_signature_key(
                pack_monomial(sig.monomial), sig.index)
            self.assertEqual(sorted(signatures, key=queue.order.key),
                             sorted(signatures, key=compact))
            for sig in signatures:
                self.assertEqual(queue._uncompact_signature_key(compact(sig)),
                                 queue.order.key(sig))

    def test_equal_weighted_monomials_use_permuted_position_rank(self):
        b = IntegerPolynomialRing(['x', 'y'], 'grevlex')
        mo = ModuleOrder(b, [b.parse('x^2'), b.parse('y^2'), b.parse('x^2')],
                         'g2', positions=[2, 0, 1])
        queue = CandidateQueue(b, mo, Stats())
        signatures = [Signature(1, (0, 2), 0), Signature(1, (2, 0), 1),
                      Signature(1, (0, 2), 2)]
        self.assertEqual([s.index for s in sorted(signatures, key=mo.key)], [2, 0, 1])
        for sig in signatures:
            queue.push_j(sig, (3, 3), lambda: b.one)
        self.assertEqual([queue.pop().signature.index for _ in range(3)], [2, 0, 1])

    def test_large_input_weight_and_other_polynomial_order_keep_exact_fallback(self):
        for poly_order, weight in (('grevlex', 128), ('grevlex', 2**100), ('grlex', 2)):
            b = IntegerPolynomialRing(['x', 'y'], poly_order)
            polynomial = b.ring.from_dict({(weight, 0): 1})
            mo = ModuleOrder(b, [polynomial, b.one], 'g2')
            queue = CandidateQueue(b, mo, Stats())
            self.assertFalse(queue._compact_order)
            queue._insert_j_packed(2**4097, 1, 0, 2, lambda: b.one)
            candidate = queue.pop()
            self.assertEqual(candidate.signature, Signature(2**4097, (1, 0), 0))
            self.assertEqual(candidate.leading_monomial, (2, 0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
