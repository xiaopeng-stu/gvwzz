"""Exact native P-cover proofs, strict cutoffs and arbitrary dimensions."""
import os
import random
import unittest
from unittest.mock import patch

from gvwzz.core import Label, Signature
from gvwzz.monomial_packing import pack_monomial
from gvwzz.orders import ModuleOrder
from gvwzz.polynomial import IntegerPolynomialRing
from gvwzz.structures import CoverIndex, Stats, SyzygyStore

try:
    if os.environ.get('GVWZZ_NO_NATIVE'):
        raise ImportError('native disabled')
    from gvwzz._native_cover import NativeCoverQuery
except ImportError:
    NativeCoverQuery = None


def make_indices(nvars, order='grevlex'):
    ring = IntegerPolynomialRing([f'x{i}' for i in range(nvars)], order)
    module = ModuleOrder(ring, [ring.one, ring.one], 'pot')
    syzygies = SyzygyStore(nvars, 2, Stats())
    return (ring, syzygies, CoverIndex(ring, module, syzygies, Stats()),
            CoverIndex(ring, module, syzygies, Stats(), mode='naive'))


class PackedCoverChecks(unittest.TestCase):
    def test_packed_full_check_retains_mixed_syzygy_gcd(self):
        ring, syzygies, cover, naive = make_indices(3)
        syzygies.insert_and_complete(Signature(6, (0, 0, 0), 0))
        label = Label(Signature(10, (0, 0, 0), 0), ring.one)
        for index in (cover, naive):
            index.insert(label, (1, 0, 0))
        q = Signature(-2, (0, 0, 1), 0)
        W = (3, 0, 1)
        self.assertTrue(cover.check_packed(q.coefficient, pack_monomial(q.monomial),
                                         q.index, pack_monomial(W), unit_syzygy_checked=True))
        self.assertEqual(cover.stats.counters['cover_calls'], 1)
        self.assertTrue(naive.check(q, W))
        if cover.native is not None:
            self.assertEqual(cover.native.hits, 0)

    def test_large_stored_exponent_uses_complete_exact_fallback(self):
        ring, syzygies, cover, naive = make_indices(3)
        label = Label(Signature(1, (0, 0, 0), 0), ring.one)
        for index in (cover, naive):
            index.insert(label, (128, 0, 0))
        q, W = Signature(1, (0, 0, 0), 0), (100, 100, 0)
        self.assertTrue(cover.check_packed(1, 0, 0, pack_monomial(W)))
        self.assertTrue(naive.check(q, W))
        if cover.native is not None:
            self.assertEqual(cover.native.rows, 0)

    def test_full_packed_query_matches_naive_in_high_coordinates(self):
        rng = random.Random(280034)
        for nvars in (1, 9, 17, 33, 65, 129):
            for order in ('lex', 'grlex', 'grevlex'):
                ring, syzygies, cover, naive = make_indices(nvars, order)
                coordinates = sorted({0, nvars // 2, nvars - 1})
                def vector(bound):
                    result = [0] * nvars
                    for k in coordinates:
                        result[k] = rng.randrange(bound)
                    return tuple(result)
                for step in range(35):
                    m, lm = vector(4), vector(5)
                    coefficient = rng.choice([-30, -6, 2, 3, 5, 14, 2**130 * 3])
                    sig = Signature(coefficient, m, rng.randrange(2))
                    if step % 11 == 0:
                        syzygies.insert_and_complete(sig)
                    else:
                        label = Label(sig, ring.one)
                        cover.insert(label, lm)
                        naive.insert(label, lm)
                    for _ in range(8):
                        q = Signature(rng.choice([-30, -2, 1, 6, 14, 2**150]),
                                      vector(7), rng.randrange(2))
                        W = vector(8)
                        actual = cover.check_packed(q.coefficient, pack_monomial(q.monomial),
                                                    q.index, pack_monomial(W))
                        self.assertEqual(actual, naive.check(q, W),
                                         (nvars, order, step, q, W))


@unittest.skipIf(NativeCoverQuery is None, 'native cover extension unavailable')
class NativeCoverChecks(unittest.TestCase):
    def test_gcd_proof_is_generalized_and_strict_in_129_variables(self):
        native = NativeCoverQuery(129, 2, 'grevlex')
        a, b = [0] * 129, [0] * 129
        a[63], b[128] = 1, 1
        la, lb = a[:], b[:]
        la[10], lb[11] = 2, 2
        native.insert(6, tuple(a), 0, tuple(la))
        native.insert(-10, tuple(b), 0, tuple(lb))
        m = [max(x, y) for x, y in zip(a, b)]
        W = m[:]
        W[20] = 3
        self.assertTrue(native.positive(-2, pack_monomial(tuple(m)), 0, pack_monomial(tuple(W))))
        self.assertEqual(native.witness_hits, 0)
        m[100] += 1
        W[100] += 1
        self.assertTrue(native.positive(14, pack_monomial(tuple(m)), 0, pack_monomial(tuple(W))))
        self.assertEqual(native.witness_hits, 1)
        # The stored proof needs both original divisors, including variable 129.
        m[128], W[128] = 0, 0
        self.assertFalse(native.positive(2, pack_monomial(tuple(m)), 0, pack_monomial(tuple(W))))
        m[128], W[128] = 1, 1
        # At an equal greatest delta the strict cover bound must fail.
        W = m[:]
        W[10] += 2
        self.assertFalse(native.positive(2, pack_monomial(tuple(m)), 0, pack_monomial(tuple(W))))
        self.assertFalse(native.positive(2, pack_monomial(tuple(m)), 1, pack_monomial(tuple(W))))

    def test_signed_arbitrary_precision_coefficients_are_never_truncated(self):
        native = NativeCoverQuery(33, 1, 'lex')
        zero, lead = (0,) * 33, (1,) + (0,) * 32
        factor = (1 << 4096) + 1
        native.insert(-6 * factor, zero, 0, lead)
        native.insert(10 * factor, zero, 0, lead)
        bound = pack_monomial((2,) + (0,) * 32)
        self.assertTrue(native.positive(-2 * factor, 0, 0, bound))
        self.assertTrue(native.positive(14 * factor, 0, 0, bound))
        self.assertFalse(native.positive(2 * factor + 1, 0, 0, bound))
        self.assertFalse(native.positive(factor, 0, 0, bound))

    def test_machine_word_boundary_and_inexact_coefficient_rejection(self):
        for coefficient in (2**64 - 1, 2**64, 2**64 + 1):
            native = NativeCoverQuery(1, 1, 'grlex')
            native.insert(coefficient, (0,), 0, (1,))
            self.assertTrue(native.positive(-3 * coefficient, 0, 0, 2))
            self.assertFalse(native.positive(coefficient - 1, 0, 0, 2))
            self.assertFalse(native.positive(coefficient, 0, 0, 1))
        for coefficient in (1.5, True):
            native = NativeCoverQuery(1, 1, 'lex')
            with self.assertRaises(ValueError):
                native.insert(coefficient, (0,), 0, (0,))
            native.insert(1, (0,), 0, (0,))
            with self.assertRaises(ValueError):
                native.positive(coefficient, 0, 0, 1)

    def test_positive_cover_does_not_allocate_a_signature(self):
        ring, syzygies, cover, naive = make_indices(17)
        zero = (0,) * 17
        label = Label(Signature(1, zero, 0), ring.one)
        cover.insert(label, zero)
        with patch('gvwzz.structures.unpack_monomial', side_effect=AssertionError('decoded')):
            self.assertTrue(cover.check_packed(1, 0, 0, 1))
        self.assertEqual(cover.stats.counters['cover_calls'], 1)
        self.assertEqual(cover.stats.counters['native_cover_proofs'], 1)

    def test_portable_bitset_word_boundaries_and_insertion_thresholds(self):
        native = NativeCoverQuery(65, 1, 'grevlex')
        # More than two mask words and thresholds inserted out of order test
        # both inherited postings and short historical bit-mask vectors.
        for i in range(150):
            m = [0] * 65
            m[0], m[32], m[64] = i % 5, (i // 5) % 5, (i // 25) % 6
            lm = m[:]
            lm[1] = 1
            native.insert(3, tuple(m), 0, tuple(lm))
        q = [0] * 65
        q[0], q[32], q[64] = 4, 4, 5
        W = q[:]
        W[1] = 2
        self.assertTrue(native.positive(3, pack_monomial(tuple(q)), 0, pack_monomial(tuple(W))))
        self.assertFalse(native.positive(2, pack_monomial(tuple(q)), 0, pack_monomial(tuple(W))))


if __name__ == '__main__':
    unittest.main(verbosity=2)
