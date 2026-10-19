"""Post20 exact queue/cover accelerator regression checks."""
import random
import unittest
from math import gcd
from unittest.mock import patch

from gvwzz.core import Label, Signature
from gvwzz.monomial_packing import pack_monomial
from gvwzz.orders import ModuleOrder
from gvwzz.polynomial import IntegerPolynomialRing
from gvwzz.structures import CandidateQueue, CoverBucket, CoverIndex, Stats, SyzygyStore


def setup(order='grevlex', nvars=2):
    ring = IntegerPolynomialRing([f'x{i}' for i in range(nvars)], order)
    inputs = [ring.one, ring.one]
    return ring, ModuleOrder(ring, inputs, 'pot')


class QueueCoverChecks(unittest.TestCase):
    def test_prefix_gcd_breakpoints_match_all_strict_cutoffs(self):
        rng = random.Random(204621)
        bucket = CoverBucket()
        records = []
        # Nonunits first exercise gcds formed across multiple keys, updates at
        # an existing key, redundant interior keys, and a late unit insertion.
        for _ in range(500):
            key, coeff = (rng.randrange(-15, 16),), rng.choice([6, 10, 14, 21, 35, 77, 2**130*3])
            records.append((key, coeff))
            bucket.insert(key, -coeff)
            for cutoff in range(-16, 18):
                expected = 0
                for k, c in records:
                    if k < (cutoff,):
                        expected = gcd(expected, c)
                self.assertEqual(bucket.gcd_below((cutoff,)), expected)
            self.assertTrue(all(a != b for a, b in zip(bucket.prefix, bucket.prefix[1:])))
        bucket.insert((-20,), 1)
        self.assertEqual(bucket.keys, [(-20,)])
        self.assertEqual(bucket.gcd_below((-20,)), 0)
        self.assertEqual(bucket.gcd_below((-19,)), 1)

    def test_indexed_cover_matches_naive_signed_ratio_keys(self):
        rng = random.Random(621052)
        for order in ('lex', 'grlex', 'grevlex'):
            b, mo = setup(order)
            syz = SyzygyStore(2, 2, Stats())
            indexed = CoverIndex(b, mo, syz, Stats(), cache_size=0)
            naive = CoverIndex(b, mo, syz, Stats(), mode='naive')
            for step in range(100):
                m = tuple(rng.randrange(5) for _ in range(2))
                sig = Signature(rng.choice([2, 3, 6, -10, 15]), m, rng.randrange(2))
                if step % 9 == 0:
                    syz.insert_and_complete(sig)
                else:
                    lm = tuple(rng.randrange(5) for _ in range(2))
                    label = Label(sig, b.one)
                    indexed.insert(label, lm)
                    naive.insert(label, lm)
                for _ in range(5):
                    q = Signature(rng.choice([-35, -7, 1, 3, 10, 42]),
                                  tuple(rng.randrange(6) for _ in range(2)), rng.randrange(2))
                    W = tuple(rng.randrange(6) for _ in range(2))
                    self.assertEqual(indexed.check(q, W), naive.check(q, W))

    def test_packed_and_tuple_queue_interoperate(self):
        b, mo = setup()
        for tie in ('fifo', 'small-coeff', 'small-lm'):
            q = CandidateQueue(b, mo, Stats(), tie_policy=tie)
            sig = Signature(-6, (1, 2), 1)
            m = pack_monomial(sig.monomial)
            self.assertTrue(q.push_j(sig, (4, 1), lambda: b.one))
            self.assertFalse(q._accept_j_packed(-6, m, 1, pack_monomial((4, 1))))
            self.assertTrue(q._accept_j_packed(-6, m, 1, pack_monomial((2, 1))))
            q._insert('J', sig, (2, 1), factory=lambda: b.one, dedup_key=(-6, m, 1))
            # The exact signed coefficient remains part of deduplication.
            self.assertTrue(q.push_j(Signature(6, (1, 2), 1), (2, 1), lambda: b.one))
            first = q.pop()
            self.assertEqual(first.signature, sig)
            q.settle_j(first)
            self.assertFalse(q.push_j(sig, (3, 1), lambda: b.one))
            self.assertFalse(q._accept_j_packed(-6, m, 1, pack_monomial((3, 1))))
            self.assertEqual(q.stats.counters['j_dedup_settled'], 2)

    def test_packed_dedup_precedes_filter_and_replacement(self):
        b, mo = setup()
        q = CandidateQueue(b, mo, Stats())
        sig = Signature(1, (0, 0), 0)
        q.push_j(sig, (3, 0), lambda: b.one)
        q.generation_filter_packed = lambda *args: True
        self.assertFalse(q._accept_j_packed(1, 0, 0, pack_monomial((2, 0))))
        self.assertFalse(q.active)
        self.assertFalse(q.pending_j)
        self.assertEqual(q.stats.counters['j_dedup_replaced'], 1)
        q.generation_filter_packed = None
        q.push_j(sig, (1, 0), lambda: b.one)
        q.settle_j(q.pop())
        q.generation_filter_packed = lambda *args: self.fail('settled bound must run first')
        self.assertFalse(q._accept_j_packed(1, 0, 0, pack_monomial((2, 0))))

    def test_new_packed_rejection_needs_no_decode_or_signature(self):
        b, mo = setup(nvars=48)
        q = CandidateQueue(b, mo, Stats(), generation_filter_packed=lambda *args: True)
        with patch('gvwzz.structures.unpack_monomial', side_effect=AssertionError('decode')):
            with patch('gvwzz.structures.Signature', side_effect=AssertionError('signature allocation')):
                self.assertFalse(q._accept_j_packed(1, 1, 1, 2))

    def test_settle_reuses_admission_key(self):
        b, mo = setup()
        q = CandidateQueue(b, mo, Stats(), tie_policy='small-lm')
        q.push_j(Signature(1, (0, 0), 0), (2, 1), lambda: b.one)
        with patch.object(b, 'order_key', side_effect=AssertionError('recomputed immutable key')):
            q.settle_j(q.pop())

    def test_packed_cover_and_checked_syzygy_match_naive_signed_coefficients(self):
        rng = random.Random(21931)
        for order in ('lex', 'grlex', 'grevlex'):
            for nvars in (3, 48):
                b, mo = setup(order, nvars)
                syz = SyzygyStore(nvars, 2, Stats())
                indexed = CoverIndex(b, mo, syz, Stats(), cache_size=0)
                naive = CoverIndex(b, mo, syz, Stats(), mode='naive')
                def vector(bound):
                    return tuple(rng.randrange(bound) if k < 3 else 0 for k in range(nvars))
                for step in range(80):
                    m = vector(6)
                    if step % 19 == 0:
                        m = (2**100,) + m[1:]
                    sig = Signature(rng.choice([-10, -3, 1, 2, 6, 15]), m, rng.randrange(2))
                    if step % 7 == 0:
                        syz.insert_and_complete(sig)
                    else:
                        label, lm = Label(sig, b.one), vector(5)
                        indexed.insert(label, lm)
                        naive.insert(label, lm)
                    for _ in range(6):
                        q = Signature(rng.choice([-21, -7, 1, 2, 5, 6, 30]), vector(7), rng.randrange(2))
                        W = vector(7)
                        packed = pack_monomial(q.monomial)
                        checked = not syz._has_divisor_packed(q.coefficient, packed, q.index)
                        expected = naive.check(q, W)
                        self.assertEqual(indexed.check(q, W, packed_monomial=packed,
                                                       unit_syzygy_checked=checked), expected)
                        if step % 13 == 0:
                            with patch('gvwzz.structures._native_divisor_mask_packed', None):
                                self.assertEqual(indexed.check(q, W, packed_monomial=packed,
                                                               unit_syzygy_checked=checked), expected)

    def test_checked_unit_proof_skips_only_duplicate_unit_existence(self):
        b, mo = setup()
        syz = SyzygyStore(2, 2, Stats())
        syz.insert_and_complete(Signature(1, (2, 0), 0))
        cover = CoverIndex(b, mo, syz, Stats(), cache_size=0)
        cover.insert(Label(Signature(1, (0, 0), 0), b.one), (1, 0))
        q = Signature(1, (1, 0), 0)
        self.assertFalse(syz._has_divisor_packed(1, 1, 0))
        with patch.object(syz, 'has_divisor', side_effect=AssertionError('duplicate S query')):
            self.assertTrue(cover.check(q, (3, 0), packed_monomial=1, unit_syzygy_checked=True))
        # The negative single-S-divisor result does not allow skipping a
        # nonunit S coefficient needed in a combined S/P gcd cover.
        syz = SyzygyStore(2, 2, Stats())
        syz.insert_and_complete(Signature(6, (0, 0), 0))
        cover = CoverIndex(b, mo, syz, Stats(), cache_size=0)
        cover.insert(Label(Signature(10, (0, 0), 0), b.one), (1, 0))
        q = Signature(2, (1, 0), 0)
        self.assertFalse(syz._has_divisor_packed(2, 1, 0))
        self.assertTrue(cover.check(q, (3, 0), packed_monomial=1, unit_syzygy_checked=True))

    def test_exponent_overflow_uses_exact_tuple_fallback(self):
        b, mo = setup()
        q = CandidateQueue(b, mo, Stats())
        sig = Signature(2**180, (128, 2**80), 1)
        self.assertTrue(q.push_j(sig, (2, 1), lambda: b.one))
        self.assertFalse(q.push_j(sig, (3, 1), lambda: b.one))
        q.settle_j(q.pop())
        self.assertFalse(q.push_j(sig, (2, 1), lambda: b.one))


if __name__ == '__main__':
    unittest.main(verbosity=2)
