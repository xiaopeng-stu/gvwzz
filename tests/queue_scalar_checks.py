"""Exact scalar queue ordering, overflow transitions and packed proof checks."""
import random
import unittest
from unittest.mock import patch

from gvwzz.core import Signature
from gvwzz.monomial_packing import pack_monomial
from gvwzz.orders import ModuleOrder
from gvwzz.polynomial import IntegerPolynomialRing
from gvwzz.structures import CandidateQueue, PackedCandidate, Stats


class ScalarQueueChecks(unittest.TestCase):
    def assert_logical_queue_counts_equal(self, fast, plain):
        # Path counters intentionally report whether the compiled heap ran.
        normalize = lambda q: {k: v for k, v in q.stats.counters.items()
                               if not k.startswith('native_queue_')}
        self.assertEqual(normalize(fast), normalize(plain))

    def make_pair(self, tie='fifo', dedup=True, nvars=5):
        b = IntegerPolynomialRing([f'x{i}' for i in range(nvars)], 'grevlex')
        mo = ModuleOrder(b, [b.one] * 3, 'pot', positions=[2, 0, 1])
        fast = CandidateQueue(b, mo, Stats(), tie_policy=tie, deduplicate=dedup)
        plain = CandidateQueue(b, mo, Stats(), tie_policy=tie, deduplicate=dedup)
        plain._disable_compact_order()
        return b, fast, plain

    def test_scalar_order_is_exact_for_random_sparse_and_dense_vectors(self):
        rng = random.Random(20261015)
        for n in (1, 2, 5, 48, 137):
            b, fast, _ = self.make_pair(nvars=n)
            vectors = [tuple(rng.randrange(128) for _ in range(n)) for _ in range(200)]
            vectors += [tuple(int(i == j) for i in range(n)) for j in range(n)]
            self.assertEqual(sorted(vectors, key=b.order_key),
                             sorted(vectors, key=lambda m: fast._packed_leading_key(pack_monomial(m))))

    def test_random_admission_pop_and_settlement_match_unbounded_queue(self):
        for tie in ('fifo', 'small-lm', 'small-coeff'):
            for dedup in (True, False):
                for overflow in (False, True):
                    with self.subTest(tie=tie, dedup=dedup, overflow=overflow):
                        rng = random.Random(2119)
                        b, fast, plain = self.make_pair(tie, dedup)
                        for step in range(1500):
                            m = tuple(rng.randrange(3) for _ in range(5))
                            W = tuple(rng.randrange(5) for _ in range(5))
                            if overflow and step == 600:
                                W = (2**100, 0, 1, 0, 0)
                            if overflow and step == 900:
                                m = (0, 0, 0, 0, 2**160)
                            sig = Signature(rng.choice([-6, -1, 1, 2, 6]), m, rng.randrange(3))
                            if step % 7 == 0:
                                for q in (fast, plain):
                                    q.push_g_lazy(sig, W, 3, lambda: b.one)
                            else:
                                self.assertEqual(fast.push_j(sig, W, lambda: b.one),
                                                 plain.push_j(sig, W, lambda: b.one))
                            if step % 11 == 0:
                                a, z = fast.pop(), plain.pop()
                                self.assertEqual((a.token, a.signature, a.leading_monomial),
                                                 (z.token, z.signature, z.leading_monomial))
                                if a.kind == 'J':
                                    fast.settle_j(a)
                                    plain.settle_j(z)
                        while fast:
                            a, z = fast.pop(), plain.pop()
                            self.assertEqual((a.token, a.signature, a.leading_monomial),
                                             (z.token, z.signature, z.leading_monomial))
                            if a.kind == 'J':
                                fast.settle_j(a)
                                plain.settle_j(z)
                        self.assertFalse(plain)
                        self.assert_logical_queue_counts_equal(fast, plain)
                        self.assertEqual(set(fast.settled_j_bounds), set(plain.settled_j_bounds))

    def test_overflow_preserves_pending_replacement_and_settled_bound(self):
        b, fast, plain = self.make_pair()
        sig = Signature(1, (1, 0, 0, 0, 0), 0)
        other = Signature(2, (0, 0, 0, 0, 0), 1)
        held = []
        for q in (fast, plain):
            q.push_j(sig, (1, 1, 0, 0, 0), lambda: b.one)
            held.append(q.pop())
            q.push_j(other, (3, 0, 0, 0, 0), lambda: b.one)
        for q, candidate in zip((fast, plain), held):
            q.push_j(Signature(3, (0, 0, 0, 0, 0), 2), (128, 0, 0, 0, 0), lambda: b.one)
            q.settle_j(candidate)
            self.assertFalse(q._accept_j_packed(1, pack_monomial(sig.monomial), 0,
                                               pack_monomial((2, 1, 0, 0, 0))))
            self.assertTrue(q.push_j(other, (2, 0, 0, 0, 0), lambda: b.one))
        self.assert_logical_queue_counts_equal(fast, plain)
        self.assertEqual(fast.settled_j_bounds, plain.settled_j_bounds)

    def test_known_syzygy_preserves_dedup_order_and_disabled_filter(self):
        b, q, _ = self.make_pair()
        sig = Signature(1, (0,) * 5, 0)
        q.push_j(sig, (3, 0, 0, 0, 0), lambda: b.one)
        q.generation_filter_packed = lambda *args: self.fail('proof bypasses search')
        self.assertFalse(q._accept_j_packed(1, 0, 0, 2, True))
        self.assertEqual(q.stats.counters['j_dedup_replaced'], 1)
        self.assertEqual(q.stats.counters['j_generation_tests'], 1)
        self.assertEqual(q.stats.counters['j_generation_syzygy_rejected'], 1)
        q.generation_filter_packed = None
        self.assertTrue(q._accept_j_packed(1, 0, 0, 2, True))

    def test_packed_admission_and_settlement_do_not_decode(self):
        b, q, _ = self.make_pair()
        def unused():
            self.fail('uncovered polynomial recipe executed')
        with patch('gvwzz.structures.unpack_monomial', side_effect=AssertionError('decode')):
            with patch('gvwzz.structures._signature_from_arithmetic', side_effect=AssertionError('signature')):
                candidate = q._insert_j_packed(-6, 1, 0, 3, unused)
                self.assertIsInstance(candidate, PackedCandidate)
                self.assertEqual(candidate._packed_signature, (-6, 1, 0))
                self.assertEqual(candidate._packed_leading, 3)
                q.settle_j(q.pop())
        self.assertIsNone(candidate._signature)
        self.assertIsNone(candidate._leading_monomial)
        self.assertEqual(q.stats.counters['candidate_signature_fast_builds'], 0)

    def test_packed_properties_are_cached_and_eager_mode_is_respected(self):
        b, q, _ = self.make_pair()
        calls = []
        def recipe():
            calls.append(1)
            return b.one
        candidate = q._insert_j_packed(2, 1, 0, 3, recipe)
        self.assertEqual(candidate.signature, Signature(2, (1, 0, 0, 0, 0), 0))
        self.assertIs(candidate.signature, candidate.signature)
        self.assertEqual(candidate.leading_monomial, (3, 0, 0, 0, 0))
        self.assertIs(candidate.leading_monomial, candidate.leading_monomial)
        self.assertIs(candidate.label, candidate.label)
        self.assertEqual(calls, [1])
        self.assertEqual(q.stats.counters['candidate_signature_fast_builds'], 1)
        q.lazy = False
        eager = q._insert_j_packed(2, 1, 1, 3, recipe)
        self.assertIsNotNone(eager._label)
        self.assertEqual(calls, [1, 1])

    def test_mixed_lazy_and_ordinary_candidates_match_through_overflow(self):
        for tie in ('fifo', 'small-lm', 'small-coeff'):
            for g_first in (True, False):
                b, fast, plain = self.make_pair(tie)
                fast.g_first = plain.g_first = g_first
                rng = random.Random(22193)
                for step in range(600):
                    m = tuple(rng.randrange(3) for _ in range(5))
                    W = tuple(rng.randrange(5) for _ in range(5))
                    sig = Signature(rng.choice([-6, -1, 1, 2, 6]), m, rng.randrange(3))
                    if step == 300:
                        sig = Signature(3, (128, 0, 0, 0, 0), 1)
                        for q in (fast, plain):
                            q.push_g_lazy(sig, (128, 0, 0, 0, 0), 3, lambda: b.one)
                    elif step % 7 == 0:
                        for q in (fast, plain):
                            q.push_g_lazy(sig, W, 3, lambda: b.one)
                    else:
                        pm, pw = pack_monomial(m), pack_monomial(W)
                        yes = fast._accept_j_packed(sig.coefficient, pm, sig.index, pw)
                        self.assertEqual(yes, plain._accept_j(sig, W))
                        if yes:
                            fast._insert_j_packed(sig.coefficient, pm, sig.index, pw, lambda: b.one)
                            plain._insert('J', sig, W, factory=lambda: b.one)
                    if step % 9 == 0:
                        a, z = fast.pop(), plain.pop()
                        self.assertEqual((a.token, a.signature, a.leading_monomial),
                                         (z.token, z.signature, z.leading_monomial))
                        if a.kind == 'J':
                            fast.settle_j(a)
                            plain.settle_j(z)
                while fast:
                    a, z = fast.pop(), plain.pop()
                    self.assertEqual((a.token, a.signature, a.leading_monomial),
                                     (z.token, z.signature, z.leading_monomial))
                    if a.kind == 'J':
                        fast.settle_j(a)
                        plain.settle_j(z)
                self.assertEqual(fast.settled_j_bounds, plain.settled_j_bounds)
                for key in ('j_enqueued', 'j_popped', 'g_enqueued', 'g_popped',
                            'j_dedup_discarded', 'j_dedup_settled', 'j_dedup_replaced',
                            'queue_stale_entries', 'queue_peak'):
                    self.assertEqual(fast.stats.counters[key], plain.stats.counters[key])

    def test_high_dimension_queue_preserves_every_coordinate_and_signed_coefficient(self):
        rng = random.Random(280129)
        for nvars in (17, 32, 64, 129):
            with self.subTest(nvars=nvars):
                b, fast, plain = self.make_pair('small-coeff', nvars=nvars)
                for step in range(180):
                    # Every coordinate participates; late coordinates cannot be truncated.
                    monomial = tuple(rng.randrange(3) for _ in range(nvars))
                    leading = tuple(rng.randrange(4) for _ in range(nvars))
                    coefficient = (-1 if step % 2 else 1) * (2**4097 + step % 7)
                    signature = Signature(coefficient, monomial, step % 3)
                    for queue in (fast, plain):
                        if step % 11 == 0:
                            queue.push_g_lazy(signature, leading, 3, lambda: b.one)
                        else:
                            queue.push_j(signature, leading, lambda: b.one)
                while fast:
                    a, z = fast.pop(), plain.pop()
                    self.assertEqual((a.token, a.kind, a.signature, a.leading_monomial),
                                     (z.token, z.kind, z.signature, z.leading_monomial))
                    if a.kind == 'J':
                        fast.settle_j(a)
                        plain.settle_j(z)
                self.assertFalse(plain)
                self.assert_logical_queue_counts_equal(fast, plain)


if __name__ == '__main__':
    unittest.main(verbosity=2)
