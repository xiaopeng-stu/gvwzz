"""Exact parent-row checks for the ordinary native management loop.

The reference is the established per-pair implementation in the same engine;
only the optional bulk emitter is disabled. Logical work and ordered labels
must agree, including nonunit coefficients and every fallback boundary.
"""
import random
import unittest
from unittest.mock import patch

from gvwzz import Options, compute
from gvwzz.core import Label, Signature
from gvwzz.orders import ModuleOrder
from gvwzz.polynomial import IntegerPolynomialRing
from gvwzz.solver import GVWInteger
import gvwzz.solver as solver
from tests.differential_checks import LOGICAL_COUNTERS


def records(result):
    return [(p.signature.coefficient, p.signature.monomial, p.signature.index,
             tuple(sorted(p.polynomial.items()))) for p in result.basis]


@unittest.skipIf(solver._native_emit_parent_pairs is None,
                 'optional native ordinary parent emitter unavailable')
class NativeParentPairsChecks(unittest.TestCase):
    def assert_semantics(self, actual, reference):
        self.assertEqual(actual.status, reference.status)
        self.assertEqual(records(actual), records(reference))
        self.assertEqual(actual.syzygies, reference.syzygies)
        for key in LOGICAL_COUNTERS:
            self.assertEqual(actual.stats.counters[key], reference.stats.counters[key], key)

    def compare(self, polynomials, variables=('x', 'y'), module='g2', **switches):
        options = Options(max_seconds=10, **switches)
        kwargs = dict(options=options, module_order=module,
                      show_basis=False, show_u=False)
        actual = compute(polynomials, variables, 'grevlex', **kwargs)
        with patch.object(solver, '_native_emit_parent_pairs', None):
            reference = compute(polynomials, variables, 'grevlex', **kwargs)
        self.assert_semantics(actual, reference)
        return actual

    def test_signed_nonunit_and_arbitrary_precision_coefficients(self):
        huge = 2 ** 4097 + 1
        for polynomials in (
            ['2*x^2+3*y', '3*x*y-2', '5*y^2-x'],
            ['-6*x^2+y', '10*x*y+1', '-15*y^2+x'],
            [f'{huge}*x', f'{huge + 2}*y', 'x+y'],
            ['x^2-y', 'x*y-1', '0'],
        ):
            with self.subTest(polynomials=polynomials):
                result = self.compare(polynomials)
                self.assertEqual(result.status, 'completed')
                self.assertGreater(result.stats.counters['native_parent_pairs'], 0)

    def test_unsigned_coefficient_boundaries_keep_exact_signed_scalars(self):
        for first, second in ((2 ** 64 - 1, 2 ** 63 + 1),
                              (-(2 ** 64 - 1), 2 ** 64 - 3),
                              (2 ** 64, -(2 ** 64 + 1))):
            with self.subTest(first=first, second=second):
                result = self.compare([f'{first}*x', f'{second}*y', 'x+y'])
                self.assertEqual(result.status, 'completed')

    def test_dedup_superset_tracks_all_queue_admission_paths(self):
        from gvwzz._native_pairs import ParentDedupMembership
        from gvwzz.monomial_packing import pack_monomial
        backend = IntegerPolynomialRing(('x', 'y'), 'grevlex')
        inputs = [backend.parse('x'), backend.parse('y')]
        order = ModuleOrder(backend, inputs, 'g2')
        queue = GVWInteger(backend, inputs, order, Options()).queue
        coefficient = 2 ** 300 + 1
        first = pack_monomial((1, 2))
        queue._insert_j_packed(coefficient, first, 0, first, lambda: inputs[0])
        membership = ParentDedupMembership(queue)
        queue._native_j_membership = membership
        self.assertTrue(membership.might_contain(coefficient, first, 0))
        second = pack_monomial((3, 4))
        queue._insert_j_packed(-coefficient, second, 1, second, lambda: inputs[1])
        self.assertTrue(membership.might_contain(-coefficient, second, 1))
        # The ordinary admission path is used by unsupported parent fallbacks.
        sig = Signature(coefficient, (5, 6), 0)
        queue._insert('J', sig, (5, 6), factory=lambda: inputs[0])
        self.assertTrue(membership.might_contain(coefficient, pack_monomial(sig.monomial), 0))
        # Removal never invalidates a previously admitted signature's bit set.
        while queue:
            queue.pop()
        self.assertTrue(membership.might_contain(coefficient, first, 0))
        self.assertTrue(membership.might_contain(-coefficient, second, 1))

        # A caller can hold a popped candidate while the optional gate is
        # first constructed. Publishing its settled bound must register it.
        held_queue = GVWInteger(backend, inputs, order, Options()).queue
        held_queue._insert_j_packed(coefficient, first, 0, first, lambda: inputs[0])
        held = held_queue.pop()
        late_membership = ParentDedupMembership(held_queue)
        held_queue._native_j_membership = late_membership
        self.assertFalse(late_membership.might_contain(coefficient, first, 0))
        held_queue.settle_j(held)
        self.assertTrue(late_membership.might_contain(coefficient, first, 0))

    def test_dedup_coefficient_hash_collisions_only_add_exact_lookups(self):
        import sys
        from gvwzz._native_pairs import ParentDedupMembership
        backend = IntegerPolynomialRing(('x', 'y'), 'grevlex')
        inputs = [backend.parse('x')]
        order = ModuleOrder(backend, inputs, 'g2')
        # Python integer hashes have deliberate collisions; the Bloom gate
        # must never treat their hash as an algebraic coefficient identity.
        huge = 2**4097 + 1
        for first, collision in ((-1, -2), (1, 1 + sys.hash_info.modulus),
                                 (huge, huge + sys.hash_info.modulus)):
            self.assertEqual(hash(first), hash(collision))
            queue = GVWInteger(backend, inputs, order, Options()).queue
            queue._insert_j_packed(first, 1, 0, 1, lambda: inputs[0])
            membership = ParentDedupMembership(queue)
            queue._native_j_membership = membership
            self.assertTrue(membership.might_contain(first, 1, 0))
            self.assertTrue(membership.might_contain(collision, 1, 0))
            self.assertTrue(queue._accept_j_packed(collision, 1, 0, 1))
            queue._insert_j_packed(collision, 1, 0, 1, lambda: inputs[0])
            self.assertEqual(len(queue.pending_j), 2)

    def test_all_generation_options(self):
        polynomials = ['2*x^2-y', '3*x*y-1', '5*y^2-x']
        for switches in ({}, {'lazy_candidates': False}, {'j_deduplicate': False},
                         {'j_generation_filter': False}, {'g_precheck': False},
                         {'queue_tie': 'fifo'}, {'queue_tie': 'small-lm'},
                         {'g_first': False}, {'debug': True}):
            with self.subTest(switches=switches):
                self.assertEqual(self.compare(polynomials, **switches).status, 'completed')

    def test_budget_and_certificate_paths_keep_original_emitter(self):
        polynomials = ['2*x^2-y', '3*x*y-1', '5*y^2-x']
        for switches in ({'max_candidates': 1}, {'max_candidates': 7},
                         {'record_paths': True}, {'track_representations': True}):
            with self.subTest(switches=switches):
                result = self.compare(polynomials, **switches)
                self.assertEqual(result.stats.counters['native_parent_pairs'], 0)

    def test_direct_rows_all_orders_and_packing_boundaries(self):
        rng = random.Random(2026101607)
        for nvars in (2, 8, 9, 12, 16, 17):
            backend = IntegerPolynomialRing(['x' + str(k) for k in range(nvars)], 'grevlex')
            for module in ('pot', 'top', 'g1', 'g2', 'schreyer', 'block'):
                inputs, labels = [], []
                for i in range(7):
                    lm = tuple(rng.randrange(4) if k < 2 else 0 for k in range(nvars))
                    sig = tuple(rng.randrange(4) if k < 2 else 0 for k in range(nvars))
                    if i == 2:
                        lm = (70,) + lm[1:]
                    elif i == 4:
                        lm = (63,) + lm[1:]
                        sig = (63,) + sig[1:]
                    lc = rng.choice((-6, -3, 1, 2, 5, 2 ** 130 + 1))
                    poly = backend.ring.from_dict({lm: lc})
                    inputs.append(poly)
                    labels.append(Label(Signature(rng.choice((-3, 1, 2 ** 200)), sig, i), poly))
                order = ModuleOrder(backend, inputs, module,
                                    positions=tuple(reversed(range(7))),
                                    block_sizes=(3, 4) if module == 'block' else None)
                native, reference = (GVWInteger(backend, inputs, order, Options()) for _ in range(2))
                for label in labels:
                    native._insert_basis(label)
                    reference._insert_basis(label)
                for j in range(1, len(labels)):
                    native._emit_pairs(j)
                    for i in range(j):
                        reference._emit_pair(i, j)
                with self.subTest(nvars=nvars, module=module):
                    self.assertEqual(native.S.all_signatures(), reference.S.all_signatures())
                    self.assertEqual(native.queue.heap, reference.queue.heap)
                    for key in LOGICAL_COUNTERS:
                        self.assertEqual(native.stats.counters[key], reference.stats.counters[key], key)
                    while native.queue:
                        self.assertEqual(native.queue.pop().label, reference.queue.pop().label)
                    self.assertFalse(reference.queue)
                    self.assertGreater(native.stats.counters['native_parent_pairs'], 0)
                    self.assertGreater(native.stats.counters['native_parent_fallback_pairs'], 0)
                    if nvars > 16:
                        self.assertGreater(native.stats.counters['native_parent_dynamic_pairs'], 0)


if __name__ == '__main__':
    unittest.main()
