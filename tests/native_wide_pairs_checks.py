"""Fixed- and dynamic-width parent rows preserve exact Python pair semantics."""
import unittest
from unittest.mock import patch

from gvwzz import Options, compute
from gvwzz.core import Label, Signature
from gvwzz.monomial_packing import pack_monomial
from gvwzz.orders import ModuleOrder
from gvwzz.polynomial import IntegerPolynomialRing
from gvwzz.solver import GVWInteger
from gvwzz.structures import Stats, SyzygyStore
import gvwzz.solver as solver
from tests.differential_checks import LOGICAL_COUNTERS
from tests.native_parent_pairs_checks import records


def monomial(nvars, **powers):
    result = [0] * nvars
    for coordinate, exponent in powers.items():
        result[int(coordinate[1:])] = exponent
    return tuple(result)


@unittest.skipIf(solver._native_emit_parent_pairs is None,
                 'optional native parent emitter unavailable')
class NativeWidePairsChecks(unittest.TestCase):
    def assert_counts(self, native, reference):
        for key in LOGICAL_COUNTERS:
            self.assertEqual(native.stats.counters[key], reference.stats.counters[key], key)

    def test_completed_small_ideals_use_high_coordinates_and_generation_switches(self):
        for nvars in (9, 12, 16, 17, 32, 64, 129):
            variables = tuple(f'x{k}' for k in range(nvars))
            x, y = variables[7], variables[-1]
            inputs = [f'2*{x}^2-{y}', f'3*{x}*{y}-1', f'5*{y}^2-{x}']
            for switches in ({}, {'j_generation_filter': False},
                             {'g_precheck': False}, {'j_deduplicate': False},
                             {'queue_tie': 'small-lm'}, {'queue_tie': 'fifo'},
                             {'lazy_candidates': False}):
                with self.subTest(nvars=nvars, switches=switches):
                    options = Options(**switches)
                    actual = compute(inputs, variables, 'grevlex', module_order='g2',
                                     options=options, show_basis=False, show_u=False)
                    with patch.object(solver, '_native_emit_parent_pairs', None):
                        expected = compute(inputs, variables, 'grevlex', module_order='g2',
                                           options=options, show_basis=False, show_u=False)
                    self.assertEqual(actual.status, 'completed')
                    self.assertEqual(records(actual), records(expected))
                    self.assertEqual(actual.syzygies, expected.syzygies)
                    self.assert_counts(actual, expected)
                    self.assertGreater(actual.stats.counters['native_parent_wide_pairs'], 0)
                    self.assertEqual(actual.stats.counters['parent_fallback_dimension_rows'], 0)
                    self.assertEqual(actual.stats.counters['native_parent_dynamic_pairs'] > 0, nvars > 16)

    def test_direct_rows_cross_word_boundaries_with_signed_huge_coefficients(self):
        huge = (1 << 4097) + 1
        for nvars in (9, 12, 16, 17, 32, 64, 129):
            backend = IntegerPolynomialRing(tuple(f'x{k}' for k in range(nvars)), 'grevlex')
            locations = (7, min(16, nvars - 1), nvars - 1)
            labels = []
            for i, scalar in enumerate((-6, 10, -huge, huge + 2, 1, -1, 3, 5)):
                leading, signature = [0] * nvars, [0] * nvars
                leading[locations[i % 3]] = (63 if i == 2 else 64 if i == 4 else 1 + i % 3)
                signature[locations[(i + 1) % 3]] = 63 if i == 2 else i % 3
                # Two rows have exactly equal formal signature ratios.
                if i in (0, 1):
                    signature = leading.copy()
                poly = backend.ring.from_dict({tuple(leading): scalar})
                labels.append(Label(Signature((-huge if i == 2 else i + 1),
                                              tuple(signature), 0 if i < 2 else i), poly))
            inputs = [label.polynomial for label in labels]
            for module in ('g2', 'pot'):
                for generation_filter in (False, True):
                    with self.subTest(nvars=nvars, module=module, filter=generation_filter):
                        order = ModuleOrder(backend, inputs, module)
                        options = Options(j_generation_filter=generation_filter)
                        native, reference = [GVWInteger(backend, inputs, order, options) for _ in range(2)]
                        for label in labels:
                            native._insert_basis(label)
                            reference._insert_basis(label)
                        for j in range(1, len(labels)):
                            native._emit_pairs(j)
                            for i in range(j):
                                reference._emit_pair(i, j)
                        self.assertEqual(native.S.all_signatures(), reference.S.all_signatures())
                        self.assertEqual(native.queue.heap, reference.queue.heap)
                        self.assert_counts(native, reference)
                        self.assertGreater(native.stats.counters['native_parent_wide_pairs'], 0)
                        self.assertGreater(native.stats.counters['native_parent_fallback_pairs'], 0)
                        self.assertGreater(native.stats.counters['parent_fallback_exponent_rows'], 0)
                        while native.queue:
                            self.assertEqual(native.queue.pop().label, reference.queue.pop().label)
                        self.assertFalse(reference.queue)

    def test_dynamic_rows_keep_all_coordinates_in_dense_descriptors(self):
        for nvars in (17, 32, 64, 129):
            backend = IntegerPolynomialRing(tuple(f'x{k}' for k in range(nvars)), 'grevlex')
            labels = []
            for i, scalar in enumerate((2, -3, 5, -7, (1 << 4097) + 1)):
                leading = tuple((k + i) % 4 for k in range(nvars))
                signature = tuple((2 * k + i) % 3 for k in range(nvars))
                polynomial = backend.ring.from_dict({leading: scalar})
                labels.append(Label(Signature(i + 1, signature, i), polynomial))
            inputs = [label.polynomial for label in labels]
            order = ModuleOrder(backend, inputs, 'g2')
            options = Options(j_generation_filter=False, g_precheck=False)
            actual, expected = [GVWInteger(backend, inputs, order, options) for _ in range(2)]
            for engine in (actual, expected):
                for label in labels:
                    engine._insert_basis(label)
            for j in range(1, len(labels)):
                actual._emit_pairs(j)
                for i in range(j):
                    expected._emit_pair(i, j)
            self.assertEqual(actual.S.all_signatures(), expected.S.all_signatures())
            self.assert_counts(actual, expected)
            self.assertEqual(actual.stats.counters['native_parent_dynamic_pairs'], 10)
            while actual.queue:
                self.assertEqual(actual.queue.pop().label, expected.queue.pop().label)
            self.assertFalse(expected.queue)

    def test_fallback_parent_converts_heap_before_later_native_admission(self):
        nvars = 9
        backend = IntegerPolynomialRing(tuple(f'x{k}' for k in range(nvars)), 'grevlex')
        inputs = [backend.parse('x7'), backend.parse('x8')]
        order = ModuleOrder(backend, inputs, 'g2')
        zero = (0,) * nvars
        labels = [
            Label(Signature(1, zero, 0),
                  backend.ring.from_dict({monomial(nvars, x8=128): 2})),
            Label(Signature(1, zero, 0),
                  backend.ring.from_dict({monomial(nvars, x7=2): 5})),
            Label(Signature(1, zero, 1), inputs[1] * 3),
        ]
        options = Options(j_generation_filter=False, g_precheck=False)
        native, reference = [GVWInteger(backend, inputs, order, options) for _ in range(2)]
        for engine in (native, reference):
            for label in labels:
                engine._insert_basis(label)
            engine.queue._insert_j_packed(1, 0, 0, 0, lambda: backend.one)
            self.assertTrue(engine.queue._compact_order)
        original_heap = native.queue.heap
        native._emit_pairs(2)
        reference._emit_pair(0, 2)
        reference._emit_pair(1, 2)
        self.assertFalse(native.queue._compact_order)
        self.assertIsNot(native.queue.heap, original_heap)
        self.assertEqual(native.stats.counters['native_parent_wide_pairs'], 1)
        self.assertEqual(native.stats.counters['native_parent_fallback_pairs'], 1)
        self.assertEqual(native.queue.heap, reference.queue.heap)
        self.assertEqual(native.S.all_signatures(), reference.S.all_signatures())
        self.assert_counts(native, reference)
        while native.queue:
            self.assertEqual(native.queue.pop().label, reference.queue.pop().label)
        self.assertFalse(reference.queue)

    def test_syzygy_cache_and_witness_compare_both_limbs_exactly(self):
        from gvwzz._native_indices import PackedSyzygyQueries
        huge = (1 << 4097) + 17
        for nvars in (9, 12, 16, 17, 32, 64, 129):
            store = SyzygyStore(nvars, 1, Stats())
            last = nvars - 1
            low = monomial(nvars, x7=1)
            high = list(low)
            high[last] = 127
            store.insert_and_complete(Signature(huge, tuple(high), 0))
            oracle = PackedSyzygyQueries(store)
            self.assertTrue(oracle.enabled)
            for coefficient in (huge, -huge, 3 * huge, huge + 1, 1):
                for exponent in (0, 126, 127, 128, 255):
                    query = list(low)
                    query[last] = exponent
                    target = Signature(coefficient, tuple(query), 0)
                    expected = any(s.divides(target) for s in store.components[0])
                    # Repeated queries exercise cache keys and recent witnesses.
                    for _ in range(2):
                        self.assertEqual(oracle.query(coefficient, pack_monomial(target.monomial, 255), 0), expected)
            store.insert_and_complete(Signature(1, low, 0))
            oracle.refresh(0)
            self.assertTrue(oracle.query(1, pack_monomial(low), 0))
            oracle.flush()

    def test_bloom_gate_tracks_high_limb_fallback_and_settlement_keys(self):
        from gvwzz._native_pairs import ParentDedupMembership
        for nvars in (9, 16, 17, 32, 64, 129):
            variables = tuple(f'x{k}' for k in range(nvars))
            backend = IntegerPolynomialRing(variables, 'grevlex')
            inputs = [backend.parse(variables[-1])]
            engine = GVWInteger(backend, inputs, ModuleOrder(backend, inputs, 'g2'), Options())
            queue = engine.queue
            membership = ParentDedupMembership(queue)
            queue._native_j_membership = membership
            huge = (1 << 4097) + 1
            for i in range(1, 5):
                powers = [0] * nvars
                powers[7], powers[-1] = i, 63 + i
                packed = pack_monomial(tuple(powers))
                coefficient = huge if i % 2 else -huge
                if i % 2:
                    queue._insert_j_packed(coefficient, packed, 0, packed, lambda: inputs[0])
                else:
                    queue._insert('J', Signature(coefficient, tuple(powers), 0), tuple(powers),
                                  factory=lambda: inputs[0])
                self.assertTrue(membership.might_contain(coefficient, packed, 0))
            held = queue.pop()
            queue.settle_j(held)
            while queue:
                entry = queue.pop()
                signature = entry.signature
                self.assertTrue(membership.might_contain(signature.coefficient,
                                                        pack_monomial(signature.monomial), 0))


if __name__ == '__main__':
    unittest.main()
