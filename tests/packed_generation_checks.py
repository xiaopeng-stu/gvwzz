"""Exact descriptor boundary, allocation and tuple-fallback regression checks."""
import random
import unittest
from unittest.mock import patch

from gvwzz import GVWInteger, Options, Signature
from gvwzz.core import Label, lcm_monomials, add_monomials, subtract_monomials
from gvwzz.monomial_packing import PackedMonomials, pack_monomial, unpack_monomial
from gvwzz.orders import ModuleOrder
from gvwzz.polynomial import IntegerPolynomialRing


class PackedGenerationChecks(unittest.TestCase):
    def test_arithmetic_and_lane_boundaries(self):
        rng = random.Random(112020)
        for nvars in (1, 2, 8, 48):
            codec = PackedMonomials(nvars)
            for _ in range(200):
                a = tuple(rng.randrange(64) for _ in range(nvars))
                b = tuple(rng.randrange(64) for _ in range(nvars))
                sig = tuple(rng.randrange(64) for _ in range(nvars))
                pa, pb, ps = map(pack_monomial, (a, b, sig))
                w = codec.lcm(pa, pb)
                self.assertEqual(codec.unpack(w), lcm_monomials(a, b))
                upper = w - pa + ps
                self.assertEqual(codec.unpack(upper),
                    add_monomials(subtract_monomials(lcm_monomials(a, b), a), sig))
                self.assertEqual(codec.unpack(pb + ps), add_monomials(b, sig))
            self.assertEqual(codec.unpack(codec.lcm(0, pack_monomial((127,) * nvars))),
                             (127,) * nvars)
            self.assertEqual(unpack_monomial(pack_monomial((0,) * nvars), nvars),
                             (0,) * nvars)
        self.assertIsNone(pack_monomial((128,)))
        self.assertIsNone(pack_monomial((10**100,)))
        self.assertIsNone(pack_monomial((64,), 63))

    def test_rejected_pair_allocates_no_signature_or_dense_descriptors(self):
        b = IntegerPolynomialRing(['x' + str(i) for i in range(48)], 'grevlex')
        fs = [b.parse('x0'), b.parse('x1')]
        sol = GVWInteger(b, fs, ModuleOrder(b, fs, 'pot'), Options())
        for i, f in enumerate(fs):
            sol._insert_basis(Label(Signature(1, b.zero_monomial, i), f))
        sol.S.insert_and_complete(Signature(1, b.zero_monomial, 1))
        with patch.object(sol, '_skip_principal_pair', return_value=True):
            with patch('gvwzz.solver._signature_from_arithmetic', side_effect=AssertionError('signature allocation')):
                with patch.object(PackedMonomials, 'unpack', side_effect=AssertionError('dense descriptor')):
                    sol._emit_pair(0, 1)
        self.assertEqual(sol.stats.counters['j_generated'], 1)
        self.assertEqual(sol.stats.counters['j_generation_syzygy_rejected'], 1)
        self.assertEqual(sol.stats.counters['candidate_signature_objects_avoided'], 1)
        self.assertEqual(sol.stats.counters['g_omitted_comparable'], 1)

    def test_redundant_principal_seed_allocates_no_dense_descriptor(self):
        b = IntegerPolynomialRing(['x' + str(i) for i in range(48)], 'grevlex')
        fs = [b.parse('x0'), b.parse('x1')]
        sol = GVWInteger(b, fs, ModuleOrder(b, fs, 'pot'), Options())
        for i, f in enumerate(fs):
            sol._insert_basis(Label(Signature(1, b.zero_monomial, i), f))
        sol.S.insert_and_complete(Signature(1, b.zero_monomial, 1))
        with patch('gvwzz.solver.add_monomials', side_effect=AssertionError('dense sum')):
            with patch('gvwzz.solver._signature_from_arithmetic', side_effect=AssertionError('signature allocation')):
                with patch.object(PackedMonomials, 'unpack', side_effect=AssertionError('dense descriptor')):
                    sol._emit_pair(0, 1)
        self.assertEqual(sol.stats.counters['principal_syzygies_detected'], 1)
        self.assertEqual(sol.stats.counters['syzygy_seed_rejected'], 1)
        self.assertEqual(sol.stats.counters['principal_syzygy_objects_avoided'], 1)

    def test_packed_and_tuple_pairs_match_all_orders(self):
        rng = random.Random(224020)
        semantic = ('parent_pairs', 'j_generated', 'g_generated', 'g_discarded',
                    'g_omitted_comparable', 'j_enqueued', 'g_enqueued',
                    'j_generation_tests', 'j_generation_syzygy_rejected',
                    'j_dedup_discarded', 'j_dedup_replaced', 'j_dedup_settled',
                    'principal_syzygies_detected', 'principal_syzygies_unknown',
                    'syzygy_seed_rejected', 'bezout_adjustments')
        for dim in (2, 8, 48):
            b = IntegerPolynomialRing(['x' + str(i) for i in range(dim)], 'grevlex')
            for module in ('pot', 'top', 'g1', 'g2', 'schreyer', 'block'):
                for coefficients in ((1, -1, 1), (4, -6, 10), (2**200, -(2**201), 2**202)):
                    fs, labels = [], []
                    for i, c in enumerate(coefficients):
                        lm = tuple(rng.randrange(3) if k < 3 else 0 for k in range(dim))
                        sig = tuple(rng.randrange(3) if k < 3 else 0 for k in range(dim))
                        f = b.ring.from_dict({lm: c})
                        fs.append(f)
                        labels.append(Label(Signature(rng.choice((-3, 1, 2)), sig, i), f))
                    order = ModuleOrder(b, fs, module, positions=(2, 0, 1),
                                        block_sizes=(1, 2) if module == 'block' else None)
                    options = Options(g_precheck=False)
                    a, z = (GVWInteger(b, fs, order, options) for _ in range(2))
                    for label in labels:
                        a._insert_basis(label)
                        z._insert_basis(label)
                    z.packed_parents[:] = [None] * len(z.P)
                    for i, j in ((0, 1), (0, 2), (1, 2), (0, 1)):
                        a._emit_pair(i, j)
                        z._emit_pair(i, j)
                    self.assertEqual(a.S.all_signatures(), z.S.all_signatures())
                    for key in semantic:
                        self.assertEqual(a.stats.counters[key], z.stats.counters[key],
                                         (dim, module, coefficients, key))
                    self.assertEqual(a.queue.heap, z.queue.heap)
                    while a.queue:
                        left, right = a.queue.pop(), z.queue.pop()
                        self.assertEqual(left.signature, right.signature)
                        self.assertEqual(left.leading_monomial, right.leading_monomial)
                        self.assertEqual(left.label, right.label)

    def test_unbounded_exponents_use_original_tuple_path(self):
        b = IntegerPolynomialRing(['x', 'y'], 'grevlex')
        fs = [b.parse('x^128'), b.parse('y^129')]
        sol = GVWInteger(b, fs, ModuleOrder(b, fs, 'pot'), Options())
        for i, f in enumerate(fs):
            sol._insert_basis(Label(Signature(1, b.zero_monomial, i), f))
        self.assertEqual(sol.packed_parents, [None, None])
        with patch.object(sol.queue, '_accept_j_packed', side_effect=AssertionError('overflow packed')):
            sol._emit_pair(0, 1)
        self.assertEqual(sol.stats.counters['parent_pairs'], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
