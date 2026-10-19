"""Queue batching preserves ordered settlement and exact cover accounting."""
import unittest
from unittest.mock import patch

from gvwzz import GVWInteger, Options
from gvwzz.core import Label, Signature
from gvwzz.orders import ModuleOrder
from gvwzz.polynomial import IntegerPolynomialRing
from gvwzz.monomial_packing import pack_monomial
from gvwzz import solver
from tests.differential_checks import LOGICAL_COUNTERS


def make_engine(nvars=2, **options):
    ring = IntegerPolynomialRing([f'x{k}' for k in range(nvars)], 'grevlex')
    inputs = [ring.one]
    return GVWInteger(ring, inputs, ModuleOrder(ring, inputs, 'g2'),
                      Options(progress_interval=0, **options))


@unittest.skipIf(solver._native_select_candidate is None, 'native selection unavailable')
class NativeQueueDrainChecks(unittest.TestCase):
    def test_covered_packed_candidates_need_no_signature_or_tuple(self):
        for nvars in (2, 17, 65, 129):
            engine = make_engine(nvars)
            zero = (0,) * nvars
            engine.cover.insert(Label(Signature(1, zero, 0), engine.backend.one), zero)
            for k in (1, 2, 3):
                mono = (0,) * (nvars - 1) + (k,)
                W = (0,) * (nvars - 1) + (k + 1,)
                engine.queue._insert_j_packed(1, pack_monomial(mono), 0,
                                             pack_monomial(W), lambda: engine.backend.one)
            with patch('gvwzz.structures.unpack_monomial', side_effect=AssertionError('decode')):
                candidate, checked = solver._native_select_candidate(engine)
            self.assertIsNone(candidate)
            self.assertFalse(checked)
            self.assertFalse(engine.queue)
            for counter in ('iterations', 'j_popped', 'cover_calls', 'cover_rejected_j',
                            'native_queue_drain_covered'):
                self.assertEqual(engine.stats.counters[counter], 3)
            self.assertEqual(len(engine.queue.settled_j_bounds), 3)

    def test_batch_limit_and_first_uncovered_candidate(self):
        engine = make_engine()
        engine.S.insert_and_complete(Signature(1, (1, 0), 0))
        for k in range(1, 5):
            engine.queue._insert_j_packed(1, pack_monomial((k, 0)), 0,
                                         pack_monomial((k + 1, 0)), lambda: engine.backend.one)
        candidate, checked = solver._native_select_candidate(engine, 2)
        self.assertIsNone(candidate)
        self.assertEqual(len(engine.queue.active), 2)
        self.assertEqual(engine.stats.counters['j_popped'], 2)
        solver._native_select_candidate(engine, 2)
        self.assertFalse(engine.queue)
        engine.queue._insert_j_packed(-6, pack_monomial((0, 5)), 0,
                                     pack_monomial((0, 6)), lambda: engine.backend.one)
        candidate, checked = solver._native_select_candidate(engine)
        self.assertTrue(checked)
        self.assertEqual(candidate.signature, Signature(-6, (0, 5), 0))
        self.assertEqual(engine.stats.counters['cover_calls'], 5)
        self.assertEqual(engine.stats.counters['cover_rejected_j'], 4)
        self.assertNotIn(candidate._dedup_key, engine.queue.settled_j_bounds)

    def test_g_is_returned_before_any_cover(self):
        engine = make_engine()
        engine.S.insert_and_complete(Signature(1, (0, 0), 0))
        engine.queue.push_g_lazy(Signature(1, (1, 0), 0), (2, 0), 1,
                                 lambda: engine.backend.one)
        candidate, checked = solver._native_select_candidate(engine)
        self.assertEqual(candidate.kind, 'G')
        self.assertFalse(checked)
        self.assertEqual(engine.stats.counters['cover_calls'], 0)
        self.assertEqual(engine.stats.counters['g_popped'], 1)

    def test_budget_exception_precedes_next_pop(self):
        engine = make_engine()
        engine.queue._insert_j_packed(1, 1, 0, 2, lambda: engine.backend.one)
        with patch.object(engine, '_check_budget', side_effect=solver.ComputationLimit('test')):
            with self.assertRaises(solver.ComputationLimit):
                solver._native_select_candidate(engine)
        self.assertEqual(len(engine.queue.active), 1)
        self.assertEqual(engine.stats.counters['iterations'], 0)

    def test_complete_runs_match_reference_selection(self):
        cases = [(['x', 'y'], ['6*x^2+y', '10*x*y+1', '15*y^2-x']),
                 ([f'x{k}' for k in range(33)],
                  ['6*x0^2+x32', '10*x0*x32+1', '15*x32^2-x0'])]
        for names, polynomials in cases:
            for tie in ('fifo', 'small-coeff', 'small-lm'):
                outputs = []
                for native in (True, False):
                    ring = IntegerPolynomialRing(names, 'grevlex')
                    inputs = [ring.parse(p) for p in polynomials]
                    engine = GVWInteger(ring, inputs, ModuleOrder(ring, inputs, 'g2'),
                                        Options(queue_tie=tie, progress_interval=0))
                    if native:
                        result = engine.run()
                    else:
                        with patch('gvwzz.solver._native_select_candidate', None):
                            result = engine.run()
                    self.assertEqual(result.status, 'completed')
                    outputs.append(([(p.signature, ring.serialize(p.polynomial))
                                     for p in result.basis], result.syzygies,
                                    {k: result.stats.counters[k] for k in LOGICAL_COUNTERS}))
                self.assertEqual(outputs[0], outputs[1])


if __name__ == '__main__':
    unittest.main()
