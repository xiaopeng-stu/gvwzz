"""Private compute engines retire before output without requiring cyclic GC."""
from contextlib import contextmanager
import gc
import unittest
import weakref
from unittest.mock import patch

from gvwzz import Options, compute, verify_generation_paths
from gvwzz.orders import ModuleOrder
from gvwzz.polynomial import IntegerPolynomialRing
from gvwzz.solver import GVWInteger
from gvwzz.validation import result_structure_errors
from gvwzz.verification import verify_result


@contextmanager
def disabled_gc():
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


class ComputeCleanupChecks(unittest.TestCase):
    inputs = ('4*x^2', '9*y^4', '6*x*y^2+y')

    @contextmanager
    def recorded_engines(self):
        references = []
        original = GVWInteger.__init__

        def record(engine, *args, **kwargs):
            original(engine, *args, **kwargs)
            references.extend((weakref.ref(engine), weakref.ref(engine.queue)))

        with patch.object(GVWInteger, '__init__', record):
            yield references

    def solve(self, **switches):
        return compute(self.inputs, ('x', 'y'), 'lex', 'pot',
                       options=Options(g_first=False, **switches),
                       show_basis=False, show_u=False, show_forest=False)

    def test_private_engine_released_and_result_remains_complete(self):
        for switches in ({}, {'record_paths': True}, {'track_representations': True},
                         {'lazy_candidates': False, 'j_generation_filter': False}):
            with self.subTest(switches=switches), disabled_gc(), self.recorded_engines() as refs:
                result = self.solve(**switches)
                self.assertEqual(len(refs), 2)
                self.assertTrue(all(ref() is None for ref in refs))
                self.assertEqual(result.status, 'completed')
                self.assertEqual(result_structure_errors(result), [])
                self.assertGreaterEqual(result.runtime_cleanup_seconds, 0.0)
                # The result snapshot retains all mathematical records after
                # the engine's parent/index structures have been destroyed.
                saved = result.to_dict()
                self.assertEqual(len(saved['basis']), len(result.basis))
                self.assertEqual(len(saved['syzygy_signatures']), len(result.syzygies))
                for label, item in zip(result.basis, saved['basis']):
                    self.assertEqual(result.backend.deserialize(item['polynomial']), label.polynomial)
                if switches.get('record_paths'):
                    self.assertTrue(verify_generation_paths(result)['signature_basis_certified'])
                elif switches.get('track_representations'):
                    self.assertTrue(verify_result(result, strong=True,
                        require_certificates=True)['signature_basis_certified'])

    def test_private_engine_released_after_partial_run(self):
        for switches in ({}, {'record_paths': True}, {'track_representations': True}):
            with self.subTest(switches=switches), disabled_gc(), self.recorded_engines() as refs:
                result = self.solve(max_candidates=1, **switches)
                self.assertEqual(result.status, 'limit')
                self.assertEqual(result_structure_errors(result), [])
                self.assertTrue(all(ref() is None for ref in refs))
                saved = result.to_dict()
                self.assertEqual(len(saved['basis']), len(result.basis))
                if switches.get('record_paths'):
                    result.generation_paths.validate(result)
                elif switches.get('track_representations'):
                    self.assertEqual(len(saved['representation_certificate']['basis']), len(result.basis))

    def test_unexpected_error_and_interrupt_release_private_callbacks(self):
        for exception in (RuntimeError, KeyboardInterrupt):
            with self.subTest(exception=exception), disabled_gc(), self.recorded_engines() as refs:
                # A genuine raise also exercises the traceback's engine frame.
                def fail(engine):
                    raise exception('test interruption')

                with patch.object(GVWInteger, 'run', fail), self.assertRaises(exception):
                    self.solve()
                self.assertEqual(len(refs), 2)
                self.assertTrue(all(ref() is None for ref in refs))

    def test_direct_run_keeps_its_existing_inspectable_engine(self):
        backend = IntegerPolynomialRing(('x', 'y'), 'lex')
        inputs = [backend.parse(f) for f in self.inputs]
        engine = GVWInteger(backend, inputs, ModuleOrder(backend, inputs, 'pot'),
                            Options(g_first=False))
        original_filter = engine.queue.generation_filter
        original_cache = engine.queue._packed_leading_key
        result = engine.run()
        self.assertEqual(result.status, 'completed')
        self.assertEqual(result.basis, engine.P)
        self.assertIs(engine.queue.generation_filter, original_filter)
        self.assertIs(engine.queue._packed_leading_key, original_cache)
        self.assertFalse(hasattr(result, 'runtime_cleanup_seconds'))
        with self.assertRaisesRegex(RuntimeError, 'only once'):
            engine.run()


if __name__ == '__main__':
    unittest.main()
