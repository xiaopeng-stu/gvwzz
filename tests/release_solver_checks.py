"""Regression checks for public solver options and aborted stage cleanup."""
import gc
import unittest
import weakref
from unittest.mock import patch

from gvwzz.incremental import IncrementalPOT, _POTStage
from gvwzz.options import Options
from gvwzz.orders import ModuleOrder
from gvwzz.polynomial import IntegerPolynomialRing
from gvwzz.solver import GVWInteger, compute
from gvwzz.validation import result_structure_errors


class PublicOptionsChecks(unittest.TestCase):
    def test_invalid_options_are_never_silently_replaced(self):
        backend = IntegerPolynomialRing(('x',), 'grevlex')
        inputs = [backend.parse('2*x')]
        order = ModuleOrder(backend, inputs, 'pot')
        for bad in (False, True, 0, 1, {}, [], 'defaults', object()):
            with self.subTest(options=bad):
                for factory in (GVWInteger, IncrementalPOT):
                    with self.subTest(factory=factory.__name__):
                        with self.assertRaisesRegex(ValueError, 'Options instance'):
                            factory(backend, inputs, order, bad)
                for show_u in (False, True):
                    with self.subTest(show_u=show_u):
                        with self.assertRaisesRegex(ValueError, 'Options instance'):
                            compute(['2*x'], ['x'], options=bad,
                                    show_basis=False, show_u=show_u)

    def test_default_and_explicit_options_still_work(self):
        for options in (None, Options()):
            with self.subTest(options=options):
                result = compute(['2*x'], ['x'], options=options,
                                 show_basis=False, show_u=False)
                self.assertEqual(result.status, 'completed')
                self.assertEqual(str(result.basis[0].polynomial), '2*x')

    def test_unrepresentable_time_options_report_value_errors(self):
        for name in ('max_seconds', 'progress_interval'):
            for value in (10**400, 1e308, float('inf'), float('nan')):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, 'finite'):
                        Options(**{name: value}).validate()


class AbortedStageLifetimeChecks(unittest.TestCase):
    def test_limited_stages_release_without_cyclic_collection(self):
        for switches in ({}, {'record_paths': True}, {'track_representations': True}):
            with self.subTest(switches=switches):
                references = []
                original = _POTStage.__init__

                def record(stage, *args, **kwargs):
                    original(stage, *args, **kwargs)
                    references.extend((weakref.ref(stage), weakref.ref(stage.queue)))

                backend = IntegerPolynomialRing(('x', 'y'), 'grevlex')
                inputs = [backend.parse(f) for f in ('4*x^2', '9*y^4', '6*x*y^2+y')]
                owner = IncrementalPOT(backend, inputs, ModuleOrder(backend, inputs, 'pot'),
                                       Options(max_candidates=1, **switches))
                enabled = gc.isenabled()
                gc.disable()
                try:
                    with patch.object(_POTStage, '__init__', record):
                        result = owner.run()
                    self.assertEqual(result.status, 'limit')
                    self.assertEqual(result.incremental_metadata['stages_completed'], 1)
                    self.assertEqual(result.incremental_metadata['partial_stage_active_records'], 2)
                    self.assertEqual(result_structure_errors(result), [])
                    self.assertTrue(all(reference() is None for reference in references))
                    self.assertEqual(result.basis, owner.basis)
                    if switches.get('record_paths'):
                        result.generation_paths.validate(result)
                        self.assertIs(owner.paths.polynomials, owner.basis)
                        self.assertEqual(owner.paths.basis, owner.path_roots)
                    if switches.get('track_representations'):
                        self.assertEqual(len(result.basis_representations), len(result.basis))
                finally:
                    if enabled:
                        gc.enable()


if __name__ == '__main__':
    unittest.main()
