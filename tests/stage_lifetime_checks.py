"""Completed POT stages release their queues without a cyclic-GC collection."""
import gc
import unittest
import weakref
from unittest.mock import patch

from gvwzz.incremental import _POTStage
from gvwzz.orders import ModuleOrder
from gvwzz.polynomial import IntegerPolynomialRing
from gvwzz.solver import GVWInteger, Options


class StageLifetimeChecks(unittest.TestCase):
    def test_completed_stages_release_without_cyclic_collection(self):
        for switches in ({}, {'record_paths': True}, {'track_representations': True},
                         {'lazy_candidates': False, 'cover_mode': 'naive'}):
            with self.subTest(switches=switches):
                references = []
                original = _POTStage.__init__

                def record(stage, *args, **kwargs):
                    original(stage, *args, **kwargs)
                    references.extend((weakref.ref(stage), weakref.ref(stage.queue)))

                backend = IntegerPolynomialRing(['x', 'y', 'z', 'w'], 'grevlex')
                inputs = [backend.parse(p) for p in
                          ('x*z-w', 'x*y*z-y', 'y*z*w-x', 'y*z*w-w')]
                enabled = gc.isenabled()
                gc.disable()
                try:
                    with patch.object(_POTStage, '__init__', record):
                        result = GVWInteger(backend, inputs, ModuleOrder(backend, inputs, 'pot'),
                            Options(incremental_pot=True, **switches)).run()
                    self.assertEqual(result.status, 'completed')
                    self.assertEqual(len(references), 8)
                    self.assertTrue(all(reference() is None for reference in references))
                finally:
                    if enabled:
                        gc.enable()


if __name__ == '__main__':
    unittest.main()
