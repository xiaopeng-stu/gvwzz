"""Lazy recipes preserve exact arithmetic, budgets, lifetime and certificates."""
import gc
import unittest
import weakref
from dataclasses import replace
from unittest.mock import patch

from gvwzz import GVWInteger, Options, compute
from gvwzz.core import Signature
from gvwzz.monomial_packing import pack_monomial
from gvwzz.orders import ModuleOrder
from gvwzz.polynomial import IntegerPolynomialRing
from gvwzz.recipes import (
    JPolynomialRecipe, GPolynomialRecipe,
    PythonJPolynomialRecipe, PythonGPolynomialRecipe,
)
from gvwzz.reconstruction import verify_generation_paths
from gvwzz.solver import ComputationLimit
from gvwzz.verification import verify_result
import gvwzz.solver as solver_module
from tests.differential_checks import LOGICAL_COUNTERS


def engine(nvars=2, options=None, solver_type=GVWInteger):
    names = [f'x{i}' for i in range(nvars)]
    backend = IntegerPolynomialRing(names, 'grevlex')
    inputs = [backend.parse('4*x0^2'), backend.parse('9*x1^4'),
              backend.parse('6*x0*x1^2+x1')]
    return solver_type(backend, inputs, ModuleOrder(backend, inputs, 'g2'), options)


class QueueRecipesChecks(unittest.TestCase):
    def recipe_implementations(self):
        return list(dict.fromkeys((
            (PythonJPolynomialRecipe, PythonGPolynomialRecipe),
            (JPolynomialRecipe, GPolynomialRecipe),
        )))

    def test_wide_signed_j_and_g_arithmetic_remains_lazy(self):
        for j_recipe, g_recipe in self.recipe_implementations():
            for nvars in (2, 9, 16, 17):
                with self.subTest(implementation=j_recipe, nvars=nvars):
                    sol = engine(nvars)
                    shift = (2,) + (0,) * (nvars - 2) + (3,)
                    scalar = -(2 ** 1024 + 3)
                    packed_shift = pack_monomial(shift)
                    recipe = j_recipe(sol._recipe_context, sol.inputs[0], scalar, packed_shift)
                    self.assertEqual(sol.stats.counters['j_materialized'], 0)
                    candidate = sol.queue._insert_j_packed(scalar, packed_shift, 0,
                                                          packed_shift, recipe)
                    expected = sol.backend._multiply_term_from_arithmetic(sol.inputs[0], scalar, shift)
                    self.assertEqual(candidate.label.polynomial, expected)
                    self.assertIs(candidate.label, candidate.label)
                    self.assertEqual(sol.stats.counters['j_materialized'], 1)
                    g = g_recipe(sol._recipe_context, sol.inputs[0], scalar, shift,
                                 sol.inputs[1], -scalar + 7, (0,) * nvars)
                    self.assertEqual(sol.stats.counters['g_materialized'], 0)
                    self.assertEqual(g(), expected + (-scalar + 7) * sol.inputs[1])
                    self.assertEqual(sol.stats.counters['g_materialized'], 1)

    def test_unpacked_exponents_and_eager_queue_keep_original_behavior(self):
        for j_recipe, _ in self.recipe_implementations():
            sol = engine(options=Options(lazy_candidates=False))
            shift = (2 ** 70, 0)
            recipe = j_recipe(sol._recipe_context, sol.inputs[0], 5, shift, packed=False)
            sig = Signature(5, shift, 0)
            sol.queue.push_j(sig, shift, recipe)
            candidate = sol.queue.active[sol.queue.serial]
            self.assertIsNotNone(candidate._label)
            self.assertEqual(candidate.label.polynomial,
                             sol.backend._multiply_term_from_arithmetic(sol.inputs[0], 5, shift))
            self.assertEqual(sol.stats.counters['j_materialized'], 1)

    def test_budget_errors_keep_type_message_and_attempt_counter(self):
        for j_recipe, _ in self.recipe_implementations():
            for budget, value, message in (
                ('max_polynomial_terms', 1, 'max_polynomial_terms exceeded'),
                ('max_coefficient_bits', 2, 'max_coefficient_bits exceeded'),
            ):
                with self.subTest(implementation=j_recipe, budget=budget):
                    sol = engine(options=Options(**{budget: value}))
                    recipe = j_recipe(sol._recipe_context, sol.inputs[2], 1, 0)
                    with self.assertRaisesRegex(ComputationLimit, message):
                        recipe()
                    self.assertEqual(sol.stats.counters['j_materialized'], 1)

    def test_replacing_engine_options_changes_pending_recipe_budget(self):
        for j_recipe, _ in self.recipe_implementations():
            sol = engine()
            recipe = j_recipe(sol._recipe_context, sol.inputs[2], 1, 0)
            sol.options = replace(sol.options, max_polynomial_terms=1)
            with self.assertRaisesRegex(ComputationLimit, 'max_polynomial_terms exceeded'):
                recipe()
            self.assertEqual(sol.stats.counters['j_materialized'], 1)

    def test_retained_candidate_does_not_keep_ordinary_solver_alive(self):
        for j_recipe, _ in self.recipe_implementations():
            sol = engine()
            expected = sol.inputs[0] * 3
            recipe = j_recipe(sol._recipe_context, sol.inputs[0], 3, 0)
            candidate = sol.queue._insert_j_packed(3, 0, 0, 0, recipe)
            ref = weakref.ref(sol)
            del sol, recipe
            gc.collect()
            self.assertIsNone(ref())
            self.assertEqual(candidate.label.polynomial, expected)

    def test_specialized_materialization_and_check_hooks_are_preserved(self):
        class SpecializedSolver(GVWInteger):
            def _check_polynomial(self, polynomial):
                self.specialized_checks = getattr(self, 'specialized_checks', 0) + 1
                return super()._check_polynomial(polynomial)

            def _materialize_polynomial(self, factory, kind):
                self.specialized_materializations = getattr(self, 'specialized_materializations', 0) + 1
                return super()._materialize_polynomial(factory, kind)

        for j_recipe, _ in self.recipe_implementations():
            sol = engine(solver_type=SpecializedSolver)
            recipe = j_recipe(sol._recipe_context, sol.inputs[0], 3, 0)
            self.assertEqual(recipe(), sol.inputs[0] * 3)
            self.assertEqual(sol.specialized_checks, 1)
            self.assertEqual(sol.specialized_materializations, 1)

    def test_python_and_selected_recipes_preserve_paths_and_witnesses(self):
        for switches in ({'record_paths': True, 'lazy_candidates': False},
                         {'track_representations': True}):
            with self.subTest(switches=switches):
                with patch.object(solver_module, '_native_emit_parent_pairs', None):
                    actual = engine(options=Options(**switches)).run()
                    with patch.object(solver_module, 'JPolynomialRecipe', PythonJPolynomialRecipe), \
                         patch.object(solver_module, 'GPolynomialRecipe', PythonGPolynomialRecipe):
                        reference = engine(options=Options(**switches)).run()
                self.assertEqual(actual.status, 'completed')
                self.assertEqual(actual.basis, reference.basis)
                self.assertEqual(actual.syzygies, reference.syzygies)
                self.assertEqual(actual.insertion_history, reference.insertion_history)
                self.assertEqual(actual.basis_representations, reference.basis_representations)
                self.assertEqual(actual.syzygy_representations, reference.syzygy_representations)
                for key in LOGICAL_COUNTERS:
                    self.assertEqual(actual.stats.counters[key], reference.stats.counters[key], key)
                report = (verify_generation_paths(actual) if switches.get('record_paths')
                          else verify_result(actual, strong=True))
                self.assertTrue(report['signature_basis_certified'])

    def test_incremental_stages_account_materialization_in_shared_stats(self):
        result = compute(['x*y-z', 'x*z-y', 'y*z-x'], ['x', 'y', 'z'],
                         'grevlex', module_order='pot',
                         options=Options(incremental_pot=True), show_basis=False, show_u=False)
        self.assertEqual(result.status, 'completed')
        counts = result.stats.counters
        self.assertGreater(counts['j_materialized'], 0)
        self.assertEqual(counts['j_materialized'], counts['j_reduction_calls'])
        self.assertGreaterEqual(counts['candidate_recipes_compact'],
                                counts['j_materialized'] + counts['g_materialized'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
