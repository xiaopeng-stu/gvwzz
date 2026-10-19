"""Exact incremental principal omission against the original queried path.

The reference restores both the old Python skip hook and current-row dispatch,
so it genuinely computes and queries every original current/current principal
seed. Its descriptors are checked independently against the frozen strong
terms, including integer coefficient divisibility.
"""
from contextlib import ExitStack
import random
import unittest
from unittest.mock import patch

from gvwzz import Options, compute, verify_generation_paths
import gvwzz.incremental as incremental
from gvwzz.solver import GVWInteger
from gvwzz.verification import verify_result

ALGEBRAIC_COUNTERS = (
    'parent_pairs', 'j_generated', 'g_generated', 'g_discarded',
    'j_inserted', 'g_inserted', 'zero_reductions', 'j_zero_reductions',
    'g_zero_reductions', 'regular_steps', 'singular_steps', 'euclidean_steps',
    'tail_steps', 'j_tail_steps', 'g_tail_steps', 'j_reduction_calls',
)


def labels(basis):
    return [(p.signature.coefficient, p.signature.monomial, p.signature.index,
             tuple(sorted(p.polynomial.items()))) for p in basis]


def random_cases():
    rng = random.Random(220042)
    for _ in range(40):
        polynomials = []
        for _ in range(3):
            terms = []
            for a, b in ((0,0), (1,0), (0,1), (2,0), (1,1), (0,2)):
                c = rng.randint(-3,3)
                if c and rng.random() < 0.5:
                    terms.append(f'({c})*x**{a}*y**{b}')
            polynomials.append('+'.join(terms) or 'x')
        yield polynomials


class PrincipalPrefixChecks(unittest.TestCase):
    def run_case(self, polynomials, *, reference=False, **switches):
        stages, proofs = [], []
        old_minimal = incremental._minimal_records
        old_insert = GVWInteger._insert_syzygy

        def capture(engine):
            chosen = old_minimal(engine)
            stages.append((labels(engine.P), tuple(chosen),
                tuple((s.coefficient,s.monomial,s.index)
                      for s in engine.S.all_signatures())))
            return chosen

        def checked_insert(engine, signature, **kwargs):
            if isinstance(signature, tuple):
                c, m, k = signature
                if kwargs.get('monomial_packed'):
                    m = engine.pair_packing.unpack(m)
                self.assertEqual(k, engine.position)
                frozen = engine.leading[:engine.frozen_count]
                # This is an independent term-by-term scan, not the optimized
                # lookup or S, and checks a single coefficient-aware divisor.
                self.assertTrue(any(c % d == 0 and all(a <= b for a,b in zip(mu,m))
                                    for mu,d in frozen), (signature, frozen))
                proofs.append((abs(c) != 1, any(d != 1 for _,d in frozen)))
            return old_insert(engine, signature, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(patch.object(incremental, '_minimal_records', capture))
            if reference:
                stack.enter_context(patch.object(incremental._POTStage,
                    '_skip_principal_pair', lambda e,i,j: i < e.frozen_count or j < e.frozen_count))
                stack.enter_context(patch.object(incremental._POTStage,
                    '_principal_pairs_redundant', False))
                stack.enter_context(patch.object(incremental,
                    '_native_emit_unit_current_pairs', None))
                stack.enter_context(patch.object(incremental._POTStage,
                    '_insert_syzygy', checked_insert))
            result = compute(polynomials, ('x','y'), 'grevlex',
                options=Options(incremental_pot=True, max_seconds=10, **switches),
                show_basis=False, show_u=False, show_forest=False)
        return result, stages, proofs

    def compare(self, polynomials, **switches):
        actual, stages, _ = self.run_case(polynomials, **switches)
        reference, old_stages, proofs = self.run_case(polynomials, reference=True, **switches)
        self.assertEqual(actual.status, reference.status)
        self.assertEqual(stages, old_stages)
        self.assertEqual(labels(actual.basis), labels(reference.basis))
        for key in ALGEBRAIC_COUNTERS:
            self.assertEqual(actual.stats.counters[key], reference.stats.counters[key], key)
        self.assertEqual(actual.incremental_metadata['stages_completed'], len(stages))
        self.assertEqual(actual.incremental_metadata['partial_stage_active_records'],
                         reference.incremental_metadata['partial_stage_active_records'])
        return actual, reference, proofs

    def test_forty_complete_integer_systems_and_certificates(self):
        proof_count = nonunit = mixed_prefix = skipped = 0
        profiles = ({}, {'queue_tie':'fifo'}, {'lazy_candidates':False},
                    {'j_generation_filter':False}, {'cover_mode':'naive'},
                    {'track_representations':True}, {'record_paths':True},
                    {'j_deduplicate':False})
        for i, polynomials in enumerate(random_cases()):
            switches = profiles[i % len(profiles)]
            with self.subTest(case=i, switches=switches):
                actual, reference, proofs = self.compare(polynomials, **switches)
                # An interrupted/limited prefix is not a successful comparison.
                self.assertEqual(actual.status, 'completed', actual.message)
                self.assertEqual(reference.status, 'completed', reference.message)
                self.assertEqual(len(actual.incremental_metadata['stages']), 3)
                self.assertEqual(actual.stats.counters['principal_syzygies_detected'], 0)
                proof_count += len(proofs)
                nonunit += sum(p[0] for p in proofs)
                mixed_prefix += sum(p[1] for p in proofs)
                skipped += actual.stats.counters['incremental_current_principal_skipped']
                if switches.get('record_paths'):
                    check = verify_generation_paths(actual, timeout=10, max_reduction_steps=0)
                    self.assertTrue(check['passed'], check)
                if switches.get('track_representations'):
                    self.assertEqual(actual.basis_representations, reference.basis_representations)
                    check = verify_result(actual, timeout=10, mode='static',
                        require_certificates=True, max_reduction_steps=0)
                    self.assertTrue(check['passed'], check)
        self.assertGreater(proof_count, 0)
        self.assertGreater(nonunit, 0)
        self.assertGreater(mixed_prefix, 0)
        self.assertGreater(skipped, 0)
        print(f'40 completed integer systems: {proof_count} frozen principal proofs; '
              f'{nonunit} nonunit coefficients; {mixed_prefix} mixed frozen prefixes')

    def test_candidate_budgets_keep_original_partial_prefix(self):
        polynomials = ['x^2*y-y^2', 'x*y^2-x', 'x^2-y^3']
        for budget in (1, 5, 10, 30, 75):
            with self.subTest(budget=budget):
                actual, reference, _ = self.compare(polynomials, max_candidates=budget)
                self.assertEqual(actual.message, reference.message)

    def test_nonincremental_solver_has_no_prefix_proof_flag(self):
        self.assertFalse(getattr(GVWInteger, '_principal_pairs_redundant', False))
        self.assertTrue(incremental._POTStage._principal_pairs_redundant)


if __name__ == '__main__':
    unittest.main()
