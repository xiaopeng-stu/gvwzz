"""Twenty-four complete systems versus legacy parent/principal emission.

No candidate-count budget is used: such a budget intentionally disables bulk
pruning. Each run must finish within its time limit. The comparison captures
every ordered stage P, selected minimal rows, stage syzygies, final labels and
algebraic reduction counters; timings and moved rejection points may differ.
"""
from contextlib import ExitStack
import hashlib
import json
from random import Random
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
    'tail_steps', 'j_tail_steps', 'g_tail_steps',
    'j_reduction_calls',
)


def label_records(basis):
    return [(p.signature.coefficient, p.signature.monomial, p.signature.index,
             tuple((m, int(c)) for m, c in sorted(p.polynomial.items()))) for p in basis]


def make_cases():
    rng = Random(220152)
    profiles = ({}, {'queue_tie': 'fifo'}, {'lazy_candidates': False},
                {'j_deduplicate': False}, {'cover_mode': 'naive'},
                {'record_paths': True}, {'track_representations': True},
                {'queue_tie': 'fifo', 'lazy_candidates': False})
    cases = []
    for number in range(20):
        variables = ('x', 'y', 'z') if number < 12 else ('x', 'y', 'z', 'w')
        powers = [rng.choice((2, 3, 4)) for _ in variables]
        v = [f'{name}^{power}' for name, power in zip(variables, powers)]
        x, y, z = v[:3]
        if len(v) == 3:
            polynomials = [f'{x}*{y}-{z}', f'{x}*{z}-{y}', f'{y}*{z}-{x}']
        else:
            w = v[3]
            polynomials = [f'{x}*{z}-{w}', f'{x}*{y}*{z}-{y}',
                           f'{y}*{z}*{w}-{x}', f'{y}*{z}*{w}-{w}']
        rng.shuffle(polynomials)
        # Positive and negative unit leading coefficients exercise the sign
        # convention used by retained frozen/current upper-parent witnesses.
        polynomials = [f'-({p})' if rng.randrange(2) else p for p in polynomials]
        cases.append((f'power-system-{number + 1:02d}', polynomials, variables,
                      dict(profiles[number % len(profiles)])))
    # Nonunit frozen leading terms exercise coefficient-aware principal
    # redundancy over ZZ; no field/monic assumption can justify these cases.
    cases.extend([
        ('nonunit-01', ['2*x^2', '3*y^2', '5*x*y+y'], ('x', 'y'),
         {'track_representations': True}),
        ('nonunit-02', ['-4*x^2+2*y', '6*x*y', '9*y^2'], ('x', 'y'),
         {'record_paths': True}),
        ('nonunit-03', ['6*x^2+y^2', '10*x*y+2*y^2', '15*y^2'], ('x', 'y'),
         {'g_precheck': False, 'queue_tie': 'fifo'}),
        ('nonunit-04', ['x^2-y', '2*y^2+x', '3*x*y+1'], ('x', 'y'),
         {'lazy_candidates': False}),
    ])
    return cases


class GeneralPowerDifferentialChecks(unittest.TestCase):
    def run_case(self, polynomials, variables, switches, *, reference):
        stage_records, general_selections = [], []
        original_minimal = incremental._minimal_records
        original_selector = incremental._native_minimal_frozen_mask

        def capture_stage(engine):
            chosen = original_minimal(engine)
            stage_records.append((label_records(engine.P), tuple(chosen),
                tuple((s.coefficient, s.monomial, s.index)
                      for s in engine.S.all_signatures()),
                tuple(engine.stats.counters[k] for k in ALGEBRAIC_COUNTERS)))
            return chosen

        def capture_selector(engine, parent, mask):
            result = original_selector(engine, parent, mask)
            if result is not None and any(any(e > 1 for e in engine.leading[i][0])
                                          for i in range(engine.frozen_count)):
                general_selections.append((engine.frozen_count, mask.bit_count(), len(result)))
            return result

        with ExitStack() as stack:
            stack.enter_context(patch.object(incremental, '_minimal_records', capture_stage))
            if reference:
                stack.enter_context(patch.object(incremental._POTStage, '_emit_pairs',
                                                 GVWInteger._emit_pairs))
                # Preserve the old principal construction/S query too. Merely
                # replacing _emit_pairs would inherit the new skip hook and
                # fail to test whether omitted current seeds were redundant.
                stack.enter_context(patch.object(incremental._POTStage, '_skip_principal_pair',
                    lambda engine, i, j: i < engine.frozen_count or j < engine.frozen_count))
            elif original_selector is not None:
                stack.enter_context(patch.object(incremental, '_native_minimal_frozen_mask',
                                                 capture_selector))
            result = compute(polynomials, variables, 'grevlex',
                options=Options(incremental_pot=True, max_seconds=30, **switches),
                show_basis=False, show_u=False)
        self.assertEqual(result.status, 'completed', result.message)
        self.assertEqual(len(stage_records), len(polynomials))
        return result, stage_records, general_selections

    def test_twenty_four_complete_systems_with_active_general_pruning(self):
        native_available = incremental._native_minimal_frozen_mask is not None
        generic_cases = dominated_cases = current_dominated_cases = 0
        skipped_principal_pairs = legacy_principal_queries = 0
        digest = hashlib.sha256()
        for name, polynomials, variables, switches in make_cases():
            with self.subTest(case=name, options=switches):
                actual, stages, selections = self.run_case(
                    polynomials, variables, switches, reference=False)
                reference, old_stages, _ = self.run_case(
                    polynomials, variables, switches, reference=True)
                self.assertEqual(stages, old_stages, name)
                self.assertEqual(label_records(actual.basis), label_records(reference.basis), name)
                for key in ALGEBRAIC_COUNTERS:
                    self.assertEqual(actual.stats.counters[key], reference.stats.counters[key],
                                     (name, key))
                self.assertEqual(actual.stats.counters['principal_syzygies_detected'], 0)
                self.assertEqual(actual.stats.counters['principal_syzygies_unknown'], 0)
                skipped_principal_pairs += actual.stats.counters['incremental_current_principal_skipped']
                legacy_principal_queries += reference.stats.counters['principal_syzygies_detected']
                if native_available and name.startswith('power-system-'):
                    self.assertTrue(selections, f'{name}: generic native selector was never used')
                    generic_cases += 1
                dominated_cases += bool(actual.stats.counters['j_dominance_rejected'])
                current_dominated_cases += bool(actual.stats.counters['j_current_dominance_rejected'])
                if switches.get('record_paths'):
                    self.assertTrue(verify_generation_paths(actual, timeout=30,
                        max_reduction_steps=0)['passed'])
                if switches.get('track_representations'):
                    self.assertTrue(verify_result(actual, timeout=30, mode='static',
                        require_certificates=True, max_reduction_steps=0)['passed'])
                digest.update(json.dumps((name, stages, label_records(actual.basis)),
                    sort_keys=True, separators=(',', ':')).encode())
        self.assertGreater(dominated_cases, 0)
        self.assertGreater(skipped_principal_pairs, 0)
        self.assertGreater(legacy_principal_queries, 0)
        if native_available:
            self.assertEqual(generic_cases, 20)
            self.assertGreater(current_dominated_cases, 0)
        print(f'24 completed differential cases; generic native active={generic_cases}; '
              f'frozen/current dominance active={dominated_cases}/{current_dominated_cases}; '
              f'current principal skipped/legacy queries={skipped_principal_pairs}/{legacy_principal_queries}; '
              f'ordered-stage SHA256={digest.hexdigest()}')


if __name__ == '__main__':
    unittest.main()
