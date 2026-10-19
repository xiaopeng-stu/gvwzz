"""Regression tests for staged POT strong ideal bases (Python standard unittest)."""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from gvwzz import (Options, compute, postprocess_basis, verify_generation_paths,
                   reconstruct_syzygies, format_basis, format_forest)
from gvwzz.cli import build_parser, resolve_orders, load_result, execute_case, flatten
from gvwzz.core import Label, Signature, add_monomials
from gvwzz.incremental import _POTStage, _minimal_records
from gvwzz.verification import verify_result

PAPER = ['4*x^2', '9*y^4', '6*x*y^2+y']


def solve(polys=PAPER, names=('x', 'y'), *, mode='none', order='lex', **options):
    opt = Options(incremental_pot=True, record_paths=mode == 'paths',
                  track_representations=mode == 'vectors', max_seconds=10,
                  **options)
    return compute(polys, names, order, options=opt, show_basis=False, show_u=False)


def canonical(r):
    return [tuple(sorted(f.items())) for f in postprocess_basis(r, reduced=True).reduced_basis]


class IncrementalChecks(unittest.TestCase):
    def certified(self, r, paths=False):
        self.assertEqual(r.status, 'completed', r.message)
        check = (verify_generation_paths(r, timeout=0, max_reduction_steps=0)
                 if paths else verify_result(r, timeout=0, mode='static', require_certificates=True,
                                             max_reduction_steps=0))
        self.assertTrue(check['passed'], check)
        self.assertTrue(check['membership_verified'], check)
        self.assertTrue(check['strong_basis_certified'], check)
        self.assertFalse(check['signature_basis_certified'], check)
        return check

    def test_01_default_is_off(self):
        self.assertFalse(Options().incremental_pot)
        with self.assertRaises(ValueError):
            Options(incremental_pot=1).validate()

    def test_02_automatic_pot(self):
        self.assertEqual(solve().module_order.name, 'pot')

    def test_03_other_module_orders_rejected(self):
        for order in ('g2','top','schreyer'):
            with self.subTest(order=order), self.assertRaises(ValueError):
                compute(PAPER, ('x','y'), module_order=order,
                        options=Options(incremental_pot=True), show_basis=False)

    def test_04_nonstandard_positions_rejected(self):
        with self.assertRaises(ValueError):
            compute(PAPER, ('x','y'), module_order='pot', positions=(2,1,0),
                    options=Options(incremental_pot=True), show_basis=False)

    def test_05_paper_minimal_result(self):
        r = solve(mode='vectors')
        self.certified(r)
        self.assertEqual([str(p.polynomial) for p in r.basis], ['y','4*x**2'])
        self.assertEqual([s['minimal_basis_size'] for s in r.incremental_metadata['stages']], [1,3,2])

    def test_06_integer_strong_not_ideal_minimal(self):
        r = solve(['2*x','3*y'], mode='vectors')
        self.certified(r)
        self.assertEqual({str(p.polynomial) for p in r.basis}, {'2*x','3*y','x*y'})

    def test_07_never_divide_content(self):
        r = solve(['-6*x'], mode='vectors')
        self.certified(r)
        self.assertEqual(str(r.basis[0].polynomial), '6*x')
        self.assertEqual(r.basis[0].signature.coefficient, -1)

    def test_08_zero_inputs(self):
        for ps in ([], ['0'], ['0','0','0']):
            with self.subTest(ps=ps):
                r = solve(ps, mode='vectors')
                self.certified(r)
                self.assertEqual(r.basis, [])
                self.assertEqual(r.incremental_metadata['stages_completed'], len(ps))

    def test_09_zeros_and_duplicates(self):
        r = solve(['0','2*x','2*x','0','x'], mode='vectors')
        self.certified(r)
        self.assertEqual([str(p.polynomial) for p in r.basis], ['x'])

    def test_10_constant_nonunits(self):
        r = solve(['6','10','15'], mode='vectors')
        self.certified(r)
        self.assertEqual([str(p.polynomial) for p in r.basis], ['1'])

    def test_11_all_inputs_after_unit(self):
        r = solve(['1','x+y','3*x^2','0','7'], mode='vectors')
        self.certified(r)
        self.assertEqual(len(r.basis), 1)
        self.assertEqual(r.incremental_metadata['stages_completed'], 5)

    def test_12_compaction_only_after_queue_empty(self):
        calls=[]
        def inspect(engine):
            self.assertFalse(engine.queue.active)
            self.assertFalse(engine.queue.heap)
            calls.append(engine.position)
            return _minimal_records(engine)
        with patch('gvwzz.incremental._minimal_records', inspect):
            solve()
        self.assertEqual(calls, [0,1,2])

    def test_13_only_current_syzygy_component(self):
        original = _POTStage._process_queue
        seen=[]
        def inspect(engine):
            original(engine)
            self.assertTrue(all(s.index == engine.position for s in engine.S.all_signatures()))
            self.assertTrue(all(p.signature.index <= engine.position for p in engine.P))
            seen.append(engine.position)
        with patch.object(_POTStage, '_process_queue', inspect):
            solve(debug=True)
        self.assertEqual(seen,[0,1,2])

    def test_14_cross_principal_skip_has_exact_divisor(self):
        original = _POTStage._skip_principal_pair
        proofs=[]
        def inspect(engine, i, j):
            answer=original(engine,i,j)
            if answer and (i < engine.frozen_count or j < engine.frozen_count):
                old, new = (i,j) if i < engine.frozen_count else (j,i)
                mu,c=engine.leading[old]
                sig=engine.P[new].signature
                self.assertTrue(engine.S._has_divisor_parts(c*sig.coefficient,
                                add_monomials(mu,sig.monomial),sig.index))
                proofs.append(1)
            return answer
        with patch.object(_POTStage, '_skip_principal_pair', inspect):
            solve()
        self.assertTrue(proofs)

    def test_15_frozen_pairs_and_cover_rows_skipped(self):
        r=solve()
        c=r.stats.counters
        self.assertGreater(c['incremental_frozen_pairs_skipped'],0)
        self.assertGreater(c['incremental_cover_records_skipped'],0)
        self.assertGreater(c['incremental_principal_skipped'],0)

    def test_16_path_certificate_across_compaction(self):
        r=solve(mode='paths')
        self.certified(r,paths=True)
        self.assertTrue(all(e.polynomial_index is None for n in r.generation_paths.nodes for e in n.edges))
        self.assertEqual(r.generation_paths.syzygy_roots, ())

    def test_17_path_sign_normalization(self):
        self.certified(solve(['-6*x','9*y','-x*y-1'],mode='paths'),paths=True)

    def test_18_vectors_keep_original_input_dimension(self):
        r=solve(['2*x','3*y','x*y-1','0','x+2'],mode='vectors')
        self.certified(r)
        self.assertTrue(all(len(v)==5 for v in r.basis_representations))

    def test_19_serialized_paths_reload(self):
        r=solve(mode='paths')
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'r.json';path.write_text(json.dumps(r.to_dict()))
            loaded=load_result(path)
            self.certified(loaded,paths=True)
            self.assertEqual(loaded.incremental_metadata,r.incremental_metadata)

    def test_20_serialized_vectors_reload(self):
        r=solve(mode='vectors')
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'r.json';path.write_text(json.dumps(r.to_dict()))
            self.certified(load_result(path))

    def test_21_no_signature_claim(self):
        r=solve()
        d=r.to_dict()
        self.assertFalse(d['solver_claims_signature_basis'])
        self.assertTrue(d['solver_claims_strong_basis'])
        self.assertFalse(d['signature_basis_certified'])
        self.assertEqual(d['solver_mode'],'incremental-pot')
        self.assertEqual(r.syzygies,[])

    def test_22_verifier_uses_S_and_G_without_strong_switch(self):
        r=solve(mode='vectors')
        v=self.certified(r)
        self.assertGreater(v['counts']['ordinary_s_pairs_checked'],0)
        self.assertGreater(v['counts']['ordinary_g_pairs_checked'],0)
        self.assertEqual(v['counts']['j_pairs_checked'],0)

    def test_23_no_certificate_not_certified(self):
        v=verify_result(solve(),mode='static',timeout=0)
        self.assertTrue(v['passed'],v)
        self.assertFalse(v['membership_verified'])
        self.assertFalse(v['strong_basis_certified'])
        self.assertFalse(v['signature_basis_certified'])
        self.assertEqual(verify_result(solve(),require_certificates=True)['status'],'missing_certificate')

    def test_24_tampered_vector_rejected(self):
        r=solve(mode='vectors')
        r.basis_representations[0]=tuple(r.backend.zero for _ in r.inputs)
        self.assertEqual(verify_result(r)['status'],'invalid_certificate')

    def test_25_missing_necessary_G_element_rejected(self):
        r=solve(['2*x','3*y'])
        r.basis=[p for p in r.basis if str(p.polynomial)!='x*y']
        v=verify_result(r,timeout=0)
        self.assertFalse(v['passed'],v)
        self.assertIn('G-polynomial',' '.join(v['errors']))

    def test_26_mode_mismatch_on_load_rejected(self):
        r=solve();d=r.to_dict();d['solver_mode']='all-inputs-at-once'
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'bad.json';path.write_text(json.dumps(d))
            with self.assertRaises(ValueError):load_result(path)

    def test_27_active_basis_budget_returns_prefix_only(self):
        r=solve(mode='paths',max_basis=2)
        self.assertEqual(r.status,'limit')
        self.assertEqual(r.incremental_metadata['stages_completed'],1)
        self.assertEqual([str(p.polynomial) for p in r.basis],['4*x**2'])
        self.assertEqual(verify_result(r)['status'],'not_completed')
        v=verify_generation_paths(r,criteria=False,timeout=0)
        self.assertTrue(v['passed'],v)
        self.assertFalse(v['strong_basis_certified'])

    def test_28_active_not_cumulative_basis_budget(self):
        r=solve(['x','2*x','3*x','4*x','5*x'],mode='vectors',max_basis=2)
        self.certified(r)
        self.assertEqual(r.stats.counters['incremental_input_insertions'],5)

    def test_29_global_candidate_budget(self):
        r=solve(max_candidates=2)
        self.assertEqual(r.status,'limit')
        self.assertEqual(r.incremental_metadata['stages_completed'],1)

    def test_30_deadline_not_reset(self):
        original=_POTStage._initialize_and_run
        deadlines=[]
        def inspect(engine):
            deadlines.append(engine.deadline)
            original(engine)
        with patch.object(_POTStage,'_initialize_and_run',inspect):
            solve()
        self.assertEqual(len(set(deadlines)),1)

    def test_31_interrupt_returns_valid_prefix(self):
        original=_POTStage._initialize_and_run
        def inspect(engine):
            if engine.position==2:raise KeyboardInterrupt()
            original(engine)
        with patch.object(_POTStage,'_initialize_and_run',inspect):
            r=solve(mode='paths')
        self.assertEqual(r.status,'interrupted')
        self.assertEqual(r.incremental_metadata['stages_completed'],2)
        self.assertTrue(verify_generation_paths(r,criteria=False,timeout=0)['passed'])

    def test_32_eager_lazy_all_kernels(self):
        reference=canonical(solve(order='grevlex'))
        for kernel in ('heap','copy','packed'):
            for lazy in (False,True):
                with self.subTest(kernel=kernel,lazy=lazy):
                    r=solve(order='grevlex',mode='paths',reduction_kernel=kernel,lazy_candidates=lazy)
                    self.certified(r,paths=True)
                    self.assertEqual(canonical(r),reference)

    def test_33_presentation_scope(self):
        r=solve()
        self.assertIn('not a full signature basis',format_basis(r))
        self.assertIn('no global',format_forest(r).lower())

    def test_34_no_global_syzygy_reconstruction(self):
        with self.assertRaises(ValueError):reconstruct_syzygies(solve(mode='paths'))

    def test_35_cli_order_resolution(self):
        parser=build_parser();args=parser.parse_args(['run','--case','eco3','--incremental-pot'])
        self.assertEqual(resolve_orders(args,{'module_order':'gvw-g2'}),('grevlex','pot',None))
        args=parser.parse_args(['run','--case','eco3','--incremental-pot','--module-order','g2'])
        with self.assertRaises(ValueError):resolve_orders(args,{})

    def test_36_output_minimal_and_reduced_views(self):
        r=solve(['5','x+6'],mode='vectors')
        self.certified(r)
        p=postprocess_basis(r,reduced=True)
        self.assertEqual([str(f) for f in p.reduced_basis],['5','x + 1'])

    def test_37_timing_partition(self):
        r=solve();t=r.stats.to_dict()['timings_seconds']
        from gvwzz.structures import TIMERS
        self.assertAlmostEqual(t['total'],sum(t[k] for k in TIMERS)+t['incremental_minimization']+t['other'],places=8)
        self.assertAlmostEqual(t['syzygy'],t['syzygy_check']+t['syzygy_complete'],places=8)

    def test_38_stage_results_match_independent_prefix_runs(self):
        from gvwzz.core import divides_monomial
        original=_minimal_records
        def inspect(engine):
            chosen=original(engine)
            old=compute(engine.inputs[:engine.position+1],engine.backend.names,
                        engine.backend.order_name,'pot',show_basis=False,
                        options=Options(max_seconds=5))
            expected={tuple(sorted(f.items())) for f in postprocess_basis(old).minimal_leading_terms}
            actual={tuple(sorted(engine.backend.term(abs(engine.leading[j][1]),engine.leading[j][0]).items())) for j,_ in chosen}
            self.assertEqual(actual,expected)
            return chosen
        with patch('gvwzz.incremental._minimal_records',inspect):
            solve(['2*x','3*y','x*y-1','x+y+2'])

    def test_39_disabling_mode_keeps_normal_signature_output(self):
        normal=compute(PAPER,('x','y'),'lex','pot',options=Options(),show_basis=False)
        self.assertTrue(normal.to_dict()['solver_claims_signature_basis'])
        self.assertIsNone(normal.incremental_metadata)
        self.assertGreater(len(normal.syzygies),0)

    def test_40_stage_progress_is_optional(self):
        output=io.StringIO()
        with redirect_stderr(output):r=solve(progress_interval=100)
        self.assertIn('[stage 3/3 completed]',output.getvalue())
        output=io.StringIO()
        with redirect_stderr(output):solve()
        self.assertEqual(output.getvalue(),'')


    def test_41_direct_coordinator_validates_and_marks_mode(self):
        from gvwzz import IncrementalPOT
        from gvwzz.polynomial import IntegerPolynomialRing
        from gvwzz.orders import ModuleOrder
        b=IntegerPolynomialRing(['x'],'grevlex')
        f=[b.parse('2*x')]
        o=ModuleOrder(b,f,'pot')
        r=IncrementalPOT(b,f,o,Options(track_representations=True)).run()
        self.assertTrue(r.options.incremental_pot)
        self.assertTrue(verify_result(r,timeout=0)['strong_basis_certified'])
        with self.assertRaises(ValueError):
            IncrementalPOT(b,f,ModuleOrder(b,f,'gvw-g2'))

    def test_42_partial_paths_never_certify_full_basis(self):
        r=solve(PAPER,['x','y'],max_basis=2,mode='paths')
        v=verify_generation_paths(r,criteria=False,timeout=0)
        self.assertTrue(v['passed'],v)
        self.assertFalse(v['strong_basis_certified'])
        self.assertFalse(v['signature_basis_certified'])
        self.assertEqual(v['status'],'prefix_paths_verified')
        self.assertFalse(verify_generation_paths(r,criteria=True,timeout=0)['passed'])

if __name__=='__main__':
    unittest.main(verbosity=2)
