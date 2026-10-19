"""Regression checks for the G-only tail switch and source-aware diagnostics."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest

from gvwzz import Options, compute, GVWInteger, reconstruct_syzygies, verify_generation_paths
from gvwzz.cli import build_parser, options_from_args, main, load_result, write_json, timing_report
from gvwzz.core import Label, Signature
from gvwzz.orders import ModuleOrder
from gvwzz.polynomial import IntegerPolynomialRing
from gvwzz.structures import Stats
from gvwzz.solver import ComputationLimit
from gvwzz.verification import verify_result

F = ['4*x^2', '9*y^4', '6*x*y^2+y']

def solve(**settings):
    return compute(F, ['x', 'y'], 'grevlex', 'pot',
                   options=Options(max_seconds=10, max_candidates=100000,
                                   **settings), show_basis=False, show_u=False)

def engine(**settings):
    b = IntegerPolynomialRing(['x'], 'lex')
    fs = [b.parse('6*x+1'), b.parse('10*x+2')]
    return GVWInteger(b, fs, ModuleOrder(b, fs, 'pot'), Options(**settings))

class TailChecks(unittest.TestCase):
    def test_01_default_and_type_validation(self):
        self.assertTrue(Options().g_tail_reduce)
        for invalid in (None, 0, 1, 'off'):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                Options(g_tail_reduce=invalid).validate()

    def test_02_cli_run_and_benchmark(self):
        for command in ('run', 'benchmark'):
            for flags, expected in (([], True), (['--g-tail-reduce'], True),
                                    (['--no-g-tail-reduce'], False)):
                with self.subTest(command=command, flags=flags):
                    args = build_parser().parse_args([command, *flags])
                    self.assertEqual(options_from_args(args).g_tail_reduce, expected)

    def test_03_conflicting_flags_rejected(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(['run', '--g-tail-reduce', '--no-g-tail-reduce'])

    def test_04_skip_only_g_and_return_original_label(self):
        s = engine(g_tail_reduce=False)
        label = Label(Signature(1, (0,), 1), s.backend.parse('2*x+3'))
        with patch.object(s, '_reduce_tail', return_value=label) as tail:
            self.assertIs(s._reduce_candidate_tail(label, 'G'), label)
            tail.assert_not_called()
            self.assertIs(s._reduce_candidate_tail(label, 'J'), label)
            tail.assert_called_once_with(label)
        self.assertEqual(s.stats.counters['g_tail_skipped'], 1)
        self.assertEqual(s.stats.counters['g_tail_steps'], 0)

    def test_05_master_switch(self):
        for flag in (True, False):
            with self.subTest(g_tail=flag):
                r = solve(tail_reduce=False, g_tail_reduce=flag)
                self.assertEqual(r.status, 'completed')
                self.assertEqual(r.stats.counters['tail_steps'], 0)

    def test_06_live_progress_includes_current_call(self):
        s = engine(progress_interval=1)
        s.current_candidate_kind = 'G'
        s.current_candidate_serial = 7
        s.phase = 'G tail reduction'
        s._tail_kind, s._tail_steps_start = 'G', 10
        s.stats.counters.update(j_tail_steps=3, g_tail_steps=7, tail_steps=15)
        out = StringIO()
        with redirect_stderr(out):
            s._check_budget()
        text = out.getvalue()
        self.assertIn('phase=G tail reduction candidate=G#7', text)
        self.assertIn('tail(J/G)=3/12', text)
        self.assertIn('popped(J/G)=', text)

    def test_07_exception_accounting(self):
        s = engine()
        label = Label(Signature(1, (0,), 1), s.backend.parse('2*x+3'))
        def interrupted(_):
            s.stats.counters['tail_steps'] += 3
            s.stats.counters['tail_exact_steps'] += 2
            s.stats.counters['tail_euclidean_steps'] += 1
            s.stats.counters['tail_reduction_calls'] += 1
            raise ComputationLimit('test interruption')
        with patch.object(s, '_reduce_tail', side_effect=interrupted), self.assertRaises(ComputationLimit):
            s._reduce_candidate_tail(label, 'G')
        self.assertEqual(s.stats.counters['g_tail_steps'], 3)
        self.assertEqual(s.stats.counters['g_tail_exact_steps'], 2)
        self.assertEqual(s.stats.counters['g_tail_euclidean_steps'], 1)
        self.assertIsNone(s._tail_kind)
        self.assertEqual(s.stats.times_ns['g_tail_reduction'], s.stats.times_ns['tail_reduction'])

    def test_08_timing_does_not_double_count(self):
        s = Stats()
        s.total_ns = 1000000000
        s.times_ns.update(tail_reduction=500000000, j_tail_reduction=200000000,
                          g_tail_reduction=300000000)
        t = s.to_dict()['timings_seconds']
        self.assertEqual(t['other'], .5)
        out = StringIO()
        with redirect_stdout(out):
            timing_report(t)
        self.assertIn('1.000000s / 1.000000s', out.getvalue())
        self.assertNotIn('New-element tail reductions', out.getvalue())
        old = Stats().to_dict()['timings_seconds']
        self.assertNotIn('j_tail_reduction', old)
        out = StringIO()
        with redirect_stdout(out): timing_report(old)
        self.assertIn('New-element tail reductions', out.getvalue())

    def test_09_real_kernel_and_mode_matrix(self):
        for kernel in ('copy', 'heap', 'packed'):
            for flag in (True, False):
                with self.subTest(kernel=kernel, g_tail=flag):
                    r = solve(g_tail_reduce=flag, reduction_kernel=kernel,
                              track_representations=True, debug=True)
                    self.assertEqual(r.status, 'completed')
                    for mode in ('static', 'naive'):
                        report=verify_result(r, mode=mode, require_certificates=True)
                        self.assertTrue(report['signature_basis_certified'], report)
                    p=solve(g_tail_reduce=flag, reduction_kernel=kernel, record_paths=True, debug=True)
                    self.assertEqual(r.basis, p.basis)
                    self.assertEqual(r.syzygies, p.syzygies)
                    self.assertTrue(verify_generation_paths(p)['signature_basis_certified'])
                    self.assertEqual(reconstruct_syzygies(p, verify=True).syzygies,
                                     r.syzygy_representations)
                    c=r.stats.counters
                    self.assertEqual(c['tail_steps'], c['j_tail_steps']+c['g_tail_steps'])
                    if not flag:
                        self.assertEqual(c['g_tail_steps'], 0)
                        self.assertEqual(c['g_tail_skipped'], c['g_inserted'])
                        self.assertGreater(c['g_inserted'], 0)

    def test_10_roundtrip_and_legacy_default(self):
        r = solve(g_tail_reduce=False, record_paths=True)
        with TemporaryDirectory() as d:
            p=Path(d)/'new.json'; write_json(p, r.to_dict())
            loaded=load_result(p)
            self.assertFalse(loaded.options.g_tail_reduce)
            self.assertTrue(verify_generation_paths(loaded)['signature_basis_certified'])
            data=solve().to_dict()
            del data['options']['g_tail_reduce']
            write_json(p, data)
            self.assertTrue(load_result(p).options.g_tail_reduce)

    def test_11_real_cli_run(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'run.json'; out=StringIO()
            args=['run','--vars','x,y','--poly',F[0],'--poly',F[1],'--poly',F[2],
                  '--poly-order','grevlex','--module-order','pot','--no-g-tail-reduce',
                  '--record-paths','--verify','--verify-mode','static','--no-show-basis',
                  '--no-show-u','--no-print-ideal','--output',str(p)]
            with redirect_stdout(out), redirect_stderr(StringIO()):
                self.assertEqual(main(args),0)
            data=json.loads(p.read_text())
            self.assertFalse(data['options']['g_tail_reduce'])
            self.assertEqual(data['counters']['g_tail_steps'],0)
            self.assertTrue(data['verification']['signature_basis_certified'])
            self.assertIn('Tail enabled: J=on | G=off',out.getvalue())

    def test_12_real_benchmark_cli(self):
        with TemporaryDirectory() as d, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            base=Path(d)/'bench'
            self.assertEqual(main(['benchmark','--cases','gcd_line','--poly-orders','grevlex',
                '--module-orders','pot','--repeat','1','--no-g-tail-reduce','--output',str(base)]),0)
            data=json.loads(Path(str(base)+'.raw.jsonl').read_text().splitlines()[0])
            self.assertFalse(data['options']['g_tail_reduce'])

    def test_13_selective_and_local_cap(self):
        for policy in ('all','selective'):
            for cap in (0,1):
                with self.subTest(policy=policy,cap=cap):
                    r=solve(g_tail_reduce=False,tail_policy=policy,tail_step_cap=cap,
                            track_representations=True)
                    self.assertEqual(r.status,'completed')
                    self.assertEqual(r.stats.counters['g_tail_steps'],0)
                    self.assertTrue(verify_result(r,mode='static',require_certificates=True)['passed'])

    def test_14_boundaries(self):
        for fs in ([],['0'],['0','0'],['6','10']):
            with self.subTest(fs=fs):
                r=compute(fs,['x'],options=Options(g_tail_reduce=False),show_basis=False)
                self.assertEqual(r.status,'completed')
                self.assertEqual(r.stats.counters['g_tail_steps'],0)

    def test_15_invalid_source(self):
        s=engine(); label=Label(Signature(1,(0,),0),s.inputs[0])
        with self.assertRaises(ValueError): s._reduce_candidate_tail(label,'S')

    def test_16_unmodified_g_has_legal_reducible_tail(self):
        from gvwzz.cases import get_case
        c=get_case('katsura4')
        on=compute(c['polynomials'],c['variables'],'grevlex','gvw-g2',
                   options=Options(g_tail_reduce=True),show_basis=False)
        off=compute(c['polynomials'],c['variables'],'grevlex','gvw-g2',
                    options=Options(g_tail_reduce=False,track_representations=True),show_basis=False)
        self.assertGreater(on.stats.counters['g_tail_steps'],0)
        self.assertEqual(off.stats.counters['g_tail_steps'],0)
        self.assertGreater(off.stats.counters['j_tail_steps'],0)
        self.assertTrue(verify_result(off,mode='static',require_certificates=True)['passed'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
