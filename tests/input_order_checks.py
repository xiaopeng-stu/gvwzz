"""Input-order regression checks; no large benchmark computation is started."""
from __future__ import annotations
import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from gvwzz import Options, compute, reverse_generators, verify_generation_paths
from gvwzz.cases import CASES, SUITES, get_case, input_metadata, reverse_case_generators
from gvwzz.cli import (build_parser, main, load_case, resolve_orders, load_result,
                       execute_case, flatten, options_from_args, parse_positions)
from gvwzz.verification import verify_result
ROOT=Path(__file__).resolve().parents[1]
PAPER={'name':'paper','variables':['x','y'],'polynomials':['4*x^2','9*y^4','6*x*y^2+y']}
def cli(*args):
    out,err=io.StringIO(),io.StringIO()
    with redirect_stdout(out),redirect_stderr(err): rc=main(list(args))
    return rc,out.getvalue(),err.getvalue()
def solve(case,inc=False,paths=False,vectors=False,order='pot'):
    return compute(case['polynomials'],case['variables'],'lex',order,
        options=Options(incremental_pot=inc,record_paths=paths,track_representations=vectors,max_seconds=10),
        show_basis=False,show_u=False)
def state(r):
    d=r.to_dict()
    return {k:d.get(k) for k in ('input_polynomials','basis','syzygy_signatures','insertion_history',
                                'representation_certificate','generation_paths','counters')}
class InputOrderChecks(unittest.TestCase):
    def test_01_default_off(self):
        for argv in (['run','--case','eco3'],['export-case','eco3'],['benchmark'],['random']):
            self.assertFalse(build_parser().parse_args(argv).reverse_generators)
    def test_02_aliases(self):
        for f in ('--reverse-generators','--reversed'):
            self.assertTrue(build_parser().parse_args(['run','--case','eco3',f]).reverse_generators)
    def test_03_repeated_flag_is_not_a_toggle(self):
        a=build_parser().parse_args(['run','--case','eco3','--reversed','--reversed'])
        self.assertEqual(load_case(a)['polynomials'],get_case('eco3')['polynomials'][::-1])
    def test_04_disable(self):
        for f in ('--no-reverse-generators','--no-reversed'):
            a=build_parser().parse_args(['run','--case','eco3','--reversed',f])
            self.assertEqual(load_case(a),get_case('eco3'))
    def test_05_no_strategy_change(self):
        p=build_parser()
        a=p.parse_args(['run','--case','eco3']);b=p.parse_args(['run','--case','eco3','--reversed'])
        self.assertEqual(asdict(options_from_args(a)),asdict(options_from_args(b)))
    def test_06_catalog_cleanup(self):
        self.assertEqual(len(CASES),109)
        self.assertFalse(any('reversed' in n for n in CASES))
        self.assertFalse(any('reversed' in n for v in SUITES.values() for n in v))
        rc,out,err=cli('list');self.assertEqual(rc,0,err);self.assertNotIn('reversed',out)
    def test_07_removed_name_migration(self):
        with self.assertRaisesRegex(ValueError,'--reverse-generators'):get_case('eder_yang1_reversed')
        rc,out,err=cli('export-case','eder_yang1_reversed')
        self.assertEqual(rc,1);self.assertIn('--reverse-generators',err)
    def test_08_all_builtin_inputs(self):
        for name in CASES:
            with self.subTest(case=name):
                a=deepcopy(get_case(name));b=get_case(name,reverse_generators=True)
                self.assertEqual(b['polynomials'],a['polynomials'][::-1]);self.assertEqual(b['name'],name)
                for k in ('variables','polynomial_order','module_order','positions','signature_blocks'):
                    self.assertEqual(b.get(k),a.get(k))
                self.assertEqual(b['input_order']['effective_to_source'],list(range(len(a['polynomials']),0,-1)))
                self.assertEqual(get_case(name),a)
    def test_09_no_shared_mutation(self):
        a=deepcopy(get_case('eder_yang1'));b=get_case('eder_yang1',reverse_generators=True)
        b['polynomials'][0]='0';b['variables'][0]='changed';b['provenance']['note']='changed'
        b['input_order']['source_metadata']['provenance']['note']='changed'
        self.assertEqual(get_case('eder_yang1'),a)
    def test_10_yang_fingerprint(self):
        a=get_case('eder_yang1');b=get_case('eder_yang1',reverse_generators=True)
        self.assertEqual(input_metadata(a)['input_sha256'],
                         'b29eb0f7692701ed7aa7ca52913d41cd1a778fe9d7067ba8bd6b9739a2dcb21a')
        self.assertNotEqual(input_metadata(a)['input_sha256'],input_metadata(b)['input_sha256'])
        self.assertEqual(len(b['polynomials']),66)
        for i in range(66):self.assertEqual(b['polynomials'][i],a['polynomials'][65-i])
    def test_11_empty_singleton_zeros_duplicates(self):
        for ps in ([],['-6*x'],['0','2*x','2*x','-3*x','0']):
            c={'variables':['x'],'polynomials':ps}
            self.assertEqual(reverse_case_generators(c)['polynomials'],ps[::-1])
    def test_12_coefficients_and_scaled_input_unchanged(self):
        a={'variables':['x','y'],'polynomials':['-6*x+12*y','10*x^2','2*y']}
        self.assertEqual(reverse_case_generators(a)['polynomials'],['2*y','10*x^2','-6*x+12*y'])
        c=get_case('cyclic3_scaled')
        self.assertEqual(get_case('cyclic3_scaled',reverse_generators=True)['polynomials'],c['polynomials'][::-1])
    def test_13_denominator_metadata(self):
        a=get_case('henrion5');b=reverse_case_generators(a)
        self.assertEqual(b['provenance']['denominator_multipliers'],a['provenance']['denominator_multipliers'][::-1])
        self.assertEqual(b['input_order']['source_metadata']['provenance'],a['provenance'])
    def test_14_positions_and_blocks_unchanged(self):
        a=dict(PAPER,positions=[3,1,2],signature_blocks=[1,2]);b=reverse_case_generators(a)
        self.assertEqual(b['positions'],a['positions']);self.assertEqual(b['signature_blocks'],a['signature_blocks'])
        self.assertEqual(parse_positions(None,b),[2,0,1])
    def test_15_two_reversals_restore_input_and_annotations(self):
        for n in ('henrion5','eder_yang1','katsura2_qfirst','cyclic3_scaled','all_zero'):
            a=get_case(n);b=reverse_case_generators(reverse_case_generators(a));info=b.pop('input_order')
            self.assertEqual(b,a);self.assertEqual(info['direction'],'original')
            self.assertEqual(info['effective_to_source'],list(range(1,len(a['polynomials'])+1)))
    def test_16_bad_arguments(self):
        for x in (1,'false',None):
            with self.assertRaises(ValueError):get_case('eco3',reverse_generators=x)
        for x in (None,[],{'polynomials':'x'}):
            with self.assertRaises(ValueError):reverse_case_generators(x)
    def test_17_bad_tracking_metadata(self):
        base=reverse_case_generators(PAPER)
        for k,v in (('effective_to_source',[3,3,1]),('effective_to_source',[True,2,3]),
                    ('effective_to_source',[1,2,3]),('reversals',0),('reversals',True),('source_metadata',None)):
            a=deepcopy(base);a['input_order'][k]=v
            with self.subTest(k=k,v=v),self.assertRaises(ValueError):reverse_case_generators(a)
    def test_18_stale_fingerprint_removed(self):
        a=dict(PAPER,benchmark_input={'input_sha256':'stale'});b=reverse_case_generators(a)
        self.assertNotIn('benchmark_input',b);self.assertEqual(a['benchmark_input']['input_sha256'],'stale')
        self.assertNotEqual(input_metadata(b)['input_sha256'],input_metadata(PAPER)['input_sha256'])
    def test_19_builtin_load_auto_pot(self):
        a=build_parser().parse_args(['run','--case','eder_yang1','--reversed','--incremental-pot'])
        c=load_case(a);self.assertEqual(resolve_orders(a,c),('grevlex','pot',None))
        self.assertEqual(c['polynomials'][0],get_case('eder_yang1')['polynomials'][-1])
    def test_20_inline_input(self):
        a=build_parser().parse_args(['run','--vars','x,y','--poly','2*x','--poly','3*y','--reversed'])
        c=load_case(a);self.assertEqual(c['polynomials'],['3*y','2*x']);self.assertEqual(c['variables'],['x','y'])
    def test_21_json_input_not_modified(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'in.json';p.write_text(json.dumps(PAPER));raw=p.read_bytes()
            a=build_parser().parse_args(['run','--input',str(p),'--reversed'])
            self.assertEqual(load_case(a)['polynomials'],PAPER['polynomials'][::-1]);self.assertEqual(p.read_bytes(),raw)
    def test_22_export_is_already_transformed(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'in.json';rc,out,err=cli('export-case','eco3','--reversed','--output',str(p))
            self.assertEqual(rc,0,err);d=json.loads(p.read_text());a=build_parser().parse_args(['run','--input',str(p)])
            self.assertEqual(load_case(a)['polynomials'],d['polynomials'])
            self.assertEqual(d['polynomials'],get_case('eco3')['polynomials'][::-1])
            self.assertEqual(d['benchmark_input']['input_sha256'],input_metadata(d)['input_sha256'])
            a=build_parser().parse_args(['run','--input',str(p),'--reversed'])
            b=load_case(a);self.assertEqual(b['polynomials'],get_case('eco3')['polynomials'])
            self.assertEqual(b['input_order']['direction'],'original')
    def test_23_public_helper(self):
        self.assertEqual(reverse_generators(PAPER),reverse_case_generators(PAPER))
    def test_24_run_json_csv_console(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'r.json'
            rc,out,err=cli('run','--vars','x,y','--poly','2*x','--poly','3*y','--reversed','--certificates',
                '--verify','--verify-mode','static','--no-show-basis','--no-show-u','--output',str(p))
            self.assertEqual(rc,0,err+out);self.assertIn('Input generator order: reversed',out)
            d=json.loads(p.read_text());r=load_result(p)
            self.assertEqual([str(f) for f in r.inputs],['3*y','2*x']);self.assertEqual(d['case'],'custom')
            self.assertEqual(d['input_order']['effective_to_source'],[2,1]);self.assertTrue(d['verification']['membership_verified'])
            with p.with_suffix('.csv').open(encoding='utf-8-sig',newline='') as fh:row=next(csv.DictReader(fh))
            self.assertEqual(row['input_generator_order'],'reversed');self.assertEqual(json.loads(row['input_generator_indices']),[2,1])
    def test_25_benchmark_each_repeat_once(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'b';rc,out,err=cli('benchmark','--cases','gcd_line,zero_and_signs','--module-orders','pot',
                '--repeat','2','--reversed','--certificates','--verify','--verify-mode','static','--output',str(p))
            self.assertEqual(rc,0,err+out)
            rows=[json.loads(s) for s in Path(str(p)+'.raw.jsonl').read_text().splitlines()];self.assertEqual(len(rows),4)
            for r in rows:
                c=get_case(r['case'],reverse_generators=True)
                self.assertEqual(r['input_order'],c['input_order'])
                self.assertEqual(r['benchmark_input']['input_sha256'],input_metadata(c)['input_sha256'])
                self.assertTrue(r['verification']['passed'],r)
    def test_26_random_export(self):
        with tempfile.TemporaryDirectory() as t:
            a,b=Path(t)/'a.json',Path(t)/'b.json'
            self.assertEqual(cli('random','--seed','10','--output',str(a))[0],0)
            self.assertEqual(cli('random','--seed','10','--reversed','--output',str(b))[0],0)
            x,y=json.loads(a.read_text()),json.loads(b.read_text())
            self.assertEqual(y['polynomials'],x['polynomials'][::-1]);self.assertEqual(y['variables'],x['variables'])
    def test_27_manual_equivalence_and_vectors_both_modes(self):
        manual=dict(PAPER,polynomials=PAPER['polynomials'][::-1]);transformed=reverse_case_generators(PAPER)
        for inc in (False,True):
            a=solve(manual,inc=inc,vectors=True);b=solve(transformed,inc=inc,vectors=True)
            self.assertEqual(a.status,'completed',a.message);self.assertEqual(state(a),state(b))
            v=verify_result(b,require_certificates=True,timeout=0,mode='static',max_reduction_steps=0)
            self.assertTrue(v['passed'],v);self.assertTrue(v['membership_verified'])
            self.assertTrue(v['strong_basis_certified'] if inc else v['signature_basis_certified'])
    def test_28_paths_reload_both_modes(self):
        c=reverse_case_generators(PAPER)
        for inc in (False,True):
            with self.subTest(incremental=inc),tempfile.TemporaryDirectory() as t:
                data,r=execute_case(c,'lex','pot',Options(incremental_pot=inc,record_paths=True),
                    return_result=True,verify=True,verify_mode='static',verify_timeout=0,verify_max_steps=0)
                self.assertTrue(data['verification']['passed'],data['verification'])
                p=Path(t)/'r.json';p.write_text(json.dumps(data))
                v=verify_generation_paths(load_result(p),timeout=0,max_reduction_steps=0)
                self.assertTrue(v['passed'],v);self.assertTrue(v['membership_verified'])
    def test_29_other_module_orders(self):
        c=reverse_case_generators(PAPER)
        for order in ('g2','top','schreyer'):
            with self.subTest(order=order):
                r=solve(c,vectors=True,order=order)
                v=verify_result(r,require_certificates=True,timeout=0,mode='static',max_reduction_steps=0)
                self.assertTrue(v['passed'],v)
                self.assertEqual([str(f) for f in r.inputs],['6*x*y**2 + y','9*y**4','4*x**2'])
    def test_30_summary_only_and_quiet_ideal(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'r.json';rc,out,err=cli('run','--case','gcd_line','--reversed','--summary-only',
                '--no-print-ideal','--no-show-basis','--output',str(p))
            self.assertEqual(rc,0,err+out);self.assertIn('Input generator order: reversed',out)
            d=json.loads(p.read_text());self.assertIn('input_order',d);self.assertNotIn('basis',d)
    def test_31_default_has_no_transform_annotation(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'r.json';rc,out,err=cli('run','--case','gcd_line','--no-show-basis','--output',str(p))
            self.assertEqual(rc,0,err+out);d=json.loads(p.read_text())
            self.assertNotIn('input_order',d);self.assertNotIn('Input generator order:',out)
            self.assertNotIn('input_generator_order',flatten(d))
    def test_32_export_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'r.json';self.assertEqual(cli('export-case','eco3','--output',str(p))[0],0);raw=p.read_bytes()
            self.assertEqual(cli('export-case','eco3','--reversed','--output',str(p))[0],1);self.assertEqual(raw,p.read_bytes())
    def test_33_removed_files(self):
        for f in ('examples/eder_yang1_reversed.json','YANG1_REVERSED.md','yang1_order_checks.py'):
            self.assertFalse((ROOT/f).exists())
    def test_34_yang_reversed_two_generator_prefix(self):
        c=get_case('eder_yang1',reverse_generators=True);c['polynomials']=c['polynomials'][:2]
        for paths in (False,True):
            r=solve(c,inc=True,paths=paths,vectors=not paths)
            v=(verify_generation_paths(r,timeout=0,max_reduction_steps=0) if paths else
               verify_result(r,require_certificates=True,timeout=0,mode='static',max_reduction_steps=0))
            self.assertTrue(v['passed'],v);self.assertTrue(v['strong_basis_certified'])
if __name__=='__main__':unittest.main(verbosity=2)
