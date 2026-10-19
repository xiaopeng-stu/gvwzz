"""Regression checks for minimal/reduced strong polynomial bases over ZZ."""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from dataclasses import replace
from io import StringIO
from itertools import combinations, product, permutations
from math import gcd
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import random
import unittest

from gvwzz import (Options, Result, compute, postprocess_basis, minimal_groebner_basis,
                   reduced_groebner_basis, minimal_leading_terms, format_result,
                   verify_generation_paths, reconstruct_syzygies)
from gvwzz.cli import build_parser, main, load_result, write_json
from gvwzz.core import Label, Signature, divides_monomial, lcm_monomials, subtract_monomials
from gvwzz.orders import ModuleOrder
from gvwzz.output import write_json as write_streamed_json
from gvwzz.polynomial import IntegerPolynomialRing
from gvwzz.postprocessing import (balanced_divmod, PolynomialPostprocessingError,
                                 _minimal_projection, _validate_minimal_projection)
from gvwzz.structures import Stats
from gvwzz.verification import strong_remainder, verify_result

FLAGS = ('show_minimal_gb', 'show_reduced_gb', 'show_minimal_lt')
PAPER = ['4*x^2', '9*y^4', '6*x*y^2+y']


def solve(fs, variables=('x', 'y'), order='lex', module='pot', **kwargs):
    return compute(fs, variables, order, module,
                   options=Options(max_seconds=15, max_candidates=200000, **kwargs),
                   show_basis=False, show_u=False)


def known_basis(fs, variables=('x', 'y'), order='lex', **kwargs):
    """A structurally valid result for testing polynomial post-processing alone.

    The tests supply a known strong polynomial basis; this helper does not
    assert that these artificial signature labels form a signature basis.
    """
    b= IntegerPolynomialRing(variables, order)
    polys=[b.parse(f) for f in fs]
    labels=[Label(Signature(1,b.zero_monomial,i), f) for i,f in enumerate(polys)]
    return Result('completed','test polynomial basis',b,ModuleOrder(b,polys,'pot'),
                  polys,labels,[],Stats(),Options(**kwargs))


def assert_strong(test, source, output):
    """Independent ordinary S/G checks and mutual ideal inclusions."""
    b=source.backend
    old=[p.polynomial for p in source.basis]
    for f in old:
        test.assertFalse(strong_remainder(f,output,b))
    for f in output:
        test.assertFalse(strong_remainder(f,old,b))
    for f,g in combinations(output,2):
        M,c=b.leading(f);N,d=b.leading(g)
        L=lcm_monomials(M,N);mu=subtract_monomials(L,M);nu=subtract_monomials(L,N)
        lc=abs(c*d)//gcd(c,d)
        sp=b.multiply_term(f,lc//c,mu)-b.multiply_term(g,lc//d,nu)
        a,z,gg=b.bezout(c,d)
        gp=b.multiply_term(f,a,mu)+b.multiply_term(g,z,nu)
        test.assertFalse(strong_remainder(sp,output,b))
        test.assertFalse(strong_remainder(gp,output,b))


class PolynomialChecks(unittest.TestCase):
    def test_indexed_minimal_projection_matches_scan(self):
        """Compare generic ZZ coefficients, order, signs and legacy counters."""
        rng = random.Random(210015)
        for nvars in (1, 4, 48):
            names = tuple('x' + str(i) for i in range(nvars))
            for order in ('lex', 'grlex', 'grevlex'):
                b = IntegerPolynomialRing(names, order)
                monomials = []
                for _ in range(75):
                    m = [0] * nvars
                    for _ in range(rng.randrange(1, min(5, nvars + 1))):
                        m[rng.randrange(nvars)] = rng.choice((0, 1, 2, 3, 257, 10000))
                    monomials.append(tuple(m))
                # Repeated monomials, associates, incomparable coefficients,
                # and arbitrary-precision coefficients exercise ZZ semantics.
                coefficients = (2, 3, 5, 6, 10, 15, 30, 10**90 + 7)
                polys = [b.term(rng.choice((-1, 1)) * rng.choice(coefficients),
                                rng.choice(monomials)) for _ in range(180)]
                r = known_basis(polys, names, order)
                rows = [(m, abs(c), i, 1 if c > 0 else -1)
                        for i, f in enumerate(polys) for m, c in [b.leading(f)]]
                rows.sort(key=lambda row: (b.order_key(row[0]), row[1], row[2]))
                expected, comparisons = [], 0
                for row in rows:
                    for n, d, _, _ in expected:
                        comparisons += 1
                        if row[1] % d == 0 and divides_monomial(n, row[0]):
                            break
                    else:
                        expected.append(row)
                actual, minimal, count = _minimal_projection(r)
                self.assertEqual(actual, expected)
                self.assertEqual(count, comparisons)
                self.assertEqual(minimal, [polys[i] if sign > 0 else -polys[i]
                                           for _, _, i, sign in expected])

    def test_indexed_validation_checks_final_output_independently(self):
        rows = [((1, 0), 2, 0, 1), ((0, 1), 3, 1, 1),
                ((1, 1), 6, 2, 1)]
        _validate_minimal_projection(rows, rows[:2], 2)
        with self.assertRaisesRegex(PolynomialPostprocessingError, 'lost a source'):
            _validate_minimal_projection(rows, rows[:1], 2)
        with self.assertRaisesRegex(PolynomialPostprocessingError, 'not an antichain'):
            _validate_minimal_projection(rows, list(reversed(rows)), 2)
        # A broken selection index must not silently pass verification.
        r = known_basis(['2*x', '4*x^2'], ('x',))
        with patch('gvwzz.postprocessing.SyzygyIndex.divisors', return_value=iter(())):
            with self.assertRaisesRegex(PolynomialPostprocessingError, 'not an antichain'):
                _minimal_projection(r)

    def test_balanced_residue_exhaustive(self):
        for a,b in product(range(-80,81),range(1,31)):
            q,r=balanced_divmod(a,b)
            self.assertEqual(a,q*b+r)
            self.assertTrue(-b < 2*r <= b)
            rank=lambda n:(abs(n),n<0)
            self.assertTrue(all(rank(r)<=rank(a-k*b) for k in range(q-2,q+3)))

    def test_balanced_big_integers_and_invalid_types(self):
        for a,b in ((10**5000+3,10**4500+2),(-10**5000-7,4),(2,4),(-2,4)):
            q,r=balanced_divmod(a,b);self.assertEqual(a,q*b+r);self.assertTrue(-b<2*r<=b)
        for a,b in ((True,4),(3.0,4),(1,False),(1,0),(1,-2)):
            with self.assertRaises((ValueError,TypeError)):
                balanced_divmod(a,b)

    def test_paper_example(self):
        r=solve(PAPER,g_first=False)
        p=postprocess_basis(r,reduced=True)
        expected=[r.backend.parse('y'),r.backend.parse('4*x^2')]
        self.assertEqual(p.minimal_basis,expected)
        self.assertEqual(p.reduced_basis,expected)
        self.assertEqual(p.minimal_leading_terms,expected)
        assert_strong(self,r,p.reduced_basis)

    def test_minimal_is_not_a_minimum_ideal_generating_set(self):
        r=solve(['2*x','3*y'])
        p=postprocess_basis(r,reduced=True)
        self.assertEqual(set(map(str,p.reduced_basis)),{'2*x','3*y','x*y'})
        self.assertEqual(len(p.minimal_basis),3)
        assert_strong(self,r,p.reduced_basis)

    def test_never_divide_by_content(self):
        for fs,expected in ((['2*x'],['2*x']), (['-6'],['6']),(['6*x','10*x'],['2*x'])):
            r=solve(fs,variables=('x',));p=postprocess_basis(r,reduced=True)
            self.assertEqual(list(map(str,p.reduced_basis)),expected)
            assert_strong(self,r,p.reduced_basis)

    def test_exact_irreducibility_does_not_suffice(self):
        r=known_basis(['5','x+6'],variables=('x',))
        p=postprocess_basis(r,reduced=True)
        self.assertEqual(list(map(str,p.minimal_basis)),['5','x + 6'])
        self.assertEqual(list(map(str,p.reduced_basis)),['5','x + 1'])
        self.assertEqual(p.statistics['coefficient_remainder_steps'],1)

    def test_positive_half_modulus_tie(self):
        for c in ('2','-2','6','-6'):
            r=known_basis(['4',f'x+({c})'],variables=('x',))
            p=postprocess_basis(r,reduced=True)
            self.assertEqual(str(p.reduced_basis[1]),'x + 2')
        p=postprocess_basis(known_basis(['4','x-2'],variables=('x',)),reduced=True)
        self.assertEqual(p.statistics['equal_absolute_value_tie_steps'],1)

    def test_duplicate_and_associate_lts(self):
        r=known_basis(['-5','5','-x-6','x+1','2*x+12'],variables=('x',))
        p=postprocess_basis(r,reduced=True)
        self.assertEqual(p.source_indices,[0,2])
        self.assertEqual(p.source_signs,[-1,-1])
        self.assertEqual(list(map(str,p.reduced_basis)),['5','x + 1'])

    def test_same_monomial_different_coefficients(self):
        r=known_basis(['6*x','10*x','-2*x','4*x^2'],variables=('x',))
        p=postprocess_basis(r,reduced=True)
        self.assertEqual(list(map(str,p.minimal_leading_terms)),['2*x'])

    def test_zero_ideal(self):
        r=solve(['0','0']);p=postprocess_basis(r,reduced=True)
        self.assertEqual((p.minimal_basis,p.minimal_leading_terms,p.reduced_basis),([],[],[]))
        self.assertIn('empty: zero ideal',format_result(r,show_basis=False,show_reduced_gb=True))

    def test_unit_ideal(self):
        r=solve(['2','3','x']);p=postprocess_basis(r,reduced=True)
        self.assertEqual(list(map(str,p.reduced_basis)),['1'])

    def test_source_records_never_modified(self):
        for opt in ({'track_representations':True},{'record_paths':True},{}):
            r=solve(PAPER,g_first=False,**opt);before=deepcopy(r.to_dict())
            p=postprocess_basis(r,reduced=True)
            self.assertEqual(r.to_dict(),before)
            self.assertTrue(all(all(f is not g.polynomial for g in r.basis) for f in p.minimal_basis))
            self.assertTrue(all(f is not g for f,g in zip(p.minimal_basis,p.reduced_basis)))
            if opt.get('record_paths'):
                self.assertTrue(verify_generation_paths(r)['passed'])
                reconstruct_syzygies(r,verify=True)
            if opt.get('track_representations'):
                self.assertTrue(verify_result(r,require_certificates=True)['passed'])

    def test_only_lts_does_not_reduce(self):
        r=solve(PAPER)
        with patch('gvwzz.postprocessing._reduce_minimal',side_effect=AssertionError):
            txt=format_result(r,show_basis=False,show_minimal_lt=True)
        self.assertIn('Leading terms of the minimal strong',txt)
        self.assertNotIn('Reduced strong Groebner basis',txt)

    def test_no_switch_no_work(self):
        r=solve(PAPER)
        with patch('gvwzz.postprocessing.postprocess_basis',side_effect=AssertionError):
            self.assertEqual(format_result(r,show_basis=False),'')
            solve(['2*x'])
        self.assertIsNone(r.polynomial_postprocessing)
        self.assertNotIn('polynomial_postprocessing',r.to_dict())

    def test_api_convenience(self):
        r=solve(['5','x+6'],variables=('x',))
        self.assertEqual(minimal_groebner_basis(r),postprocess_basis(r).minimal_basis)
        self.assertEqual(reduced_groebner_basis(r),postprocess_basis(r,reduced=True).reduced_basis)
        self.assertEqual(minimal_leading_terms(r),postprocess_basis(r).minimal_leading_terms)
        with self.assertRaises(ValueError):postprocess_basis(r,reduced=1)
        with self.assertRaises(ValueError):format_result(r,show_minimal_lt='yes')

    def test_compute_stores_separate_output(self):
        with redirect_stdout(StringIO()) as sink:
            r=compute(PAPER,['x','y'],'lex','pot',show_basis=False,show_minimal_lt=True,show_reduced_gb=True)
        self.assertEqual(len(r.basis),10)
        self.assertIsNotNone(r.polynomial_postprocessing)
        data=r.to_dict()['polynomial_postprocessing']
        self.assertEqual(data['minimal_basis_size'],2)
        self.assertIn('reduced_basis',data)
        self.assertNotIn('Minimal strong Groebner basis (',sink.getvalue())

    def test_partial_source(self):
        r=solve(['x']);r.status='limit'
        with self.assertRaises(PolynomialPostprocessingError):postprocess_basis(r)
        txt=format_result(r,show_basis=False,show_reduced_gb=True)
        self.assertIn('skipped',txt)
        self.assertNotIn('Reduced strong Groebner basis (',txt)

    def test_malformed_source(self):
        r=solve(['x']);r.basis[0]=Label(r.basis[0].signature,r.backend.zero)
        with self.assertRaises(PolynomialPostprocessingError):postprocess_basis(r)

    def test_detect_bad_leading_coefficients(self):
        r=known_basis(['6*x','10*x'],variables=('x',))
        with self.assertRaises(PolynomialPostprocessingError):postprocess_basis(r,reduced=True)

    def test_solver_caps_do_not_truncate_postprocessing(self):
        r=known_basis(['5','x+6*y^4+7*y^3+8*y^2+9*y+11'],tail_step_cap=1,g_tail_reduce=False)
        p=postprocess_basis(r,reduced=True)
        self.assertEqual(p.statistics['reduction_steps'],5)
        assert_strong(self,r,p.reduced_basis)

    def test_all_polynomial_orders(self):
        for order in ('lex','grlex','grevlex'):
            r=solve(['5','x+6*y+8','y^2+7'],order=order)
            p=postprocess_basis(r,reduced=True)
            assert_strong(self,r,p.reduced_basis)
            q=postprocess_basis(known_basis(p.reduced_basis,order=order),reduced=True)
            self.assertEqual(q.reduced_basis,p.reduced_basis)
            self.assertEqual(q.statistics['reduction_steps'],0)

    def test_canonical_across_source_permutations(self):
        fs=['8','x+5*y+6','y^2+7*y+10']
        for order in ('lex','grlex','grevlex'):
            reference=None
            for perm in permutations(fs):
                r=known_basis(list(perm),order=order)
                p=postprocess_basis(r,reduced=True)
                assert_strong(self,r,p.reduced_basis)
                if reference is None:reference=p.reduced_basis
                self.assertEqual(p.reduced_basis,reference)

    def test_canonical_across_equal_lt_representatives(self):
        for extra in ('x+5*y+6','x+13*y-18','x-11*y+30'):
            r=known_basis(['8',extra,'y^2+7*y+10'])
            p=postprocess_basis(r,reduced=True)
            ref=postprocess_basis(known_basis(['8','x+5*y+6','y^2+7*y+10']),reduced=True)
            self.assertEqual(p.reduced_basis,ref.reduced_basis)

    def test_big_coefficient_serialization(self):
        r=known_basis([-(10**5000)],variables=('x',))
        p=postprocess_basis(r,reduced=True);d=p.to_dict()
        saved=d['reduced_basis'][0]
        self.assertEqual(r.backend.deserialize(saved),p.reduced_basis[0])
        self.assertTrue(saved['terms'][0]['coefficient'].startswith('0x'))

    def test_streamed_postprocessing_matches_eager_records(self):
        """Streaming preserves all records and exact large integer encodings."""
        for r in (known_basis(['5', 'x+6'], variables=('x',)),
                  known_basis([-(10**5000)], variables=('x',)),
                  known_basis([], variables=('x',))):
            for reduced in (False, True):
                post = postprocess_basis(r, reduced=reduced)
                for include_basis in (False, True):
                    expected = post.to_dict(include_basis=include_basis)
                    with TemporaryDirectory() as directory:
                        target = Path(directory) / 'post.json'
                        write_streamed_json(target, post.to_json_dict(include_basis=include_basis))
                        self.assertEqual(json.loads(target.read_text()), expected)

    def test_streamed_postprocessing_is_lazy_and_reusable(self):
        post = postprocess_basis(known_basis(['5', 'x+6'], variables=('x',)),
                                 reduced=True)
        with patch.object(post.backend, 'serialize', wraps=post.backend.serialize) as serialize:
            streamed = post.to_json_dict()
            self.assertEqual(serialize.call_count, 0)
            with TemporaryDirectory() as directory:
                first, second = Path(directory) / 'first.json', Path(directory) / 'second.json'
                write_streamed_json(first, streamed)
                self.assertEqual(serialize.call_count, 6)
                write_streamed_json(second, streamed)
                self.assertEqual(serialize.call_count, 12)
                self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_postprocessing_reuses_saved_positive_source_polynomials(self):
        from gvwzz.output import JsonSavedPolynomial
        r = known_basis(['5', 'x+6'], variables=('x',))
        with TemporaryDirectory() as directory:
            source, target = Path(directory) / 'source.json', Path(directory) / 'post.json'
            receipt = write_streamed_json(source, r.to_json_dict())
            post = postprocess_basis(r)
            expected = post.to_dict()
            raw_source = source.read_bytes()
            with patch.object(post.backend, 'serialize', wraps=post.backend.serialize) as serialize:
                streamed = post.to_json_dict(_source_json=receipt)
                records = list(streamed['minimal_basis'])
                self.assertTrue(all(isinstance(record, JsonSavedPolynomial) for record in records))
                self.assertEqual(serialize.call_count, 0)
                write_streamed_json(target, streamed)
                # Only the one-term leading-term views are serialized anew.
                self.assertEqual(serialize.call_count, len(post.minimal_leading_terms))
            self.assertEqual(json.loads(target.read_text()), expected)
            for index in post.source_indices:
                start, length = receipt.basis_polynomial_spans[index]
                self.assertIn(raw_source[start:start + length], target.read_bytes())

    def test_postprocessing_saved_source_fallbacks_preserve_signs_and_reduced_tails(self):
        r = known_basis(['-5', 'x+6'], variables=('x',))
        with TemporaryDirectory() as directory:
            source, target = Path(directory) / 'source.json', Path(directory) / 'post.json'
            receipt = write_streamed_json(source, r.to_json_dict())
            post = postprocess_basis(r, reduced=True)
            expected = post.to_dict()
            self.assertEqual(post.source_signs, [-1, 1])
            self.assertEqual(list(map(str, post.reduced_basis)), ['5', 'x + 1'])
            with patch.object(post.backend, 'serialize', wraps=post.backend.serialize) as serialize:
                write_streamed_json(target, post.to_json_dict(_source_json=receipt))
                serialized_ids = [id(call.args[0]) for call in serialize.call_args_list]
                self.assertIn(id(post.minimal_basis[0]), serialized_ids)
                self.assertNotIn(id(post.minimal_basis[1]), serialized_ids)
                self.assertTrue(all(id(f) in serialized_ids for f in post.reduced_basis))
            self.assertEqual(json.loads(target.read_text()), expected)

    def test_postprocessing_rejects_unrelated_saved_source_for_reuse(self):
        r = known_basis(['5', 'x+6'], variables=('x',))
        unrelated = known_basis(['7', 'x+8'], variables=('x',))
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'source.json'
            wrong_source = Path(directory) / 'wrong-source.json'
            target = Path(directory) / 'post.json'
            write_streamed_json(source, r.to_json_dict())
            wrong_receipt = write_streamed_json(wrong_source, unrelated.to_json_dict())
            post = postprocess_basis(r)
            expected = post.to_dict()
            with patch.object(post.backend, 'serialize', wraps=post.backend.serialize) as serialize:
                write_streamed_json(target, post.to_json_dict(_source_json=wrong_receipt))
                self.assertEqual(serialize.call_count, 4)
            self.assertEqual(json.loads(target.read_text()), expected)
            # A summary has no source polynomial ranges: every record falls back.
            summary = Path(directory) / 'summary.json'
            summary_receipt = write_streamed_json(summary, r.to_json_dict(include_basis=False))
            with patch.object(post.backend, 'serialize', wraps=post.backend.serialize) as serialize:
                write_streamed_json(target, post.to_json_dict(_source_json=summary_receipt))
                self.assertEqual(serialize.call_count, 4)
            self.assertEqual(json.loads(target.read_text()), expected)

    def test_postprocessing_reports_validation_and_selection_times(self):
        post = postprocess_basis(known_basis(['5', 'x+6'], variables=('x',)))
        times = post.statistics
        self.assertGreaterEqual(times['source_validation_seconds'], 0)
        self.assertGreaterEqual(times['minimal_selection_seconds'], 0)
        self.assertAlmostEqual(times['minimal_seconds'],
                               times['source_validation_seconds'] + times['minimal_selection_seconds'])

    def test_cli_default_flags(self):
        for command in ('run','show'):
            argv=[command]+(['test.json'] if command=='show' else [])
            args=build_parser().parse_args(argv)
            self.assertTrue(all(getattr(args,x) is False for x in FLAGS))
            for flag in FLAGS:
                arg='--'+flag.replace('_','-')
                self.assertTrue(getattr(build_parser().parse_args(argv+[arg]),flag))
                no='--no-'+flag.replace('_','-')
                self.assertFalse(getattr(build_parser().parse_args(argv+[no]),flag))
                with redirect_stderr(StringIO()),self.assertRaises(SystemExit):
                    build_parser().parse_args(argv+[arg,no])

    def test_cli_masks_and_roundtrip(self):
        with TemporaryDirectory() as directory:
            for mask in product((False,True),repeat=3):
                target=Path(directory)/('mask'+''.join(map(str,map(int,mask)))+'.json')
                args=['run','--vars','x','--poly','5','--poly','x+6','--poly-order','lex',
                      '--module-order','pot','--record-paths','--verify','--no-show-basis',
                      '--no-show-u','--output',str(target)]
                args += ['--'+name.replace('_','-') for name,on in zip(FLAGS,mask) if on]
                with redirect_stdout(StringIO()) as sink,redirect_stderr(StringIO()):
                    self.assertEqual(main(args),0)
                text=sink.getvalue()
                self.assertEqual('Minimal strong Groebner basis (' in text,mask[0])
                self.assertEqual('Reduced strong Groebner basis (' in text,mask[1])
                self.assertEqual('Leading terms of the minimal strong Groebner basis (' in text,mask[2])
                data=json.loads(target.read_text())
                self.assertEqual('polynomial_postprocessing' in data,any(mask))
                r=load_result(target);self.assertIsNone(r.polynomial_postprocessing)
                self.assertTrue(verify_generation_paths(r)['signature_basis_certified'])
                self.assertEqual('reduced_basis' in data.get('polynomial_postprocessing',{}),mask[1])
                old=target.read_bytes()
                with redirect_stdout(StringIO()),redirect_stderr(StringIO()):
                    self.assertEqual(main(['show',str(target),'--no-show-basis','--no-show-u','--show-reduced-gb']),0)
                self.assertEqual(target.read_bytes(),old)

    def test_cli_skips_partial(self):
        with TemporaryDirectory() as directory,redirect_stdout(StringIO()) as sink,redirect_stderr(StringIO()):
            p=Path(directory)/'partial.json'
            code=main(['run','--case','cyclic3','--max-candidates','1','--show-reduced-gb',
                       '--no-show-basis','--output',str(p)])
            self.assertEqual(code,2)
            data=json.loads(p.read_text());self.assertEqual(data['polynomial_postprocessing']['status'],'skipped')
            self.assertNotIn('Reduced strong Groebner basis (',sink.getvalue())

    def test_cli_interruption_saves_source(self):
        with TemporaryDirectory() as directory,redirect_stdout(StringIO()),redirect_stderr(StringIO()):
            p=Path(directory)/'interrupted.json'
            with patch('gvwzz.cli.postprocess_basis',side_effect=KeyboardInterrupt):
                code=main(['run','--case','cyclic3','--show-reduced-gb','--record-paths',
                           '--no-show-basis','--output',str(p)])
            self.assertEqual(code,130)
            d=json.loads(p.read_text());self.assertEqual(d['status'],'completed')
            self.assertEqual(d['polynomial_postprocessing']['status'],'interrupted')
            self.assertTrue(verify_generation_paths(load_result(p))['passed'])

    def test_cli_failure_saves_source(self):
        with TemporaryDirectory() as directory,redirect_stdout(StringIO()),redirect_stderr(StringIO()):
            p=Path(directory)/'failed.json'
            with patch('gvwzz.cli.postprocess_basis',side_effect=ValueError('test failure')):
                code=main(['run','--case','cyclic3','--show-reduced-gb','--no-show-basis','--output',str(p)])
            self.assertEqual(code,1)
            self.assertEqual(json.loads(p.read_text())['polynomial_postprocessing']['status'],'failed')
            self.assertEqual(load_result(p).status,'completed')

    def test_cached_output_is_not_trusted_by_show(self):
        with TemporaryDirectory() as directory:
            r=solve(['5','x+6'],variables=('x',));p=Path(directory)/'test.json'
            data=r.to_dict();data['polynomial_postprocessing']={'reduced_basis':['1']}
            write_json(p,data)
            with redirect_stdout(StringIO()) as sink:
                self.assertEqual(main(['show',str(p),'--no-show-basis','--show-reduced-gb']),0)
            self.assertIn('x + 1',sink.getvalue())
            self.assertIn('Reduced strong Groebner basis (2)',sink.getvalue())

    def test_summary_only(self):
        with TemporaryDirectory() as directory,redirect_stdout(StringIO()):
            p=Path(directory)/'summary.json'
            self.assertEqual(main(['run','--case','cyclic3','--summary-only','--show-reduced-gb',
                                   '--no-show-basis','--output',str(p)]),0)
            d=json.loads(p.read_text())['polynomial_postprocessing']
            self.assertNotIn('reduced_basis',d);self.assertIsNotNone(d['reduced_basis_size'])

    def test_failed_source_verification_blocks_postprocessing(self):
        with TemporaryDirectory() as directory, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            target=Path(directory)/'not_verified.json'
            failed={'passed':False,'status':'test_failed','membership_verified':False,
                    'signature_basis_certified':False,'errors':['deliberate test failure']}
            with patch('gvwzz.cli.verify_result',return_value=failed), patch('gvwzz.cli.postprocess_basis') as post:
                code=main(['run','--case','cyclic3','--verify','--show-reduced-gb',
                           '--no-show-basis','--output',str(target)])
                self.assertEqual(code,3);post.assert_not_called()
            self.assertEqual(json.loads(target.read_text())['polynomial_postprocessing']['status'],'skipped')

    def test_formatting_does_not_store_and_supports_unicode(self):
        r=solve(PAPER)
        before=r.to_dict()
        text=format_result(r,show_basis=False,show_minimal_lt=True,style='unicode',width=40)
        self.assertIn('x²',text)
        self.assertEqual(r.to_dict(),before)
        self.assertIsNone(r.polynomial_postprocessing)
        report=postprocess_basis(r,reduced=True).to_dict()
        self.assertIs(report['source_basis_verified_by_postprocessing'],False)


    def test_random_triangular_strong_bases(self):
        rng=random.Random(104)
        for _ in range(24):
            b=IntegerPolynomialRing(('x','y'),'lex');x,y=b.ring.gens
            m=rng.randrange(2,25)
            a=sum((rng.randrange(-35,36)*y**j for j in range(5)),b.zero)
            z=sum((rng.randrange(-35,36)*y**j for j in range(3)),b.zero)
            r=known_basis([b.ring(m),x+a,y**3+z])
            p=postprocess_basis(r,reduced=True)
            assert_strong(self,r,p.reduced_basis)
            # Adding multiples of the integer generator preserves the ideal/LTs.
            r2=known_basis([b.ring(-m),x+a+m*(y**6-1),y**3+z+3*m*y])
            p2=postprocess_basis(r2,reduced=True)
            self.assertEqual(p.reduced_basis,p2.reduced_basis)


if __name__ == '__main__':
    unittest.main(verbosity=2)
