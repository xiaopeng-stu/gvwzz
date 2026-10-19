"""Post17: exact short-circuit queries and deferred generation, not new criteria."""
import io
import random
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from gvwzz import GVWInteger, Options, Signature
from gvwzz.core import (_signature_from_arithmetic, add_monomials,
                        subtract_monomials, lcm_monomials)
from gvwzz.orders import ModuleOrder
from gvwzz.polynomial import IntegerPolynomialRing
from gvwzz.structures import SyzygyStore, SyzygyIndex, Stats, CandidateQueue, CoverIndex
from gvwzz.cli import timing_report
from gvwzz.verification import verify_result
from gvwzz.reconstruction import verify_generation_paths, reconstruct_syzygies


def engine(options=None):
    b = IntegerPolynomialRing(['x','y'], 'grevlex')
    fs = [b.parse(s) for s in ('4*x^2', '9*y^4', '6*x*y^2+y')]
    return GVWInteger(b, fs, ModuleOrder(b,fs,'g2'), options or Options())


def s(c,m,i=0): return Signature(c,tuple(m),i)


def uncompleted(coeffs):
    # An intentionally uncompleted collection distinguishes a single divisor
    # from a coefficient gcd. No completion routine is invoked on this fixture.
    st=SyzygyStore(1,1,Stats())
    st.components[0].extend([s(c,(1,)) for c in coeffs])
    st.tries[0].set((1,),list(coeffs))
    st.nonunit_counts[0]=sum(c!=1 for c in coeffs)
    return st


class GenerationChecks(unittest.TestCase):
    def test_01_empty(self):
        st=SyzygyStore(2,2,Stats())
        self.assertFalse(st.has_divisor(s(-7,(3,4),1)))

    def test_02_unit_query_never_enumerates(self):
        st=SyzygyStore(2,1,Stats())
        for i in range(100): st.insert_and_complete(s(1,(i,99-i)))
        with patch.object(st,'divisor_coefficients',side_effect=AssertionError('all coefficients')):
            with patch.object(st.tries[0],'divisors',side_effect=AssertionError('payload enumeration')):
                self.assertTrue(st.has_divisor(s(37,(100,100))))
                self.assertFalse(st.has_divisor(s(37,(0,0))))

    def test_03_mixed_query_is_not_gcd(self):
        st=uncompleted((2,3))
        self.assertFalse(st.has_divisor(s(5,(3,))))
        self.assertFalse(st.has_divisor(s(1,(3,))))
        self.assertTrue(st.has_divisor(s(6,(3,))))

    def test_04_mixed_first_witness_shortcircuits(self):
        st=uncompleted((2,3,5,7,11))
        before=st.stats.counters['syzygy_existence_coefficients_tested']
        self.assertTrue(st.has_divisor(s(-8,(2,))))
        self.assertEqual(st.stats.counters['syzygy_existence_coefficients_tested']-before,1)

    def test_05_unit_witness_inside_mixed(self):
        st=uncompleted((2,3,1,5))
        self.assertTrue(st.has_divisor(s(7,(2,))))
        self.assertEqual(st.stats.counters['syzygy_existence_coefficients_tested'],3)

    def test_06_signed_cache_keys(self):
        st=uncompleted((6,))
        self.assertTrue(st.has_divisor(s(12,(2,))))
        before=st.stats.counters['syzygy_existence_cache_hits']
        self.assertTrue(st.has_divisor(s(-12,(2,))))
        self.assertEqual(st.stats.counters['syzygy_existence_cache_hits']-before,1)
        self.assertFalse(st.has_divisor(s(5,(2,))))

    def test_07_unit_cache_shares_coefficients(self):
        st=SyzygyStore(1,1,Stats());st.insert_and_complete(s(1,(1,)))
        st.has_divisor(s(25,(2,)))
        before=st.stats.counters['syzygy_existence_cache_hits']
        self.assertTrue(st.has_divisor(s(-13,(2,))))
        self.assertEqual(st.stats.counters['syzygy_existence_cache_hits']-before,1)

    def test_08_negative_is_invalidated(self):
        st=SyzygyStore(2,1,Stats());st.insert_and_complete(s(1,(3,0)))
        self.assertFalse(st.has_divisor(s(1,(0,3))))
        st.insert_and_complete(s(1,(0,1)))
        self.assertTrue(st.has_divisor(s(1,(0,3))))

    def test_09_positive_survives_replacement(self):
        st=SyzygyStore(1,1,Stats());st.insert_and_complete(s(6,(3,)))
        self.assertTrue(st.has_divisor(s(12,(4,))))
        st.insert_and_complete(s(3,(2,)))
        before=st.stats.counters['syzygy_existence_cache_hits']
        self.assertTrue(st.has_divisor(s(12,(4,))))
        self.assertEqual(st.stats.counters['syzygy_existence_cache_hits']-before,1)
        self.assertEqual(st.all_signatures(),[s(3,(2,))])

    def test_10_component_versions_are_independent(self):
        st=SyzygyStore(1,2,Stats());st.insert_and_complete(s(2,(2,),0))
        st.has_divisor(s(3,(3,),0));v=st.component_versions[0]
        st.insert_and_complete(s(1,(0,),1))
        before=st.stats.counters['syzygy_existence_cache_hits']
        self.assertFalse(st.has_divisor(s(3,(3,),0)))
        self.assertEqual(st.component_versions[0],v)
        self.assertEqual(st.stats.counters['syzygy_existence_cache_hits']-before,1)

    def test_11_unit_to_mixed_keeps_coefficient_check(self):
        st=SyzygyStore(2,1,Stats());st.insert_and_complete(s(1,(1,0)))
        self.assertFalse(st.has_divisor(s(1,(0,3))))
        st.insert_and_complete(s(2,(0,1)))
        self.assertTrue(st.has_divisor(s(2,(0,3))))
        self.assertFalse(st.has_divisor(s(1,(0,3))))

    def test_12_mixed_to_unit(self):
        st=SyzygyStore(2,1,Stats());st.insert_and_complete(s(2,(0,1)))
        self.assertFalse(st.has_divisor(s(1,(1,1))))
        st.insert_and_complete(s(1,(0,0)))
        self.assertTrue(st.has_divisor(s(1,(1,1))))
        self.assertEqual(st.nonunit_counts[0],0)

    def test_13_boolean_cache_bounded(self):
        st=SyzygyStore(1,1,Stats());st.insert_and_complete(s(1,(0,)))
        for i in range(9000):self.assertTrue(st.has_divisor(s(1,(i,))))
        self.assertLessEqual(len(st.existence_cache[0]),8192)
        self.assertEqual(st.divisor_cache[0],{})

    def test_14_full_coefficient_api_preserved(self):
        st=uncompleted((2,3))
        self.assertEqual(st.divisor_coefficients((3,),0),(2,3))
        self.assertFalse(st.has_divisor(s(5,(3,))))
        self.assertEqual(st.divisor_coefficients((3,),0),(2,3))

    def test_15_random_streams_and_queries(self):
        rng=random.Random(173816)
        for dim in (1,2,8,48):
            st=SyzygyStore(dim,3,Stats())
            for _ in range(90):
                m=tuple(rng.randrange(4) if k<3 else 0 for k in range(dim))
                st.insert_and_complete(s(rng.choice([-35,-6,-1,1,2,3,10]),m,rng.randrange(3)))
                for _ in range(3):
                    q=s(rng.choice([-21,-1,1,4,9,30]),
                        (rng.randrange(6) if k<3 else 0 for k in range(dim)),rng.randrange(3))
                    expected=any(x.divides(q) for x in st.all_signatures())
                    self.assertEqual(st.has_divisor(q),expected)

    def test_16_large_coefficients_exact(self):
        st=uncompleted((2**3000*3,2**2500*5))
        self.assertTrue(st.has_divisor(s(-(2**3001*3),(3,))))
        self.assertFalse(st.has_divisor(s(2**2499,(3,))))

    def test_17_compaction_preserves_boolean_answers(self):
        st=SyzygyStore(2,1,Stats())
        for i in range(130):st.insert_and_complete(s(1,(i,129-i)))
        self.assertTrue(st.has_divisor(s(7,(130,130))))
        self.assertFalse(st.has_divisor(s(1,(0,0))))
        st.insert_and_complete(s(1,(0,0)))
        self.assertGreater(st.stats.counters['syzygy_index_compactions'],0)
        self.assertTrue(st.has_divisor(s(1,(0,0))))

    def test_18_deferred_rejection_allocates_no_signature(self):
        sol=engine();sol.S.insert_and_complete(s(1,(0,0)))
        with patch('gvwzz.solver._signature_from_arithmetic',side_effect=AssertionError('allocation')):
            with patch.object(sol.S,'_complete_uncovered',side_effect=AssertionError('completion')):
                sol._insert_syzygy((6,(3,5),0),check_budget=False)
        self.assertEqual(sol.stats.counters['principal_syzygy_objects_avoided'],1)

    def test_19_rejected_seed_has_no_witness_snapshot(self):
        sol=engine(Options(record_paths=True));sol.S.insert_and_complete(s(1,(0,0)))
        with patch.object(type(sol.S.components[0]),'__iter__',side_effect=AssertionError('snapshot')):
            sol._insert_syzygy((2,(3,4),0),check_budget=False,
                path_factory=lambda: self.fail('witness factory on rejected seed'))

    def test_20_redundancy_phase_and_restore(self):
        sol=engine(Options(progress_interval=20));sol.S.insert_and_complete(s(1,(0,0)))
        sol.phase='candidate generation';sol._progress_next=0
        buf=io.StringIO()
        with redirect_stderr(buf):sol._insert_syzygy((2,(3,4),0),check_budget=False)
        self.assertIn('phase=syzygy redundancy check',buf.getvalue())
        self.assertEqual(sol.phase,'candidate generation')

    def test_21_completion_phase_and_restore(self):
        sol=engine(Options(progress_interval=20));sol.phase='initialization';sol._progress_next=0
        buf=io.StringIO()
        with redirect_stderr(buf):sol._insert_syzygy(s(2,(1,0)),check_budget=False)
        self.assertIn('phase=syzygy completion',buf.getvalue())
        self.assertEqual(sol.phase,'initialization')

    def test_22_exception_phase_and_timing(self):
        sol=engine();sol.phase='original'
        with patch.object(sol.S,'_accept_seed_parts',side_effect=RuntimeError('query failed')):
            with self.assertRaises(RuntimeError):sol._insert_syzygy(s(2,(1,0)))
        self.assertEqual(sol.phase,'original')
        self.assertEqual(sol.stats.times_ns['syzygy'],sol.stats.times_ns['syzygy_check'])

    def test_23_timing_partitions_do_not_double_count(self):
        sol=engine();r=sol.run();t=r.stats.times_ns
        self.assertEqual(t['syzygy'],t['syzygy_check']+t['syzygy_complete'])
        d=r.stats.to_dict()['timings_seconds'];buf=io.StringIO()
        with redirect_stdout(buf):timing_report(d)
        self.assertIn('Syzygy seed redundancy checks',buf.getvalue())
        self.assertIn('Syzygy completion / update',buf.getvalue())
        self.assertNotIn('Syzygy insert/complete/simplify',buf.getvalue())
        self.assertLess(abs(d['syzygy']-d['syzygy_check']-d['syzygy_complete']),1e-12)

    def test_24_legacy_timing_display(self):
        d=engine().run().stats.to_dict()['timings_seconds'];d.pop('syzygy_check');d.pop('syzygy_complete')
        buf=io.StringIO()
        with redirect_stdout(buf):timing_report(d)
        self.assertIn('Syzygy insert/complete/simplify',buf.getvalue())

    def test_25_arithmetic_constructor_matches_checked(self):
        rng=random.Random(112342)
        for _ in range(300):
            c=rng.randrange(1,10**30)*rng.choice((-1,1));m=tuple(rng.randrange(20) for _ in range(48));i=rng.randrange(60)
            a=_signature_from_arithmetic(c,m,i);b=Signature(c,m,i)
            self.assertEqual(a,b);self.assertEqual(hash(a),hash(b));self.assertEqual(a.to_dict(),b.to_dict())

    def test_26_zero_signature_still_rejected(self):
        with self.assertRaises(ValueError):_signature_from_arithmetic(0,(0,),0)
        with self.assertRaises(ValueError):Signature(0,(0,),0)

    def test_27_public_validation_unchanged(self):
        for c,m,i in ((True,(0,),0),(1,(-1,),0),(1,(False,),0),(1,(0,),-1),(1,(0.2,),0),(1.2,(0,),0)):
            with self.assertRaises(ValueError):Signature(c,m,i)

    def test_28_ratio_shift_formula(self):
        rng=random.Random(7000)
        for dim in (1,8,48):
            for _ in range(100):
                p=tuple(rng.randrange(6) for _ in range(dim));q=tuple(rng.randrange(6) for _ in range(dim));sig=tuple(rng.randrange(4) for _ in range(dim))
                W=lcm_monomials(p,q)
                actual=add_monomials(W,subtract_monomials(sig,p))
                self.assertEqual(actual,add_monomials(subtract_monomials(W,p),sig))
                self.assertTrue(all(e>=0 for e in actual))

    def test_29_unit_multicover_does_not_enumerate(self):
        sol=engine();sol.S.insert_and_complete(s(1,(1,0)))
        with patch.object(sol.S,'divisor_coefficients',side_effect=AssertionError('enumerate all')):
            self.assertTrue(sol.cover.check(s(6,(2,0)),(3,0)))

    def test_30_mixed_multicover_still_combines_coefficients(self):
        b=IntegerPolynomialRing(['x'],'lex');f=[b.parse('x')];st=uncompleted((2,3));c=CoverIndex(b,ModuleOrder(b,f,'pot'),st,Stats())
        self.assertFalse(st.has_divisor(s(5,(2,))))
        self.assertTrue(c.check(s(5,(2,)),(1,)))

    def test_31_queue_public_and_deferred_paths_equal(self):
        sol=engine();b,o=sol.backend,sol.order
        for tie in ('fifo','small-coeff','small-lm'):
            for dedup in (True,False):
                a=CandidateQueue(b,o,Stats(),dedup,tie_policy=tie);z=CandidateQueue(b,o,Stats(),dedup,tie_policy=tie)
                for k in range(90):
                    sig=s(k%3+1,(k%5,0),k%3);W=(k%7,k%2)
                    x=a.push_j(sig,W,lambda: b.one)
                    y=z._accept_j(sig,W)
                    if y:z._insert('J',sig,W,factory=lambda: b.one)
                    self.assertEqual(x,y);self.assertEqual(a.heap,z.heap)
                self.assertEqual(a.stats.counters,z.stats.counters)
                while a:
                    ca,cz=a.pop(),z.pop();self.assertEqual((ca.token,ca.signature),(cz.token,cz.signature))

    def test_32_replacement_before_filter_is_preserved(self):
        sol=engine();q=CandidateQueue(sol.backend,sol.order,Stats(),generation_filter=lambda sig:False)
        sig=s(1,(0,0));q.push_j(sig,(3,0),lambda:sol.backend.one)
        q.generation_filter=lambda sig:True
        self.assertFalse(q._accept_j(sig,(2,0)))
        self.assertFalse(q.active);self.assertFalse(q.pending_j)
        self.assertEqual(q.stats.counters['j_dedup_replaced'],1)

    def test_33_settled_dedup_precedes_filter(self):
        sol=engine();q=CandidateQueue(sol.backend,sol.order,Stats())
        sig=s(1,(0,0));q.push_j(sig,(1,0),lambda:sol.backend.one);q.settle_j(q.pop())
        q.generation_filter=lambda sig:self.fail('filter called before settled check')
        self.assertFalse(q._accept_j(sig,(2,0)))

    def test_34_eager_and_lazy_online_certificates(self):
        results=[]
        for lazy in (True,False):
            r=engine(Options(lazy_candidates=lazy,track_representations=True)).run()
            report=verify_result(r,mode='static',strong=True,require_certificates=True)
            self.assertTrue(report['signature_basis_certified']);results.append(r)
        self.assertEqual(results[0].basis,results[1].basis)
        self.assertEqual(results[0].basis_representations,results[1].basis_representations)

    def test_35_eager_paths_and_replay(self):
        r=engine(Options(lazy_candidates=False,record_paths=True,g_precheck=False)).run()
        self.assertTrue(verify_generation_paths(r)['signature_basis_certified'])
        reconstruct_syzygies(r,verify=True)

    def test_36_all_orders_and_kernels(self):
        for order in ('pot','top','g1','g2','schreyer','block'):
            for kernel in ('copy','heap','packed'):
                sol=engine(Options(reduction_kernel=kernel,record_paths=True,g_first=False))
                sol=GVWInteger(sol.backend,sol.inputs,ModuleOrder(sol.backend,sol.inputs,order,block_sizes=(1,2) if order=='block' else None),sol.options)
                r=sol.run();self.assertTrue(verify_generation_paths(r)['signature_basis_certified'],(order,kernel))

    def test_37_completed_pair_counts_preserved(self):
        r=engine().run();c=r.stats.counters;n=len(r.basis)
        self.assertEqual(c['parent_pairs'],n*(n-1)//2)
        self.assertEqual(c['g_generated'],n*(n-1)//2)
        self.assertEqual(c['g_generated'],c['g_discarded']+c['g_inserted'])

    def test_38_internal_gate_is_required_for_mutation(self):
        sol=engine();sol.S.insert_and_complete(s(1,(0,0)))
        before=sol.S.version
        with patch.object(sol.S,'_complete_uncovered',side_effect=AssertionError('unnecessary mutation')):
            sol._insert_syzygy(s(17,(4,5)),check_budget=False)
        self.assertEqual(sol.S.version,before)

    def test_39_no_new_solver_defaults(self):
        o=Options()
        self.assertTrue(o.euclidean and o.tail_reduce and o.g_tail_reduce)
        self.assertEqual(o.reducer_selection,'sparse');self.assertEqual(o.queue_tie,'small-coeff')
        self.assertIsNone(o.max_seconds)

    def test_40_logical_rejections_and_updates(self):
        sol=engine()
        sol._insert_syzygy((1,(1,0),0),check_budget=False)
        sol._insert_syzygy((6,(3,0),0),check_budget=False)
        c=sol.stats.counters
        self.assertEqual(c['syzygy_insert_calls'],2)
        self.assertEqual(c['syzygy_seed_rejected'],1)
        self.assertEqual(sol.S.version,1)

    def test_41_recent_witness_does_not_skip_coefficient_test(self):
        st=uncompleted((2,))
        self.assertTrue(st.has_divisor(s(6,(3,))))
        self.assertFalse(st.has_divisor(s(7,(4,))))
        with patch.object(st.tries[0],'_divisor_mask',side_effect=AssertionError('bitset query')):
            self.assertTrue(st.has_divisor(s(12,(9,))))
        self.assertEqual(st.stats.counters['syzygy_recent_witness_hits'],1)

    def test_42_recent_witness_survives_exact_replacement(self):
        st=SyzygyStore(2,1,Stats());st.insert_and_complete(s(4,(3,0)))
        self.assertTrue(st.has_divisor(s(8,(4,0))))
        st.insert_and_complete(s(2,(1,0)))
        self.assertNotIn(s(4,(3,0)),st.all_signatures())
        with patch.object(st.tries[0],'_divisor_mask',side_effect=AssertionError('bitset query')):
            self.assertTrue(st.has_divisor(s(12,(9,9))))
        self.assertTrue(any(x.divides(s(12,(9,9))) for x in st.all_signatures()))

    def test_43_degree_bound_needs_no_full_scan(self):
        idx=SyzygyIndex(2);idx.set((4,2),[1]);idx.set((3,3),[1])
        self.assertEqual(idx._divisor_mask((1,1)),0)
        self.assertEqual(list(idx.divisors((4,2))),[[1]])
        self.assertEqual(list(idx.divisors((2,4))),[])

    def test_44_stale_degree_bound_is_conservative(self):
        idx=SyzygyIndex(1);idx.set((1,),[1]);idx.remove((1,));idx.set((5,),[2])
        self.assertEqual(idx.minimum_degree,1)
        self.assertEqual(list(idx.divisors((1,))),[])
        self.assertEqual(list(idx.divisors((3,))),[])
        self.assertEqual(list(idx.divisors((6,))),[[2]])

    def test_45_unit_witness_in_mixed_component_is_safe(self):
        st=SyzygyStore(2,1,Stats());st.insert_and_complete(s(1,(1,0)))
        self.assertTrue(st.has_divisor(s(5,(2,0))))
        st.insert_and_complete(s(6,(0,1)))
        self.assertFalse(st.has_divisor(s(5,(0,3))))
        self.assertTrue(st.has_divisor(s(17,(3,3))))


if __name__=='__main__': unittest.main(verbosity=2)
