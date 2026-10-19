"""Incremental POT ideal bases over ZZ, with frozen minimal prefix bases.

This is a distinct solver mode, not deletion inside an unfinished signature
basis. At stage i the frozen H is a strong basis of (f_1,...,f_{i-1}). Only
signatures in position i are processed. Lower-position labels are ordinary
reducers; their discarded signatures/syzygies are NOT reused for cover.

The working relative module is {(u,v): v-u*f_i belongs to (H)}. Its zero
signature part is reduced by H. LT(H)*e_i seeds valid syzygies. Frozen/frozen
pairs are settled by the prefix invariant. Every known current/current
principal term lies in the frozen leading-term ideal, so it needs no seed
construction or redundancy query. A stage is compacted only after its queue
is empty.
The returned labels retain genuine original-input representations, but the
result is an ideal basis, NOT a complete original-module signature basis.
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace
from time import perf_counter_ns, process_time_ns

from .core import _signature_from_arithmetic, Label, Signature
from .forest import InsertionOrigin
from .frozen import FrozenSeedIndex
from .paths import Edge, GenerationPaths, PathTracker
from .solver import ComputationLimit, GVWInteger, Options, Result
from .structures import CoverIndex, Stats, SyzygyIndex, SyzygyStore, is_complete

try:
    if os.environ.get('GVWZZ_NO_NATIVE'):
        raise ImportError('native accelerators disabled')
    from ._native_pairs import emit_frozen_pairs as _native_emit_frozen_pairs
    try:
        from ._native_pairs import emit_unit_current_pairs as _native_emit_unit_current_pairs
        from ._native_pairs import commit_frozen_parents as _native_commit_frozen_parents
    except ImportError:
        _native_emit_unit_current_pairs = None
        _native_commit_frozen_parents = None
    try:
        from ._native_pairs import minimal_frozen_parents as _native_minimal_frozen_parents
        from ._native_pairs import minimal_frozen_mask as _native_minimal_frozen_mask
    except ImportError:
        _native_minimal_frozen_parents = None
        _native_minimal_frozen_mask = None
except ImportError:
    _native_emit_frozen_pairs = None
    _native_emit_unit_current_pairs = None
    _native_commit_frozen_parents = None
    _native_minimal_frozen_parents = None
    _native_minimal_frozen_mask = None


class _StageCover(CoverIndex):
    """Lower positions cannot align with a current-position signature."""
    def __init__(self, *args, position: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.position = position

    def insert(self, label, lm):
        if label.signature.index == self.position:
            super().insert(label, lm)
        else:
            self.stats.counters['incremental_cover_records_skipped'] += 1


class _IncrementalPaths(PathTracker):
    """Term-only DAG edges survive removal/reindexing of active polynomials.

Ordinary PathTracker principal edges reference final P indices. That is unsafe
across compaction. Expand ONLY scalar polynomial multipliers into term edges;
representation vectors are still never expanded online.
    """
    def __init__(self, backend, ncomponents):
        super().__init__(backend, ncomponents)
        self.polynomials = []

    def _poly_edges(self, root, polynomial, sign=1):
        return [Edge(root, sign * int(c), tuple(mu)) for mu, c in polynomial.items() if c]

    def principal(self, i, j, signature):
        edges = self._poly_edges(self.basis[i], self.polynomials[j].polynomial)
        edges += self._poly_edges(self.basis[j], self.polynomials[i].polynomial, -1)
        return self._add('principal', signature, edges)

    def prefix_syzygy(self, old_index, input_root, f, signature):
        # h*e_i - f_i*u_h, whose leading position is i.
        edges = self._poly_edges(input_root, self.polynomials[old_index].polynomial)
        edges += self._poly_edges(self.basis[old_index], f, -1)
        return self._add('principal', signature, edges)


class _POTStage(GVWInteger):
    # Only a natural incremental POT stage has the strong-prefix invariant.
    # Native pair emission also serves ordinary GVWInteger instances, so it
    # must opt into this proof explicitly rather than infer it from unit LCs.
    _principal_pairs_redundant = True

    def __init__(self, owner, position):
        super().__init__(owner.backend, owner.inputs, owner.order,
                         replace(owner.options, incremental_pot=False), _validated_inputs=True)
        self.owner, self.position = owner, position
        self.frozen_count = len(owner.basis)
        self._frozen_seed_index = None
        self.stats = owner.stats
        self._recipe_context.counts = self.stats.counters
        self.S = SyzygyStore(len(self.backend.names), len(self.inputs), self.stats, self.options.debug)
        self.cover = _StageCover(self.backend, self.order, self.S, self.stats,
                                 self.options.cover_mode, self.options.cover_cache_size,
                                 position=position)
        # The base queue is still empty and its bound callbacks resolve self.S
        # dynamically. Reuse it instead of leaving an unreachable old queue
        # with bound callbacks retaining this entire stage until cyclic GC.
        self.queue.stats = self.stats
        self.deadline = owner.deadline
        self._progress_start = owner.start
        self._progress_next = owner.progress_next
        if owner.paths is not None:
            self.paths = owner.paths
            self.paths.basis = []
            self.paths.syzygies = {}
            self.paths.polynomials = self.P
        self.started = True

    def _report_progress(self, now):
        self._progress_next = now + int(self.options.progress_interval * 1e9)
        self.owner.progress_next = self._progress_next
        c, sig = self.stats.counters, self.current_signature
        signature = f'e{sig.index+1}:{sig.monomial}' if sig is not None else '-'
        jt, gt = c['j_tail_steps'], c['g_tail_steps']
        if self._tail_kind == 'J':
            jt += c['tail_steps'] - self._tail_steps_start
        elif self._tail_kind == 'G':
            gt += c['tail_steps'] - self._tail_steps_start
        candidate = (f'{self.current_candidate_kind}#{self.current_candidate_serial}'
                     if self.current_candidate_kind else '-')
        print(f'[progress {(now-self._progress_start)/1e9:.1f}s] '
              f'stage={self.position+1}/{len(self.inputs)} phase={self.phase} '
              f'candidate={candidate} P={len(self.P)} frozen={self.frozen_count} S_current={len(self.S)} '
              f'queue={len(self.queue.active)} generated={c["j_generated"]+c["g_generated"]} '
              f'head={c["regular_steps"]+c["singular_steps"]+c["euclidean_steps"]} '
              f'tail={c["tail_steps"]} tail(J/G)={jt}/{gt} '
              f'popped(J/G)={c["j_popped"]}/{c["g_popped"]} signature={signature}',
              file=sys.stderr, flush=True)

    def _skip_principal_pair(self, i, j):
        """Every known stage principal seed is covered by the frozen ideal.

        For current labels write p = U*f_k + h_p and q = V*f_k + h_q,
        with h_p,h_q in the completed prefix ideal I. Their principal current
        component q*U-p*V = U*h_q-V*h_p belongs to I. Whenever the usual
        principal descriptor is nonzero, it is its exact leading term, hence
        one LT of the frozen strong basis divides it, coefficient included.
        The initial LT(H)*e_k seeds and their retained divisors already cover
        it. A frozen/current pair has the same direct prefix-seed witness.

        This does not infer a new descriptor when the two leading terms
        cancel: the ordinary emitter did not insert such an unknown seed.
        """
        return True

    def _emit_pair(self, i, j):
        if i < self.frozen_count and j < self.frozen_count:
            raise AssertionError('Frozen prefix pairs must not be regenerated')
        super()._emit_pair(i, j)

    def _emit_pairs(self, j):
        """Remove frozen-parent J covers by an exact immutable seed index.

        A unit frozen seed x^a e_k covers the J from a current label
        (C*x^s e_k, LT=B*x^b), with unit B, exactly when min(a,b)<=s.
        Completing/deleting seeds preserves this cover: every deleted seed
        has a retained divisor. These J candidates therefore need neither a
        signature descriptor nor a queue entry. Their comparable G candidates
        have the usual exact parent top reducer and are counted separately.
        """
        frozen = self.frozen_count
        # A candidate-count budget promises the original first violating pair.
        # Keep that order and its exact partial-state accounting on this path.
        if (not frozen or not self.options.j_generation_filter or
                self.options.max_candidates is not None or
                abs(self.leading[j][1]) != 1):
            return super()._emit_pairs(j)
        if self._frozen_seed_index is None:
            if all(abs(self.leading[i][1]) == 1 for i in range(frozen)):
                self._frozen_seed_index = FrozenSeedIndex(
                    len(self.backend.names), [self.leading[i][0] for i in range(frozen)])
            else:
                self._frozen_seed_index = False
        if self._frozen_seed_index is False:
            return super()._emit_pairs(j)
        self.phase = 'candidate generation'
        self._check_budget()
        start = perf_counter_ns()
        sig = self.P[j].signature
        screen_start = perf_counter_ns()
        survivor_mask = self._frozen_seed_index.survivor_mask(
            self.leading[j][0], sig.monomial)
        self.stats.times_ns['frozen_seed_screen'] += perf_counter_ns() - screen_start
        survivor_count = survivor_mask.bit_count()
        seed_omitted = frozen - survivor_count
        minimal_start = perf_counter_ns()
        minimal = (_native_minimal_frozen_mask(self, j, survivor_mask)
                   if _native_minimal_frozen_mask is not None else None)
        if minimal is None:
            minimal = self._frozen_seed_index.minimal_parents_mask(
                self.leading[j][0], survivor_mask)
        self.stats.times_ns['frozen_multiplier_minimal'] += perf_counter_ns() - minimal_start
        dominance_omitted = survivor_count - len(minimal)
        survivors = minimal
        omitted = seed_omitted + dominance_omitted
        if omitted:
            counts = self.stats.counters
            for name in ('parent_pairs', 'j_generated', 'g_generated',
                         'g_omitted_comparable', 'g_discarded',
                         'incremental_principal_skipped',
                         'candidate_signature_objects_avoided'):
                counts[name] += omitted
            counts['j_generation_tests'] += seed_omitted
            counts['j_generation_syzygy_rejected'] += seed_omitted
            counts['j_frozen_seed_rejected'] += seed_omitted
            counts['j_dominance_rejected'] += dominance_omitted
        self.stats.times_ns['generation'] += perf_counter_ns() - start
        if _native_emit_frozen_pairs is not None and self.packed_parents[j] is not None:
            _native_emit_frozen_pairs(self, j, survivors)
        else:
            for i in survivors:
                self._emit_pair(i, j)
        if _native_commit_frozen_parents is not None:
            _native_commit_frozen_parents(self, j)
        if _native_emit_unit_current_pairs is not None:
            _native_emit_unit_current_pairs(self, j, frozen)
        else:
            for i in range(frozen, j):
                self._emit_pair(i, j)
        self._check_budget()

    def _seed_prefix_syzygies(self, f, input_root):
        """Install the known-complete LT(H)*e_i without redoing its pair closure."""
        start = perf_counter_ns()
        previous = self.phase
        self.phase = 'incremental prefix syzygies'
        try:
            seeds = []
            groups = {}
            for j in range(self.frozen_count):
                self._check_budget()
                mu, c = self.leading[j]
                if c <= 0:
                    raise AssertionError('Frozen prefix leading coefficients must be positive')
                s = _signature_from_arithmetic(c, mu, self.position)
                seeds.append(s)
                groups.setdefault(mu, []).append(c)
                if self.witnesses is not None:
                    v = [-f * u for u in self.witnesses.basis[j]]
                    v[self.position] += self.P[j].polynomial
                    self.witnesses.syzygies[s] = tuple(v)
                if self.paths is not None:
                    self.paths.syzygies[s] = self.paths.prefix_syzygy(j, input_root, f, s)
            store = self.S
            store.components[self.position].extend(seeds)
            for mu, coeffs in groups.items():
                store.tries[self.position].set(mu, coeffs)
            store.nonunit_counts[self.position] = sum(s.coefficient != 1 for s in seeds)
            if seeds:
                store.component_versions[self.position] += 1
                store.version += 1
            counts = self.stats.counters
            counts['incremental_prefix_syzygy_seeds'] += len(seeds)
            counts['syzygy_index_records_written'] += len(groups)
            counts['s_peak'] = max(counts['s_peak'], len(store))
            if self.options.debug:
                assert is_complete(seeds), 'Frozen strong basis must give gcd-complete LT seeds'
        finally:
            elapsed = perf_counter_ns() - start
            self.stats.times_ns['syzygy'] += elapsed
            self.stats.times_ns['syzygy_complete'] += elapsed
            self.phase = previous

    def _initialize_and_run(self):
        owner, k = self.owner, self.position
        self.phase = 'incremental frozen basis loading'
        for j, p in enumerate(owner.basis):
            if p.signature.index >= k:
                raise AssertionError('A frozen label has not strictly lower POT position')
            if self.witnesses is not None:
                self.witnesses.current = owner.vectors[j]
            if self.paths is not None:
                self.paths.current_root = owner.path_roots[j]
                self.paths.current_edges = None
            self._current_origin = None
            self._insert_basis(p)
            self.stats.counters['incremental_frozen_loads'] += 1
        self.stats.counters['incremental_frozen_pairs_skipped'] += self.frozen_count * (self.frozen_count-1)//2
        f, sig = self.inputs[k], Signature(1, self.backend.zero_monomial, k)
        self.current_signature = sig
        if self.witnesses is not None:
            self.witnesses.current = self.witnesses.unit(k)
        if self.paths is not None:
            self.paths.input(k, sig)
            input_root = self.paths.current_root
        else:
            input_root = None
        if not f:
            self._insert_syzygy(sig)
            self._process_queue()
            return
        self.phase = 'incremental input insertion'
        self._current_origin = InsertionOrigin('input', input_index=k)
        j = self._insert_basis(Label(sig, f))
        self.stats.counters['incremental_input_insertions'] += 1
        self._seed_prefix_syzygies(f, input_root)
        self._emit_pairs(j)
        self._process_queue()


def _minimal_records(engine):
    """Exact coefficient-aware minimal strong projection; no tail reduction.

    Input must already be a completed strong basis. Sorting places every
    possible proper term divisor first. The index stores positive coefficients,
    never monic/primitive-part replacements. The selected labels are original
    representatives up to a unit sign, so optional witnesses can be retained.
    """
    b = engine.backend
    engine.phase = 'incremental minimal strong basis'
    descriptors = [(mu, abs(c), j, 1 if c > 0 else -1)
                   for j, (mu, c) in enumerate(engine.leading)]
    descriptors.sort(key=lambda x: (b.order_key(x[0]), x[1], x[2]))
    chosen = []
    # For small squarefree unit leading terms, every proper divisor is one
    # of at most 255 support subsets. Membership in the complete input set is
    # sufficient: if a divisor is itself discarded, a minimal divisor below
    # it still exists. Thus no evolving index or repeated prefix insertion is
    # needed when every completed stage reloads its large frozen antichain.
    # A byte encoding still works as a support bitset because all coordinates
    # are zero or one; the set bits need not occupy consecutive positions.
    packed = []
    for mu, c, _, _ in descriptors:
        if c != 1 or sum(mu) > 8 or max(mu, default=0) > 1:
            packed = None
            break
        packed.append(int.from_bytes(bytes(mu), 'little'))
    if packed is not None:
        first = {}
        for (_, _, j, _), code in zip(descriptors, packed):
            first.setdefault(code, j)
        has_unit = 0 in first
        for (_, _, j, sign), code in zip(descriptors, packed):
            engine._check_budget()
            if first[code] != j:
                continue
            redundant = bool(code) and has_unit
            subset = (code - 1) & code
            while subset and not redundant:
                redundant = subset in first
                subset = (subset - 1) & code
            if not redundant:
                chosen.append((j, sign))
    else:
        index = SyzygyIndex(len(b.names))
        coeffs = {}
        for mu, c, j, sign in descriptors:
            engine._check_budget()
            if any(c % d == 0 for group in index.divisors(mu) for d in group):
                continue
            chosen.append((j, sign))
            bucket = coeffs.setdefault(mu, [])
            bucket.append(c)
            index.set(mu, bucket)
    if engine.options.debug:
        from .core import divides_monomial
        leads = [(engine.leading[j][0], abs(engine.leading[j][1])) for j, _ in chosen]
        for mu, c, _, _ in descriptors:
            assert any(c % d == 0 and divides_monomial(nu, mu) for nu, d in leads)
        for j, (mu, c) in enumerate(leads):
            assert not any(c % d == 0 and divides_monomial(nu, mu)
                           for h, (nu, d) in enumerate(leads) if h != j)
    return chosen


class IncrementalPOT:
    """Coordinator for I_i=(f_1,...,f_i); original input order is fixed."""
    def __init__(self, backend, inputs, order, options: Options | None = None):
        from .orders import ModuleOrder
        from .polynomial import IntegerPolynomialRing
        if not isinstance(backend, IntegerPolynomialRing) or not isinstance(order, ModuleOrder):
            raise ValueError('IncrementalPOT requires an integer polynomial ring and a compatible ModuleOrder')
        if options is not None and not isinstance(options, Options):
            raise ValueError('options must be an Options instance or None')
        inputs = [backend.parse(f) for f in inputs]
        order.validate_for(backend, inputs)
        options = replace(options, incremental_pot=True) if options is not None else Options(incremental_pot=True)
        options.validate()
        if options.reduction_kernel == 'packed' and backend.order_name not in ('grlex', 'grevlex'):
            raise ValueError('packed reduction requires grlex or grevlex; use heap/copy for lex')
        if order.name != 'pot' or order.positions != tuple(range(len(inputs))):
            raise ValueError('--incremental-pot requires natural-order POT (e1 < ... < em); '
                             'do not use g2, TOP, blocks or permuted --positions')
        self.backend, self.inputs, self.order, self.options = backend, inputs, order, options
        self.stats = Stats()
        self.stats.times_ns.update(j_tail_reduction=0, g_tail_reduction=0,
                                   syzygy_check=0, syzygy_complete=0,
                                   incremental_minimization=0)
        self.basis = []
        self.vectors = [] if options.track_representations else None
        self.paths = _IncrementalPaths(backend, len(inputs)) if options.record_paths else None
        self.path_roots = []
        self.stages = []
        self.start = self.progress_next = 0
        self.deadline = None
        self.started = False

    def run(self):
        if self.started:
            raise RuntimeError('An incremental solver instance may be run only once')
        self.started = True
        self.start = perf_counter_ns()
        cpu_start = process_time_ns()
        self.progress_next = self.start + int(self.options.progress_interval * 1e9)
        self.deadline = (self.start + int(self.options.max_seconds*1e9)
                         if self.options.max_seconds is not None else None)
        status, message, engine = 'completed', 'All POT stages completed and compacted', None
        current_stage = 0
        partial_active_size = None
        try:
            for k in range(len(self.inputs)):
                current_stage = k+1
                started = perf_counter_ns()
                before_counts = self.stats.counters.copy()
                before_times = self.stats.times_ns.copy()
                frozen = len(self.basis)
                engine = _POTStage(self, k)
                engine._check_budget()
                engine._initialize_and_run()
                engine._check_accounting()
                if engine.queue.active or engine.queue.heap:
                    raise AssertionError('Stage compression requires an exhausted queue')
                before = len(engine.P)
                t = perf_counter_ns()
                chosen = _minimal_records(engine)
                basis, vectors, roots = [], [], []
                for j, sign in chosen:
                    p = engine.P[j]
                    if sign < 0:
                        p = Label(Signature(-p.signature.coefficient, p.signature.monomial, p.signature.index),
                                  -p.polynomial)
                    basis.append(p)
                    if self.vectors is not None:
                        v = engine.witnesses.basis[j]
                        vectors.append(v if sign > 0 else tuple(-u for u in v))
                    if self.paths is not None:
                        root = self.paths.basis[j]
                        if sign < 0:
                            root = self.paths._add('unit', p.signature,
                                      [Edge(root, -1, self.backend.zero_monomial)])
                        roots.append(root)
                self.stats.times_ns['incremental_minimization'] += perf_counter_ns()-t
                counters = self.stats.counters
                counters['incremental_stages_completed'] += 1
                counters['incremental_p_discarded'] += before - len(basis)
                counters['incremental_s_retired'] += len(engine.S)
                stage = {
                    'input_index': k+1, 'status': 'completed', 'frozen_basis_size': frozen,
                    'stage_basis_size': before, 'minimal_basis_size': len(basis),
                    'discarded_records': before-len(basis), 'retired_syzygies': len(engine.S),
                    'seconds': (perf_counter_ns()-started)/1e9,
                    'counters': {name: int(value-before_counts[name]) for name, value in counters.items()
                                 if value != before_counts[name] and name not in ('p_peak','s_peak','queue_peak')},
                    'timings_seconds': {name: (value-before_times[name])/1e9
                                        for name, value in self.stats.times_ns.items()
                                        if value != before_times[name]},
                }
                self.basis, self.vectors, self.path_roots, self.stages = (
                    basis, vectors if self.vectors is not None else None, roots, self.stages + [stage])
                if self.options.progress_interval:
                    print(f'[stage {k+1}/{len(self.inputs)} completed] '
                          f'P={before} -> minimal={len(basis)} | retired S={len(engine.S)} | '
                          f'seconds={self.stages[-1]["seconds"]:.3f}', file=sys.stderr, flush=True)
                if self.paths is not None:
                    self.paths.basis = list(roots)
                    self.paths.syzygies = {}
                    self.paths.polynomials = self.basis
                # All live stage queues/caches/syzygies are now retired. No pending
                # candidate and no earlier-stage cover evidence survives.
                # Break queue -> bound filter -> engine before retiring the
                # stage, so non-retained P rows need not wait for cyclic GC.
                engine.queue.generation_filter = None
                engine.queue.generation_filter_packed = None
                # The bounded key cache also wraps a bound queue method.
                # Retired queues must release that self-cycle promptly.
                engine.queue._packed_leading_key.cache_clear()
                engine.queue._packed_leading_key = None
                engine = None
        except ComputationLimit as exc:
            status, message = 'limit', str(exc) + '; partial incremental output is NOT a basis of the full input ideal'
        except KeyboardInterrupt:
            status, message = 'interrupted', 'Interrupted; partial incremental output is NOT a basis of the full input ideal'
        finally:
            self.stats.total_ns = perf_counter_ns()-self.start
            self.stats.cpu_ns = process_time_ns()-cpu_start
            if engine is not None:
                partial_active_size = len(engine.P)
                # Aborted stages are private and cannot be resumed. Release
                # their bound callbacks and pending recipes just as completed
                # stages do, rather than retaining all indices until cyclic GC.
                engine.queue.generation_filter = None
                engine.queue.generation_filter_packed = None
                engine.queue._packed_leading_key.cache_clear()
                engine.queue._packed_leading_key = None
                engine.queue.active.clear()
                engine.queue.pending_j.clear()
                engine.queue.heap.clear()
                if self.paths is not None:
                    self.paths.basis = list(self.path_roots)
                    self.paths.syzygies = {}
                    self.paths.polynomials = self.basis
                engine = None
        # An interrupted stage is not a restart checkpoint. Publish only the
        # last committed prefix basis; its roots/vectors remain valid even if
        # the in-flight stage has appended more DAG nodes.
        graph = (GenerationPaths(tuple(self.paths.nodes), tuple(self.path_roots), (),
                                 self.paths.reduction_steps_recorded)
                 if self.paths is not None else None)
        metadata = {
            'format': 'gvwzz-incremental-pot-v1', 'input_order': list(range(1, len(self.inputs)+1)),
            'stages_completed': len(self.stages), 'stages_total': len(self.inputs),
            'current_stage': current_stage, 'last_completed_prefix': len(self.stages), 'stages': self.stages,
            'partial_stage_active_records': partial_active_size,
            'compression': 'minimal strong leading-term antichain over ZZ; unit signs only',
            'syzygy_scope': 'Only the current stage is maintained; no global syzygy basis is returned',
            'signature_scope': 'Valid original-input labels, but not a complete signature basis',
            'verification': 'Ordinary integer S/G zero remainders and input reductions; membership needs vectors or paths',
            'budget_scope': 'time, candidates and reduction steps are global; max_basis bounds active stage P',
        }
        return Result(status, message, self.backend, self.order, self.inputs,
                      self.basis, [], self.stats, self.options,
                      basis_representations=self.vectors,
                      syzygy_representations=[] if self.vectors is not None else None,
                      insertion_history=None, generation_paths=graph,
                      incremental_metadata=metadata)
