"""The general all-inputs-at-once GVW-type solver over ``ZZ``.

The default public algorithm is concentrated in this file: insert every input,
generate all initial parent pairs, and process candidates in nondecreasing
signature-monomial order. The optional incremental POT engine is implemented
separately in incremental.py; the default all-inputs-at-once path is unchanged.
"""
from __future__ import annotations

import sys
from bisect import insort
from collections import OrderedDict
from heapq import merge
from math import gcd
from time import perf_counter_ns, process_time_ns
from typing import Iterable

from .core import (
    Label, Monomial, Signature, add_monomials, divides_monomial,
    lcm_monomials, subtract_monomials, exact_integer, _signature_from_arithmetic,
)
from .certificates import WitnessTracker
from .paths import PathTracker
from .forest import InsertionOrigin
# These public names remain importable from solver for API compatibility.
from .options import Options
from .result import Result
from .runtime import environment
from .orders import ModuleOrder
from .monomial_packing import PackedMonomials, pack_monomial
from .polynomial import IntegerPolynomialRing, PackedWorkspace, PolynomialWorkspace
from .structures import CandidateQueue, CoverIndex, DivisorIndex, Stats, SyzygyStore
from .recipes import RecipeContext, JPolynomialRecipe, GPolynomialRecipe, check_polynomial

# The optional extension preserves arbitrary-size integer coefficients and
# the reference reducer policy. A source-only installation remains supported.
import os as _native_os
_native_coefficient_backend = 'python'
try:
    if _native_os.environ.get('GVWZZ_NO_NATIVE'):
        raise ImportError('native acceleration disabled')
    # The Windows wheel registers its bundled GMP DLL directory on import.
    # Load it before the optional extension's native dependencies are resolved.
    try:
        import gmpy2 as _native_gmpy2
    except (ImportError, OSError):
        pass
    from ._native_reduction import (find_reducer as _native_find_reducer,
                                    reduce_working as _native_reduce_working)
    from . import _native_reduction as _native_reduction_module
    _native_coefficient_backend = getattr(_native_reduction_module, 'COEFFICIENT_BACKEND', 'python')
except (ImportError, OSError):
    _native_find_reducer = _native_reduce_working = None

try:
    if _native_os.environ.get('GVWZZ_NO_NATIVE'):
        raise ImportError('native acceleration disabled')
    from ._native_pairs import emit_parent_pairs as _native_emit_parent_pairs
except (ImportError, OSError):
    _native_emit_parent_pairs = None

try:
    if _native_os.environ.get('GVWZZ_NO_NATIVE'):
        raise ImportError('native acceleration disabled')
    from ._native_queue_drain import select_candidate as _native_select_candidate
except (ImportError, OSError):
    _native_select_candidate = None


class ComputationLimit(RuntimeError):
    pass


class GVWInteger:
    def __init__(
        self,
        backend: IntegerPolynomialRing,
        inputs: list,
        module_order: ModuleOrder,
        options: Options | None = None,
        *,
        _validated_inputs: bool = False,
    ) -> None:
        if not isinstance(backend, IntegerPolynomialRing):
            raise ValueError("GVWInteger requires an IntegerPolynomialRing")
        if not isinstance(module_order, ModuleOrder):
            raise ValueError("GVWInteger requires a compatible ModuleOrder")
        if options is not None and not isinstance(options, Options):
            raise ValueError("options must be an Options instance or None")
        if _validated_inputs:
            # Internal staged construction only: IncrementalPOT has already
            # parsed/copied these inputs and checked this exact ModuleOrder.
            # Public construction retains the full validation below.
            inputs = list(inputs)
        else:
            inputs = [backend.parse(f) for f in inputs]
            module_order.validate_for(backend, inputs)
        self.backend = backend
        self.inputs = inputs
        self.order = module_order
        self.options = Options() if options is None else options
        self.options.validate()
        if self.options.reduction_kernel == "packed" and backend.order_name not in ("grlex", "grevlex"):
            raise ValueError("packed reduction requires grlex or grevlex; use heap/copy for lex")

        self.witnesses = WitnessTracker(backend, len(inputs)) if self.options.track_representations else None
        self.paths = PathTracker(backend, len(inputs)) if self.options.record_paths else None
        self.stats = Stats()
        self.stats.times_ns.update(j_tail_reduction=0, g_tail_reduction=0,
                                   syzygy_check=0, syzygy_complete=0)
        self.S = SyzygyStore(len(backend.names), len(inputs), self.stats, self.options.debug)
        self.P: list[Label] = []
        # Lightweight provenance is separate from algebraic labels/certificates.
        self.insertion_history: list[InsertionOrigin | None] = []
        self._current_origin: InsertionOrigin | None = None

        # Derived metadata parallel to P.  The algebraic record itself remains
        # exactly Label(signature, polynomial).
        self.leading: list[tuple[Monomial, int]] = []
        self.reducer_buckets: dict[Monomial, list[int]] = {}
        self.reducer_index = DivisorIndex(len(backend.names))
        self.unit_reducer_index = DivisorIndex(len(backend.names))
        self.unit_reducer_buckets: dict[Monomial, list[int]] = {}
        self.top_cache = OrderedDict()
        self.reducer_query_cache = OrderedDict()
        self.reducer_rank: list[tuple] = []
        self.coefficient_heights: list[int] = []
        self.ratio_keys: list[tuple] = []
        self.ratio_exponents: list[Monomial] = []
        # Pair generation only needs compact arithmetic until a candidate
        # survives the exact queue and syzygy tests. 63 leaves room for sums
        # of two parent coordinates without crossing the byte-lane guard.
        self.pair_packing = PackedMonomials(len(backend.names))
        materialize_override = None
        if (type(self)._materialize_polynomial is not GVWInteger._materialize_polynomial
                or type(self)._check_polynomial is not GVWInteger._check_polynomial):
            materialize_override = self._materialize_polynomial
        self._recipe_context = RecipeContext(
            backend, self.pair_packing, self.stats.counters, self.options,
            ComputationLimit, materialize_override, self)
        self.packed_parents: list[tuple[int, int] | None] = []
        self.top_nonunit_buckets: dict[Monomial, dict[int, int]] = {}
        self.top_nonunit_index = DivisorIndex(len(backend.names))
        self.cover = CoverIndex(
            backend, module_order, self.S, self.stats,
            self.options.cover_mode, self.options.cover_cache_size,
        )
        self.queue = self._new_queue()
        self._native_queue_syzygy = None

        self.deadline: int | None = None
        self.started = False
        self.phase = "initialization"
        self._progress_start = 0
        self._progress_next = 0
        self.current_signature: Signature | None = None
        self.current_candidate_kind: str | None = None
        self.current_candidate_serial: int | None = None
        self._tail_kind: str | None = None
        self._tail_steps_start = 0

    # ------------------------------------------------------------------
    # State initialization, limits, and insertion
    # ------------------------------------------------------------------

    def _new_queue(self) -> CandidateQueue:
        return CandidateQueue(
            self.backend, self.order, self.stats,
            self.options.j_deduplicate, self.options.g_first,
            self.options.lazy_candidates,
            self._j_generation_covered if self.options.j_generation_filter else None,
            tie_policy=self.options.queue_tie,
            generation_filter_packed=(self._j_generation_covered_packed
                                      if self.options.j_generation_filter else None))

    def _check_budget(self) -> None:
        now = perf_counter_ns()
        if self.deadline is not None and now > self.deadline:
            raise ComputationLimit('max_seconds exceeded (cooperative time limit)')
        if self.options.progress_interval and now >= self._progress_next:
            self._report_progress(now)
        c, o = self.stats.counters, self.options
        if o.max_candidates is not None and c['j_generated'] + c['g_generated'] > o.max_candidates:
            total = c['j_generated'] + c['g_generated']
            raise ComputationLimit(f'max_candidates exceeded: generated J+G={total:,} > {o.max_candidates:,}; '
                                   'set --max-candidates 0 to disable this budget')
        if o.max_reduction_steps is not None and c['regular_steps'] + c['singular_steps'] + c['euclidean_steps'] + c['tail_steps'] >= o.max_reduction_steps:
            raise ComputationLimit('max_reduction_steps reached')

    def _report_progress(self, now: int) -> None:
        self._progress_next = now + int(self.options.progress_interval * 1e9)
        c = self.stats.counters
        sig = self.current_signature
        signature = (f'e{sig.index+1}:{sig.monomial}' if sig is not None else '-')
        j_tail = c['j_tail_steps']
        g_tail = c['g_tail_steps']
        # Include the in-flight call without extra work on each reduction step.
        if self._tail_kind == 'J':
            j_tail += c['tail_steps'] - self._tail_steps_start
        elif self._tail_kind == 'G':
            g_tail += c['tail_steps'] - self._tail_steps_start
        candidate_id = (f'{self.current_candidate_kind}#{self.current_candidate_serial}'
                        if self.current_candidate_kind is not None else '-')
        print(f'[progress {((now-self._progress_start)/1e9):.1f}s] '
              f'phase={self.phase} candidate={candidate_id} P={len(self.P)} S={len(self.S)} '
              f'queue={len(self.queue.active)} '
              f'generated={c["j_generated"] + c["g_generated"]} '
              f'head={c["regular_steps"]+c["singular_steps"]+c["euclidean_steps"]} '
              f'tail={c["tail_steps"]} tail(J/G)={j_tail}/{g_tail} '
              f'popped(J/G)={c["j_popped"]}/{c["g_popped"]} '
              f'signature={signature}',
              file=sys.stderr, flush=True)

    def _check_polynomial(self, p) -> None:
        check_polynomial(p, self.options, ComputationLimit)

    def _insert_basis(self, label: Label, *, timing_bucket: str | None = 'index') -> int:
        self._check_budget()
        self._check_polynomial(label.polynomial)
        if self.options.max_basis is not None and len(self.P) >= self.options.max_basis:
            raise ComputationLimit('max_basis reached')
        start = perf_counter_ns()
        try:
            lm, lc = self.backend.leading(label.polynomial)
            i = len(self.P)
            self.reducer_query_cache.clear()
            self.P.append(label)
            self.insertion_history.append(self._current_origin)
            self._current_origin = None
            self.leading.append((lm, lc))
            packed_lm = pack_monomial(lm, 63)
            packed_sig = pack_monomial(label.signature.monomial, 63)
            self.packed_parents.append((packed_lm, packed_sig)
                if packed_lm is not None and packed_sig is not None else None)
            height = max(abs(int(c)).bit_length() for c in label.polynomial.values())
            self.coefficient_heights.append(height)
            delta = subtract_monomials(label.signature.monomial, lm)
            ratio_key = self.order.key_monomial(delta, label.signature.index)
            self.ratio_keys.append(ratio_key)
            self.ratio_exponents.append(delta)
            if self.options.reducer_selection == 'newest':
                rank = (-i,)
            elif self.options.reducer_selection == 'unit':
                rank = (abs(lc) != 1, len(label.polynomial), height, i)
            elif self.options.reducer_selection in ('height', 'balanced'):
                relative_height = height - abs(lc).bit_length()
                nterms = len(label.polynomial)
                rank = ((relative_height, nterms, abs(lc).bit_length(), i)
                        if self.options.reducer_selection == 'height' else
                        (nterms * (1 + relative_height), relative_height, nterms, i))
            else:
                rank = (len(label.polynomial), abs(lc).bit_length(), i)
            self.reducer_rank.append(rank)
            if lm not in self.reducer_buckets:
                self.reducer_buckets[lm] = []
                self.reducer_index.set(lm, self.reducer_buckets[lm])
            if self.options.reducer_selection == 'oldest':
                self.reducer_buckets[lm].append(i)
            else:
                insort(self.reducer_buckets[lm], i, key=self.reducer_rank.__getitem__)
            if abs(lc) == 1:
                if lm not in self.unit_reducer_buckets:
                    self.unit_reducer_buckets[lm] = []
                    self.unit_reducer_index.set(lm, self.unit_reducer_buckets[lm])
                bucket = self.unit_reducer_buckets[lm]
                if not bucket or ratio_key < self.ratio_keys[bucket[0]]:
                    bucket[:] = [i]
            else:
                if lm not in self.top_nonunit_buckets:
                    self.top_nonunit_buckets[lm] = {}
                    self.top_nonunit_index.set(lm, self.top_nonunit_buckets[lm])
                bucket = self.top_nonunit_buckets[lm]
                previous = bucket.get(abs(lc))
                if previous is None or ratio_key < self.ratio_keys[previous]:
                    bucket[abs(lc)] = i
            self.cover.insert(label, lm)
            self.stats.counters['p_peak'] = max(
                self.stats.counters['p_peak'], len(self.P))
            self.stats.counters['max_polynomial_terms'] = max(
                self.stats.counters['max_polynomial_terms'], len(label.polynomial))
            self.stats.counters['max_coefficient_bits'] = max(
                self.stats.counters['max_coefficient_bits'],
                height)
            self.stats.counters['max_signature_bits'] = max(
                self.stats.counters['max_signature_bits'], abs(label.signature.coefficient).bit_length())
            if self.witnesses is not None:
                self.witnesses.append_basis()
            if self.paths is not None:
                self.paths.append_basis(label.signature)
            return i
        finally:
            if timing_bucket is not None:
                self.stats.times_ns[timing_bucket] += perf_counter_ns() - start


    # ------------------------------------------------------------------
    # Parent-pair generation
    # ------------------------------------------------------------------

    def _insert_syzygy(self, sig: Signature | tuple, *, check_budget: bool = True,
                       representation=None, path_factory=None,
                       monomial_packed: bool = False) -> None:
        """Check first; materialize a principal seed and snapshot S only if needed.

        Tuples are internal (coefficient, monomial, component) descriptors made
        solely from checked parents in _emit_pair. Public input paths use the
        validating Signature constructor. No S mutation occurs before the gate.
        """
        start = perf_counter_ns()
        previous_phase = self.phase
        self.phase = 'syzygy redundancy check'
        completion_start = None
        try:
            if isinstance(sig, Signature):
                coefficient, monomial, index = sig.coefficient, sig.monomial, sig.index
                deferred = False
            else:
                coefficient, monomial, index = sig
                deferred = True
            if monomial_packed:
                self.stats.counters['syzygy_insert_calls'] += 1
                accepted = not self.S._has_divisor_packed(coefficient, monomial, index)
                if not accepted:
                    self.stats.counters['syzygy_seed_rejected'] += 1
            else:
                accepted = self.S._accept_seed_parts(coefficient, monomial, index)
            if not accepted:
                if deferred:
                    self.stats.counters['principal_syzygy_objects_avoided'] += 1
            else:
                completion_start = perf_counter_ns()
                self.phase = 'syzygy completion'
                if deferred:
                    if monomial_packed:
                        monomial = self.pair_packing.unpack(monomial)
                    sig = _signature_from_arithmetic(coefficient, monomial, index)
                old = tuple(self.S.components[index]) if (self.witnesses is not None or self.paths is not None) else ()
                self.S._complete_uncovered(sig)
                if self.witnesses is not None:
                    vector = representation() if callable(representation) else representation
                    if vector is None:
                        vector = self.witnesses.current
                    if vector is None:
                        raise RuntimeError("Missing representation of syzygy seed")
                    self.witnesses.update_syzygies(sig, vector, old, self.S.components[index])
                if self.paths is not None:
                    node = path_factory() if path_factory is not None else self.paths.freeze_current(sig)
                    self.paths.update_syzygies(sig, node, old, self.S.components[index])
        finally:
            end = perf_counter_ns()
            self.stats.times_ns['syzygy'] += end - start
            if completion_start is None:
                self.stats.times_ns['syzygy_check'] += end - start
            else:
                self.stats.times_ns['syzygy_check'] += completion_start - start
                self.stats.times_ns['syzygy_complete'] += end - completion_start
            try:
                if self.options.progress_interval and end >= self._progress_next:
                    self._report_progress(end)
            finally:
                self.phase = previous_phase
        if check_budget:
            self._check_budget()

    def _materialize_polynomial(self, factory, kind):
        # Lazy recipes use immutable parent polynomials, never u-vectors.
        self.stats.counters[kind.lower() + '_materialized'] += 1
        poly = factory()
        self._check_polynomial(poly)
        return poly

    def _j_generation_covered(self, sig: Signature) -> bool:
        """Only an already present exact syzygy divisor may reject a J here.

        This is a single cover and requires no prefix completeness. In contrast,
        a G candidate must not be syzygy-rejected at generation time.
        """
        t = perf_counter_ns()
        try:
            self.stats.counters['j_generation_tests'] += 1
            return self.S.has_divisor(sig)
        finally:
            self.stats.times_ns['j_generation_test'] += perf_counter_ns() - t

    def _skip_principal_pair(self, i: int, j: int) -> bool:
        return False

    def _emit_pairs(self, j: int) -> None:
        """Emit the ordered parent row; staged engines can prune known covers."""
        # Subclasses retain their own pair hooks and staged invariants.
        counts = self.stats.counters
        counts['parent_dispatch_rows'] += 1
        if (type(self) is GVWInteger and _native_emit_parent_pairs is not None
                and _native_emit_parent_pairs(self, j)):
            return
        counts['parent_python_rows'] += 1
        if _native_emit_parent_pairs is None:
            counts['parent_fallback_unavailable_rows'] += 1
        elif type(self) is not GVWInteger:
            counts['parent_fallback_engine_rows'] += 1
        for i in range(j):
            self._emit_pair(i, j)

    def _j_generation_covered_packed(self, coefficient: int, packed: int, index: int) -> bool:
        """The same exact single-syzygy test without allocating a Signature."""
        t = perf_counter_ns()
        try:
            self.stats.counters['j_generation_tests'] += 1
            return self.S._has_divisor_packed(coefficient, packed, index)
        finally:
            self.stats.times_ns['j_generation_test'] += perf_counter_ns() - t

    def _emit_pair(self, i: int, j: int) -> None:
        # The end-of-pair check also accounts for the principal-syzygy update.
        # Avoid three identical budget scans for every parent pair.
        self.phase = 'candidate generation'
        counts, times = self.stats.counters, self.stats.times_ns
        counts['parent_pairs'] += 1
        p, q = self.P[i], self.P[j]
        mp, cp = self.leading[i]
        mq, cq = self.leading[j]
        sp, sq = p.signature, q.signature
        start = perf_counter_ns()
        filter_ns = 0
        try:
            # These two comparisons have the same sign:
            #   LM(q)*sig(p) versus LM(p)*sig(q),
            #   (L/LM(p))*sig(p) versus (L/LM(q))*sig(q).
            # Compute it once from cached formal ratios, without temporary labels.
            ki, kj = self.ratio_keys[i], self.ratio_keys[j]
            cmp = 1 if ki > kj else (-1 if ki < kj else 0)
            packed_p, packed_q = self.packed_parents[i], self.packed_parents[j]
            packed = packed_p is not None and packed_q is not None
            syzygy_packed = packed
            if packed:
                packed_mp, packed_sp = packed_p
                packed_mq, packed_sq = packed_q
            skip_principal = self._skip_principal_pair(i, j)
            if skip_principal:
                syz = None
            elif cmp > 0:
                syz = (cq * sp.coefficient,
                       packed_mq + packed_sp if packed else add_monomials(mq, sp.monomial), sp.index)
            elif cmp < 0:
                syz = (-cp * sq.coefficient,
                       packed_mp + packed_sq if packed else add_monomials(mp, sq.monomial), sq.index)
            else:
                c = cq * sp.coefficient - cp * sq.coefficient
                syz = ((c, packed_mq + packed_sp if packed else add_monomials(mq, sp.monomial), sp.index)
                       if c else None)

            if packed:
                packed_W = self.pair_packing.lcm(packed_mp, packed_mq)
                W = None
            else:
                W = lcm_monomials(mp, mq)
            # Divisible leading coefficients need neither gcd nor Bezout.
            # This is especially common over ZZ with monic input families.
            if cp == cq or cp == -cq:
                d, ell, comparable = abs(cp), abs(cp), True
            elif cp % cq == 0:
                d, ell, comparable = abs(cq), abs(cp), True
            elif cq % cp == 0:
                d, ell, comparable = abs(cp), abs(cq), True
            else:
                d = gcd(cp, cq)
                ell, comparable = abs((cp // d) * cq), False
            jdesc = None
            # Reuse the leading-side exponent in J construction and G precheck.
            if packed:
                if cmp >= 0:
                    upper_exp, upper_index = packed_W - packed_mp + packed_sp, sp.index
                else:
                    upper_exp, upper_index = packed_W - packed_mq + packed_sq, sq.index
            elif cmp >= 0:
                upper_exp, upper_index = add_monomials(W, self.ratio_exponents[i]), sp.index
            else:
                upper_exp, upper_index = add_monomials(W, self.ratio_exponents[j]), sq.index
            if cmp:
                counts['j_generated'] += 1
                if cmp > 0:
                    c = ell // cp
                    jdesc = c * sp.coefficient, p.polynomial, c, mp
                else:
                    c = ell // cq
                    jdesc = c * sq.coefficient, q.polynomial, c, mq

            # Count one labelled G-candidate per unordered parent pair, even
            # when an exact top witness rejects it before materialization.
            counts['g_generated'] += 1
            gdata = None
            if not comparable:
                # G reduction APIs need ordinary exponent vectors. This
                # conversion is skipped altogether for comparable parents.
                if packed:
                    W = self.pair_packing.unpack(packed_W)
                    upper_exp = self.pair_packing.unpack(upper_exp)
                    packed = False
                reject = False
                if self.options.g_precheck:
                    t = perf_counter_ns()
                    counts['reducer_queries'] += 1
                    reject = self._top_reducer_index(upper_exp, upper_index, W, d) is not None
                    filter_ns = perf_counter_ns() - t
                    times['g_generation_test'] += filter_ns
                    counts['g_generation_tests'] += 1
                if reject:
                    counts['g_generation_top_rejected'] += 1
                    counts['g_discarded'] += 1
                else:
                    # A/B construction is done only for the surviving G's.
                    # Keep exactly the same gcdex and adjustment as algebra.py.
                    up, uq = subtract_monomials(W, mp), subtract_monomials(W, mq)
                    a, b, _ = self.backend.bezout(cp, cq)
                    adjusted = False
                    if cmp > 0:
                        if not a:
                            a += cq // d; b -= cp // d; adjusted = True
                        coefficient = a * sp.coefficient
                    elif cmp < 0:
                        if not b:
                            a += cq // d; b -= cp // d; adjusted = True
                        coefficient = b * sq.coefficient
                    else:
                        coefficient = a * sp.coefficient + b * sq.coefficient
                        if not coefficient:
                            a += cq // d; b -= cp // d; adjusted = True
                            coefficient = a * sp.coefficient + b * sq.coefficient
                    sig = _signature_from_arithmetic(coefficient, upper_exp, upper_index)
                    counts['candidate_signature_fast_builds'] += 1
                    counts['bezout_adjustments'] += int(adjusted)
                    factory = GPolynomialRecipe(
                        self._recipe_context, p.polynomial, a, up, q.polynomial, b, uq)
                    gdata = sig, d, factory
            else:
                # If either LC divides the other, that parent exactly
                # top-reduces the G-candidate. No Bezout construction or
                # queue entry is needed, but the candidate is still counted.
                counts['g_omitted_comparable'] += 1
                counts['g_discarded'] += 1
        finally:
            times['generation'] += perf_counter_ns() - start - filter_ns
        if syz is not None:
            counts['principal_syzygies_detected'] += 1
            if self.paths is not None:
                self._insert_syzygy(syz, check_budget=False,
                    monomial_packed=syzygy_packed,
                    path_factory=lambda: self.paths.principal(i, j, _signature_from_arithmetic(
                        syz[0], self.pair_packing.unpack(syz[1]) if syzygy_packed else syz[1], syz[2])))
            elif self.witnesses is None:
                self._insert_syzygy(syz, check_budget=False, monomial_packed=syzygy_packed)
            else:
                def principal_vector():
                    return tuple(q.polynomial * u - p.polynomial * v
                                 for u, v in zip(self.witnesses.basis[i], self.witnesses.basis[j]))
                self._insert_syzygy(syz, check_budget=False, representation=principal_vector,
                                   monomial_packed=syzygy_packed)
        elif skip_principal:
            counts['incremental_principal_skipped'] += 1
            if (getattr(self, '_principal_pairs_redundant', False)
                    and i >= self.frozen_count and j >= self.frozen_count):
                counts['incremental_current_principal_skipped'] += 1
        else:
            counts['principal_syzygies_unknown'] += 1
        start = perf_counter_ns()
        j_filter_before = times['j_generation_test']
        try:
            if jdesc is not None:
                coefficient, f, c, parent_lm = jdesc
                if packed:
                    inserted = self.queue._accept_j_packed(coefficient, upper_exp, upper_index, packed_W)
                    if inserted:
                        packed_shift = packed_W - (packed_mp if cmp > 0 else packed_mq)
                        factory = JPolynomialRecipe(self._recipe_context, f, c, packed_shift)
                        entry = self.queue._insert_j_packed(coefficient, upper_exp, upper_index, packed_W,
                            factory)
                        if self.paths is not None or self.witnesses is not None:
                            m = self.pair_packing.unpack(packed_shift)
                    else:
                        counts['candidate_signature_objects_avoided'] += 1
                else:
                    sig = _signature_from_arithmetic(coefficient, upper_exp, upper_index)
                    counts['candidate_signature_fast_builds'] += 1
                    inserted = self.queue._accept_j(sig, W)
                    if inserted:
                        m = subtract_monomials(W, parent_lm)
                        factory = JPolynomialRecipe(self._recipe_context, f, c, m, packed=False)
                        self.queue._insert('J', sig, W, factory=factory)
                        entry = self.queue.active[self.queue.serial]
                if inserted:
                    entry.generating_parents = (i, j)
                if inserted and self.paths is not None:
                    parent = i if cmp > 0 else j
                    entry.path_recipe = ((parent, c, m),)
                if inserted and self.witnesses is not None:
                    parent = i if cmp > 0 else j
                    entry.representation_recipe = (
                        lambda parent=parent, c=c, m=m: self.witnesses.scale(self.witnesses.basis[parent], c, m))
            if gdata is not None:
                sig, lc, factory = gdata
                self.queue.push_g_lazy(sig, W, lc, factory)
                self.queue.active[self.queue.serial].generating_parents = (i, j)
                if self.paths is not None:
                    self.queue.active[self.queue.serial].path_recipe = ((i, a, up), (j, b, uq))
                if self.witnesses is not None:
                    self.queue.active[self.queue.serial].representation_recipe = (
                        lambda i=i, j=j, a=a, b=b, up=up, uq=uq:
                        self.witnesses.combination(i, a, up, j, b, uq))
        finally:
            # The callback's time belongs to the J-generation filter, not twice
            # to both queue time and filter time.
            times['queue'] += perf_counter_ns() - start - (times['j_generation_test'] - j_filter_before)
        self._check_budget()

    # ------------------------------------------------------------------
    # Reducer lookup
    # ------------------------------------------------------------------

    def _top_reducer_index(self, exponent, component, W, coeff):
        """Return any exact top witness without constructing scaled signatures.

        For existence queries only, at a fixed polynomial LM and |LC| the
        smallest formal ratio dominates the others. Actual reduction selection
        still uses the complete, unchanged P and its original selection policy.
        """
        key = (exponent, component, W, abs(coeff))
        hit = self.top_cache.get(key)
        if hit is not None and (hit[1] is not None or hit[0] == len(self.P)):
            self.stats.counters['top_cache_hits'] += 1
            self.top_cache.move_to_end(key)
            return hit[1]
        target = self.order.key_monomial(subtract_monomials(exponent, W), component)
        chosen = None
        for bucket in self.unit_reducer_index.divisors(W):
            i = bucket[0]
            self.stats.counters['reducer_candidates'] += 1
            if self.ratio_keys[i] <= target:
                chosen = i
                break
        if chosen is None and abs(coeff) != 1:
            for bucket in self.top_nonunit_index.divisors(W):
                for c, i in bucket.items():
                    self.stats.counters['reducer_candidates'] += 1
                    if coeff % c == 0 and self.ratio_keys[i] <= target:
                        chosen = i
                        break
                if chosen is not None:
                    break
        if self.options.cover_cache_size:
            self.top_cache[key] = (len(self.P), chosen)
            self.top_cache.move_to_end(key)
            if len(self.top_cache) > self.options.cover_cache_size:
                self.top_cache.popitem(last=False)
        return chosen

    def _find_top_reducer(self, signature, W, coeff):
        chosen = self._top_reducer_index(signature.monomial, signature.index, W, coeff)
        if chosen is None:
            return None
        q = self.P[chosen]
        M, c = self.leading[chosen]
        scalar = coeff // c
        shift = subtract_monomials(W, M)
        target = self.order.key_monomial(
            subtract_monomials(signature.monomial, W), signature.index)
        kind = ('regular' if self.ratio_keys[chosen] < target else
                'super' if scalar * q.signature.coefficient == signature.coefficient else 'singular')
        return chosen, scalar, shift, kind

    def _find_reducer(self, signature: Signature, leading, mode: str,
                      allow_euclidean: bool = False):
        """Exact strict/any tests and signature-safe mixed reduction.

With signature-first, a partial Euclidean step of the preferred signature
class can precede an exact step of the other class. With exact-first, partial
steps are used only after all executable exact head steps are exhausted.

        Never use Euclidean-only reducibility to discard a G-polynomial:
        the theorem requires strong (exact coefficient) top reducibility.
        """
        if _native_find_reducer is not None and mode != 'any':
            return _native_find_reducer(self, signature, leading, mode, allow_euclidean)
        self.stats.counters['reducer_queries'] += 1
        W, coeff = leading
        if mode == 'any':
            return self._find_top_reducer(signature, W, coeff)
        # During head/tail reduction P is fixed and signature monomials do
        # not change. Reuse the geometric query independently of coefficients.
        cache_key = (signature.monomial, signature.index, W)
        cached = self.reducer_query_cache.get(cache_key)
        if cached is not None:
            self.stats.counters['reducer_geometry_cache_hits'] += 1
            self.reducer_query_cache.move_to_end(cache_key)
            target_key, ids = cached
        else:
            target_key = self.order.key_monomial(
                subtract_monomials(signature.monomial, W), signature.index)
            buckets = list(self.reducer_index.divisors(W))
            ordered = (buckets[0] if len(buckets) == 1 else
                       merge(*buckets, key=(self.reducer_rank.__getitem__
                             if self.options.reducer_selection != 'oldest' else None)))
            ids = tuple(i for i in ordered if self.ratio_keys[i] <= target_key)
            self.reducer_query_cache[cache_key] = (target_key, ids)
            if len(self.reducer_query_cache) > 4096:
                self.reducer_query_cache.popitem(last=False)
        saved = None
        euclidean_choice = None
        euclidean_score = None
        preferred = 'regular' if self.options.reducer_policy == 'regular-first' else 'singular'
        for i in ids:
            self.stats.counters['reducer_candidates'] += 1
            M, c = self.leading[i]
            exact = coeff % c == 0
            if not exact and not (mode == 'mixed' and allow_euclidean):
                continue
            q = self.P[i]
            key = self.ratio_keys[i]
            if key > target_key:
                continue
            if exact:
                scalar = coeff // c
            else:
                # Nearest-integer quotient, with ties toward +infinity after
                # making the divisor positive. All arithmetic remains integral.
                b = abs(c)
                quotient = (coeff + b // 2) // b
                scalar = quotient if c > 0 else -quotient
                rem = coeff - scalar * c
                if not scalar or abs(rem) >= abs(coeff):
                    continue
            if key < target_key:
                rtype = 'regular'
            elif scalar * q.signature.coefficient == signature.coefficient:
                rtype = 'super'
            else:
                rtype = 'singular'
            if not exact:
                if rtype == 'super':
                    self.stats.counters['euclidean_sigdrop_skipped'] += 1
                    continue
                rank = (0 if rtype == preferred else 1) if self.options.euclidean_priority == 'signature-first' else 0
                score = (rank, abs(rem), len(q.polynomial), abs(c).bit_length(), i)
                if euclidean_score is None or score < euclidean_score:
                    euclidean_choice = (i, scalar, subtract_monomials(W, M), 'euclidean-' + rtype)
                    euclidean_score = score
                continue
            choice = (i, scalar, subtract_monomials(W, M), rtype)
            if mode == 'any':
                return choice
            if mode == 'strict':
                if rtype == 'regular':
                    return choice
                continue
            if rtype == 'super':
                continue
            if self.options.reducer_policy == 'first' or rtype == preferred:
                return choice
            if saved is None:
                saved = choice
        if (self.options.euclidean_priority == 'signature-first' and saved is not None
                and euclidean_choice is not None and self.options.reducer_policy != 'first'
                and euclidean_choice[3] == 'euclidean-' + preferred):
            return euclidean_choice
        return saved if saved is not None else euclidean_choice

    # ------------------------------------------------------------------
    # Head and tail reduction
    # ------------------------------------------------------------------

    def _apply_reduction(self, label: Label, reducer, *, tail_term=None) -> Label:
        i, coefficient, monomial, kind = reducer
        euclidean = kind.startswith('euclidean-')
        if euclidean:
            kind = kind[len('euclidean-'):]
        q = self.P[i]
        sig = label.signature
        poly = label.polynomial - self.backend._multiply_term_from_arithmetic(q.polynomial, coefficient, monomial)
        if kind == 'singular':
            sig = Signature(sig.coefficient - coefficient * q.signature.coefficient,
                            sig.monomial, sig.index)
        elif kind != 'regular':
            raise AssertionError('Super reductions must never be executed')
        if tail_term is not None:
            self.stats.counters['tail_steps'] += 1
            self.stats.counters['tail_euclidean_steps' if euclidean else 'tail_exact_steps'] += 1
        elif euclidean:
            self.stats.counters['euclidean_steps'] += 1
            self.stats.counters['euclidean_' + kind + '_steps'] += 1
        else:
            self.stats.counters[kind + '_steps'] += 1
        self._check_polynomial(poly)
        if self.options.debug:
            assert sig.monomial == label.signature.monomial and sig.index == label.signature.index
            if tail_term is None and poly:
                old_lm, old_c = self.backend.leading(label.polynomial)
                new_lm, new_c = self.backend.leading(poly)
                assert (self.backend.order_key(new_lm), abs(new_c)) < (self.backend.order_key(old_lm), abs(old_c))
            elif tail_term is not None:
                assert self.backend.leading(poly) == self.backend.leading(label.polynomial)
                assert abs(int(poly.get(tail_term, 0))) < abs(int(label.polynomial[tail_term]))
        if self.witnesses is not None:
            self.witnesses.subtract(i, coefficient, monomial)
        if self.paths is not None:
            self.paths.subtract(i, coefficient, monomial)
        return Label(sig, poly)

    def _workspace_step(self, signature, work, reducer, *, tail=False):
        i, coefficient, shift, kind = reducer
        euclidean = kind.startswith('euclidean-')
        if euclidean:
            kind = kind[len('euclidean-'):]
        q = self.P[i]
        old_lead = work.leading() if self.options.debug else None
        if kind == 'singular':
            signature = Signature(signature.coefficient - coefficient * q.signature.coefficient,
                                  signature.monomial, signature.index)
        elif kind != 'regular':
            raise AssertionError('Super reductions must never be executed')
        work.subtract_multiple(q.polynomial, coefficient, shift)
        if self.witnesses is not None:
            self.witnesses.subtract(i, coefficient, shift)
        if self.paths is not None:
            self.paths.subtract(i, coefficient, shift)
        c = self.stats.counters
        if tail:
            c['tail_steps'] += 1
            c['tail_euclidean_steps' if euclidean else 'tail_exact_steps'] += 1
        elif euclidean:
            c['euclidean_steps'] += 1
            c['euclidean_' + kind + '_steps'] += 1
        else:
            c[kind + '_steps'] += 1
        # A rigorous coefficient-height upper bound avoids re-scanning every
        # unchanged coefficient at each step. Fall back to an exact scan when
        # the bound is too loose to certify the configured safety limit.
        limit = self.options.max_polynomial_terms
        if limit is not None and len(work.polynomial) > limit:
            raise ComputationLimit('max_polynomial_terms exceeded')
        bound = self.options.max_coefficient_bits
        if bound is not None:
            work.height_upper = max(work.height_upper,
                abs(coefficient).bit_length() + self.coefficient_heights[i]) + 1
            if work.height_upper > bound:
                self.stats.counters['coefficient_bound_rescans'] += 1
                work.height_upper = max((abs(int(c)).bit_length()
                    for c in work.polynomial.values()), default=0)
                if work.height_upper > bound:
                    raise ComputationLimit('max_coefficient_bits exceeded')
        if self.options.debug:
            assert old_lead is not None
            new_lead = work.leading()
            if new_lead is not None:
                assert (self.backend.order_key(new_lead[0]), abs(new_lead[1])) < (
                    self.backend.order_key(old_lead[0]), abs(old_lead[1]))
        return signature

    def _reduce_working(self, label: Label, *, tail=False) -> Label:
        if (_native_reduce_working is not None
                and self.backend.order_name in ('grlex', 'grevlex')):
            return _native_reduce_working(self, label, tail)
        if self.options.reduction_kernel == 'packed':
            work = PackedWorkspace(self.backend, label.polynomial)
            self.stats.counters['packed_workspaces'] += 1
        else:
            work = PolynomialWorkspace(self.backend, label.polynomial)
        sig = label.signature
        work.height_upper = max((abs(int(c)).bit_length()
            for c in label.polynomial.values()), default=0)
        if tail:
            work.finish_term()  # Polynomial leading term remains untouched.
        tail_cap = self.options.tail_step_cap if tail else 0
        steps = 0
        while (leading := work.leading()) is not None:
            self._check_budget()
            if tail_cap and steps >= tail_cap:
                # Optional tail normalisation may stop; head reduction NEVER does.
                # Keep the current label, preserving polynomial LT and signature monomial.
                self.stats.counters['tail_cap_reached'] += 1
                break
            reducer = self._find_reducer(sig, leading, 'mixed',
                                         allow_euclidean=self.options.euclidean)
            if reducer is None:
                if not tail:
                    break
                work.finish_term()
                continue
            sig = self._workspace_step(sig, work, reducer, tail=tail)
            steps += 1
        return Label(sig, work.result())

    def _reduce_after_first(self, label: Label) -> Label:
        if self.options.reduction_kernel in ('heap', 'packed'):
            return self._reduce_working(label)
        while label.polynomial:
            self._check_budget()
            reducer = self._find_reducer(label.signature, self.backend.leading(label.polynomial), 'mixed',
                                         allow_euclidean=self.options.euclidean)
            if reducer is None:
                break
            label = self._apply_reduction(label, reducer)
        return label

    def _reduce_j(self, label: Label) -> Label:
        self.stats.counters['j_reduction_calls'] += 1
        self._check_budget()
        if (_native_reduce_working is not None
                and self.options.reduction_kernel in ('heap', 'packed')
                and self.backend.order_name in ('grlex', 'grevlex')
                and self.options.max_polynomial_terms is None
                and self.options.max_coefficient_bits is None):
            label = _native_reduce_working(self, label, False, True)
        else:
            reducer = self._find_reducer(label.signature, self.backend.leading(label.polynomial), 'strict')
            if reducer is None:
                raise AssertionError('Every retained regular J-pair must have a strict first reducer')
            # First step ALWAYS exact, strictly smaller multiplied signature.
            label = self._reduce_after_first(self._apply_reduction(label, reducer))
        if self.options.debug and label.polynomial:
            assert self._find_reducer(label.signature, self.backend.leading(label.polynomial), 'any') is None, (
                'A super reducer at the end would contradict the exact multi-cover pretest')
        return label

    def _reduce_tail(self, label: Label) -> Label:
        """Signature-safe optional tail reduction, without changing polynomial LT.

        Consider tail terms in decreasing monomial order. Skip a remainder once
        no usable reducer exists. No old P record is modified in place.
        """
        if not self.options.tail_reduce or len(label.polynomial) <= 1:
            return label
        lead = self.backend.leading(label.polynomial)
        if self.options.tail_policy == 'selective':
            W, coeff = lead
            # This is only a decision to omit OPTIONAL tail work, never a
            # permission to perform a signature-unsafe reduction or discard.
            ordinary_divisor = any(self.unit_reducer_index.divisors(W))
            if not ordinary_divisor:
                ordinary_divisor = any(coeff % c == 0
                    for bucket in self.top_nonunit_index.divisors(W) for c in bucket)
            if ordinary_divisor:
                self.stats.counters['tail_selective_skipped'] += 1
                return label
        self.stats.counters['tail_reduction_calls'] += 1
        if self.options.reduction_kernel in ('heap', 'packed'):
            result = self._reduce_working(label, tail=True)
            assert self.backend.leading(result.polynomial) == lead
            return result
        cutoff = self.backend.order_key(lead[0])
        steps, tail_cap = 0, self.options.tail_step_cap
        while True:
            self._check_budget()
            todo = [m for m in label.polynomial if self.backend.order_key(m) < cutoff]
            if not todo:
                break
            W = max(todo, key=self.backend.order_key)
            while W in label.polynomial:
                self._check_budget()
                if tail_cap and steps >= tail_cap:
                    self.stats.counters['tail_cap_reached'] += 1
                    assert self.backend.leading(label.polynomial) == lead
                    return label
                c = int(label.polynomial[W])
                red = self._find_reducer(label.signature, (W, c), 'mixed',
                                          allow_euclidean=self.options.euclidean)
                if red is None:
                    break
                label = self._apply_reduction(label, red, tail_term=W)
                steps += 1
            cutoff = self.backend.order_key(W)
        assert self.backend.leading(label.polynomial) == lead
        return label

    def _reduce_candidate_tail(self, label: Label, kind: str) -> Label:
        """Dispatch optional tails and attribute their costs to the source type.

        Skipping a G tail returns its original label after the unchanged exact
        top test. It does not reject the candidate, and it does not reduce its
        head. J tails, path recipes and online witnesses are otherwise unchanged.
        """
        if kind not in ('J', 'G'):
            raise ValueError("Candidate kind must be J or G")
        if kind == 'G' and not self.options.g_tail_reduce:
            self.phase = 'G tail skipped'
            self.stats.counters['g_tail_skipped'] += 1
            return label
        keys = ('tail_reduction_calls', 'tail_steps', 'tail_exact_steps',
                'tail_euclidean_steps')
        counts = self.stats.counters
        before = tuple(counts[key] for key in keys)
        self._tail_kind = kind
        self._tail_steps_start = counts['tail_steps']
        self.phase = f'{kind} tail reduction'
        start = perf_counter_ns()
        try:
            return self._reduce_tail(label)
        finally:
            # Finalise also on a budget exception or KeyboardInterrupt. Keep
            # the original total; subtotals must not be counted a second time.
            elapsed = perf_counter_ns() - start
            self.stats.times_ns['tail_reduction'] += elapsed
            self.stats.times_ns[kind.lower() + '_tail_reduction'] += elapsed
            for key, value in zip(keys, before):
                counts[kind.lower() + '_' + key] += counts[key] - value
            self._tail_kind = None

    # ------------------------------------------------------------------
    # Main candidate loop
    # ------------------------------------------------------------------

    def _start_witness(self, candidate) -> None:
        if self.paths is not None:
            if candidate.path_recipe is None:
                raise RuntimeError("Candidate has no generation path")
            self.paths.start(candidate.path_recipe, candidate.kind)
            candidate.path_recipe = None
        if self.witnesses is not None:
            if candidate.representation_recipe is None:
                raise RuntimeError("Candidate has no representation recipe")
            self.witnesses.current = candidate.representation_recipe()
            candidate.representation_recipe = None

    def _process_queue(self) -> None:
        """Process the main candidate queue to exhaustion."""
        last_sig_key = None
        native_selection = (_native_select_candidate
                            if type(self) is GVWInteger
                            and self.options.cover_mode == 'indexed'
                            and not self.options.debug else None)
        while self.queue:
            self.phase = 'queue selection'
            self.current_candidate_kind = None
            self.current_candidate_serial = None
            cover_checked = False
            if native_selection is not None:
                candidate, cover_checked = native_selection(self)
                if candidate is None:
                    continue
            else:
                self._check_budget()
                t = perf_counter_ns()
                candidate = self.queue.pop()
                self.stats.times_ns['queue'] += perf_counter_ns() - t
                self.stats.counters['iterations'] += 1
            packed_signature = getattr(candidate, '_packed_signature', None)
            syzygy_checked_version = None
            if (not cover_checked and candidate.kind == 'J' and packed_signature is not None
                    and self.options.cover_mode == 'indexed' and not self.options.debug):
                # The same positive S-cover as CoverIndex.check, before lazy
                # candidates expand their Signature and leading exponent tuple.
                # A negative answer continues through the full multi-cover test.
                t = perf_counter_ns()
                covered = self.S._has_divisor_packed(*packed_signature)
                self.stats.times_ns['cover'] += perf_counter_ns() - t
                if covered:
                    self.stats.counters['cover_calls'] += 1
                    self.stats.counters['cover_gcd_updates'] += 1
                    self.stats.counters['cover_rejected_j'] += 1
                    self.stats.counters['packed_queue_cover_rejections'] += 1
                    self.queue.settle_j(candidate)
                    continue
                syzygy_checked_version = self.S.version
            self.current_signature = candidate.signature
            self.current_candidate_kind = candidate.kind
            self.current_candidate_serial = getattr(candidate, "token", None)
            key = self.order.key(candidate.signature)
            if last_sig_key is not None and key < last_sig_key:
                raise AssertionError('Queue signatures decreased: incompatible order or bad label')
            last_sig_key = key
            if candidate.kind == 'G':
                # G-candidates never undergo a cover or syzygy-cover test.
                self.phase = 'G top test'
                t = perf_counter_ns()
                try:
                    lead = (candidate.leading_monomial, candidate.leading_coefficient)
                    reducer = self._find_reducer(candidate.signature, lead, 'any')
                finally:
                    self.stats.times_ns['g_top_test'] += perf_counter_ns() - t
                if reducer is not None:
                    # Reject regular, singular, and super top-reducible G-candidates.
                    # Never perform a G top reduction: the retained candidate must
                    # preserve its gcd leading term and maximal lifted signature monomial.
                    self.stats.counters['g_top_rejected'] += 1
                    self.stats.counters['g_discarded'] += 1
                    continue
                t = perf_counter_ns()
                try:
                    r = candidate.label
                    self._start_witness(candidate)
                finally:
                    self.stats.times_ns['materialization'] += perf_counter_ns() - t
            else:
                self.phase = 'J cover check'
                t = perf_counter_ns()
                try:
                    if cover_checked:
                        covered = False  # Checked against the same P and S in selection.
                    elif packed_signature is not None:
                        covered = self.cover.check(candidate.signature, candidate.leading_monomial,
                            packed_monomial=packed_signature[1],
                            unit_syzygy_checked=(syzygy_checked_version == self.S.version))
                    else:
                        covered = self.cover.check(candidate.signature, candidate.leading_monomial)
                finally:
                    self.stats.times_ns['cover'] += perf_counter_ns() - t
                if covered:
                    self.stats.counters['cover_rejected_j'] += 1
                    self.queue.settle_j(candidate)
                    continue
                t = perf_counter_ns()
                try:
                    original = candidate.label
                    self._start_witness(candidate)
                finally:
                    self.stats.times_ns['materialization'] += perf_counter_ns() - t
                t = perf_counter_ns()
                try:
                    self.phase = 'J head reduction'
                    r = self._reduce_j(original)
                finally:
                    self.stats.times_ns['reduction'] += perf_counter_ns() - t
            if not r.polynomial:
                self.stats.counters['zero_reductions'] += 1
                self.stats.counters[candidate.kind.lower() + '_zero_reductions'] += 1
                self._insert_syzygy(r.signature)
                if candidate.kind == 'J':
                    self.queue.settle_j(candidate)
                continue
            r = self._reduce_candidate_tail(r, candidate.kind)
            self.phase = f'{candidate.kind} insertion'
            actual_parents = getattr(candidate, 'generating_parents', None)
            self._current_origin = (InsertionOrigin(candidate.kind, actual_parents)
                                    if actual_parents is not None else None)
            j = self._insert_basis(r)
            self.stats.counters[candidate.kind.lower() + '_inserted'] += 1
            if candidate.kind == 'J':
                # The old J now has a cover, even when singular steps changed
                # the final signature coefficient. Settle BEFORE new emission.
                self.queue.settle_j(candidate)
            self._emit_pairs(j)
        self.current_signature = None
        self.current_candidate_kind = None
        self.current_candidate_serial = None
        self.phase = 'finished'
        self.queue.compact_when_empty()

    def _initialize_and_run(self) -> None:
        for i, f in enumerate(self.inputs):
            if self.witnesses is not None:
                self.witnesses.current = self.witnesses.unit(i)
            sig = Signature(1, self.backend.zero_monomial, i)
            if self.paths is not None:
                self.paths.input(i, sig)
            if f:
                self._current_origin = InsertionOrigin("input", input_index=i)
                self._insert_basis(Label(sig, f))
            else:
                self._insert_syzygy(sig)
        for j in range(len(self.P)):
            self._emit_pairs(j)
        self._process_queue()

    # ------------------------------------------------------------------
    # Public execution
    # ------------------------------------------------------------------

    def run(self) -> Result:
        if self.started:
            raise RuntimeError("A solver instance may be run only once")
        self.started = True
        if self.options.incremental_pot:
            from .incremental import IncrementalPOT
            return IncrementalPOT(self.backend, self.inputs, self.order, self.options).run()
        start = perf_counter_ns()
        cpu_start = process_time_ns()
        self._progress_start = start
        self._progress_next = start + int(self.options.progress_interval * 1e9)
        if self.options.max_seconds is not None:
            self.deadline = start + int(self.options.max_seconds * 1e9)

        status = "completed"
        message = "All candidates have been settled"
        try:
            self._initialize_and_run()
            self._check_accounting()
        except ComputationLimit as exc:
            status = "limit"
            message = str(exc) + "; partial output is NOT certified as a basis"
        except KeyboardInterrupt:
            status = "interrupted"
            message = "Interrupted; partial output is NOT certified as a basis"
        finally:
            self.stats.total_ns = perf_counter_ns() - start
            self.stats.cpu_ns = process_time_ns() - cpu_start

        return Result(
            status, message, self.backend, self.order, self.inputs,
            list(self.P), self.S.all_signatures(), self.stats, self.options,
            insertion_history=list(self.insertion_history),
            generation_paths=(self.paths.snapshot(self.S.all_signatures())
                              if self.paths is not None else None),
            basis_representations=list(self.witnesses.basis) if self.witnesses is not None else None,
            syzygy_representations=([self.witnesses.syzygies[s] for s in self.S.all_signatures()]
                                   if self.witnesses is not None else None),
        )

    def _check_accounting(self) -> None:
        counters = self.stats.counters
        assert counters["j_generated"] == (
            counters["j_popped"]
            + counters["j_dedup_discarded"]
            + counters["j_dedup_replaced"]
            + counters["j_generation_syzygy_rejected"]
            + counters["j_dominance_rejected"]
        )
        assert counters["j_popped"] == counters["cover_rejected_j"] + counters["j_reduction_calls"]
        assert counters["j_reduction_calls"] == counters["j_zero_reductions"] + counters["j_inserted"]
        assert counters["cover_calls"] == counters["j_popped"]
        assert counters["g_generated"] == counters["parent_pairs"]
        assert counters["g_generated"] == (
            counters["g_omitted_comparable"]
            + counters["g_generation_top_rejected"]
            + counters["g_enqueued"]
        )
        assert counters["g_enqueued"] == counters["g_popped"]
        assert counters["g_popped"] == (
            counters["g_top_rejected"]
            + counters["g_inserted"]
            + counters["g_zero_reductions"]
        )
        assert counters["g_discarded"] == (
            counters["g_omitted_comparable"]
            + counters["g_generation_top_rejected"]
            + counters["g_top_rejected"]
        )
        assert counters["g_generated"] == (
            counters["g_discarded"] + counters["g_inserted"] + counters["g_zero_reductions"]
        )
        assert counters["zero_reductions"] == counters["j_zero_reductions"] + counters["g_zero_reductions"]
        for name in ('tail_reduction_calls', 'tail_steps', 'tail_exact_steps', 'tail_euclidean_steps'):
            assert counters[name] == counters['j_' + name] + counters['g_' + name]
        if not self.options.g_tail_reduce:
            assert counters['g_tail_steps'] == 0
            assert counters['g_tail_skipped'] == counters['g_inserted']


def compute(
    polynomials: Iterable[str | int],
    variables: Iterable[str],
    polynomial_order: str = "grevlex",
    module_order: str | None = None,
    positions: Iterable[int] | None = None,
    block_sizes: Iterable[int] | None = None,
    options: Options | None = None,
    *,
    show_basis: bool = True,
    show_forest: bool = False,
    show_u: bool | None = None,
    print_style: str = "ascii",
    print_width: int = 100,
    show_minimal_gb: bool = False,
    show_reduced_gb: bool = False,
    show_minimal_lt: bool = False,
) -> Result:
    """Run the solver, then print P/S and available u-vectors by default.

    show_u=None follows show_basis and certificate availability. Explicit True
    enables representation tracking on a copy of Options and prints vectors
    even when show_basis=False. False hides vectors without disabling tracking.
    Polynomial-basis views are off by default and use detached post-processing.
    Requested output is saved separately in result.polynomial_postprocessing.
    Printing is outside measured computation. GVWInteger.run() stays silent.
    """
    from .printing import print_result, print_polynomial_output, validate_display_options
    from .postprocessing import postprocess_basis, validate_polynomial_output_flags
    from dataclasses import replace
    if options is not None and not isinstance(options, Options):
        raise ValueError("options must be an Options instance or None")
    validate_display_options(show_basis, show_forest, print_style, print_width, show_u=show_u)
    validate_polynomial_output_flags(show_minimal_gb, show_reduced_gb, show_minimal_lt)
    if show_u is True:
        if options is None:
            options = Options(track_representations=True)
        elif not options.track_representations:
            options = replace(options, track_representations=True)
    if options is not None and options.incremental_pot and show_forest:
        raise ValueError("--incremental-pot retires stage history; a global ratio forest is not available")
    if module_order is None:
        module_order = "pot" if options is not None and options.incremental_pot else "gvw-g2"
    start = perf_counter_ns()
    backend = IntegerPolynomialRing(variables, polynomial_order)
    inputs = [backend.parse(polynomial) for polynomial in polynomials]
    order = ModuleOrder(backend, inputs, module_order, positions, block_sizes=block_sizes)
    setup_seconds = (perf_counter_ns() - start) / 1e9
    engine = GVWInteger(backend, inputs, order, options)
    try:
        result = engine.run()
    finally:
        # This engine is private to compute(). Its Result owns the completed
        # records, but queue callbacks and the bound key cache otherwise retain
        # the engine and its large indices until cyclic GC happens to run.
        # Retire them before display/serialization, outside solver timings.
        cleanup_start = perf_counter_ns()
        engine.queue.generation_filter = None
        engine.queue.generation_filter_packed = None
        engine.queue._packed_leading_key.cache_clear()
        engine.queue._packed_leading_key = None
        # Limited runs can still own certificate recipe closures that point
        # back to the engine. Pending candidates are not part of Result.
        engine.queue.active.clear()
        engine.queue.pending_j.clear()
        engine.queue.heap.clear()
        del engine
        cleanup_seconds = (perf_counter_ns() - cleanup_start) / 1e9
    result.runtime_cleanup_seconds = cleanup_seconds
    result.setup_seconds = setup_seconds
    print_result(result, show_basis=show_basis, show_forest=show_forest,
                 style=print_style, width=print_width, show_u=show_u)
    if show_minimal_gb or show_reduced_gb or show_minimal_lt:
        if result.status != "completed":
            print("Polynomial post-processing skipped: source computation is not completed.")
        else:
            post = postprocess_basis(result, reduced=show_reduced_gb)
            result.polynomial_postprocessing = post
            print_polynomial_output(post, show_minimal_gb=show_minimal_gb,
                                    show_reduced_gb=show_reduced_gb,
                                    show_minimal_lt=show_minimal_lt,
                                    style=print_style, width=print_width)
    return result
