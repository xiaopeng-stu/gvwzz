# cython: language_level=3, boundscheck=False, wraparound=False
"""Optional exact parent-row emitters and incremental multiplier antichains.

The ordinary emitter batches descriptors and exact rejection tests for
any number of variables, retaining single- and double-word specializations
through sixteen. Wider descriptors use exact arbitrary-width packed integers.
Coefficients remain arbitrary-precision Python integers.
The incremental POT emitters retain arbitrary-width packed monomials. Every
packed parent coordinate is <=63, so lifted coordinates remain <=126; larger
parents use the original emitter. The independent multiplier antichain accepts
uint32 powers and up to 65535 variables, with an exact Python fallback.
"""
from time import perf_counter_ns
from libc.stdint cimport uint64_t, uint32_t
from libcpp.vector cimport vector
from cpython.bytes cimport PyBytes_AS_STRING
from cpython.ref cimport PyObject
from cpython.object cimport PyObject_Hash
from gvwzz._native_indices cimport PackedSyzygyQueries
from gvwzz.core import _signature_from_arithmetic
from math import gcd

cdef extern from *:
    """
    #include <chrono>
    static inline uint64_t gvw_management_clock_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    """
    uint64_t gvw_management_clock_ns() noexcept

cdef extern from *:
    """
    #include <algorithm>
    #include <cstdint>
    #include <vector>
    #include <utility>
    struct GVWShiftRow {
        uint64_t mask;
        size_t parent;
        unsigned degree;
    };
    static unsigned gvw_popcount(uint64_t x) {
        x -= (x >> 1) & UINT64_C(0x5555555555555555);
        x = (x & UINT64_C(0x3333333333333333)) + ((x >> 2) & UINT64_C(0x3333333333333333));
        x = (x + (x >> 4)) & UINT64_C(0x0f0f0f0f0f0f0f0f);
        return unsigned((x * UINT64_C(0x0101010101010101)) >> 56);
    }
    static std::vector<size_t> gvw_select_shift_rows(std::vector<GVWShiftRow>& rows) {
        std::sort(rows.begin(), rows.end(), [](const GVWShiftRow& a, const GVWShiftRow& b) {
            return a.degree < b.degree || (a.degree == b.degree && a.parent < b.parent);
        });
        std::vector<uint64_t> minimal;
        std::vector<size_t> answer;
        for (const GVWShiftRow& row : rows) {
            bool covered = false;
            for (uint64_t divisor : minimal) {
                if ((row.mask & divisor) == divisor) { covered = true; break; }
            }
            if (!covered) {
                minimal.push_back(row.mask);
                answer.push_back(row.parent);
            }
        }
        std::sort(answer.begin(), answer.end());
        return answer;
    }
    static std::vector<size_t> gvw_minimal_shifts(
            const std::vector<uint64_t>& supports, uint64_t current,
            const std::vector<size_t>& parents) {
        std::vector<GVWShiftRow> rows;
        rows.reserve(parents.size());
        for (size_t i : parents) {
            uint64_t shift = supports[i] & ~current;
            rows.push_back({shift, i, gvw_popcount(shift)});
        }
        return gvw_select_shift_rows(rows);
    }
    """
    vector[size_t] gvw_minimal_shifts(const vector[uint64_t]&, uint64_t,
                                    const vector[size_t]&) except +


cdef extern from *:
    """
    struct GVWPowerTerm {
        size_t variable;
        uint32_t exponent;
    };
    struct GVWPowerShift {
        std::vector<GVWPowerTerm> terms;
        size_t parent;
        uint64_t degree;
        uint64_t support;
    };
    static bool gvw_power_divides(const GVWPowerShift& a, const GVWPowerShift& b) {
        if ((a.support & b.support) != a.support) return false;
        size_t j = 0;
        for (const auto& x : a.terms) {
            while (j < b.terms.size() && b.terms[j].variable < x.variable) ++j;
            if (j == b.terms.size() || b.terms[j].variable != x.variable ||
                    b.terms[j].exponent < x.exponent) return false;
        }
        return true;
    }
    static std::vector<size_t> gvw_minimal_power_shifts(
            const std::vector<std::vector<GVWPowerTerm>>& frozen,
            const std::vector<uint32_t>& current, const std::vector<size_t>& parents,
            std::vector<GVWPowerShift>& retained) {
        std::vector<GVWPowerShift> shifts;
        shifts.reserve(parents.size());
        for (size_t i : parents) {
            GVWPowerShift row;
            row.parent = i;
            row.degree = row.support = 0;
            row.terms.reserve(frozen[i].size());
            for (const auto& term : frozen[i]) {
                if (term.exponent > current[term.variable]) {
                    uint32_t e = term.exponent - current[term.variable];
                    row.terms.push_back({term.variable, e});
                    row.degree += e;
                    if (term.variable < 64) row.support |= uint64_t(1) << term.variable;
                }
            }
            shifts.push_back(std::move(row));
        }
        // A one-variable multiplier can only be dominated by a smaller
        // multiplier in that variable (or the unit). Extract these exact
        // witnesses before sorting; they usually remove most lifted rows.
        const size_t absent = size_t(-1);
        std::vector<size_t> single(current.size(), absent);
        size_t unit = absent;
        for (size_t k = 0; k < shifts.size(); ++k) {
            const auto& row = shifts[k];
            if (row.terms.empty()) {
                if (unit == absent || row.parent < shifts[unit].parent) unit = k;
            } else if (row.terms.size() == 1) {
                size_t v = row.terms[0].variable;
                size_t old = single[v];
                if (old == absent || row.degree < shifts[old].degree ||
                    (row.degree == shifts[old].degree && row.parent < shifts[old].parent)) single[v] = k;
            }
        }
        retained.clear();
        std::vector<size_t> answer;
        if (unit != absent) {
            answer.push_back(shifts[unit].parent);
            retained.push_back(std::move(shifts[unit]));
            return answer;
        }
        std::vector<uint32_t> single_exponent(current.size(), 0);
        for (size_t v = 0; v < single.size(); ++v) {
            if (single[v] != absent) {
                auto& row = shifts[single[v]];
                single_exponent[v] = row.terms[0].exponent;
                answer.push_back(row.parent);
                retained.push_back(std::move(row));
            }
        }
        std::vector<GVWPowerShift> remaining;
        remaining.reserve(shifts.size());
        for (auto& row : shifts) {
            if (row.terms.size() <= 1) continue;
            bool covered = false;
            for (const auto& term : row.terms) {
                uint32_t e = single_exponent[term.variable];
                if (e && e <= term.exponent) { covered = true; break; }
            }
            if (!covered) remaining.push_back(std::move(row));
        }
        std::sort(remaining.begin(), remaining.end(), [](const GVWPowerShift& a, const GVWPowerShift& b) {
            return a.degree < b.degree || (a.degree == b.degree && a.parent < b.parent);
        });
        // Every retained non-singleton is assigned one of its variables.
        // A divisor must use an anchor occurring in the query; selecting a
        // shortest bucket reduces candidate visits without approximation.
        std::vector<std::vector<size_t>> buckets(current.size());
        for (auto& row : remaining) {
            bool covered = false;
            size_t anchor = row.terms[0].variable;
            for (const auto& term : row.terms) {
                size_t v = term.variable;
                if (buckets[v].size() < buckets[anchor].size()) anchor = v;
                for (size_t r : buckets[v]) {
                    if (gvw_power_divides(retained[r], row)) { covered = true; break; }
                }
                if (covered) break;
            }
            if (!covered) {
                buckets[anchor].push_back(retained.size());
                answer.push_back(row.parent);
                retained.push_back(std::move(row));
            }
        }
        std::sort(answer.begin(), answer.end());
        return answer;
    }
    static bool gvw_power_covers_bytes(const std::vector<GVWPowerShift>& divisors,
                                       const unsigned char* coordinates, size_t nvars) {
        uint64_t support = 0;
        for (size_t k = 0; k < std::min(nvars, size_t(64)); ++k)
            if (coordinates[k]) support |= uint64_t(1) << k;
        for (const auto& divisor : divisors) {
            if ((divisor.support & support) != divisor.support) continue;
            bool divides = true;
            for (const auto& term : divisor.terms) {
                if (term.exponent > coordinates[term.variable]) { divides = false; break; }
            }
            if (divides) return true;
        }
        return false;
    }
    """
    cdef cppclass GVWPowerTerm:
        size_t variable
        uint32_t exponent
    cdef cppclass GVWPowerShift:
        pass
    vector[size_t] gvw_minimal_power_shifts(const vector[vector[GVWPowerTerm]]&,
        const vector[uint32_t]&, const vector[size_t]&, vector[GVWPowerShift]&) except +
    bint gvw_power_covers_bytes(const vector[GVWPowerShift]&, const unsigned char*, size_t)


cdef class _FrozenAntichain:
    cdef vector[uint64_t] supports
    cdef vector[uint64_t] cover_shifts
    cdef vector[vector[uint64_t]] covers_by_parent
    cdef vector[vector[GVWPowerTerm]] powers
    cdef vector[GVWPowerShift] power_cover_shifts
    cdef vector[vector[GVWPowerShift]] power_covers_by_parent
    cdef Py_ssize_t cover_parent
    cdef Py_ssize_t nvars
    cdef bint eligible, power_eligible

    def __init__(self, engine):
        cdef Py_ssize_t i, k
        cdef uint64_t support
        cdef GVWPowerTerm term
        cdef vector[GVWPowerTerm] row
        self.nvars = len(engine.backend.names)
        self.cover_parent = -1
        self.eligible = self.nvars <= 64
        # Limits ensure coordinate conversion and degree sums cannot overflow.
        # Unsupported dimensions/exponents retain the exact Python fallback.
        self.power_eligible = self.nvars <= 65535
        if not self.power_eligible:
            self.eligible = False
            return
        for i in range(engine.frozen_count):
            support = 0
            row.clear()
            mu = engine.leading[i][0]
            if len(mu) != self.nvars:
                raise ValueError('Wrong exponent-vector length')
            for k in range(self.nvars):
                e = mu[k]
                if e < 0 or e > 4294967295:
                    self.eligible = self.power_eligible = False
                    return
                if e:
                    term.variable, term.exponent = k, e
                    row.push_back(term)
                    if k < 64:
                        support |= (<uint64_t>1) << k
                    if e != 1:
                        self.eligible = False
            self.supports.push_back(support)
            self.powers.push_back(row)

    cdef object _select(self, leading, const vector[size_t]& parents,
                        Py_ssize_t parent):
        cdef Py_ssize_t k
        cdef size_t i
        cdef uint64_t current = 0
        cdef vector[size_t] chosen
        cdef vector[uint32_t] coordinates
        cdef bint squarefree = self.eligible
        self.cover_parent = -1
        self.cover_shifts.clear()
        self.power_cover_shifts.clear()
        if not self.power_eligible:
            return None
        if len(leading) != self.nvars:
            raise ValueError('Wrong exponent-vector length')
        for k in range(self.nvars):
            e = leading[k]
            if e < 0 or e > 4294967295:
                return None
            coordinates.push_back(e)
            if e and k < 64:
                current |= (<uint64_t>1) << k
            if e > 1:
                squarefree = False
        if squarefree:
            chosen = gvw_minimal_shifts(self.supports, current, parents)
            for i in chosen:
                self.cover_shifts.push_back(self.supports[i] & ~current)
        else:
            chosen = gvw_minimal_power_shifts(self.powers, coordinates, parents,
                                             self.power_cover_shifts)
        self.cover_parent = parent
        return [i for i in chosen]

    def select(self, leading, parents, Py_ssize_t parent=-1):
        cdef size_t i
        cdef vector[size_t] rows
        if not self.power_eligible:
            self.cover_parent = -1
            return None
        for value in parents:
            if value < 0 or value >= self.supports.size():
                raise IndexError('frozen parent index out of range')
            i = value
            rows.push_back(i)
        return self._select(leading, rows, parent)

    def select_mask(self, leading, parent_mask, Py_ssize_t parent=-1):
        cdef size_t i
        cdef vector[size_t] rows
        cdef bytes mask_bytes
        cdef const unsigned char* mask
        if not self.power_eligible:
            self.cover_parent = -1
            return None
        if parent_mask < 0 or parent_mask.bit_length() > self.supports.size():
            raise ValueError('frozen parent bitset out of range')
        mask_bytes = parent_mask.to_bytes((self.supports.size() + 7) // 8, 'little')
        mask = <const unsigned char*>PyBytes_AS_STRING(mask_bytes)
        for i in range(self.supports.size()):
            if mask[i >> 3] & (1 << (i & 7)):
                rows.push_back(i)
        return self._select(leading, rows, parent)

    cdef void commit(self, Py_ssize_t parent):
        if parent < 0 or self.cover_parent != parent:
            return
        if self.covers_by_parent.size() <= <size_t>parent:
            self.covers_by_parent.resize(parent + 1)
            self.power_covers_by_parent.resize(parent + 1)
        self.covers_by_parent[parent] = self.cover_shifts
        self.power_covers_by_parent[parent] = self.power_cover_shifts

    cdef bint covers_shift(self, Py_ssize_t parent, object shift) except -1:
        cdef bytes encoded
        cdef const unsigned char* coordinates
        cdef uint64_t support = 0, divisor
        cdef Py_ssize_t k
        if (parent < 0 or <size_t>parent >= self.covers_by_parent.size()):
            return False
        if (self.covers_by_parent[parent].empty() and
                self.power_covers_by_parent[parent].empty()):
            return False
        encoded = shift.to_bytes(self.nvars, 'little')
        coordinates = <const unsigned char*>PyBytes_AS_STRING(encoded)
        if not self.power_covers_by_parent[parent].empty():
            return gvw_power_covers_bytes(self.power_covers_by_parent[parent],
                                           coordinates, self.nvars)
        for k in range(self.nvars):
            if coordinates[k]:
                support |= (<uint64_t>1) << k
        for divisor in self.covers_by_parent[parent]:
            if (support & divisor) == divisor:
                return True
        return False


def minimal_frozen_parents(engine, Py_ssize_t j, parents):
    """Minimal unit J shifts; exact sparse nonnegative-exponent fast path.

    Equal shifts retain their earliest parent. Proper shift divisibility gives
    a strictly smaller signature, whose eventual cover also covers the larger
    J. The Python fallback handles coordinates above 2**32-1 or more than
    65535 variables; no truncation or approximate comparison is used.
    """
    index = getattr(engine, '_native_antichain', None)
    if index is None:
        index = _FrozenAntichain(engine)
        engine._native_antichain = index
    return index.select(engine.leading[j][0], parents, j)


def minimal_frozen_mask(engine, Py_ssize_t j, parent_mask):
    """The same antichain directly from a Python-int bitset of frozen rows."""
    index = getattr(engine, '_native_antichain', None)
    if index is None:
        index = _FrozenAntichain(engine)
        engine._native_antichain = index
    return index.select_mask(engine.leading[j][0], parent_mask, j)


def commit_frozen_parents(engine, Py_ssize_t j):
    """Publish the multiplier cover witnesses after the frozen row succeeds."""
    index = getattr(engine, '_native_antichain', None)
    if isinstance(index, _FrozenAntichain):
        (<_FrozenAntichain>index).commit(j)

cdef void _flush(object counts, Py_ssize_t pairs,
                 Py_ssize_t accepted, Py_ssize_t avoided):
    if pairs:
        counts['parent_pairs'] += pairs
        counts['j_generated'] += pairs
        counts['g_generated'] += pairs
        counts['g_omitted_comparable'] += pairs
        counts['g_discarded'] += pairs
        counts['incremental_principal_skipped'] += pairs
        counts['native_parent_pairs'] += pairs
    if accepted:
        counts['native_packed_j_enqueued'] += accepted
    if avoided:
        counts['candidate_signature_objects_avoided'] += avoided


def emit_frozen_pairs(engine, Py_ssize_t j, parents):
    """Emit ascending frozen row IDs, preserving the existing exact J filter."""
    cdef Py_ssize_t i, pending = 0, accepted = 0, avoided = 0
    cdef object packed_p, packed_mp, packed_mq, packed_sq, high, select_a
    cdef object packed_W, upper_exp, m, entry, factory
    cdef bint inserted
    counts, times = engine.stats.counters, engine.stats.times_ns
    packed = engine.packed_parents
    packed_mq, packed_sq = packed[j]
    q = engine.P[j]
    cq = engine.leading[j][1]
    # Both leading coefficients are units. ell=1, and 1/cq=cq.
    coefficient = cq * q.signature.coefficient
    index = q.signature.index
    f = q.polynomial
    packing, backend, queue = engine.pair_packing, engine.backend, engine.queue
    recipe_context = engine._recipe_context
    guard = packing.guard
    accept = queue._accept_j_packed
    insert = queue._insert_j_packed
    materialize = engine._materialize_polynomial
    paths, witnesses = engine.paths, engine.witnesses
    start = perf_counter_ns()
    generation_before = times['generation']
    queue_before = times['queue']
    filter_before = times['j_generation_test']
    try:
        for i in parents:
            packed_p = packed[i]
            if packed_p is None:
                _flush(counts, pending, accepted, avoided)
                pending = accepted = avoided = 0
                engine._emit_pair(i, j)
                continue
            pending += 1
            packed_mp = packed_p[0]
            high = ((packed_mp | guard) - packed_mq) & guard
            select_a = high - (high >> 7)
            packed_W = (packed_mp & select_a) | (packed_mq & ~select_a)
            upper_exp = packed_W - packed_mq + packed_sq
            inserted = accept(coefficient, upper_exp, index, packed_W)
            if inserted:
                accepted += 1
                # LCM>=LM(q) in every byte, so subtraction cannot borrow.
                factory = JPolynomialRecipe(recipe_context, f, cq, packed_W - packed_mq)
                entry = insert(coefficient, upper_exp, index, packed_W, factory)
                entry.generating_parents = (i, j)
                if paths is not None:
                    m = packing.unpack(packed_W - packed_mq)
                    entry.path_recipe = ((j, cq, m),)
                if witnesses is not None:
                    m = packing.unpack(packed_W - packed_mq)
                    entry.representation_recipe = (
                        lambda j=j, c=cq, m=m, witnesses=witnesses:
                        witnesses.scale(witnesses.basis[j], c, m))
            else:
                avoided += 1
            if pending >= 128:
                _flush(counts, pending, accepted, avoided)
                pending = accepted = avoided = 0
                engine._check_budget()
    finally:
        _flush(counts, pending, accepted, avoided)
        # Native descriptors and queue dispatch are fused. Keep S-filter time
        # exclusive, and avoid recounting any original-emitter fallback time.
        times['generation'] += (perf_counter_ns() - start
            - (times['generation'] - generation_before)
            - (times['queue'] - queue_before)
            - (times['j_generation_test'] - filter_before))
    engine._check_budget()


cdef void _flush_unit(object counts, Py_ssize_t pairs, Py_ssize_t js,
                      Py_ssize_t detected, Py_ssize_t unknown,
                      Py_ssize_t accepted, Py_ssize_t avoided, Py_ssize_t dominated,
                      bint skip_principal):
    if pairs:
        counts['parent_pairs'] += pairs
        counts['g_generated'] += pairs
        counts['g_omitted_comparable'] += pairs
        counts['g_discarded'] += pairs
        counts['native_parent_pairs'] += pairs
        counts['native_current_parent_pairs'] += pairs
        if skip_principal:
            counts['incremental_principal_skipped'] += pairs
            counts['incremental_current_principal_skipped'] += pairs
    if js:
        counts['j_generated'] += js
    if detected:
        counts['principal_syzygies_detected'] += detected
    if unknown:
        counts['principal_syzygies_unknown'] += unknown
    if accepted:
        counts['native_packed_j_enqueued'] += accepted
    if avoided:
        counts['candidate_signature_objects_avoided'] += avoided
    if dominated:
        counts['j_dominance_rejected'] += dominated
        counts['j_current_dominance_rejected'] += dominated


def emit_unit_current_pairs(engine, Py_ssize_t j, Py_ssize_t first):
    """Exact unit-LC current/current rows; all unsupported pairs use Python.

    Signature comparisons and principal-seed completion retain their original
    order. A cached multiplier from the upper parent's successfully emitted
    frozen row can prove a future exact cover before J dedup/S filtering.
    Certificate modes and candidate-count budgets retain their original
    per-pair path; packed coordinates remain Python integers.
    """
    cdef Py_ssize_t i, pending = 0, js = 0, detected = 0, unknown = 0
    cdef Py_ssize_t accepted = 0, avoided = 0, parent
    cdef Py_ssize_t dominated = 0
    cdef bint skip_principal = bool(getattr(engine, '_principal_pairs_redundant', False))
    cdef _FrozenAntichain frozen_covers = None
    cdef int cmp
    cdef object cp, cq, p, q, sp, sq, pp, mq, ssq, mp, ssp
    cdef object ki, kj, high, select_a, W, shift, upper, coefficient, c
    cdef object syz, syz_c, entry, factory
    leading, packed = engine.leading, engine.packed_parents
    cq = leading[j][1]
    if (abs(cq) != 1 or packed[j] is None or engine.paths is not None or
            engine.witnesses is not None or engine.options.max_candidates is not None):
        for i in range(first, j):
            engine._emit_pair(i, j)
        return
    counts, times = engine.stats.counters, engine.stats.times_ns
    labels, ratios = engine.P, engine.ratio_keys
    q, kj = labels[j], ratios[j]
    sq = q.signature
    mq, ssq = packed[j]
    packing, backend, queue = engine.pair_packing, engine.backend, engine.queue
    recipe_context = engine._recipe_context
    guard = packing.guard
    accept, insert = queue._accept_j_packed, queue._insert_j_packed
    insert_syzygy = engine._insert_syzygy
    materialize = engine._materialize_polynomial
    frozen = getattr(engine, 'frozen_count', 0)
    cover_index = getattr(engine, '_native_antichain', None)
    if isinstance(cover_index, _FrozenAntichain):
        frozen_covers = cover_index
    start = perf_counter_ns()
    generation_before, queue_before = times['generation'], times['queue']
    filter_before, syzygy_before = times['j_generation_test'], times['syzygy']
    try:
        for i in range(first, j):
            cp, pp = leading[i][1], packed[i]
            if abs(cp) != 1 or pp is None or i < frozen:
                _flush_unit(counts, pending, js, detected, unknown, accepted, avoided, dominated, skip_principal)
                pending = js = detected = unknown = accepted = avoided = dominated = 0
                engine._emit_pair(i, j)
                continue
            pending += 1
            p, ki = labels[i], ratios[i]
            sp = p.signature
            mp, ssp = pp
            cmp = 1 if ki > kj else (-1 if ki < kj else 0)
            if cmp:
                js += 1
            if not skip_principal:
                # Exactly the original principal signature, before J admission.
                if cmp > 0:
                    syz = (cq * sp.coefficient, mq + ssp, sp.index)
                elif cmp < 0:
                    syz = (-cp * sq.coefficient, mp + ssq, sq.index)
                else:
                    syz_c = cq * sp.coefficient - cp * sq.coefficient
                    syz = (syz_c, mq + ssp, sp.index) if syz_c else None
                if syz is not None:
                    detected += 1
                    insert_syzygy(syz, check_budget=False, monomial_packed=True)
                else:
                    unknown += 1
            if cmp:
                high = ((mp | guard) - mq) & guard
                select_a = high - (high >> 7)
                W = (mp & select_a) | (mq & ~select_a)
                if cmp > 0:
                    parent, c, f = i, cp, p.polynomial
                    shift = W - mp
                    upper, index = shift + ssp, sp.index
                    coefficient = cp * sp.coefficient
                else:
                    parent, c, f = j, cq, q.polynomial
                    shift = W - mq
                    upper, index = shift + ssq, sq.index
                    coefficient = cq * sq.coefficient
                # The same upper parent and unit coefficient as its earlier
                # frozen batch. Its retained smaller/equal multiplier acquires
                # an exact cover before this J can be reduced. The principal
                # syzygy is processed unless the incremental stage already
                # proves all current principal seeds redundant.
                if frozen_covers is not None and frozen_covers.covers_shift(parent, shift):
                    dominated += 1
                    avoided += 1
                elif accept(coefficient, upper, index, W):
                    accepted += 1
                    factory = JPolynomialRecipe(recipe_context, f, c, shift)
                    entry = insert(coefficient, upper, index, W, factory)
                    entry.generating_parents = (i, j)
                else:
                    avoided += 1
            if pending >= 128:
                _flush_unit(counts, pending, js, detected, unknown, accepted, avoided, dominated, skip_principal)
                pending = js = detected = unknown = accepted = avoided = dominated = 0
                engine._check_budget()
    finally:
        _flush_unit(counts, pending, js, detected, unknown, accepted, avoided, dominated, skip_principal)
        times['generation'] += (perf_counter_ns() - start
            - (times['generation'] - generation_before)
            - (times['queue'] - queue_before)
            - (times['j_generation_test'] - filter_before)
            - (times['syzygy'] - syzygy_before))
    engine._check_budget()


include "_top_query.pxi"


cdef extern from *:
    """
    struct GVWPairParent {
        uint64_t leading, signature, leading_high, signature_high, small_coefficient;
        Py_ssize_t component;
        PyObject *coefficient, *signature_coefficient, *polynomial, *ratio;
        PyObject *packed_leading, *packed_signature;
        bool packed, small, positive;
    };
    static inline uint64_t gvw_pair_gcd(uint64_t a, uint64_t b) {
        while (b) { uint64_t t = a % b; a = b; b = t; }
        return a;
    }
    static inline uint64_t gvw_pair_hash(uint64_t value) {
        value ^= value >> 30;
        value *= UINT64_C(0xbf58476d1ce4e5b9);
        value ^= value >> 27;
        value *= UINT64_C(0x94d049bb133111eb);
        return value ^ (value >> 31);
    }
    """
    cdef cppclass GVWPairParent:
        uint64_t leading, signature, leading_high, signature_high, small_coefficient
        Py_ssize_t component
        PyObject *coefficient
        PyObject *signature_coefficient
        PyObject *polynomial
        PyObject *ratio
        PyObject *packed_leading
        PyObject *packed_signature
        bint packed, small, positive
    uint64_t gvw_pair_gcd(uint64_t, uint64_t) noexcept
    uint64_t gvw_pair_hash(uint64_t) noexcept


cdef class _ParentRows:
    """Borrow immutable label fields while their owning lists remain alive."""
    cdef vector[GVWPairParent] rows
    cdef object labels, leading, ratios, packed_parents

    def __init__(self, engine):
        self.labels, self.leading, self.ratios = engine.P, engine.leading, engine.ratio_keys
        self.packed_parents = engine.packed_parents
        self.refresh(engine)

    cdef void refresh(self, object engine) except *:
        cdef Py_ssize_t i
        cdef object label, signature, packed, coefficient, magnitude
        cdef GVWPairParent row
        for i in range(self.rows.size(), len(self.labels)):
            label = self.labels[i]
            signature = label.signature
            packed = engine.packed_parents[i]
            row.packed = packed is not None
            if row.packed:
                row.leading = packed[0] & 0xffffffffffffffff
                row.signature = packed[1] & 0xffffffffffffffff
                row.leading_high = (packed[0] >> 64) & 0xffffffffffffffff
                row.signature_high = (packed[1] >> 64) & 0xffffffffffffffff
                row.packed_leading, row.packed_signature = <PyObject*>packed[0], <PyObject*>packed[1]
            else:
                row.leading = row.signature = row.leading_high = row.signature_high = 0
                row.packed_leading = row.packed_signature = NULL
            coefficient = self.leading[i][1]
            magnitude = abs(coefficient)
            row.small = magnitude <= 0xffffffffffffffff
            row.small_coefficient = <uint64_t>magnitude if row.small else 0
            row.positive = coefficient > 0
            row.component = signature.index
            row.coefficient = <PyObject*>coefficient
            row.signature_coefficient = <PyObject*>signature.coefficient
            row.polynomial = <PyObject*>label.polynomial
            row.ratio = <PyObject*>self.ratios[i]
            self.rows.push_back(row)


cdef class ParentDedupMembership:
    """An append-only Bloom superset of every admitted packed J signature.

    The coefficient uses Python's exact integer hash, so equal integers of any
    size always enter the same bits. Hash collisions only create false positives:
    every positive answer still performs the original exact dictionary lookup.
    Deletions need no update. Queue hooks include Python fallback admissions
    and new settlement bounds, so a negative cannot bypass an existing key.
    """
    cdef vector[uint64_t] bits

    def __init__(self, queue):
        self.bits.resize(65536, 0)
        for key in queue.pending_j:
            self.add_signature(key[0], key[1], key[2])
        for key in queue.settled_j_bounds:
            self.add_signature(key[0], key[1], key[2])

    cpdef add_signature(self, object coefficient, object monomial, Py_ssize_t index):
        cdef uint64_t packed, high, h1, h2, h3
        if not isinstance(monomial, int) or monomial < 0:
            return
        if monomial.bit_length() <= 128:
            packed, high = monomial & 0xffffffffffffffff, monomial >> 64
            h1 = gvw_pair_hash(packed ^ gvw_pair_hash(high) ^ gvw_pair_hash(<uint64_t>index)
                ^ gvw_pair_hash(<uint64_t>PyObject_Hash(coefficient)))
        else:
            h1 = gvw_pair_hash(<uint64_t>PyObject_Hash(monomial)
                ^ gvw_pair_hash(<uint64_t>index)
                ^ gvw_pair_hash(<uint64_t>PyObject_Hash(coefficient)))
        h2 = gvw_pair_hash(h1 + <uint64_t>0x9e3779b97f4a7c15)
        h3 = h1 + h2
        self.bits[(h1 >> 6) & 65535] |= <uint64_t>1 << (h1 & 63)
        self.bits[(h2 >> 6) & 65535] |= <uint64_t>1 << (h2 & 63)
        self.bits[(h3 >> 6) & 65535] |= <uint64_t>1 << (h3 & 63)

    cdef bint may_contain(self, object coefficient, uint64_t packed,
                         Py_ssize_t index) except -1:
        return self.may_contain128(coefficient, packed, 0, index)

    cdef bint may_contain128(self, object coefficient, uint64_t packed, uint64_t high,
                            Py_ssize_t index) except -1:
        cdef uint64_t h1 = gvw_pair_hash(packed ^ gvw_pair_hash(high) ^ gvw_pair_hash(<uint64_t>index)
            ^ gvw_pair_hash(<uint64_t>PyObject_Hash(coefficient)))
        cdef uint64_t h2 = gvw_pair_hash(h1 + <uint64_t>0x9e3779b97f4a7c15)
        cdef uint64_t h3 = h1 + h2
        return ((self.bits[(h1 >> 6) & 65535] & (<uint64_t>1 << (h1 & 63))) != 0
            and (self.bits[(h2 >> 6) & 65535] & (<uint64_t>1 << (h2 & 63))) != 0
            and (self.bits[(h3 >> 6) & 65535] & (<uint64_t>1 << (h3 & 63))) != 0)

    cdef bint may_contain_packed(self, object coefficient, object monomial,
                                Py_ssize_t index) except -1:
        cdef uint64_t h1, h2, h3
        if monomial.bit_length() <= 128:
            return self.may_contain128(coefficient, monomial & 0xffffffffffffffff,
                                       monomial >> 64, index)
        h1 = gvw_pair_hash(<uint64_t>PyObject_Hash(monomial)
            ^ gvw_pair_hash(<uint64_t>index)
            ^ gvw_pair_hash(<uint64_t>PyObject_Hash(coefficient)))
        h2 = gvw_pair_hash(h1 + <uint64_t>0x9e3779b97f4a7c15)
        h3 = h1 + h2
        return ((self.bits[(h1 >> 6) & 65535] & (<uint64_t>1 << (h1 & 63))) != 0
            and (self.bits[(h2 >> 6) & 65535] & (<uint64_t>1 << (h2 & 63))) != 0
            and (self.bits[(h3 >> 6) & 65535] & (<uint64_t>1 << (h3 & 63))) != 0)

    def might_contain(self, coefficient, monomial, Py_ssize_t index):
        """Conservative checked entry point for diagnostics and tests."""
        if not isinstance(monomial, int) or monomial < 0:
            return True
        return bool(self.may_contain_packed(coefficient, monomial, index))


include "_queue_admission.pxi"


include "_candidate_recipes.pxi"

# Ordinary all-inputs-at-once parent rows. The native loop keeps the original
# ordering of G top tests, principal completion, exact-J dedup and J S tests.
# Only bounded monomial coordinates use machine words; coefficients remain
# arbitrary-precision Python integers throughout.
cdef tuple _parent_counter_names = (
    'parent_pairs', 'j_generated', 'g_generated', 'g_omitted_comparable',
    'g_discarded', 'g_generation_tests', 'g_generation_top_rejected',
    'principal_syzygies_detected', 'principal_syzygies_unknown',
    'syzygy_insert_calls', 'syzygy_seed_rejected',
    'principal_syzygy_objects_avoided', 'j_generation_tests',
    'j_generation_syzygy_rejected', 'candidate_signature_objects_avoided',
    'candidate_signature_fast_builds', 'bezout_adjustments',
    'reducer_queries', 'native_parent_pairs')

cdef void _flush_parent_counts(object counts, uint64_t* pending):
    cdef Py_ssize_t k
    for k in range(19):
        if pending[k]:
            counts[_parent_counter_names[k]] += pending[k]
            pending[k] = 0


cdef void _flush_parent_times(object times, uint64_t* elapsed):
    times['generation'] += elapsed[0]
    times['g_generation_test'] += elapsed[1]
    times['syzygy_check'] += elapsed[2]
    times['syzygy_complete'] += elapsed[3]
    times['syzygy'] += elapsed[2] + elapsed[3]
    times['j_generation_test'] += elapsed[4]
    times['queue'] += elapsed[5]
    cdef Py_ssize_t k
    for k in range(6):
        elapsed[k] = 0


cdef extern from *:
    """
    // Each exponent occupies a byte and is <=126 after lifting. Operations
    // below are lane-safe: subtraction is used only after componentwise >=,
    // addition cannot carry, and the LCM comparison reserves each high bit.
    // Two uint64_t words also compile on MSVC without __int128.
    struct GVWPairWord {
        uint64_t low, high;
        GVWPairWord(uint64_t l=0, uint64_t h=0) : low(l), high(h) {}
        GVWPairWord operator+(const GVWPairWord& b) const {
            return {low + b.low, high + b.high};
        }
        GVWPairWord operator-(const GVWPairWord& b) const {
            return {low - b.low, high - b.high};
        }
    };
    static inline GVWPairWord gvw_pair_lcm(const GVWPairWord& a, const GVWPairWord& b) {
        const uint64_t guard = UINT64_C(0x8080808080808080);
        uint64_t lo = ((a.low | guard) - b.low) & guard;
        uint64_t hi = ((a.high | guard) - b.high) & guard;
        lo -= lo >> 7; hi -= hi >> 7;
        return {(a.low & lo) | (b.low & ~lo), (a.high & hi) | (b.high & ~hi)};
    }
    """
    cdef cppclass GVWPairWord:
        GVWPairWord() noexcept
        GVWPairWord(uint64_t, uint64_t) noexcept
        uint64_t low, high
        GVWPairWord operator+(const GVWPairWord&) noexcept
        GVWPairWord operator-(const GVWPairWord&) noexcept
    GVWPairWord gvw_pair_lcm(const GVWPairWord&, const GVWPairWord&) noexcept


cdef inline object _pair_word_python(GVWPairWord word):
    if word.high:
        return (<object>word.high << 64) | <object>word.low
    return word.low


def emit_parent_pairs(engine, Py_ssize_t j):
    """Dispatch native rows for any dimension, with exact exponent fallback."""
    counts = engine.stats.counters
    options = engine.options
    nvars = len(engine.backend.names)
    if engine.packed_parents[j] is None:
        counts['parent_fallback_exponent_rows'] += 1
        return False
    if engine.paths is not None or engine.witnesses is not None:
        counts['parent_fallback_certificate_rows'] += 1
        return False
    if options.max_candidates is not None:
        counts['parent_fallback_budget_rows'] += 1
        return False
    if type(engine).__name__ != 'GVWInteger':
        counts['parent_fallback_engine_rows'] += 1
        return False
    if nvars <= 8:
        return _emit_parent_pairs64(engine, j)
    if nvars <= 16:
        return _emit_parent_pairs128(engine, j)
    return _emit_parent_pairs_dynamic(engine, j)


cdef _emit_parent_pairs64(engine, Py_ssize_t j):
    """Emit one ordinary parent row in order, with exact wide-integer fallback.

    Returning False asks the Python engine to emit the entire row. Per-pair
    fallback is used only for unbounded exponents, so later bounded parents
    can still benefit. Certificate modes and strict candidate budgets retain
    the original callbacks and the original per-pair checks.
    """
    cdef Py_ssize_t i, k, index, syz_index, since_check = 0
    cdef Py_ssize_t parent = 0
    cdef uint64_t mp, mq, ssp, ssq, W, upper, guard, high, select_a, syz_exp
    cdef uint64_t shift = 0
    cdef uint64_t d_small = 1
    cdef uint64_t scalar_small = 0
    cdef uint64_t membership_queries = 0, membership_positives = 0
    cdef uint64_t tick, after, queue_tick, filter_elapsed
    cdef uint64_t pending[19]
    cdef uint64_t elapsed[6]
    cdef int cmp
    cdef bint comparable, rejected, inserted, adjusted, have_syz
    cdef bint deduplicate, generation_filter, g_precheck
    cdef object cp, cq, scq, ki, kj, pp
    cdef object coefficient, j_coefficient, j_scalar, syz_coefficient
    cdef object d, a, b, up, uq, sig, factory, gdata, entry, dkey
    cdef object unpacked_W = None
    cdef PackedSyzygyQueries queries
    cdef NativeTopQuery top_queries
    cdef _ParentRows parent_rows
    cdef GVWPairParent *prow
    cdef GVWPairParent *qrow
    cdef ParentDedupMembership membership = None
    cdef NativeQueueAdmission admission
    options = engine.options
    for k in range(19):
        pending[k] = 0
    for k in range(6):
        elapsed[k] = 0
    counts, times = engine.stats.counters, engine.stats.times_ns
    counts['native_parent_rows'] += 1
    backend, packing, queue = engine.backend, engine.pair_packing, engine.queue
    recipe_context = engine._recipe_context
    store = engine.S
    queries = getattr(engine, '_native_syzygy_queries', None)
    if queries is None:
        queries = PackedSyzygyQueries(store)
        engine._native_syzygy_queries = queries
    queries.refresh()
    top_queries = getattr(engine, '_native_top_query', None)
    if top_queries is None:
        top_queries = NativeTopQuery(engine)
        engine._native_top_query = top_queries
    else:
        top_queries.refresh(engine)
    parent_rows = getattr(engine, '_native_parent_rows', None)
    if parent_rows is None:
        parent_rows = _ParentRows(engine)
        engine._native_parent_rows = parent_rows
    else:
        parent_rows.refresh(engine)
    qrow = &parent_rows.rows[j]
    cq, scq, kj = <object>qrow.coefficient, <object>qrow.signature_coefficient, <object>qrow.ratio
    mq, ssq = qrow.leading, qrow.signature
    guard = packing.guard
    admission = NativeQueueAdmission(queue)
    materialize = engine._materialize_polynomial
    complete = store._complete_uncovered
    check_budget = engine._check_budget
    # A packed oracle compares the existing upper parent's formal ratio.
    # No signature/vector construction is required for a rejected G.
    top = engine._top_reducer_index
    deduplicate = queue.deduplicate
    if deduplicate:
        membership = queue._native_j_membership
        if membership is None:
            membership = ParentDedupMembership(queue)
            queue._native_j_membership = membership
    generation_filter = options.j_generation_filter
    g_precheck = options.g_precheck
    engine.phase = 'candidate generation'
    try:
        for i in range(j):
            prow = &parent_rows.rows[i]
            if not prow.packed:
                counts['native_parent_fallback_pairs'] += 1
                _flush_parent_counts(counts, pending)
                _flush_parent_times(times, elapsed)
                queries.flush()
                top_queries.flush()
                engine._emit_pair(i, j)
                queries.refresh()
                continue
            tick = gvw_management_clock_ns()
            cp = <object>prow.coefficient
            mp, ssp = prow.leading, prow.signature
            if top_queries.enabled:
                cmp = top_queries.compare_parents(i, j)
            else:
                ki = <object>prow.ratio
                cmp = 1 if ki > kj else (-1 if ki < kj else 0)
            pending[0] += 1
            pending[2] += 1
            pending[18] += 1
            if cmp > 0:
                syz_coefficient = cq * <object>prow.signature_coefficient
                syz_exp, syz_index = mq + ssp, prow.component
                have_syz = True
            elif cmp < 0:
                syz_coefficient = -(cp * scq)
                syz_exp, syz_index = mp + ssq, qrow.component
                have_syz = True
            else:
                syz_coefficient = cq * <object>prow.signature_coefficient - cp * scq
                syz_exp, syz_index = mq + ssp, prow.component
                have_syz = bool(syz_coefficient)
            high = ((mp | guard) - mq) & guard
            select_a = high - (high >> 7)
            W = (mp & select_a) | (mq & ~select_a)
            if prow.small and qrow.small:
                if prow.small_coefficient % qrow.small_coefficient == 0:
                    d_small = qrow.small_coefficient
                    comparable = True
                elif qrow.small_coefficient % prow.small_coefficient == 0:
                    d_small = prow.small_coefficient
                    comparable = True
                else:
                    d_small = gvw_pair_gcd(prow.small_coefficient, qrow.small_coefficient)
                    comparable = False
                d = d_small
            elif cp == cq or cp == -cq:
                d, comparable = abs(cp), True
            elif cp % cq == 0:
                d, comparable = abs(cq), True
            elif cq % cp == 0:
                d, comparable = abs(cp), True
            else:
                d, comparable = gcd(cp, cq), False
            if cmp >= 0:
                upper, index = W - mp + ssp, prow.component
            else:
                upper, index = W - mq + ssq, qrow.component
            if cmp:
                pending[1] += 1
                if cmp > 0:
                    parent, shift = i, W - mp
                    if prow.small and qrow.small:
                        scalar_small = qrow.small_coefficient // d_small
                        j_scalar = scalar_small
                    else:
                        j_scalar = abs(cq) // d
                    if not prow.positive:
                        j_scalar = -j_scalar
                    j_coefficient = j_scalar * <object>prow.signature_coefficient
                else:
                    parent, shift = j, W - mq
                    if prow.small and qrow.small:
                        scalar_small = prow.small_coefficient // d_small
                        j_scalar = scalar_small
                    else:
                        j_scalar = abs(cp) // d
                    if not qrow.positive:
                        j_scalar = -j_scalar
                    j_coefficient = j_scalar * scq
            gdata = None
            if comparable:
                pending[3] += 1
                pending[4] += 1
                elapsed[0] += gvw_management_clock_ns() - tick
            else:
                rejected = False
                unpacked_W = None
                elapsed[0] += gvw_management_clock_ns() - tick
                if g_precheck:
                    tick = gvw_management_clock_ns()
                    pending[5] += 1
                    pending[17] += 1
                    if top_queries.enabled:
                        rejected = top_queries.query64(W, d, i if cmp >= 0 else j)
                    else:
                        unpacked_W = packing.unpack(W)
                        rejected = top(packing.unpack(upper), index, unpacked_W, d) is not None
                    elapsed[1] += gvw_management_clock_ns() - tick
                tick = gvw_management_clock_ns()
                if rejected:
                    pending[4] += 1
                    pending[6] += 1
                else:
                    if unpacked_W is None:
                        unpacked_W = packing.unpack(W)
                    up, uq = packing.unpack(W - mp), packing.unpack(W - mq)
                    a, b, _ = backend.bezout(cp, cq)
                    adjusted = False
                    if cmp > 0:
                        if not a:
                            a += cq // d
                            b -= cp // d
                            adjusted = True
                        coefficient = a * <object>prow.signature_coefficient
                    elif cmp < 0:
                        if not b:
                            a += cq // d
                            b -= cp // d
                            adjusted = True
                        coefficient = b * scq
                    else:
                        coefficient = a * <object>prow.signature_coefficient + b * scq
                        if not coefficient:
                            a += cq // d
                            b -= cp // d
                            adjusted = True
                            coefficient = a * <object>prow.signature_coefficient + b * scq
                    sig = _signature_from_arithmetic(coefficient, packing.unpack(upper), index)
                    pending[15] += 1
                    pending[16] += adjusted
                    factory = GPolynomialRecipe(recipe_context,
                        <object>prow.polynomial, a, up, <object>qrow.polynomial, b, uq)
                    gdata = sig, d, factory
                elapsed[0] += gvw_management_clock_ns() - tick

            tick = gvw_management_clock_ns()
            if have_syz:
                pending[7] += 1
                pending[9] += 1
                if queries.query64(syz_coefficient, syz_exp, syz_index):
                    pending[10] += 1
                    pending[11] += 1
                    elapsed[2] += gvw_management_clock_ns() - tick
                else:
                    after = gvw_management_clock_ns()
                    elapsed[2] += after - tick
                    engine.phase = 'syzygy completion'
                    complete(_signature_from_arithmetic(
                        syz_coefficient, packing.unpack(syz_exp), syz_index))
                    queries.refresh(syz_index)
                    elapsed[3] += gvw_management_clock_ns() - after
                    engine.phase = 'candidate generation'
            else:
                pending[8] += 1
                elapsed[0] += gvw_management_clock_ns() - tick

            queue_tick = gvw_management_clock_ns()
            filter_elapsed = 0
            if cmp:
                inserted = True
                if deduplicate:
                    membership_queries += 1
                    if membership.may_contain(j_coefficient, upper, index):
                        membership_positives += 1
                        dkey = (j_coefficient, upper, index)
                        inserted = admission.accept_bound(dkey, W)
                if inserted and generation_filter:
                    tick = gvw_management_clock_ns()
                    pending[12] += 1
                    if queries.query64(j_coefficient, upper, index):
                        pending[13] += 1
                        inserted = False
                    filter_elapsed = gvw_management_clock_ns() - tick
                    elapsed[4] += filter_elapsed
                if inserted:
                    factory = JPolynomialRecipe(recipe_context,
                        <object>parent_rows.rows[parent].polynomial, j_scalar, shift)
                    entry = admission.insert(j_coefficient, upper, index, W, factory)
                    entry.generating_parents = (i, j)
                else:
                    pending[14] += 1
            if gdata is not None:
                sig, d, factory = gdata
                queue.push_g_lazy(sig, unpacked_W, d, factory)
                queue.active[queue.serial].generating_parents = (i, j)
            elapsed[5] += gvw_management_clock_ns() - queue_tick - filter_elapsed
            since_check += 1
            if since_check >= 128:
                _flush_parent_counts(counts, pending)
                _flush_parent_times(times, elapsed)
                queries.flush()
                top_queries.flush()
                check_budget()
                since_check = 0
    finally:
        _flush_parent_counts(counts, pending)
        _flush_parent_times(times, elapsed)
        queries.flush()
        top_queries.flush()
        counts['native_j_membership_queries'] += membership_queries
        counts['native_j_membership_positives'] += membership_positives
    check_budget()
    return True


cdef _emit_parent_pairs128(engine, Py_ssize_t j):
    """Emit one ordinary parent row in order, with exact wide-integer fallback.

    Returning False asks the Python engine to emit the entire row. Per-pair
    fallback is used only for unbounded exponents, so later bounded parents
    can still benefit. Certificate modes and strict candidate budgets retain
    the original callbacks and the original per-pair checks.
    """
    cdef Py_ssize_t i, k, index, syz_index, since_check = 0
    cdef Py_ssize_t parent = 0
    cdef GVWPairWord mp, mq, ssp, ssq, W, upper, syz_exp, shift
    cdef uint64_t d_small = 1
    cdef uint64_t scalar_small = 0
    cdef uint64_t membership_queries = 0, membership_positives = 0
    cdef uint64_t tick, after, queue_tick, filter_elapsed
    cdef uint64_t pending[19]
    cdef uint64_t elapsed[6]
    cdef int cmp
    cdef bint comparable, rejected, inserted, adjusted, have_syz
    cdef bint deduplicate, generation_filter, g_precheck
    cdef object cp, cq, scq, ki, kj, pp
    cdef object coefficient, j_coefficient, j_scalar, syz_coefficient
    cdef object d, a, b, up, uq, sig, factory, gdata, entry, dkey
    cdef object unpacked_W = None
    cdef object packed_upper, packed_W
    cdef uint64_t wide_pairs = 0
    cdef PackedSyzygyQueries queries
    cdef NativeTopQuery top_queries
    cdef _ParentRows parent_rows
    cdef GVWPairParent *prow
    cdef GVWPairParent *qrow
    cdef ParentDedupMembership membership = None
    cdef NativeQueueAdmission admission
    options = engine.options
    for k in range(19):
        pending[k] = 0
    for k in range(6):
        elapsed[k] = 0
    counts, times = engine.stats.counters, engine.stats.times_ns
    counts['native_parent_rows'] += 1
    counts['native_parent_wide_rows'] += 1
    backend, packing, queue = engine.backend, engine.pair_packing, engine.queue
    recipe_context = engine._recipe_context
    store = engine.S
    queries = getattr(engine, '_native_syzygy_queries', None)
    if queries is None:
        queries = PackedSyzygyQueries(store)
        engine._native_syzygy_queries = queries
    queries.refresh()
    top_queries = getattr(engine, '_native_top_query', None)
    if top_queries is None:
        top_queries = NativeTopQuery(engine)
        engine._native_top_query = top_queries
    else:
        top_queries.refresh(engine)
    parent_rows = getattr(engine, '_native_parent_rows', None)
    if parent_rows is None:
        parent_rows = _ParentRows(engine)
        engine._native_parent_rows = parent_rows
    else:
        parent_rows.refresh(engine)
    qrow = &parent_rows.rows[j]
    cq, scq, kj = <object>qrow.coefficient, <object>qrow.signature_coefficient, <object>qrow.ratio
    mq = GVWPairWord(qrow.leading, qrow.leading_high)
    ssq = GVWPairWord(qrow.signature, qrow.signature_high)
    admission = NativeQueueAdmission(queue)
    materialize = engine._materialize_polynomial
    complete = store._complete_uncovered
    check_budget = engine._check_budget
    # A packed oracle compares the existing upper parent's formal ratio.
    # No signature/vector construction is required for a rejected G.
    top = engine._top_reducer_index
    deduplicate = queue.deduplicate
    if deduplicate:
        membership = queue._native_j_membership
        if membership is None:
            membership = ParentDedupMembership(queue)
            queue._native_j_membership = membership
    generation_filter = options.j_generation_filter
    g_precheck = options.g_precheck
    engine.phase = 'candidate generation'
    try:
        for i in range(j):
            prow = &parent_rows.rows[i]
            if not prow.packed:
                counts['native_parent_fallback_pairs'] += 1
                _flush_parent_counts(counts, pending)
                _flush_parent_times(times, elapsed)
                queries.flush()
                top_queries.flush()
                engine._emit_pair(i, j)
                queries.refresh()
                continue
            tick = gvw_management_clock_ns()
            cp = <object>prow.coefficient
            mp = GVWPairWord(prow.leading, prow.leading_high)
            ssp = GVWPairWord(prow.signature, prow.signature_high)
            if top_queries.enabled:
                cmp = top_queries.compare_parents(i, j)
            else:
                ki = <object>prow.ratio
                cmp = 1 if ki > kj else (-1 if ki < kj else 0)
            pending[0] += 1
            pending[2] += 1
            pending[18] += 1
            wide_pairs += 1
            if cmp > 0:
                syz_coefficient = cq * <object>prow.signature_coefficient
                syz_exp, syz_index = mq + ssp, prow.component
                have_syz = True
            elif cmp < 0:
                syz_coefficient = -(cp * scq)
                syz_exp, syz_index = mp + ssq, qrow.component
                have_syz = True
            else:
                syz_coefficient = cq * <object>prow.signature_coefficient - cp * scq
                syz_exp, syz_index = mq + ssp, prow.component
                have_syz = bool(syz_coefficient)
            W = gvw_pair_lcm(mp, mq)
            if prow.small and qrow.small:
                if prow.small_coefficient % qrow.small_coefficient == 0:
                    d_small = qrow.small_coefficient
                    comparable = True
                elif qrow.small_coefficient % prow.small_coefficient == 0:
                    d_small = prow.small_coefficient
                    comparable = True
                else:
                    d_small = gvw_pair_gcd(prow.small_coefficient, qrow.small_coefficient)
                    comparable = False
                d = d_small
            elif cp == cq or cp == -cq:
                d, comparable = abs(cp), True
            elif cp % cq == 0:
                d, comparable = abs(cq), True
            elif cq % cp == 0:
                d, comparable = abs(cp), True
            else:
                d, comparable = gcd(cp, cq), False
            if cmp >= 0:
                upper, index = W - mp + ssp, prow.component
            else:
                upper, index = W - mq + ssq, qrow.component
            if cmp:
                pending[1] += 1
                if cmp > 0:
                    parent, shift = i, W - mp
                    if prow.small and qrow.small:
                        scalar_small = qrow.small_coefficient // d_small
                        j_scalar = scalar_small
                    else:
                        j_scalar = abs(cq) // d
                    if not prow.positive:
                        j_scalar = -j_scalar
                    j_coefficient = j_scalar * <object>prow.signature_coefficient
                else:
                    parent, shift = j, W - mq
                    if prow.small and qrow.small:
                        scalar_small = prow.small_coefficient // d_small
                        j_scalar = scalar_small
                    else:
                        j_scalar = abs(cp) // d
                    if not qrow.positive:
                        j_scalar = -j_scalar
                    j_coefficient = j_scalar * scq
            gdata = None
            if comparable:
                pending[3] += 1
                pending[4] += 1
                elapsed[0] += gvw_management_clock_ns() - tick
            else:
                rejected = False
                unpacked_W = None
                elapsed[0] += gvw_management_clock_ns() - tick
                if g_precheck:
                    tick = gvw_management_clock_ns()
                    pending[5] += 1
                    pending[17] += 1
                    if top_queries.enabled:
                        rejected = top_queries.query128(W.low, W.high, d, i if cmp >= 0 else j)
                    else:
                        unpacked_W = packing.unpack(_pair_word_python(W))
                        rejected = top(packing.unpack(_pair_word_python(upper)), index, unpacked_W, d) is not None
                    elapsed[1] += gvw_management_clock_ns() - tick
                tick = gvw_management_clock_ns()
                if rejected:
                    pending[4] += 1
                    pending[6] += 1
                else:
                    if unpacked_W is None:
                        unpacked_W = packing.unpack(_pair_word_python(W))
                    up, uq = packing.unpack(_pair_word_python(W - mp)), packing.unpack(_pair_word_python(W - mq))
                    a, b, _ = backend.bezout(cp, cq)
                    adjusted = False
                    if cmp > 0:
                        if not a:
                            a += cq // d
                            b -= cp // d
                            adjusted = True
                        coefficient = a * <object>prow.signature_coefficient
                    elif cmp < 0:
                        if not b:
                            a += cq // d
                            b -= cp // d
                            adjusted = True
                        coefficient = b * scq
                    else:
                        coefficient = a * <object>prow.signature_coefficient + b * scq
                        if not coefficient:
                            a += cq // d
                            b -= cp // d
                            adjusted = True
                            coefficient = a * <object>prow.signature_coefficient + b * scq
                    sig = _signature_from_arithmetic(coefficient, packing.unpack(_pair_word_python(upper)), index)
                    pending[15] += 1
                    pending[16] += adjusted
                    factory = GPolynomialRecipe(recipe_context,
                        <object>prow.polynomial, a, up, <object>qrow.polynomial, b, uq)
                    gdata = sig, d, factory
                elapsed[0] += gvw_management_clock_ns() - tick

            tick = gvw_management_clock_ns()
            if have_syz:
                pending[7] += 1
                pending[9] += 1
                if queries.query128(syz_coefficient, syz_exp.low, syz_exp.high, syz_index):
                    pending[10] += 1
                    pending[11] += 1
                    elapsed[2] += gvw_management_clock_ns() - tick
                else:
                    after = gvw_management_clock_ns()
                    elapsed[2] += after - tick
                    engine.phase = 'syzygy completion'
                    complete(_signature_from_arithmetic(
                        syz_coefficient, packing.unpack(_pair_word_python(syz_exp)), syz_index))
                    queries.refresh(syz_index)
                    elapsed[3] += gvw_management_clock_ns() - after
                    engine.phase = 'candidate generation'
            else:
                pending[8] += 1
                elapsed[0] += gvw_management_clock_ns() - tick

            queue_tick = gvw_management_clock_ns()
            filter_elapsed = 0
            if cmp:
                packed_upper = packed_W = None
                inserted = True
                if deduplicate:
                    membership_queries += 1
                    if membership.may_contain128(j_coefficient, upper.low, upper.high, index):
                        membership_positives += 1
                        packed_upper = _pair_word_python(upper)
                        dkey = (j_coefficient, packed_upper, index)
                        packed_W = _pair_word_python(W)
                        inserted = admission.accept_bound(dkey, packed_W)
                if inserted and generation_filter:
                    tick = gvw_management_clock_ns()
                    pending[12] += 1
                    if queries.query128(j_coefficient, upper.low, upper.high, index):
                        pending[13] += 1
                        inserted = False
                    filter_elapsed = gvw_management_clock_ns() - tick
                    elapsed[4] += filter_elapsed
                if inserted:
                    if packed_upper is None:
                        packed_upper = _pair_word_python(upper)
                    if packed_W is None:
                        packed_W = _pair_word_python(W)
                    factory = JPolynomialRecipe(recipe_context,
                        <object>parent_rows.rows[parent].polynomial, j_scalar, _pair_word_python(shift))
                    entry = admission.insert(j_coefficient, packed_upper, index, packed_W, factory)
                    entry.generating_parents = (i, j)
                else:
                    pending[14] += 1
            if gdata is not None:
                sig, d, factory = gdata
                queue.push_g_lazy(sig, unpacked_W, d, factory)
                queue.active[queue.serial].generating_parents = (i, j)
            elapsed[5] += gvw_management_clock_ns() - queue_tick - filter_elapsed
            since_check += 1
            if since_check >= 128:
                _flush_parent_counts(counts, pending)
                _flush_parent_times(times, elapsed)
                queries.flush()
                top_queries.flush()
                check_budget()
                since_check = 0
    finally:
        _flush_parent_counts(counts, pending)
        _flush_parent_times(times, elapsed)
        queries.flush()
        top_queries.flush()
        counts['native_parent_wide_pairs'] += wide_pairs
        counts['native_j_membership_queries'] += membership_queries
        counts['native_j_membership_positives'] += membership_positives
    check_budget()
    return True


cdef _emit_parent_pairs_dynamic(engine, Py_ssize_t j):
    """Fuse one parent row using arbitrarily many exact packed byte lanes.

    Python's arbitrary-width integer operations implement lane arithmetic in C;
    the surrounding dispatch and admission loops stay compiled. Only exponent
    overflow takes the ordinary per-pair path, and later packed pairs still
    benefit. No coefficient or monomial dimension is truncated.
    """
    cdef Py_ssize_t i, k, index, syz_index, since_check = 0
    cdef Py_ssize_t parent = 0
    cdef object mp, mq, ssp, ssq, W, upper, syz_exp, shift
    cdef object lane_guard, high_bits, select_a
    cdef uint64_t d_small = 1
    cdef uint64_t scalar_small = 0
    cdef uint64_t membership_queries = 0, membership_positives = 0
    cdef uint64_t tick, after, queue_tick, filter_elapsed
    cdef uint64_t pending[19]
    cdef uint64_t elapsed[6]
    cdef int cmp
    cdef bint comparable, rejected, inserted, adjusted, have_syz
    cdef bint deduplicate, generation_filter, g_precheck
    cdef object cp, cq, scq, ki, kj, pp
    cdef object coefficient, j_coefficient, j_scalar, syz_coefficient
    cdef object d, a, b, up, uq, sig, factory, gdata, entry, dkey
    cdef object unpacked_W = None
    cdef object packed_upper, packed_W
    cdef uint64_t wide_pairs = 0
    cdef PackedSyzygyQueries queries
    cdef NativeTopQuery top_queries
    cdef _ParentRows parent_rows
    cdef GVWPairParent *prow
    cdef GVWPairParent *qrow
    cdef ParentDedupMembership membership = None
    cdef NativeQueueAdmission admission
    options = engine.options
    for k in range(19):
        pending[k] = 0
    for k in range(6):
        elapsed[k] = 0
    counts, times = engine.stats.counters, engine.stats.times_ns
    counts['native_parent_rows'] += 1
    counts['native_parent_wide_rows'] += 1
    counts['native_parent_dynamic_rows'] += 1
    backend, packing, queue = engine.backend, engine.pair_packing, engine.queue
    lane_guard = packing.guard
    recipe_context = engine._recipe_context
    store = engine.S
    queries = getattr(engine, '_native_syzygy_queries', None)
    if queries is None:
        queries = PackedSyzygyQueries(store)
        engine._native_syzygy_queries = queries
    queries.refresh()
    top_queries = getattr(engine, '_native_top_query', None)
    if top_queries is None:
        top_queries = NativeTopQuery(engine)
        engine._native_top_query = top_queries
    else:
        top_queries.refresh(engine)
    parent_rows = getattr(engine, '_native_parent_rows', None)
    if parent_rows is None:
        parent_rows = _ParentRows(engine)
        engine._native_parent_rows = parent_rows
    else:
        parent_rows.refresh(engine)
    qrow = &parent_rows.rows[j]
    cq, scq, kj = <object>qrow.coefficient, <object>qrow.signature_coefficient, <object>qrow.ratio
    mq = <object>qrow.packed_leading
    ssq = <object>qrow.packed_signature
    admission = NativeQueueAdmission(queue)
    materialize = engine._materialize_polynomial
    complete = store._complete_uncovered
    check_budget = engine._check_budget
    # A packed oracle compares the existing upper parent's formal ratio.
    # No signature/vector construction is required for a rejected G.
    top = engine._top_reducer_index
    deduplicate = queue.deduplicate
    if deduplicate:
        membership = queue._native_j_membership
        if membership is None:
            membership = ParentDedupMembership(queue)
            queue._native_j_membership = membership
    generation_filter = options.j_generation_filter
    g_precheck = options.g_precheck
    engine.phase = 'candidate generation'
    try:
        for i in range(j):
            prow = &parent_rows.rows[i]
            if not prow.packed:
                counts['native_parent_fallback_pairs'] += 1
                _flush_parent_counts(counts, pending)
                _flush_parent_times(times, elapsed)
                queries.flush()
                top_queries.flush()
                engine._emit_pair(i, j)
                queries.refresh()
                continue
            tick = gvw_management_clock_ns()
            cp = <object>prow.coefficient
            mp = <object>prow.packed_leading
            ssp = <object>prow.packed_signature
            if top_queries.enabled:
                cmp = top_queries.compare_parents(i, j)
            else:
                ki = <object>prow.ratio
                cmp = 1 if ki > kj else (-1 if ki < kj else 0)
            pending[0] += 1
            pending[2] += 1
            pending[18] += 1
            wide_pairs += 1
            if cmp > 0:
                syz_coefficient = cq * <object>prow.signature_coefficient
                syz_exp, syz_index = mq + ssp, prow.component
                have_syz = True
            elif cmp < 0:
                syz_coefficient = -(cp * scq)
                syz_exp, syz_index = mp + ssq, qrow.component
                have_syz = True
            else:
                syz_coefficient = cq * <object>prow.signature_coefficient - cp * scq
                syz_exp, syz_index = mq + ssp, prow.component
                have_syz = bool(syz_coefficient)
            high_bits = ((mp | lane_guard) - mq) & lane_guard
            select_a = high_bits - (high_bits >> 7)
            W = (mp & select_a) | (mq & ~select_a)
            if prow.small and qrow.small:
                if prow.small_coefficient % qrow.small_coefficient == 0:
                    d_small = qrow.small_coefficient
                    comparable = True
                elif qrow.small_coefficient % prow.small_coefficient == 0:
                    d_small = prow.small_coefficient
                    comparable = True
                else:
                    d_small = gvw_pair_gcd(prow.small_coefficient, qrow.small_coefficient)
                    comparable = False
                d = d_small
            elif cp == cq or cp == -cq:
                d, comparable = abs(cp), True
            elif cp % cq == 0:
                d, comparable = abs(cq), True
            elif cq % cp == 0:
                d, comparable = abs(cp), True
            else:
                d, comparable = gcd(cp, cq), False
            if cmp >= 0:
                upper, index = W - mp + ssp, prow.component
            else:
                upper, index = W - mq + ssq, qrow.component
            if cmp:
                pending[1] += 1
                if cmp > 0:
                    parent, shift = i, W - mp
                    if prow.small and qrow.small:
                        scalar_small = qrow.small_coefficient // d_small
                        j_scalar = scalar_small
                    else:
                        j_scalar = abs(cq) // d
                    if not prow.positive:
                        j_scalar = -j_scalar
                    j_coefficient = j_scalar * <object>prow.signature_coefficient
                else:
                    parent, shift = j, W - mq
                    if prow.small and qrow.small:
                        scalar_small = prow.small_coefficient // d_small
                        j_scalar = scalar_small
                    else:
                        j_scalar = abs(cp) // d
                    if not qrow.positive:
                        j_scalar = -j_scalar
                    j_coefficient = j_scalar * scq
            gdata = None
            if comparable:
                pending[3] += 1
                pending[4] += 1
                elapsed[0] += gvw_management_clock_ns() - tick
            else:
                rejected = False
                unpacked_W = None
                elapsed[0] += gvw_management_clock_ns() - tick
                if g_precheck:
                    tick = gvw_management_clock_ns()
                    pending[5] += 1
                    pending[17] += 1
                    if top_queries.enabled:
                        rejected = top_queries.query_packed(W, d, i if cmp >= 0 else j)
                    else:
                        unpacked_W = packing.unpack(W)
                        rejected = top(packing.unpack(upper), index, unpacked_W, d) is not None
                    elapsed[1] += gvw_management_clock_ns() - tick
                tick = gvw_management_clock_ns()
                if rejected:
                    pending[4] += 1
                    pending[6] += 1
                else:
                    if unpacked_W is None:
                        unpacked_W = packing.unpack(W)
                    up, uq = packing.unpack(W - mp), packing.unpack(W - mq)
                    a, b, _ = backend.bezout(cp, cq)
                    adjusted = False
                    if cmp > 0:
                        if not a:
                            a += cq // d
                            b -= cp // d
                            adjusted = True
                        coefficient = a * <object>prow.signature_coefficient
                    elif cmp < 0:
                        if not b:
                            a += cq // d
                            b -= cp // d
                            adjusted = True
                        coefficient = b * scq
                    else:
                        coefficient = a * <object>prow.signature_coefficient + b * scq
                        if not coefficient:
                            a += cq // d
                            b -= cp // d
                            adjusted = True
                            coefficient = a * <object>prow.signature_coefficient + b * scq
                    sig = _signature_from_arithmetic(coefficient, packing.unpack(upper), index)
                    pending[15] += 1
                    pending[16] += adjusted
                    factory = GPolynomialRecipe(recipe_context,
                        <object>prow.polynomial, a, up, <object>qrow.polynomial, b, uq)
                    gdata = sig, d, factory
                elapsed[0] += gvw_management_clock_ns() - tick

            tick = gvw_management_clock_ns()
            if have_syz:
                pending[7] += 1
                pending[9] += 1
                if queries.query(syz_coefficient, syz_exp, syz_index):
                    pending[10] += 1
                    pending[11] += 1
                    elapsed[2] += gvw_management_clock_ns() - tick
                else:
                    after = gvw_management_clock_ns()
                    elapsed[2] += after - tick
                    engine.phase = 'syzygy completion'
                    complete(_signature_from_arithmetic(
                        syz_coefficient, packing.unpack(syz_exp), syz_index))
                    queries.refresh(syz_index)
                    elapsed[3] += gvw_management_clock_ns() - after
                    engine.phase = 'candidate generation'
            else:
                pending[8] += 1
                elapsed[0] += gvw_management_clock_ns() - tick

            queue_tick = gvw_management_clock_ns()
            filter_elapsed = 0
            if cmp:
                packed_upper = packed_W = None
                inserted = True
                if deduplicate:
                    membership_queries += 1
                    if membership.may_contain_packed(j_coefficient, upper, index):
                        membership_positives += 1
                        packed_upper = upper
                        dkey = (j_coefficient, packed_upper, index)
                        packed_W = W
                        inserted = admission.accept_bound(dkey, packed_W)
                if inserted and generation_filter:
                    tick = gvw_management_clock_ns()
                    pending[12] += 1
                    if queries.query(j_coefficient, upper, index):
                        pending[13] += 1
                        inserted = False
                    filter_elapsed = gvw_management_clock_ns() - tick
                    elapsed[4] += filter_elapsed
                if inserted:
                    if packed_upper is None:
                        packed_upper = upper
                    if packed_W is None:
                        packed_W = W
                    factory = JPolynomialRecipe(recipe_context,
                        <object>parent_rows.rows[parent].polynomial, j_scalar, shift)
                    entry = admission.insert(j_coefficient, packed_upper, index, packed_W, factory)
                    entry.generating_parents = (i, j)
                else:
                    pending[14] += 1
            if gdata is not None:
                sig, d, factory = gdata
                queue.push_g_lazy(sig, unpacked_W, d, factory)
                queue.active[queue.serial].generating_parents = (i, j)
            elapsed[5] += gvw_management_clock_ns() - queue_tick - filter_elapsed
            since_check += 1
            if since_check >= 128:
                _flush_parent_counts(counts, pending)
                _flush_parent_times(times, elapsed)
                queries.flush()
                top_queries.flush()
                check_budget()
                since_check = 0
    finally:
        _flush_parent_counts(counts, pending)
        _flush_parent_times(times, elapsed)
        queries.flush()
        top_queries.flush()
        counts['native_parent_wide_pairs'] += wide_pairs
        counts['native_parent_dynamic_pairs'] += wide_pairs
        counts['native_j_membership_queries'] += membership_queries
        counts['native_j_membership_positives'] += membership_positives
    check_budget()
    return True
