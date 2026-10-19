# cython: language_level=3, boundscheck=False, wraparound=False
"""Optional exact packed syzygy lookups.

Only loop positions and the eight exponent bytes in one block use C types.
Signature coefficients, full packed exponent vectors and all row bit masks
remain arbitrary-precision Python integers. Stored exponent thresholds remain
unbounded, so rows above 127 are still correctly excluded from packed queries.
"""
from bisect import bisect_right
from libc.stdint cimport uint64_t
from libcpp.vector cimport vector


cdef extern from *:
    """
    #include <cstdint>
    #include <vector>
    #include <unordered_map>
    struct GVWSyzygyQueryKey {
        uint64_t monomial;
        uint64_t high;
        uint64_t coefficient;
        bool operator==(const GVWSyzygyQueryKey& b) const {
            return monomial == b.monomial && high == b.high && coefficient == b.coefficient;
        }
    };
    struct GVWSyzygyQueryHash {
        size_t operator()(const GVWSyzygyQueryKey& a) const {
            uint64_t x = a.monomial ^ (a.high * UINT64_C(0x9e3779b97f4a7c15))
                         ^ (a.coefficient + UINT64_C(0x9e3779b97f4a7c15));
            x ^= x >> 30; x *= UINT64_C(0xbf58476d1ce4e5b9);
            x ^= x >> 27; x *= UINT64_C(0x94d049bb133111eb);
            return size_t(x ^ (x >> 31));
        }
    };
    struct GVWSyzygyQueryRow {
        uint64_t monomial;
        uint64_t high;
        uint64_t coefficient;
        size_t slot;
        bool small;
    };
    struct GVWSyzygyQueryComponent {
        std::vector<GVWSyzygyQueryRow> rows;
        std::unordered_map<GVWSyzygyQueryKey, bool, GVWSyzygyQueryHash> cache;
        uint64_t witness_monomial = 0;
        uint64_t witness_high = 0;
        uint64_t witness_coefficient = 0;
        bool witness_valid = false;
        bool witness_small = false;
        bool unit_only = true;
        int cached(uint64_t monomial, uint64_t high, uint64_t coefficient) const {
            auto it = cache.find({monomial, high, coefficient});
            return it == cache.end() ? -1 : int(it->second);
        }
        void remember(uint64_t monomial, uint64_t high, uint64_t coefficient, bool answer) {
            // A bounded native cache needs no Python key, version tuple or LRU node.
            if (cache.size() >= 8192) cache.clear();
            cache[{monomial, high, coefficient}] = answer;
        }
        void invalidate_negative() {
            // A removed syzygy is replaced by a retained divisor, so positive
            // answers remain valid. Negative answers depend on the current S.
            for (auto it = cache.begin(); it != cache.end();) {
                if (!it->second) it = cache.erase(it);
                else ++it;
            }
        }
    };
    struct GVWSyzygyQueryState {
        std::vector<GVWSyzygyQueryComponent> components;
        uint64_t queries = 0, cache_hits = 0, witness_hits = 0;
        uint64_t unit_queries = 0, coefficient_tests = 0;
        explicit GVWSyzygyQueryState(size_t n) : components(n) {}
    };
    """
    cppclass GVWSyzygyQueryRow:
        uint64_t monomial
        uint64_t high
        uint64_t coefficient
        size_t slot
        bint small
    cppclass GVWSyzygyQueryComponent:
        vector[GVWSyzygyQueryRow] rows
        uint64_t witness_monomial
        uint64_t witness_high
        uint64_t witness_coefficient
        bint witness_valid
        bint witness_small
        bint unit_only
        int cached(uint64_t, uint64_t, uint64_t) noexcept
        void remember(uint64_t, uint64_t, uint64_t, bint) except +
        void invalidate_negative() except +
    cppclass GVWSyzygyQueryState:
        GVWSyzygyQueryState(size_t) except +
        vector[GVWSyzygyQueryComponent] components
        uint64_t queries
        uint64_t cache_hits
        uint64_t witness_hits
        uint64_t unit_queries
        uint64_t coefficient_tests


cdef class PackedSyzygyQueries:
    """Versioned exact S queries for the fused native parent-row loop.

    Only bounded exponent encodings and proved-small coefficient magnitudes
    enter machine words. Every other coefficient uses Python's exact modulo;
    arbitrary exponents retain the exact fallback. Wider rings use the compiled
    arbitrary-width block index, without a variable-count limit.
    Refresh after S changes and flush before publishing statistics.
    """

    def __cinit__(self, store):
        self.state = NULL
        self.store = store
        self.nvars = store.nvars
        self.enabled = self.nvars >= 0
        self.guard = 0
        if self.enabled:
            self.guard = int.from_bytes(bytes([128]) * min(self.nvars, 8), 'little')
        self.guard_high = int.from_bytes(bytes([128]) * min(8, max(0, self.nvars - 8)), 'little') if self.enabled else 0
        self.versions = [None] * len(store.components)
        self.coefficients = [[] for _ in store.components]
        self.witness_coefficients = [None] * len(store.components)
        self.state = <void*>new GVWSyzygyQueryState(len(store.components))
        self.refresh()

    def __dealloc__(self):
        cdef GVWSyzygyQueryState* state = <GVWSyzygyQueryState*>self.state
        if self.state != NULL:
            del state

    cpdef refresh(self, Py_ssize_t index=-1):
        cdef GVWSyzygyQueryState* state = <GVWSyzygyQueryState*>self.state
        cdef GVWSyzygyQueryComponent* component
        cdef GVWSyzygyQueryRow row
        cdef Py_ssize_t begin, end, i
        cdef bint bounded
        cdef object trie, monomial, payload, coefficient, magnitude, values, version, exponent
        if not self.enabled or self.nvars > 16:
            return
        if index < 0:
            begin, end = 0, len(self.versions)
        else:
            begin, end = index, index + 1
        for i in range(begin, end):
            version = self.store.component_versions[i]
            if self.versions[i] == version:
                continue
            component = &state.components[i]
            component.invalidate_negative()
            component.rows.clear()
            component.unit_only = self.store.nonunit_counts[i] == 0
            trie = self.store.tries[i]
            values = []
            # Preserve the index's active row and coefficient order. Large
            # stored exponents cannot divide a validated <=127 query lane.
            for monomial, payload in zip(trie.monomials, trie.payloads):
                if payload is None:
                    continue
                bounded = True
                for exponent in monomial:
                    if exponent < 0 or exponent > 127:
                        bounded = False
                        break
                if not bounded:
                    continue
                row.monomial = int.from_bytes(bytes(monomial[:8]), 'little')
                row.high = int.from_bytes(bytes(monomial[8:]), 'little')
                for coefficient in payload:
                    magnitude = abs(coefficient)
                    row.small = magnitude <= 0xffffffffffffffff
                    row.coefficient = <uint64_t>magnitude if row.small else 0
                    row.slot = len(values)
                    values.append(magnitude)
                    component.rows.push_back(row)
            self.coefficients[i] = values
            self.versions[i] = version

    cdef bint query64(self, object coefficient, uint64_t packed,
                      Py_ssize_t index) except -1:
        return self.query128(coefficient, packed, 0, index)

    cdef bint query128(self, object coefficient, uint64_t packed, uint64_t high,
                       Py_ssize_t index) except -1:
        cdef GVWSyzygyQueryState* state = <GVWSyzygyQueryState*>self.state
        cdef GVWSyzygyQueryComponent* component = &state.components[index]
        cdef GVWSyzygyQueryRow* row
        cdef object magnitude = None
        cdef uint64_t small_coefficient = 0
        cdef bint small = True, answer = False, divides
        cdef int cached
        cdef size_t k
        state.queries += 1
        if component.rows.empty():
            return False
        if not component.unit_only:
            magnitude = abs(coefficient)
            small = magnitude <= 0xffffffffffffffff
            if small:
                small_coefficient = magnitude
        if small:
            cached = component.cached(packed, high, small_coefficient)
            if cached >= 0:
                state.cache_hits += 1
                return cached != 0
        if (component.witness_valid and
                (((packed | self.guard) - component.witness_monomial) & self.guard) == self.guard and
                (((high | self.guard_high) - component.witness_high) & self.guard_high) == self.guard_high):
            if component.witness_small and component.witness_coefficient == 1:
                divides = True
            elif small and component.witness_small and not component.unit_only:
                divides = small_coefficient % component.witness_coefficient == 0
            else:
                divides = coefficient % self.witness_coefficients[index] == 0
            if divides:
                state.witness_hits += 1
                return True
        if component.unit_only:
            state.unit_queries += 1
        for k in range(component.rows.size()):
            row = &component.rows[k]
            if ((((packed | self.guard) - row.monomial) & self.guard) != self.guard or
                    (((high | self.guard_high) - row.high) & self.guard_high) != self.guard_high):
                continue
            if component.unit_only:
                divides = True
            else:
                state.coefficient_tests += 1
                if small and row.small:
                    divides = small_coefficient % row.coefficient == 0
                else:
                    divides = coefficient % self.coefficients[index][row.slot] == 0
            if divides:
                component.witness_monomial = row.monomial
                component.witness_high = row.high
                component.witness_coefficient = row.coefficient
                component.witness_small = row.small
                component.witness_valid = True
                self.witness_coefficients[index] = self.coefficients[index][row.slot]
                answer = True
                break
        if small:
            component.remember(packed, high, small_coefficient, answer)
        return answer

    cpdef bint query(self, object coefficient, object packed,
                    Py_ssize_t index) except -1:
        if index < 0 or index >= len(self.versions):
            raise IndexError('syzygy component out of range')
        if packed < 0 or packed.bit_length() > 8 * self.nvars:
            raise ValueError('packed exponent vector has the wrong length')
        if packed & self.store.packed_guard:
            # High query lanes have no guard bit left for scalar subtraction.
            return self.store._has_divisor_parts(
                coefficient, tuple(packed.to_bytes(self.nvars, 'little')), index)
        if self.nvars > 16:
            return has_divisor_packed(self.store, coefficient, packed, index)
        if not self.enabled:
            return self.store._has_divisor_packed(coefficient, packed, index)
        return self.query128(coefficient, packed & 0xffffffffffffffff, packed >> 64, index)

    cpdef flush(self):
        cdef GVWSyzygyQueryState* state = <GVWSyzygyQueryState*>self.state
        cdef object counts = self.store.stats.counters
        counts['syzygy_existence_queries'] += state.queries
        counts['syzygy_existence_cache_hits'] += state.cache_hits
        counts['syzygy_recent_witness_hits'] += state.witness_hits
        counts['syzygy_unit_mask_queries'] += state.unit_queries
        counts['syzygy_existence_coefficients_tested'] += state.coefficient_tests
        state.queries = state.cache_hits = state.witness_hits = 0
        state.unit_queries = state.coefficient_tests = 0


cdef object _divisor_mask_packed(object trie, object packed):
    cdef object mask = trie.all_mask
    cdef object blocks = trie.packed_blocks
    cdef object values_by_coordinate = trie.values
    cdef object masks_by_coordinate = trie.masks
    cdef object cache, chunk, allowed, values
    cdef Py_ssize_t start, stop, k, pos
    cdef uint64_t remaining
    cdef unsigned exponent
    for start, stop, cache in blocks:
        chunk = packed & 0xffffffffffffffff
        packed = packed >> 64
        allowed = cache.get(chunk)
        if allowed is None:
            allowed = trie.all_mask
            remaining = chunk
            for k in range(start, stop):
                exponent = remaining & 255
                remaining >>= 8
                values = values_by_coordinate[k]
                pos = bisect_right(values, exponent)
                if pos < len(values):
                    allowed = allowed & ~masks_by_coordinate[k][values[pos]]
                    if not allowed:
                        break
            if len(cache) >= 2048:
                cache.clear()
            cache[chunk] = allowed
        mask = mask & allowed
        if not mask:
            return 0
    return mask


def divisor_mask_packed(trie, packed):
    return _divisor_mask_packed(trie, packed)


def has_divisor_packed(store, coefficient, packed, index):
    cdef object counts = store.stats.counters
    cdef object trie, key, cache, hit, version, witness, guard, mask, bit, c
    cdef object witness_coefficient = None
    cdef Py_ssize_t row, witness_row = -1
    cdef bint unit_only, answer
    counts['syzygy_existence_queries'] += 1
    trie = store.tries[index]
    if not trie.all_mask:
        return False
    unit_only = store.nonunit_counts[index] == 0
    key = packed if unit_only else (packed, abs(coefficient))
    cache = store.packed_existence_cache[index]
    hit = cache.get(key)
    version = store.component_versions[index]
    if hit is not None and (hit[1] or hit[0] == version):
        cache.move_to_end(key)
        counts['syzygy_existence_cache_hits'] += 1
        return hit[1]
    witness = store.packed_recent_witness[index]
    guard = store.packed_guard
    if (witness is not None and coefficient % witness[0] == 0
            and (((packed | guard) - witness[1]) & guard) == guard):
        counts['syzygy_recent_witness_hits'] += 1
        return True
    mask = _divisor_mask_packed(trie, packed)
    if unit_only:
        counts['syzygy_unit_mask_queries'] += 1
        answer = bool(mask)
        if answer:
            witness_row = (mask & -mask).bit_length() - 1
            witness_coefficient = 1
    else:
        answer = False
        while mask:
            bit = mask & -mask
            row = bit.bit_length() - 1
            for c in trie.payloads[row]:
                counts['syzygy_existence_coefficients_tested'] += 1
                if coefficient % c == 0:
                    answer = True
                    witness_row = row
                    witness_coefficient = c
                    break
            if answer:
                break
            mask = mask ^ bit
    if witness_row >= 0:
        store.packed_recent_witness[index] = (witness_coefficient,
            int.from_bytes(bytes(trie.monomials[witness_row]), 'little'))
    cache[key] = (version, answer)
    cache.move_to_end(key)
    if len(cache) > 8192:
        cache.popitem(last=False)
    return answer
