"""Exact generation-only top witnesses for packed G2/grevlex parent rows.

The complete reduction index and its selection policy remain unchanged.  This
index only answers existence: a unit witness dominates another when both its
leading monomial divides and its formal signature ratio is no larger.
"""

from libc.stdint cimport int32_t, int64_t

cdef extern from *:
    """
    #include <cstdint>
    struct GVWTopWitness {
        uint64_t monomial, high;
        int64_t degree;
        int32_t rank;
        int32_t coordinates[16];
        std::vector<int32_t> extra_coordinates;
        std::vector<uint64_t> words;
        bool packed;
        GVWTopWitness() : monomial(0), high(0), degree(0), rank(0),
                          coordinates{}, packed(true) {}
    };
    static inline bool gvw_top_divides(uint64_t divisor, uint64_t target) {
        const uint64_t guard = UINT64_C(0x8080808080808080);
        return (((target | guard) - divisor) & guard) == guard;
    }
    static inline bool gvw_top_rows_divide(const GVWTopWitness& a,
                                           const GVWTopWitness& b) {
        if (!gvw_top_divides(a.monomial, b.monomial) ||
                !gvw_top_divides(a.high, b.high)) return false;
        for (size_t k = 2; k < a.words.size(); ++k)
            if (!gvw_top_divides(a.words[k], b.words[k])) return false;
        return true;
    }
    static inline bool gvw_top_words_divide(const GVWTopWitness& a,
                                            const std::vector<uint64_t>& words) {
        for (size_t k = 0; k < a.words.size(); ++k)
            if (!gvw_top_divides(a.words[k], words[k])) return false;
        return true;
    }
    static inline int gvw_top_ratio_compare(const GVWTopWitness& a,
                                            const GVWTopWitness& b,
                                            size_t dimensions) {
        if (a.degree != b.degree) return a.degree < b.degree ? -1 : 1;
        for (size_t k = dimensions; k-- > 0;) {
            const int32_t x = k < 16 ? a.coordinates[k] : a.extra_coordinates[k - 16];
            const int32_t y = k < 16 ? b.coordinates[k] : b.extra_coordinates[k - 16];
            if (x != y) return x > y ? -1 : 1;
        }
        // Equal G2 weights and rank imply equal unweighted exponents too,
        // since the rank identifies the input component uniquely.
        return a.rank < b.rank ? -1 : (a.rank > b.rank ? 1 : 0);
    }
    static inline bool gvw_top_ratio_leq(const GVWTopWitness& a,
                                         const GVWTopWitness& b,
                                         size_t dimensions) {
        return gvw_top_ratio_compare(a, b, dimensions) <= 0;
    }
    """
    cdef cppclass GVWTopWitness:
        GVWTopWitness() noexcept
        uint64_t monomial, high
        int64_t degree
        int32_t rank
        int32_t coordinates[16]
        vector[int32_t] extra_coordinates
        vector[uint64_t] words
        bint packed
    bint gvw_top_divides(uint64_t, uint64_t) noexcept
    bint gvw_top_rows_divide(const GVWTopWitness&, const GVWTopWitness&) noexcept
    bint gvw_top_words_divide(const GVWTopWitness&, const vector[uint64_t]&) noexcept
    int gvw_top_ratio_compare(const GVWTopWitness&, const GVWTopWitness&,
                              size_t) noexcept
    bint gvw_top_ratio_leq(const GVWTopWitness&, const GVWTopWitness&,
                           size_t) noexcept


cdef class NativeTopQuery:
    """Incremental exact witnesses; memory is bounded by the basis size.

    Python integers are retained for coefficients of every size.  Only checked
    monomial coordinates and order descriptors enter fixed-width storage.
    Unsupported orders or ratio coordinates use the ordinary Python lookup.
    """
    cdef readonly bint enabled
    cdef Py_ssize_t nvars, count
    cdef vector[GVWTopWitness] rows
    cdef vector[size_t] units, nonunits
    cdef list nonunit_coefficients
    cdef dict nonunit_positions
    cdef object counts
    cdef uint64_t candidates, queries

    def __init__(self, engine):
        self.nvars = len(engine.backend.names)
        self.enabled = (0 < self.nvars
                        and engine.order.name == 'gvw-g2'
                        and engine.backend.order_name == 'grevlex')
        self.count = 0
        self.nonunit_coefficients = []
        self.nonunit_positions = {}
        self.counts = engine.stats.counters
        self.refresh(engine)

    cpdef refresh(self, engine):
        cdef Py_ssize_t i, k, component
        cdef size_t j, kept, old
        cdef GVWTopWitness row
        cdef object signature, leading, weight, value, coefficient, key, position, degree
        cdef bint covered
        if not self.enabled:
            return
        for i in range(self.count, len(engine.P)):
            signature = engine.P[i].signature
            component = signature.index
            leading = engine.leading[i][0]
            weight = engine.order.weights[component]
            row = GVWTopWitness()
            if self.nvars > 16:
                row.extra_coordinates.resize(self.nvars - 16)
                row.words.resize((self.nvars + 7) // 8, 0)
            degree = 0
            value = engine.order.rank[component]
            if value < 0 or value > 2147483647:
                self.enabled = False
                return
            row.rank = value
            for k in range(self.nvars):
                value = signature.monomial[k] - leading[k] + weight[k]
                # Check each coordinate and the total in Python before narrowing.
                # The dynamically sized tail removes any fixed dimension limit.
                if value < -2147483648 or value > 2147483647:
                    self.enabled = False
                    return
                if k < 16:
                    row.coordinates[k] = value
                else:
                    row.extra_coordinates[k - 16] = value
                degree += value
                value = leading[k]
                if value < 0 or value > 127:
                    # Such a monomial cannot divide a packed query.  Its
                    # order descriptor is still usable as a target parent.
                    row.packed = False
                else:
                    if k < 8:
                        row.monomial |= (<uint64_t>value) << (8 * k)
                    elif k < 16:
                        row.high |= (<uint64_t>value) << (8 * (k - 8))
                    if self.nvars > 16:
                        row.words[k // 8] |= (<uint64_t>value) << (8 * (k % 8))
            if degree < -9223372036854775808 or degree > 9223372036854775807:
                self.enabled = False
                return
            row.degree = degree
            self.rows.push_back(row)
            self.count += 1
            if not row.packed:
                continue
            coefficient = abs(int(engine.leading[i][1]))
            if coefficient == 1:
                covered = False
                for j in range(self.units.size()):
                    old = self.units[j]
                    if (gvw_top_rows_divide(self.rows[old], row)
                            and gvw_top_ratio_leq(self.rows[old], row, self.nvars)):
                        covered = True
                        break
                if covered:
                    continue
                kept = 0
                for j in range(self.units.size()):
                    old = self.units[j]
                    if (gvw_top_rows_divide(row, self.rows[old])
                            and gvw_top_ratio_leq(row, self.rows[old], self.nvars)):
                        continue
                    self.units[kept] = old
                    kept += 1
                self.units.resize(kept)
                self.units.push_back(i)
            else:
                key = (row.monomial, row.high, coefficient) if self.nvars <= 16 else (leading, coefficient)
                position = self.nonunit_positions.get(key)
                if position is None:
                    self.nonunit_positions[key] = self.nonunits.size()
                    self.nonunits.push_back(i)
                    self.nonunit_coefficients.append(coefficient)
                else:
                    j = position
                    old = self.nonunits[j]
                    if gvw_top_ratio_leq(row, self.rows[old], self.nvars):
                        self.nonunits[j] = i

    cdef int compare_parents(self, Py_ssize_t left, Py_ssize_t right) noexcept:
        return gvw_top_ratio_compare(self.rows[left], self.rows[right], self.nvars)

    cdef bint query64(self, uint64_t W, object coefficient,
                      Py_ssize_t target_parent) except -1:
        return self.query128(W, 0, coefficient, target_parent)

    cdef bint query128(self, uint64_t W, uint64_t high, object coefficient,
                       Py_ssize_t target_parent) except -1:
        cdef size_t j, parent
        self.queries += 1
        for j in range(self.units.size()):
            parent = self.units[j]
            if (gvw_top_divides(self.rows[parent].monomial, W) and
                    gvw_top_divides(self.rows[parent].high, high)):
                self.candidates += 1
                if gvw_top_ratio_leq(self.rows[parent], self.rows[target_parent], self.nvars):
                    return True
        if coefficient == 1 or coefficient == -1:
            return False
        for j in range(self.nonunits.size()):
            parent = self.nonunits[j]
            if (gvw_top_divides(self.rows[parent].monomial, W) and
                    gvw_top_divides(self.rows[parent].high, high)):
                self.candidates += 1
                if (gvw_top_ratio_leq(self.rows[parent], self.rows[target_parent], self.nvars)
                        and coefficient % self.nonunit_coefficients[j] == 0):
                    return True
        return False

    cdef bint query_packed(self, object packed, object coefficient,
                          Py_ssize_t target_parent) except -1:
        cdef vector[uint64_t] words
        cdef object encoded
        cdef const unsigned char* data
        cdef size_t j, k, parent, nwords = (self.nvars + 7) // 8
        cdef unsigned lane
        if self.nvars <= 16:
            return self.query128(packed & 0xffffffffffffffff, packed >> 64,
                                  coefficient, target_parent)
        encoded = packed.to_bytes(nwords * 8, 'little')
        data = <const unsigned char*>PyBytes_AS_STRING(encoded)
        words.resize(nwords, 0)
        for k in range(nwords):
            for lane in range(8):
                words[k] |= (<uint64_t>data[k * 8 + lane]) << (8 * lane)
        self.queries += 1
        for j in range(self.units.size()):
            parent = self.units[j]
            if gvw_top_words_divide(self.rows[parent], words):
                self.candidates += 1
                if gvw_top_ratio_leq(self.rows[parent], self.rows[target_parent], self.nvars):
                    return True
        if coefficient == 1 or coefficient == -1:
            return False
        for j in range(self.nonunits.size()):
            parent = self.nonunits[j]
            if gvw_top_words_divide(self.rows[parent], words):
                self.candidates += 1
                if (gvw_top_ratio_leq(self.rows[parent], self.rows[target_parent], self.nvars)
                        and coefficient % self.nonunit_coefficients[j] == 0):
                    return True
        return False

    cpdef flush(self):
        if self.candidates:
            self.counts['reducer_candidates'] += self.candidates
            self.candidates = 0
        if self.queries:
            self.counts['native_generation_top_queries'] += self.queries
            self.queries = 0

    def has_top(self, packed_W, coefficient, target_parent):
        """Checked Python entry point for differential tests and diagnostics."""
        cdef uint64_t W, high
        cdef Py_ssize_t parent
        if not self.enabled:
            raise ValueError('Native top index is not eligible for this basis/order')
        if not isinstance(packed_W, int) or packed_W < 0 or packed_W.bit_length() > 8 * self.nvars:
            raise ValueError('Packed target has too many coordinates')
        if packed_W & int.from_bytes(bytes([128]) * self.nvars, 'little'):
            raise ValueError('Packed exponent coordinates must be at most 127')
        parent = target_parent
        if parent < 0 or parent >= self.count:
            raise IndexError('Target parent is outside the indexed basis')
        return bool(self.query_packed(packed_W, coefficient, parent))

    def sizes(self):
        return {'basis': self.rows.size(), 'unit_witnesses': self.units.size(),
                'nonunit_witnesses': self.nonunits.size()}

    def compare(self, left, right):
        """Checked diagnostic entry point for the native parent comparison."""
        cdef Py_ssize_t i = left, j = right
        if not self.enabled:
            raise ValueError('Native top index is not eligible for this basis/order')
        if i < 0 or i >= self.count or j < 0 or j >= self.count:
            raise IndexError('Target parent is outside the indexed basis')
        return self.compare_parents(i, j)
