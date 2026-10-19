cdef class _PackedRatioKeys:
    """Exact bounded grevlex ratio keys, separate from integer coefficients."""
    cdef Py_ssize_t bits, n, high, count
    cdef object keys
    cdef GvwzzReducerRows *rows
    cdef GvwzzWideReducerRows *wide_rows
    cdef bint fast_geometry, wide
    cdef uint64_t guard_mask
    cdef GvwzzWideKey wide_guard_mask

    def __cinit__(self):
        self.rows = NULL
        self.wide_rows = NULL

    def __dealloc__(self):
        if self.rows is not NULL:
            del self.rows
        if self.wide_rows is not NULL:
            del self.wide_rows

    def __init__(self, Py_ssize_t bits, Py_ssize_t n):
        cdef Py_ssize_t k
        self.bits, self.n, self.high, self.count = bits, n, bits * n, 0
        self.keys = []
        # Keep the existing one-word implementation for its original domain.
        self.wide = bits * (n + 1) > 63 or n * (bits + 1) > 64
        self.fast_geometry = n * (bits + 1) <= (128 if self.wide else 64)
        self.guard_mask = 0
        self.wide_guard_mask = GvwzzWideKey(0)
        if self.fast_geometry:
            for k in range(n):
                if self.wide:
                    self.wide_guard_mask = self.wide_guard_mask | (
                        GvwzzWideKey(1) << (k * (bits + 1) + bits))
                else:
                    self.guard_mask |= (<uint64_t>1) << (k * (bits + 1) + bits)
        if gvwzz_gmp_available():
            if self.wide:
                self.wide_rows = new GvwzzWideReducerRows()
            else:
                self.rows = new GvwzzReducerRows()

    cdef long long encode(self, tuple exponent) except *:
        cdef Py_ssize_t k
        cdef long long digit, degree = 0, value = 0
        for k in range(self.n - 1, -1, -1):
            digit = exponent[k]
            degree += digit
            value = (value << self.bits) + digit
        return (degree << self.high) - value

    cdef GvwzzWideKey encode_wide(self, tuple exponent) except *:
        cdef Py_ssize_t k
        cdef uint64_t digit, degree = 0
        cdef GvwzzWideKey value = GvwzzWideKey(0)
        for k in range(self.n - 1, -1, -1):
            digit = exponent[k]
            degree += digit
            value = (value << self.bits) + GvwzzWideKey(digit)
        return (GvwzzWideKey(degree) << self.high) - value

    cdef uint64_t guarded(self, tuple exponent) except *:
        cdef Py_ssize_t k
        cdef uint64_t value = 0, digit
        for k in range(self.n - 1, -1, -1):
            digit = exponent[k]
            value = (value << (self.bits + 1)) | digit
        return value

    cdef GvwzzWideKey guarded_wide(self, tuple exponent) except *:
        cdef Py_ssize_t k
        cdef uint64_t digit
        cdef GvwzzWideKey value = GvwzzWideKey(0)
        for k in range(self.n - 1, -1, -1):
            digit = exponent[k]
            value = (value << (self.bits + 1)) | GvwzzWideKey(digit)
        return value

    cdef void update(self, solver):
        cdef Py_ssize_t i
        cdef long long value
        cdef GvwzzWideKey wide_value, wide_monomial
        cdef object sig, lc, sig_coefficient, ordered
        for i in range(self.count, len(solver.P)):
            sig = solver.P[i].signature
            if self.wide:
                wide_value = (self.encode_wide(sig.monomial)
                    + self.encode_wide(solver.order.weights[sig.index])
                    - self.encode_wide(solver.leading[i][0]))
                self.keys.append((gvwzz_wide_to_python(wide_value, True),
                                  solver.order.rank[sig.index]))
            else:
                value = (self.encode(sig.monomial)
                    + self.encode(solver.order.weights[sig.index])
                    - self.encode(solver.leading[i][0]))
                self.keys.append((value, solver.order.rank[sig.index]))
            if self.rows is not NULL or self.wide_rows is not NULL:
                lc = _gmp_integer(int(solver.leading[i][1]))
                sig_coefficient = _gmp_integer(sig.coefficient)
                if self.wide:
                    wide_monomial = (self.guarded_wide(solver.leading[i][0])
                        if self.fast_geometry else GvwzzWideKey(0))
                    self.wide_rows.add(wide_value, solver.order.rank[sig.index],
                        <PyObject *>lc, <PyObject *>sig_coefficient,
                        len(solver.P[i].polynomial),
                        abs(int(solver.leading[i][1])).bit_length(), wide_monomial)
                else:
                    self.rows.add(value, solver.order.rank[sig.index], <PyObject *>lc,
                        <PyObject *>sig_coefficient, len(solver.P[i].polynomial),
                        abs(int(solver.leading[i][1])).bit_length(),
                        self.guarded(solver.leading[i][0]) if self.fast_geometry else 0)
        if (self.fast_geometry and self.count != len(solver.P)
                and (self.rows is not NULL or self.wide_rows is not NULL)):
            ordered = list(range(len(solver.P)))
            if solver.options.reducer_selection != 'oldest':
                ordered.sort(key=solver.reducer_rank.__getitem__)
            if self.wide:
                self.wide_rows.set_order(<PyObject *>ordered, self.wide_guard_mask)
            else:
                self.rows.set_order(<PyObject *>ordered, self.guard_mask)
        self.count = len(solver.P)


cdef class _PackedRatioContext:
    cdef bint enabled
    cdef _PackedRatioKeys codec
    cdef long long signature_value
    cdef GvwzzWideKey wide_signature_value
    cdef object signature_rank, ratio_keys
    cdef object signature_coefficient, native_signature_coefficient

    @property
    def native_geometry_enabled(self):
        return bool(self.enabled and self.codec.fast_geometry
                    and (self.codec.rows is not NULL or self.codec.wide_rows is not NULL))

    @property
    def geometry_cache_info(self):
        """Read-only diagnostics for the active ratio codec's retained cache."""
        if not self.enabled:
            return None
        if self.codec.wide_rows is not NULL:
            return self.codec.wide_rows.geometry_cache_info()
        if self.codec.rows is not NULL:
            return self.codec.rows.geometry_cache_info()
        return None

    def __init__(self, solver, signature, degree_bound):
        cdef object state, max_signature, max_leading, codecs, bound, sig
        cdef Py_ssize_t old_count, i, bits, n = len(solver.backend.names)
        self.enabled = False
        if solver.order.name != 'gvw-g2' or solver.backend.order_name != 'grevlex':
            return
        state = getattr(solver, '_native_ratio_bounds', None)
        if state is None:
            old_count, max_signature, max_leading, codecs = 0, 0, 0, {}
        else:
            old_count, max_signature, max_leading, codecs = state
        for i in range(old_count, len(solver.P)):
            sig = solver.P[i].signature
            max_signature = max(max_signature,
                sum(sig.monomial) + sum(solver.order.weights[sig.index]))
            max_leading = max(max_leading, sum(solver.leading[i][0]))
        solver._native_ratio_bounds = (len(solver.P), max_signature, max_leading, codecs)
        max_signature = max(max_signature,
            sum(signature.monomial) + sum(solver.order.weights[signature.index]))
        bound = max_signature + max(max_leading, degree_bound)
        bits = max(1, bound.bit_length())
        # Every ratio coordinate lies in [-L,S], so differences are < B.
        # E(a)=|a|*B**n-sum(a[k]*B**k) preserves grevlex on these ratios.
        # Each nonnegative E(input) is below 2**127. Since the combined
        # signature and weight degree is bounded by S, their sum also fits;
        # subtraction is exact in signed 128 bits. Larger inputs fall back.
        if bits * (n + 1) > 127:
            return
        self.codec = codecs.get(bits)
        if self.codec is None:
            self.codec = _PackedRatioKeys(bits, n)
            codecs[bits] = self.codec
        self.codec.update(solver)
        if self.codec.wide:
            self.wide_signature_value = (self.codec.encode_wide(signature.monomial)
                + self.codec.encode_wide(solver.order.weights[signature.index]))
        else:
            self.signature_value = (self.codec.encode(signature.monomial)
                + self.codec.encode(solver.order.weights[signature.index]))
        self.signature_rank = solver.order.rank[signature.index]
        self.ratio_keys = self.codec.keys
        self.signature_coefficient = signature.coefficient
        if gvwzz_gmp_available():
            self.native_signature_coefficient = _gmp_integer(signature.coefficient)
        self.enabled = True
