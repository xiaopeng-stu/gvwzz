# cython: language_level=3, boundscheck=False, wraparound=False
"""Optional exact sparse reduction kernel.

Coefficients use exact Python or optional GMP arbitrary-precision integers.
The GMP accumulator uses bounded monomial keys only after proving that the
whole graded encoding fits; larger keys retain the arbitrary-width path.
Retained SymPy polynomials remain immutable. No floating point is used.
"""
import heapq
from libc.stdint cimport uint64_t
from cpython.ref cimport PyObject
from cpython.exc cimport PyErr_CheckSignals
from bisect import bisect_right
from collections import OrderedDict
from functools import lru_cache

from sympy.polys.domains import ZZ
from .core import Label, Signature

cdef extern from "gvwzz_wide_key.hpp":
    cppclass GvwzzWideKey:
        uint64_t lo, hi
        GvwzzWideKey()
        GvwzzWideKey(uint64_t)
        GvwzzWideKey(uint64_t, uint64_t)
        GvwzzWideKey operator+(const GvwzzWideKey &)
        GvwzzWideKey operator-(const GvwzzWideKey &)
        GvwzzWideKey operator&(const GvwzzWideKey &)
        GvwzzWideKey operator|(const GvwzzWideKey &)
        GvwzzWideKey operator<<(unsigned int)
        GvwzzWideKey operator>>(unsigned int)
        bint operator==(const GvwzzWideKey &)
        bint operator!=(const GvwzzWideKey &)
        bint operator<(const GvwzzWideKey &)
    bint gvwzz_wide_signed_less(const GvwzzWideKey &, const GvwzzWideKey &)
    int gvwzz_wide_from_python(PyObject *, GvwzzWideKey *) except -1
    object gvwzz_wide_to_python(const GvwzzWideKey &, bint signed_value)


cdef extern from *:
    """
    #include <Python.h>
    #include <stdint.h>
    #include <unordered_map>
    #include <queue>
    #include <vector>
    #include <new>
    #include <stdexcept>
    #if defined(GVWZZ_HAVE_GMP) && GVWZZ_HAVE_GMP
    // Include GMP before gmpy2.h, whose public C API has extern-C linkage.
    #include "gmp.h"
    #include "gmpy2.h"
    static bool gvwzz_gmp_available() { return true; }
    static int gvwzz_gmp_initialize() { return import_gmpy2(); }

    // The top key bit is free under the verified 63-bit monomial bound.
    // Cache common reducer coefficients in the same 16-byte term record;
    // larger coefficients retain an owned arbitrary-precision GMP object.
    // Accumulator values always remain arbitrary-precision GMP integers.
    struct GvwzzEncodedTerm {
        static constexpr uint64_t SMALL = UINT64_C(1) << 63;
        uint64_t key_and_flag;
        union { PyObject *coefficient; int64_t small; } value;
        GvwzzEncodedTerm(uint64_t key, PyObject *coefficient) : key_and_flag(key) {
            const size_t bits = mpz_sizeinbase(MPZ(coefficient), 2);
            // Windows unsigned long is 32 bits even on 64-bit Python.
            // The 63-bit guard also excludes INT64_MIN before negation.
            if (bits <= 63 && bits <= sizeof(unsigned long) * 8) {
                const uint64_t magnitude = mpz_get_ui(MPZ(coefficient));
                value.small = mpz_sgn(MPZ(coefficient)) < 0
                    ? -static_cast<int64_t>(magnitude) : static_cast<int64_t>(magnitude);
                key_and_flag |= SMALL;
            } else {
                value.coefficient = coefficient;
                Py_INCREF(coefficient);
            }
        }
        GvwzzEncodedTerm(const GvwzzEncodedTerm &other)
            : key_and_flag(other.key_and_flag), value(other.value) {
            if (!(key_and_flag & SMALL)) Py_INCREF(value.coefficient);
        }
        GvwzzEncodedTerm &operator=(const GvwzzEncodedTerm &) = delete;
        ~GvwzzEncodedTerm() {
            if (!(key_and_flag & SMALL)) Py_DECREF(value.coefficient);
        }
    };
    class GvwzzEncodedTerms {
        size_t coefficient_bytes = 0;
    public:
        std::vector<GvwzzEncodedTerm> terms;
        void reserve(size_t count) { terms.reserve(count); }
        void add(uint64_t key, PyObject *coefficient) {
            terms.emplace_back(key, coefficient);
            if (!(terms.back().key_and_flag & GvwzzEncodedTerm::SMALL)) {
                coefficient_bytes += sizeof(MPZ_Object)
                    + static_cast<size_t>(MPZ(coefficient)->_mp_alloc) * sizeof(mp_limb_t);
            }
        }
        size_t size() const { return terms.size(); }
        // This bounds retained encoded storage, not process RSS: source
        // polynomials, Python cache metadata and active borrowers also live.
        size_t storage_bytes() const {
            return sizeof(*this) + terms.capacity() * sizeof(GvwzzEncodedTerm)
                   + coefficient_bytes;
        }
    };
    class GvwzzSparseAccumulator {
        static constexpr uint64_t EMPTY = UINT64_MAX;
        static constexpr uint64_t DELETED = UINT64_MAX - 1;
        struct Entry {
            uint64_t key;
            mpz_t value;
            bool pending;
            Entry() : key(EMPTY), pending(false) { mpz_init(value); }
            Entry(const Entry &other) : key(other.key), pending(other.pending) {
                mpz_init_set(value, other.value);
            }
            Entry &operator=(const Entry &) = delete;
            ~Entry() { mpz_clear(value); }
        };
        std::vector<Entry> values;
        std::priority_queue<uint64_t> heap;
        size_t live, filled;
        static size_t hash(uint64_t key) {
            key ^= key >> 33;
            key *= UINT64_C(0xff51afd7ed558ccd);
            key ^= key >> 33;
            key *= UINT64_C(0xc4ceb9fe1a85ec53);
            key ^= key >> 33;
            return static_cast<size_t>(key);
        }
        void resize(size_t capacity) {
            std::vector<Entry> replacement(capacity);
            for (Entry &entry : values) {
                if (entry.key >= DELETED) continue;
                size_t slot = hash(entry.key) & (capacity - 1);
                while (replacement[slot].key != EMPTY) slot = (slot + 1) & (capacity - 1);
                replacement[slot].key = entry.key;
                replacement[slot].pending = entry.pending;
                mpz_swap(replacement[slot].value, entry.value);
            }
            values.swap(replacement);
            filled = live;
        }
        Entry &locate(uint64_t key, bool create) {
            if (create && filled * 10 >= values.size() * 6)
                resize(live * 3 < values.size() ? values.size() : values.size() * 2);
            const size_t mask = values.size() - 1;
            size_t slot = hash(key) & mask, deleted = values.size();
            while (values[slot].key != key) {
                if (values[slot].key == EMPTY) {
                    if (deleted != values.size()) slot = deleted;
                    else ++filled;
                    ++live;
                    values[slot].key = key;
                    return values[slot];
                }
                if (values[slot].key == DELETED && deleted == values.size()) deleted = slot;
                slot = (slot + 1) & mask;
            }
            return values[slot];
        }
        void erase(Entry &entry) {
            mpz_set_ui(entry.value, 0);
            entry.key = DELETED;
            entry.pending = false;
            --live;
        }
    public:
        GvwzzSparseAccumulator() : values(64), live(0), filled(0) {}
        void initial(uint64_t key, PyObject *value) {
            Entry &entry = locate(key, true);
            mpz_set(entry.value, MPZ(value));
            if (!entry.pending) { heap.push(key); entry.pending = true; }
        }
        void prune() {
            while (!heap.empty()) {
                Entry &entry = locate(heap.top(), false);
                if (mpz_sgn(entry.value)) return;
                heap.pop();
                erase(entry);
            }
        }
        bool empty() const { return heap.empty(); }
        uint64_t top() const { return heap.top(); }
        PyObject *as_integer(const Entry &entry) const {
            MPZ_Object *temporary = GMPy_MPZ_New(NULL);
            if (temporary == NULL) return NULL;
            mpz_set(temporary->z, entry.value);
            PyObject *result = PyNumber_Long(reinterpret_cast<PyObject *>(temporary));
            Py_DECREF(temporary);
            return result;
        }
        PyObject *top_value() { return as_integer(locate(heap.top(), false)); }
        mpz_srcptr top_coefficient() { return locate(heap.top(), false).value; }
        void finish() {
            Entry &entry = locate(heap.top(), false);
            heap.pop();
            if (!mpz_sgn(entry.value)) erase(entry);
            else entry.pending = false;
        }
        int subtract_terms(const GvwzzEncodedTerms &terms, uint64_t shift,
                           PyObject *coefficient, int mode) {
            return subtract_terms_mpz(terms, shift, MPZ(coefficient));
        }
        int subtract_terms_mpz(const GvwzzEncodedTerms &terms, uint64_t shift,
                               mpz_srcptr coefficient) {
            try {
                for (const GvwzzEncodedTerm &term : terms.terms) {
                    const uint64_t key = (term.key_and_flag & (GvwzzEncodedTerm::SMALL - 1)) + shift;
                    Entry &entry = locate(key, true);
                    if (term.key_and_flag & GvwzzEncodedTerm::SMALL) {
                        if (term.value.small >= 0)
                            mpz_submul_ui(entry.value, coefficient, static_cast<unsigned long>(term.value.small));
                        else
                            mpz_addmul_ui(entry.value, coefficient, static_cast<unsigned long>(-term.value.small));
                    } else {
                        mpz_submul(entry.value, coefficient, MPZ(term.value.coefficient));
                    }
                    if (mpz_sgn(entry.value)) {
                        if (!entry.pending) { heap.push(key); entry.pending = true; }
                    } else if (!entry.pending) erase(entry);
                }
                return 0;
            } catch (const std::bad_alloc &) {
                PyErr_NoMemory();
                return -1;
            }
        }
        PyObject *snapshot() const {
            PyObject *result = PyDict_New();
            if (result == NULL) return NULL;
            for (const Entry &entry : values) {
                if (!mpz_sgn(entry.value)) continue;
                PyObject *key = PyLong_FromUnsignedLongLong(entry.key);
                PyObject *coefficient = as_integer(entry);
                if (key == NULL || coefficient == NULL || PyDict_SetItem(result, key, coefficient) < 0) {
                    Py_XDECREF(key);
                    Py_XDECREF(coefficient);
                    Py_DECREF(result);
                    return NULL;
                }
                Py_DECREF(key);
                Py_DECREF(coefficient);
            }
            return result;
        }
    };
    #else
    static bool gvwzz_gmp_available() { return false; }
    static int gvwzz_gmp_initialize() { return 0; }
    // Keep the extension buildable without GMP. These types cannot be
    // instantiated: Python's existing exact accumulator is selected below.
    class GvwzzEncodedTerms {
    public:
        GvwzzEncodedTerms() { throw std::logic_error("GMP workspace is unavailable"); }
        void reserve(size_t) {}
        void add(uint64_t, PyObject *) {}
        size_t size() const { return 0; }
        size_t storage_bytes() const { return 0; }
    };
    class GvwzzSparseAccumulator {
    public:
        GvwzzSparseAccumulator() { throw std::logic_error("GMP workspace is unavailable"); }
        void initial(uint64_t, PyObject *) {}
        void prune() {}
        bool empty() const { return true; }
        uint64_t top() const { return 0; }
        PyObject *top_value() { Py_RETURN_NONE; }
        void finish() {}
        int subtract_terms(const GvwzzEncodedTerms &, uint64_t, PyObject *, int) { return 0; }
        PyObject *snapshot() const { return PyDict_New(); }
    };
    #endif
    """
    bint gvwzz_gmp_available()
    int gvwzz_gmp_initialize() except -1
    cppclass GvwzzEncodedTerms:
        GvwzzEncodedTerms() except +
        void reserve(size_t count) except +
        void add(uint64_t key, PyObject *coefficient) except +
        size_t size()
        size_t storage_bytes()
    cppclass GvwzzSparseAccumulator:
        GvwzzSparseAccumulator() except +
        void initial(uint64_t key, PyObject *value) except +
        void prune()
        bint empty()
        uint64_t top()
        object top_value()
        void finish()
        int subtract_terms(const GvwzzEncodedTerms &terms, uint64_t shift,
                           PyObject *coefficient, int mode) except -1
        object snapshot()

if gvwzz_gmp_available():
    from gmpy2 import mpz as _gmp_integer
    gvwzz_gmp_initialize()
else:
    _gmp_integer = None
COEFFICIENT_BACKEND = 'gmp' if gvwzz_gmp_available() else 'python'
BULK_REDUCTION_AVAILABLE = gvwzz_gmp_available()

cdef class _CompactTerms:
    cdef GvwzzEncodedTerms *packed

    def __cinit__(self, Py_ssize_t capacity=0):
        self.packed = new GvwzzEncodedTerms()
        if capacity:
            self.packed.reserve(capacity)

    def __dealloc__(self):
        del self.packed

    def __len__(self):
        return self.packed.size()


cdef class _CompactTermCache:
    """Backend-wide LRU for exact reducer encodings of all packing widths."""
    cdef object entries
    cdef size_t budget_bytes, cached_bytes
    cdef unsigned long long hits, misses, evictions

    def __init__(self, budget_bytes=128 * 1024 * 1024):
        # Shared across packing widths. Windows stores more coefficients as
        # owned GMP values because its unsigned long is only 32 bits.
        self.entries = OrderedDict()
        self.set_budget(budget_bytes)

    cdef void set_budget(self, budget_bytes) except *:
        if budget_bytes < 1:
            raise ValueError('Encoded term cache budget must be positive')
        self.budget_bytes = budget_bytes
        self.trim()

    cdef void trim(self) except *:
        cdef object entry
        while self.cached_bytes > self.budget_bytes:
            _, entry = self.entries.popitem(last=False)
            self.cached_bytes -= <size_t>entry[3]
            self.evictions += 1

    cdef object get(self, key):
        cdef object entry = self.entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        self.entries.move_to_end(key)
        return entry

    cdef void put(self, key, polynomial, encoded, degree, size_t storage) except *:
        cdef object previous
        # A single oversized reducer must not evict useful smaller entries.
        if storage > self.budget_bytes:
            return
        previous = self.entries.pop(key, None)
        if previous is not None:
            self.cached_bytes -= <size_t>previous[3]
        self.entries[key] = (polynomial, encoded, degree, storage)
        self.cached_bytes += storage
        self.trim()

    def snapshot(self):
        return {'budget_bytes': self.budget_bytes,
                'cached_bytes': self.cached_bytes, 'entries': len(self.entries),
                'hits': self.hits, 'misses': self.misses, 'evictions': self.evictions}


def _set_compact_cache_budget(backend, budget_bytes):
    """Internal diagnostic hook; existing contexts keep the same cache owner."""
    cdef _CompactTermCache cache
    state = getattr(backend, '_native_compact_terms_cache', None)
    if state is None:
        cache = _CompactTermCache(budget_bytes)
        backend._native_compact_terms_cache = cache
    else:
        cache = state
        cache.set_budget(budget_bytes)


def _compact_cache_info(backend):
    cache = getattr(backend, '_native_compact_terms_cache', None)
    return None if cache is None else cache.snapshot()


cdef tuple _subtract(tuple left, tuple right):
    cdef Py_ssize_t k, n = len(left)
    cdef list out = [None] * n
    for k in range(n):
        out[k] = left[k] - right[k]
    return tuple(out)


cdef class _Context:
    cdef public object encode, decode
    cdef object mask, high_unit
    cdef Py_ssize_t bits, n, high
    cdef bint grevlex
    cdef object cache
    cdef Py_ssize_t terms
    cdef _CompactTermCache compact_cache

    def __init__(self, backend, Py_ssize_t bits):
        cdef object one = 1
        self.bits = bits
        self.n = len(backend.names)
        self.high = bits * self.n
        self.mask = (one << bits) - 1
        self.high_unit = one << self.high
        self.grevlex = backend.order_name == 'grevlex'
        self.encode = lru_cache(maxsize=65536)(self._encode)
        self.decode = lru_cache(maxsize=65536)(self._decode)
        self.cache = OrderedDict()
        self.terms = 0
        if gvwzz_gmp_available():
            cache = getattr(backend, '_native_compact_terms_cache', None)
            if cache is None:
                cache = backend._native_compact_terms_cache = _CompactTermCache()
            self.compact_cache = cache

    def _encode(self, tuple monomial):
        cdef Py_ssize_t k
        cdef object degree = 0, value = 0, exponent
        cdef bytearray data
        cdef unsigned int e, bit_offset
        cdef Py_ssize_t offset, byte_index
        if len(monomial) != self.n:
            raise ValueError('Invalid exponent vector')
        for exponent in monomial:
            if exponent < 0:
                raise ValueError('Invalid exponent vector')
            degree += exponent
        if degree > self.mask:
            raise ValueError('Exponent packing degree bound exceeded')
        if self.bits <= 8:
            # Packing through a byte buffer avoids one arbitrary-precision
            # shift/allocation per coordinate, while retaining the exact key.
            data = bytearray((self.high + 7) // 8)
            for k in range(self.n):
                e = monomial[k] if self.grevlex else monomial[self.n - 1 - k]
                offset = k * self.bits
                byte_index = offset >> 3
                bit_offset = offset & 7
                data[byte_index] |= (e << bit_offset) & 255
                if bit_offset + self.bits > 8:
                    data[byte_index + 1] |= e >> (8 - bit_offset)
            value = int.from_bytes(data, 'little')
            return ((degree << self.high) - value if self.grevlex else
                    (degree << self.high) + value)
        if self.grevlex:
            for k in range(self.n - 1, -1, -1):
                value = (value << self.bits) | monomial[k]
            return (degree << self.high) - value
        for exponent in monomial:
            value = (value << self.bits) | exponent
        return (degree << self.high) + value

    def _decode(self, value):
        cdef Py_ssize_t k
        cdef object degree, digits
        cdef list out = [None] * self.n
        cdef bytes data
        cdef unsigned int e, bit_offset, small_mask
        cdef Py_ssize_t offset, byte_index
        if value < 0:
            raise ValueError('Negative packed exponent')
        if self.grevlex:
            degree = (value + self.high_unit - 1) >> self.high
            digits = (degree << self.high) - value
        else:
            digits = value & (self.high_unit - 1)
        if self.bits <= 8:
            data = digits.to_bytes((self.high + 7) // 8, 'little')
            small_mask = (1 << self.bits) - 1
            for k in range(self.n):
                offset = k * self.bits
                byte_index = offset >> 3
                bit_offset = offset & 7
                e = data[byte_index] >> bit_offset
                if bit_offset + self.bits > 8:
                    e |= data[byte_index + 1] << (8 - bit_offset)
                out[k if self.grevlex else self.n - 1 - k] = e & small_mask
            return tuple(out)
        if self.grevlex:
            for k in range(self.n):
                out[k] = digits & self.mask
                digits >>= self.bits
        else:
            for k in range(self.n - 1, -1, -1):
                out[k] = digits & self.mask
                digits >>= self.bits
        return tuple(out)

    cdef object get_terms(self, polynomial):
        cdef object key = id(polynomial), entry, m, c, encoded, packed
        cdef _CompactTerms compact
        if self.compact_cache is not None and self.bits * (self.n + 1) <= 63:
            # The backend fixes variable count and order. A retained source
            # polynomial prevents identity reuse; width distinguishes codecs.
            key = (self.bits, key)
            entry = self.compact_cache.get(key)
            if entry is not None:
                return entry[1], entry[2]
            compact = _CompactTerms(len(polynomial))
            for m, c in polynomial.items():
                packed = self.encode(m)
                c = _gmp_integer(int(c))
                compact.packed.add(<uint64_t>packed, <PyObject *>c)
            degree = max((sum(m) for m in polynomial), default=0)
            self.compact_cache.put(key, polynomial, compact, degree,
                                   compact.packed.storage_bytes())
            return compact, degree
        entry = self.cache.get(key)
        if entry is not None:
            self.cache.move_to_end(key)
            return entry[1], entry[2]
        encoded = [(self.encode(m), int(c)) for m, c in polynomial.items()]
        degree = max((sum(m) for m in polynomial), default=0)
        if len(encoded) <= 500000:
            self.cache[key] = (polynomial, encoded, degree)
            self.terms += len(encoded)
            while self.terms > 500000:
                _, entry = self.cache.popitem(last=False)
                self.terms -= len(entry[1])
        return encoded, degree


cdef class NativeWorkspace:
    cdef public object backend, height_upper
    cdef object _polynomial
    cdef GvwzzSparseAccumulator *compact
    cdef bint small_keys
    cdef _Context context
    cdef object heap, pending, bound

    def __cinit__(self):
        self.compact = NULL

    def __dealloc__(self):
        if self.compact is not NULL:
            del self.compact

    @property
    def polynomial(self):
        return self.compact.snapshot() if self.small_keys else self._polynomial

    def __init__(self, backend, polynomial):
        cdef object contexts, m, c, key
        if backend.order_name not in ('grlex', 'grevlex'):
            raise ValueError('Native workspaces require a graded monomial order')
        self.backend = backend
        self.bound = max((sum(m) for m in polynomial), default=0)
        bits = max(1, self.bound.bit_length())
        contexts = getattr(backend, '_native_reduction_contexts', None)
        if contexts is None:
            contexts = backend._native_reduction_contexts = OrderedDict()
        if bits not in contexts:
            contexts[bits] = _Context(backend, bits)
            while len(contexts) > 3:
                contexts.popitem(last=False)
        contexts.move_to_end(bits)
        self.context = contexts[bits]
        # The graded degree bound proves that every key and shifted key
        # fits. Larger exponent vectors keep the arbitrary-width path.
        self.small_keys = gvwzz_gmp_available() and bits * (self.context.n + 1) <= 63
        if self.small_keys:
            self.compact = new GvwzzSparseAccumulator()
            for m, c in polynomial.items():
                key = self.context.encode(m)
                c = _gmp_integer(int(c))
                self.compact.initial(<uint64_t>key, <PyObject *>c)
        else:
            self._polynomial = {self.context.encode(m): int(c)
                                for m, c in polynomial.items()}
            self.pending = set(self._polynomial)
            self.heap = [-m for m in self._polynomial]
            heapq.heapify(self.heap)

    cpdef leading(self):
        cdef object pair = self.peek()
        if pair is not None:
            return self.context.decode(pair[0]), pair[1]
        return None

    cdef object peek(self):
        cdef object m, c
        if self.small_keys:
            self.compact.prune()
            if self.compact.empty():
                return None
            return self.compact.top(), self.compact.top_value()
        while self.heap:
            m = -self.heap[0]
            c = self._polynomial.get(m, 0)
            if c:
                return m, c
            heapq.heappop(self.heap)
            self.pending.remove(m)
        return None

    cpdef finish_term(self):
        if self.small_keys:
            if self.compact.empty():
                raise IndexError('Cannot finish an empty workspace')
            self.compact.finish()
        else:
            self.pending.remove(-heapq.heappop(self.heap))

    cpdef subtract_multiple(self, other, coefficient, shift):
        cdef object terms, degree, mu
        terms, degree = self.context.get_terms(other)
        if degree + sum(shift) > self.bound:
            raise ValueError('Reduction would violate the packed degree bound')
        mu = self.context.encode(shift)
        self.subtract_encoded(terms, coefficient, mu)

    cdef void subtract_encoded(self, terms, coefficient, mu):
        cdef object monomial, c, new, value
        cdef uint64_t small_mu
        cdef int mode
        cdef _CompactTerms compact_terms
        if self.small_keys:
            if not coefficient:
                return
            small_mu = mu
            mode = 0
            compact_terms = terms
            coefficient = _gmp_integer(coefficient)
            self.compact.subtract_terms(compact_terms.packed[0], small_mu,
                                       <PyObject *>coefficient, mode)
            return
        for monomial, c in terms:
            new = monomial + mu
            value = self._polynomial.get(new, 0) - coefficient * c
            if value:
                self._polynomial[new] = value
                if new not in self.pending:
                    self.pending.add(new)
                    heapq.heappush(self.heap, -new)
            else:
                self._polynomial.pop(new, None)

    cpdef result(self):
        cdef object m, c
        return self.backend.ring.from_dict({self.context.decode(m): ZZ(c)
                                           for m, c in self.polynomial.items()})


cdef list _divisors(index, tuple target):
    """The same exact bitset intersection and insertion-order enumeration."""
    cdef Py_ssize_t k, pos, start, stop, block
    cdef object mask, bit, payload, one = 1, state, caches, cache, key, allowed, all_mask
    cdef list result = []
    if index.dirty:
        index._refresh()
    state = getattr(index, '_native_blocks', None)
    if state is None or state[0] != len(index.payloads):
        caches = [{} for _ in range(0, len(target), 8)]
        index._native_blocks = (len(index.payloads), caches)
    else:
        caches = state[1]
    all_mask = (one << len(index.payloads)) - 1
    mask = all_mask
    block = 0
    for start in range(0, len(target), 8):
        stop = min(start + 8, len(target))
        key = target[start:stop]
        cache = caches[block]
        allowed = cache.get(key)
        if allowed is None:
            allowed = all_mask
            for k in range(start, stop):
                pos = bisect_right(index.values[k], target[k]) - 1
                if pos < 0:
                    allowed = 0
                    break
                allowed &= index.prefix[k][pos]
                if not allowed:
                    break
            if len(cache) >= 2048:
                cache.clear()
            cache[key] = allowed
        mask &= allowed
        if not mask:
            return result
        block += 1
    while mask:
        bit = mask & -mask
        payload = index.payloads[bit.bit_length() - 1]
        if payload is not None:
            result.append(payload)
        mask ^= bit
    return result


cdef object _first_frozen_reducer(solver, NativeWorkspace work, packed, coefficient,
                                  signature):
    """A proof-checked shortcut for the first ordinary POT reducer.

Every bucket is already sorted by the selected reducer policy. If their
minimum belongs to a lower POT component and divides the coefficient, it is
the first exact regular reducer in the reference search. Other situations
use the complete policy implementation without changing a reduction choice.
    """
    cdef object index = solver.reducer_index
    cdef _Context context = work.context
    cdef object state, caches, cache, cache_key, allowed, all_mask, mask
    cdef object one = 1, chunk_mask, digits, degree, bit, bucket, chosen = None
    cdef object rank, best_rank = None, i, c
    cdef Py_ssize_t start, stop, block, k, pos
    cdef unsigned int e, small_mask
    cdef uint64_t chunk, remaining
    if context.bits > 8:
        return None
    if index.dirty:
        index._refresh()
    state = getattr(index, '_native_packed_blocks', None)
    if state is None or state[0] != len(index.payloads):
        caches = [{} for _ in range(0, context.n, 8)]
        index._native_packed_blocks = (len(index.payloads), caches)
    else:
        caches = state[1]
    if context.grevlex:
        degree = (packed + context.high_unit - 1) >> context.high
        digits = (degree << context.high) - packed
    else:
        # Grlex stores the variables in the opposite digit order. The
        # generic branch handles it without allocating another conversion.
        return None
    all_mask = (one << len(index.payloads)) - 1
    mask = all_mask
    small_mask = (1 << context.bits) - 1
    chunk_mask = (one << (8 * context.bits)) - 1
    block = 0
    for start in range(0, context.n, 8):
        stop = min(start + 8, context.n)
        chunk = digits & chunk_mask
        digits >>= 8 * context.bits
        cache = caches[block]
        cache_key = (context.bits, chunk)
        allowed = cache.get(cache_key)
        if allowed is None:
            allowed = all_mask
            remaining = chunk
            for k in range(start, stop):
                e = remaining & small_mask
                remaining >>= context.bits
                pos = bisect_right(index.values[k], e) - 1
                if pos < 0:
                    allowed = 0
                    break
                allowed &= index.prefix[k][pos]
                if not allowed:
                    break
            if len(cache) >= 2048:
                cache.clear()
            cache[cache_key] = allowed
        mask &= allowed
        if not mask:
            return None
        block += 1
    while mask:
        bit = mask & -mask
        bucket = index.payloads[bit.bit_length() - 1]
        if bucket:
            i = bucket[0]
            rank = solver.reducer_rank[i]
            if chosen is None or rank < best_rank:
                chosen, best_rank = i, rank
        mask ^= bit
    if chosen is None:
        return None
    # 'oldest' buckets are insertion ordered, but their reducer_rank field
    # intentionally follows the sparse policy; compare actual IDs instead.
    # It is excluded by the caller, as are all non-regular-first policies.
    if solver.ratio_keys[chosen][0] >= solver.order.rank[signature.index]:
        return None
    c = solver.leading[chosen][1]
    if coefficient % c:
        return None
    solver.stats.counters['reducer_queries'] += 1
    solver.stats.counters['reducer_candidates'] += 1
    solver.stats.counters['native_frozen_reducer_shortcuts'] += 1
    return chosen, coefficient // c, None, 'regular'


include "_reducer_rows.pxi"


cdef extern from *:
    """
    #if defined(GVWZZ_HAVE_GMP) && GVWZZ_HAVE_GMP
    // Private, bounded chunks keep the exact selector and arithmetic in C++.
    // The caller retains every cached term owner and the solver throughout.
    template <class Rows, class RatioKey, class GeometryKey>
    class GvwzzBulkReductionImpl {
        GvwzzSparseAccumulator *work;
        Rows *rows;
        std::vector<const GvwzzEncodedTerms *> terms;
        std::vector<uint64_t> leading;
        GvwzzNativeReducerChoice pending_choice;
        uint64_t pending_packed = 0;
        bool pending = false;
        mpz_t signature;
        unsigned work_bits, ratio_bits, n, high;
        uint64_t high_unit, mask;
        RatioKey signature_value;
        Py_ssize_t signature_rank;
        bool tail, euclidean, signature_first;
        int policy;
    public:
        unsigned long long counts[11] = {};
        GvwzzBulkReductionImpl(GvwzzSparseAccumulator *work_, Rows *rows_,
                PyObject *signature_, unsigned wb, unsigned rb, unsigned dimensions,
                RatioKey sv, Py_ssize_t sr, bool t, bool e, int p, bool sf, size_t count)
            : work(work_), rows(rows_), terms(count, NULL), leading(count, 0),
              work_bits(wb), ratio_bits(rb), n(dimensions),
              high(wb * dimensions), high_unit(UINT64_C(1) << high),
              mask((UINT64_C(1) << wb) - 1), signature_value(sv),
              signature_rank(sr), tail(t), euclidean(e), signature_first(sf), policy(p) {
            mpz_init_set(signature, MPZ(signature_));
        }
        GvwzzBulkReductionImpl(const GvwzzBulkReductionImpl &) = delete;
        GvwzzBulkReductionImpl &operator=(const GvwzzBulkReductionImpl &) = delete;
        ~GvwzzBulkReductionImpl() { mpz_clear(signature); }
        void set_terms(size_t id, const GvwzzEncodedTerms *value, uint64_t lm) {
            terms[id] = value;
            leading[id] = lm;
        }
        Py_ssize_t requested_id() const { return pending_choice.id; }
        int run_chunk(size_t iterations) {
            try {
                while (true) {
                    if (!pending && counts[0] >= iterations) return 0;
                    uint64_t packed;
                    GvwzzNativeReducerChoice choice;
                    if (pending) {
                        packed = pending_packed;
                        choice = pending_choice;
                        pending = false;
                    } else {
                        work->prune();
                        if (work->empty()) return 1;
                        packed = work->top();
                        const uint64_t degree = (packed + high_unit - 1) >> high;
                        uint64_t digits = (degree << high) - packed;
                        RatioKey ratio_digits(0);
                        GeometryKey guarded(0);
                        for (unsigned k = 0; k < n; ++k) {
                            const uint64_t digit = digits & mask;
                            digits >>= work_bits;
                            ratio_digits = ratio_digits | (RatioKey(digit) << (k * ratio_bits));
                            guarded = guarded | (GeometryKey(digit) << (k * (ratio_bits + 1)));
                        }
                        const RatioKey encoded = (RatioKey(degree) << (ratio_bits * n)) - ratio_digits;
                        choice = rows->select_geometry_native(guarded,
                            signature_value - encoded, signature_rank,
                            work->top_coefficient(), signature, true, euclidean,
                            policy, signature_first);
                        ++counts[0];
                        counts[1] += choice.candidates;
                        counts[2] += choice.skipped;
                        if (choice.id < 0) {
                            if (!tail) return 1;
                            work->finish();
                            continue;
                        }
                    }
                    const size_t id = static_cast<size_t>(choice.id);
                    if (id >= terms.size()) {
                        PyErr_SetString(PyExc_AssertionError, "Missing bounded bulk reducer");
                        return -1;
                    }
                    if (terms[id] == NULL) {
                        pending_choice = choice;
                        pending_packed = packed;
                        pending = true;
                        return 2;
                    }
                    const bool is_euclidean = choice.kind > 10;
                    const int kind = is_euclidean ? choice.kind - 10 : choice.kind;
                    if (kind == 2)
                        mpz_submul(signature, choice.scalar, rows->row_signature(choice.id));
                    else if (kind != 1) {
                        PyErr_SetString(PyExc_AssertionError, "Super reductions must never be executed");
                        return -1;
                    }
                    if (work->subtract_terms_mpz(*terms[id], packed - leading[id], choice.scalar) < 0)
                        return -1;
                    if (tail) {
                        ++counts[8];
                        ++counts[is_euclidean ? 10 : 9];
                    } else if (is_euclidean) {
                        ++counts[5];
                        ++counts[kind == 1 ? 6 : 7];
                    } else ++counts[kind == 1 ? 3 : 4];
                }
            } catch (const std::bad_alloc &) {
                PyErr_NoMemory();
                return -1;
            }
        }
        PyObject *signature_integer() const {
            MPZ_Object *temporary = GMPy_MPZ_New(NULL);
            if (temporary == NULL) return NULL;
            mpz_set(temporary->z, signature);
            PyObject *result = PyNumber_Long(reinterpret_cast<PyObject *>(temporary));
            Py_DECREF(temporary);
            return result;
        }
    };
    #else
    template <class Rows, class RatioKey, class GeometryKey>
    class GvwzzBulkReductionImpl {
    public:
        unsigned long long counts[11] = {};
        GvwzzBulkReductionImpl(GvwzzSparseAccumulator *, Rows *, PyObject *,
            unsigned, unsigned, unsigned, RatioKey, Py_ssize_t, bool, bool, int, bool, size_t) {
            throw std::logic_error("GMP bulk reduction is unavailable");
        }
        void set_terms(size_t, const GvwzzEncodedTerms *, uint64_t) {}
        Py_ssize_t requested_id() const { return -1; }
        int run_chunk(size_t) { return 1; }
        PyObject *signature_integer() const { Py_RETURN_NONE; }
    };
    #endif
    typedef GvwzzBulkReductionImpl<GvwzzReducerRows, int64_t, uint64_t> GvwzzBulkReduction;
    typedef GvwzzBulkReductionImpl<GvwzzWideReducerRows, GvwzzWideKey, GvwzzWideKey> GvwzzWideBulkReduction;
    """
    cppclass GvwzzBulkReduction:
        GvwzzBulkReduction(GvwzzSparseAccumulator *, GvwzzReducerRows *, PyObject *,
            unsigned int, unsigned int, unsigned int, long long, Py_ssize_t,
            bint, bint, int, bint, size_t) except +
        unsigned long long counts[11]
        void set_terms(size_t, const GvwzzEncodedTerms *, uint64_t)
        Py_ssize_t requested_id()
        int run_chunk(size_t) except -1
        object signature_integer()
    cppclass GvwzzWideBulkReduction:
        GvwzzWideBulkReduction(GvwzzSparseAccumulator *, GvwzzWideReducerRows *, PyObject *,
            unsigned int, unsigned int, unsigned int, GvwzzWideKey, Py_ssize_t,
            bint, bint, int, bint, size_t) except +
        unsigned long long counts[11]
        void set_terms(size_t, const GvwzzEncodedTerms *, uint64_t)
        Py_ssize_t requested_id()
        int run_chunk(size_t) except -1
        object signature_integer()


include "_ratio_context.pxi"


cdef object _make_reducer_lookup(solver, signature, NativeWorkspace work):
    cdef _PackedRatioContext lookup = _PackedRatioContext(solver, signature, work.bound)
    return lookup if lookup.enabled else None


cdef object _select_gmp_reducer(solver, signature, tuple W, coeff,
                               _PackedRatioContext lookup, mode, bint allow_euclidean,
                               bint packed_shift, ids=None, target_key=None):
    cdef object native_coefficient, native_choice, i, scalar, rtype
    cdef object counters = solver.stats.counters
    cdef int native_policy
    cdef GvwzzWideKey wide_target
    native_coefficient = _gmp_integer(coeff)
    if lookup.signature_coefficient is not signature.coefficient:
        lookup.signature_coefficient = signature.coefficient
        lookup.native_signature_coefficient = _gmp_integer(signature.coefficient)
    native_policy = (0 if solver.options.reducer_policy == 'regular-first' else
                     1 if solver.options.reducer_policy == 'singular-first' else 2)
    if lookup.codec.wide:
        wide_target = lookup.wide_signature_value - lookup.codec.encode_wide(W)
        if ids is None:
            native_choice = lookup.codec.wide_rows.select_geometry(
                lookup.codec.guarded_wide(W), wide_target, lookup.signature_rank,
                <PyObject *>native_coefficient, <PyObject *>lookup.native_signature_coefficient,
                mode == 'mixed', allow_euclidean, native_policy,
                solver.options.euclidean_priority == 'signature-first')
        else:
            native_choice = lookup.codec.wide_rows.select(<PyObject *>ids,
                wide_target, lookup.signature_rank, <PyObject *>native_coefficient,
                <PyObject *>lookup.native_signature_coefficient, mode == 'mixed',
                allow_euclidean, native_policy,
                solver.options.euclidean_priority == 'signature-first')
    elif ids is None:
        native_choice = lookup.codec.rows.select_geometry(lookup.codec.guarded(W),
            lookup.signature_value - lookup.codec.encode(W), lookup.signature_rank,
            <PyObject *>native_coefficient, <PyObject *>lookup.native_signature_coefficient,
            mode == 'mixed', allow_euclidean, native_policy,
            solver.options.euclidean_priority == 'signature-first')
    else:
        native_choice = lookup.codec.rows.select(<PyObject *>ids, target_key[0],
            target_key[1], <PyObject *>native_coefficient,
            <PyObject *>lookup.native_signature_coefficient, mode == 'mixed',
            allow_euclidean, native_policy,
            solver.options.euclidean_priority == 'signature-first')
    if native_choice[3]:
        counters['reducer_candidates'] += native_choice[3]
    if native_choice[4]:
        counters['euclidean_sigdrop_skipped'] += native_choice[4]
    i, scalar = native_choice[0], native_choice[1]
    if i < 0:
        return None
    rtype = ('regular' if native_choice[2] == 1 else
             'singular' if native_choice[2] == 2 else
             'euclidean-regular' if native_choice[2] == 11 else 'euclidean-singular')
    return (i, scalar, None if packed_shift else _subtract(W, solver.leading[i][0]), rtype)


def find_reducer(solver, signature, leading, mode, bint allow_euclidean=False,
                 bint packed_shift=False, lookup_context=None):
    """Compiled version of the reference reducer policy, including ZZ steps."""
    cdef object W, coeff, cache_key, cached, target_key, buckets, ordered, ids
    cdef object saved = None, euclidean_choice = None, euclidean_score = None
    cdef object M, c, q, key, scalar, rem, b, quotient, rank, score, choice, rtype
    cdef object preferred, i
    cdef object geometry_state, geometry_cache, geometric_ids, geometry_entry, new_leading
    cdef Py_ssize_t new_id, axis
    cdef bint divides
    cdef object target_rank, reducer_rank, ratio_keys
    cdef object native_coefficient, native_choice
    cdef int native_policy
    cdef _PackedRatioContext lookup
    cdef bint packed_ratio = lookup_context is not None
    cdef bint native_selection
    cdef bint exact, pot = solver.order.name == 'pot', regular_signature
    counters = solver.stats.counters
    counters['reducer_queries'] += 1
    W, coeff = leading
    if packed_ratio:
        lookup = lookup_context
        packed_ratio = lookup.enabled
    ratio_keys = lookup.ratio_keys if packed_ratio else solver.ratio_keys
    native_selection = (packed_ratio and (lookup.codec.rows is not NULL
                        or lookup.codec.wide_rows is not NULL)
                        and mode in ('mixed', 'strict'))
    if native_selection and lookup.codec.fast_geometry:
        return _select_gmp_reducer(solver, signature, W, coeff, lookup, mode,
                                   allow_euclidean, packed_shift)
    cache_key = ((lookup.codec.bits, native_selection, signature.monomial, signature.index, W)
                 if packed_ratio else (signature.monomial, signature.index, W))
    cached = None if pot else solver.reducer_query_cache.get(cache_key)
    target_rank = solver.order.rank[signature.index] if pot else None
    if cached is not None:
        counters['reducer_geometry_cache_hits'] += 1
        solver.reducer_query_cache.move_to_end(cache_key)
        target_key, ids = cached
    else:
        if packed_ratio:
            target_key = ((gvwzz_wide_to_python(lookup.wide_signature_value
                          - lookup.codec.encode_wide(W), True)
                          if lookup.codec.wide else
                          lookup.signature_value - lookup.codec.encode(W)), lookup.signature_rank)
        else:
            target_key = (None if pot else solver.order.key_monomial(
                _subtract(signature.monomial, W), signature.index))
        if solver.order.name == 'gvw-g2' and solver.backend.order_name == 'grevlex':
            # Monomial divisibility does not depend on a candidate signature.
            # Reuse it across distinct signatures until the immutable basis grows.
            geometry_state = getattr(solver, '_native_reducer_geometry_append', None)
            if geometry_state is None:
                geometry_cache = OrderedDict()
                solver._native_reducer_geometry_append = geometry_cache
            else:
                geometry_cache = geometry_state
            geometry_entry = geometry_cache.get(W)
            geometric_ids = (None if geometry_entry is None
                or len(solver.P) - geometry_entry[0] > 8 else geometry_entry[1])
            if geometric_ids is None:
                buckets = _divisors(solver.reducer_index, W)
                # Sorting the exact divisor result in CPython avoids heapq's
                # Python generator/heap bookkeeping. Every policy includes a
                # unique insertion ID, preserving its total order.
                ordered = []
                for bucket in buckets:
                    ordered.extend(bucket)
                if solver.options.reducer_selection == 'oldest':
                    ordered.sort()
                else:
                    ordered.sort(key=solver.reducer_rank.__getitem__)
                geometric_ids = tuple(ordered)
                geometry_cache[W] = (len(solver.P), geometric_ids)
                if len(geometry_cache) > 8192:
                    geometry_cache.popitem(last=False)
            else:
                counters['native_monomial_divisor_cache_hits'] += 1
                if geometry_entry[0] != len(solver.P):
                    ordered = list(geometric_ids)
                    for new_id in range(geometry_entry[0], len(solver.P)):
                        new_leading = solver.leading[new_id][0]
                        divides = True
                        for axis in range(len(W)):
                            if new_leading[axis] > W[axis]:
                                divides = False
                                break
                        if divides:
                            ordered.append(new_id)
                    if solver.options.reducer_selection == 'oldest':
                        ordered.sort()
                    else:
                        ordered.sort(key=solver.reducer_rank.__getitem__)
                    geometric_ids = tuple(ordered)
                    geometry_cache[W] = (len(solver.P), geometric_ids)
                geometry_cache.move_to_end(W)
            if pot:
                ids = geometric_ids
            else:
                ids = (geometric_ids if native_selection else
                       tuple(i for i in geometric_ids if ratio_keys[i] <= target_key))
                solver.reducer_query_cache[cache_key] = (target_key, ids)
                if len(solver.reducer_query_cache) > 4096:
                    solver.reducer_query_cache.popitem(last=False)
        else:
            # Monomial divisibility does not depend on a candidate signature.
            # Reuse it across distinct signatures until the immutable basis grows.
            geometry_state = getattr(solver, '_native_reducer_geometry', None)
            if geometry_state is None or geometry_state[0] != len(solver.P):
                geometry_cache = OrderedDict()
                solver._native_reducer_geometry = (len(solver.P), geometry_cache)
            else:
                geometry_cache = geometry_state[1]
            geometric_ids = geometry_cache.get(W)
            if geometric_ids is None:
                buckets = _divisors(solver.reducer_index, W)
                # Sorting the exact divisor result in CPython avoids heapq's
                # Python generator/heap bookkeeping. Every policy includes a
                # unique insertion ID, preserving its total order.
                ordered = []
                for bucket in buckets:
                    ordered.extend(bucket)
                if solver.options.reducer_selection == 'oldest':
                    ordered.sort()
                else:
                    ordered.sort(key=solver.reducer_rank.__getitem__)
                geometric_ids = tuple(ordered)
                geometry_cache[W] = geometric_ids
                if len(geometry_cache) > 8192:
                    geometry_cache.popitem(last=False)
            else:
                counters['native_monomial_divisor_cache_hits'] += 1
                geometry_cache.move_to_end(W)
            if pot:
                ids = geometric_ids
            else:
                ids = (geometric_ids if native_selection else
                       tuple(i for i in geometric_ids if ratio_keys[i] <= target_key))
                solver.reducer_query_cache[cache_key] = (target_key, ids)
                if len(solver.reducer_query_cache) > 4096:
                    solver.reducer_query_cache.popitem(last=False)
    if native_selection:
        return _select_gmp_reducer(solver, signature, W, coeff, lookup, mode,
                                   allow_euclidean, packed_shift, ids, target_key)
    preferred = 'regular' if solver.options.reducer_policy == 'regular-first' else 'singular'
    for i in ids:
        key = ratio_keys[i]
        if pot:
            reducer_rank = key[0]
            if reducer_rank > target_rank:
                continue
            regular_signature = reducer_rank < target_rank
            if not regular_signature:
                if target_key is None:
                    target_key = solver.order.key_monomial(
                        _subtract(signature.monomial, W), signature.index)
                if key > target_key:
                    continue
                regular_signature = key < target_key
        else:
            regular_signature = key < target_key
        counters['reducer_candidates'] += 1
        M, c = solver.leading[i]
        exact = coeff % c == 0
        if not exact and not (mode == 'mixed' and allow_euclidean):
            continue
        q = solver.P[i]
        if exact:
            scalar = coeff // c
        else:
            b = abs(c)
            quotient = (coeff + b // 2) // b
            scalar = quotient if c > 0 else -quotient
            rem = coeff - scalar * c
            if not scalar or abs(rem) >= abs(coeff):
                continue
        if regular_signature:
            rtype = 'regular'
        elif scalar * q.signature.coefficient == signature.coefficient:
            rtype = 'super'
        else:
            rtype = 'singular'
        if not exact:
            if rtype == 'super':
                counters['euclidean_sigdrop_skipped'] += 1
                continue
            rank = (0 if rtype == preferred else 1) if solver.options.euclidean_priority == 'signature-first' else 0
            score = (rank, abs(rem), len(q.polynomial), abs(c).bit_length(), i)
            if euclidean_score is None or score < euclidean_score:
                euclidean_choice = (i, scalar, None if packed_shift else _subtract(W, M), 'euclidean-' + rtype)
                euclidean_score = score
            continue
        choice = (i, scalar, None if packed_shift else _subtract(W, M), rtype)
        if mode == 'strict':
            if rtype == 'regular':
                return choice
            continue
        if rtype == 'super':
            continue
        if solver.options.reducer_policy == 'first' or rtype == preferred:
            return choice
        if saved is None:
            saved = choice
    if (solver.options.euclidean_priority == 'signature-first' and saved is not None
            and euclidean_choice is not None and solver.options.reducer_policy != 'first'
            and euclidean_choice[3] == 'euclidean-' + preferred):
        return euclidean_choice
    return saved if saved is not None else euclidean_choice


_BULK_COUNTER_KEYS = ('reducer_queries', 'reducer_candidates', 'euclidean_sigdrop_skipped',
    'regular_steps', 'singular_steps', 'euclidean_steps', 'euclidean_regular_steps',
    'euclidean_singular_steps', 'tail_steps', 'tail_exact_steps', 'tail_euclidean_steps')


cdef void _flush_bulk_counts(unsigned long long *counts, counters) except *:
    cdef Py_ssize_t k
    for k in range(11):
        if counts[k]:
            counters[_BULK_COUNTER_KEYS[k]] += counts[k]
            counts[k] = 0


cdef object _bulk_reduce(solver, signature, NativeWorkspace work,
                        _PackedRatioContext lookup, bint tail):
    cdef GvwzzBulkReduction *driver = NULL
    cdef GvwzzWideBulkReduction *wide_driver = NULL
    cdef unsigned long long *counts = NULL
    cdef bint wide = lookup.codec.wide
    cdef _CompactTerms packed_terms
    cdef object native_signature = _gmp_integer(signature.coefficient)
    cdef object terms, degree, final_coefficient
    # Context caches may evict entries while this list is being assembled.
    # Retain the Python owner of every borrowed native pointer until exit.
    cdef list owners = []
    cdef Py_ssize_t i, loaded = 0
    cdef int done, policy = (0 if solver.options.reducer_policy == 'regular-first' else
                            1 if solver.options.reducer_policy == 'singular-first' else 2)
    cdef object counters = solver.stats.counters
    cdef bint progress = bool(solver.options.progress_interval)
    try:
        if wide:
            wide_driver = new GvwzzWideBulkReduction(work.compact, lookup.codec.wide_rows,
                <PyObject *>native_signature, work.context.bits, lookup.codec.bits,
                work.context.n, lookup.wide_signature_value, lookup.signature_rank,
                tail, solver.options.euclidean, policy,
                solver.options.euclidean_priority == 'signature-first', len(solver.P))
            counts = wide_driver.counts
            counters['native_wide_bulk_reduction_workspaces'] += 1
        else:
            driver = new GvwzzBulkReduction(work.compact, lookup.codec.rows,
                <PyObject *>native_signature, work.context.bits, lookup.codec.bits,
                work.context.n, lookup.signature_value, lookup.signature_rank,
                tail, solver.options.euclidean, policy,
                solver.options.euclidean_priority == 'signature-first', len(solver.P))
            counts = driver.counts
        counters['native_bulk_reduction_workspaces'] += 1
        while True:
            done = wide_driver.run_chunk(1024) if wide else driver.run_chunk(1024)
            if done == 2:
                # Resolve only reducers actually selected in this workspace.
                # The pending choice owns no workspace coefficient pointer;
                # its scalar borrows unchanged selector scratch until resume.
                i = wide_driver.requested_id() if wide else driver.requested_id()
                terms, degree = work.context.get_terms(solver.P[i].polynomial)
                packed_terms = terms
                owners.append(terms)
                if wide:
                    wide_driver.set_terms(i, packed_terms.packed,
                        work.context.encode(solver.leading[i][0]))
                else:
                    driver.set_terms(i, packed_terms.packed,
                        work.context.encode(solver.leading[i][0]))
                loaded += 1
                continue
            _flush_bulk_counts(counts, counters)
            # Flush completed work before either callback can interrupt.
            PyErr_CheckSignals()
            if progress:
                solver._check_budget()
            if done:
                break
        final_coefficient = (wide_driver.signature_integer() if wide
                             else driver.signature_integer())
        return (signature if final_coefficient == signature.coefficient else
                Signature(final_coefficient, signature.monomial, signature.index))
    finally:
        if counts is not NULL:
            try:
                _flush_bulk_counts(counts, counters)
                counters['native_bulk_terms_loaded'] += loaded
            finally:
                if wide_driver is not NULL:
                    del wide_driver
                if driver is not NULL:
                    del driver


def reduce_working(solver, label, bint tail=False, bint first_strict=False):
    cdef NativeWorkspace work = NativeWorkspace(solver.backend, label.polynomial)
    cdef object sig = label.signature, leading, reducer
    cdef object lookup_context = _make_reducer_lookup(solver, sig, work)
    cdef Py_ssize_t steps = 0
    cdef object tail_cap = solver.options.tail_step_cap if tail else 0
    cdef bint fast_step = (not solver.options.debug and solver.witnesses is None
                          and solver.paths is None
                          and solver.options.max_polynomial_terms is None
                          and solver.options.max_coefficient_bits is None)
    cdef bint check_budget = (solver.deadline is not None or solver.options.progress_interval
                             or solver.options.max_candidates is not None
                             or solver.options.max_reduction_steps is not None)
    cdef bint euclidean
    cdef object i, coefficient, kind, q, terms, degree, mu, packed_leading
    cdef bint frozen_shortcut = (fast_step and solver.order.name == 'pot'
        and solver.backend.order_name == 'grevlex'
        and solver.order.rank[sig.index] > 0
        and solver.options.reducer_policy in ('regular-first', 'first')
        and solver.options.reducer_selection != 'oldest')
    cdef object counters = solver.stats.counters
    solver.stats.counters['native_reduction_workspaces'] += 1
    if lookup_context is not None and lookup_context.native_geometry_enabled:
        counters['native_gmp_geometry_workspaces'] += 1
    if solver.options.reduction_kernel == 'packed':
        solver.stats.counters['packed_workspaces'] += 1
    work.height_upper = max((abs(int(c)).bit_length()
                            for c in label.polynomial.values()), default=0)
    if tail:
        work.finish_term()
    if first_strict:
        # J-pairs must begin with an exact, strictly signature-decreasing
        # reduction. Execute that step in the same private accumulator,
        # avoiding a temporary product and a second polynomial copy.
        reducer = find_reducer(solver, sig, work.leading(), 'strict', False, False,
                               lookup_context)
        if reducer is None:
            raise AssertionError('Every retained regular J-pair must have a strict first reducer')
        sig = solver._workspace_step(sig, work, reducer)
    if (fast_step and work.small_keys and lookup_context is not None
            and lookup_context.native_geometry_enabled
            and not solver.options.tail_step_cap
            and solver.deadline is None
            and solver.options.max_seconds is None
            and solver.options.max_basis is None
            and solver.options.max_candidates is None
            and solver.options.max_reduction_steps is None):
        sig = _bulk_reduce(solver, sig, work, lookup_context, tail)
        return Label(sig, work.result())
    while True:
        packed_leading = work.peek()
        if packed_leading is None:
            break
        if check_budget:
            solver._check_budget()
        if tail_cap and steps >= tail_cap:
            solver.stats.counters['tail_cap_reached'] += 1
            break
        reducer = (_first_frozen_reducer(solver, work, packed_leading[0],
                    packed_leading[1], sig) if frozen_shortcut else None)
        if reducer is None:
            leading = (work.context.decode(packed_leading[0]), packed_leading[1])
            reducer = find_reducer(solver, sig, leading, 'mixed', solver.options.euclidean,
                                   fast_step, lookup_context)
        if reducer is None:
            if not tail:
                break
            work.finish_term()
            continue
        if fast_step:
            i, coefficient, _, kind = reducer
            euclidean = kind.startswith('euclidean-')
            if euclidean:
                kind = kind[10:]
            q = solver.P[i]
            if kind == 'singular':
                sig = Signature(sig.coefficient - coefficient * q.signature.coefficient,
                                sig.monomial, sig.index)
            elif kind != 'regular':
                raise AssertionError('Super reductions must never be executed')
            # The encoding is additive. The graded-order reducer invariant
            # proves that every shifted term stays inside this degree bound.
            # Avoid reconstructing and re-encoding a full exponent tuple.
            mu = packed_leading[0] - work.context.encode(solver.leading[i][0])
            terms, degree = work.context.get_terms(q.polynomial)
            work.subtract_encoded(terms, coefficient, mu)
            if tail:
                counters['tail_steps'] += 1
                counters['tail_euclidean_steps' if euclidean else 'tail_exact_steps'] += 1
            elif euclidean:
                counters['euclidean_steps'] += 1
                counters['euclidean_' + kind + '_steps'] += 1
            else:
                counters[kind + '_steps'] += 1
        else:
            sig = solver._workspace_step(sig, work, reducer, tail=tail)
        steps += 1
    return Label(sig, work.result())
