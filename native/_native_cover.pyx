# cython: language_level=3, boundscheck=False, wraparound=False
"""Exact P-cover witnesses, with dynamically sized exponent coordinates.

The byte encoding is checked at the boundary. Its size is the ring dimension,
not a machine-word limit. Big integer coefficients stay Python integers.
This accelerator only proves positive P covers; callers retain the full S/P
multi-cover test whenever the native subset does not establish a cover.
"""
from libc.stdint cimport uint64_t
from libcpp.vector cimport vector
from cpython.bytes cimport PyBytes_AS_STRING
from cpython.long cimport PyLong_CheckExact
from math import gcd
from gvwzz.core import exact_integer


cdef extern from *:
    """
    #include <algorithm>
    #include <cstdint>
    #include <limits>
    #include <utility>
    #include <vector>
    struct GVWCoverRow {
        std::vector<unsigned char> monomial;
        std::vector<short> delta;
        int64_t degree = 0;
        uint64_t coefficient = 0;
        bool small = false;
    };
    struct GVWCoverColumn {
        std::vector<unsigned char> values;
        std::vector<std::vector<uint64_t>> masks;
        void insert(unsigned char exponent, size_t row) {
            if (!exponent) return;
            auto it = std::lower_bound(values.begin(), values.end(), exponent);
            size_t pos = size_t(it - values.begin());
            if (it == values.end() || *it != exponent) {
                std::vector<uint64_t> inherited;
                if (pos < masks.size()) inherited = masks[pos];
                values.insert(it, exponent);
                masks.insert(masks.begin() + pos, std::move(inherited));
            }
            const size_t word = row / 64;
            const uint64_t bit = UINT64_C(1) << (row % 64);
            for (size_t i = 0; i <= pos; ++i) {
                if (masks[i].size() <= word) masks[i].resize(word + 1, 0);
                masks[i][word] |= bit;
            }
        }
    };
    struct GVWCoverComponent {
        std::vector<GVWCoverRow> rows;
        std::vector<GVWCoverColumn> columns;
        std::vector<GVWCoverRow> witnesses;
        size_t next_witness = 0;
    };
    struct GVWCoverState {
        size_t nvars;
        int order;
        std::vector<GVWCoverComponent> components;
        GVWCoverRow query, proof;
        std::vector<uint64_t> mask;
        size_t mask_word = 0;
        uint64_t word_bits = 0;
        bool proof_empty = true;
        explicit GVWCoverState(size_t n, size_t rank, int ord)
            : nvars(n), order(ord), components(rank) {}
        bool less(const GVWCoverRow& a, const GVWCoverRow& b) const {
            if (order && a.degree != b.degree) return a.degree < b.degree;
            if (order == 2) {
                for (size_t i = nvars; i > 0; --i)
                    if (a.delta[i - 1] != b.delta[i - 1])
                        return a.delta[i - 1] > b.delta[i - 1];
            } else {
                for (size_t i = 0; i < nvars; ++i)
                    if (a.delta[i] != b.delta[i]) return a.delta[i] < b.delta[i];
            }
            return false;
        }
        bool divides(const GVWCoverRow& row) const {
            for (size_t k = 0; k < nvars; ++k)
                if (row.monomial[k] > query.monomial[k]) return false;
            return true;
        }
        GVWCoverRow make_row(const unsigned char* sig, const unsigned char* lm) const {
            GVWCoverRow row;
            row.monomial.assign(sig, sig + nvars);
            row.delta.resize(nvars);
            for (size_t k = 0; k < nvars; ++k) {
                row.delta[k] = short(lm[k]) - short(sig[k]);
                row.degree += row.delta[k];
            }
            return row;
        }
        void insert(size_t component, const unsigned char* sig,
                    const unsigned char* lm, bool small, uint64_t coefficient) {
            GVWCoverComponent& c = components[component];
            if (c.columns.empty()) c.columns.resize(nvars);
            GVWCoverRow row = make_row(sig, lm);
            row.small = small;
            row.coefficient = coefficient;
            const size_t slot = c.rows.size();
            for (size_t k = 0; k < nvars; ++k) c.columns[k].insert(sig[k], slot);
            c.rows.push_back(std::move(row));
        }
        void prepare(const unsigned char* sig, const unsigned char* lm) {
            // Reuse query storage for the millions of covered queue entries.
            query.monomial.resize(nvars);
            query.delta.resize(nvars);
            query.degree = 0;
            for (size_t k = 0; k < nvars; ++k) {
                query.monomial[k] = sig[k];
                query.delta[k] = short(lm[k]) - short(sig[k]);
                query.degree += query.delta[k];
            }
            proof_empty = true;
        }
        bool witness_matches(size_t component, size_t slot) const {
            const GVWCoverRow& row = components[component].witnesses[slot];
            return divides(row) && less(row, query);
        }
        size_t witness_count(size_t component) const {
            return components[component].witnesses.size();
        }
        void begin(size_t component) {
            const GVWCoverComponent& c = components[component];
            const size_t words = (c.rows.size() + 63) / 64;
            mask.assign(words, ~UINT64_C(0));
            if (words && c.rows.size() % 64)
                mask.back() = (UINT64_C(1) << (c.rows.size() % 64)) - 1;
            for (size_t k = 0; k < c.columns.size(); ++k) {
                const GVWCoverColumn& column = c.columns[k];
                const auto it = std::upper_bound(column.values.begin(), column.values.end(),
                                                  query.monomial[k]);
                if (it == column.values.end()) continue;
                const auto& excluded = column.masks[size_t(it - column.values.begin())];
                for (size_t w = 0; w < excluded.size(); ++w) mask[w] &= ~excluded[w];
            }
            mask_word = 0;
            word_bits = words ? mask[0] : 0;
        }
        static unsigned lowbit_index(uint64_t value) {
            // Portable to MSVC; avoid compiler-specific ctz intrinsics.
            unsigned result = 0;
            if (!(value & UINT64_C(0xffffffff))) { value >>= 32; result += 32; }
            if (!(value & UINT64_C(0xffff))) { value >>= 16; result += 16; }
            if (!(value & UINT64_C(0xff))) { value >>= 8; result += 8; }
            if (!(value & UINT64_C(0xf))) { value >>= 4; result += 4; }
            if (!(value & UINT64_C(0x3))) { value >>= 2; result += 2; }
            if (!(value & UINT64_C(0x1))) ++result;
            return result;
        }
        size_t next(size_t component) {
            while (mask_word < mask.size()) {
                if (!word_bits) {
                    if (++mask_word < mask.size()) word_bits = mask[mask_word];
                    continue;
                }
                const size_t slot = 64 * mask_word + lowbit_index(word_bits);
                word_bits &= word_bits - 1;
                if (less(components[component].rows[slot], query)) return slot;
            }
            return std::numeric_limits<size_t>::max();
        }
        void contribute(size_t component, size_t slot) {
            const GVWCoverRow& row = components[component].rows[slot];
            if (proof_empty) { proof = row; proof_empty = false; return; }
            for (size_t k = 0; k < nvars; ++k)
                proof.monomial[k] = std::max(proof.monomial[k], row.monomial[k]);
            if (less(proof, row)) { proof.delta = row.delta; proof.degree = row.degree; }
        }
        size_t remember(size_t component, bool small, uint64_t coefficient) {
            GVWCoverComponent& c = components[component];
            proof.small = small;
            proof.coefficient = coefficient;
            const size_t slot = c.next_witness;
            if (slot == c.witnesses.size()) c.witnesses.push_back(proof);
            else c.witnesses[slot] = proof;
            c.next_witness = (slot + 1) % 16;
            return slot;
        }
    };
    """
    cppclass GVWCoverRow:
        uint64_t coefficient
        bint small
    cppclass GVWCoverComponent:
        vector[GVWCoverRow] rows
        vector[GVWCoverRow] witnesses
    cppclass GVWCoverState:
        GVWCoverState(size_t, size_t, int) except +
        vector[GVWCoverComponent] components
        void insert(size_t, const unsigned char*, const unsigned char*, bint, uint64_t) except +
        void prepare(const unsigned char*, const unsigned char*) except +
        bint witness_matches(size_t, size_t) noexcept
        size_t witness_count(size_t) noexcept
        void begin(size_t) except +
        size_t next(size_t) noexcept
        void contribute(size_t, size_t) except +
        size_t remember(size_t, bint, uint64_t) except +


cdef uint64_t _gcd64(uint64_t a, uint64_t b) noexcept:
    cdef uint64_t r
    while b:
        r = a % b
        a, b = b, r
    return a


cdef class NativeCoverQuery:
    cdef GVWCoverState* state
    cdef Py_ssize_t nvars, rank
    cdef object coefficients, witness_coefficients
    cdef public unsigned long long queries, hits, witness_hits, rows

    def __cinit__(self, Py_ssize_t nvars, Py_ssize_t rank, order):
        self.state = NULL
        if nvars < 0 or rank < 0:
            raise ValueError('Negative cover dimension')
        self.nvars, self.rank = nvars, rank
        self.state = new GVWCoverState(nvars, rank, {'lex': 0, 'grlex': 1, 'grevlex': 2}[order])
        self.coefficients = [[] for _ in range(rank)]
        self.witness_coefficients = [[None] * 16 for _ in range(rank)]

    def __dealloc__(self):
        if self.state != NULL:
            del self.state

    def insert(self, coefficient, monomial, Py_ssize_t index, leading):
        cdef bytes sig, lm
        cdef bint small
        cdef uint64_t value
        if not 0 <= index < self.rank:
            raise ValueError('Wrong module component')
        if len(monomial) != self.nvars or len(leading) != self.nvars:
            raise ValueError('Wrong exponent-vector length')
        if any(e < 0 or e > 127 for e in monomial) or any(e < 0 or e > 127 for e in leading):
            return False
        coefficient = abs(coefficient if PyLong_CheckExact(coefficient)
                          else exact_integer(coefficient, 'signature coefficient'))
        if not coefficient:
            raise ValueError('Zero signature coefficient')
        sig, lm = bytes(monomial), bytes(leading)
        small = coefficient <= 0xffffffffffffffff
        value = <uint64_t>coefficient if small else 0
        self.state.insert(index, <const unsigned char*>PyBytes_AS_STRING(sig),
                          <const unsigned char*>PyBytes_AS_STRING(lm), small, value)
        self.coefficients[index].append(coefficient)
        self.rows += 1
        return True

    def positive(self, coefficient, packed_signature, Py_ssize_t index, packed_leading):
        """Prove a P-only multi-cover; False leaves the exact full query to the caller."""
        cdef bytes sig, lm
        cdef Py_ssize_t i
        cdef size_t slot, remembered
        cdef GVWCoverRow* row
        cdef bint small, small_acc = True
        cdef uint64_t value, acc64 = 0, next64
        cdef object acc = None, new_acc, magnitude
        if not 0 <= index < self.rank:
            raise ValueError('Wrong module component')
        if packed_signature < 0 or packed_leading < 0:
            return False
        try:
            sig = packed_signature.to_bytes(self.nvars, 'little')
            lm = packed_leading.to_bytes(self.nvars, 'little')
        except OverflowError:
            return False
        # Internal descriptors are already checked, but keep the native public
        # entry point safe if called directly by a diagnostic or extension.
        for i in range(self.nvars):
            if (<unsigned char*>PyBytes_AS_STRING(sig))[i] > 127 or (<unsigned char*>PyBytes_AS_STRING(lm))[i] > 127:
                return False
        magnitude = abs(coefficient if PyLong_CheckExact(coefficient)
                        else exact_integer(coefficient, 'signature coefficient'))
        if not magnitude:
            raise ValueError('Zero signature coefficient')
        self.queries += 1
        small = magnitude <= 0xffffffffffffffff
        value = <uint64_t>magnitude if small else 0
        self.state.prepare(<const unsigned char*>PyBytes_AS_STRING(sig),
                           <const unsigned char*>PyBytes_AS_STRING(lm))
        for slot in range(self.state.witness_count(index)):
            if not self.state.witness_matches(index, slot):
                continue
            row = &self.state.components[index].witnesses[slot]
            if row.small and small:
                if value % row.coefficient:
                    continue
            elif magnitude % self.witness_coefficients[index][slot]:
                continue
            self.hits += 1
            self.witness_hits += 1
            return True
        self.state.begin(index)
        while True:
            slot = self.state.next(index)
            if slot == <size_t>-1:
                return False
            row = &self.state.components[index].rows[slot]
            if small_acc and row.small:
                next64 = _gcd64(acc64, row.coefficient)
                if next64 == acc64:
                    continue
                acc64 = next64
                self.state.contribute(index, slot)
                if small:
                    if value % acc64:
                        continue
                elif magnitude % acc64:
                    continue
                acc = acc64
            else:
                if small_acc:
                    acc = acc64
                    small_acc = False
                new_acc = gcd(acc, self.coefficients[index][slot])
                if new_acc == acc:
                    continue
                acc = new_acc
                self.state.contribute(index, slot)
                if magnitude % acc:
                    continue
            small_acc = acc <= 0xffffffffffffffff
            remembered = self.state.remember(index, small_acc, <uint64_t>acc if small_acc else 0)
            self.witness_coefficients[index][remembered] = acc
            self.hits += 1
            return True
