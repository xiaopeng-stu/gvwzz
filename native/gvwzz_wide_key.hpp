#ifndef GVWZZ_WIDE_KEY_HPP
#define GVWZZ_WIDE_KEY_HPP

#include <Python.h>
#include <stdint.h>
#include <stddef.h>

// Portable two-limb integer used only for bounded monomial encodings.
// Polynomial and signature coefficients always retain arbitrary precision.
// Unsigned operations are modulo 2^128, including subtraction of signed
// encoded ratios. Callers prove that signed ratios lie in [-2^127, 2^127).
struct GvwzzWideKey {
    uint64_t lo, hi;
    GvwzzWideKey() : lo(0), hi(0) {}
    GvwzzWideKey(uint64_t low) : lo(low), hi(0) {}
    GvwzzWideKey(uint64_t low, uint64_t high) : lo(low), hi(high) {}

    GvwzzWideKey operator+(const GvwzzWideKey &other) const {
        const uint64_t low = lo + other.lo;
        return GvwzzWideKey(low, hi + other.hi + (low < lo));
    }
    GvwzzWideKey operator-(const GvwzzWideKey &other) const {
        return GvwzzWideKey(lo - other.lo, hi - other.hi - (lo < other.lo));
    }
    GvwzzWideKey operator&(const GvwzzWideKey &other) const {
        return GvwzzWideKey(lo & other.lo, hi & other.hi);
    }
    GvwzzWideKey operator|(const GvwzzWideKey &other) const {
        return GvwzzWideKey(lo | other.lo, hi | other.hi);
    }
    GvwzzWideKey operator<<(unsigned amount) const {
        if (amount >= 128) return GvwzzWideKey();
        if (amount >= 64) return GvwzzWideKey(0, lo << (amount - 64));
        if (!amount) return *this;
        return GvwzzWideKey(lo << amount, (hi << amount) | (lo >> (64 - amount)));
    }
    GvwzzWideKey operator>>(unsigned amount) const {
        if (amount >= 128) return GvwzzWideKey();
        if (amount >= 64) return GvwzzWideKey(hi >> (amount - 64));
        if (!amount) return *this;
        return GvwzzWideKey((lo >> amount) | (hi << (64 - amount)), hi >> amount);
    }
    bool operator==(const GvwzzWideKey &other) const {
        return hi == other.hi && lo == other.lo;
    }
    bool operator!=(const GvwzzWideKey &other) const { return !(*this == other); }
    bool operator<(const GvwzzWideKey &other) const {
        return hi < other.hi || (hi == other.hi && lo < other.lo);
    }
};

inline bool gvwzz_wide_signed_less(const GvwzzWideKey &left,
                                   const GvwzzWideKey &right) {
    const uint64_t sign = UINT64_C(1) << 63;
    const uint64_t left_high = left.hi ^ sign, right_high = right.hi ^ sign;
    return left_high < right_high || (left_high == right_high && left.lo < right.lo);
}

inline bool gvwzz_ratio_less(int64_t left, int64_t right) { return left < right; }
inline bool gvwzz_ratio_less(const GvwzzWideKey &left, const GvwzzWideKey &right) {
    return gvwzz_wide_signed_less(left, right);
}

struct GvwzzWideHash {
    size_t operator()(const GvwzzWideKey &key) const {
        uint64_t value = key.lo ^ (key.hi + UINT64_C(0x9e3779b97f4a7c15)
                                       + (key.lo << 6) + (key.lo >> 2));
        value ^= value >> 33;
        value *= UINT64_C(0xff51afd7ed558ccd);
        value ^= value >> 33;
        value *= UINT64_C(0xc4ceb9fe1a85ec53);
        return static_cast<size_t>(value ^ (value >> 33));
    }
};

// The caller has already checked the encoding bound. Public CPython numeric
// APIs keep this compatible with both Python 3.12 and Python 3.13 on Windows.
inline int gvwzz_wide_from_python(PyObject *value, GvwzzWideKey *result) {
    const uint64_t low = PyLong_AsUnsignedLongLongMask(value);
    if (PyErr_Occurred()) return -1;
    PyObject *shift = PyLong_FromLong(64);
    if (!shift) return -1;
    PyObject *upper = PyNumber_Rshift(value, shift);
    Py_DECREF(shift);
    if (!upper) return -1;
    const uint64_t high = PyLong_AsUnsignedLongLongMask(upper);
    Py_DECREF(upper);
    if (PyErr_Occurred()) return -1;
    *result = GvwzzWideKey(low, high);
    return 0;
}

inline PyObject *gvwzz_wide_to_python(const GvwzzWideKey &value, bool signed_value) {
    PyObject *high = (signed_value && (value.hi >> 63))
        ? PyLong_FromLongLong(-1 - static_cast<int64_t>(~value.hi))
        : PyLong_FromUnsignedLongLong(value.hi);
    if (!high) return NULL;
    PyObject *shift = PyLong_FromLong(64);
    if (!shift) { Py_DECREF(high); return NULL; }
    PyObject *upper = PyNumber_Lshift(high, shift);
    Py_DECREF(high); Py_DECREF(shift);
    if (!upper) return NULL;
    PyObject *low = PyLong_FromUnsignedLongLong(value.lo);
    if (!low) { Py_DECREF(upper); return NULL; }
    PyObject *result = PyNumber_Add(upper, low);
    Py_DECREF(upper); Py_DECREF(low);
    return result;
}

#endif
