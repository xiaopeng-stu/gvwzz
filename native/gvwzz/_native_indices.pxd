from libc.stdint cimport uint64_t


cdef class PackedSyzygyQueries:
    cdef object store, versions, coefficients, witness_coefficients
    cdef void* state
    cdef Py_ssize_t nvars
    cdef uint64_t guard, guard_high
    cdef public bint enabled
    cdef bint query64(self, object coefficient, uint64_t packed,
                      Py_ssize_t index) except -1
    cdef bint query128(self, object coefficient, uint64_t packed, uint64_t high,
                       Py_ssize_t index) except -1
    cpdef bint query(self, object coefficient, object packed,
                    Py_ssize_t index) except -1
    cpdef refresh(self, Py_ssize_t index=*)
    cpdef flush(self)
