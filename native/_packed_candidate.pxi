"""Lazy packed candidate storage; integer coordinates are never truncated."""

from gvwzz.core import Label as _QueueLabel
from gvwzz.monomial_packing import unpack_monomial as _queue_unpack


cdef class NativePackedCandidate:
    cdef public object token, _packed_signature, _packed_leading, _stats
    cdef public Py_ssize_t _nvars
    cdef public object leading_coefficient, _factory, kind, _label, _signature
    cdef public object _leading_monomial, representation_recipe, path_recipe
    cdef public object generating_parents, _dedup_key, _leading_key

    def __init__(self, token, _packed_signature, _packed_leading, _nvars, _stats,
                 leading_coefficient=None, _factory=None, kind='J', _label=None,
                 _signature=None, _leading_monomial=None, representation_recipe=None,
                 path_recipe=None, generating_parents=None, _dedup_key=None,
                 _leading_key=None):
        self.token = token
        self._packed_signature, self._packed_leading = _packed_signature, _packed_leading
        self._nvars, self._stats = _nvars, _stats
        self.leading_coefficient, self._factory = leading_coefficient, _factory
        self.kind, self._label, self._signature = kind, _label, _signature
        self._leading_monomial = _leading_monomial
        self.representation_recipe, self.path_recipe = representation_recipe, path_recipe
        self.generating_parents = generating_parents
        self._dedup_key, self._leading_key = _dedup_key, _leading_key

    @property
    def signature(self):
        if self._signature is None:
            coefficient, packed, index = self._packed_signature
            self._signature = _signature_from_arithmetic(
                coefficient, _queue_unpack(packed, self._nvars), index)
            self._stats.counters['candidate_signature_fast_builds'] += 1
        return self._signature

    @property
    def leading_monomial(self):
        if self._leading_monomial is None:
            self._leading_monomial = _queue_unpack(self._packed_leading, self._nvars)
        return self._leading_monomial

    @property
    def packed_signature(self):
        return self._packed_signature[1]

    @property
    def packed_leading_monomial(self):
        return self._packed_leading

    @property
    def signature_coefficient(self):
        return self._packed_signature[0]

    @property
    def signature_index(self):
        return self._packed_signature[2]

    @property
    def label(self):
        if self._label is None:
            if self._factory is None:
                raise RuntimeError('Missing polynomial recipe')
            self._label = _QueueLabel(self.signature, self._factory())
            self._factory = None
        return self._label
