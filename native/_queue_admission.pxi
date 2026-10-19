"""Compact J admission and exact bound checks for the fused parent row."""

include "_packed_candidate.pxi"
include "_candidate_heap.pxi"

from heapq import heappush as _packed_queue_heappush


cdef class NativeQueueAdmission:
    """Cache stable queue references once per row, with exact overflow fallback.

    CandidateQueue owns every retained object and every heap/dedup mutation.
    This helper only compiles its existing packed admission procedure; it does
    not change comparisons, candidate identity, or polynomial materialization.
    """
    cdef object queue, candidate_type, stats, counts
    cdef object leading_key, signature_key, fallback
    cdef dict active, pending, settled
    cdef object heap
    cdef NativeCandidateHeap native_heap
    cdef Py_ssize_t nvars
    cdef int tie_policy
    cdef bint deduplicate, lazy, g_first

    def __init__(self, queue):
        from gvwzz.structures import PackedCandidate
        self.queue = queue
        self.candidate_type = PackedCandidate
        self.stats, self.counts = queue.stats, queue.stats.counters
        self.active, self.pending, self.heap = queue.active, queue.pending_j, queue.heap
        self.settled = queue.settled_j_bounds
        self.native_heap = self.heap if isinstance(self.heap, NativeCandidateHeap) else None
        self.leading_key, self.signature_key = queue._packed_leading_key, queue._packed_signature_key
        self.fallback = queue._insert_j_packed
        self.nvars = queue.nvars
        self.deduplicate, self.lazy, self.g_first = queue.deduplicate, queue.lazy, queue.g_first
        self.tie_policy = (1 if queue.tie_policy == 'small-coeff'
                           else 2 if queue.tie_policy == 'small-lm' else 0)

    cdef bint accept_bound(self, object dkey, object W):
        """The signed-coefficient dictionary remains the exact authority."""
        cdef object settled, pending, key, pending_key
        if not self.queue._compact_order:
            return self.queue._accept_j_bound(dkey, None, W)
        self.counts['native_queue_dedup_checks'] += 1
        settled = self.settled.get(dkey)
        pending = self.pending.get(dkey)
        if settled is None and pending is None:
            return True
        key = self.leading_key(W)
        if settled is not None and settled <= key:
            self.counts['j_dedup_discarded'] += 1
            self.counts['j_dedup_settled'] += 1
            return False
        if pending is not None:
            pending_key = ((<NativePackedCandidate>pending)._leading_key
                           if isinstance(pending, NativePackedCandidate)
                           else pending._leading_key)
            if pending_key <= key:
                self.counts['j_dedup_discarded'] += 1
                return False
            del self.active[pending.token]
            del self.pending[dkey]
            self.counts['j_dedup_replaced'] += 1
        return True

    cdef object insert(self, object coefficient, object monomial,
                       Py_ssize_t index, object W, object factory):
        cdef object dkey, token, entry, signature_key, tie, magnitude, leading = None
        cdef NativePackedCandidate packed_entry
        cdef Py_ssize_t size
        # A preceding Python fallback in this row can replace the heap and
        # convert all scalar keys. Continue through that same exact path.
        if not self.queue._compact_order:
            return self.fallback(coefficient, monomial, index, W, factory)
        dkey = (coefficient, monomial, index)
        if self.deduplicate and self.queue._native_j_membership is not None:
            self.queue._native_j_membership.add_signature(coefficient, monomial, index)
        token = self.queue.serial + 1
        self.queue.serial = token
        if self.deduplicate or self.tie_policy:
            leading = self.leading_key(W)
        if self.candidate_type is NativePackedCandidate:
            # Construction stays in Cython; all unused lazy fields start None.
            packed_entry = NativePackedCandidate.__new__(NativePackedCandidate)
            packed_entry.token = token
            packed_entry._packed_signature, packed_entry._packed_leading = dkey, W
            packed_entry._nvars, packed_entry._stats = self.nvars, self.stats
            packed_entry._factory, packed_entry.kind = factory, 'J'
            packed_entry._dedup_key = dkey if self.deduplicate else None
            packed_entry._leading_key = leading
            entry = packed_entry
        else:
            entry = self.candidate_type(token, dkey, W, self.nvars, self.stats, None, factory)
            if self.deduplicate:
                entry._dedup_key = dkey
            entry._leading_key = leading
        if not self.lazy:
            _ = entry.label
        signature_key = self.signature_key(monomial, index)
        if self.tie_policy:
            magnitude = abs(coefficient)
            if self.tie_policy == 1:
                if self.native_heap is not None:
                    self.native_heap.push_fields(signature_key, <int>self.g_first,
                                                 magnitude, leading, magnitude, token)
                else:
                    tie = (magnitude, leading, magnitude)
            else:
                if self.native_heap is not None:
                    self.native_heap.push_fields(signature_key, <int>self.g_first,
                                                 leading, magnitude, magnitude, token)
                else:
                    tie = (leading, magnitude, magnitude)
        else:
            if self.native_heap is not None:
                self.native_heap.push_fields(signature_key, <int>self.g_first, 0, 0, 0, token)
            else:
                tie = ()
        if self.native_heap is None:
            _packed_queue_heappush(self.heap, (signature_key, <int>self.g_first, tie, token))
        else:
            self.counts['native_queue_pushes'] += 1
        self.active[token] = entry
        if self.deduplicate:
            self.pending[dkey] = entry
        self.counts['j_enqueued'] += 1
        self.counts['candidate_signature_objects_deferred'] += 1
        size = len(self.active)
        if size > self.counts['queue_peak']:
            self.counts['queue_peak'] = size
        return entry
