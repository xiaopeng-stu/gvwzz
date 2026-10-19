# cython: language_level=3, boundscheck=False, wraparound=False
"""Bounded queue selection with exact cover checks before descriptor expansion.

P and S cannot change inside this loop. A covered J is settled in the same
heap order as the reference solver; G and uncovered J candidates are returned
immediately for ordinary processing. No polynomial is constructed here.
"""
from time import perf_counter_ns
from gvwzz._native_indices cimport PackedSyzygyQueries


def select_candidate(engine, Py_ssize_t max_pops=256):
    """Return (candidate, cover_checked), or (None, False) after a bounded batch.

The caller enables this only for the ordinary non-debug indexed-cover engine.
Every pop and cover call retains its exclusive timing bucket. Budget/progress
checks run at the same cooperative boundary as the reference loop.
"""
    if max_pops < 1:
        raise ValueError('max_pops must be positive')
    cdef object queue = engine.queue
    cdef object counts = engine.stats.counters
    cdef object times = engine.stats.times_ns
    cdef object candidate, packed, tick
    cdef bint covered
    cdef Py_ssize_t popped
    cdef PackedSyzygyQueries syzygies = engine._native_queue_syzygy
    if syzygies is None:
        syzygies = PackedSyzygyQueries(engine.S)
        engine._native_queue_syzygy = syzygies
    else:
        syzygies.refresh()
    counts['native_queue_drain_calls'] += 1
    try:
        for popped in range(max_pops):
            if not queue:
                return None, False
            engine._check_budget()
            tick = perf_counter_ns()
            try:
                candidate = queue.pop()
            finally:
                times['queue'] += perf_counter_ns() - tick
            counts['iterations'] += 1
            counts['native_queue_drain_pops'] += 1
            packed = getattr(candidate, '_packed_signature', None)
            if candidate.kind != 'J' or packed is None:
                return candidate, False
            tick = perf_counter_ns()
            try:
                covered = syzygies.query(packed[0], packed[1], packed[2])
                if covered:
                    counts['cover_calls'] += 1
                    counts['cover_gcd_updates'] += 1
                    counts['packed_queue_cover_rejections'] += 1
                else:
                    covered = engine.cover.check_packed(
                        packed[0], packed[1], packed[2],
                        candidate.packed_leading_monomial,
                        unit_syzygy_checked=True)
            finally:
                times['cover'] += perf_counter_ns() - tick
            if not covered:
                return candidate, True
            counts['cover_rejected_j'] += 1
            counts['native_queue_drain_covered'] += 1
            queue.settle_j(candidate)
            # Release the recipe before selecting the next candidate. This matches
            # the reference loop's lifetime apart from its single local reference.
            candidate = None
        return None, False
    finally:
        syzygies.flush()
