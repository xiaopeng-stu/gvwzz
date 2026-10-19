"""Runtime data structures used by the solver.

The classes in this module are implementation aids only: counters, divisor
indices, the multi-cover index, the syzygy store, and the candidate queue.
They do not perform polynomial reduction.
"""
from __future__ import annotations

import heapq
import os as _native_os
from bisect import bisect_left, bisect_right, insort
from collections import Counter, OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from math import gcd
from typing import Any, Callable, Iterator

from .core import (
    Label, Monomial, Signature, divides_monomial, lcm_monomials,
    subtract_monomials, _signature_from_arithmetic,
)
from .monomial_packing import pack_monomial, unpack_monomial

try:
    if _native_os.environ.get('GVWZZ_NO_NATIVE'):
        raise ImportError('native accelerators disabled')
    from ._native_indices import has_divisor_packed as _native_has_divisor_packed
    from ._native_indices import divisor_mask_packed as _native_divisor_mask_packed
except ImportError:
    _native_has_divisor_packed = None
    _native_divisor_mask_packed = None

try:
    if _native_os.environ.get('GVWZZ_NO_NATIVE'):
        raise ImportError('native accelerators disabled')
    from ._native_cover import NativeCoverQuery as _NativeCoverQuery
except ImportError:
    _NativeCoverQuery = None

# g_generated includes all parent-pair G labels, including early rejections.
# g_omitted_comparable is retained for compatibility: omitted from the QUEUE,
# not from g_generated. It is one disjoint component of g_discarded.
COUNTERS = """parent_pairs j_generated g_generated g_discarded g_omitted_comparable
j_enqueued g_enqueued j_popped g_popped j_dedup_discarded j_dedup_replaced
j_dedup_settled queue_stale_entries queue_peak iterations
cover_calls cover_rejected_j cover_cache_hits
cover_bucket_queries cover_gcd_updates cover_naive_records
j_reduction_calls zero_reductions regular_steps singular_steps
reducer_queries reducer_candidates g_top_rejected g_inserted j_inserted
principal_syzygies_detected principal_syzygies_unknown
syzygy_insert_calls syzygy_seed_rejected syzygy_g_signatures
syzygy_g_skipped_old syzygy_g_skipped_seed syzygy_new_signatures
syzygy_simplified s_peak p_peak bezout_adjustments
euclidean_steps euclidean_regular_steps euclidean_singular_steps euclidean_sigdrop_skipped
tail_reduction_calls tail_steps tail_exact_steps tail_euclidean_steps
j_materialized g_materialized g_generation_tests g_generation_top_rejected
j_zero_reductions g_zero_reductions
top_cache_hits syzygy_divisor_cache_hits j_generation_tests
j_generation_syzygy_rejected
j_tail_reduction_calls g_tail_reduction_calls j_tail_steps g_tail_steps
j_tail_exact_steps g_tail_exact_steps j_tail_euclidean_steps g_tail_euclidean_steps
g_tail_skipped
syzygy_g_shortcuts syzygy_g_materialized
syzygy_index_appends syzygy_index_rebuilds syzygy_index_records_written
syzygy_multiple_queries syzygy_multiple_candidates syzygy_component_reuses
syzygy_index_deletions syzygy_index_compactions
syzygy_existence_queries syzygy_existence_cache_hits syzygy_unit_mask_queries
syzygy_existence_coefficients_tested
principal_syzygy_objects_avoided candidate_signature_fast_builds candidate_signature_objects_deferred
syzygy_recent_witness_hits""".split()

TIMERS = (
    "cover", "reduction", "g_top_test", "syzygy", "generation", "queue",
    "index", "g_generation_test", "materialization", "tail_reduction",
    "j_generation_test",
)


@dataclass
class Stats:
    counters: Counter = field(default_factory=Counter)
    times_ns: Counter = field(default_factory=Counter)
    total_ns: int = 0
    cpu_ns: int = 0

    def to_dict(self) -> dict:
        counts = {key: int(self.counters[key]) for key in COUNTERS}
        counts.update({key: int(value) for key, value in self.counters.items() if key not in counts})
        times = {key: self.times_ns[key] / 1e9 for key in TIMERS}
        times["total"] = self.total_ns / 1e9
        times["other"] = (self.total_ns - sum(self.times_ns[key] for key in TIMERS)) / 1e9
        times["process_cpu"] = self.cpu_ns / 1e9
        # These are a breakdown of tail_reduction, NOT extra exclusive sections.
        # Old saved runs have no such breakdown; do not invent zero subtotals.
        if 'j_tail_reduction' in self.times_ns or 'g_tail_reduction' in self.times_ns:
            for key in ('j_tail_reduction', 'g_tail_reduction'):
                times[key] = self.times_ns[key] / 1e9
        # These refine the existing syzygy total; they are not extra totals.
        if 'syzygy_check' in self.times_ns or 'syzygy_complete' in self.times_ns:
            for key in ('syzygy_check', 'syzygy_complete'):
                times[key] = self.times_ns[key] / 1e9
        if 'incremental_minimization' in self.times_ns:
            times['incremental_minimization'] = self.times_ns['incremental_minimization'] / 1e9
            times['other'] -= times['incremental_minimization']
        # Subsections of generation, not additional exclusive wall time.
        for key in ('frozen_seed_screen', 'frozen_multiplier_minimal'):
            if key in self.times_ns:
                times[key] = self.times_ns[key] / 1e9
        return {"counters": counts, "timings_seconds": times}


_PY_ONE = 1  # Force arbitrary-precision shifts in optional source builds.


class DivisorIndex:
    """Incremental exact divisor queries with Python-integer bitsets.

    Arithmetic on polynomials is unchanged. Each bit identifies an exponent
    vector. For variable k intersect the masks exponent[k] <= target[k].
    Insertion/update is deterministic; payload enumeration uses insertion IDs,
    not a monomial order (the public query does not promise an output order).
    """
    def __init__(self, nvars: int) -> None:
        self.nvars = nvars
        self.positions = {}
        self.payloads = []
        self.values = [[] for _ in range(nvars)]
        self.exact = [{} for _ in range(nvars)]
        self.prefix = [[] for _ in range(nvars)]
        self.dirty = False

    def set(self, monomial: Monomial, payload: Any) -> None:
        if len(monomial) != self.nvars:
            raise ValueError('Wrong exponent-vector length')
        i = self.positions.get(monomial)
        if i is not None:
            self.payloads[i] = payload
            return
        i = len(self.payloads)
        self.positions[monomial] = i
        self.payloads.append(payload)
        bit = _PY_ONE << i
        for k, e in enumerate(monomial):
            col = self.exact[k]
            if e not in col:
                insort(self.values[k], e)
                col[e] = bit
            else:
                col[e] |= bit
        self.dirty = True

    def _refresh(self):
        for k, vals in enumerate(self.values):
            acc = 0
            out = []
            col = self.exact[k]
            for e in vals:
                acc |= col[e]
                out.append(acc)
            self.prefix[k] = out
        self.dirty = False

    def divisors(self, target: Monomial) -> Iterator[Any]:
        if len(target) != self.nvars:
            raise ValueError('Wrong exponent-vector length')
        if self.dirty:
            self._refresh()
        mask = (_PY_ONE << len(self.payloads)) - 1
        for k, e in enumerate(target):
            pos = bisect_right(self.values[k], e) - 1
            if pos < 0:
                return
            mask &= self.prefix[k][pos]
            if not mask:
                return
        while mask:
            bit = mask & -mask
            payload = self.payloads[bit.bit_length()-1]
            if payload is not None:
                yield payload
            mask ^= bit


class SyzygyIndex:
    """Exact, two-way monomial index for one syzygy component.

    A bit identifies a distinct monomial, in insertion order. Only positive
    exponents have postings: masks[k][e] contains the monomials whose exponent
    in variable k is at least e. The stored thresholds are observed exponent
    values, NOT all integers up to the largest exponent. Zero coordinates cost
    no insertion work. All arithmetic uses Python integers (no bit-width cap).

    Insertions update just the affected postings. Divisor queries exclude
    exponents above the target; multiple queries intersect its positive
    constraints. Neither query scans the component's signature list.
    A deletion clears the row's active bit and releases its payload. Stale
    posting bits are always intersected with all_mask. Row IDs are not reused,
    so enumeration remains in active insertion order. Occasional compaction
    bounds tombstones without rebuilding on each deletion.
    """
    def __init__(self, nvars: int) -> None:
        self.nvars = nvars
        self.positions = {}
        self.monomials = []
        self.payloads = []
        # Incremental POT stages touch only a few module components. Keep the
        # ordinary index object/list interface, but allocate its coordinate
        # postings only on the first insertion into this particular component.
        self.values = []
        self.masks = []
        self.all_mask = 0
        # A conservative degree lower bound. Deletions need not rescan it:
        # retaining a lower value only forgoes shortcuts, never rejects a divisor.
        self.minimum_degree = None
        # Packed-query blocks cover eight byte-sized exponent coordinates.
        # Their cached masks combine the same exact threshold postings as the
        # tuple query. Small integer keys avoid repeatedly hashing 48-tuples.
        self.packed_blocks = []

    def set(self, monomial: Monomial, payload: Any) -> None:
        if len(monomial) != self.nvars:
            raise ValueError('Wrong exponent-vector length')
        i = self.positions.get(monomial)
        if i is not None:
            self.payloads[i] = payload
            return
        if not self.payloads:
            self.values = [[] for _ in range(self.nvars)]
            self.masks = [{} for _ in range(self.nvars)]
            self.packed_blocks = [(start, min(start + 8, self.nvars), {})
                                  for start in range(0, self.nvars, 8)]
        i = len(self.payloads)
        bit = _PY_ONE << i
        degree = sum(monomial)
        if self.minimum_degree is None or degree < self.minimum_degree:
            self.minimum_degree = degree
        self.positions[monomial] = i
        self.monomials.append(monomial)
        self.payloads.append(payload)
        self.all_mask |= bit
        # Every new row must be considered by every block, including blocks
        # whose coordinates in this row are all zero. Deletions need no clear:
        # the query always intersects the cached masks with active all_mask.
        for _, _, cache in self.packed_blocks:
            cache.clear()
        for k, e in enumerate(monomial):
            if not e:
                continue
            values, masks = self.values[k], self.masks[k]
            pos = bisect_left(values, e)
            if pos == len(values) or values[pos] != e:
                # Existing rows with exponent >= e are exactly those in the
                # next higher observed threshold. Do not lose them when a
                # newly inserted row introduces a threshold between old ones.
                masks[e] = masks[values[pos]] if pos < len(values) else 0
                values.insert(pos, e)
            for exponent in values[:pos + 1]:
                masks[exponent] |= bit

    def remove(self, monomial: Monomial) -> bool:
        """Remove one monomial bucket, without touching the other postings."""
        i = self.positions.pop(monomial, None)
        if i is None:
            return False
        self.all_mask &= ~(_PY_ONE << i)
        self.payloads[i] = None
        self.monomials[i] = None
        return True

    def needs_compaction(self) -> bool:
        # Amortize a rebuild against the deletions since the preceding one.
        # At most about twice the live row count plus 64 slots are retained.
        dead = len(self.payloads) - len(self.positions)
        return dead >= 64 and dead >= len(self.positions)

    def _divisor_mask(self, target: Monomial) -> int:
        if len(target) != self.nvars:
            raise ValueError('Wrong exponent-vector length')
        mask = self.all_mask
        if not mask:
            return 0
        degree = sum(target)
        if degree < self.minimum_degree:
            return 0
        if degree == self.minimum_degree:
            # Equal-degree monomials divide iff they are identical. The bound
            # may be stale after deletion, but active positions remain exact.
            position = self.positions.get(target)
            return 0 if position is None else (_PY_ONE << position) & mask
        for k, e in enumerate(target):
            if e < 0:
                return 0
            values = self.values[k]
            pos = bisect_right(values, e)
            if pos < len(values):
                mask &= ~self.masks[k][values[pos]]
                if not mask:
                    return 0
        return mask

    def _multiple_mask(self, target: Monomial) -> int:
        if len(target) != self.nvars:
            raise ValueError('Wrong exponent-vector length')
        mask = self.all_mask
        if not mask:
            return 0
        for k, e in enumerate(target):
            if e <= 0:
                continue
            values = self.values[k]
            pos = bisect_left(values, e)
            if pos == len(values):
                return 0
            mask &= self.masks[k][values[pos]]
            if not mask:
                return 0
        return mask

    def _divisor_mask_packed(self, packed: int) -> int:
        """Exact divisor rows for validated little-endian byte exponents.

        Only the query has a bounded encoding. Stored thresholds remain
        arbitrary-precision integers, including exponents above 127 or 255.
        A cached block excludes a row iff one of its eight coordinates is
        too large. In particular, zero coordinates are checked, not skipped.
        """
        mask = self.all_mask
        for start, stop, cache in self.packed_blocks:
            chunk = packed & 0xffffffffffffffff
            packed >>= 64
            allowed = cache.get(chunk)
            if allowed is None:
                allowed = self.all_mask
                remaining = chunk
                for k in range(start, stop):
                    exponent = remaining & 255
                    remaining >>= 8
                    values = self.values[k]
                    pos = bisect_right(values, exponent)
                    if pos < len(values):
                        allowed &= ~self.masks[k][values[pos]]
                        if not allowed:
                            break
                # Bound auxiliary memory even on dense, high-degree inputs.
                # Dict clearing is cheaper than maintaining LRU recency here.
                if len(cache) >= 2048:
                    cache.clear()
                cache[chunk] = allowed
            mask &= allowed
            if not mask:
                return 0
        return mask

    def divisors(self, target: Monomial) -> Iterator[Any]:
        mask = self._divisor_mask(target)
        while mask:
            bit = mask & -mask
            payload = self.payloads[bit.bit_length() - 1]
            if payload is not None:
                yield payload
            mask ^= bit

    def divisors_packed(self, packed: int) -> Iterator[Any]:
        """The same exact buckets from an internal validated byte descriptor."""
        mask = (self._divisor_mask_packed(packed) if _native_divisor_mask_packed is None
                else _native_divisor_mask_packed(self, packed))
        while mask:
            bit = mask & -mask
            payload = self.payloads[bit.bit_length() - 1]
            if payload is not None:
                yield payload
            mask ^= bit

    def multiples(self, target: Monomial) -> Iterator[tuple[Monomial, Any]]:
        mask = self._multiple_mask(target)
        while mask:
            bit = mask & -mask
            i = bit.bit_length() - 1
            payload = self.payloads[i]
            if payload is not None:
                yield self.monomials[i], payload
            mask ^= bit


class CoverBucket:
    __slots__ = ('keys', 'coefficients', 'prefix')
    def __init__(self) -> None:
        self.keys: list[tuple] = []
        self.coefficients: list[int] = []
        self.prefix: list[int] = []

    def insert(self, key: tuple, coefficient: int) -> None:
        coefficient = abs(coefficient)
        pos = bisect_left(self.keys, key)
        before = self.prefix[pos - 1] if pos else 0
        # Only changes of the prefix gcd affect a strict-below query. Once an
        # earlier coefficient gcd divides this coefficient, this record is
        # redundant for every present AND future cutoff (new insertions can
        # only decrease a gcd). In particular unit buckets need one key.
        if before and coefficient % before == 0:
            return
        if pos < len(self.keys) and self.keys[pos] == key:
            value = gcd(self.coefficients[pos], coefficient)
            if value == self.coefficients[pos]:
                return
            self.coefficients[pos] = value
        else:
            self.keys.insert(pos, key)
            self.coefficients.insert(pos, coefficient)
            self.prefix.insert(pos, 0)
        acc, j = before, pos
        while j < len(self.keys):
            value = gcd(acc, self.coefficients[j])
            if value == acc:
                del self.keys[j]
                del self.coefficients[j]
                del self.prefix[j]
                continue
            old = self.prefix[j]
            self.prefix[j] = acc = value
            if acc == 1:
                del self.keys[j + 1:]
                del self.coefficients[j + 1:]
                del self.prefix[j + 1:]
                return
            if old == acc:
                return
            j += 1

    def gcd_below(self, key: tuple) -> int:
        if not self.keys or key <= self.keys[0]:
            return 0
        if len(self.keys) == 1:
            return self.prefix[0]
        pos = bisect_left(self.keys, key) - 1  # STRICT polynomial bound
        return self.prefix[pos] if pos >= 0 else 0


class CoverIndex:
    def __init__(self, backend, module_order, syzygies, stats: Stats,
                 mode: str = 'indexed', cache_size: int = 8192) -> None:
        if mode not in ('indexed', 'naive'):
            raise ValueError('cover mode must be indexed or naive')
        self.backend, self.order, self.syzygies, self.stats = backend, module_order, syzygies, stats
        self.mode, self.cache_size = mode, cache_size
        self.buckets: dict[tuple[int, Monomial], CoverBucket] = {}
        # P is append-only, but rebuilding every coordinate's prefix masks
        # after each insertion still costs O(nvars * stored exponents).
        # Incremental positive-exponent postings also skip zero coordinates.
        self.tries = [SyzygyIndex(len(backend.names)) for _ in module_order.rank]
        self.entries: list[tuple[Label, Monomial]] = []
        self.cache: OrderedDict = OrderedDict()
        self.p_version = 0
        self.native = (_NativeCoverQuery(len(backend.names), len(module_order.rank),
                                        backend.order_name)
                       if mode == 'indexed' and _NativeCoverQuery is not None else None)

    def insert(self, label: Label, lm: Monomial) -> None:
        self.entries.append((label, lm))
        self.p_version += 1
        if self.mode == 'naive':
            return
        sig = label.signature
        if self.native is not None:
            self.native.insert(sig.coefficient, sig.monomial, sig.index, lm)
        bkey = (sig.index, sig.monomial)
        if bkey not in self.buckets:
            self.buckets[bkey] = CoverBucket()
            self.tries[sig.index].set(sig.monomial, self.buckets[bkey])
        delta = subtract_monomials(lm, sig.monomial)
        self.buckets[bkey].insert(self.backend.order_key(delta), sig.coefficient)

    def check_packed(self, coefficient: int, packed_signature: int, index: int,
                     packed_leading: int, *, unit_syzygy_checked: bool = False) -> bool:
        """Exact cover without expanding positive native P-cover witnesses.

        A native proof combines retained P rows: its monomial is their LCM,
        its coefficient their GCD, and its strict cutoff their greatest formal
        leading ratio. This implication survives every later P/S update.
        A miss still executes the complete S/P coefficient-GCD criterion.
        """
        if self.native is not None:
            self.stats.counters['native_cover_queries'] += 1
            if self.native.positive(coefficient, packed_signature, index, packed_leading):
                self.stats.counters['cover_calls'] += 1
                self.stats.counters['cover_gcd_updates'] += 1
                self.stats.counters['native_cover_proofs'] += 1
                return True
            self.stats.counters['native_cover_fallbacks'] += 1
        nvars = len(self.backend.names)
        sig = _signature_from_arithmetic(coefficient, unpack_monomial(packed_signature, nvars), index)
        return self.check(sig, unpack_monomial(packed_leading, nvars),
                          packed_monomial=packed_signature,
                          unit_syzygy_checked=unit_syzygy_checked)

    def check(self, sig: Signature, W: Monomial, *, packed_monomial=None,
              unit_syzygy_checked: bool = False) -> bool:
        """Exact cover, optionally reusing an internal packed descriptor/proof.

        unit_syzygy_checked means an exact S existence query just returned
        False with S unchanged. It can replace the unit-only S query, but never
        the coefficient gcd accumulation required for a nonunit component.
        """
        c = abs(sig.coefficient)
        version = (self.p_version, self.syzygies.version)
        ckey = (c, sig.monomial, sig.index, W)
        self.stats.counters['cover_calls'] += 1
        if self.mode == 'indexed' and self.cache_size:
            cached = self.cache.get(ckey)
            # True results persist: P is retained; S deletions preserve covers.
            if cached is not None and (cached[1] or cached[0] == version):
                self.stats.counters['cover_cache_hits'] += 1
                self.cache.move_to_end(ckey)
                return cached[1]
        if self.mode == 'naive':
            ans = self._naive(sig, W)
        else:
            ans = self._indexed(sig, W, packed_monomial=packed_monomial,
                                unit_syzygy_checked=unit_syzygy_checked)
        if self.mode == 'indexed' and self.cache_size:
            self.cache[ckey] = (version, ans)
            self.cache.move_to_end(ckey)
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return ans

    def _indexed(self, sig: Signature, W: Monomial, *, packed_monomial=None,
                 unit_syzygy_checked: bool = False) -> bool:
        acc, c = 0, abs(sig.coefficient)
        if self.syzygies.nonunit_counts[sig.index] == 0:
            # In a unit-only component, a single monomial divisor supplies gcd 1.
            # Do not allocate a tuple of hundreds of identical coefficients.
            if not unit_syzygy_checked and self.syzygies.has_divisor(sig):
                self.stats.counters['cover_gcd_updates'] += 1
                return True
        else:
            for coeff in self.syzygies.divisor_coefficients(sig.monomial, sig.index):
                acc = gcd(acc, coeff)
                self.stats.counters['cover_gcd_updates'] += 1
                if c % acc == 0:
                    return True
        # Translation-invariant extension of the monomial order to Z^n:
        # lm(vq) + sig.exp - sq.exp < W iff lm(vq)-sq.exp < W-sig.exp.
        cutoff = self.backend.order_key(subtract_monomials(W, sig.monomial))
        trie = self.tries[sig.index]
        buckets = (trie.divisors(sig.monomial) if packed_monomial is None
                   else trie.divisors_packed(packed_monomial))
        for bucket in buckets:
            self.stats.counters['cover_bucket_queries'] += 1
            b = bucket.gcd_below(cutoff)
            if b:
                acc = gcd(acc, b)
                self.stats.counters['cover_gcd_updates'] += 1
                if c % acc == 0:
                    return True
        return False

    def _naive(self, sig: Signature, W: Monomial) -> bool:
        acc, c = 0, abs(sig.coefficient)
        for s in self.syzygies.all_signatures():
            self.stats.counters['cover_naive_records'] += 1
            if s.index == sig.index and divides_monomial(s.monomial, sig.monomial):
                acc = gcd(acc, abs(s.coefficient))
                self.stats.counters['cover_gcd_updates'] += 1
                if c % acc == 0:
                    return True
        Wkey = self.backend.order_key(W)
        for rec, lm in self.entries:
            self.stats.counters['cover_naive_records'] += 1
            s = rec.signature
            if s.index != sig.index or not divides_monomial(s.monomial, sig.monomial):
                continue
            shifted = tuple(w + u - v for w, u, v in zip(lm, sig.monomial, s.monomial))
            if self.backend.order_key(shifted) < Wkey:
                acc = gcd(acc, abs(s.coefficient))
                self.stats.counters['cover_gcd_updates'] += 1
                if c % acc == 0:
                    return True
        return False

def gcd_signature(a: Signature, b: Signature) -> Signature | None:
    """Formula on signature terms (also useful to check equal-label records).

The main algorithm calls it only for distinct syzygy records.
    """
    if a.index != b.index:
        return None
    return Signature(gcd(a.coefficient, b.coefficient),
                     lcm_monomials(a.monomial, b.monomial), a.index)


def minimal_signatures(items) -> list[Signature]:
    """Finite exact-divisibility antichain; one positive associate representative."""
    ordered = sorted({s.positive() for s in items},
                     key=lambda s: (s.index, sum(s.monomial), s.coefficient, s.monomial))
    keep: list[Signature] = []
    # A proper divisor has smaller degree or (at equal degree) the same
    # monomial and a smaller positive coefficient, so it is encountered first.
    for s in ordered:
        if not any(t.divides(s) for t in keep):
            keep.append(s)
    return keep


def is_complete(items) -> bool:
    items = tuple(items)
    for a, b in combinations(items, 2):
        h = gcd_signature(a, b)
        if h is not None and not any(s.divides(h) for s in items):
            return False
    return True


class _SyzygyComponent(Sequence):
    """Ordered exact signatures with O(1)-average append and keyed deletion.

    Iteration/indexing expose the same active order as the former list:
    survivors first, retained new signatures last. An immutable tuple snapshot
    is built only for explicit indexing, not on every insertion or deletion.
    This is an internal sequence, not a mutable-list API.
    """
    __slots__ = ('_entries', '_snapshot')

    def __init__(self) -> None:
        self._entries = {}
        self._snapshot = None

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries.values())

    def __getitem__(self, key):
        if self._snapshot is None:
            self._snapshot = tuple(self._entries.values())
        return self._snapshot[key]

    def __eq__(self, other):
        if isinstance(other, Sequence):
            return tuple(self) == tuple(other)
        return NotImplemented

    def __repr__(self):
        return repr(list(self))

    def extend(self, signatures) -> None:
        for s in signatures:
            self._entries[(s.monomial, s.coefficient)] = s
        self._snapshot = None

    def remove_keys(self, keys) -> None:
        for key in keys:
            del self._entries[key]
        self._snapshot = None


class SyzygyStore:
    def __init__(self, nvars: int, ncomponents: int, stats: Stats,
                 debug: bool = False) -> None:
        self.nvars, self.stats, self.debug = nvars, stats, debug
        self.components = [_SyzygyComponent() for _ in range(ncomponents)]
        self.tries = [SyzygyIndex(nvars) for _ in range(ncomponents)]
        self.nonunit_counts = [0] * ncomponents
        self.version = 0
        self.divisor_cache = [OrderedDict() for _ in range(ncomponents)]
        # Boolean queries must not enumerate all coefficients for multi-cover.
        # Positive divisibility answers persist: every removed signature is
        # replaced by a retained divisor. Negative answers are versioned per
        # component, so an insertion cannot leave behind a stale rejection.
        self.existence_cache = [OrderedDict() for _ in range(ncomponents)]
        self.component_versions = [0] * ncomponents
        # One exact recent witness per component (coefficient, sparse support).
        # Like positive cache entries, its implication survives replacement by
        # a retained divisor. This is a proved shortcut, not an approximate filter.
        self.recent_witness = [None] * ncomponents
        self.packed_existence_cache = [OrderedDict() for _ in range(ncomponents)]
        self.packed_recent_witness = [None] * ncomponents
        self.packed_guard = int.from_bytes(bytes([128]) * nvars, 'little')

    def all_signatures(self) -> list[Signature]:
        return [s for group in self.components for s in group]

    def __len__(self) -> int:
        return sum(map(len, self.components))

    def divisor_coefficients(self, monomial: Monomial, index: int):
        cache = self.divisor_cache[index]
        coeffs = cache.get(monomial)
        if coeffs is None:
            coeffs = tuple(c for group in self.tries[index].divisors(monomial) for c in group)
            cache[monomial] = coeffs
            if len(cache) > 8192:
                cache.popitem(last=False)
        else:
            cache.move_to_end(monomial)
            self.stats.counters['syzygy_divisor_cache_hits'] += 1
        return coeffs

    def has_divisor(self, s: Signature) -> bool:
        if len(s.monomial) != self.nvars:
            raise ValueError('Wrong exponent-vector length')
        return self._has_divisor_parts(s.coefficient, s.monomial, s.index)

    def _has_divisor_parts(self, coefficient: int, monomial: Monomial, index: int) -> bool:
        """Exact existence only, from an already validated signature or parents.

        A multi-cover can combine coefficients; this query MUST NOT do that.
        A coefficient gcd dividing the target does not by itself prove the
        existence of a single signature divisor. Mixed components enumerate
        actual divisor buckets and return at the first coefficient witness.
        """
        # Seed/cover callers already have a tuple. In many-variable rings a
        # C-level byte conversion is still cheaper than hashing and examining
        # that tuple repeatedly; it also shares generation's compact caches.
        # Unbounded exponents retain the original exact tuple path.
        if self.nvars >= 16:
            try:
                packed = int.from_bytes(bytes(monomial), 'little')
            except (ValueError, OverflowError):
                pass
            else:
                if not (packed & self.packed_guard):
                    return self._has_divisor_packed(coefficient, packed, index)
        counts = self.stats.counters
        counts['syzygy_existence_queries'] += 1
        trie = self.tries[index]
        if not trie.all_mask:
            return False
        unit_only = self.nonunit_counts[index] == 0
        # Coefficient zero is a cache-key tag for a unit-only component, not
        # a stored signature. A mixed-component query must check its real c.
        key = (monomial, 0 if unit_only else abs(coefficient))
        cache = self.existence_cache[index]
        hit = cache.get(key)
        version = self.component_versions[index]
        if hit is not None and (hit[1] or hit[0] == version):
            cache.move_to_end(key)
            counts['syzygy_existence_cache_hits'] += 1
            return hit[1]
        witness = self.recent_witness[index]
        if witness is not None and coefficient % witness[0] == 0:
            if all(monomial[k] >= e for k, e in witness[1]):
                counts['syzygy_recent_witness_hits'] += 1
                return True
        mask = trie._divisor_mask(monomial)
        witness_row = None
        witness_coefficient = None
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
                for c in trie.payloads[bit.bit_length() - 1]:
                    counts['syzygy_existence_coefficients_tested'] += 1
                    if coefficient % c == 0:
                        answer = True
                        witness_row = bit.bit_length() - 1
                        witness_coefficient = c
                        break
                if answer:
                    break
                mask ^= bit
        if witness_row is not None:
            self.recent_witness[index] = (witness_coefficient, tuple(
                (k, e) for k, e in enumerate(trie.monomials[witness_row]) if e))
        cache[key] = (version, answer)
        cache.move_to_end(key)
        if len(cache) > 8192:
            cache.popitem(last=False)
        return answer

    def _accept_seed_parts(self, coefficient: int, monomial: Monomial, index: int) -> bool:
        """The common counted gate for public and deferred solver insertions."""
        self.stats.counters['syzygy_insert_calls'] += 1
        if self._has_divisor_parts(coefficient, monomial, index):
            self.stats.counters['syzygy_seed_rejected'] += 1
            return False
        return True

    def _has_divisor_packed(self, coefficient: int, packed: int, index: int) -> bool:
        """Exact existence for a validated 0..127 exponent byte encoding.

        This is the deferred-generation gate: rejected candidates need neither
        a full exponent tuple nor a Signature object. Coefficient divisibility
        is still tested against a single stored witness, never a gcd cover.
        """
        if _native_has_divisor_packed is not None:
            return _native_has_divisor_packed(self, coefficient, packed, index)
        return self._has_divisor_packed_python(coefficient, packed, index)

    def _has_divisor_packed_python(self, coefficient: int, packed: int, index: int) -> bool:
        """Portable reference path, also used to differential-check native queries."""
        counts = self.stats.counters
        counts['syzygy_existence_queries'] += 1
        trie = self.tries[index]
        if not trie.all_mask:
            return False
        unit_only = self.nonunit_counts[index] == 0
        key = packed if unit_only else (packed, abs(coefficient))
        cache = self.packed_existence_cache[index]
        hit = cache.get(key)
        version = self.component_versions[index]
        if hit is not None and (hit[1] or hit[0] == version):
            cache.move_to_end(key)
            counts['syzygy_existence_cache_hits'] += 1
            return hit[1]
        witness = self.packed_recent_witness[index]
        guard = self.packed_guard
        if (witness is not None and coefficient % witness[0] == 0
                and (((packed | guard) - witness[1]) & guard) == guard):
            counts['syzygy_recent_witness_hits'] += 1
            return True
        mask = trie._divisor_mask_packed(packed)
        witness_row = None
        witness_coefficient = None
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
                mask ^= bit
        if witness_row is not None:
            # A divisor of this validated query also fits the byte lanes.
            self.packed_recent_witness[index] = (witness_coefficient,
                int.from_bytes(bytes(trie.monomials[witness_row]), 'little'))
        cache[key] = (version, answer)
        cache.move_to_end(key)
        if len(cache) > 8192:
            cache.popitem(last=False)
        return answer

    def insert_and_complete(self, z: Signature) -> bool:
        """Return whether the active simplified set changed.

1. Freeze the old component (which is gcd-complete).
2. Form G(z,old_i), ignoring terms already divided by an OLD signature.
3. Also ignore multiples of z itself (an immediate exact divisor).
4. After ALL candidates have been considered, simplify their union and
   remove old multiples of the retained new signatures. Never mutate the
   snapshot while generating combinations.
        """
        self.stats.counters['syzygy_insert_calls'] += 1
        if self.has_divisor(z):
            self.stats.counters['syzygy_seed_rejected'] += 1
            return False
        return self._complete_uncovered(z)

    def _complete_uncovered(self, z: Signature) -> bool:
        """Private continuation after the exact seed gate, with S unchanged.

        The solver also uses this continuation to delay Signature allocation,
        witness snapshots, and completion timing until a seed actually survives.
        """
        z = z.positive()
        # The component is read-only throughout candidate generation. Copy it
        # only for debug assertions; the solver separately snapshots it when
        # an explicit-vector/path witness needs the pre-update state.
        old = self.components[z.index]
        old_size = len(old)
        old_snapshot = tuple(old) if self.debug else ()
        if self.debug and not is_complete(old):
            raise AssertionError('One-pass completion requires complete old S')
        counts = self.stats.counters
        pending = {z}
        # This remains a logical parent-pair count, not an allocation count.
        counts['syzygy_g_signatures'] += len(old)
        if self.nonunit_counts[z.index] == 0:
            # Every OLD parent divides its G-signature, for any seed coefficient.
            # Avoid even the per-pair LCM/allocation/query loop in this common case.
            counts['syzygy_g_skipped_old'] += len(old)
            counts['syzygy_g_shortcuts'] += len(old)
        else:
            for s in old:
                if z.coefficient % s.coefficient == 0:
                    # gcd(lc(z), lc(s)) == lc(s), so s is an exact divisor.
                    counts['syzygy_g_skipped_old'] += 1
                    counts['syzygy_g_shortcuts'] += 1
                    continue
                if s.coefficient % z.coefficient == 0:
                    # The seed divides the G-signature. Preserve the old-first
                    # counter classification without constructing a Signature.
                    monomial = lcm_monomials(z.monomial, s.monomial)
                    old_divisor = self._has_divisor_parts(z.coefficient, monomial, z.index)
                    counter = ('syzygy_g_skipped_old' if old_divisor else
                               'syzygy_g_skipped_seed')
                    counts[counter] += 1
                    counts['syzygy_g_shortcuts'] += 1
                    continue
                h = gcd_signature(z, s)
                assert h is not None
                counts['syzygy_g_materialized'] += 1
                if self.has_divisor(h):  # index still refers to OLD S
                    counts['syzygy_g_skipped_old'] += 1
                    continue
                if z.divides(h):
                    counts['syzygy_g_skipped_seed'] += 1
                    continue
                pending.add(h)
        new_min = [z] if len(pending) == 1 else minimal_signatures(pending)
        trie = self.tries[z.index]
        removed = set()
        # A new signature can remove only OLD monomial multiples; query the
        # reverse index first, then check the INTEGER coefficient divisibility.
        # In particular, monomial divisibility alone never deletes a signature.
        for h in new_min:
            counts['syzygy_multiple_queries'] += 1
            for monomial, coefficients in trie.multiples(h.monomial):
                counts['syzygy_multiple_candidates'] += len(coefficients)
                for coefficient in coefficients:
                    if coefficient % h.coefficient == 0:
                        removed.add((monomial, coefficient))
        counts['syzygy_new_signatures'] += len(pending)
        counts['syzygy_simplified'] += len(pending) - len(new_min) + len(removed)
        if not removed:
            # No tuple/list copy of the entire component on the append-only
            # path. Its existing order is unchanged, followed by new_min.
            old.extend(new_min)
            result = old
            for s in new_min:
                position = trie.positions.get(s.monomial)
                coeffs = [] if position is None else list(trie.payloads[position])
                coeffs.append(s.coefficient)
                trie.set(s.monomial, coeffs)
            counts['syzygy_index_appends'] += 1
            counts['syzygy_index_records_written'] += len(new_min)
            counts['syzygy_component_reuses'] += 1
        else:
            # Delete only matching buckets. Keep survivors in their old order,
            # append new_min, and leave unrelated posting bits untouched.
            old.remove_keys(removed)
            old.extend(new_min)
            result = old
            affected = {m for m, _ in removed}
            for m in affected:
                position = trie.positions[m]
                coeffs = [c for c in trie.payloads[position] if (m, c) not in removed]
                if coeffs:
                    trie.set(m, coeffs)
                else:
                    trie.remove(m)
            for s in new_min:
                position = trie.positions.get(s.monomial)
                coeffs = [] if position is None else list(trie.payloads[position])
                coeffs.append(s.coefficient)
                trie.set(s.monomial, coeffs)
            counts['syzygy_index_deletions'] += len(removed)
            counts['syzygy_index_records_written'] += len(new_min)
            if trie.needs_compaction():
                compact = SyzygyIndex(self.nvars)
                by_monomial: dict[Monomial, list[int]] = {}
                for s in result:
                    by_monomial.setdefault(s.monomial, []).append(s.coefficient)
                for m, coeffs in by_monomial.items():
                    compact.set(m, coeffs)
                self.tries[z.index] = compact
                counts['syzygy_index_rebuilds'] += 1
                counts['syzygy_index_compactions'] += 1
                counts['syzygy_index_records_written'] += len(result)
        self.nonunit_counts[z.index] += (sum(s.coefficient != 1 for s in new_min)
                                        - sum(c != 1 for _, c in removed))
        self.divisor_cache[z.index].clear()
        self.component_versions[z.index] += 1
        self.version += 1
        self.stats.counters['s_peak'] = max(self.stats.counters['s_peak'], len(self))
        if self.debug:
            assert len(result) == old_size - len(removed) + len(new_min)
            assert self.nonunit_counts[z.index] == sum(s.coefficient != 1 for s in result)
            assert is_complete(result), 'One-pass completion failed'
            assert all(any(s.divides(t) for s in result) for t in old_snapshot + (z,))
            assert all(not a.divides(b) and not b.divides(a)
                       for a, b in combinations(result, 2))
        return True

@dataclass(slots=True)
class Candidate:
    token: int
    kind: str
    signature: Signature
    leading_monomial: Monomial
    leading_coefficient: int | None = None
    _label: Label | None = None
    _factory: Callable[[], Any] | None = None
    representation_recipe: Callable[[], Any] | None = None
    # Local R-linear construction, only allocated when path mode is enabled.
    path_recipe: tuple | None = None
    # Actual generating pair, not the reducers used later.
    generating_parents: tuple[int, int] | None = None
    # Immutable construction data reused by exact-J deduplication. Keys are
    # computed once after admission, never for a new signature rejected by S.
    _dedup_key: tuple | None = None
    _leading_key: int | tuple | None = None

    @property
    def label(self) -> Label:
        if self._label is None:
            if self._factory is None:
                raise RuntimeError('Missing polynomial recipe')
            self._label = Label(self.signature, self._factory())
            self._factory = None
        return self._label


@dataclass(slots=True)
class PackedCandidate:
    """A J candidate whose exponent vectors are decoded only when needed.

    The packed descriptors come from checked parent arithmetic, with each byte
    in [0, 127]. They are internal data, never a public input representation.
    Polynomial recipes can keep their packed shift too, so an established
    syzygy can discard this whole entry without constructing either vector.
    """
    token: int
    _packed_signature: tuple[int, int, int]
    _packed_leading: int
    _nvars: int
    _stats: Stats
    leading_coefficient: int | None = None
    _factory: Callable[[], Any] | None = None
    kind: str = 'J'
    _label: Label | None = None
    _signature: Signature | None = None
    _leading_monomial: Monomial | None = None
    representation_recipe: Callable[[], Any] | None = None
    path_recipe: tuple | None = None
    generating_parents: tuple[int, int] | None = None
    _dedup_key: tuple | None = None
    _leading_key: int | tuple | None = None

    @property
    def signature(self) -> Signature:
        if self._signature is None:
            coefficient, packed, index = self._packed_signature
            self._signature = _signature_from_arithmetic(
                coefficient, unpack_monomial(packed, self._nvars), index)
            self._stats.counters['candidate_signature_fast_builds'] += 1
        return self._signature

    @property
    def leading_monomial(self) -> Monomial:
        if self._leading_monomial is None:
            self._leading_monomial = unpack_monomial(self._packed_leading, self._nvars)
        return self._leading_monomial

    @property
    def packed_signature(self) -> int:
        return self._packed_signature[1]

    @property
    def packed_leading_monomial(self) -> int:
        return self._packed_leading

    @property
    def signature_coefficient(self) -> int:
        return self._packed_signature[0]

    @property
    def signature_index(self) -> int:
        return self._packed_signature[2]

    @property
    def label(self) -> Label:
        if self._label is None:
            if self._factory is None:
                raise RuntimeError('Missing polynomial recipe')
            self._label = Label(self.signature, self._factory())
            self._factory = None
        return self._label


_NativeCandidateHeap = None
if not _native_os.environ.get('GVWZZ_NO_NATIVE'):
    try:
        from ._native_pairs import NativeCandidateHeap as _NativeCandidateHeap
        from ._native_pairs import NativePackedCandidate as PackedCandidate
    except ImportError:
        pass


class CandidateQueue:
    def __init__(self, backend, module_order, stats: Stats, deduplicate: bool = True,
                 g_first: bool = True, lazy: bool = True, generation_filter=None,
                 tie_policy: str = 'fifo', generation_filter_packed=None) -> None:
        self.backend, self.order, self.stats = backend, module_order, stats
        self.deduplicate, self.g_first, self.lazy = deduplicate, g_first, lazy
        self.heap: list = []
        self.active: dict[int, Candidate] = {}
        self.pending_j: dict[tuple, Candidate] = {}
        self.settled_j_bounds: dict[tuple, tuple] = {}
        # Optional append-only negative-membership gate. A positive answer
        # still runs the exact signed-coefficient deduplication dictionary.
        self._native_j_membership = None
        self.serial = 0
        self.generation_filter = generation_filter
        self.generation_filter_packed = generation_filter_packed
        self.tie_policy = tie_policy
        self.nvars = len(backend.names)
        # A grevlex monomial with byte-lane exponents has the exact scalar key
        # degree * 256**n + complement(reverse-lex exponent digits).  Combining
        # the bounded degree field with POT rank also makes the heap's primary
        # key one integer. For G2, the weighted monomial comes first and rank
        # occupies the low bits. All arithmetic uses unbounded Python integers.
        self._compact_g2 = module_order.name == 'gvw-g2'
        self._packed_weights = (tuple(pack_monomial(m) for m in module_order.weights)
                                if self._compact_g2 else ())
        self._compact_order = (backend.order_name == 'grevlex'
                               and (module_order.name == 'pot'
                                    or (self._compact_g2
                                        and all(m is not None for m in self._packed_weights))))
        self._order_bits = 8 * self.nvars
        self._order_mask = (1 << self._order_bits) - 1
        self._rank_shift = self._order_bits + (127 * self.nvars).bit_length()
        self._signature_rank_bits = max(1, (len(module_order.rank) - 1).bit_length())
        self._signature_rank_mask = (1 << self._signature_rank_bits) - 1
        self._packed_leading_key = lru_cache(maxsize=32768)(self._make_packed_leading_key)
        self._native_heap = None
        if self._compact_order and _NativeCandidateHeap is not None:
            self._native_heap = _NativeCandidateHeap(self.tie_policy)
            self.heap = self._native_heap

    def _make_packed_leading_key(self, packed: int) -> int:
        return ((sum(packed.to_bytes(self.nvars, 'little')) << self._order_bits)
                | (self._order_mask ^ packed))

    def _uncompact_leading_key(self, key: int) -> tuple:
        packed = (key & self._order_mask) ^ self._order_mask
        return self.backend.order_key(unpack_monomial(packed, self.nvars))

    def _packed_signature_key(self, monomial: int, index: int) -> int:
        if self._compact_g2:
            # Each summand has lanes <=127, hence weighted lanes <=254 and
            # cannot carry into a neighbouring byte. Equal weighted monomials
            # and ranks imply equal unweighted monomials, so G2's final tie
            # key is redundant. Coefficients never enter a module-order key.
            weighted = monomial + self._packed_weights[index]
            return ((self._packed_leading_key(weighted) << self._signature_rank_bits)
                    | self.order.rank[index])
        return ((self.order.rank[index] << self._rank_shift)
                | self._packed_leading_key(monomial))

    def _uncompact_signature_key(self, key: int) -> tuple:
        if self._compact_g2:
            rank = key & self._signature_rank_mask
            index = self.order.positions[rank]
            weighted_key = key >> self._signature_rank_bits
            weighted = (weighted_key & self._order_mask) ^ self._order_mask
            monomial = unpack_monomial(weighted - self._packed_weights[index], self.nvars)
            return self.order.key_monomial(monomial, index)
        low_mask = (1 << self._rank_shift) - 1
        return (key >> self._rank_shift,
                self._uncompact_leading_key(key & low_mask))

    def _disable_compact_order(self) -> None:
        """Switch every retained key together when a byte lane overflows.

        Keeping stale heap entries preserves their original pop/counter
        behaviour. Settled bounds must be converted too; a tagged mixed key
        would incorrectly order large and small exponent candidates.
        """
        if not self._compact_order:
            return
        self._compact_order = False
        for candidate in self.active.values():
            if candidate._leading_key is not None:
                candidate._leading_key = self._uncompact_leading_key(candidate._leading_key)
        for dkey, key in self.settled_j_bounds.items():
            self.settled_j_bounds[dkey] = self._uncompact_leading_key(key)
        heap = []
        for sigkey, priority, tie, token in self.heap:
            sigkey = self._uncompact_signature_key(sigkey)
            if self.tie_policy == 'small-coeff':
                tie = (tie[0], self._uncompact_leading_key(tie[1]), tie[2])
            elif self.tie_policy == 'small-lm':
                tie = (self._uncompact_leading_key(tie[0]), tie[1], tie[2])
            heap.append((sigkey, priority, tie, token))
        self.heap = heap
        self._native_heap = None
        heapq.heapify(self.heap)
        self._packed_leading_key.cache_clear()

    def __bool__(self) -> bool:
        return bool(self.active)

    def compact_when_empty(self) -> None:
        """Drop stale heap entries after the queue has been exhausted."""
        if self.active:
            raise RuntimeError('Cannot compact a nonempty candidate queue')
        self.heap.clear()
        self.pending_j.clear()

    @staticmethod
    def _signature_key(sig: Signature) -> tuple:
        packed = pack_monomial(sig.monomial)
        return (sig.coefficient, sig.monomial if packed is None else packed, sig.index)

    def _insert(self, kind, sig, W, *, label=None, factory=None, lc=None,
                dedup_key=None, leading_packed=None) -> None:
        self.serial += 1
        entry = Candidate(self.serial, kind, sig, W, lc, label, factory)
        if kind == 'J' and self.deduplicate:
            entry._dedup_key = self._signature_key(sig) if dedup_key is None else dedup_key
            if self._native_j_membership is not None:
                self._native_j_membership.add_signature(*entry._dedup_key)
        need_leading_key = (kind == 'J' and self.deduplicate) or self.tie_policy != 'fifo'
        if self._compact_order:
            dkey = entry._dedup_key if entry._dedup_key is not None else dedup_key
            packed_sig = (dkey[1] if dkey is not None
                          and isinstance(dkey[1], int) else pack_monomial(sig.monomial))
            packed_W = ((leading_packed if leading_packed is not None else pack_monomial(W))
                        if need_leading_key else 0)
            if packed_sig is None or packed_W is None:
                self._disable_compact_order()
            else:
                signature_key = self._packed_signature_key(packed_sig, sig.index)
                if need_leading_key:
                    entry._leading_key = self._packed_leading_key(packed_W)
        if not self._compact_order:
            signature_key = self.order.key(sig)
            if need_leading_key:
                entry._leading_key = self.backend.order_key(W)
        if not self.lazy:
            _ = entry.label
        priority = (0 if kind == 'G' else 1) if self.g_first else (0 if kind == 'J' else 1)
        # Signatures remain the PRIMARY queue key. Heuristics only break
        # ties, never bypass the strictly-smaller-signature completeness bound.
        if self.tie_policy == 'small-coeff':
            tie = (abs(lc) if kind == 'G' and lc is not None else abs(sig.coefficient),
                   entry._leading_key, abs(sig.coefficient))
        elif self.tie_policy == 'small-lm':
            tie = (entry._leading_key,
                   abs(lc) if kind == 'G' and lc is not None else abs(sig.coefficient),
                   abs(sig.coefficient))
        else:
            tie = ()
        if self._native_heap is not None:
            self._native_heap.push((signature_key, priority, tie, self.serial))
            self.stats.counters['native_queue_pushes'] += 1
        else:
            heapq.heappush(self.heap, (signature_key, priority, tie, self.serial))
        self.active[self.serial] = entry
        if kind == 'J' and self.deduplicate:
            self.pending_j[entry._dedup_key] = entry
        self.stats.counters[kind.lower() + '_enqueued'] += 1
        self.stats.counters['queue_peak'] = max(self.stats.counters['queue_peak'], len(self.active))

    def _insert_j_packed(self, coefficient: int, monomial: int, index: int,
                         W: int, factory: Callable, lc: int | None = None):
        """Admit an already-filtered packed J, deferring both exponent vectors.

        This matches _insert('J', ...) exactly, including tie policies and eager
        mode. The native emitter supplies trusted bounded byte descriptors.
        Unsupported orders use the established ordinary Candidate path.
        """
        dkey = (coefficient, monomial, index)
        if self.deduplicate and self._native_j_membership is not None:
            self._native_j_membership.add_signature(coefficient, monomial, index)
        if not self._compact_order:
            sig = _signature_from_arithmetic(
                coefficient, unpack_monomial(monomial, self.nvars), index)
            self.stats.counters['candidate_signature_fast_builds'] += 1
            self._insert('J', sig, unpack_monomial(W, self.nvars), factory=factory,
                         lc=lc, dedup_key=dkey)
            return self.active[self.serial]
        self.serial += 1
        entry = PackedCandidate(self.serial, dkey, W, self.nvars, self.stats, lc, factory)
        if self.deduplicate:
            entry._dedup_key = dkey
        if self.deduplicate or self.tie_policy != 'fifo':
            entry._leading_key = self._packed_leading_key(W)
        if not self.lazy:
            _ = entry.label
        signature_key = self._packed_signature_key(monomial, index)
        if self.tie_policy == 'small-coeff':
            tie = (abs(coefficient), entry._leading_key, abs(coefficient))
        elif self.tie_policy == 'small-lm':
            tie = (entry._leading_key, abs(coefficient), abs(coefficient))
        else:
            tie = ()
        if self._native_heap is not None:
            self._native_heap.push((signature_key, int(self.g_first), tie, self.serial))
            self.stats.counters['native_queue_pushes'] += 1
        else:
            heapq.heappush(self.heap, (signature_key, int(self.g_first), tie, self.serial))
        self.active[self.serial] = entry
        if self.deduplicate:
            self.pending_j[dkey] = entry
        counts = self.stats.counters
        counts['j_enqueued'] += 1
        counts['candidate_signature_objects_deferred'] += 1
        counts['queue_peak'] = max(counts['queue_peak'], len(self.active))
        return entry

    def _accept_j(self, sig: Signature, W: Monomial) -> bool:
        """Unchanged exact-J dedup/filter order, before allocating a recipe."""
        if self.deduplicate:
            if not self._accept_j_bound(self._signature_key(sig), W):
                return False
        if self.generation_filter is not None and self.generation_filter(sig):
            self.stats.counters['j_generation_syzygy_rejected'] += 1
            return False
        return True

    def _accept_j_bound(self, dkey: tuple, W: Monomial | None,
                        W_packed: int | None = None) -> bool:
        settled = self.settled_j_bounds.get(dkey)
        pending = self.pending_j.get(dkey)
        if settled is None and pending is None:
            return True
        # The common new-signature path neither decodes the leading monomial
        # nor constructs its order key. Bounds retained in the queue are keys
        # already, so a hit needs only the new candidate's key.
        if self._compact_order:
            packed = W_packed if W is None else pack_monomial(W)
            if packed is None:
                self._disable_compact_order()
                settled = self.settled_j_bounds.get(dkey)
            else:
                key = self._packed_leading_key(packed)
        if not self._compact_order:
            if W is None:
                W = unpack_monomial(W_packed, self.nvars)
            key = self.backend.order_key(W)
        if settled is not None and settled <= key:
            self.stats.counters['j_dedup_discarded'] += 1
            self.stats.counters['j_dedup_settled'] += 1
            return False
        if pending is not None:
            if pending._leading_key <= key:
                self.stats.counters['j_dedup_discarded'] += 1
                return False
            del self.active[pending.token]
            del self.pending_j[dkey]
            self.stats.counters['j_dedup_replaced'] += 1
        return True

    def _accept_j_packed(self, coefficient: int, monomial: int, index: int,
                         W: int, known_syzygy: bool = False) -> bool:
        """Exact byte-lane descriptors; preserve dedup BEFORE generation S test."""
        if self.deduplicate:
            if not self._accept_j_bound((coefficient, monomial, index), None, W):
                return False
        if known_syzygy and (self.generation_filter_packed is not None
                             or self.generation_filter is not None):
            self.stats.counters['j_generation_tests'] += 1
            self.stats.counters['j_generation_syzygy_rejected'] += 1
            return False
        if self.generation_filter_packed is not None:
            rejected = self.generation_filter_packed(coefficient, monomial, index)
        elif self.generation_filter is not None:
            sig = Signature(coefficient, unpack_monomial(monomial, self.nvars), index)
            rejected = self.generation_filter(sig)
        else:
            rejected = False
        if rejected:
            self.stats.counters['j_generation_syzygy_rejected'] += 1
            return False
        return True

    def push_j(self, sig: Signature, W: Monomial, polynomial_factory: Callable,
               leading_coefficient: int | None = None) -> bool:
        if not self._accept_j(sig, W):
            return False
        self._insert('J', sig, W, factory=polynomial_factory, lc=leading_coefficient)
        return True

    def push_g_lazy(self, sig, W, lc, polynomial_factory) -> None:
        self._insert('G', sig, W, factory=polynomial_factory, lc=lc)

    def pop(self) -> Candidate:
        if self._native_heap is not None:
            return self._native_heap.pop_candidate(self)
        while self.heap:
            token = heapq.heappop(self.heap)[3]
            candidate = self.active.pop(token, None)
            if candidate is None:
                self.stats.counters['queue_stale_entries'] += 1
                continue
            if candidate.kind == 'J' and self.deduplicate:
                del self.pending_j[candidate._dedup_key]
            self.stats.counters[candidate.kind.lower() + '_popped'] += 1
            return candidate
        raise IndexError('No active candidate')

    def settle_j(self, candidate: Candidate) -> None:
        """Call only AFTER the original exact signature has acquired a cover."""
        if self._native_heap is not None:
            self._native_heap.settle_candidate(self, candidate)
            return
        if not self.deduplicate:
            return
        dkey, key = candidate._dedup_key, candidate._leading_key
        # A caller may hold a popped candidate while a subsequent insertion
        # triggers the unbounded fallback. It is no longer in active.values().
        if not self._compact_order and isinstance(key, int):
            key = self._uncompact_leading_key(key)
            candidate._leading_key = key
        old = self.settled_j_bounds.get(dkey)
        if old is None or key < old:
            self.settled_j_bounds[dkey] = key
            if self._native_j_membership is not None:
                self._native_j_membership.add_signature(*dkey)
