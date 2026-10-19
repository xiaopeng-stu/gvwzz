"""Bulk, exact frozen-prefix rejection for incremental POT stages.

The index contains immutable parent row IDs, including repeated monomials.
It is deliberately separate from the evolving syzygy store: simplification of
that store must neither renumber these parents nor erase the original proof.
"""
from __future__ import annotations

from bisect import bisect_right


class FrozenSeedIndex:
    """Find frozen parents not rejected by their own unit prefix seed.

    Let a frozen parent have leading monomial ``a`` and the current parent
    have leading monomial ``b`` and signature monomial ``s``. For unit leading
    coefficients the frozen seed ``a * e_k`` divides their J signature iff

        a <= max(a, b) - b + s  iff  min(a, b) <= s.

    Failure is equivalent to the existence of a coordinate k with both
    ``b[k] > s[k]`` and ``a[k] > s[k]``. Threshold postings turn this existential
    condition into a union of bitsets, avoiding all rejected parent pairs.
    The caller must establish unit leading coefficients and natural-order
    incremental POT. For general coefficients an additional divisibility
    condition is required and omission based on this index alone is invalid.

    A removed prefix seed is still covered by a retained divisor, so later
    syzygy simplification cannot invalidate a rejection. Returned survivors
    merely fail this particular witness; they still need all ordinary gates.
    """

    def __init__(self, nvars: int, leading_monomials) -> None:
        self.nvars = nvars
        self.count = 0
        self.monomials = []
        self.squarefree_supports = []
        self.squarefree = True
        self._packed_rows = None
        self._sparse_monomials = []
        self._largest_degree = self._largest_exponent = 0
        exact = [{} for _ in range(nvars)]
        for row, monomial in enumerate(leading_monomials):
            if len(monomial) != nvars:
                raise ValueError('Wrong exponent-vector length')
            bit = 1 << row
            support = 0
            sparse = []
            degree = 0
            for k, exponent in enumerate(monomial):
                if exponent < 0:
                    raise ValueError('Negative monomial exponent')
                if exponent:
                    sparse.append((k, exponent))
                    degree += exponent
                    if exponent > self._largest_exponent:
                        self._largest_exponent = exponent
                    support |= 1 << k
                    if exponent != 1:
                        self.squarefree = False
                    buckets = exact[k]
                    buckets[exponent] = buckets.get(exponent, 0) | bit
            self.monomials.append(monomial)
            self._sparse_monomials.append(sparse)
            self._largest_degree = max(self._largest_degree, degree)
            self.squarefree_supports.append(support)
            self.count += 1
        self.all_mask = (1 << self.count) - 1
        self.values = []
        self.masks = []
        for buckets in exact:
            values = sorted(buckets)
            masks = [0] * len(values)
            mask = 0
            for j in range(len(values) - 1, -1, -1):
                mask |= buckets[values[j]]
                masks[j] = mask
            self.values.append(values)
            self.masks.append(masks)

    def survivor_mask(self, leading, signature) -> int:
        """Parent IDs whose own seed does not divide the proposed J signature.

        Inputs are validated polynomial/signature exponent tuples. This path
        has no bounded exponent encoding and never approximates divisibility.
        """
        if len(leading) != self.nvars or len(signature) != self.nvars:
            raise ValueError('Wrong exponent-vector length')
        mask = 0
        for k, (b, s) in enumerate(zip(leading, signature)):
            if b <= s:
                continue
            values = self.values[k]
            pos = bisect_right(values, s)
            if pos < len(values):
                mask |= self.masks[k][pos]
        return mask

    def surviving_parents(self, leading, signature):
        """Return ascending surviving IDs and the number of omitted pairs."""
        mask = self.survivor_mask(leading, signature)
        survivors = []
        while mask:
            bit = mask & -mask
            survivors.append(bit.bit_length() - 1)
            mask ^= bit
        return tuple(survivors), self.count - len(survivors)

    intersection_nondivisors = survivor_mask

    def minimal_parents(self, leading, parents):
        """Keep earliest rows for divisibility-minimal frozen J multipliers.

        This is a future-cover criterion, not a present syzygy witness.
        With a fixed current parent p and unit frozen leading coefficients,
        all these J candidates are the same signed p times
        ``u = max(frozen_lm - leading, 0)``. If u1 properly divides u2,
        J1 has strictly smaller signature and is processed first. Its eventual
        strict cover, multiplied by u2/u1, is a strict cover of J2. An existing
        same-signature/lower-bound dedup successor of J1 suffices too. Frozen
        pair emission never updates S, so selecting the whole batch is safe.

        Equal multipliers retain the earliest original row. Returning retained
        rows in ascending original order preserves all meaningful FIFO ties.
        The caller must not confuse omitted rows with immediate S rejections.
        """
        if len(leading) != self.nvars:
            raise ValueError('Wrong exponent-vector length')
        if self.squarefree and all(e <= 1 for e in leading):
            current = sum(1 << k for k, e in enumerate(leading) if e)
            unique = {}
            for row in parents:
                u = self.squarefree_supports[row] & ~current
                old = unique.get(u)
                if old is None or row < old:
                    unique[u] = row
            keep, nonunits, unit_variables = [], [], 0
            for u, row in sorted(unique.items(), key=lambda item: (item[0].bit_count(), item[1])):
                if not u:
                    return (row,)
                if u & unit_variables:
                    continue
                if any(v & u == v for v in nonunits):
                    continue
                keep.append(row)
                if not (u & (u - 1)):
                    unit_variables |= u
                else:
                    nonunits.append(u)
            return tuple(sorted(keep))
        return self._minimal_packed(leading, parents)

    def minimal_parents_mask(self, leading, parents_mask):
        """The same exact selection directly from a parent-row bitset."""
        def rows():
            remaining = parents_mask
            while remaining:
                bit = remaining & -remaining
                yield bit.bit_length() - 1
                remaining ^= bit
        return self.minimal_parents(leading, rows())

    def _prepare_packed_rows(self):
        # One spare high bit in every lane stops subtraction borrows at the
        # lane boundary. The data width also holds a *whole monomial degree*,
        # so base-1 reduction computes that degree without unpacking lanes.
        # Width is derived from Python integers: there is no exponent limit.
        self._lane_bits = max(1, self._largest_degree.bit_length()) + 1
        self._lane_mask = (1 << self._lane_bits) - 1
        self._lane_ones = sum(1 << (k * self._lane_bits) for k in range(self.nvars))
        self._lane_guards = self._lane_ones << (self._lane_bits - 1)
        self._packed_rows = [sum(e << (k * self._lane_bits) for k, e in sparse)
                             | self._lane_guards for sparse in self._sparse_monomials]

    def _minimal_packed(self, leading, parents):
        if self._packed_rows is None:
            self._prepare_packed_rows()
        bits, guards, ones = self._lane_bits, self._lane_guards, self._lane_ones
        maximum = self._largest_exponent
        # b > max(a) has exactly the same positive difference as b=max(a).
        # Clamping prevents an arbitrary large current exponent borrowing
        # across packed lanes, while preserving max(a-b, 0) exactly.
        current = sum((b if b < maximum else maximum) << (k * bits)
                      for k, b in enumerate(leading))
        packed_rows = self._packed_rows
        unique = {}
        unit_variables = 0
        for row in parents:
            difference = packed_rows[row] - current
            nonnegative = difference & guards
            u = difference & (nonnegative - (nonnegative >> (bits - 1)))
            old = unique.get(u)
            if old is None or row < old:
                unique[u] = row
            if u & ones and not u & (u - 1):
                unit_variables |= u << (bits - 1)
        zero = unique.get(0)
        if zero is not None:
            return (zero,)
        # Every retained multiplier is an exact arbitrary-width packed value.
        # Support buckets are useful for sparse monomials: a divisor's support
        # must be a subset. Within each bucket the guarded subtraction remains
        # the full exponent-wise divisibility test, including powers > 1.
        keep, minima, by_support, records = [], [], {}, []
        units = unit_variables >> (bits - 1)
        while units:
            unit = units & -units
            keep.append(unique[unit])
            units ^= unit
        for u, row in unique.items():
            support = ((u | guards) - ones) & guards
            if support & unit_variables:
                continue
            records.append((u % self._lane_mask, row, u, support))
        for degree, row, u, support in sorted(records):
            target = u | guards
            found = False
            if (1 << support.bit_count()) <= len(minima):
                subset = support
                while subset:
                    for divisor in by_support.get(subset, ()):
                        if (target - divisor) & guards == guards:
                            found = True
                            break
                    if found:
                        break
                    subset = (subset - 1) & support
            else:
                for divisor, divisor_support, divisor_degree in minima:
                    if divisor_degree == degree:
                        break
                    if (divisor_support & support == divisor_support and
                            (target - divisor) & guards == guards):
                        found = True
                        break
            if found:
                continue
            keep.append(row)
            minima.append((u, support, degree))
            by_support.setdefault(support, []).append(u)
        return tuple(sorted(keep))
