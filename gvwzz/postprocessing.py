"""Polynomial-only post-processing of a completed strong signature basis.

Minimality uses divisibility of COMPLETE leading terms over ZZ. Reduction uses
0 < 1 < -1 < 2 < -2 < ... on coefficients (positive leading coefficients and
balanced remainders, with positive half-modulus ties). No signature restriction,
primitive-part normalization, field conversion or solver tail-step cap is used.

Precondition: the polynomial projection of P is a strong Groebner basis. This
module checks structural/normalization invariants, but does not independently
certify that precondition or membership in the original input ideal. Use the
existing certificate/path verifier on the source result for that purpose.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import TYPE_CHECKING

from .core import divides_monomial, exact_integer, subtract_monomials
from .polynomial import PolynomialWorkspace
from .structures import DivisorIndex, SyzygyIndex

if TYPE_CHECKING:
    from .polynomial import IntegerPolynomialRing
    from .solver import Result


class PolynomialPostprocessingError(ValueError):
    """The source is incomplete/invalid or a post-processing invariant fails."""


def validate_polynomial_output_flags(show_minimal_gb=False, show_reduced_gb=False,
                                     show_minimal_lt=False) -> None:
    for name, value in (("show_minimal_gb", show_minimal_gb),
                        ("show_reduced_gb", show_reduced_gb),
                        ("show_minimal_lt", show_minimal_lt)):
        if type(value) is not bool:
            raise ValueError(f"{name} must be a boolean")


def balanced_divmod(a: int, b: int) -> tuple[int, int]:
    """Return exact q,r with a=q*b+r and -b/2 < r <= b/2; b must be positive."""
    a, b = exact_integer(a, "dividend"), exact_integer(b, "positive modulus")
    if b <= 0:
        raise ValueError("The balanced-remainder modulus must be positive")
    q, r = divmod(a, b)
    if 2 * r > b:
        q, r = q + 1, r - b
    return q, r


@dataclass
class PolynomialBasisOutput:
    """Detached output: all polynomials are copies, not replacements for P."""
    backend: IntegerPolynomialRing
    source_size: int
    minimal_basis: list
    minimal_leading_terms: list
    source_indices: list[int]       # zero-based P indices for the minimal representatives
    source_signs: list[int]         # only unit multipliers +1/-1
    reduced_basis: list | None
    statistics: dict

    def to_dict(self, include_basis: bool = True) -> dict:
        b = self.backend
        data = {
            "format": "gvwzz-polynomial-bases-v1",
            "status": "completed",
            "coefficient_ring": "ZZ",
            "polynomial_order": b.order_name,
            "scope": "Polynomial projection of P only; original signature records are unchanged",
            "source_requirement": "The completed polynomial projection P0 must be a strong Groebner basis",
            "source_basis_verified_by_postprocessing": False,
            "certification_note": "Post-processing is not a source certificate. Verify the original P/S and its vectors or paths separately",
            "normalization": "Positive leading coefficients; only unit signs are normalized",
            "minimality": "No leading term divides another; coefficients are included",
            "reduced_convention": "balanced-positive-ties: -b/2 < r <= b/2 for positive b",
            "source_basis_size": self.source_size,
            "minimal_basis_size": len(self.minimal_basis),
            "reduced_basis_size": None if self.reduced_basis is None else len(self.reduced_basis),
            "minimal_source_records": [i + 1 for i in self.source_indices],
            "minimal_source_signs": list(self.source_signs),
            "checks": {
                "leading_term_antichain": True,
                "source_leading_terms_covered": True,
                "positive_leading_coefficients": True,
                "reduced_leading_terms_unchanged": None if self.reduced_basis is None else True,
                "reduced_terms_irreducible": None if self.reduced_basis is None else True,
            },
            "statistics": dict(self.statistics),
        }
        if include_basis:
            data["minimal_basis"] = [b.serialize(f) for f in self.minimal_basis]
            data["minimal_leading_terms"] = [b.serialize(f) for f in self.minimal_leading_terms]
            if self.reduced_basis is not None:
                data["reduced_basis"] = [b.serialize(f) for f in self.reduced_basis]
        return data

    def to_json_dict(self, include_basis: bool = True, *, _source_json=None) -> dict:
        """Return the same JSON records without materializing every polynomial.

        The output writer consumes one polynomial at a time. Public ``to_dict``
        remains eager and returns ordinary JSON-compatible lists as before.

        ``_source_json`` is private to the CLI's immediate save/postprocess
        sequence: it must be the receipt for the unchanged source used to
        create this unmodified output. Positive-sign representatives can reuse
        that saved polynomial record. Sign changes and reduced tails require
        fresh serialization. Default public serialization never trusts a file.
        """
        from .output import JsonSavedPolynomial, JsonSequence
        data = self.to_dict(include_basis=False)
        if include_basis:
            b = self.backend

            def minimal_records():
                for j, f in enumerate(self.minimal_basis):
                    source_index = self.source_indices[j]
                    if (_source_json is not None
                            and getattr(self, '_json_source_identity', None) is not None
                            and _source_json.basis_source_identity is self._json_source_identity
                            and self.source_signs[j] > 0
                            and _source_json.has_saved_polynomial(source_index)):
                        yield JsonSavedPolynomial(_source_json, source_index)
                    else:
                        yield b.serialize(f)

            data["minimal_basis"] = JsonSequence(minimal_records)
            data["minimal_leading_terms"] = JsonSequence(
                lambda: (b.serialize(f) for f in self.minimal_leading_terms))
            if self.reduced_basis is not None:
                data["reduced_basis"] = JsonSequence(
                    lambda: (b.serialize(f) for f in self.reduced_basis))
        return data


def _validate_minimal_projection(descriptors, kept, nvars):
    """Independently check coverage and the complete-LT antichain.

    Selection uses the incremental threshold-exclusion index. Validation
    builds a separate, static prefix-inclusion index from the final output;
    it does not trust the selection witnesses, sort order or active masks.
    Every positive witness is also checked with the ordinary exact predicate.
    The static index is built once, so its prefixes are never rebuilt between
    insertions and queries.
    """
    index = DivisorIndex(nvars)
    buckets = {}
    for j, row in enumerate(kept):
        m = row[0]
        buckets.setdefault(m, []).append((j, row))
    for m, bucket in buckets.items():
        index.set(m, bucket)
    for m, c, _, _ in descriptors:
        if not any(c % row[1] == 0 and divides_monomial(row[0], m)
                   for bucket in index.divisors(m) for _, row in bucket):
            raise PolynomialPostprocessingError("Minimal extraction lost a source leading-term divisor")
    for j, (m, c, _, _) in enumerate(kept):
        for bucket in index.divisors(m):
            for k, (n, d, _, _) in bucket:
                if j != k and c % d == 0 and divides_monomial(n, m):
                    raise PolynomialPostprocessingError("Minimal leading terms are not an antichain")


def _minimal_projection(result):
    b = result.backend
    descriptors = []
    for i, label in enumerate(result.basis):
        m, c = b.leading(label.polynomial)
        descriptors.append((m, abs(c), i, 1 if c > 0 else -1))
    # A term divisor occurs earlier: smaller monomial, or same monomial and
    # smaller positive coefficient. Equal LTs keep the earliest P record.
    descriptors.sort(key=lambda row: (b.order_key(row[0]), row[1], row[2]))
    kept = []
    index = SyzygyIndex(len(b.names))
    buckets = {}
    comparisons = 0
    for row in descriptors:
        m, c, _, _ = row
        witness = None
        # Buckets and their rows are in original kept order. Same-monomial
        # records are contiguous in the descriptor sort, even over ZZ where
        # several incomparable coefficients may have to be retained.
        for bucket in index.divisors(m):
            for j, d in bucket:
                if c % d == 0:
                    witness = j
                    break
            if witness is not None:
                break
        # Retain the historical counter (the number of comparisons of the
        # reference scan), although indexing skips the irrelevant predicates.
        comparisons += len(kept) if witness is None else witness + 1
        if witness is None:
            if m not in buckets:
                bucket = buckets[m] = []
                index.set(m, bucket)
            buckets[m].append((len(kept), c))
            kept.append(row)
    _validate_minimal_projection(descriptors, kept, len(b.names))
    polynomials = [result.basis[i].polynomial.copy() if sign > 0
                   else -result.basis[i].polynomial for _, _, i, sign in kept]
    return kept, polynomials, comparisons


def _reduce_minimal(minimal, b):
    """Reduce tails in increasing leading-monomial order.

    A tail reducer has smaller LM than its target. Already reduced earlier
    elements suffice; later changes cannot affect earlier polynomials. Every
    replacement has the same LT and preserves the generated ideal. Thus an
    initially strong minimal basis stays strong. A minimal strong ZZ basis
    with positive LCs has no reducible leading coefficient. Check this explicitly.
    """
    leads = [b.leading(f) for f in minimal]
    by_coefficient = sorted(range(len(leads)), key=lambda i: (leads[i][1], i))
    for i, (m, c) in enumerate(leads):
        for j in by_coefficient:
            n, d = leads[j]
            if j != i and divides_monomial(n, m) and balanced_divmod(c, d)[0]:
                raise PolynomialPostprocessingError(
                    "Source projection is not a minimal strong basis: a leading coefficient "
                    "is reducible. Verify the original source; no completion is performed here."
                )

    @lru_cache(maxsize=8192)
    def reducer_index(m):
        for j in by_coefficient:
            if divides_monomial(leads[j][0], m):
                return j
        return None

    output = []
    steps = exact_steps = remainder_steps = tie_steps = terms_examined = 0
    try:
        for i, f in enumerate(minimal):
            workspace = PolynomialWorkspace(b, f)
            workspace.finish_term()   # protect the original leading term
            while workspace.leading() is not None:
                m, c = workspace.leading()
                terms_examined += 1
                j = reducer_index(m)
                if j is not None:
                    if j >= i:
                        raise PolynomialPostprocessingError("A tail reducer does not have smaller leading monomial")
                    n, d = leads[j]
                    q, r = balanced_divmod(c, d)
                    if q:
                        workspace.subtract_multiple(output[j], q, subtract_monomials(m, n))
                        steps += 1
                        if r == 0:
                            exact_steps += 1
                        else:
                            remainder_steps += 1
                            if abs(r) == abs(c):
                                tie_steps += 1
                        # The remaining coefficient is canonical modulo the
                        # smallest positive applicable LC, hence irreducible by
                        # every single applicable LC. Lower terms cannot change it.
                # The heap still has the processed term, even when cancelled.
                workspace.finish_term()
            g = workspace.result()
            if b.leading(g) != leads[i]:
                raise PolynomialPostprocessingError("Post-processing changed a minimal leading term")
            output.append(g)
        # Explicitly test all terms, including heads, against every other LT.
        for i, f in enumerate(output):
            for m, c in f.items():
                for j in by_coefficient:
                    n, d = leads[j]
                    if i != j and divides_monomial(n, m):
                        if balanced_divmod(c, d)[0]:
                            raise PolynomialPostprocessingError("Reduced-basis irreducibility check failed")
                        # This is the smallest applicable positive LC. All
                        # remaining moduli are larger, so their canonical
                        # intervals contain this one as well.
                        break
    finally:
        reducer_index.cache_clear()
    return output, {
        "reduction_steps": steps,
        "exact_reduction_steps": exact_steps,
        "coefficient_remainder_steps": remainder_steps,
        "equal_absolute_value_tie_steps": tie_steps,
        "terms_examined": terms_examined,
    }


def postprocess_basis(result: Result, *, reduced: bool = False) -> PolynomialBasisOutput:
    """Return the minimal strong basis, its LTs, and optionally the reduced basis.

    The supplied Result is not mutated. Partial executions are rejected. These
    are ordinary polynomial ideal bases, NOT minimal signature bases or syzygy
    bases. Validity as bases is conditional on the source being a strong basis.
    Timings are separate from Result.stats, and solver budgets/tail caps do not
    truncate this computation. KeyboardInterrupt propagates rather than marking
    a partly reduced list as a reduced Groebner basis.
    """
    if type(reduced) is not bool:
        raise ValueError("reduced must be a boolean")
    from .validation import result_structure_errors
    start = perf_counter()
    if getattr(result, "status", None) != "completed":
        raise PolynomialPostprocessingError("Polynomial basis post-processing requires a completed source result")
    errors = result_structure_errors(result)
    if errors:
        raise PolynomialPostprocessingError("Invalid source result: " + "; ".join(errors))
    validated = perf_counter()
    kept, minimal, comparisons = _minimal_projection(result)
    leading = [result.backend.term(c, m) for m, c, _, _ in kept]
    middle = perf_counter()
    reduced_basis = None
    counters = {"reduction_steps": 0, "exact_reduction_steps": 0,
                "coefficient_remainder_steps": 0, "equal_absolute_value_tie_steps": 0,
                "terms_examined": 0}
    if reduced:
        reduced_basis, counters = _reduce_minimal(minimal, result.backend)
    end = perf_counter()
    counters.update(minimal_seconds=middle - start,
                    source_validation_seconds=validated - start,
                    minimal_selection_seconds=middle - validated,
                    reduced_seconds=(end - middle) if reduced else 0.0,
                    total_seconds=end - start,
                    leading_term_comparisons=comparisons,
                    source_records_removed=len(result.basis) - len(minimal))
    output = PolynomialBasisOutput(result.backend, len(result.basis), minimal, leading,
                                   [row[2] for row in kept], [row[3] for row in kept],
                                   reduced_basis, counters)
    # An opaque token binds the CLI's immediate saved-source reuse to this
    # result, without keeping the result or its solver workspaces alive.
    output._json_source_identity = getattr(result, '_json_source_identity', None)
    return output


def minimal_groebner_basis(result: Result) -> list:
    """Return a normalized minimal strong basis of the polynomial projection."""
    return postprocess_basis(result).minimal_basis


def reduced_groebner_basis(result: Result) -> list:
    """Return the balanced-positive-ties reduced strong polynomial basis."""
    return postprocess_basis(result, reduced=True).reduced_basis


def minimal_leading_terms(result: Result) -> list:
    """Return full minimal leading terms, including their integer coefficients."""
    return postprocess_basis(result).minimal_leading_terms
