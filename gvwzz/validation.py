"""Checks at public and serialization boundaries, before indices are built."""
from __future__ import annotations

from .core import Label, Signature, exact_integer, validate_monomial
from .orders import ModuleOrder
from .polynomial import IntegerPolynomialRing


def validate_signature(signature, nvars: int, ncomponents: int) -> None:
    if not isinstance(signature, Signature):
        raise ValueError("Expected a Signature record")
    if exact_integer(signature.coefficient, "signature coefficient") == 0:
        raise ValueError("Signature coefficient must be nonzero")
    i = exact_integer(signature.index, "signature index")
    if not 0 <= i < ncomponents:
        raise ValueError("Signature component is outside the input module")
    validate_monomial(signature.monomial, nvars, "signature exponents")


def result_structure_errors(result) -> list[str]:
    """Return diagnostic errors instead of running arithmetic on malformed data."""
    errors = []
    try:
        b, order = result.backend, result.module_order
        if not isinstance(b, IntegerPolynomialRing) or not isinstance(order, ModuleOrder):
            raise ValueError("Result requires a valid integer ring and module order")
        if not isinstance(result.inputs, (list, tuple)):
            raise ValueError("Result inputs must be a list")
        for f in result.inputs:
            b.validate_polynomial(f)
        order.validate_for(b, result.inputs)
        if not isinstance(result.basis, (list, tuple)) or not isinstance(result.syzygies, (list, tuple)):
            raise ValueError("Basis and syzygies must be lists")
    except (ValueError, TypeError, AttributeError, IndexError) as exc:
        return [str(exc)]
    n, m = len(b.names), len(result.inputs)
    for i, p in enumerate(result.basis):
        try:
            if not isinstance(p, Label):
                raise ValueError("Expected a Label record")
            validate_signature(p.signature, n, m)
            b.validate_polynomial(p.polynomial, nonzero=True)
        except (ValueError, TypeError, AttributeError) as exc:
            errors.append(f"P[{i}]: {exc}")
    for i, s in enumerate(result.syzygies):
        try:
            validate_signature(s, n, m)
        except (ValueError, TypeError, AttributeError) as exc:
            errors.append(f"S[{i}]: {exc}")
    for field, count in (("basis_representations", len(result.basis)),
                         ("syzygy_representations", len(result.syzygies))):
        vectors = getattr(result, field, None)
        if vectors is None:
            continue
        if not isinstance(vectors, (list, tuple)) or len(vectors) != count:
            errors.append(f"{field}: wrong number of vectors")
            continue
        for i, vector in enumerate(vectors):
            try:
                if not isinstance(vector, (list, tuple)) or len(vector) != m:
                    raise ValueError("Representation vector has wrong module dimension")
                for component in vector:
                    b.validate_polynomial(component)
            except (ValueError, TypeError, AttributeError) as exc:
                errors.append(f"{field}[{i}]: {exc}")
    if ((getattr(result, "basis_representations", None) is None) !=
            (getattr(result, "syzygy_representations", None) is None)):
        errors.append("A certificate must include both P and S representations")
    return errors
