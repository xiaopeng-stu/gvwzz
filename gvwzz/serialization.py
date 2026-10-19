"""Exact polynomial output with a bounded cache of printed monomials.

Only presentation is accelerated. Sparse terms, coefficient encoding, and their
order are unchanged; unsupported symbol spelling uses the reference printer.
"""
from __future__ import annotations

from functools import lru_cache
import re
import weakref

from .core import DECIMAL_BOUND, encode_integer, exact_integer


def make_polynomial_serializer(backend):
    import sympy

    backend_ref = weakref.ref(backend)

    if (sympy.__version__ != "1.14.0"
            or not all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name)
                       for name in backend.names)):
        def reference(polynomial):
            return backend_ref()._serialize_reference(polynomial)
        return reference

    ring = backend.ring

    @lru_cache(maxsize=32768)
    def monomial_text(monomial):
        # The pinned SymPy printer still owns symbol and exponent spelling.
        return str(ring.from_dict({monomial: ring.domain.one}))

    def serialize(polynomial):
        backend_ref().validate_polynomial(polynomial)
        terms = polynomial.terms()
        records = [{"coefficient": encode_integer(coefficient),
                    "exponents": list(monomial)}
                   for monomial, coefficient in terms]
        if any(abs(exact_integer(coefficient)) >= DECIMAL_BOUND
               for _, coefficient in terms):
            expression = None
        else:
            fragments = []
            try:
                for monomial, coefficient in terms:
                    value = exact_integer(coefficient)
                    magnitude = abs(value)
                    if monomial == ring.zero_monom:
                        body = str(magnitude)
                    else:
                        body = monomial_text(monomial)
                        if magnitude != 1:
                            body = str(magnitude) + "*" + body
                    if fragments:
                        fragments.append(" - " if value < 0 else " + ")
                    elif value < 0:
                        fragments.append("-")
                    fragments.append(body)
                expression = "".join(fragments) if fragments else "0"
            except ValueError:
                expression = None
        return {"expression": expression, "terms": records}

    return serialize
