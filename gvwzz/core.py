"""Core algebraic records and monomial arithmetic.

This module contains only transparent mathematical data: monomial arithmetic,
coefficient-bearing signatures, and labels.  Candidate construction is written
once, in the solver method that actually executes it.
"""
from __future__ import annotations

from dataclasses import dataclass
from operator import add, sub, index as integer_index
import re
from typing import Any, TypeAlias

Monomial: TypeAlias = tuple[int, ...]


def exact_integer(value: Any, name: str = "value") -> int:
    """Accept exact integer objects, never truncating rationals/floats or booleans."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an exact integer, not bool")
    try:
        return integer_index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an exact integer") from exc


def validate_monomial(value: Any, nvars: int | None = None,
                      name: str = "exponents") -> Monomial:
    if not isinstance(value, (tuple, list)):
        raise ValueError(f"{name} must be an exponent vector")
    if nvars is not None and len(value) != nvars:
        raise ValueError(f"{name} must have exactly {nvars} components")
    result = tuple(exact_integer(e, name) for e in value)
    if any(e < 0 for e in result):
        raise ValueError(f"{name} must contain nonnegative integers")
    return result


def _same_dimension(a: Monomial, b: Monomial) -> None:
    if len(a) != len(b):
        raise ValueError("Monomial exponent vectors have different dimensions")


# CPython's smallest configurable nonzero decimal digit limit is 640.
# Keep this threshold FIXED, independent of the reader/writer's environment.
# In particular, neither exceptions from str(int) nor sys settings choose the
# persisted representation. Changing this convention requires a format revision.
INTEGER_ENCODING = "gvwzz-decimal640-hex-v1"
DECIMAL_DIGITS = 640
DECIMAL_BOUND = 10 ** DECIMAL_DIGITS


def encode_integer(value: int) -> str:
    """Stable exact encoding: up to 640 decimal digits, otherwise signed hex.

    No global interpreter settings are read or changed. Small coefficients keep
    their historical decimal spelling. Large coefficients always use the same
    spelling on every supported interpreter, including when its guard is off.
    """
    value = exact_integer(value)
    return str(value) if -DECIMAL_BOUND < value < DECIMAL_BOUND else hex(value)


def _decode_decimal_digits(digits: str) -> int:
    """Decode validated unsigned legacy decimal digits in balanced small parts.

    Each built-in decimal conversion is at most 640 digits. Balanced assembly
    avoids the long sequential multiply-by-ten loop and does not disable the
    interpreter's process-wide safety setting. This is for exact mathematical
    data, not evaluation of arbitrary expressions.
    """
    if len(digits) <= DECIMAL_DIGITS:
        return int(digits)
    middle = len(digits) // 2
    left = _decode_decimal_digits(digits[:middle])
    right = _decode_decimal_digits(digits[middle:])
    return left * 10 ** (len(digits) - middle) + right


def decode_integer(value: str | int) -> int:
    """Decode current hex/decimal data and legacy long decimal coefficients.

    Accepts exact integer objects or strictly signed decimal/hex strings only.
    The string path is independent of sys.int_max_str_digits. It can also be
    used as json.loads(parse_int=...) when loading saved mathematical records.
    """
    if not isinstance(value, str):
        return exact_integer(value, "coefficient")
    if not re.fullmatch(r"[+-]?(?:[0-9]+|0[xX][0-9a-fA-F]+)", value):
        raise ValueError("A coefficient must be an integer, decimal string or hex string")
    if value.lower().startswith(("0x", "-0x", "+0x")):
        return int(value, 16)
    negative = value.startswith("-")
    digits = value[1:] if value.startswith(("+", "-")) else value
    result = _decode_decimal_digits(digits)
    return -result if negative else result


def divides_monomial(a: Monomial, b: Monomial) -> bool:
    _same_dimension(a, b)
    return all(x <= y for x, y in zip(a, b))


def add_monomials(a: Monomial, b: Monomial) -> Monomial:
    _same_dimension(a, b)
    return tuple(map(add, a, b))


def subtract_monomials(a: Monomial, b: Monomial) -> Monomial:
    """Subtract exponent vectors; negative entries are allowed for ratio keys."""
    _same_dimension(a, b)
    return tuple(map(sub, a, b))


def lcm_monomials(a: Monomial, b: Monomial) -> Monomial:
    _same_dimension(a, b)
    return tuple(x if x >= y else y for x, y in zip(a, b))


@dataclass(frozen=True, slots=True)
class Signature:
    """A complete signature ``coefficient * monomial * e_index``.

    ``index`` is zero-based internally.  Equality and divisibility include the
    coefficient; module-order comparisons are performed separately by
    :class:`gvwzz.orders.ModuleOrder` and ignore the coefficient.
    """

    coefficient: int
    monomial: Monomial
    index: int

    def __post_init__(self) -> None:
        coefficient = exact_integer(self.coefficient, "signature coefficient")
        index = exact_integer(self.index, "signature index")
        monomial = validate_monomial(self.monomial, name="signature exponents")
        if coefficient == 0:
            raise ValueError("A stored signature must have a nonzero coefficient")
        if index < 0:
            raise ValueError("Signature index must be nonnegative")
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "monomial", monomial)

    def multiply(self, coefficient: int, monomial: Monomial) -> Signature:
        return Signature(
            self.coefficient * exact_integer(coefficient, "signature multiplier"),
            add_monomials(self.monomial, monomial),
            self.index,
        )

    def divides(self, other: Signature) -> bool:
        return (
            self.index == other.index
            and other.coefficient % self.coefficient == 0
            and divides_monomial(self.monomial, other.monomial)
        )

    def positive(self) -> Signature:
        """Normalize a syzygy signature only by the unit ``-1``."""
        return self if self.coefficient > 0 else Signature(-self.coefficient, self.monomial, self.index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "coefficient": encode_integer(self.coefficient),
            "exponents": list(self.monomial),
            "index": self.index + 1,
        }


def _signature_from_arithmetic(coefficient: int, monomial: Monomial, index: int) -> Signature:
    """Internal constructor for arithmetic on already validated solver records.

    NOT an input/deserialization API. Callers must supply Python integer fields
    and a nonnegative exponent tuple of the solver dimension. These properties
    follow from the checked parents and monomial arithmetic at the call sites.
    Public Signature construction, loading, and verification retain validation.
    The nonzero check is still mandatory (e.g. a Bezout cancellation bug).
    """
    if coefficient == 0:
        raise ValueError("A stored signature must have a nonzero coefficient")
    result = object.__new__(Signature)
    object.__setattr__(result, 'coefficient', coefficient)
    object.__setattr__(result, 'monomial', monomial)
    object.__setattr__(result, 'index', index)
    return result


@dataclass(frozen=True, slots=True)
class Label:
    """The complete stored record: an exact signature and a polynomial part.

    The representation vector ``u`` is deliberately not stored.
    """

    signature: Signature
    polynomial: Any
