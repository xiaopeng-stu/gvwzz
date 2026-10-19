"""Exact compact descriptors for small exponents, with an unbounded fallback.

Each exponent occupies one little-endian byte. Its high bit is reserved for
independent lane comparisons, so stored query coordinates lie in [0, 127].
These descriptors are internal accelerators, never an external representation
or a restriction on polynomial/signature exponents.
"""
from __future__ import annotations


def pack_monomial(monomial: tuple[int, ...], bound: int = 127) -> int | None:
    """Pack checked, nonnegative exponents, or return None for the exact fallback."""
    if max(monomial, default=0) > bound:
        return None
    return int.from_bytes(bytes(monomial), 'little')


def unpack_monomial(packed: int, nvars: int) -> tuple[int, ...]:
    return tuple(packed.to_bytes(nvars, 'little'))


class PackedMonomials:
    """Byte-lane arithmetic on trusted nonnegative exponent vectors."""

    __slots__ = ('nvars', 'guard')

    def __init__(self, nvars: int) -> None:
        self.nvars = nvars
        self.guard = int.from_bytes(b'\x80' * nvars, 'little')

    def lcm(self, a: int, b: int) -> int:
        # Adding a guard before subtraction prevents inter-lane borrows.
        # Each guard bit selects the larger input in its own coordinate.
        high = ((a | self.guard) - b) & self.guard
        select_a = high - (high >> 7)
        return (a & select_a) | (b & ~select_a)

    def unpack(self, packed: int) -> tuple[int, ...]:
        return unpack_monomial(packed, self.nvars)
