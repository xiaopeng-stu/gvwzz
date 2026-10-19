"""Compact, lazy polynomial recipes shared by Python and native emitters.

An accepted candidate owns one record, rather than two nested functions with
their own default-argument tuples. The shared context normally has no reference
to the solver or queue, and preserves materialization counters and budgets.
"""
from __future__ import annotations

import os
import weakref


class RecipeContext:
    __slots__ = ('backend', 'packing', 'counts', 'options', 'limit_error', '_override', '_source')

    def __init__(self, backend, packing, counts, options, limit_error, override=None, source=None):
        self.backend = backend
        self.packing = packing
        self.counts = counts
        self.options = options
        self.limit_error = limit_error
        # Preserve specialized subclass hooks; the ordinary path owns no
        # bound solver method and cannot create queue -> recipe -> solver.
        self._override = override
        # A weak reference observes a caller's later engine.options replacement
        # without retaining the engine. Retained candidates also work after it
        # has been collected, using their original immutable options.
        self._source = weakref.ref(source) if source is not None else None

    def materialize(self, factory, kind):
        if self._override is not None:
            return self._override(factory, kind)
        self.counts[kind.lower() + '_materialized'] += 1
        polynomial = factory()
        source = self._source() if self._source is not None else None
        check_polynomial(polynomial, source.options if source is not None else self.options,
                         self.limit_error)
        return polynomial


def check_polynomial(polynomial, options, limit_error):
    """The same polynomial budgets at insertion and lazy materialization."""
    if options.max_polynomial_terms is not None and len(polynomial) > options.max_polynomial_terms:
        raise limit_error('max_polynomial_terms exceeded')
    bound = options.max_coefficient_bits
    if bound is not None and any(abs(int(c)).bit_length() > bound for c in polynomial.values()):
        raise limit_error('max_coefficient_bits exceeded')


class PythonJPolynomialRecipe:
    __slots__ = ('context', 'polynomial', 'coefficient', 'shift', 'packed')

    def __init__(self, context, polynomial, coefficient, shift, packed=True):
        self.context = context
        self.polynomial = polynomial
        self.coefficient = coefficient
        self.shift = shift
        self.packed = packed
        context.counts['candidate_recipes_compact'] += 1

    def _multiply(self):
        context = self.context
        shift = context.packing.unpack(self.shift) if self.packed else self.shift
        return context.backend._multiply_term_from_arithmetic(
            self.polynomial, self.coefficient, shift)

    def __call__(self):
        return self.context.materialize(self._multiply, 'J')


class PythonGPolynomialRecipe:
    __slots__ = ('context', 'polynomial_p', 'a', 'up', 'polynomial_q', 'b', 'uq')

    def __init__(self, context, polynomial_p, a, up, polynomial_q, b, uq):
        self.context = context
        self.polynomial_p, self.a, self.up = polynomial_p, a, up
        self.polynomial_q, self.b, self.uq = polynomial_q, b, uq
        context.counts['candidate_recipes_compact'] += 1

    def _combine(self):
        multiply = self.context.backend._multiply_term_from_arithmetic
        return (multiply(self.polynomial_p, self.a, self.up)
                + multiply(self.polynomial_q, self.b, self.uq))

    def __call__(self):
        return self.context.materialize(self._combine, 'G')


JPolynomialRecipe = PythonJPolynomialRecipe
GPolynomialRecipe = PythonGPolynomialRecipe
try:
    if os.environ.get('GVWZZ_NO_NATIVE'):
        raise ImportError('native acceleration disabled')
    from ._native_pairs import JPolynomialRecipe, GPolynomialRecipe
except (ImportError, OSError):
    pass
