"""One compact callable per candidate, with arbitrary-width packed shifts."""


cdef class JPolynomialRecipe:
    cdef object context, polynomial, coefficient, shift
    cdef bint packed

    def __init__(self, context, polynomial, coefficient, shift, packed=True):
        self.context = context
        self.polynomial, self.coefficient, self.shift = polynomial, coefficient, shift
        self.packed = packed
        context.counts['candidate_recipes_compact'] += 1
        context.counts['candidate_recipes_native'] += 1

    def _multiply(self):
        cdef object shift = (self.context.packing.unpack(self.shift)
                             if self.packed else self.shift)
        return self.context.backend._multiply_term_from_arithmetic(
            self.polynomial, self.coefficient, shift)

    def __call__(self):
        return self.context.materialize(self._multiply, 'J')


cdef class GPolynomialRecipe:
    cdef object context, polynomial_p, a, up, polynomial_q, b, uq

    def __init__(self, context, polynomial_p, a, up, polynomial_q, b, uq):
        self.context = context
        self.polynomial_p, self.a, self.up = polynomial_p, a, up
        self.polynomial_q, self.b, self.uq = polynomial_q, b, uq
        context.counts['candidate_recipes_compact'] += 1
        context.counts['candidate_recipes_native'] += 1

    def _combine(self):
        cdef object multiply = self.context.backend._multiply_term_from_arithmetic
        return (multiply(self.polynomial_p, self.a, self.up)
                + multiply(self.polynomial_q, self.b, self.uq))

    def __call__(self):
        return self.context.materialize(self._combine, 'G')
