"""Polynomial backend and the three reduction workspaces.

All polynomial arithmetic is delegated to SymPy 1.14.0 over ``ZZ``.
The workspaces only organize transient sparse updates; retained basis
polynomials are never mutated.
"""
from __future__ import annotations

import ast
import heapq
import keyword
from collections import OrderedDict
from functools import lru_cache
from typing import Any, Iterable

from sympy.polys.domains import ZZ
from sympy.polys.rings import PolyElement, ring

from .core import (Monomial, encode_integer, decode_integer, exact_integer,
                   validate_monomial, DECIMAL_BOUND)

ORDER_ALIASES = {"lex": "lex", "grlex": "grlex", "deglex": "grlex",
                 "grevlex": "grevlex", "degrevlex": "grevlex"}


class IntegerPolynomialRing:
    def __init__(self, names: Iterable[str], order: str = "grevlex") -> None:
        if isinstance(names, (str, bytes)):
            raise ValueError("Variable names must be a sequence, not a single string")
        self.names = tuple(names)
        if any(not isinstance(name, str) for name in self.names):
            raise ValueError("Variable names must be strings")
        if not isinstance(order, str):
            raise ValueError("Polynomial order must be a string")
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("Specify distinct variable names (at least one)")
        if any(not x.isidentifier() or keyword.iskeyword(x) or x.startswith('_')
               for x in self.names):
            raise ValueError("Variable names must be ordinary Python identifiers")
        try:
            self.order_name = ORDER_ALIASES[order.lower()]
        except KeyError as exc:
            raise ValueError(f"Unknown polynomial order: {order}") from exc
        self.ring, *self.gens = ring(','.join(self.names), ZZ, order=self.order_name)
        self.zero = self.ring.zero
        self.one = self.ring.one
        self.zero_monomial = (0,) * len(self.names)
        # Cache derived order keys only. Polynomial arithmetic stays in SymPy.
        # Negative exponents also occur in formal-ratio comparisons.
        self.order_key = lru_cache(maxsize=32768)(self.ring.order)
        self._variables = dict(zip(self.names, self.gens))
        self._support_cache = OrderedDict()
        self._support_cache_terms = 0
        self.support_cache_max_terms = 100000

    def parse(self, text: str | int | PolyElement) -> PolyElement:
        if isinstance(text, PolyElement):
            self.validate_polynomial(text)
            return text.copy()
        if isinstance(text, int) and not isinstance(text, bool):
            return self.ring(text)
        if not isinstance(text, str):
            raise TypeError("Polynomials must be integer expressions or ring elements")
        if len(text) > 200_000:
            raise ValueError("Polynomial expression is too long")
        try:
            node = ast.parse(text.replace('^', '**'), mode='eval')
            if sum(1 for _ in ast.walk(node)) > 30_000:
                raise ValueError("Polynomial expression has too many syntax nodes")
            return self._parse_node(node.body)
        except (SyntaxError, RecursionError) as exc:
            raise ValueError(f"Invalid polynomial expression: {text[:160]}") from exc

    def _parse_node(self, node: ast.AST) -> PolyElement:
        # Deliberately no eval/sympify: reject function calls, attributes, division,
        # floats, negative exponents and all executable syntax.
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return self.ring(node.value)
        if isinstance(node, ast.Name) and node.id in self._variables:
            return self._variables[node.id]
        if isinstance(node, ast.UnaryOp):
            q = self._parse_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return q
            if isinstance(node.op, ast.USub):
                return -q
        if isinstance(node, ast.BinOp):
            q = self._parse_node(node.left)
            if isinstance(node.op, ast.Pow):
                if not isinstance(node.right, ast.Constant) or type(node.right.value) is not int:
                    raise ValueError("Exponent must be a nonnegative integer literal")
                e = node.right.value
                if not 0 <= e <= 10_000:
                    raise ValueError("Exponent must be between 0 and 10000")
                return q ** e
            r = self._parse_node(node.right)
            if isinstance(node.op, ast.Add):
                return q + r
            if isinstance(node.op, ast.Sub):
                return q - r
            if isinstance(node.op, ast.Mult):
                return q * r
        raise ValueError("Use only integer coefficients, declared variables, +, -, *, ^ or **")

    def validate_polynomial(self, p: PolyElement, *, nonzero: bool = False) -> None:
        """Validate a public input or saved record before any arithmetic/indexing."""
        if not isinstance(p, PolyElement) or p.ring != self.ring:
            raise ValueError("Polynomial belongs to a different ring/order (expected ZZ)")
        for m, c in p.items():
            validate_monomial(m, len(self.names), "polynomial exponents")
            if exact_integer(c, "polynomial coefficient") == 0:
                raise ValueError("Sparse polynomial contains an explicit zero term")
        if nonzero and not p:
            raise ValueError("P must contain only nonzero polynomials")

    def deserialize(self, data: Any, *, allow_expression: bool = True) -> PolyElement:
        """Read exact sparse terms. The optional expression is display-only.

        Legacy expression strings are accepted only through the existing safe
        parser. Sparse records have no parser-literal degree cap.
        """
        if not isinstance(data, dict):
            if not allow_expression:
                raise ValueError("Expected an exact sparse polynomial record")
            return self.parse(data)
        if not isinstance(data.get("terms"), list):
            raise ValueError("Sparse polynomial requires a list of terms")
        if "expression" in data and data["expression"] is not None and not isinstance(data["expression"], str):
            raise ValueError("Polynomial expression must be a string or null")
        terms = {}
        for term in data["terms"]:
            if not isinstance(term, dict) or "exponents" not in term or "coefficient" not in term:
                raise ValueError("Every polynomial term needs coefficient and exponents")
            m = validate_monomial(term["exponents"], len(self.names), "polynomial exponents")
            c = decode_integer(term["coefficient"])
            if c == 0:
                raise ValueError("Sparse polynomial terms must have nonzero coefficients")
            if m in terms:
                raise ValueError("Duplicate exponent vector in sparse polynomial")
            terms[m] = ZZ(c)
        return self.ring.from_dict(terms)

    def leading(self, p: PolyElement) -> tuple[Monomial, int]:
        if not isinstance(p, PolyElement) or p.ring != self.ring:
            raise ValueError("Polynomial belongs to a different ring/order (expected ZZ)")
        if not p:
            raise ValueError("Zero polynomial has no nonzero leading term")
        m = p.leading_expv()
        return m, exact_integer(p[m], "leading coefficient")

    def multiply_term(self, p: PolyElement, c: int, m: Monomial) -> PolyElement:
        if not isinstance(p, PolyElement) or p.ring != self.ring:
            raise ValueError("Polynomial belongs to a different ring/order (expected ZZ)")
        m = validate_monomial(m, len(self.names))
        return p.mul_term((m, ZZ(exact_integer(c, "coefficient"))))

    def _multiply_term_from_arithmetic(self, p: PolyElement, c: int,
                                       m: Monomial) -> PolyElement:
        """Internal multiplication of already checked solver records.

        Parent polynomials belong to this ring; the scalar and nonnegative
        shift come from exact coefficient division and monomial arithmetic.
        Keep public ``multiply_term`` validation at the input boundary.
        """
        return p.mul_term((m, ZZ(c)))

    def term(self, c: int, m: Monomial) -> PolyElement:
        m = validate_monomial(m, len(self.names))
        c = exact_integer(c, "coefficient")
        return self.ring.from_dict({m: ZZ(c)}) if c else self.zero.copy()

    @staticmethod
    def bezout(a: int, b: int) -> tuple[int, int, int]:
        a, b = exact_integer(a, "Bézout coefficient input"), exact_integer(b, "Bézout coefficient input")
        s, t, g = ZZ.gcdex(ZZ(a), ZZ(b))
        return int(s), int(t), int(g)

    @staticmethod
    def total_degree(p: PolyElement) -> int:
        return max((sum(m) for m in p), default=0)

    def shifted_support(self, polynomial, shift):
        """Cache immutable shifted exponent tuples, not polynomial arithmetic.

        Retain a reference to each source polynomial to prevent Python id reuse.
        Storage is bounded by the number of exponent vectors, not entry count.
        """
        if not any(shift):
            return polynomial.keys()
        key = (id(polynomial), shift)
        entry = self._support_cache.get(key)
        if entry is not None:
            self._support_cache.move_to_end(key)
            return entry[1]
        mul = self.ring.monomial_mul
        terms = frozenset(mul(m, shift) for m in polynomial)
        budget = self.support_cache_max_terms
        if len(terms) <= budget:
            self._support_cache[key] = (polynomial, terms)
            self._support_cache_terms += len(terms)
            while self._support_cache_terms > budget:
                _, (_, old) = self._support_cache.popitem(last=False)
                self._support_cache_terms -= len(old)
        return terms


    def serialize(self, p: PolyElement) -> dict[str, Any]:
        serializer = getattr(self, "_output_serializer", None)
        if serializer is None:
            from .serialization import make_polynomial_serializer
            serializer = self._output_serializer = make_polynomial_serializer(self)
        return serializer(p)

    def _serialize_reference(self, p: PolyElement) -> dict[str, Any]:
        """Pinned backend output, retained for fallback and differential checks."""
        self.validate_polynomial(p)
        # Display strings are not mathematical data. Avoid making even this
        # optional field depend on the writer's decimal conversion guard.
        if any(abs(exact_integer(c)) >= DECIMAL_BOUND for c in p.values()):
            expression = None
        else:
            try:
                expression = str(p)
            except ValueError:
                expression = None  # Exact sparse terms remain fully available.
        return {"expression": expression, "terms": [
            {"coefficient": encode_integer(c), "exponents": list(m)}
            for m, c in p.terms()]}

# The implementation is pinned to SymPy 1.14.0 and uses this exact sparse
# fused add-multiply primitive.
fused_add_multiple = PolyElement._iadd_poly_monom

@lru_cache(maxsize=65536)
def reverse_key(order: str, m: tuple[int,...]) -> tuple:
    if order == 'lex':
        return tuple(-x for x in m)
    if order == 'grlex':
        return (-sum(m), tuple(-x for x in m))
    return (-sum(m), tuple(reversed(m)))

class PolynomialWorkspace:
    def __init__(self, backend, polynomial):
        self.backend = backend
        self.polynomial = polynomial.copy()
        self.pending = set(polynomial)
        self.heap = [(reverse_key(backend.order_name, m), m) for m in polynomial]
        heapq.heapify(self.heap)

    def leading(self):
        while self.heap:
            m = self.heap[0][1]
            c = self.polynomial.get(m, 0)
            if c:
                return m, int(c)
            heapq.heappop(self.heap)
            self.pending.remove(m)
        return None

    def finish_term(self):
        """Keep an irreducible term in the polynomial but stop considering it."""
        _, m = heapq.heappop(self.heap)
        self.pending.remove(m)

    def subtract_multiple(self, other, coefficient, shift):
        # Fused sparse add-multiply provided by the pinned backend, avoiding
        # materialization of the product and a whole-polynomial copy per step.
        if hasattr(self.polynomial, '_iadd_poly_monom'):
            self.polynomial = fused_add_multiple(self.polynomial, other, (shift, ZZ(-coefficient)))
        else:
            self.polynomial = self.polynomial - self.backend.multiply_term(other, coefficient, shift)
        # Most shifted monomials are already queued. Do this set difference
        # in CPython rather than visiting every support term in a Python loop.
        missing = self.backend.shifted_support(other, shift) - self.pending
        for new in missing:
            if self.polynomial.get(new, 0):
                heapq.heappush(self.heap, (reverse_key(self.backend.order_name, new), new))
                self.pending.add(new)

    def result(self):
        # SymPy's mutating primitive may have cached a hash while checking for
        # generator objects. This polynomial has not been used as a hash key.
        self.polynomial._hash = None
        return self.polynomial

# A Python global deliberately prevents fixed-width C shifts in source builds.
_PY_ONE = 1


class PackedContext:
    def __init__(self, backend, bits: int, cache_terms: int = 500_000,
                 support_terms: int = 200_000):
        if backend.order_name not in ('grlex', 'grevlex'):
            raise ValueError('PackedContext requires a graded monomial order')
        if bits < 1:
            raise ValueError('bits must be positive')
        self.backend = backend
        self.bits = bits
        self.n = len(backend.names)
        self.mask = (_PY_ONE << bits) - 1
        self.high = bits * self.n
        self.high_unit = _PY_ONE << self.high
        self.ring, _ = ring('_gvw_packed', ZZ, order='lex')
        self.cache = OrderedDict()
        self.terms = 0
        self.limit = cache_terms
        self.support_cache = OrderedDict()
        self.support_terms = 0
        self.support_limit = support_terms
        self.encode = lru_cache(maxsize=65536)(self._encode)
        self.decode = lru_cache(maxsize=65536)(self._decode)

    def _encode(self, monomial: tuple[int, ...]) -> int:
        if len(monomial) != self.n or any(e < 0 for e in monomial):
            raise ValueError('Invalid exponent vector')
        degree = sum(monomial)
        if degree > self.mask:
            raise ValueError('Exponent packing degree bound exceeded')
        value = 0
        if self.backend.order_name == 'grevlex':
            for exponent in reversed(monomial):
                value = (value << self.bits) | exponent
            return (degree << self.high) - value
        for exponent in monomial:
            value = (value << self.bits) | exponent
        return (degree << self.high) + value

    def _decode(self, value: int) -> tuple[int, ...]:
        if value < 0:
            raise ValueError('Negative packed exponent')
        if self.backend.order_name == 'grevlex':
            degree = (value + self.high_unit - 1) >> self.high
            digits = (degree << self.high) - value
            out = []
            for _ in range(self.n):
                out.append(digits & self.mask)
                digits >>= self.bits
            return tuple(out)
        digits = value & (self.high_unit - 1)
        out = []
        for _ in range(self.n):
            out.append(digits & self.mask)
            digits >>= self.bits
        return tuple(reversed(out))

    def polynomial_entry(self, polynomial):
        """Cache immutable source polynomials, their encoding, and total degree."""
        key = id(polynomial)
        entry = self.cache.get(key)
        if entry is not None:
            self.cache.move_to_end(key)
            return entry[1], entry[2]
        degree = max((sum(m) for m in polynomial), default=0)
        if degree > self.mask:
            raise ValueError('Reducer exceeds the exponent packing capacity')
        encoded = self.ring.from_dict({(self.encode(m),): c
                                       for m, c in polynomial.items()})
        if len(encoded) <= self.limit:
            self.cache[key] = (polynomial, encoded, degree)
            self.terms += len(encoded)
            while self.terms > self.limit:
                _, old = self.cache.popitem(last=False)
                self.terms -= len(old[1])
        return encoded, degree

    def shifted_support(self, polynomial, shift: int):
        if shift == 0:
            return polynomial.keys()
        key = (id(polynomial), shift)
        entry = self.support_cache.get(key)
        if entry is not None:
            self.support_cache.move_to_end(key)
            return entry[1]
        result = frozenset((m[0] + shift,) for m in polynomial)
        if len(result) <= self.support_limit:
            self.support_cache[key] = (polynomial, result)
            self.support_terms += len(result)
            while self.support_terms > self.support_limit:
                _, old = self.support_cache.popitem(last=False)
                self.support_terms -= len(old[1])
        return result


class PackedWorkspace:
    """Heap workspace, with the same interface as PolynomialWorkspace."""
    def __init__(self, backend, polynomial):
        if backend.order_name not in ('grlex', 'grevlex'):
            raise ValueError('packed reduction requires grlex or grevlex')
        self.backend = backend
        self.bound = max((sum(m) for m in polynomial), default=0)
        bits = max(1, self.bound.bit_length())
        contexts = getattr(backend, '_packed_contexts', None)
        if contexts is None:
            contexts = backend._packed_contexts = OrderedDict()
        if bits not in contexts:
            contexts[bits] = PackedContext(backend, bits)
            # Calls with different degree ranges need different encodings.
            # Limit retained ranges rather than letting caches grow forever.
            while len(contexts) > 3:
                contexts.popitem(last=False)
        contexts.move_to_end(bits)
        self.ctx = contexts[bits]
        # Always a privately owned accumulator, never an encoded cached reducer.
        self.polynomial = self.ctx.ring.from_dict({(self.ctx.encode(m),): c
                                                  for m, c in polynomial.items()})
        self.pending = set(self.polynomial)
        self.heap = [(-m[0], m) for m in self.polynomial]
        heapq.heapify(self.heap)

    def leading(self):
        while self.heap:
            m = self.heap[0][1]
            coefficient = self.polynomial.get(m, 0)
            if coefficient:
                return self.ctx.decode(m[0]), int(coefficient)
            heapq.heappop(self.heap)
            self.pending.remove(m)
        return None

    def finish_term(self):
        _, monomial = heapq.heappop(self.heap)
        self.pending.remove(monomial)

    def subtract_multiple(self, other, coefficient: int, shift: tuple[int, ...]):
        q, degree = self.ctx.polynomial_entry(other)
        if degree + sum(shift) > self.bound:
            raise ValueError('Reduction would violate the packed degree bound')
        mu = self.ctx.encode(shift)
        self.polynomial = fused_add_multiple(self.polynomial, q,
                                             ((mu,), ZZ(-coefficient)))
        missing = self.ctx.shifted_support(q, mu) - self.pending
        for monomial in missing:
            if self.polynomial.get(monomial, 0):
                heapq.heappush(self.heap, (-monomial[0], monomial))
                self.pending.add(monomial)

    def result(self):
        return self.backend.ring.from_dict({self.ctx.decode(m[0]): c
                                            for m, c in self.polynomial.items()})
