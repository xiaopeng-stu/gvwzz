"""Optional exact representation witnesses, separate from signature-only labels.

No witness participates in candidate selection, cover tests or reduction lookup.
The final verifier checks the saved vectors rather than replaying the solver.
"""
from __future__ import annotations

from math import gcd
from .core import Signature, lcm_monomials, subtract_monomials


class WitnessTracker:
    def __init__(self, backend, ncomponents: int):
        self.backend = backend
        self.ncomponents = ncomponents
        self.current = None
        self.basis = []
        self.syzygies = {}

    def unit(self, index):
        b = self.backend
        return tuple(b.one.copy() if i == index else b.zero.copy()
                     for i in range(self.ncomponents))

    def scale(self, vector, coefficient, monomial):
        return tuple(self.backend.multiply_term(f, coefficient, monomial) for f in vector)

    @staticmethod
    def add(left, right):
        return tuple(a + b for a, b in zip(left, right))

    def combination(self, i, a, mu, j, b, nu):
        return self.add(self.scale(self.basis[i], a, mu), self.scale(self.basis[j], b, nu))

    def subtract(self, i, coefficient, monomial):
        if self.current is None:
            raise RuntimeError("Missing representation of the current candidate")
        reducer = self.scale(self.basis[i], coefficient, monomial)
        self.current = tuple(a - b for a, b in zip(self.current, reducer))

    def append_basis(self):
        if self.current is None:
            raise RuntimeError("Missing representation of inserted pair")
        self.basis.append(tuple(f.copy() for f in self.current))

    def update_syzygies(self, signature, vector, old, retained):
        """Produce witnesses only for the signatures the existing store retained."""
        seed = signature.positive()
        if signature.coefficient < 0:
            vector = self.scale(vector, -1, self.backend.zero_monomial)
        new = dict(self.syzygies)
        for h in retained:
            if h in self.syzygies:
                continue
            if h == seed:
                new[h] = vector
                continue
            for s in old:
                L = lcm_monomials(seed.monomial, s.monomial)
                if h != Signature(gcd(seed.coefficient, s.coefficient), L, seed.index):
                    continue
                a, b, _ = self.backend.bezout(seed.coefficient, s.coefficient)
                new[h] = self.add(
                    self.scale(vector, a, subtract_monomials(L, seed.monomial)),
                    self.scale(self.syzygies[s], b, subtract_monomials(L, s.monomial)),
                )
                break
            else:
                raise RuntimeError("Missing construction for a retained syzygy signature")
        keep = set(retained)
        self.syzygies = {s: v for s, v in new.items() if s.index != seed.index or s in keep}
