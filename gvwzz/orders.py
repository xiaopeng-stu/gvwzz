"""Compatible module orders, including GVW weights and a pure TOP/POT block order.

For block sizes ``(b1,...,br)``, input positions are partitioned consecutively.
Blocks are compared by POT; positions inside one block are compared by TOP.
Thus ``(1,...,1)`` is POT and ``(m,)`` is TOP.
"""
from __future__ import annotations
from functools import lru_cache
from typing import Iterable
from .core import Signature, Monomial, add_monomials, exact_integer
from .polynomial import IntegerPolynomialRing

ALIASES = {
    "pot": "pot", "top": "top", "g1": "gvw-g1", "g2": "gvw-g2",
    "gvw-g1": "gvw-g1", "gvw-g2": "gvw-g2", "schreyer": "schreyer",
    "block": "block", "blocked": "block", "block-top-pot": "block",
}


class ModuleOrder:
    def __init__(self, backend: IntegerPolynomialRing, inputs: list,
                 name: str = "gvw-g2", positions: Iterable[int] | None = None,
                 block_sizes: Iterable[int] | None = None) -> None:
        if not isinstance(backend, IntegerPolynomialRing):
            raise ValueError("ModuleOrder requires an IntegerPolynomialRing")
        inputs = [backend.parse(f) for f in inputs]
        self.input_snapshot = tuple(tuple(sorted(f.items())) for f in inputs)
        self.backend = backend
        if not isinstance(name, str):
            raise ValueError("Module order must be a string")
        try:
            self.name = ALIASES[name.lower()]
        except KeyError as exc:
            raise ValueError(f"Unknown module order: {name}") from exc
        m = len(inputs)
        self.positions = tuple(range(m)) if positions is None else tuple(exact_integer(i, "position index") for i in positions)
        if sorted(self.positions) != list(range(m)):
            raise ValueError("Position order must be a permutation of the input indices")
        rank = {index: j for j, index in enumerate(self.positions)}
        self.rank = tuple(rank[i] for i in range(m))

        if self.name == "block":
            if block_sizes is None:
                raise ValueError("Block module order requires signature block sizes")
            self.block_sizes = tuple(exact_integer(x, "signature block size") for x in block_sizes)
            if not self.block_sizes or any(x <= 0 for x in self.block_sizes):
                raise ValueError("Signature block sizes must be positive integers")
            if sum(self.block_sizes) != m:
                raise ValueError(
                    f"Signature block sizes sum to {sum(self.block_sizes)}, but there are {m} inputs")
            block_of_rank: list[int] = []
            for block, size in enumerate(self.block_sizes):
                block_of_rank.extend([block] * size)
            self.block_of_index = tuple(block_of_rank[self.rank[i]] for i in range(m))
        else:
            if block_sizes is not None:
                raise ValueError("Signature block sizes may only be used with module order 'block'")
            self.block_sizes = None
            self.block_of_index = None

        if self.name == "gvw-g1" and backend.order_name == "lex" and len(backend.names) > 1:
            raise ValueError("GVW g1 is degree-first and incompatible with multivariate lex. "
                             "Use grlex/grevlex with g1, or pot/top/gvw-g2 with lex.")
        # Zero input has no lm. Explicit extension: weight monomial 1 and degree 0.
        # Its basis signature is inserted into S, so all its multiples are rejected.
        self.weights = tuple(backend.leading(f)[0] if f else backend.zero_monomial
                             for f in inputs)
        self.degrees = tuple(backend.total_degree(f) for f in inputs)
        self.key_monomial = lru_cache(maxsize=32768)(self._key_monomial)

    def validate_for(self, backend: IntegerPolynomialRing, inputs: list) -> None:
        """Reject a stale/wrong-ring order instead of using incompatible weights."""
        if self.backend.ring != backend.ring:
            raise ValueError("ModuleOrder and solver use different rings/orders")
        snapshot = tuple(tuple(sorted(f.items())) for f in inputs)
        if self.input_snapshot != snapshot:
            raise ValueError("ModuleOrder was constructed for different input polynomials")
        expected = ModuleOrder(backend, inputs, self.name, self.positions, self.block_sizes)
        for field in ("rank", "weights", "degrees", "block_of_index"):
            if getattr(self, field) != getattr(expected, field):
                raise ValueError(f"Inconsistent ModuleOrder {field}")

    def _key_monomial(self, exponent: Monomial, index: int) -> tuple:
        key = self.backend.order_key
        rank = self.rank[index]
        if self.name == "pot":
            return rank, key(exponent)
        if self.name == "top":
            return key(exponent), rank
        if self.name == "block":
            # POT between blocks and TOP within a block.
            return self.block_of_index[index], key(exponent), rank
        if self.name == "gvw-g1":
            return sum(exponent) + self.degrees[index], key(exponent), rank
        # g2 is input-lm weighted and breaks ties with POT. Schreyer is the
        # same weighted comparison with the explicitly REVERSED position tie.
        weighted = key(add_monomials(exponent, self.weights[index]))
        if self.name == "gvw-g2":
            return weighted, rank, key(exponent)
        return weighted, -rank, key(exponent)

    def key(self, sig: Signature) -> tuple:
        return self.key_monomial(sig.monomial, sig.index)

    def compare(self, a: Signature, b: Signature) -> int:
        ka, kb = self.key(a), self.key(b)
        return (ka > kb) - (ka < kb)

    def metadata(self) -> dict:
        data = {
            "module_order": self.name,
            "positions_low_to_high": [i + 1 for i in self.positions],
            "input_lm_weights": [list(x) for x in self.weights],
            "input_degree_weights": list(self.degrees),
            "zero_input_weight_convention": "monomial 1, degree 0",
        }
        if self.block_sizes is not None:
            data["signature_blocks"] = list(self.block_sizes)
            ranges = []
            start = 0
            for size in self.block_sizes:
                ranges.append([i + 1 for i in self.positions[start:start + size]])
                start += size
            data["signature_block_positions"] = ranges
            data["block_order_description"] = "POT between blocks; TOP within each block"
        return data
