"""Exact local generation paths without online representation vectors.

Edges describe R-linear operations. A term edge stores (integer, exponents);
a principal-syzygy edge additionally references an immutable P polynomial.
No multiplication of representation vectors occurs here. Node IDs increase
strictly along construction, independently of signature order or reductions.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Any

from .core import (Signature, encode_integer, decode_integer, exact_integer,
                   validate_monomial, lcm_monomials, subtract_monomials)

FORMAT = "gvwzz-generation-paths-v1"
KINDS = {"input", "J", "G", "principal", "sigG", "unit"}


@dataclass(frozen=True, slots=True)
class Edge:
    parent: int
    coefficient: int
    monomial: tuple[int, ...]
    polynomial_index: int | None = None


@dataclass(frozen=True, slots=True)
class Node:
    kind: str
    signature: Signature
    edges: tuple[Edge, ...] = ()
    input_index: int | None = None


@dataclass(frozen=True)
class GenerationPaths:
    nodes: tuple[Node, ...]
    basis_roots: tuple[int, ...]
    syzygy_roots: tuple[int, ...]
    reduction_steps_recorded: int = 0

    def to_dict(self) -> dict:
        return {
            "format": FORMAT,
            "index_base": 0,
            "nodes": [dict(kind=n.kind, signature=n.signature.to_dict(),
                           input_index=n.input_index,
                           edges=[dict(parent=e.parent, coefficient=encode_integer(e.coefficient),
                                       exponents=list(e.monomial), polynomial_index=e.polynomial_index)
                                  for e in n.edges]) for n in self.nodes],
            "basis_roots": list(self.basis_roots),
            "syzygy_roots": list(self.syzygy_roots),
            "reduction_steps_recorded": self.reduction_steps_recorded,
        }

    def validate(self, result) -> None:
        """Structural validation only; algebraic validation is separate."""
        nv, rank = len(result.backend.names), len(result.inputs)
        if len(self.basis_roots) != len(result.basis) or len(self.syzygy_roots) != len(result.syzygies):
            raise ValueError("Generation-path root counts do not match P/S")
        for j, node in enumerate(self.nodes):
            if node.kind not in KINDS:
                raise ValueError("Unsupported generation-path node kind")
            sig = node.signature
            if not 0 <= sig.index < rank:
                raise ValueError("Path signature index outside the input module")
            validate_monomial(sig.monomial, nv)
            if node.kind == "input":
                i = exact_integer(node.input_index, "path input index")
                if not 0 <= i < rank or node.edges:
                    raise ValueError("Invalid input node")
                if sig != Signature(1, result.backend.zero_monomial, i):
                    raise ValueError("Input path has wrong signature")
            elif node.input_index is not None or not node.edges:
                raise ValueError("Linear path node requires nonempty edges and no input index")
            for e in node.edges:
                parent = exact_integer(e.parent, "parent node ID")
                if not 0 <= parent < j:
                    raise ValueError("A path parent must have a strictly smaller node ID (no cycles)")
                if exact_integer(e.coefficient, "edge coefficient") == 0:
                    raise ValueError("Zero edge coefficient is not canonical")
                validate_monomial(e.monomial, nv)
                if e.polynomial_index is not None:
                    k = exact_integer(e.polynomial_index, "P polynomial index")
                    if not 0 <= k < len(result.basis):
                        raise ValueError("Path polynomial reference outside P")
                    if not 0 <= self.basis_roots[k] < j:
                        raise ValueError("Path polynomial must belong to an earlier P node")
        for roots, signatures in ((self.basis_roots, [p.signature for p in result.basis]),
                                   (self.syzygy_roots, result.syzygies)):
            for root, sig in zip(roots, signatures):
                root = exact_integer(root, "path root ID")
                if not 0 <= root < len(self.nodes) or self.nodes[root].signature != sig:
                    raise ValueError("Generation-path root has wrong ID or complete signature")
        if exact_integer(self.reduction_steps_recorded, "recorded reduction steps") < 0:
            raise ValueError("Recorded reduction count must be nonnegative")

    @classmethod
    def from_dict(cls, data: Any, result) -> "GenerationPaths":
        if not isinstance(data, dict) or data.get("format") != FORMAT or exact_integer(data.get("index_base"), "path index base") != 0:
            raise ValueError("Unsupported generation-path format")
        for name in ("nodes", "basis_roots", "syzygy_roots"):
            if not isinstance(data.get(name), list):
                raise ValueError(f"Generation-path {name} must be a list")
        nodes = []
        for item in data["nodes"]:
            if not isinstance(item, dict) or not isinstance(item.get("edges"), list):
                raise ValueError("Invalid generation-path node")
            s = item["signature"]
            signature = Signature(decode_integer(s["coefficient"]),
                                  validate_monomial(s["exponents"], len(result.backend.names)),
                                  exact_integer(s["index"], "signature index") - 1)
            edges = []
            for e in item["edges"]:
                if not isinstance(e, dict):
                    raise ValueError("Invalid path edge")
                pi = e.get("polynomial_index")
                edges.append(Edge(exact_integer(e["parent"], "path parent"),
                                  decode_integer(e["coefficient"]),
                                  validate_monomial(e["exponents"], len(result.backend.names)),
                                  None if pi is None else exact_integer(pi, "polynomial index")))
            ii = item.get("input_index")
            nodes.append(Node(item["kind"], signature, tuple(edges),
                              None if ii is None else exact_integer(ii, "input index")))
        graph = cls(tuple(nodes),
                    tuple(exact_integer(x, "P root") for x in data["basis_roots"]),
                    tuple(exact_integer(x, "S root") for x in data["syzygy_roots"]),
                    exact_integer(data.get("reduction_steps_recorded", 0), "recorded reductions"))
        graph.validate(result)
        return graph


class PathTracker:
    """Compact sidecar; has no role in algebraic decisions or reducer lookup."""
    def __init__(self, backend, ncomponents: int):
        self.backend, self.ncomponents = backend, ncomponents
        self.nodes: list[Node] = []
        self.basis: list[int] = []
        self.syzygies: dict[Signature, int] = {}
        self.current_edges: list[Edge] | None = None
        self.current_root: int | None = None
        self.current_kind = "input"
        self.reduction_steps_recorded = 0

    def _add(self, kind: str, signature: Signature, edges=(), input_index=None) -> int:
        # Coalesce repeated term contributions; no polynomial or u multiplication.
        weights: dict[tuple, int] = {}
        for e in edges:
            key = (e.parent, e.monomial, e.polynomial_index)
            weights[key] = weights.get(key, 0) + int(e.coefficient)
        combined = tuple(Edge(parent, c, monomial, pi)
                         for (parent, monomial, pi), c in weights.items() if c)
        node_id = len(self.nodes)
        if kind != "input" and not combined:
            raise RuntimeError("A nonzero signed path collapsed to the zero expression")
        if any(not 0 <= e.parent < node_id for e in combined):
            raise RuntimeError("Path graph construction order violated")
        self.nodes.append(Node(kind, signature, combined, input_index))
        return node_id

    def input(self, index: int, signature: Signature) -> None:
        self.current_root = self._add("input", signature, input_index=index)
        self.current_edges = None
        self.current_kind = "input"

    def start(self, recipe: tuple, kind: str) -> None:
        self.current_edges = [Edge(self.basis[i], int(c), tuple(mu))
                              for i, c, mu in recipe if c]
        self.current_root = None
        self.current_kind = kind

    def subtract(self, i: int, coefficient: int, monomial: tuple) -> None:
        if self.current_edges is None:
            raise RuntimeError("No active candidate path")
        self.current_edges.append(Edge(self.basis[i], -int(coefficient), tuple(monomial)))
        self.current_root = None
        self.reduction_steps_recorded += 1

    def freeze_current(self, signature: Signature) -> int:
        if self.current_root is None:
            if self.current_edges is None:
                raise RuntimeError("Missing current generation path")
            self.current_root = self._add(self.current_kind, signature, self.current_edges)
        if self.nodes[self.current_root].signature != signature:
            raise RuntimeError("Current path complete signature mismatch")
        return self.current_root

    def append_basis(self, signature: Signature) -> None:
        self.basis.append(self.freeze_current(signature))

    def principal(self, i: int, j: int, signature: Signature) -> int:
        z = self.backend.zero_monomial
        return self._add("principal", signature,
                         (Edge(self.basis[i], 1, z, j), Edge(self.basis[j], -1, z, i)))

    def update_syzygies(self, signature: Signature, node: int, old, retained) -> None:
        seed = signature.positive()
        if signature.coefficient < 0:
            node = self._add("unit", seed, (Edge(node, -1, self.backend.zero_monomial),))
        # Keep node storage immutable, even if old active signatures are deleted.
        new = dict(self.syzygies)
        for h in retained:
            if h in self.syzygies:
                continue
            if h == seed:
                new[h] = node
                continue
            for s in old:
                L = lcm_monomials(seed.monomial, s.monomial)
                if h != Signature(gcd(seed.coefficient, s.coefficient), L, seed.index):
                    continue
                a, b, _ = self.backend.bezout(seed.coefficient, s.coefficient)
                new[h] = self._add("sigG", h, (
                    Edge(node, a, subtract_monomials(L, seed.monomial)),
                    Edge(self.syzygies[s], b, subtract_monomials(L, s.monomial))))
                break
            else:
                raise RuntimeError("Missing local construction for a retained sigG signature")
        keep = set(retained)
        self.syzygies = {s: r for s, r in new.items() if s.index != seed.index or s in keep}

    def snapshot(self, syzygies) -> GenerationPaths:
        return GenerationPaths(tuple(self.nodes), tuple(self.basis),
                               tuple(self.syzygies[s] for s in syzygies),
                               self.reduction_steps_recorded)
