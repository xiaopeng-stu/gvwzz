"""Insertion provenance and the paper's coefficient-free ratio-layer forests.

History is descriptive execution metadata, not a representation certificate.
Only successful nonzero insertions become vertices.  Discarded candidates and
syzygies do not.  All internal indices are zero-based; JSON/display indices
are one-based.  No forest is reconstructed by guessing parents from final
signatures or by re-running the solver.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .core import exact_integer, subtract_monomials
from .validation import validate_signature

if TYPE_CHECKING:
    from .solver import Result


class ForestUnavailable(ValueError):
    """The result has no complete generating-parent history."""


@dataclass(frozen=True, slots=True)
class InsertionOrigin:
    kind: str
    parents: tuple[int, int] | None = None
    input_index: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("input", "J", "G"):
            raise ValueError("Insertion kind must be input, J or G")
        if self.kind == "input":
            if self.parents is not None or self.input_index is None:
                raise ValueError("An input origin needs an input index and no parents")
            index = exact_integer(self.input_index, "input index")
            if index < 0:
                raise ValueError("Input index must be nonnegative")
            object.__setattr__(self, "input_index", index)
        else:
            if self.input_index is not None or not isinstance(self.parents, (tuple, list)) or len(self.parents) != 2:
                raise ValueError("A J/G origin needs exactly two generating parents")
            parents = tuple(exact_integer(i, "generating-parent index") for i in self.parents)
            if min(parents) < 0 or parents[0] == parents[1]:
                raise ValueError("Generating parents must be distinct nonnegative indices")
            object.__setattr__(self, "parents", tuple(sorted(parents)))

    def to_dict(self, insertion_index: int) -> dict:
        item = {"insertion_index": insertion_index + 1, "kind": self.kind}
        if self.kind == "input":
            item["input_index"] = self.input_index + 1
        else:
            item["parents"] = [p + 1 for p in self.parents]
        return item


def validate_history(history, nvertices: int, ninputs: int) -> None:
    if not isinstance(history, (list, tuple)) or len(history) != nvertices:
        raise ValueError("Insertion history must have one entry per P record")
    for i, origin in enumerate(history):
        # Unknown origins are allowed for externally constructed labels, but
        # the forest builder will explicitly decline to invent their edges.
        if origin is None:
            continue
        if not isinstance(origin, InsertionOrigin):
            raise ValueError("Invalid insertion-history record")
        if origin.kind == "input":
            if not 0 <= origin.input_index < ninputs:
                raise ValueError("Input origin is outside the input module")
        elif any(p >= i for p in origin.parents):
            raise ValueError("Generating parents must precede the child's insertion")


def serialize_history(history, nvertices: int, ninputs: int) -> dict:
    validate_history(history, nvertices, ninputs)
    return {"format": "gvwzz-insertion-history-v1", "entries": [
        origin.to_dict(i) if origin is not None else None
        for i, origin in enumerate(history)
    ]}


def deserialize_history(data, nvertices: int, ninputs: int) -> list[InsertionOrigin | None]:
    if not isinstance(data, dict) or data.get("format") != "gvwzz-insertion-history-v1":
        raise ValueError("Unsupported insertion-history format")
    entries = data.get("entries")
    if not isinstance(entries, list) or len(entries) != nvertices:
        raise ValueError("Insertion history has the wrong number of entries")
    history = []
    for i, item in enumerate(entries):
        if item is None:
            history.append(None)
            continue
        if not isinstance(item, dict):
            raise ValueError("Invalid insertion-history entry")
        if exact_integer(item.get("insertion_index"), "insertion index") != i + 1:
            raise ValueError("Insertion indices must be consecutive and start at 1")
        kind = item.get("kind")
        if kind == "input":
            if item.get("parents") is not None:
                raise ValueError("An input origin cannot have generating parents")
            origin = InsertionOrigin(kind, input_index=exact_integer(item.get("input_index"), "input index") - 1)
        elif kind in ("J", "G"):
            parents = item.get("parents")
            if not isinstance(parents, list) or len(parents) != 2 or item.get("input_index") is not None:
                raise ValueError("A J/G history entry needs two parents and no input index")
            origin = InsertionOrigin(kind, tuple(exact_integer(p, "parent index") - 1 for p in parents))
        else:
            raise ValueError("Invalid insertion kind in history")
        history.append(origin)
    validate_history(history, nvertices, ninputs)
    return history


# A ratio is x^delta * e_index, with delta possibly negative, ignoring coefficients.
Ratio = tuple[int, tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class ForestVertex:
    insertion_index: int  # one-based
    ratio: Ratio
    origin: InsertionOrigin
    parent: int | None  # one-based selected forest parent
    children: tuple[int, ...]  # one-based, chronological


@dataclass(frozen=True, slots=True)
class RatioLayer:
    ratio: Ratio
    vertices: tuple[int, ...]
    roots: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RatioForest:
    vertices: tuple[ForestVertex, ...]
    layers: tuple[RatioLayer, ...]


def build_forest(result: Result) -> RatioForest:
    """Build same-ratio edges from the actual insertion history.

    A vertex with no same-ratio generating parent is a root.  With one such
    parent, choose it.  With two, choose the one inserted later.  Layers are
    ordered by the actual module order, not by lexical order of printed text.
    This also works for a finite partial execution; it proves no termination
    or correctness claim about that partial result.
    """
    if result.options.incremental_pot:
        raise ForestUnavailable("incremental POT retires stage history; no global ratio-layer forest is returned")
    if not result.basis:
        return RatioForest((), ())
    history = result.insertion_history
    if history is None or any(origin is None for origin in history):
        raise ForestUnavailable(
            "generating-parent history is missing; the actual forest cannot "
            "be recovered from final (signature, polynomial) records alone. "
            "Recompute with version 0.10.0.post5 or later."
        )
    validate_history(history, len(result.basis), len(result.inputs))
    b, order = result.backend, result.module_order
    order.validate_for(b, result.inputs)
    ratios, keys = [], []
    groups: dict[Ratio, list[int]] = {}
    for i, label in enumerate(result.basis):
        validate_signature(label.signature, len(b.names), len(result.inputs))
        lm, _ = b.leading(label.polynomial)
        delta = subtract_monomials(label.signature.monomial, lm)
        ratio = (label.signature.index, delta)
        ratios.append(ratio)
        keys.append(order.key_monomial(delta, label.signature.index))
        groups.setdefault(ratio, []).append(i)

    parents: list[int | None] = [None] * len(ratios)
    children: list[list[int]] = [[] for _ in ratios]
    for i, origin in enumerate(history):
        if origin.kind == "input":
            s = result.basis[i].signature
            if (s.coefficient != 1 or any(s.monomial) or s.index != origin.input_index
                    or result.basis[i].polynomial != result.inputs[origin.input_index]):
                raise ValueError(f"Input origin of g{i + 1} does not match its record")
            continue
        parent_keys = [keys[p] for p in origin.parents]
        if origin.kind == "J":
            if keys[i] <= max(parent_keys):
                raise ValueError(f"J-sourced g{i + 1} must have strictly larger ratio than both parents")
        elif keys[i] != max(parent_keys):
            raise ValueError(f"G-sourced g{i + 1} must have the maximum parent ratio")
        same = [p for p in origin.parents if ratios[p] == ratios[i]]
        if same:
            selected = max(same)
            parents[i] = selected + 1
            children[selected].append(i + 1)

    vertices = tuple(ForestVertex(i + 1, ratios[i], history[i], parents[i], tuple(children[i]))
                     for i in range(len(ratios)))
    ordered = sorted(groups, key=lambda r: keys[groups[r][0]])
    layers = tuple(RatioLayer(r, tuple(i + 1 for i in groups[r]),
                              tuple(i + 1 for i in groups[r] if parents[i] is None))
                   for r in ordered)
    return RatioForest(vertices, layers)
