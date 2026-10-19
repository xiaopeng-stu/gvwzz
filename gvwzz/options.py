"""Immutable solver configuration and validation, independent of the run loop."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from .core import exact_integer


@dataclass(frozen=True)
class Options:
    """Solver switches; incremental_pot selects the separate staged ideal-basis engine."""

    cover_mode: str = "indexed"
    j_deduplicate: bool = True
    reducer_policy: str = "regular-first"
    g_first: bool = True
    queue_tie: str = "small-coeff"
    debug: bool = False
    track_representations: bool = False
    record_paths: bool = False
    euclidean: bool = True
    tail_reduce: bool = True
    tail_policy: str = "all"
    tail_step_cap: int = 0
    reduction_kernel: str = "heap"
    lazy_candidates: bool = True
    g_precheck: bool = True
    j_generation_filter: bool = True
    # Compatibility field: G-candidates with an exact top reducer are always discarded.
    g_policy: Literal["discard"] = "discard"
    reducer_selection: str = "sparse"
    euclidean_priority: str = "signature-first"
    # No solver abort budgets by default. Explicit positive limits still apply.
    # Verification and user-input parser guards are independent.
    max_seconds: float | None = None
    max_basis: int | None = None
    max_candidates: int | None = None
    max_reduction_steps: int | None = None
    max_polynomial_terms: int | None = None
    max_coefficient_bits: int | None = None
    cover_cache_size: int = 8192
    progress_interval: float = 0.0
    # Extra gate for G tails only; tail_reduce remains the master switch.
    g_tail_reduce: bool = True
    # Separate ideal-basis mode: complete a POT stage, then retain a minimal strong basis.
    incremental_pot: bool = False

    def validate(self) -> None:
        for name in ("j_deduplicate", "g_first", "debug", "euclidean", "tail_reduce",
                     "lazy_candidates", "g_precheck", "j_generation_filter", "track_representations", "record_paths", "g_tail_reduce", "incremental_pot"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in ("max_basis", "max_candidates", "max_reduction_steps",
                     "max_polynomial_terms", "max_coefficient_bits", "cover_cache_size", "tail_step_cap"):
            value = getattr(self, name)
            if value is not None:
                exact_integer(value, name)
            elif name in ("cover_cache_size", "tail_step_cap"):
                raise ValueError(f"{name} must be an integer")
        for name in ("max_seconds", "progress_interval"):
            value = getattr(self, name)
            if value is None and name == "max_seconds":
                continue
            try:
                # Deadlines and progress intervals are stored in nanoseconds.
                representable = type(value) in (float, int) and isfinite(value * 1e9)
            except OverflowError:
                representable = False
            if not representable:
                raise ValueError(f"{name} must be finite and representable in nanoseconds")
        if self.record_paths and self.track_representations:
            raise ValueError("record_paths and track_representations are mutually exclusive; path mode never expands u online")
        if self.cover_mode not in ("indexed", "naive"):
            raise ValueError("cover_mode must be indexed or naive")
        if self.reducer_policy not in ("regular-first", "singular-first", "first"):
            raise ValueError("Unknown reducer policy")
        if self.g_policy != "discard":
            raise ValueError(
                "Only g_policy='discard' is supported. "
                "The G top-reduction mode has been removed."
            )
        if self.euclidean_priority not in ("exact-first", "signature-first"):
            raise ValueError("Unknown Euclidean priority")
        for name in (
            "max_seconds", "max_basis", "max_candidates",
            "max_reduction_steps", "max_polynomial_terms",
            "max_coefficient_bits",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or None")
        if self.reducer_selection not in (
            "oldest", "sparse", "height", "balanced", "newest", "unit"
        ):
            raise ValueError("Unknown reducer selection")
        if self.reduction_kernel not in ("copy", "heap", "packed"):
            raise ValueError("reduction_kernel must be copy, heap or packed")
        if self.queue_tie not in ("fifo", "small-coeff", "small-lm"):
            raise ValueError("Unknown equal-signature tie strategy")
        if self.tail_policy not in ("all", "selective"):
            raise ValueError("tail_policy must be all or selective")
        if self.tail_step_cap < 0:
            raise ValueError("tail_step_cap must be nonnegative")
        if self.progress_interval < 0:
            raise ValueError("progress_interval must be nonnegative")
        if self.cover_cache_size < 0:
            raise ValueError("cover_cache_size must be nonnegative")

