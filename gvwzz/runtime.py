"""Environment information reported by the CLI and serialized results."""
from __future__ import annotations

import platform
import sys
from collections.abc import Mapping

import sympy

from ._version import __version__


_PARENT_FALLBACK_REASONS = {
    "dimension": "a legacy native batch rejected the variable count",
    "exponent": "parent exponents do not fit the checked packed representation",
    "certificate": "generation paths or representation certificates are enabled",
    "budget": "a generated-candidate budget is enabled",
    "engine": "this engine uses its own parent-generation callbacks",
    "unavailable": "the native parent-batch entry point is unavailable",
}


def native_usage(counters: Mapping, *, incremental_pot: bool = False) -> dict:
    """Summarize observed dispatch counters, separately from import availability.

    Non-bulk native workspaces still use native reduction. The existing
    reduction counters do not count Python workspaces or identify why bulk
    dispatch was rejected, so those quantities are intentionally not guessed.
    """
    native_pairs = int(counters.get("native_parent_pairs", 0))
    parent = {
        "native_batch_rows": int(counters.get("native_parent_rows", 0)),
        "native_batch_pairs": native_pairs,
        "native_wide_rows": int(counters.get("native_parent_wide_rows", 0)),
        "native_wide_pairs": int(counters.get("native_parent_wide_pairs", 0)),
        "native_dynamic_rows": int(counters.get("native_parent_dynamic_rows", 0)),
        "native_dynamic_pairs": int(counters.get("native_parent_dynamic_pairs", 0)),
        "python_rows": (int(counters.get("parent_python_rows", 0))
                        if ("parent_dispatch_rows" in counters or
                            "parent_python_rows" in counters) else None),
        "python_pairs": None,
        "within_batch_python_pairs": int(counters.get("native_parent_fallback_pairs", 0)),
        "fallback_reasons_rows": {},
        "row_scope": "ordinary parent dispatch",
    }
    # Incremental mode also counts pairs skipped by mathematical invariants.
    # Subtracting its native count would mislabel those skipped pairs as Python.
    if not incremental_pot and "parent_pairs" in counters:
        parent["python_pairs"] = int(counters["parent_pairs"]) - native_pairs
    for reason, label in _PARENT_FALLBACK_REASONS.items():
        value = counters.get(f"parent_fallback_{reason}_rows", 0)
        if value:
            parent["fallback_reasons_rows"][reason] = {
                "rows": int(value), "reason": label,
            }
    workspaces = int(counters.get("native_reduction_workspaces", 0))
    bulk = int(counters.get("native_bulk_reduction_workspaces", 0))
    reduction = {
        "native_workspaces": workspaces,
        "native_bulk_workspaces": bulk,
        "native_wide_bulk_workspaces": int(counters.get("native_wide_bulk_reduction_workspaces", 0)),
        "native_nonbulk_workspaces": workspaces - bulk,
        "python_workspaces": None,
        "bulk_fallback_reasons": None,
    }
    recipes = {
        name: int(counters[key]) for name, key in (
            ("compact", "candidate_recipes_compact"),
            ("native", "candidate_recipes_native"),
        ) if key in counters
    }
    drain = {
        "calls": int(counters.get("native_queue_drain_calls", 0)),
        "popped": int(counters.get("native_queue_drain_pops", 0)),
        "covered": int(counters.get("native_queue_drain_covered", 0)),
    }
    cover = {
        "queries": int(counters.get("native_cover_queries", 0)),
        "proofs": int(counters.get("native_cover_proofs", 0)),
        "fallbacks": int(counters.get("native_cover_fallbacks", 0)),
    }
    queue = {
        "pushes": int(counters.get("native_queue_pushes", 0)),
        "pops": int(counters.get("native_queue_pops", 0)),
        "dedup_checks": int(counters.get("native_queue_dedup_checks", 0)),
    }
    return {"parent_generation": parent, "reduction": reduction,
            "candidate_recipes": recipes, "queue_drain": drain,
            "packed_cover": cover, "candidate_queue": queue}


def environment() -> dict:
    from sympy.external.gmpy import GROUND_TYPES
    # Import the selected entry points lazily: the engines also use this report.
    from . import incremental, solver, structures

    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as source:
            for line in source:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {
        "implementation_version": __version__,
        "python": sys.version.split()[0],
        "sympy": sympy.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": cpu,
        "integer_ground_type": GROUND_TYPES,
        "polynomial_backend": "sympy.polys.rings.PolyElement over ZZ",
        "threads": "single-threaded algorithm",
        "native_reduction_coefficient_backend": solver._native_coefficient_backend,
        "native_accelerators": {
            "reduction": solver._native_reduce_working is not None,
            "pairs": incremental._native_emit_frozen_pairs is not None,
            "indices": getattr(structures, '_native_has_divisor_packed', None) is not None,
        },
        # Availability is not evidence that a case used an eligible kernel.
        "native_entry_points": {
            "parent_batch": solver._native_emit_parent_pairs is not None,
            "frozen_parent_batch": incremental._native_emit_frozen_pairs is not None,
            "candidate_heap": getattr(structures, "_NativeCandidateHeap", None) is not None,
            "packed_cover": getattr(structures, "_NativeCoverQuery", None) is not None,
            "queue_drain": getattr(solver, "_native_select_candidate", None) is not None,
        },
    }

