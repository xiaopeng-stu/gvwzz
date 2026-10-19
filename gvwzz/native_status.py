"""Report the accelerators selected by this interpreter, without running a case."""
from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path


_EXTENSIONS = {
    "reduction": "gvwzz._native_reduction",
    "pairs": "gvwzz._native_pairs",
    "indices": "gvwzz._native_indices",
}

# Older extension names remain stable in JSON. These additional entry points
# prevent a mixed build from passing --require-native with post27 binaries.
_REQUIRED_ENTRY_POINTS = {
    "parent_batch": ("gvwzz._native_pairs", "emit_parent_pairs"),
    "candidate_heap": ("gvwzz._native_pairs", "NativeCandidateHeap"),
    "packed_cover": ("gvwzz._native_cover", "NativeCoverQuery"),
    "queue_drain": ("gvwzz._native_queue_drain", "select_candidate"),
}


def native_status(*, diagnose=False) -> dict:
    # Read the functions the solver actually selected, not just whether a .so or
    # .pyd file exists. This also respects import-time environment overrides.
    from .solver import environment

    runtime = environment()
    selected = runtime["native_accelerators"]
    entries = runtime.get("native_entry_points", {})
    data = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_directory": str(Path(__file__).resolve().parent),
        "native_accelerators": selected,
        "native_entry_points": entries,
        "native_reduction_coefficient_backend": runtime.get("native_reduction_coefficient_backend", "python"),
        "all_native_available": (all(selected.get(name) for name in _EXTENSIONS)
                                 and all(entries.get(name) for name in _REQUIRED_ENTRY_POINTS)),
        "GVWZZ_NO_NATIVE": os.environ.get("GVWZZ_NO_NATIVE"),
    }
    if diagnose:
        problems = {}
        checks = [(name, module, None, selected.get(name))
                  for name, module in _EXTENSIONS.items()]
        checks.extend((name, module, attr, entries.get(name))
                      for name, (module, attr) in _REQUIRED_ENTRY_POINTS.items())
        for name, module_name, attr, available in checks:
            if available:
                continue
            if os.environ.get("GVWZZ_NO_NATIVE"):
                problems[name] = ("Disabled by GVWZZ_NO_NATIVE=" +
                                  os.environ["GVWZZ_NO_NATIVE"] +
                                  " before interpreter startup.")
                continue
            try:
                module = importlib.import_module(module_name)
                if attr is not None and not hasattr(module, attr):
                    raise ImportError(f"{module_name} lacks {attr}; rebuild this release")
            except (ImportError, OSError) as exc:
                problems[name] = f"{type(exc).__name__}: {exc}"
            else:
                problems[name] = (
                    "Extension imports, but its runtime entry point was not selected. "
                    "Check GVWZZ_NO_NATIVE and rebuild for this release/interpreter."
                )
        data["problems"] = problems
    return data


def status_line(data: dict) -> str:
    return "Native extensions available: " + " | ".join(
        f"{name}={'yes' if data['native_accelerators'].get(name) else 'no'}"
        for name in _EXTENSIONS
    )


def entry_status_lines(data: dict) -> list[str]:
    entries = data.get("native_entry_points", {})
    if not entries:
        return []
    lines = []
    if "parent_batch" in entries:
        lines.append("Native parent-batch entry point: " +
                     ("available" if entries["parent_batch"] else "unavailable"))
    names = ("candidate_heap", "packed_cover", "queue_drain")
    if any(name in entries for name in names):
        lines.append("Native queue/cover entry points: " + " | ".join(
            f"{name}={'yes' if entries.get(name) else 'no'}" for name in names))
    return lines


def usage_lines(usage: dict) -> list[str]:
    """Render measured kernel use without treating imports as activity."""
    parent, reduction = usage["parent_generation"], usage["reduction"]
    python_rows = parent["python_rows"]
    python_pairs = parent["python_pairs"]
    lines = [
        "Native parent batches used: "
        f"rows={parent['native_batch_rows']} pairs={parent['native_batch_pairs']} "
        f"| wide rows/pairs={parent['native_wide_rows']}/{parent['native_wide_pairs']} "
        f"| dynamic rows/pairs={parent.get('native_dynamic_rows', 0)}/{parent.get('native_dynamic_pairs', 0)}",
        "Python parent dispatch: "
        f"rows={python_rows if python_rows is not None else 'not recorded'} "
        f"pairs={python_pairs if python_pairs is not None else 'not recorded'} "
        f"| within native batches={parent['within_batch_python_pairs']} pairs",
    ]
    for item in parent["fallback_reasons_rows"].values():
        lines.append(f"  Parent fallback: {item['rows']} rows; {item['reason']}.")
    lines.append(
        "Native reduction workspaces used: "
        f"total={reduction['native_workspaces']} bulk={reduction['native_bulk_workspaces']} "
        f"wide={reduction['native_wide_bulk_workspaces']} "
        f"non-bulk={reduction['native_nonbulk_workspaces']}"
    )
    if reduction["native_nonbulk_workspaces"]:
        lines.append("  Non-bulk workspaces still use native reduction; bulk fallback reasons were not recorded.")
    recipes = usage.get("candidate_recipes", {})
    if recipes:
        lines.append("Compact candidate recipes created: " + " | ".join(
            f"{name}={value}" for name, value in recipes.items()))
    queue = usage.get("candidate_queue", {})
    if queue:
        lines.append(
            "Native candidate queue used: "
            f"pushes={queue['pushes']} pops={queue['pops']} dedup checks={queue['dedup_checks']}"
        )
    drain = usage.get("queue_drain", {})
    if drain:
        lines.append(
            "Native queue drain used: "
            f"calls={drain['calls']} popped={drain['popped']} covered={drain['covered']}"
        )
    cover = usage.get("packed_cover", {})
    if cover:
        lines.append(
            "Native packed cover used: "
            f"queries={cover['queries']} proofs={cover['proofs']} fallbacks={cover['fallbacks']}"
        )
    return lines


def print_native_start(*, require_native=False) -> None:
    data = native_status()
    print(status_line(data), flush=True)
    for line in entry_status_lines(data):
        print(line, flush=True)
    print("Kernel eligibility depends on the case and options; actual usage is reported after computation.",
          flush=True)
    print("Native reduction arithmetic:", data.get("native_reduction_coefficient_backend", "python"),
          "(arbitrary-precision integers)", flush=True)
    if not data["all_native_available"]:
        if data["GVWZZ_NO_NATIVE"]:
            hint = "GVWZZ_NO_NATIVE disables native acceleration; unset it before starting Python."
        else:
            hint = (
                "Some native accelerators or entry points are unavailable. From the extracted release directory, "
                "use this Python to run: python build_native.py. "
                "For details: python -m gvwzz doctor."
            )
        print(hint, flush=True)
        if require_native:
            raise ValueError("--require-native: computation was not started because an accelerator is unavailable")


def doctor_command(*, json_output=False, require_native=False) -> int:
    import json

    data = native_status(diagnose=True)
    if json_output:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print("Python:", data["python_executable"])
        print("Python version:", data["python_version"])
        print("Platform:", data["platform"])
        print("Package:", data["package_directory"])
        print(status_line(data))
        for line in entry_status_lines(data):
            print(line)
        print("Native reduction arithmetic:", data.get("native_reduction_coefficient_backend", "python"),
              "(arbitrary-precision integers)")
        if data["GVWZZ_NO_NATIVE"] is not None:
            print("GVWZZ_NO_NATIVE:", repr(data["GVWZZ_NO_NATIVE"]))
        for name, reason in data["problems"].items():
            print(f"  {name}: {reason}")
        if not data["all_native_available"]:
            print("Build in the extracted release directory with the Python shown above:")
            print("  python -m pip install -r requirements-native.txt")
            print("  python build_native.py")
            print("Windows builds require Microsoft C++ Build Tools with the C++ workload.")
            print("Unset GVWZZ_NO_NATIVE before verifying or running with acceleration.")
        else:
            print("All required native entry points are selected. Individual kernels still check case eligibility.")
    return 1 if require_native and not data["all_native_available"] else 0
