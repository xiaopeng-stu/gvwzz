"""Command-line interface for the single general solver path."""
from __future__ import annotations

import argparse
import cProfile
import csv
import json
import os
import sys
import tempfile
from dataclasses import asdict, fields, replace
from itertools import product
from math import isfinite
from pathlib import Path
from statistics import median
from time import perf_counter

from .cases import CASES, SUITES, get_case, input_metadata, random_case, reverse_case_generators
from .core import Label, Signature, decode_integer, exact_integer, validate_monomial
from .orders import ModuleOrder
from .polynomial import IntegerPolynomialRing
from .solver import Options, Result, compute, environment
from .structures import Stats, TIMERS
from .verification import verify_result
from .forest import deserialize_history
from .printing import print_result, print_polynomial_output, validate_display_options
from .postprocessing import postprocess_basis
from .output import write_json as _write_json, append_json_fields

from ._version import __version__ as VERSION


def write_json(path, data, *, progress=None):
    """Atomically stream exact JSON records without a whole-document string."""
    return _write_json(path, data, progress=progress)


def write_csv(path, rows) -> None:
    """Commit the CSV companion only after every cell has been encoded."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Column discovery and record output each consume the iterable once.
    if not isinstance(rows, (list, tuple)):
        rows = list(rows)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8-sig",
                                         prefix="." + path.name + ".", suffix=".tmp",
                                         dir=path.parent, delete=False) as sink:
            temporary = Path(sink.name)
            writer = csv.DictWriter(sink, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def positive_or_none(value):
    return None if value == 0 else value


def parse_blocks(value) -> tuple[int, ...] | None:
    """Parse ``1,2,1,2`` or ``(1,2,1,2)`` for the pure block order."""
    if value is None or value == "":
        return None
    if isinstance(value, (tuple, list)):
        blocks = tuple(exact_integer(x, "signature block size") for x in value)
    else:
        text = str(value).strip().replace("\N{FULLWIDTH COMMA}", ",")
        if len(text) >= 2 and text[0] in "([{" and text[-1] in ")]}":
            text = text[1:-1].strip()
        if not text:
            raise ValueError("--signature-blocks must contain positive integers")
        try:
            blocks = tuple(int(part.strip()) for part in text.split(","))
        except ValueError as exc:
            raise ValueError("Use --signature-blocks 1,2,1,2") from exc
    if not blocks or any(x <= 0 for x in blocks):
        raise ValueError("--signature-blocks must contain positive integers")
    return blocks


def options_from_args(args) -> Options:
    # None means no preset was explicitly selected: all unspecified budgets
    # are off, but an explicit --timeout / --max-* still takes effect.
    # True (--no-limits) intentionally clears all six, as in older versions.
    # False (--limits) restores the previous CLI defaults for unspecified fields.
    no_limits = getattr(args, "no_limits", None)

    def budget(value, limited_default):
        if no_limits is True:
            return None
        if value is None:
            value = limited_default if no_limits is False else 0
        return positive_or_none(value)

    return Options(
        cover_mode=args.cover_mode,
        j_deduplicate=not args.no_j_dedup,
        reducer_policy=args.reducer_policy,
        g_first=not args.j_first,
        queue_tie=args.queue_tie,
        debug=args.debug,
        track_representations=args.certificates or getattr(args, "show_u", None) is True,
        record_paths=getattr(args, "record_paths", False) or getattr(args, "reconstruct_syzygies", False),
        euclidean=args.euclidean,
        tail_reduce=args.tail_reduce,
        g_tail_reduce=args.g_tail_reduce,
        tail_policy=args.tail_policy,
        tail_step_cap=args.tail_step_cap,
        reduction_kernel=args.reduction_kernel,
        euclidean_priority=args.euclidean_priority,
        reducer_selection=args.reducer_selection,
        lazy_candidates=args.lazy_candidates,
        g_precheck=not args.no_g_precheck,
        g_policy=args.g_policy,
        j_generation_filter=not args.no_j_generation_filter,
        max_coefficient_bits=budget(args.max_coefficient_bits, 0),
        max_seconds=budget(args.timeout, 30.0),
        max_basis=budget(args.max_basis, 5000),
        max_candidates=budget(args.max_candidates, 250000),
        max_reduction_steps=budget(args.max_steps, 500000),
        max_polynomial_terms=budget(args.max_terms, 100000),
        progress_interval=args.progress,
        incremental_pot=getattr(args, "incremental_pot", False),
    )


def add_solver_options(parser) -> None:
    parser.add_argument("--require-native", action="store_true",
                        help="Stop unless all three native extensions are available; "
                             "individual kernel eligibility still depends on the case and options")
    parser.add_argument("--incremental-pot", action=argparse.BooleanOptionalAction, default=False,
                        help="Separate POT incremental ideal-basis mode; compact each completed stage to a minimal strong basis")
    parser.add_argument("--record-paths", action="store_true",
                        help="Record local P/S generation paths, without expanding u online")
    budget_mode = parser.add_mutually_exclusive_group()
    budget_mode.add_argument(
        "--no-limits", "--no-limit", dest="no_limits", action="store_true",
        help="Disable all six solver budgets (the default); when explicit, also "
             "override any --timeout/--max-* values. Verification limits are separate",
    )
    budget_mode.add_argument(
        "--limits", dest="no_limits", action="store_false",
        help="Restore previous finite solver-budget defaults, except explicitly "
             "set --timeout/--max-* values; coefficient bits remain off unless set",
    )
    parser.set_defaults(no_limits=None)
    parser.add_argument("--progress", type=float, default=0.0, metavar="SECONDS")
    parser.add_argument("--cover-mode", choices=["indexed", "naive"], default="indexed",
                        help="J-candidate cover checks only; G-candidates never use covers")
    parser.add_argument("--no-j-dedup", action="store_true")
    parser.add_argument("--reducer-policy", choices=["regular-first", "singular-first", "first"],
                        default="regular-first")
    parser.add_argument(
        "--g-policy", choices=["discard"], default="discard",
        help="Compatibility option: exact top-reducible G-candidates are always discarded",
    )
    euclidean = parser.add_mutually_exclusive_group()
    euclidean.add_argument("--euclidean", dest="euclidean", action="store_true",
                          help="Enable signature-safe Euclidean coefficient reductions (default: on)")
    euclidean.add_argument("--no-euclidean", dest="euclidean", action="store_false",
                          help="Disable Euclidean coefficient reductions; exact reductions remain")
    parser.set_defaults(euclidean=True)
    parser.add_argument("--euclidean-priority", choices=["signature-first", "exact-first"],
                        default="signature-first")
    parser.add_argument("--reduction-kernel", choices=["copy", "heap", "packed"], default="heap")
    tail_reduction = parser.add_mutually_exclusive_group()
    tail_reduction.add_argument("--tail-reduce", dest="tail_reduce", action="store_true",
                               help="Enable signature-safe tail reductions (default: on)")
    tail_reduction.add_argument("--no-tail-reduce", dest="tail_reduce", action="store_false",
                               help="Disable tail reduction of new records")
    parser.set_defaults(tail_reduce=True)
    g_tail = parser.add_mutually_exclusive_group()
    g_tail.add_argument("--g-tail-reduce", dest="g_tail_reduce", action="store_true",
                       help="Allow G-candidate tails when --tail-reduce is enabled (default: on)")
    g_tail.add_argument("--no-g-tail-reduce", dest="g_tail_reduce", action="store_false",
                       help="Skip only G-candidate tails; retain J reductions and G exact top tests")
    parser.set_defaults(g_tail_reduce=True)
    parser.add_argument("--tail-policy", choices=["all", "selective"], default="all")
    parser.add_argument("--tail-step-cap", type=int, default=0, metavar="N")
    parser.add_argument("--reducer-selection",
                        choices=["oldest", "sparse", "height", "balanced", "newest", "unit"],
                        default="sparse", help="Reducer ranking (default: sparse)")
    parser.add_argument("--queue-tie", choices=["fifo", "small-coeff", "small-lm"],
                        default="small-coeff", help="Queue tie-break (default: small-coeff)")
    parser.add_argument("--j-first", action="store_true")
    # Name the option after the enabled feature, matching Options.lazy_candidates.
    materialization = parser.add_mutually_exclusive_group()
    materialization.add_argument(
        "--lazy-candidates", dest="lazy_candidates", action="store_true",
        help="Enable delayed candidate polynomial expansion (default: on)",
    )
    materialization.add_argument(
        "--no-lazy-candidates", dest="lazy_candidates", action="store_false",
        help="Disable delayed expansion: expand candidate polynomials at enqueue time",
    )
    # Read old command lines without advertising a third, inverse-named switch.
    materialization.add_argument(
        "--eager-candidates", dest="lazy_candidates", action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(lazy_candidates=True)
    parser.add_argument("--no-g-precheck", action="store_true",
                        help="Disable the general G generation top test, not the comparable-LC discard")
    parser.add_argument("--no-j-generation-filter", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--certificates", action="store_true",
                        help="Track/save exact u-vectors; run prints them with the basis unless --no-show-u")

    parser.add_argument("--timeout", type=float, default=None,
                        help="Solver wall time in seconds (default: off; --limits: 30); 0 disables")
    parser.add_argument("--max-basis", type=int, default=None,
                        help="Maximum P records (default: off; --limits: 5000); 0 disables")
    parser.add_argument("--max-candidates", type=int, default=None,
                        help="Generated J+G labels, including early G discards "
                             "(default: off; --limits: 250000); 0 disables")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Head+tail steps (default: off; --limits: 500000); 0 disables")
    parser.add_argument("--max-terms", type=int, default=None,
                        help="Terms per polynomial (default: off; --limits: 100000); 0 disables")
    parser.add_argument("--max-coefficient-bits", type=int, default=None,
                        help="Coefficient bit size (default: off, also with --limits); 0 disables")

    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-strong", action="store_true",
                        help="Add ordinary integer S checks; membership needs --certificates")
    parser.add_argument("--verify-timeout", type=float, default=60.0, help="0 disables")
    parser.add_argument("--verify-max-steps", type=int, default=200000, help="0 disables")
    parser.add_argument("--verify-mode", choices=["naive", "static"], default="naive")


def load_case(args) -> dict:
    case = _load_case_source(args)
    # Validate annotations before solving: they are also used by console/CSV
    # output, so a malformed optional field must not fail after a long run.
    if not isinstance(case.get("name"), str):
        raise ValueError("Input name must be a string")
    if case.get("provenance") is not None and not isinstance(case["provenance"], dict):
        raise ValueError("Input provenance must be an object or null")
    if "input_order" in case:
        order = case["input_order"]
        if not isinstance(order, dict) or not isinstance(order.get("direction"), str):
            raise ValueError("Input input_order requires an object with a direction string")
        permutation = order.get("effective_to_source")
        if (not isinstance(permutation, list)
                or any(type(i) is not int for i in permutation)
                or sorted(permutation) != list(range(1, len(case["polynomials"]) + 1))):
            raise ValueError("Input input_order effective_to_source must be a one-based permutation")
    return reverse_case_generators(case) if getattr(args, "reverse_generators", False) else case


def _load_case_source(args) -> dict:
    if args.case:
        return get_case(args.case)
    if args.input:
        data = json.loads(Path(args.input).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("variables"), list) or not isinstance(data.get("polynomials"), list):
            raise ValueError("Input JSON requires lists 'variables' and 'polynomials'")
        data.setdefault("name", Path(args.input).stem)
        return data
    if args.poly is not None and args.vars:
        return {"name": "custom", "variables": args.vars.split(","), "polynomials": args.poly}
    raise ValueError("Use --case NAME, --input FILE, or --vars x,y --poly ...")


def parse_positions(value: str | None, case: dict) -> list[int] | None:
    if value:
        return [decode_integer(x.strip()) - 1 for x in value.split(",")]
    source = case.get("positions")
    if source is None:
        return None
    if not isinstance(source, list):
        raise ValueError("Input positions must be a list")
    return [exact_integer(x, "position index") - 1 for x in source]


def resolve_orders(args, case: dict) -> tuple[str, str, tuple[int, ...] | None]:
    polynomial_order = args.poly_order or case.get("polynomial_order", "grevlex")
    blocks = parse_blocks(getattr(args, "signature_blocks", None))
    if getattr(args, 'incremental_pot', False):
        if blocks is not None or args.module_order not in (None, 'pot'):
            raise ValueError('--incremental-pot requires --module-order pot (or omit --module-order)')
        return polynomial_order, 'pot', None
    if blocks is not None:
        if args.module_order is not None and args.module_order != "block":
            raise ValueError("--signature-blocks requires --module-order block or no explicit module order")
        module_order = "block"
    else:
        module_order = args.module_order or case.get("module_order", "gvw-g2")
    return polynomial_order, module_order, blocks


def display_sparse_terms(terms, names) -> str:
    """Display exact saved terms when SymPy cannot render a huge coefficient."""
    displayed = []
    for term in terms:
        monomial = "*".join(
            f"{name}^{power}" if power != 1 else name
            for name, power in zip(names, term["exponents"]) if power
        )
        coefficient = term["coefficient"]
        displayed.append(f"{coefficient}*{monomial}" if monomial else coefficient)
    return " + ".join(displayed) or "0"


def print_input_ideal(case: dict, polynomial_order: str) -> None:
    backend = IntegerPolynomialRing(case["variables"], polynomial_order)
    generators = [backend.parse(polynomial) for polynomial in case["polynomials"]]
    print("Input polynomial ideal:")
    print(f"  R = ZZ[{', '.join(case['variables'])}]  (order: {backend.order_name})")
    if not generators:
        print("  I = <0>")
        return
    print("  I = <")
    for index, polynomial in enumerate(generators):
        comma = "," if index + 1 < len(generators) else ""
        try:
            displayed = str(polynomial).replace('**', '^')
        except ValueError:
            # SymPy's decimal display can hit Python's integer digit guard.
            # Use exact hex-encoded sparse coefficients in that case.
            displayed = display_sparse_terms(backend.serialize(polynomial)["terms"], backend.names)
        print(f"    {displayed}{comma}")
    print("  >")
    print(f"  generators: {len(generators)} | variables: {len(case['variables'])}", flush=True)


def execute_case(
    case: dict,
    polynomial_order: str,
    module_order: str,
    options: Options,
    *,
    positions=None,
    block_sizes=None,
    verify=False,
    strong=False,
    verify_timeout=60.0,
    verify_max_steps=200000,
    verify_mode="naive",
    include_basis=True,
    profile_path=None,
    return_result=False,
    defer_basis_serialization=False,
) -> dict | tuple[dict, Result]:
    if profile_path is None:
        result = compute(
            case["polynomials"], case["variables"], polynomial_order,
            module_order, positions=positions, block_sizes=block_sizes,
            options=options, show_basis=False, show_forest=False,
        )
    else:
        path = Path(profile_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            result = compute(
                case["polynomials"], case["variables"], polynomial_order,
                module_order, positions=positions, block_sizes=block_sizes,
                options=options, show_basis=False, show_forest=False,
            )
        finally:
            profiler.disable()
            profiler.dump_stats(str(path))

    # CLI output can serialize each polynomial directly to the destination.
    # Keep the public eager return contract for benchmarks and other callers.
    data = result.to_dict(include_basis=include_basis and not defer_basis_serialization)
    data["case"] = case["name"]
    if case.get("provenance") or "input_order" in case:
        data["benchmark_input"] = input_metadata(case)
    if "input_order" in case:
        data["input_order"] = data["benchmark_input"]["input_order"]
    if profile_path is not None:
        data["profiling"] = {
            "path": str(profile_path),
            "timing_includes_profiler_overhead": True,
        }
    if verify or strong:
        if result.generation_paths is not None:
            from .reconstruction import verify_generation_paths
            data["verification"] = verify_generation_paths(
                result, strong=strong, timeout=verify_timeout, mode=verify_mode,
                max_reduction_steps=verify_max_steps)
        else:
            data["verification"] = verify_result(
                result,
                strong=strong,
                timeout=verify_timeout,
                mode=verify_mode,
                max_reduction_steps=verify_max_steps,
                require_certificates=options.track_representations,
            )
        data["signature_basis_certified"] = data["verification"]["signature_basis_certified"]
        if options.incremental_pot:
            data['strong_basis_certified'] = data['verification'].get('strong_basis_certified', False)
        data["certification_scope"] = data["verification"].get("scope", data["verification"]["status"])
    return (data, result) if return_result else data


def flatten(data: dict) -> dict:
    result = {
        key: data.get(key)
        for key in (
            "case", "polynomial_order", "module_order", "status", "message",
            "basis_size", "syzygy_size", "repeat", "setup_seconds",
        )
    }
    for key in ("input_sha256", "input_sha256_format", "declared_variable_count",
                "active_variable_count", "generator_count"):
        if key in data.get("benchmark_input", {}):
            result[key] = data["benchmark_input"][key]
    if "input_order" in data:
        result["input_generator_order"] = data["input_order"]["direction"]
        result["input_generator_indices"] = json.dumps(data["input_order"]["effective_to_source"],
                                                       separators=(",", ":"))
    for field in fields(Options):
        result[field.name] = data.get("options", {}).get(field.name)
    if data.get("signature_blocks") is not None:
        result["signature_blocks"] = tuple(data["signature_blocks"])
    result.update(data.get("counters", {}))
    result.update({key + "_seconds": value for key, value in data.get("timings_seconds", {}).items()})
    report = data.get("verification", {})
    result["verification"] = report.get("status", "not_requested")
    result["verification_level"] = report.get("verification_level", "not_requested")
    result["integer_encoding"] = data.get("integer_encoding")
    result["membership_verified"] = report.get("membership_verified", False)
    result["signature_basis_certified"] = report.get("signature_basis_certified", False)
    if data.get('options', {}).get('incremental_pot'):
        result['strong_basis_certified'] = report.get('strong_basis_certified', False)
        result['solver_mode'] = 'incremental-pot'
        result['stages_completed'] = data.get('incremental', {}).get('stages_completed', 0)
    post = data.get("polynomial_postprocessing")
    if isinstance(post, dict):
        result["polynomial_postprocessing_status"] = post.get("status")
        result["minimal_gb_size"] = post.get("minimal_basis_size")
        result["reduced_gb_size"] = post.get("reduced_basis_size")
        result["polynomial_postprocessing_seconds"] = post.get("statistics", {}).get("total_seconds")
    return result


def timing_report(timings: dict) -> None:
    rows = (
        ("generation", "Parent/signature descriptors"),
        ("g_generation_test", "G generation top filter"),
        ("j_generation_test", "J generation syzygy filter"),
        ("queue", "Queue and exact-J dedup"),
        ("syzygy", "Syzygy insert/complete/simplify"),
        ("cover", "J cover at queue removal"),
        ("reduction", "J head reductions"),
        ("g_top_test", "G top test at queue removal"),
        ("tail_reduction", "New-element tail reductions"),
        ("materialization", "Lazy polynomial materialization"),
        ("index", "P insertion and index update"),
        ("other", "Dispatch/budget checks/remainder"),
    )
    if 'j_tail_reduction' in timings and 'g_tail_reduction' in timings:
        rows = tuple(item for row in rows for item in (
            (("j_tail_reduction", "J-result tail reductions"),
             ("g_tail_reduction", "G-candidate tail reductions"))
            if row[0] == "tail_reduction" else (row,)))
    if 'syzygy_check' in timings and 'syzygy_complete' in timings:
        rows = tuple(item for row in rows for item in (
            (("syzygy_check", "Syzygy seed redundancy checks"),
             ("syzygy_complete", "Syzygy completion / update"))
            if row[0] == "syzygy" else (row,)))
    if 'incremental_minimization' in timings:
        rows = rows[:-1] + (("incremental_minimization", "Stage minimal strong basis"),) + rows[-1:]
    total = timings["total"]
    print("Time breakdown (exclusive wall-clock sections):")
    for field, name in rows:
        seconds = timings.get(field, 0.0)
        percent = 100 * seconds / total if total else 0.0
        print(f"  {name:34s} {seconds:11.6f}s  {percent:6.2f}%")
    summed = sum(timings.get(field, 0.0) for field, _ in rows)
    print(f"  {'SUM / measured wall':34s} {summed:11.6f}s / {total:.6f}s")
    details = [
        f"{label} {timings[key]:.6f}s"
        for key, label in (("frozen_seed_screen", "frozen seed screen"),
                           ("frozen_multiplier_minimal", "multiplier minimization"))
        if key in timings
    ]
    if details:
        print("  Within parent/signature descriptors: " + " | ".join(details) + " (included above).")
    print(f"  CPU {timings.get('process_cpu', 0.0):.6f}s (not added); setup/verification/JSON writing excluded.")


def console_report(data: dict) -> None:
    module_display = data["module_order"]
    if data.get("signature_blocks"):
        module_display += "(" + ",".join(str(x) for x in data["signature_blocks"]) + ")"
    print(f"Status: {data['status']} | {data['case']} | {data['polynomial_order']} / {module_display}")
    if data.get('options', {}).get('incremental_pot'):
        print(f"P={data['basis_size']} | stage-local S retired | backend: {data['environment']['polynomial_backend']}")
    else:
        print(f"P={data['basis_size']} S={data['syzygy_size']} | backend: {data['environment']['polynomial_backend']}")

    options = data["options"]
    if options.get('incremental_pot'):
        inc = data.get('incremental', {})
        print(f"Mode: incremental POT | completed stages={inc.get('stages_completed', 0)}/{inc.get('stages_total', 0)}")
        print("P is the compacted ideal basis, not all historical insertions. S is stage-local and retired.")
        print("No global signature/syzygy basis or ratio forest is claimed.")
        c = data.get('counters', {})
        print(f"Incremental: prefix pairs skipped={c.get('incremental_frozen_pairs_skipped', 0)} | "
              f"principal pairs skipped={c.get('incremental_principal_skipped', 0)} | "
              f"P records retired={c.get('incremental_p_discarded', 0)}")
        if c.get('incremental_current_principal_skipped', 0):
            print("  Current/current principal pairs covered by the prefix invariant: "
                  f"{c['incremental_current_principal_skipped']}")
        if data['status'] != 'completed':
            print("Partial output contains only the last completed prefix basis, not the in-flight stage or full input ideal.")
    print(
        f"Kernel={options['reduction_kernel']} | reducer={options['reducer_selection']} "
        f"| queue tie={options['queue_tie']} | tail={options['tail_policy']}"
    )
    accelerators = data.get('environment', {}).get('native_accelerators')
    if accelerators is not None:
        from .native_status import status_line
        print(status_line({'native_accelerators': accelerators}))
    if 'native_usage' in data:
        from .native_status import usage_lines
        for line in usage_lines(data['native_usage']):
            print(line)
    tail_on = options.get('tail_reduce', False)
    g_tail_on = tail_on and options.get('g_tail_reduce', True)
    print(f"Tail enabled: J={'on' if tail_on else 'off'} | "
          f"G={'on' if g_tail_on else 'off'}")
    budgets = (
        ("seconds", "max_seconds"),
        ("P", "max_basis"),
        ("generated J+G", "max_candidates"),
        ("head+tail steps", "max_reduction_steps"),
        ("terms/poly", "max_polynomial_terms"),
        ("coefficient bits", "max_coefficient_bits"),
    )
    print("Solver budgets: " + " | ".join(
        f"{name}={options[key] if options[key] is not None else 'off'}" for name, key in budgets
    ))

    timings = data["timings_seconds"]
    counters = data["counters"]
    print(
        f"Compute {timings['total']:.6f}s | cover {timings['cover']:.6f}s "
        f"| J reduction {timings['reduction']:.6f}s | G top test {timings['g_top_test']:.6f}s "
        f"| syzygy {timings['syzygy']:.6f}s"
    )
    print(
        f"Generated J={counters['j_generated']} G={counters['g_generated']} "
        f"| reduced J={counters['j_reduction_calls']} | zero={counters['zero_reductions']} "
        f"| regular steps={counters['regular_steps']} singular steps={counters['singular_steps']}"
    )
    print(
        f"Cover rejected J={counters['cover_rejected_j']} "
        f"| J dedup={counters['j_dedup_discarded'] + counters['j_dedup_replaced']}"
    )
    print(
        f"G discarded={counters['g_discarded']} "
        f"(comparable LC={counters['g_omitted_comparable']}, "
        f"generation top={counters['g_generation_top_rejected']}, "
        f"queue top={counters['g_top_rejected']}) "
        f"| enqueued={counters['g_enqueued']} popped={counters['g_popped']} "
        f"inserted={counters['g_inserted']}"
    )
    print(
        f"Euclidean head steps={counters.get('euclidean_steps', 0)} "
        f"| tail steps={counters.get('tail_steps', 0)} "
        f"| materialized J/G={counters.get('j_materialized', 0)}/{counters.get('g_materialized', 0)}"
    )
    if 'j_tail_reduction' in timings or 'g_tail_reduction' in timings or counters.get('g_tail_skipped'):
        print(f"Tail steps by source: J={counters.get('j_tail_steps', 0)} "
              f"G={counters.get('g_tail_steps', 0)} "
              f"| G tails bypassed={counters.get('g_tail_skipped', 0)}")
    print(
        f"J generation syzygy rejected={counters.get('j_generation_syzygy_rejected', 0)} "
        f"| multiplier dominance rejected={counters.get('j_dominance_rejected', 0)} "
        f"| queue peak={counters['queue_peak']}"
    )
    timing_report(timings)
    if counters.get("tail_cap_reached", 0):
        print(f"Optional tail cap reached {counters['tail_cap_reached']} times")
    if data.get("profiling"):
        print("Profile:", data["profiling"]["path"], "(timings include profiler overhead)")
    if "verification" in data:
        report = data["verification"]
        qualifier = (" (criteria only; NOT certified for the input ideal)"
                     if report.get("passed") and not report.get("membership_verified") else "")
        print("Verification:", report["status"] + qualifier)
        if data["verification"].get("membership_verified"):
            certificate_status = "verified"
        elif data.get("representation_certificate_available"):
            certificate_status = "NOT verified (failed or incomplete certificate check)"
        else:
            certificate_status = "NOT verified (label checks only)"
        print("Membership/signature certificate:", certificate_status)
        if options.get('incremental_pot'):
            print("Strong ideal basis certified:", report.get('strong_basis_certified', False))
            print("Full original-module signature basis certified: False (not the output of this mode)")
        for warning in report.get("warnings", []):
            print("WARNING:", warning)
        for error in data["verification"].get("errors", [])[:5]:
            print("  ", error)
    if data["status"] != "completed":
        print(data["message"])


def ensure_distinct_paths(paths: dict[str, Path]) -> None:
    """Reject output aliases before any result or input file can be overwritten."""
    items = list(paths.items())
    for index, (first_name, first) in enumerate(items):
        for second_name, second in items[index + 1:]:
            if first.resolve() == second.resolve() or (
                first.exists() and second.exists() and first.samefile(second)
            ):
                raise ValueError(f"{first_name} and {second_name} must use different files")


def run_command(args) -> int:
    command_started = perf_counter()
    if getattr(args, 'incremental_pot', False) and (args.show_forest or args.reconstruct_syzygies):
        raise ValueError('--incremental-pot returns an ideal basis, not a full syzygy basis/forest; '
                         'disable --show-forest/--reconstruct-syzygies. --record-paths and --certificates remain supported')
    validate_display_options(args.show_basis, args.show_forest, args.print_style, args.print_width,
                             show_u=args.show_u)
    json_path = Path(args.output)
    if json_path.suffix.lower() == ".csv" or json_path.name.lower() == ".csv":
        raise ValueError("--output is the JSON result file; choose a name other than .csv")
    csv_path = json_path.with_suffix(".csv")
    paths = {"JSON output": json_path, "CSV output": csv_path}
    if args.input:
        paths["input"] = Path(args.input)
    if args.profile:
        paths["profile"] = Path(args.profile)
    ensure_distinct_paths(paths)

    if args.summary_only and (args.record_paths or args.reconstruct_syzygies):
        raise ValueError("Path mode requires a full result; do not use --summary-only")
    if args.reconstruct_syzygies:
        syzygy_path = Path(args.syzygy_output) if args.syzygy_output else json_path.with_name(json_path.stem + "_syzygies.json")
        paths["syzygy output"] = syzygy_path
        ensure_distinct_paths(paths)

    case = load_case(args)
    order_info = case.get("input_order")
    if isinstance(order_info, dict):
        print("Input generator order:", order_info.get("direction", "supplied"),
              "| signature/certificate positions refer to the displayed input sequence", flush=True)
    source = case.get("provenance") or {}
    if source.get("variant"):
        print("Input variant:", source["variant"], "|", source.get("note", ""), file=sys.stderr)

    polynomial_order, module_order, blocks = resolve_orders(args, case)
    positions = parse_positions(args.positions, case)
    if not args.no_print_ideal:
        print_input_ideal(case, polynomial_order)

    from .native_status import print_native_start
    print_native_start(require_native=getattr(args, "require_native", False))
    data, result = execute_case(
        case,
        polynomial_order,
        module_order,
        options_from_args(args),
        positions=positions,
        block_sizes=blocks,
        verify=args.verify,
        strong=args.verify_strong,
        verify_timeout=args.verify_timeout,
        verify_max_steps=args.verify_max_steps,
        verify_mode=args.verify_mode,
        include_basis=not args.summary_only,
        profile_path=args.profile,
        return_result=True,
        defer_basis_serialization=True,
    )
    console_report(data)
    sys.stdout.flush()
    display_seconds = 0.0
    display_started = perf_counter()
    # Display controls are independent of --summary-only (JSON persistence).
    if args.show_basis or args.show_forest or args.show_u is True:
        print()
        print_result(result, show_basis=args.show_basis, show_forest=args.show_forest,
                     style=args.print_style, width=args.print_width, show_u=args.show_u)
    display_seconds += perf_counter() - display_started

    def output_progress(stage):
        started = perf_counter()
        next_report = started + args.progress

        def report(field, records, byte_count=0):
            nonlocal next_report
            now = perf_counter()
            if args.progress > 0 and now >= next_report:
                detail = (f"copied {byte_count / (1024 * 1024):.1f} MiB"
                          if field == "__copy__" else f"field={field} records={records}")
                print(f"[output {now - started:.1f}s] {stage}: {detail}", flush=True)
                next_report = now + args.progress
        return report

    print("[output] Saving source JSON" +
          (" (summary only)" if args.summary_only else f" (full P/S, P={len(result.basis)})") +
          ": " + str(json_path), flush=True)
    output_started = perf_counter()
    if not args.summary_only:
        data.update(result.json_basis_fields())
    saved_source = write_json(json_path, data, progress=output_progress("source JSON"))
    source_json_seconds = perf_counter() - output_started
    write_csv(csv_path, [flatten(data)])
    print(f"[output] Source JSON saved in {source_json_seconds:.3f}s", flush=True)
    post_seconds = final_json_seconds = 0.0

    def save_postprocessing(post_data):
        nonlocal final_json_seconds
        print("[output] Saving polynomial post-processing (reusing saved source records)", flush=True)
        started = perf_counter()
        append_json_fields(json_path, {"polynomial_postprocessing": post_data},
                           previous=saved_source, progress=output_progress("final JSON"))
        final_json_seconds = perf_counter() - started
        write_csv(csv_path, [flatten(data)])

    def completion_report():
        cleanup = getattr(result, "runtime_cleanup_seconds", 0.0)
        print(f"After Compute: solver cleanup {cleanup:.3f}s | source JSON {source_json_seconds:.3f}s | "
              f"polynomial post-processing {post_seconds:.3f}s | display {display_seconds:.3f}s | "
              f"final JSON {final_json_seconds:.3f}s (excluded from Compute)", flush=True)
        print(f"Command elapsed before exit: {perf_counter() - command_started:.3f}s", flush=True)

    requested_polynomials = args.show_minimal_gb or args.show_reduced_gb or args.show_minimal_lt
    if requested_polynomials:
        # The full source is already committed. An interrupted post-processing
        # pass must never lose P/S, certificates or generation paths.
        reason = None
        if data["status"] != "completed":
            reason = "source computation is not completed"
        elif not data.get("verification", {}).get("passed", True):
            reason = "requested source verification did not pass"
        if reason is not None:
            data["polynomial_postprocessing"] = {"status": "skipped", "reason": reason}
            print("Polynomial post-processing skipped:", reason)
            save_postprocessing(data["polynomial_postprocessing"])
        else:
            print("[output] Polynomial post-processing: validating source and extracting minimal terms", flush=True)
            post_started = perf_counter()
            try:
                post = postprocess_basis(result, reduced=args.show_reduced_gb)
            except KeyboardInterrupt:
                post_seconds = perf_counter() - post_started
                data["polynomial_postprocessing"] = {
                    "status": "interrupted", "reason": "No partial reduced basis is published"}
                save_postprocessing(data["polynomial_postprocessing"])
                print("Polynomial post-processing interrupted. Source saved:", json_path.resolve())
                completion_report()
                return 130
            except (ValueError, TypeError) as exc:
                post_seconds = perf_counter() - post_started
                data["polynomial_postprocessing"] = {"status": "failed", "reason": str(exc)}
                save_postprocessing(data["polynomial_postprocessing"])
                completion_report()
                raise
            post_seconds = perf_counter() - post_started
            result.polynomial_postprocessing = post
            data["polynomial_postprocessing"] = post.to_dict(include_basis=False)
            display_started = perf_counter()
            print()
            print_polynomial_output(post, show_minimal_gb=args.show_minimal_gb,
                                    show_reduced_gb=args.show_reduced_gb,
                                    show_minimal_lt=args.show_minimal_lt,
                                    style=args.print_style, width=args.print_width)
            display_seconds += perf_counter() - display_started
            save_postprocessing(post.to_json_dict(include_basis=not args.summary_only,
                                                  _source_json=saved_source))
    print("JSON:", json_path.resolve())
    if data["status"] != "completed":
        completion_report()
        return 2
    if args.reconstruct_syzygies:
        from .path_cli import replay_to_file
        # Persist the signature basis and its paths BEFORE expansion starts.
        replay_to_file(result, syzygy_path, verify=args.verify_syzygies,
                       progress_interval=args.progress, show=False)
    completion_report()
    return 0 if data.get("verification", {}).get("passed", True) else 3


def benchmark_command(args) -> int:
    if args.repeat < 1:
        raise ValueError("--repeat must be positive")
    names = args.cases.split(",") if args.cases else SUITES[args.suite]
    polynomial_orders = args.poly_orders.split(",")
    blocks = parse_blocks(args.signature_blocks)
    if getattr(args, 'incremental_pot', False):
        if blocks is not None or args.module_orders not in (None, 'pot'):
            raise ValueError('--incremental-pot benchmark requires --module-orders pot or its default')
        module_orders = ['pot']
    else:
        module_orders = ['block'] if blocks is not None else (args.module_orders or 'pot,top,gvw-g1,gvw-g2,schreyer').split(',')
    options = options_from_args(args)
    from .native_status import print_native_start
    print_native_start(require_native=getattr(args, "require_native", False))
    if getattr(args, "reverse_generators", False):
        print("Input generator order: reversed for every selected case", flush=True)

    prefix = Path(args.output)
    benchmark_paths = {
        "raw JSONL": Path(str(prefix) + ".raw.jsonl"),
        "raw CSV": Path(str(prefix) + ".raw.csv"),
        "summary CSV": Path(str(prefix) + ".summary.csv"),
        "summary JSON": Path(str(prefix) + ".summary.json"),
    }
    ensure_distinct_paths(benchmark_paths)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    raw: list[dict] = []
    grouped: dict[tuple, list[dict]] = {}
    interrupted = False
    verification_failed = False
    with benchmark_paths["raw JSONL"].open("w", encoding="utf-8") as sink:
        for name, polynomial_order, module_order in product(names, polynomial_orders, module_orders):
            case = get_case(name, reverse_generators=getattr(args, "reverse_generators", False))
            for repeat in range(1, args.repeat + 1):
                try:
                    data = execute_case(
                        case, polynomial_order, module_order, options,
                        block_sizes=blocks,
                        verify=args.verify, strong=args.verify_strong,
                        verify_timeout=args.verify_timeout,
                        verify_max_steps=args.verify_max_steps,
                        verify_mode=args.verify_mode,
                        include_basis=False,
                    )
                except ValueError as exc:
                    data = {
                        "case": name,
                        "polynomial_order": polynomial_order,
                        "module_order": module_order,
                        "status": "unsupported",
                        "message": str(exc),
                        "options": asdict(options),
                    }
                if "input_order" in case:
                    data.setdefault("input_order", case["input_order"])
                data["repeat"] = repeat
                if not data.get("verification", {}).get("passed", True):
                    verification_failed = True
                sink.write(json.dumps(data, ensure_ascii=False) + "\n")
                sink.flush()
                row = flatten(data)
                raw.append(row)
                grouped.setdefault((name, polynomial_order, module_order), []).append(row)
                elapsed = data.get("timings_seconds", {}).get("total", 0)
                print(
                    f"{name:28s} {polynomial_order:8s} {module_order:9s} "
                    f"#{repeat} {data['status']:12s} {elapsed:9.4f}s "
                    f"P={data.get('basis_size', '-')} S={data.get('syzygy_size', '-')}",
                    flush=True,
                )
                if data["status"] in ("limit", "interrupted", "unsupported"):
                    interrupted = data["status"] == "interrupted"
                    break
            if interrupted:
                break

    write_csv(benchmark_paths["raw CSV"], raw)
    summary = []
    for rows in grouped.values():
        completed = [item for item in rows if item["status"] == "completed"]
        measured = completed or rows
        row = dict(measured[0])
        row.pop("repeat", None)
        row["repeats_actual"] = len(rows)
        row["repeats_completed"] = len(completed)
        row["repeat_statuses"] = [item["status"] for item in rows]
        row["all_completed"] = len(completed) == len(rows)
        row["all_verified"] = all(item["verification"] in (
            "passed", "path_certificate_and_criteria", "ideal_criteria_only",
            "ideal_certificate_and_criteria", "path_certificate_and_ideal_criteria") for item in rows)
        row["all_membership_verified"] = all(item.get("membership_verified", False) for item in rows)
        row["all_signature_basis_certified"] = all(item.get("signature_basis_certified", False) for item in rows)
        for key, value in list(row.items()):
            if key in ("repeats_actual", "repeats_completed") or type(value) not in (int, float):
                continue
            values = [item[key] for item in measured if type(item.get(key)) in (int, float)]
            if values:
                row[key] = median(values)
        row["timing_statistic"] = ("median of completed repeats" if completed else
                                   "incomplete repeat; no completed timings")
        summary.append(row)
    write_csv(benchmark_paths["summary CSV"], summary)
    write_json(
        benchmark_paths["summary JSON"],
        {"environment": environment(), "options": asdict(options),
         "requested_repeats": args.repeat, "results": summary,
         **({"reverse_generators": True} if getattr(args, "reverse_generators", False) else {})},
    )
    print("Wrote:", str(prefix) + ".raw.jsonl/.raw.csv/.summary.csv/.summary.json")
    return 2 if interrupted else (3 if verification_failed else 0)


def load_result(path) -> Result:
    """Load exact records across interpreter digit limits.

    Only this saved-result reader overrides JSON integer conversion. User
    polynomial expressions still use the existing restricted parser. Large
    legacy decimal coefficients work whether saved as strings or JSON numbers.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"), parse_int=decode_integer)
    if not isinstance(data, dict):
        raise ValueError("Result JSON must be an object")
    if "basis" not in data or "syzygy_signatures" not in data:
        raise ValueError("Statistics-only JSON cannot be reverified")
    schema = exact_integer(data.get("result_schema_version", 1), "result schema version")
    if schema not in (1, 2):
        raise ValueError("Unsupported result schema version")
    if data.get("coefficient_ring", "ZZ") != "ZZ":
        raise ValueError("Result coefficient_ring must be ZZ")
    for key in ("variables", "input_polynomials", "basis", "syzygy_signatures"):
        if not isinstance(data.get(key), list):
            raise ValueError(f"Result field {key} must be a list")
    if not isinstance(data.get("status"), str) or not isinstance(data.get("message", ""), str):
        raise ValueError("Result status and message must be strings")
    backend = IntegerPolynomialRing(data["variables"], data["polynomial_order"])
    inputs = [backend.deserialize(f) for f in data["input_polynomials"]]
    m = len(inputs)
    positions = data.get("positions_low_to_high", list(range(1, m + 1)))
    if not isinstance(positions, list):
        raise ValueError("positions_low_to_high must be a list")
    positions = [exact_integer(i, "position index") - 1 for i in positions]
    order = ModuleOrder(backend, inputs, data["module_order"], positions,
                        block_sizes=data.get("signature_blocks"))

    def signature(item):
        if not isinstance(item, dict) or not all(k in item for k in ("coefficient", "exponents", "index")):
            raise ValueError("Invalid saved signature record")
        index = exact_integer(item["index"], "signature index")
        if not 1 <= index <= m:
            raise ValueError("Saved signature index must be between 1 and the number of inputs")
        return Signature(decode_integer(item["coefficient"]),
                         validate_monomial(item["exponents"], len(backend.names), "signature exponents"),
                         index - 1)

    basis = []
    for item in data["basis"]:
        if not isinstance(item, dict) or "signature" not in item or "polynomial" not in item:
            raise ValueError("Invalid saved P record")
        polynomial = backend.deserialize(item["polynomial"])
        backend.validate_polynomial(polynomial, nonzero=True)
        basis.append(Label(signature(item["signature"]), polynomial))
    syzygies = [signature(item) for item in data["syzygy_signatures"]]
    for key, actual in (("basis_size", len(basis)), ("syzygy_size", len(syzygies))):
        if key in data and exact_integer(data[key], key) != actual:
            raise ValueError(f"{key} disagrees with the saved records")
    option_source = data.get("options", {})
    if not isinstance(option_source, dict):
        raise ValueError("Result options must be an object")
    allowed = {field.name for field in fields(Options)}
    options = Options(**{key: value for key, value in option_source.items() if key in allowed})
    # Historical 'reduce' records remain readable as data only. Constructing
    # or running a solver with these Options is still explicitly prohibited.
    if options.g_policy == "reduce":
        replace(options, g_policy="discard").validate()
    else:
        options.validate()
    result = Result(data["status"], data.get("message", ""), backend, order,
                    inputs, basis, syzygies, Stats(), options)
    # Timings are descriptive, never evidence of algebraic correctness. Restore
    # only known stored sections; Stats derives "other" from those sections.
    def saved_seconds(value, name, *, allow_negative=False):
        if type(value) not in (int, float):
            raise ValueError(f"{name} must be a finite number")
        try:
            seconds = float(value)
        except OverflowError as exc:
            raise ValueError(f"{name} must be a finite number") from exc
        if not isfinite(seconds) or (seconds < 0 and not allow_negative):
            raise ValueError(f"{name} must be a finite nonnegative number")
        return seconds

    def saved_nanoseconds(value, name):
        scaled = saved_seconds(value, name) * 1e9
        if not isfinite(scaled):
            raise ValueError(f"{name} must fit the finite nanosecond timing range")
        return round(scaled)

    result.setup_seconds = saved_seconds(data.get("setup_seconds", 0.0), "setup_seconds")
    saved_times = data.get("timings_seconds", {})
    if not isinstance(saved_times, dict):
        raise ValueError("Result timings_seconds must be an object")
    result.stats.total_ns = saved_nanoseconds(saved_times.get("total", 0.0), "total timing")
    result.stats.cpu_ns = saved_nanoseconds(saved_times.get("process_cpu", 0.0), "process_cpu timing")
    subtimers = ("j_tail_reduction", "g_tail_reduction", "syzygy_check", "syzygy_complete",
                 "incremental_minimization", "frozen_seed_screen", "frozen_multiplier_minimal")
    for key in (*TIMERS, *subtimers):
        if key in saved_times:
            result.stats.times_ns[key] = saved_nanoseconds(saved_times[key], key + " timing")
    if "other" in saved_times:
        saved_seconds(saved_times["other"], "other timing", allow_negative=True)
    if options.incremental_pot:
        if data.get('solver_mode') != 'incremental-pot':
            raise ValueError('Incremental options and saved solver_mode disagree')
        result.incremental_metadata = data.get('incremental', {})
    elif data.get('solver_mode') == 'incremental-pot':
        raise ValueError('Incremental result requires matching Options.incremental_pot')
    if "insertion_history" in data:
        result.insertion_history = deserialize_history(
            data["insertion_history"], len(basis), m)
    # Counters are descriptive, not evidence that the algebraic tests passed.
    saved_counts = data.get("counters", {})
    if not isinstance(saved_counts, dict):
        raise ValueError("Result counters must be an object")
    for key, value in saved_counts.items():
        count = exact_integer(value, f"counter {key}")
        if count < 0:
            raise ValueError("Counters must be nonnegative")
        result.stats.counters[key] = count
    certificate = data.get("representation_certificate")
    if certificate is not None:
        if not isinstance(certificate, dict) or certificate.get("format") != "gvwzz-u-vectors-v1":
            raise ValueError("Unsupported representation certificate")
        def vectors(key, count):
            rows = certificate.get(key)
            if not isinstance(rows, list) or len(rows) != count:
                raise ValueError(f"Certificate {key} has wrong number of vectors")
            out = []
            for row in rows:
                if not isinstance(row, list) or len(row) != m:
                    raise ValueError("Certificate vector has wrong module dimension")
                out.append(tuple(backend.deserialize(f, allow_expression=False) for f in row))
            return out
        result.basis_representations = vectors("basis", len(basis))
        result.syzygy_representations = vectors("syzygies", len(syzygies))
    if "generation_paths" in data:
        from .paths import GenerationPaths
        result.generation_paths = GenerationPaths.from_dict(data["generation_paths"], result)
    return result


def add_display_options(parser) -> None:
    for name, text in (
        ("minimal-gb", "a minimal strong polynomial Groebner basis"),
        ("reduced-gb", "the balanced-remainder reduced strong polynomial Groebner basis"),
        ("minimal-lt", "minimal polynomial leading terms, including coefficients"),
    ):
        group = parser.add_mutually_exclusive_group()
        dest = "show_" + name.replace("-", "_")
        group.add_argument("--show-" + name, dest=dest, action="store_true",
                           help="Post-process completed P and print " + text + " (default: off)")
        group.add_argument("--no-show-" + name, dest=dest, action="store_false",
                           help="Do not request this polynomial output")
        parser.set_defaults(**{dest: False})
    parser.add_argument("--show-basis", action=argparse.BooleanOptionalAction, default=True,
                        help="Print aligned P and S tables (default: on)")
    parser.add_argument("--show-forest", action=argparse.BooleanOptionalAction, default=False,
                        help="Print same-ratio forests from insertion history (default: off)")
    parser.add_argument("--show-u", action=argparse.BooleanOptionalAction, default=None,
                        help="Print full u-vectors for P and S. Default: with basis if stored. "
                             "run --show-u also enables certificates; show never recomputes.")
    parser.add_argument("--print-style", choices=("ascii", "unicode"), default="ascii",
                        help="Math/branch glyphs for console output (default: ascii)")
    parser.add_argument("--print-width", type=int, default=100,
                        help="Target width for wrapping long records, at least 40 (default: 100)")


def add_input_order_options(parser) -> None:
    parser.add_argument(
        "--reverse-generators", "--reversed", dest="reverse_generators",
        action=argparse.BooleanOptionalAction, default=False,
        help="Reverse the supplied input generator list (default: off); variables, "
             "monomial order and candidate-queue strategy are unchanged",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Coefficient-aware GVW-type algorithm over ZZ")
    parser.add_argument("--version", action="version", version=f"gvwzz {VERSION}")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("list", help="List built-in test systems")

    doctor = commands.add_parser("doctor", help="Check this Python's native accelerators without solving")
    doctor.add_argument("--json", action="store_true", help="Print machine-readable diagnostics")
    doctor.add_argument("--require-native", action="store_true",
                        help="Exit with status 1 if any native accelerator is unavailable")

    export = commands.add_parser("export-case", help="Inspect or save a built-in input without solving")
    export.add_argument("name")
    add_input_order_options(export)
    export.add_argument("--output", help="Save input JSON to a new file; default: print JSON")

    run = commands.add_parser("run", help="Run a built-in, JSON, or inline system")
    source = run.add_mutually_exclusive_group()
    source.add_argument("--case")
    source.add_argument("--input")
    run.add_argument("--vars")
    run.add_argument("--poly", action="append")
    run.add_argument("--poly-order", choices=["lex", "grlex", "grevlex", "deglex", "degrevlex"])
    run.add_argument("--module-order",
                     choices=["pot", "top", "block", "gvw-g1", "gvw-g2", "g1", "g2", "schreyer"])
    run.add_argument("--signature-blocks",
                     help="Pure block module order only; all inputs are still inserted at once")
    run.add_argument("--positions", help="Input positions from lowest to highest, e.g. 1,3,2")
    run.add_argument("--output", default="results/run.json")
    run.add_argument("--no-print-ideal", action="store_true")
    add_display_options(run)
    add_input_order_options(run)
    run.add_argument("--summary-only", action="store_true")
    run.add_argument("--reconstruct-syzygies", action="store_true",
                     help="Record paths, save the basis, then replay final S only")
    run.add_argument("--syzygy-output", help="Separate syzygy JSON for --reconstruct-syzygies")
    run.add_argument("--verify-syzygies", action="store_true",
                     help="Also check each expanded zero identity (separate from replay time)")
    run.add_argument("--profile")
    add_solver_options(run)

    benchmark = commands.add_parser("benchmark", help="Compare cases and orders")
    benchmark.add_argument("--suite", choices=list(SUITES), default="quick")
    benchmark.add_argument("--cases")
    benchmark.add_argument("--poly-orders", default="grevlex")
    benchmark.add_argument("--module-orders", default=None)
    benchmark.add_argument("--signature-blocks")
    benchmark.add_argument("--repeat", type=int, default=3)
    benchmark.add_argument("--output", default="results/benchmark")
    add_solver_options(benchmark)
    add_input_order_options(benchmark)

    show = commands.add_parser("show", help="Pretty-print a saved result without recomputation")
    show.add_argument("file")
    add_display_options(show)

    verify = commands.add_parser("verify", help="Recheck a saved full result")
    verify.add_argument("file")
    verify.add_argument("--strong", action="store_true",
                        help="Add ordinary integer S checks; does not replace representation certificates")
    verify.add_argument("--require-certificates", action="store_true",
                        help="Fail unless exact representation vectors are present and valid")
    verify.add_argument("--timeout", type=float, default=60.0)
    verify.add_argument("--max-steps", type=int, default=200000)
    verify.add_argument("--verify-mode", choices=["naive", "static"], default="naive")

    from .path_cli import add_path_commands
    add_path_commands(commands)

    random_parser = commands.add_parser("random", help="Write a reproducible random input")
    add_input_order_options(random_parser)
    random_parser.add_argument("--seed", type=int, default=1)
    random_parser.add_argument("--nvars", type=int, default=2)
    random_parser.add_argument("--ngens", type=int, default=3)
    random_parser.add_argument("--degree", type=int, default=2)
    random_parser.add_argument("--terms", type=int, default=4)
    random_parser.add_argument("--coeff-bound", type=int, default=5)
    random_parser.add_argument("--output", default="examples/random.json")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            from .native_status import doctor_command
            return doctor_command(json_output=args.json, require_native=args.require_native)
        if args.command in ("reconstruct", "verify-paths", "verify-syzygies"):
            from .path_cli import dispatch_path_command
            return dispatch_path_command(args)
        if args.command == "export-case":
            data = get_case(args.name, reverse_generators=args.reverse_generators)
            data["benchmark_input"] = input_metadata(data)
            text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
            if args.output:
                path = Path(args.output)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("x", encoding="utf-8") as stream:
                    stream.write(text)
                print("Wrote", path)
            else:
                print(text, end="")
            return 0
        if args.command == "list":
            for name, case in CASES.items():
                print(f"{name:28s} {case.get('description', '')}")
            return 0
        if args.command == "run":
            return run_command(args)
        if args.command == "benchmark":
            return benchmark_command(args)
        if args.command == "show":
            validate_display_options(args.show_basis, args.show_forest, args.print_style, args.print_width,
                                     show_u=args.show_u)
            result = load_result(args.file)
            print_result(result, show_basis=args.show_basis, show_forest=args.show_forest,
                         style=args.print_style, width=args.print_width, show_u=args.show_u,
                         show_minimal_gb=args.show_minimal_gb, show_reduced_gb=args.show_reduced_gb,
                         show_minimal_lt=args.show_minimal_lt)
            return 0
        if args.command == "verify":
            result = load_result(args.file)
            data = verify_result(
                result,
                args.strong,
                args.timeout,
                mode=args.verify_mode,
                max_reduction_steps=args.max_steps,
                require_certificates=args.require_certificates,
            )
            print(json.dumps(data, ensure_ascii=False, indent=2))
            return 0 if data["passed"] else 3
        if args.command == "random":
            data = random_case(
                args.seed, args.nvars, args.ngens, args.degree,
                args.terms, args.coeff_bound,
            )
            if args.reverse_generators:
                data = reverse_case_generators(data)
                data["benchmark_input"] = input_metadata(data)
            write_json(args.output, data)
            print("Wrote", args.output)
            return 0
    except (ValueError, TypeError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 1
