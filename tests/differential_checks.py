"""Deterministic, bounded end-to-end comparison with an earlier source tree.

Run with a separately extracted reference release, for example::

    python -m tests.differential_checks --reference ../gvwzz-reference

Each version runs in a fresh interpreter. Ordered labels, syzygies, insertion
history, paths or representation vectors, and logical algorithm counters must
match exactly. Both outputs also pass independent certificate/criteria checks.
Timings, cache hits, allocations and index-work counters are deliberately not
part of the comparison: those are the implementation costs being optimized.
No random case may silently pass after hitting a computation or time limit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys


SEED = 20261015
LOGICAL_COUNTERS = """
parent_pairs j_generated g_generated g_discarded g_omitted_comparable
j_enqueued g_enqueued j_popped g_popped j_dedup_discarded j_dedup_replaced
j_dedup_settled queue_stale_entries queue_peak iterations
cover_calls cover_rejected_j j_reduction_calls zero_reductions
regular_steps singular_steps g_top_rejected g_inserted j_inserted
principal_syzygies_detected principal_syzygies_unknown
syzygy_insert_calls syzygy_seed_rejected syzygy_g_signatures
syzygy_g_skipped_old syzygy_g_skipped_seed syzygy_new_signatures
syzygy_simplified s_peak p_peak bezout_adjustments
euclidean_steps euclidean_regular_steps euclidean_singular_steps
euclidean_sigdrop_skipped tail_reduction_calls tail_steps tail_exact_steps
tail_euclidean_steps j_zero_reductions g_zero_reductions
g_generation_tests g_generation_top_rejected j_generation_tests
j_generation_syzygy_rejected j_tail_reduction_calls g_tail_reduction_calls
j_tail_steps g_tail_steps j_tail_exact_steps g_tail_exact_steps
j_tail_euclidean_steps g_tail_euclidean_steps g_tail_skipped
max_polynomial_terms max_coefficient_bits max_signature_bits
""".split()

RESULT_FIELDS = """
status message variables coefficient_ring polynomial_order module_order
positions_low_to_high input_lm_weights input_degree_weights
zero_input_weight_convention signature_blocks signature_block_positions
block_order_description input_polynomials basis_size syzygy_size solver_mode
basis syzygy_signatures insertion_history representation_certificate
generation_paths
""".split()


def make_cases() -> list[dict]:
    """Forty cases; a fixed RNG controls coefficients, never global state."""
    rng = random.Random(SEED)
    cases = []
    orders = ("pot", "top", "g1", "g2", "schreyer", "block")
    profiles = (
        dict(lazy_candidates=True, j_deduplicate=True,
             track_representations=True, reduction_kernel="heap"),
        dict(lazy_candidates=False, j_deduplicate=False, record_paths=True,
             reduction_kernel="copy", g_first=False, g_precheck=False,
             queue_tie="fifo"),
        dict(lazy_candidates=True, j_deduplicate=False, record_paths=True,
             reduction_kernel="heap", j_generation_filter=False,
             queue_tie="small-lm"),
        dict(lazy_candidates=False, j_deduplicate=True,
             track_representations=True, reduction_kernel="packed",
             g_first=False, g_precheck=False),
        dict(lazy_candidates=True, j_deduplicate=True, record_paths=True,
             reduction_kernel="copy", cover_mode="naive", euclidean=False,
             tail_reduce=False, queue_tie="fifo"),
        dict(lazy_candidates=False, j_deduplicate=False,
             track_representations=True, reduction_kernel="heap",
             tail_policy="selective", tail_step_cap=2, queue_tie="small-lm"),
    )
    for order in orders:
        for profile, switches in enumerate(profiles):
            # Degree <= 2 in two active variables keeps these tests bounded.
            a, b, c = [rng.choice((-6, -4, -3, -2, 2, 3, 4, 6))
                       for _ in range(3)]
            d, e = [rng.choice((-2, -1, 1, 2)) for _ in range(2)]
            if profile == 0:
                polynomials = [f"({a})*x^2", f"({b})*y^2",
                               f"({c})*x*y+({d})*y"]
            elif profile == 1:
                polynomials = [f"({a})*x^2+({d})*y+({e})",
                               f"({b})*y^2+({e})*x+({d})",
                               f"({c})*x*y+({d})*x+({e})*y"]
            elif profile == 2:
                polynomials = [f"({a})*x+({d})*y+({e})",
                               f"({b})*y^2+({d})", f"({c})*x*y"]
            elif profile == 3:
                polynomials = [f"({a})*x^2+({d})*y", "0",
                               f"({b})*x*y+({e})*y"]
            elif profile == 4:
                first = f"({a})*x+({d})*y"
                polynomials = [first, first, f"({b})*y^2+({e})*y"]
            else:
                polynomials = [f"({a})*x^2+({d})*y^2",
                               f"({b})*x*y", f"({c})*y^2"]
            polynomial_order = ("grevlex", "grlex", "lex", "grevlex",
                                "grlex", "grevlex")[profile]
            if order == "g1" and polynomial_order == "lex":
                polynomial_order = "grevlex"  # g1 does not support lex.
            cases.append(dict(
                name=f"{order}-{profile + 1}", variables=["x", "y"],
                polynomials=polynomials, polynomial_order=polynomial_order,
                module_order=order,
                positions=[2, 0, 1] if profile % 2 else [0, 1, 2],
                block_sizes=[1, 2] if order == "block" else None,
                options=dict(switches),
            ))
    for dimension, order in ((8, "pot"), (48, "pot"), (8, "g2"), (48, "g2")):
        names = [f"x{i}" for i in range(dimension)]
        x, y = names[0], names[-1]
        cases.append(dict(
            name=f"sparse-{dimension}-{order}", variables=names,
            polynomials=[f"-4*{x}^2", f"9*{y}^2", f"6*{x}*{y}+{y}"],
            polynomial_order="grevlex", module_order=order,
            positions=[2, 0, 1], block_sizes=None,
            options=dict(record_paths=True, lazy_candidates=dimension == 48,
                         j_deduplicate=order == "pot"),
        ))
    return cases


def run_worker(source: str) -> None:
    # This must precede all gvwzz imports; the driver imports no solver module.
    sys.path.insert(0, str(Path(source).resolve()))
    from gvwzz import Options, compute, verify_generation_paths
    from gvwzz.verification import verify_result

    case = json.load(sys.stdin)
    options = Options(**case["options"], max_seconds=15, max_basis=180,
                      max_candidates=25000, max_reduction_steps=100000)
    result = compute(case["polynomials"], case["variables"],
                     polynomial_order=case["polynomial_order"],
                     module_order=case["module_order"],
                     positions=case["positions"], block_sizes=case["block_sizes"],
                     options=options, show_basis=False, show_u=False)
    if result.status != "completed":
        raise AssertionError(f'{case["name"]}: {result.status}: {result.message}')
    if options.record_paths:
        report = verify_generation_paths(result, strong=True, timeout=15,
                                         max_reduction_steps=100000)
    else:
        report = verify_result(result, strong=True, mode="static", timeout=15,
                               max_reduction_steps=100000, require_certificates=True)
    if not report["passed"] or not report["signature_basis_certified"]:
        raise AssertionError(f'{case["name"]}: independent verification failed: {report}')
    raw = result.to_dict()
    answer = {key: raw[key] for key in RESULT_FIELDS if key in raw}
    answer["logical_counters"] = {key: raw["counters"].get(key, 0)
                                   for key in LOGICAL_COUNTERS}
    answer["independent_verification"] = dict(
        passed=report["passed"], membership_verified=report["membership_verified"],
        signature_basis_certified=report["signature_basis_certified"])
    json.dump(answer, sys.stdout, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")


def first_difference(left, right, path="result") -> str | None:
    if type(left) is not type(right):
        return f"{path}: types {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return f"{path}: keys differ: {sorted(set(left) ^ set(right))}"
        for key in sorted(left):
            difference = first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
    elif isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: lengths {len(left)} != {len(right)}"
        for index, (a, b) in enumerate(zip(left, right)):
            difference = first_difference(a, b, f"{path}[{index}]")
            if difference:
                return difference
    elif left != right:
        return f"{path}: {left!r} != {right!r}"
    return None


def evaluate(source: Path, case: dict, timeout: float) -> dict:
    try:
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(source)],
            input=json.dumps(case), text=True, capture_output=True, timeout=timeout,
            cwd=source, env={**os.environ, "PYTHONHASHSEED": "0"}, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f'{case["name"]}: {source.name} exceeded {timeout}s') from exc
    if process.returncode:
        raise RuntimeError(f'{case["name"]}: {source.name} failed\n{process.stderr}')
    return json.loads(process.stdout)


def main() -> None:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path,
                        help="Explicit path to a separately built reference source tree.")
    parser.add_argument("--candidate", type=Path, default=here)
    parser.add_argument("--case", action="append", help="Run only this named case (repeatable).")
    parser.add_argument("--timeout", type=float, default=45,
                        help="Per-version process timeout in seconds (default: 45).")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        run_worker(args.worker)
        return
    if args.reference is None:
        parser.error("--reference is required for this optional historical comparison")
    for source in (args.reference, args.candidate):
        if not (source / "gvwzz" / "solver.py").is_file():
            parser.error(f"No gvwzz source tree at {source}")
    cases = make_cases()
    if args.case:
        unknown = set(args.case) - {case["name"] for case in cases}
        if unknown:
            parser.error(f"Unknown cases: {sorted(unknown)}")
        cases = [case for case in cases if case["name"] in args.case]
    digest = hashlib.sha256()
    for case in cases:
        reference = evaluate(args.reference.resolve(), case, args.timeout)
        candidate = evaluate(args.candidate.resolve(), case, args.timeout)
        difference = first_difference(reference, candidate)
        if difference:
            raise AssertionError(f'{case["name"]}: {difference}\ninput={json.dumps(case, sort_keys=True)}')
        digest.update(json.dumps([case, candidate], sort_keys=True,
                                 separators=(",", ":")).encode())
        print(f'PASS {case["name"]}: P={candidate["basis_size"]} '
              f'S={candidate["syzygy_size"]}', flush=True)
    print(f"PASS {len(cases)} differential cases; seed={SEED}; sha256={digest.hexdigest()}")


if __name__ == "__main__":
    main()
