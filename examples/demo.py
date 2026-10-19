"""Compute the included ideal, save its generation paths, and replay syzygies."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gvwzz import (
    Options, compute, format_vector, reconstruct_syzygies,
    verify_generation_paths,
)
from gvwzz.cli import write_json


def main() -> None:
    result = compute(
        ["4*x^2", "9*y^4", "6*x*y^2+y"], ["x", "y"],
        polynomial_order="lex", module_order="pot",
        options=Options(record_paths=True, g_first=False),
        show_basis=False, show_u=False, show_forest=False,
    )
    if result.status != "completed":
        raise RuntimeError(result.message)
    report = verify_generation_paths(result)
    if not report["signature_basis_certified"]:
        raise RuntimeError(f"Path/basis verification failed: {report}")
    write_json("results/demo_paths.json", result.to_dict())
    post = reconstruct_syzygies(result, verify=True)
    write_json("results/demo_syzygies.json", post.to_dict())
    for record in post.records:
        print(f"z{record.index + 1} = {format_vector(record.vector, result.backend)}")
    print("Saved results/demo_paths.json and results/demo_syzygies.json")


if __name__ == "__main__":
    main()
