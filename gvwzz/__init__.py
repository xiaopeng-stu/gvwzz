"""Public API for exact GVW-type signature bases over integer polynomial rings."""

from ._version import __version__
from .core import Label, Signature
from .options import Options
from .result import Result
from .solver import GVWInteger, compute
from .incremental import IncrementalPOT
from .forest import build_forest
from .printing import (
    format_basis, format_forest, format_result, format_u, format_vector,
    print_basis, print_forest, print_result, print_u,
    format_polynomial_output, print_polynomial_output,
)
from .postprocessing import (
    postprocess_basis, minimal_groebner_basis, reduced_groebner_basis,
    minimal_leading_terms, PolynomialBasisOutput, PolynomialPostprocessingError,
)
from .reconstruction import (
    reconstruct_syzygies, SyzygyReplay, PathReconstruction,
    ReconstructionError, verify_generation_paths,
)


def reverse_generators(case: dict) -> dict:
    """Return a detached case with its generator list reversed; no computation."""
    from .cases import reverse_case_generators
    return reverse_case_generators(case)


# Stable public imports: implementation modules can be read independently.
__all__ = [
    "Label", "Signature", "Options", "Result", "GVWInteger", "compute",
    "IncrementalPOT", "reverse_generators",
    "format_basis", "format_forest", "format_result", "format_u", "format_vector",
    "print_basis", "print_forest", "print_result", "print_u", "build_forest",
    "postprocess_basis", "minimal_groebner_basis", "reduced_groebner_basis",
    "minimal_leading_terms", "PolynomialBasisOutput", "PolynomialPostprocessingError",
    "format_polynomial_output", "print_polynomial_output",
    "reconstruct_syzygies", "SyzygyReplay", "PathReconstruction",
    "ReconstructionError", "verify_generation_paths",
]
