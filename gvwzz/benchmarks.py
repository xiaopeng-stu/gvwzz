"""Classical benchmark inputs over ZZ, with explicit normalization conventions.

These independent generators implement mathematical definitions, not external
solver code. They use only the Python standard library. The benchmark catalog
and provenance are input data, not changes to the GVW algorithm.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations
from math import lcm, prod

REFERENCE_URL = (
    "https://raw.githubusercontent.com/sumiya11/Groebner.jl/master/"
    "src/utils/examples.jl"
)
REFERENCE_PAPER = "https://arxiv.org/abs/2304.06935v2"
REFERENCE_DATE = "2026-10-06"
FAMILY_RANGES = {"eco": (2, 13), "noon": (2, 9), "reimer": (2, 8), "henrion": (5, 7)}


def _check_index(family: str, n: int) -> None:
    low, high = FAMILY_RANGES[family]
    if type(n) is not int or not low <= n <= high:
        raise ValueError(f"{family}: choose an integer {low} <= n <= {high}")


def _record(family: str, n: int, names: list[str], polynomials: list[str],
            order: str, note: str, multipliers: list[int] | None = None) -> dict:
    return {
        "name": f"{family}{n}",
        "variables": names,
        "polynomials": polynomials,
        "description": f"{family.title()}-{n} over ZZ. {note}",
        "generator_order": order,
        "provenance": {
            "family": family, "parameter": n,
            "definition_version": "gvwzz-classic-zz-v1",
            "source_url": REFERENCE_URL, "source_accessed": REFERENCE_DATE,
            "reference_paper": REFERENCE_PAPER,
            "source_procedure": {"eco": "econ", "noon": "noonn",
                                 "reimer": "reimern", "henrion": f"henrion{n}"}[family],
            "coefficient_domain": "ZZ",
            "variant": ("denominator-cleared-ZZ" if family == "henrion"
                        else "integer-equations-over-ZZ"),
            "normalization": ("Each rational equation is multiplied by its least positive "
                              "coefficient-denominator LCM; no subsequent content removal."
                              if family == "henrion" else
                              "none; use the displayed integer coefficients without content removal"),
            "denominator_multipliers": multipliers or [1] * len(polynomials),
            "generator_order": order,
            "note": note,
            "historical_timing_reproduction": False,
        },
    }


def eco(n: int) -> dict:
    """Economic system, n variables and n equations, affine equation last."""
    _check_index("eco", n)
    names = [f"x{i}" for i in range(1, n + 1)]
    equations = []
    for k in range(1, n):
        terms = [f"x{k}"]
        terms.extend(f"x{j}*x{j+k}" for j in range(1, n - k))
        equations.append(f"x{n}*(" + " + ".join(terms) + f") - {k}")
    equations.append(" + ".join(names[:-1]) + " + 1")
    return _record("eco", n, names, equations, "k=1,...,n-1; affine equation last",
                   "The final equation is x1+...+x(n-1)+1, not sum-1.")


def noon(n: int) -> dict:
    """Nonhomogeneous Noon system in the 10, -11, 10 integer convention."""
    _check_index("noon", n)
    names = [f"x{i}" for i in range(1, n + 1)]
    equations = []
    for i in range(1, n + 1):
        squares = " + ".join(f"x{j}^2" for j in range(1, n + 1) if j != i)
        equations.append(f"10*x{i}*({squares}) - 11*x{i} + 10")
    return _record("noon", n, names, equations, "i=1,...,n",
                   "Nonhomogeneous. No h variable; do not divide all equations by 10 over ZZ.")


def reimer(n: int) -> dict:
    """Alternating power sums of degrees 2,...,n+1, all with constant -1."""
    _check_index("reimer", n)
    names = [f"x{i}" for i in range(1, n + 1)]
    equations = []
    for degree in range(2, n + 2):
        equation = "".join((" + " if i % 2 else " - ") + f"2*x{i}^{degree}"
                           for i in range(1, n + 1)).removeprefix(" + ")
        equations.append(equation + " - 1")
    return _record("reimer", n, names, equations, "ascending degree 2,...,n+1",
                   "Keep the factor 2 and the constant -1 in every equation.")


def _principal_minor(n: int, subset: tuple[int, ...]) -> Fraction:
    """Principal minor of tridiag(-1; 2,...,2,(n+1)/n; -1).

Indices are 1-based. A consecutive block of length r has determinant r+1,
except a block ending at n, whose determinant is (n+r)/n.
    """
    answer = Fraction(1)
    start = 0
    while start < len(subset):
        end = start
        while end + 1 < len(subset) and subset[end + 1] == subset[end] + 1:
            end += 1
        length = end - start + 1
        answer *= (Fraction(n + length, n) if subset[end] == n else length + 1)
        start = end + 1
    return answer


def henrion(n: int) -> dict:
    """Exact integer lifts of the reference Henrion-5, -6 and -7 systems.

The compact determinant description was checked coefficient by coefficient
against the reference equations. Henrion-5 retains the reference's auxiliary
t variable and final linear equation; it is not silently changed to a 5x5 input.
    """
    _check_index("henrion", n)
    names = [f"f{i}" for i in range(1, n + 1)]
    eigenvalues = [4 * i * i - 1 for i in range(1, n + 1)]
    equations, multipliers = [], []
    for degree in range(n, 0, -1):
        subsets = list(combinations(range(1, n + 1), degree))
        coefficients = [_principal_minor(n, subset) for subset in subsets]
        constant = sum(prod(eigenvalues[i - 1] for i in subset) for subset in subsets)
        multiplier = lcm(*(c.denominator for c in coefficients))
        terms = []
        for subset, coefficient in zip(subsets, coefficients):
            integer = coefficient * multiplier
            if integer.denominator != 1:
                raise ArithmeticError("Henrion denominator clearing failed")
            monomial = "*".join(f"f{i}" for i in subset)
            terms.append(f"{integer.numerator}*{monomial}")
        equations.append(" + ".join(terms) + f" - {constant * multiplier}")
        multipliers.append(multiplier)
    if n == 5:
        names.append("t")
        equations.append("f1 + 2*f2 + 3*f3 + 4*f4 + 5*f5 + 6*t")
        multipliers.append(1)
    note = ("Retains 6 variables f1,...,f5,t and the auxiliary linear equation."
            if n == 5 else f"Uses {n} variables f1,...,f{n}, without an auxiliary variable.")
    result = _record("henrion", n, names, equations,
                     "descending degree n,...,1; auxiliary equation last when n=5",
                     note, multipliers)
    result["provenance"]["compact_definition"] = (
        "Principal minors of tridiag(-1;2,...,2,(n+1)/n;-1), "
        "target spectrum 4*i*i-1 (i=1,...,n). "
        "An equivalent formula derived from and checked against the cited explicit equations."
    )
    return result


def classical_cases() -> dict[str, dict]:
    """Construct the four additional families without running a solver."""
    cases = {}
    for family, generator in (("eco", eco), ("noon", noon),
                               ("reimer", reimer), ("henrion", henrion)):
        low, high = FAMILY_RANGES[family]
        for n in range(low, high + 1):
            case = generator(n)
            cases[case["name"]] = case
    return cases
