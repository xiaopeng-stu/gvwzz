"""Human-readable formatting of records, forests and optional polynomial bases.

P/S, u and forest views never alter or normalize original records. The three
explicit polynomial-basis views compute detached post-processing when requested;
they do not rerun GVW or modify P/S. ASCII remains the portable default.
"""
from __future__ import annotations

import sys
import textwrap
from typing import TYPE_CHECKING, TextIO

from .core import Signature, encode_integer, exact_integer, validate_monomial
from .forest import ForestUnavailable, Ratio, build_forest

if TYPE_CHECKING:
    from .polynomial import IntegerPolynomialRing
    from .solver import Result

_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_SUP = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def validate_display_options(show_basis: bool = True, show_forest: bool = False,
                             style: str = "ascii", width: int = 100,
                             *, show_u: bool | None = None) -> None:
    if show_u is not None and type(show_u) is not bool:
        raise ValueError("show_u must be True, False, or None (automatic)")
    if type(show_basis) is not bool or type(show_forest) is not bool:
        raise ValueError("show_basis and show_forest must be booleans")
    if style not in ("ascii", "unicode"):
        raise ValueError("Print style must be ascii or unicode")
    if exact_integer(width, "print width") < 40:
        raise ValueError("Print width must be at least 40")


def _symbol(letter: str, index: int, style: str) -> str:
    number = encode_integer(index)
    if style == "unicode" and number.isdecimal():
        return letter + number.translate(_SUB)
    return f"{letter}_{number}" if letter == "e" else f"{letter}{number}"


def format_monomial(exponents, variables, *, style: str = "ascii") -> str:
    validate_display_options(style=style)
    variables = tuple(variables)
    exponents = validate_monomial(exponents, len(variables))
    factors = []
    for variable, exponent in zip(variables, exponents):
        if not exponent:
            continue
        if exponent == 1:
            factors.append(variable)
        else:
            power = encode_integer(exponent)
            factors.append(variable + (power.translate(_SUP) if style == "unicode" and power.isdecimal()
                                       else "^" + power))
    return ("·" if style == "unicode" else "*").join(factors) or "1"


def _coefficient_times(coefficient: int, monomial: str, style: str) -> str:
    coefficient = exact_integer(coefficient)
    if not coefficient:
        return "0"
    if monomial == "1":
        return encode_integer(coefficient)
    if coefficient == 1:
        return monomial
    if coefficient == -1:
        return "-" + monomial
    return encode_integer(coefficient) + ("·" if style == "unicode" else "*") + monomial


def format_signature(signature: Signature, variables, *, style: str = "ascii") -> str:
    """Format the complete signed signature, including its original coefficient."""
    if not isinstance(signature, Signature):
        raise ValueError("Expected a Signature")
    monomial = format_monomial(signature.monomial, variables, style=style)
    basis = _symbol("e", signature.index + 1, style)
    term = basis if monomial == "1" else monomial + ("·" if style == "unicode" else "*") + basis
    return _coefficient_times(signature.coefficient, term, style)


def format_polynomial(polynomial, backend: IntegerPolynomialRing, *, style: str = "ascii") -> str:
    """Format terms in the configured polynomial order, without eval/str round-trips."""
    validate_display_options(style=style)
    backend.validate_polynomial(polynomial)
    if not polynomial:
        return "0"
    parts = []
    for exponent in sorted(polynomial, key=backend.order_key, reverse=True):
        coefficient = exact_integer(polynomial[exponent])
        term = _coefficient_times(abs(coefficient), format_monomial(exponent, backend.names, style=style), style)
        if not parts:
            parts.append(("-" if coefficient < 0 else "") + term)
        else:
            parts.append((" - " if coefficient < 0 else " + ") + term)
    return "".join(parts)


def format_ratio(ratio: Ratio, variables, *, style: str = "ascii") -> str:
    """Format the coefficient-free Laurent module monomial as a reduced fraction."""
    component, delta = ratio
    variables = tuple(variables)
    if len(delta) != len(variables):
        raise ValueError("Ratio has the wrong number of exponents")
    delta = tuple(exact_integer(x, "ratio exponent") for x in delta)
    component = exact_integer(component, "ratio component")
    if component < 0:
        raise ValueError("Ratio component must be nonnegative")
    numerator = format_monomial(tuple(max(e, 0) for e in delta), variables, style=style)
    denominator = format_monomial(tuple(max(-e, 0) for e in delta), variables, style=style)
    basis = _symbol("e", component + 1, style)
    join = "·" if style == "unicode" else "*"
    numerator = basis if numerator == "1" else numerator + join + basis
    if denominator == "1":
        return numerator
    # Parentheses make products in a denominator unambiguous.
    if sum(e < 0 for e in delta) > 1:
        denominator = "(" + denominator + ")"
    return numerator + "/" + denominator


def _wrap(text: str, width: int) -> list[str]:
    return textwrap.wrap(text, width=max(1, width), break_long_words=False,
                         break_on_hyphens=False, replace_whitespace=False) or [""]


def _table(headers: tuple[str, ...], rows: list[tuple[str, ...]], width: int) -> str:
    sizes = [max(len(headers[i]), max((len(row[i]) for row in rows), default=0))
             for i in range(len(headers))]
    # Prefer keeping identifiers intact and giving the polynomial column room.
    if len(sizes) == 3 and sum(sizes) + 4 > width:
        sizes[1] = min(sizes[1], max(len(headers[1]), width // 3))
        sizes[2] = min(sizes[2], max(len(headers[2]), width - sizes[0] - sizes[1] - 4))
    elif len(sizes) == 2 and sum(sizes) + 2 > width:
        sizes[1] = max(len(headers[1]), width - sizes[0] - 2)
    output = ["  ".join(h.ljust(n) for h, n in zip(headers, sizes)).rstrip(),
              "  ".join("-" * n for n in sizes)]
    for row in rows:
        cells = [_wrap(text, n) for text, n in zip(row, sizes)]
        for line in range(max(map(len, cells))):
            output.append("  ".join((cell[line] if line < len(cell) else "").ljust(n)
                                    for cell, n in zip(cells, sizes)).rstrip())
    return "\n".join(output)


def format_basis(result: Result, *, style: str = "ascii", width: int = 100) -> str:
    """Return aligned P and S tables. z-indices only enumerate retained signatures."""
    validate_display_options(style=style, width=width)
    names = result.backend.names
    prefix = (f"Partial result (status: {result.status}); not a certified final basis.\n\n"
              if result.status != "completed" else "")
    rows = [(_symbol("g", i + 1, style), format_signature(p.signature, names, style=style),
             format_polynomial(p.polynomial, result.backend, style=style))
            for i, p in enumerate(result.basis)]
    incremental = result.options.incremental_pot
    title = "Minimal strong ideal basis P" if incremental and result.status == 'completed' else "Nonzero polynomial pairs P"
    chunks = [prefix + f"{title} ({len(rows)})"]
    chunks.append(_table(("ID", "Signature", "Polynomial"), rows, width) if rows else "(empty)")
    if incremental:
        chunks.append("\nLabels retain original-input representations; they are not a full signature basis.\n"
                      "Stage-local syzygies were retired; no global syzygy basis is returned.")
        return "\n".join(chunks)
    srows = [(_symbol("z", i + 1, style), format_signature(s, names, style=style))
             for i, s in enumerate(result.syzygies)]
    chunks.extend(["", f"Syzygy signatures S ({len(srows)})",
                   _table(("ID", "Signature"), srows, width) if srows else "(empty)"])
    return "\n".join(chunks)


def format_vector(vector, backend: IntegerPolynomialRing, *, style: str = "ascii") -> str:
    """Return all coordinates in original input order, including zero entries."""
    validate_display_options(style=style)
    if not isinstance(vector, (list, tuple)):
        raise ValueError("A representation vector must be a list or tuple")
    return "(" + ", ".join(format_polynomial(p, backend, style=style) for p in vector) + ")"


def _u_enabled(result: Result, show_basis: bool, show_u: bool | None) -> bool:
    # Automatic output follows the existing basis switch. Explicit --show-u
    # can print vectors alone without turning on the signature tables.
    if show_u is not None:
        return show_u
    return show_basis and (getattr(result, "basis_representations", None) is not None
                           or getattr(result, "syzygy_representations", None) is not None)


def format_u(result: Result, *, style: str = "ascii", width: int = 100) -> str:
    """Format full stored (u,v) pairs and retained syzygy vectors, without solving.

    Requires both certificate arrays. No vectors are inferred from signatures
    or forest edges. Structural validation is not mathematical verification.
    """
    from .validation import result_structure_errors
    validate_display_options(style=style, width=width)
    if (getattr(result, "basis_representations", None) is None
            or getattr(result, "syzygy_representations", None) is None):
        raise ValueError(
            "Full u-vectors are unavailable: this result has no complete representation certificate. "
            "Run with --certificates (or --show-u), or Options(track_representations=True), "
            "and save a full result, not --summary-only. A saved signature/forest cannot recover them."
        )
    errors = result_structure_errors(result)
    if errors:
        raise ValueError("Cannot print u-vectors from an invalid result: " + "; ".join(errors))

    lines = []
    if result.status != "completed":
        lines.extend([f"Partial result (status: {result.status}); only stored vectors are shown.", ""])
    m = len(result.inputs)
    lines.append(f"Full representation vectors (module dimension {m})")
    lines.append("Coordinates are in the original input order; zeros are retained.")
    lines.append("Stored data only; printing does not perform verification.")

    def field(name, text):
        lead = "  " + name + " = "
        chunks = _wrap(text, width - len(lead))
        lines.append(lead + chunks[0])
        lines.extend(" " * len(lead) + line for line in chunks[1:])

    lines.extend(["", f"Nonzero polynomial pairs (u, v) in P ({len(result.basis)})"])
    if not result.basis:
        lines.append("(empty)")
    for i, (pair, vector) in enumerate(zip(result.basis, result.basis_representations), 1):
        lines.append(_symbol("g", i, style))
        field("signature", format_signature(pair.signature, result.backend.names, style=style))
        field("u", format_vector(vector, result.backend, style=style))
        field("v", format_polynomial(pair.polynomial, result.backend, style=style))
        lines.append("")
    if lines[-1] != "":
        lines.append("")
    if result.options.incremental_pot:
        lines.append("No global syzygy vectors: stage-local syzygies were retired.")
        return "\n".join(lines).rstrip()
    lines.append(f"Syzygy vectors (u, 0) in S ({len(result.syzygies)})")
    if not result.syzygies:
        lines.append("(empty)")
    for i, (signature, vector) in enumerate(zip(result.syzygies, result.syzygy_representations), 1):
        lines.append(_symbol("z", i, style))
        field("signature", format_signature(signature, result.backend.names, style=style))
        field("u", format_vector(vector, result.backend, style=style))
        field("v", "0")
        lines.append("")
    return "\n".join(lines).rstrip()


def format_forest(result: Result, *, style: str = "ascii", width: int = 100) -> str:
    """Return text trees, with only the paper's selected same-ratio edges.

    Older files with no history receive an explicit unavailable message, never
    fabricated roots/edges. Iterative traversal handles paths deeper than
    Python's recursion limit. The forest contains P only, not S.
    """
    validate_display_options(style=style, width=width)
    try:
        forest = build_forest(result)
    except ForestUnavailable as exc:
        return f"Ratio-layer forests\nUnavailable: {exc}"
    title = f"Ratio-layer forests ({len(forest.layers)} layers, {len(forest.vertices)} vertices)"
    if result.status != "completed":
        title += f" [partial execution: {result.status}]"
    if not forest.vertices:
        return title + "\n(empty: P has no nonzero polynomial pairs)"
    lines = [title, "Only same-ratio edges are shown; g-indices are insertion indices."]
    tee, end, bar = ("├── ", "└── ", "│   ") if style == "unicode" else ("+-- ", "`-- ", "|   ")
    rho = "ρ" if style == "unicode" else "rho"
    names = result.backend.names
    for number, layer in enumerate(forest.layers, 1):
        lines.extend(["", f"T{number}: {rho} = {format_ratio(layer.ratio, names, style=style)}"])
        # The stack stores the visual prefix rather than using recursive calls.
        stack = [(i, "", True, True) for i in reversed(layer.roots)]
        while stack:
            ident, prefix, is_last, is_root = stack.pop()
            node = forest.vertices[ident - 1]
            p = result.basis[ident - 1]
            origin = node.origin
            if origin.kind == "input":
                source = f"input f{origin.input_index + 1}"
            else:
                parent_text = ",".join(_symbol("g", i + 1, style) for i in origin.parents)
                source = f"from {origin.kind}({parent_text})"
            description = (f"{_symbol('g', ident, style)} [{source}]  "
                           f"({format_signature(p.signature, names, style=style)}, "
                           f"{format_polynomial(p.polynomial, result.backend, style=style)})")
            lead = prefix + ("" if is_root else end if is_last else tee)
            following = prefix + ("" if is_root else "    " if is_last else bar)
            text_lines = _wrap(description, max(20, width - len(lead)))
            lines.append(lead + text_lines[0])
            lines.extend(following + text for text in text_lines[1:])
            for position in range(len(node.children) - 1, -1, -1):
                stack.append((node.children[position], following,
                              position == len(node.children) - 1, False))
    return "\n".join(lines)


def format_polynomial_output(post, *, show_minimal_gb=False, show_reduced_gb=False,
                             show_minimal_lt=False, style="ascii", width=100) -> str:
    """Format an already computed PolynomialBasisOutput without further algebra."""
    from .postprocessing import validate_polynomial_output_flags
    validate_display_options(style=style, width=width)
    validate_polynomial_output_flags(show_minimal_gb, show_reduced_gb, show_minimal_lt)
    if not (show_minimal_gb or show_reduced_gb or show_minimal_lt):
        return ""
    if show_reduced_gb and post.reduced_basis is None:
        raise ValueError("The reduced polynomial basis has not been computed")
    b = post.backend
    sections = ["Polynomial ideal basis post-processing over ZZ (" + b.order_name + ")",
                "Projection of P only; original signatures, vectors and paths are unchanged.",
                "Assumes the polynomial projection P0 is a strong basis; this output is not a new certificate."]
    def table(title, symbol, values, header):
        rows = [(_symbol(symbol, i + 1, style), format_polynomial(f, b, style=style))
                for i, f in enumerate(values)]
        return (title + f" ({len(values)})\n" +
                (_table(("ID", header), rows, width) if rows else "(empty: zero ideal)"))
    if show_minimal_gb:
        sections.append(table("Minimal strong Groebner basis", "m", post.minimal_basis, "Polynomial"))
    if show_reduced_gb:
        sections.append(table("Reduced strong Groebner basis", "r", post.reduced_basis, "Polynomial"))
        sections.append("Convention: positive leading coefficients; -b/2 < remainder <= b/2.")
    if show_minimal_lt:
        sections.append(table("Leading terms of the minimal strong Groebner basis", "lt",
                              post.minimal_leading_terms, "Leading term (with coefficient)"))
    t = post.statistics
    sections.append(f"Post-processing {t['total_seconds']:.6f}s | minimal {t['minimal_seconds']:.6f}s | "
                    f"reduced {t['reduced_seconds']:.6f}s | reduction steps={t['reduction_steps']} "
                    "(excluded from Compute)")
    return "\n\n".join(sections)


def print_polynomial_output(post, *, show_minimal_gb=False, show_reduced_gb=False,
                            show_minimal_lt=False, style="ascii", width=100,
                            file=None) -> None:
    text = format_polynomial_output(post, show_minimal_gb=show_minimal_gb,
                                    show_reduced_gb=show_reduced_gb,
                                    show_minimal_lt=show_minimal_lt, style=style, width=width)
    if text:
        print(text, file=sys.stdout if file is None else file)


def format_result(result: Result, *, show_basis: bool = True, show_forest: bool = False,
                  style: str = "ascii", width: int = 100,
                  show_u: bool | None = None, show_minimal_gb: bool = False,
                  show_reduced_gb: bool = False, show_minimal_lt: bool = False) -> str:
    """Format selected sections; optional polynomial views compute detached copies."""
    from .postprocessing import postprocess_basis, validate_polynomial_output_flags
    validate_polynomial_output_flags(show_minimal_gb, show_reduced_gb, show_minimal_lt)
    validate_display_options(show_basis, show_forest, style, width, show_u=show_u)
    # Validate requested vectors before returning any partial display text.
    u_text = format_u(result, style=style, width=width) if _u_enabled(result, show_basis, show_u) else None
    sections = []
    if show_basis:
        sections.append(format_basis(result, style=style, width=width))
    if u_text is not None:
        sections.append(u_text)
    if show_forest:
        sections.append(format_forest(result, style=style, width=width))
    if show_minimal_gb or show_reduced_gb or show_minimal_lt:
        if result.status != "completed":
            sections.append("Polynomial post-processing skipped: source computation is not completed.")
        else:
            post = postprocess_basis(result, reduced=show_reduced_gb)
            sections.append(format_polynomial_output(post, show_minimal_gb=show_minimal_gb,
                            show_reduced_gb=show_reduced_gb, show_minimal_lt=show_minimal_lt,
                            style=style, width=width))
    return "\n\n".join(sections)


def print_result(result: Result, *, show_basis: bool = True, show_forest: bool = False,
                 style: str = "ascii", width: int = 100, file: TextIO | None = None,
                 show_u: bool | None = None, show_minimal_gb: bool = False,
                 show_reduced_gb: bool = False, show_minimal_lt: bool = False) -> None:
    """Print P/S, automatically adding complete stored vectors when available."""
    text = format_result(result, show_basis=show_basis, show_forest=show_forest,
                         style=style, width=width, show_u=show_u,
                         show_minimal_gb=show_minimal_gb, show_reduced_gb=show_reduced_gb,
                         show_minimal_lt=show_minimal_lt)
    if text:
        print(text, file=sys.stdout if file is None else file)


def print_basis(result: Result, *, style: str = "ascii", width: int = 100,
                file: TextIO | None = None) -> None:
    print_result(result, style=style, width=width, file=file, show_u=False)


def print_forest(result: Result, *, style: str = "ascii", width: int = 100,
                 file: TextIO | None = None) -> None:
    print_result(result, show_basis=False, show_forest=True, style=style, width=width, file=file)


def print_u(result: Result, *, style: str = "ascii", width: int = 100,
            file: TextIO | None = None) -> None:
    """Print only complete stored vectors, without rerunning the solver."""
    print_result(result, show_basis=False, show_u=True, style=style, width=width, file=file)
