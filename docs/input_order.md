# Generator order

`--reverse-generators` (alias `--reversed`) reverses the supplied generator list
once before order construction and solver initialization. It is off by default.
Negative forms are `--no-reverse-generators` and `--no-reversed`. Repeated positive
switches do not toggle; when positive and negative forms are mixed, the last wins.

Reversal is input preparation rather than an `Options` field. It supports both
ordinary and incremental POT computation. Variables, terms within polynomials,
integer coefficients, and reducer/queue policies are unchanged. Reversal does
not imply a speedup or select a module order.

## Commands

```powershell
python -m gvwzz run --case cyclic6 --reverse-generators --module-order g2 --no-show-basis --output results/cyclic6_reverse.json
python -m gvwzz run --input examples/paper.json --reverse-generators --incremental-pot --no-show-basis --output results/paper_reverse.json
python -m gvwzz export-case eco3 --reverse-generators --output inputs/eco3_reverse.json
python -m gvwzz benchmark --cases eco3,noon2,reimer2 --reverse-generators --module-orders pot --repeat 1 --output results/reverse_benchmark
```

Use the original case name; no `_reversed` suffix is needed. Benchmark reverses
each selected input, not the order of the case list. Every repeat starts with
the same effective order. An exported reversed input already contains that list;
loading it with another positive flag reverses the actual list again. A negative
flag prevents a new reversal rather than undoing saved input preparation.

Zero and duplicate generators remain in the list. Input files are not rewritten.
No primitive normalization or content division is performed.

## Signatures and metadata

For m inputs, new position i corresponds to source position m+1-i. All module
positions, vectors, witnesses, and paths refer to the effective ordered inputs
saved in the result. Explicit `--positions` and signature blocks refer to the
new list and are not remapped. Incremental mode still requires natural POT.

CLI results and exports contain `input_order` with `direction`, a 1-based
`effective_to_source` permutation, `reversals`, and `source_metadata`. The
fingerprint is recomputed from the effective ordered equations. Benchmark
JSON/CSV carries the same order annotation. Metadata describes input preparation;
it is not a membership certificate. Older results use their saved input sequence.

## Python

```python
from gvwzz import Options, compute, reverse_generators
from gvwzz.cases import get_case

case = get_case("eco3", reverse_generators=True)
result = compute(case["polynomials"], case["variables"],
                 options=Options(incremental_pot=True), show_basis=False)

custom = {"name": "custom", "variables": ["x", "y"],
          "polynomials": ["2*x", "3*y"]}
reversed_case = reverse_generators(custom)
```

The helper returns a detached dictionary without modifying the source or running
the solver. `compute` receives the effective list; it has no reversal keyword.
The CLI adds case-level provenance and order annotations to its output.
