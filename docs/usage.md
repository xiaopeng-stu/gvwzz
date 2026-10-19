# Usage

Run these commands from the release directory. See the [README](../README.md)
for installation and each subcommand's `--help` for its complete interface.

## Inputs and orders

```powershell
python -m gvwzz list
python -m gvwzz export-case eco3
python -m gvwzz export-case henrion5 --output inputs/henrion5.json
python -m gvwzz run --input examples/paper.json --no-show-basis --output results/paper.json
python -m gvwzz run --vars x,y --poly "4*x^2" --poly "9*y^4" --poly "6*x*y^2+y" --poly-order lex --module-order pot --no-show-basis --output results/custom.json
```

`export-case` refuses to overwrite an existing path. Exports contain exact
equations, provenance, and an input fingerprint. All inputs are over `ZZ`.
Polynomial orders are `lex`, `grlex`, and `grevlex`; the default is
`grevlex / gvw-g2` when neither the input nor command specifies orders.

Plain `katsuraN` uses N variables and linear-first input order;
`katsuraN_qfirst` uses N+1 variables and quadratic-first order. Noon inputs retain
their nonhomogeneous integer coefficients. Eco's final equation is
`x1 + ... + x(n-1) + 1`. Compare equations rather than case names alone.

`--reverse-generators` (alias `--reversed`) reverses the supplied generator list.
Signatures and certificates refer to the effective order. See
[generator order](input_order.md). `--incremental-pot` selects a separate staged
polynomial-ideal workflow; see [incremental mode](incremental.md).

## Solver settings

Defaults are the heap workspace, sparse reducer selection, `small-coeff` queue
ties, Euclidean reduction, and full J/G tail reduction. G has priority at an equal
signature monomial; `--j-first` gives J priority.

| Option | Effect |
| --- | --- |
| `--progress 30` | Print progress at safe boundaries, approximately every 30 seconds |
| `--no-g-tail-reduce` | Skip G tails while retaining J tails |
| `--no-tail-reduce` | Disable both J and G tails |
| `--timeout 120` | Set a cooperative solver time budget |
| `--max-candidates 250000` | Bound generated J/G candidates |
| `--max-steps 500000` | Bound combined head and tail steps |
| `--limits` | Select predefined finite solver budgets |
| `--no-limits` | Disable solver budgets |

Budgets are off by default. Progress and time budgets are cooperative, not hard
real-time deadlines. A `limit` or `interrupted` result is partial; saved JSON is
not a restart checkpoint. Disabling tails changes the work and can increase later
reductions. It is not required for native acceleration.

Native bulk loops require eligible orders, features, and metadata. Explicit
budgets, tail-step caps, debug checking, and vector/path recording can select
other exact paths. Progress alone does not disable bulk reduction. Checked
exponent encodings select fallback before narrowing; coefficients remain
arbitrary precision.

## Display, output, and timing

```powershell
python -m gvwzz run --case cyclic7 --poly-order grevlex --module-order g2 --require-native --show-minimal-lt --no-show-basis --no-show-u --no-show-forest --progress 30 --output results/cyclic7.json
python -m gvwzz show results/cyclic7.json --show-minimal-gb --show-minimal-lt --no-show-basis --no-show-u
```

| Switch | View, off by default |
| --- | --- |
| `--show-minimal-gb` | Minimal strong polynomial basis |
| `--show-reduced-gb` | Reduced strong polynomial basis with balanced integer remainders |
| `--show-minimal-lt` | Minimal leading terms, including integer coefficients |

Each has a matching `--no-show-...` form. Views use polynomial projection P and
retain the original P/S, vectors, paths, and solver counters. Both monomial and
coefficient divisibility determine minimality. Reduced output has positive
leading coefficients and remainders `-b/2 < r <= b/2` for positive `b`; content is
not divided out. Views assume P is a strong basis and are not new certificates.
Partial results are not labelled as completed polynomial bases.

`--no-show-basis`, `--no-show-u`, and `--no-show-forest` control console display,
not full JSON contents. A requested `--show-minimal-lt` view saves the minimal
polynomial basis and leading terms unless `--summary-only` is set. `run` saves
views under `polynomial_postprocessing`; `show` computes them without rewriting
the result.

`--summary-only` saves statistics without P/S, certificates, or polynomial basis
arrays. It does not alter solver arithmetic or avoid requested post-processing.
Such files cannot be loaded by `show`, verification, or reconstruction.

`Compute` is solver time. Its exclusive breakdown is a partition; do not add
aggregate fields such as `tail_reduction` or `syzygy` to their components. Setup,
verification, cleanup, JSON output, post-processing, and display are separate.
`[output]` messages identify later work; `After Compute` shows its phases.
`Command elapsed before exit` excludes subsequent interpreter shutdown.

Full JSON is streamed and saved atomically before post-processing. A second
atomic save adds requested views, reusing validated source records where
possible. If post-processing is interrupted, the completed source result remains
available. Large polynomial records can still take substantial time to validate
or write.

## Native diagnostics

```powershell
python -m gvwzz doctor
python -m gvwzz doctor --json
python -m gvwzz doctor --require-native
```

Diagnostics show the interpreter, imported package path, extensions, entry
points, import errors, and arithmetic. `--require-native` also applies to `run`
and `benchmark`. It requires entry points, not universal fast-path eligibility
or a speedup.

After solving, usage reports distinguish native parent rows/pairs, Python
dispatch, reduction workspaces, compact recipes, queue activity, and cover
activity. The `wide` parent subset includes more than eight variables; `dynamic`
is its more-than-16-variable subset. Bulk and wide reduction counts are subsets
of total native workspaces. Do not add overlapping counts. Heap pops include
stale records; logical J/G pops count active candidates. Packed-cover fallback
means the complete exact criterion handled the query. Unrecorded values appear
as `not recorded` or JSON `null`.

To force the exact Python fallback in PowerShell:

```powershell
$env:GVWZZ_NO_NATIVE = '1'
python -m gvwzz run --case katsura3 --no-show-basis --output results/katsura3_python.json
Remove-Item Env:GVWZZ_NO_NATIVE
```

To build without GMP, set `GVWZZ_NO_GMP=1` before `python build_native.py`. Remove
it and rebuild to restore GMP. Start a fresh Python process after changing these
environment settings.

## Certificates and verification

Online vectors:

```powershell
python -m gvwzz run --input examples/paper.json --certificates --j-first --no-show-basis --no-show-u --output results/vectors.json
python -m gvwzz verify results/vectors.json --require-certificates --verify-mode static
```

Generation paths without online vectors:

```powershell
python -m gvwzz run --input examples/paper.json --record-paths --j-first --no-show-basis --no-show-u --output results/paths.json
python -m gvwzz verify-paths results/paths.json
python -m gvwzz reconstruct results/paths.json --verify --show-syzygies --progress 0 --output results/syzygies.json
```

`--certificates` and `--record-paths` are mutually exclusive. Paths also cannot
be combined with `--show-u` or `--summary-only`. `completed` means the solver
exhausted its work; independent membership needs vectors or paths. Syzygy-vector
checks establish identities and signatures, not completeness by themselves.
Verification budgets are separate from solver budgets. Incremental mode has a
different [certification scope](incremental.md).

## Repeatable benchmarks

```powershell
python -m gvwzz benchmark --suite classic-small --poly-orders grevlex --module-orders pot --j-first --repeat 3 --timeout 60 --max-candidates 250000 --output results/classic_small
```

`classic-small` includes Cyclic-3, Katsura-3, Eco-3, Noon-2, and Reimer-2. Benchmark
JSON/CSV files are summaries, not full replayable results. Compare completed runs
with matching equations, input order, polynomial/module orders, reduction
settings, interpreter, and native backend. Use complete command time or matching
measured phases.
