# GVWZZ v1

GVWZZ computes coefficient-aware GVW-type signature bases over integer polynomial
rings (`ZZ`). It uses one CPU thread and arbitrary-precision exact integers.
Optional C++/GMP extensions accelerate reduction, pair generation, indices,
queues, and cover queries. Full J/G tail reduction is enabled by default.

Repository: [xiaopeng-stu/gvwzz](https://github.com/xiaopeng-stu/gvwzz).
This standalone CPU release is based on `0.10.0.post31`. The Python package
version is `1.0.0`; the release name is `v1`. No earlier package or updater is
needed. See the [release notes](docs/release-notes.md) for the review scope.

## Contents

1. [Scope and terminology](#1-scope-and-terminology)
2. [Installation](#2-installation)
3. [First runs and command overview](#3-first-runs-and-command-overview)
4. [Input systems and orders](#4-input-systems-and-orders)
5. [Solver and performance parameters](#5-solver-and-performance-parameters)
6. [Limits and progress](#6-limits-and-progress)
7. [Output and polynomial basis views](#7-output-and-polynomial-basis-views)
8. [Independent verification](#8-independent-verification)
9. [Generation paths and syzygy reconstruction](#9-generation-paths-and-syzygy-reconstruction)
10. [Incremental POT mode](#10-incremental-pot-mode)
11. [Benchmarks and random inputs](#11-benchmarks-and-random-inputs)
12. [Timing reports and diagnostics](#12-timing-reports-and-diagnostics)
13. [Python API](#13-python-api)
14. [Tests, source layout, and contributing](#14-tests-source-layout-and-contributing)
15. [Troubleshooting](#15-troubleshooting)

## 1. Scope and terminology

All inputs are interpreted over `ZZ`. Integer coefficients are exact and have
arbitrary precision. No nonunit content division or rational normalization is
performed. This v1 package does not provide a rational-field solver, finite-field
solver, or GPU backend.

| Term | Meaning in this program |
| --- | --- |
| `P` | Nonzero labeled polynomials retained by the ordinary signature solver. |
| `S` | Retained coefficient-aware syzygy signatures; explicit vectors are optional. |
| J-candidate / J-pair | A candidate subject to signature filtering, cover, and head reduction. |
| G-candidate / G-polynomial | A candidate needed for integer coefficient handling; exact top-reducible candidates are discarded. |
| Head reduction | Reduction of the current leading term under the solver's signature conditions. |
| Tail reduction | Optional reduction below the leading term, preserving the leading term and signature monomial. |
| Minimal strong basis | A polynomial basis with no leading term divisible by another retained leading term, using both coefficient and monomial divisibility. |
| Reduced strong basis view | Additional polynomial post-processing with positive leading coefficients and balanced integer remainders. |

The polynomial projection of `P`, its minimal view, and its reduced view have
different sizes and purposes. Polynomial views do not replace the original
signature records or certify them independently. A completed computation and
an independently verified computation are reported separately.

There is no fixed user-facing variable-count limit beyond available resources.
Native monomial encodings have checked bounds; valid configurations outside a
fast path's eligibility use exact fallback paths. This does not limit coefficient
precision. Invalid order combinations are rejected rather than silently changed.

## 2. Installation

Use Python 3.10 or later. The repository is
[github.com/xiaopeng-stu/gvwzz](https://github.com/xiaopeng-stu/gvwzz).

Follow the section for your operating system. The clone commands download the
repository's default branch into a new directory named `gvwzz`.
Run each command in order, and continue only after it succeeds.

**Using the source ZIP instead?** Install the same system prerequisites, extract
the archive, and open a terminal in the directory containing this README and
`build_native.py` (normally `gvwzz-v1`). Skip the
`git clone` and `cd gvwzz` lines, then follow the remaining commands.

### 2.1 Windows (PowerShell)

Install these prerequisites:

1. [Python](https://www.python.org/downloads/windows/), version 3.10 or later.
   Make sure `python` is available in a new PowerShell window.
2. [Git for Windows](https://git-scm.com/download/win), with Git available from
   the command line.
3. [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/),
   with the **Desktop development with C++** workload.

Open a new PowerShell window in the directory where you want to keep the project:

```powershell
python --version
git --version
git clone https://github.com/xiaopeng-stu/gvwzz.git
cd gvwzz
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-native.txt
.\.venv\Scripts\python.exe build_native.py
.\.venv\Scripts\python.exe -m gvwzz doctor --require-native
```

Run Cyclic-7:

```powershell
.\.venv\Scripts\python.exe -m gvwzz run --case cyclic7 --poly-order grevlex --module-order g2 --require-native --show-minimal-lt --no-show-basis --no-show-u --no-show-forest --progress 30 --output results/cyclic7.json
```

These commands use the virtual environment directly, so PowerShell activation
is not required. In all later examples, replace `python` with
`.\.venv\Scripts\python.exe` on Windows. When opening a new terminal, return
to the source directory before running these commands.

### 2.2 Linux (Ubuntu/Debian)

Install Git, Python, the C++ compiler, Python development headers, and virtual
environment support:

```bash
sudo apt update
sudo apt install -y git python3 python3-dev python3-venv build-essential
python3 --version
git --version
```

Check that `python3` is version 3.10 or later. For other Linux distributions,
install the equivalent packages with the distribution's package manager. If you
use a separately installed Python, its development headers must match that
interpreter.

Clone, create the environment, and build:

```bash
git clone https://github.com/xiaopeng-stu/gvwzz.git
cd gvwzz
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-native.txt
python build_native.py
python -m gvwzz doctor --require-native
```

Run Cyclic-7:

```bash
python -m gvwzz run --case cyclic7 \
  --poly-order grevlex --module-order g2 --require-native \
  --show-minimal-lt --no-show-basis --no-show-u --no-show-forest \
  --progress 30 --output results/cyclic7.json
```

In a new terminal, return to the source directory and run
`source .venv/bin/activate` before using the program. Run `deactivate` when
finished. Install Python packages inside the virtual environment; use `sudo`
only for the system packages above.

This release was checked on Linux x86-64 with Python 3.12.14. See the
[validation report](docs/audit-v1.md) for the tested scope.

### 2.3 macOS (Terminal)

Install [Python for macOS](https://www.python.org/downloads/macos/), version
3.10 or later. Install Apple's Command Line Tools, which provide the C++
compiler and Git:

```bash
xcode-select --install
```

Complete the installation dialog before continuing. If the tools are already
installed, proceed with the following commands:

```bash
python3 --version
git --version
git clone https://github.com/xiaopeng-stu/gvwzz.git
cd gvwzz
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-native.txt
python build_native.py
python -m gvwzz doctor --require-native
python -m gvwzz run --case cyclic7 --poly-order grevlex --module-order g2 --require-native --show-minimal-lt --no-show-basis --no-show-u --no-show-forest --progress 30 --output results/cyclic7.json
```

In a new terminal, return to the source directory and run
`source .venv/bin/activate`. Ubuntu/Debian `apt` commands do not apply to
macOS. Local v1 validation was performed on Linux; macOS checks are configured
in GitHub Actions.

### 2.4 Select the v1 release

The default clone commands above do not require a release tag. Once the
maintainer has published the `v1` tag, select that fixed release from the
source directory **before installing dependencies and building**:

```text
git fetch --tags
git checkout v1
```

A detached-HEAD notice is normal when checking out a release tag. If you change
the checked-out version after a build, reinstall its requirements and run
`build_native.py` again with the same virtual-environment Python.


### 2.5 Build diagnostics and installation modes

The diagnostics should include:

```text
Native extensions available: reduction=yes | pairs=yes | indices=yes
Native queue/cover entry points: candidate_heap=yes | packed_cover=yes | queue_drain=yes
Native reduction arithmetic: gmp (arbitrary-precision integers)
```

`--require-native` checks entry-point availability. The arithmetic line separately
confirms GMP; the report after a run shows which native paths actually ran.
Rebuild in this directory after changing Python, gmpy2, or GVWZZ. Native binaries
from another Python version, operating system, or release are not interchangeable.

Generated C++ is included alongside Cython
sources and headers so all five extensions can be rebuilt from this package.

For exact Python-only operation, install `requirements.txt` and omit
`--require-native`. The fallback is slower and needs no compiler.
`python -m pip install .` installs the Python package and `gvwzz` command;
extensions are built separately with `python build_native.py`. For the native
source workflow, run commands from this directory. Keep the extracted source
and its built extensions together. To use this built source tree and the
`gvwzz` command from another directory, run `python -m pip install -e .` from the
source root. A regular wheel installation contains the Python fallback.
The release does not include GPU code or
precompiled platform-specific binaries.

## 3. First runs and command overview

Run examples from the source directory. On Linux/macOS, activate `.venv` first.
On Windows, replace `python` with `.\.venv\Scripts\python.exe`. The one-line
commands below work in both PowerShell and a POSIX shell with that substitution.

### 3.1 Compute and display minimal leading terms

```text
python -m gvwzz run --case cyclic7 --poly-order grevlex --module-order g2 --require-native --show-minimal-lt --no-show-basis --no-show-u --no-show-forest --progress 30 --output results/cyclic7.json
```

The command saves full source records to `results/cyclic7.json` and a statistics
row to `results/cyclic7.csv`. Output directories are created automatically.
Check `Status: completed`. Minimal-leading-term output runs after the solver and
can take additional time.

For a short installation smoke test, use `--case cyclic3`. Other family examples
include `katsura8`, `eco10`, `noon7`, and `reimer5`; changing the case does not
impose a time limit.

### 3.2 Solve now and inspect later

Save a full result without polynomial post-processing or large console tables:

```text
python -m gvwzz run --case cyclic7 --require-native --no-show-basis --no-show-u --progress 30 --output results/cyclic7-source.json
python -m gvwzz show results/cyclic7-source.json --show-minimal-gb --show-minimal-lt --no-show-basis --no-show-u
```

`show` does not rerun GVW, but requested polynomial post-processing still costs
time. It does not rewrite the saved file.

### 3.3 Available commands

| Command | Purpose |
| --- | --- |
| `list` | List built-in systems and descriptions. |
| `doctor` | Inspect native extensions for the current Python interpreter. |
| `export-case NAME` | Print or save exact input equations and provenance without solving. |
| `run` | Compute one built-in, JSON, or inline system. |
| `show FILE` | Display a full saved result and optional polynomial views. |
| `verify FILE` | Independently check a full saved result. |
| `reconstruct FILE` | Replay recorded paths to expand selected final syzygy vectors. |
| `verify-paths FILE` | Check recorded identities and, by default, final criteria. |
| `verify-syzygies FILE` | Check explicit syzygy vectors in reconstructed JSON/JSONL. |
| `benchmark` | Run selected cases, orders, and repetitions; save JSON/CSV statistics. |
| `random` | Write a reproducible random input file. |

Use `python -m gvwzz --version` for the package version, and `-h` or `--help`
at the top level or after a subcommand for its accepted arguments:

```text
python -m gvwzz run --help
python -m gvwzz benchmark --help
```

## 4. Input systems and orders

### 4.1 Input sources

Choose one input route:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--case NAME` | None | A built-in case, such as `cyclic7`. |
| `--input FILE` | None | A UTF-8 JSON input file. Mutually exclusive with `--case`. |
| `--vars x,y` and repeated `--poly EXPR` | None | Inline variable names and generator expressions; supply both and omit `--case`/`--input`. |
| `--no-print-ideal` | Off | Suppress the initial equations; it does not suppress the summary. |

Inline example:

```text
python -m gvwzz run --vars x,y --poly "4*x^2" --poly "9*y^4" --poly "6*x*y^2+y" --poly-order lex --module-order pot --show-minimal-lt --no-show-basis --output results/custom.json
```

The same input can be stored as JSON:

```json
{
  "name": "custom",
  "variables": ["x", "y"],
  "polynomials": ["4*x^2", "9*y^4", "6*x*y^2+y"],
  "polynomial_order": "lex",
  "module_order": "pot"
}
```

`variables` and `polynomials` are required lists. `name` and the order fields
are optional. Expressions use declared variables, integer coefficients,
multiplication `*`, and nonnegative integer powers (`^` or `**`). Keep the
variable sequence fixed when comparing results. An optional `positions` list in
input JSON is a 1-based position permutation as described below.

### 4.2 Built-in naming and provenance

```text
python -m gvwzz list
python -m gvwzz export-case cyclic7
python -m gvwzz export-case cyclic7 --output inputs/cyclic7.json
python -m gvwzz run --input inputs/cyclic7.json --no-show-basis --output results/from-file.json
```

`export-case` prints JSON by default; `--output FILE` saves to a new file and
refuses to overwrite an existing one. It also accepts the generator reversal
switches below.

| Family | Convention to check before comparing implementations |
| --- | --- |
| `cyclicN` | Cyclic equations with `N` variables. Inspect exported variable/generator order. |
| `katsuraN` | `N` variables, linear equation first. `katsuraN_qfirst` uses the separate `N+1`-variable, quadratic-first convention. |
| `ecoN` | Integer equations; the final equation is `x1 + ... + x(N-1) + 1`. |
| `noonN` | Nonhomogeneous integer equations with coefficients `10`, `-11`, and `10`; do not divide all equations by 10 over `ZZ`. |
| `reimerN` | Integer equations; use `export-case` to confirm the precise instance. |

Other names and provenance variants are listed by `list`. A matching family
name alone does not establish identical benchmark inputs.

### 4.3 Polynomial and module orders

| Parameter | Default when not supplied by input | Choices and effect |
| --- | --- | --- |
| `--poly-order ORDER` | `grevlex` | `lex`, `grlex`, `grevlex`; aliases `deglex` = `grlex`, `degrevlex` = `grevlex`. |
| `--module-order ORDER` | `g2` (`gvw-g2`) | `pot`, `top`, `g1`, `g2`, `schreyer`, `block`; `gvw-g1` and `gvw-g2` are explicit aliases. |
| `--positions LIST` | Natural order `1,2,...,m` | Complete permutation of the `m` input indices, listed from lowest to highest position. |
| `--signature-blocks LIST` | None | Positive block sizes summing to `m`, for example `2,1` for three inputs. Selects `block` if no module order is explicitly given. |

Explicit CLI orders override input order metadata. Without either, the default
is `grevlex / gvw-g2`. Incremental mode instead selects POT. Supplying signature
blocks with an explicit non-block module order is an error.

For a module monomial `t*e_i`, the module orders compare as follows. Position
ranks come from `--positions`:

| Module order | Comparison |
| --- | --- |
| `pot` | Position first, then `t`. |
| `top` | `t` first, then position. |
| `g1` | `degree(t) + degree(f_i)` first, then `t`, then position. |
| `g2` | `t*LM(f_i)` first, then position. |
| `schreyer` | `t*LM(f_i)` first, with the reversed position tie. |
| `block` | POT between consecutive blocks, TOP within each block. All inputs are still inserted together. |

`g1` is incompatible with multivariate `lex`. The `packed` reduction kernel
also rejects `lex`. Use `heap` with `lex`. Zero inputs use weight monomial `1`
and degree `0`. Coefficients remain part of integer signatures even though the
module order compares their monomial-position parts.

### 4.4 Generator order

`--reverse-generators` (alias `--reversed`) reverses the supplied generator list
once. It is off by default; `--no-reverse-generators` and `--no-reversed` disable
this preparation. With conflicting positive/negative switches, the last wins.

It changes input positions, not variable order or queue policy. Explicit
`--positions` and blocks refer to the resulting input sequence. Reversing an
already reversed exported input reverses its actual list again. See
[generator order](docs/input_order.md) for saved mappings and fingerprints.

## 5. Solver and performance parameters

The parameters in this section apply to both `run` and `benchmark`. Defaults
are a starting point for performance comparisons. Changing a reduction or
scheduling heuristic may change the number of generated pairs, intermediate
polynomials, and coefficient sizes; fewer steps do not always mean less time.

### 5.1 Reduction kernels and reducer choice

| Parameter | Default | Values and effect |
| --- | --- | --- |
| `--require-native` | Off | Fail if required native entry points are unavailable. Does not force every operation onto a native fast path or by itself prove GMP is active. |
| `--reduction-kernel` | `heap` | `heap`: heap-based sparse workspace; `copy`: polynomial-copy implementation for comparison; `packed`: packed monomial workspace for graded orders only. |
| `--reducer-policy` | `regular-first` | Prefer regular reducers, prefer singular reducers (`singular-first`), or use the first eligible exact reducer (`first`), considering Euclidean alternatives only if no exact reducer is eligible in that mode. This is a preference, not a switch disabling the other reduction kind. |
| `--reducer-selection` | `sparse` | Orders eligible reducers using one of the strategies in the next table. |
| `--euclidean` / `--no-euclidean` | On | Enable/disable signature-safe Euclidean coefficient steps during head and tail reduction, in addition to exact reductions. |
| `--euclidean-priority` | `signature-first` | `signature-first`: consider exact and Euclidean candidates within the requested signature-category priority; `exact-first`: try exact reduction before Euclidean steps. |
| `--debug` | Off | Enable additional internal checks. This can slow computation and change fast-path eligibility. |

Reducer selection does not bypass divisibility or signature admissibility checks:

| Selection | Preferred reducer |
| --- | --- |
| `oldest` | Earliest inserted. |
| `newest` | Most recently inserted. |
| `sparse` | Fewer terms, then smaller leading-coefficient bit length, then older insertion. |
| `height` | Smaller relative coefficient height, then fewer terms, smaller leading-coefficient bit length, then older insertion. |
| `balanced` | Smaller `term count * (1 + relative height)`, then relative height, term count, and older insertion. |
| `unit` | Unit leading coefficient first, then fewer terms, smaller maximum coefficient bit length, then older insertion. |

Relative coefficient height is the maximum coefficient bit length minus the
leading-coefficient bit length. These rankings are implementation heuristics.
`--reducer-policy` chooses
the admissible reduction category; `--reducer-selection` prioritizes reducers
within that search. Different combinations can produce different computation
traces while retaining exact arithmetic. A retained J-pair starts with an exact
regular reduction regardless of the selected policy; the policy governs the
subsequent mixed-reduction search.

### 5.2 Tail reduction

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--tail-reduce` / `--no-tail-reduce` | On | Master switch for J-result and G-candidate tail reduction. |
| `--g-tail-reduce` / `--no-g-tail-reduce` | On | Additional G-tail switch; effective only when the master switch is on. |
| `--tail-policy` | `all` | `all`: attempt tails for every eligible new polynomial. `selective`: skip optional tail work when an existing `P` leading term divides the new leading term, including coefficient divisibility. |
| `--tail-step-cap N` | `0` | At most `N` tail steps per polynomial tail-reduction call. `0` means unlimited. Does not cap head steps or the total run. |

To retain J tails but bypass G tails, use `--no-g-tail-reduce`. To bypass both,
use `--no-tail-reduce`. These switches do not remove J head reduction or G top
tests. A selective/capped tail can leave more terms for later work; it is not a
guaranteed speedup. The selective skip test ignores signatures only when deciding
whether to skip this optional work; actual tail steps remain signature-safe.

### 5.3 Pair generation, queue, and cover

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--cover-mode` | `indexed` | `indexed`: indexed cover queries; `naive`: direct reference search. Cover applies to J-candidates, not G-candidates. |
| `--no-j-dedup` | Off | Disable exact J-candidate deduplication. Deduplication is normally enabled. |
| `--queue-tie` | `small-coeff` | Tie strategy after signature monomial and J/G priority: `fifo`, `small-coeff`, or `small-lm`. |
| `--j-first` | Off | Give J priority instead of G at an equal signature monomial. |
| `--lazy-candidates` / `--no-lazy-candidates` | On | Retain recipes and defer polynomial materialization until needed. Disabling creates candidates eagerly. |
| `--no-g-precheck` | Off | Disable the general generation-time G top test. The comparable-leading-coefficient discard remains active. |
| `--no-j-generation-filter` | Off | Disable the generation-time syzygy filter for J-candidates. |
| `--g-policy discard` | `discard` | Compatibility option; `discard` is the only supported policy. Exact top-reducible G-candidates are discarded. |

The queue compares signature monomial first, J/G priority second, the selected
tie heuristic next, and insertion serial as a final tie. `fifo` uses insertion
order at that tie; `small-coeff` favors smaller absolute signature coefficients
for J and smaller absolute polynomial leading coefficients for G, then smaller
leading monomials. `small-lm` compares the leading monomial before that
coefficient. These settings do not
replace the primary signature order. `--eager-candidates` is a legacy alias for
`--no-lazy-candidates`.

### 5.4 Safe performance experiments

Change one option at a time and keep the same case, variable/generator orders,
module order, verification settings, and output requests. For example:

```text
python -m gvwzz benchmark --cases cyclic6 --module-orders g2 --repeat 3 --require-native --output results/default-tails
python -m gvwzz benchmark --cases cyclic6 --module-orders g2 --repeat 3 --require-native --no-g-tail-reduce --output results/no-g-tails
```

Certificate vectors, path recording, debug mode, solver budgets, and tail caps
can change native bulk-kernel eligibility. Progress reporting alone does not
disable those kernels. Inspect actual kernel counters after each run; entry-point
availability at startup is not an execution trace.

## 6. Limits and progress

### 6.1 Solver budgets

By default, all solver abort budgets below are **off**. An explicit positive
value enables only that budget. At the CLI, `0` disables a budget; negative
values are invalid. These limits do not change arbitrary-precision arithmetic.

| Parameter | Unit / meaning | Default | Unspecified value with `--limits` |
| --- | --- | --- | --- |
| `--timeout N` | Solver wall-clock seconds | Off | `30` |
| `--max-basis N` | Retained nonzero basis size | Off | `5000` |
| `--max-candidates N` | Generated J + G candidates, including early G discards | Off | `250000` |
| `--max-steps N` | Combined head and tail reduction steps | Off | `500000` |
| `--max-terms N` | Terms in one polynomial | Off | `100000` |
| `--max-coefficient-bits N` | Integer coefficient bit length | Off | Off |

`--limits` restores the finite defaults in the last column; explicit values
still override them. `--no-limits` (alias `--no-limit`) disables **all** solver
budgets, including explicitly supplied `--timeout` and `--max-*` values. Avoid
combining it with budgets you intend to enforce. It does not disable verifier
limits or `--tail-step-cap`.

```text
python -m gvwzz run --case noon9 --timeout 300 --max-basis 10000 --no-show-basis --progress 30 --output results/noon9-limited.json
```

Limits are checked at safe execution boundaries, so they are not hard operating
system time or memory limits. A long native operation may run past a requested
deadline before control returns. A `limit` or `interrupted` result is partial;
do not interpret it as a completed basis. Saved JSON is an output record, not
a checkpoint from which the queue can be resumed.

### 6.2 Progress

`--progress SECONDS` defaults to `0` (off). Use, for example, `--progress 30`
for periodic progress at safe boundaries. Values must be finite and nonnegative.
Progress includes the current phase, `P`/`S`/queue sizes, generated candidates,
head/tail steps, and the current signature when available. It also enables
progress messages for supported output stages. It does not set a time limit.

## 7. Output and polynomial basis views

### 7.1 Files

| `run` parameter | Default | Effect |
| --- | --- | --- |
| `--output FILE` | `results/run.json` | Full JSON result and a CSV summary at the same path with the `.csv` suffix. Use a JSON filename, not a `.csv` filename. |
| `--summary-only` | Off | Save statistics and metadata without full `P`/`S`, certificate vectors, or polynomial arrays. Does not turn off requested console output or post-processing. |
| `--profile FILE` | None | Save a Python `cProfile` recording of computation. Measured solve timings then include profiling overhead. |

Input, JSON, CSV, profile, and requested syzygy output paths must be distinct.
Use distinct result names for comparisons. A summary-only file cannot be loaded
by `show`, verification, or reconstruction. For fast statistics-only output,
combine `--summary-only --no-show-basis --no-show-u` and leave polynomial views
off.

### 7.2 Display switches (`run` and `show`)

| Parameter | Default | What it displays / computes |
| --- | --- | --- |
| `--show-basis` / `--no-show-basis` | On | Original aligned `P` and `S` tables. |
| `--show-minimal-gb` / `--no-show-minimal-gb` | Off | A minimal strong polynomial Groebner basis extracted from completed `P`. |
| `--show-reduced-gb` / `--no-show-reduced-gb` | Off | The further reduced strong polynomial basis; potentially expensive. |
| `--show-minimal-lt` / `--no-show-minimal-lt` | Off | Leading terms of the minimal strong basis, including integer coefficients. |
| `--show-u` / `--no-show-u` | Automatic | Print vectors with the basis if stored. Explicit `run --show-u` also enables certificate tracking and prints vectors even if basis tables are hidden. |
| `--show-forest` / `--no-show-forest` | Off | Same-ratio forests from insertion history. |
| `--print-style` | `ascii` | `ascii` or `unicode` mathematical/branch glyphs. |
| `--print-width N` | `100` | Target width for wrapping long records; minimum `40`. |

`--no-show-basis` and `--no-show-u` only hide console content; they do not remove
full source records from ordinary JSON or disable explicitly requested certificate
tracking. `show --show-u` displays stored vectors and never recomputes them.

`--show-minimal-lt` is not just a cheap scan of the console table: it requests
minimal polynomial basis extraction. `run` saves that view in
`polynomial_postprocessing` (full polynomial arrays unless summary-only). The
reduced view uses balanced remainders `-b/2 < r <= b/2` for positive divisor `b`
and never divides out nonunit content.

Polynomial views preserve original signatures, vectors, paths, and solver
counters. They assume the source projection is a strong basis; they are not
independent certificates. Post-processing is skipped for incomplete source runs
or when requested source verification did not pass.

## 8. Independent verification

Computation and verification are separate. `Status: completed` records solver
completion; inspect the verification report to know which independent checks
were performed and whether input-ideal membership was established.

### 8.1 Check immediately after solving

These switches are available on `run` and `benchmark`:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--verify` | Off | Request independent result checks after solving. |
| `--verify-strong` | Off | Request verification with additional ordinary integer S checks; also triggers verification without `--verify`. |
| `--verify-mode` | `naive` | `naive`: direct checking; `static`: indexed static verification. |
| `--verify-timeout N` | `60` | Verification time budget in seconds; `0` disables it. Separate from solver `--timeout`. |
| `--verify-max-steps N` | `200000` | Step bound per ordinary strong-remainder check; `0` disables it. Not a global head-plus-tail counter like solver `--max-steps`. |
| `--certificates` | Off | Track exact representation vectors during solving, enabling independent membership checks. May use substantial memory and time. |

For a small fully recorded example:

```text
python -m gvwzz run --input examples/paper.json --certificates --verify-strong --verify-mode static --verify-timeout 0 --verify-max-steps 0 --no-show-basis --no-show-u --output results/paper-certified.json
```

Both verification modes check final J cover, G top reducibility, syzygy
coefficient-gcd completeness/minimality, and strong remainders of the original
inputs. Stored vectors add exact membership and signature checks.

Without vectors or valid generation paths, algebraic criteria alone do not
independently prove that every output belongs to the original input ideal.
`--verify-strong` does not manufacture missing representations. Verification
time is excluded from solver `Compute` time.

### 8.2 Check a saved full result

```text
python -m gvwzz verify results/paper-certified.json --strong --require-certificates --verify-mode static --timeout 0 --max-steps 0
```

For the `verify` subcommand, `FILE` is required. `--strong` adds ordinary integer
S checks (default off); `--require-certificates` requires valid stored exact
vectors (default off). `--verify-mode` defaults to `naive`. Here `--timeout`
defaults to `60` and `--max-steps` to `200000`, and both apply to verification,
with `0` meaning unlimited. The step bound is per ordinary remainder check,
not a cumulative count. Use `verify-paths` for recorded path membership;
`verify --require-certificates` specifically expects explicit vectors.
`verify` does not accept `--verify-timeout` or
`--verify-max-steps`; those names belong to `run`/`benchmark`.

## 9. Generation paths and syzygy reconstruction

Path mode records arithmetic dependencies without expanding full vectors online.
It is an alternative to `--certificates`, not an additional certificate mode to
combine with it. `--record-paths` and `--certificates`/explicit `--show-u` are
mutually exclusive. Path mode requires a full result, so `--summary-only` is
incompatible.

### 9.1 Record and replay separately

Replay requires a completed full result containing recorded generation paths.
Signature-only results cannot recover missing paths after computation.

```text
python -m gvwzz run --input examples/paper.json --record-paths --no-show-basis --no-show-u --output results/paper-paths.json
python -m gvwzz verify-paths results/paper-paths.json --strong --timeout 0 --max-steps 0
python -m gvwzz reconstruct results/paper-paths.json --output results/paper-syzygies.json --verify
python -m gvwzz verify-syzygies results/paper-syzygies.json
```

### 9.2 `run` path/replay switches

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--record-paths` | Off | Store generation paths in the full result. Also accepted by `benchmark`, whose output remains summary-only. |
| `--reconstruct-syzygies` | Off | Imply path recording, save the completed source result, then replay final `S` vectors. |
| `--syzygy-output FILE` | `<result-stem>_syzygies.json` | Separate output used with `--reconstruct-syzygies`. |
| `--verify-syzygies` | Off | With reconstruction, also verify expanded zero identities; checking time is reported separately from replay. |

### 9.3 `reconstruct FILE` parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--output FILE` | Required | Destination for reconstructed syzygies. |
| `--format` | `json` | `json` or streaming `jsonl`; interrupted JSONL remains as `.partial`. |
| `--indices LIST` | All final `S` | Distinct 1-based final `S` indices, for example `1,3,5`. |
| `--verify` | Off | Check expanded zero identities as well as complete signatures. |
| `--progress SECONDS` | `5` | Replay progress interval; `0` disables it. |
| `--show-syzygies` | Off | Print expanded vectors. |
| `--syzygies-only` | Always the behavior | Compatibility flag; replay only reconstructs final syzygies. |

Reconstruction replays recorded operations, without rerunning GVW or reducing
the polynomials again. It does not expand all `P` vectors.

### 9.4 Path and explicit-syzygy checks

`verify-paths FILE` accepts `--strong` (off), `--paths-only` (off), `--timeout`
(`60` seconds), and `--max-steps` (`200000`). Zero disables either budget.
`--paths-only` checks recorded identities without final completeness criteria.
The command verifies paths without expanding all `P` representation vectors.
Its step bound applies to final remainder checks, not to the number of path
nodes. Final criteria use the static checker; there is no `--verify-mode`
option on this subcommand.

`verify-syzygies FILE` takes a reconstructed JSON/JSONL file and checks explicit
zero identities and complete signatures. It has no additional tuning switches.
These syzygy checks alone do not establish completeness of `S` or certify the
entire signature basis.

## 10. Incremental POT mode

`--incremental-pot` is off by default; `--no-incremental-pot` selects ordinary
mode. Incremental mode completes successive input prefixes, retaining a minimal
strong polynomial basis at stage boundaries. It is a separate ideal-basis
engine with different output scope.

```text
python -m gvwzz run --case eco3 --incremental-pot --module-order pot --show-minimal-lt --no-show-basis --no-show-u --progress 30 --output results/eco3-incremental.json
```

Requirements and effects:

- Omit `--module-order`, or use `pot`; blocks and nontrivial position permutations
  are rejected. Generator reversal remains available.
- `--show-forest` and `--reconstruct-syzygies` are unavailable. `--certificates`
  or `--record-paths` remain supported as alternative recording modes.
- Final output is a polynomial ideal basis, not a complete signature basis of
  the original graph module. Global `S` and a global forest are not returned.
- `signature_basis_certified` remains false in this mode. A successful full
  membership/ideal verification can establish `strong_basis_certified` instead.
- Budgets accumulate across stages. An interrupted/limited run returns only the
  last completed prefix basis and marks the full problem incomplete.

Compare incremental output through polynomial bases or independently verified
ideals, rather than requiring identical intermediate `P`/`S` or pair counts.
See [incremental POT](docs/incremental.md) for the stage invariant and checks.

## 11. Benchmarks and random inputs

### 11.1 Benchmark parameters

```text
python -m gvwzz benchmark --cases cyclic6,katsura8,eco10,noon7,reimer5 --poly-orders grevlex --module-orders g2 --repeat 3 --require-native --progress 30 --output results/families
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--suite NAME` | `quick` | A predefined collection; available names are shown by `benchmark --help`. |
| `--cases LIST` | Suite selection | Comma-separated case names, overriding `--suite`. |
| `--poly-orders LIST` | `grevlex` | Comma-separated polynomial orders. |
| `--module-orders LIST` | `pot,top,gvw-g1,gvw-g2,schreyer` | Comma-separated module orders. Use `--module-orders g2` to benchmark only the ordinary default order. |
| `--signature-blocks LIST` | None | Force the block-order benchmark with the supplied block sizes. |
| `--repeat N` | `3` | Positive repetition count per case/order combination. |
| `--output PREFIX` | `results/benchmark` | Prefix for the four files listed below; omit a filename extension. |

Solver, verification, progress, and generator reversal options are shared with
`run`. Display switches and `--positions` are not benchmark options. With
`--incremental-pot`, omitted module orders select only `pot`; explicit non-POT
orders or blocks are rejected. Supplying blocks in ordinary mode forces the
block-order selection even if `--module-orders` was also supplied.

Available suites are `smoke`, `quick`, `medium`, `cyclic`, `cyclic-scaled`,
`katsura`, `to10`, `eder`, `eder-variants`, `eder-extra`, `eder-all`, `eco`,
`noon`, `reimer`, `henrion`, `katsura-qfirst`, `classic-small`, `classic-medium`,
and `classic-paper`.

The default benchmark schedules 150 runs: 10 quick cases, five module orders,
and three repeats. Specify orders and repeats for a targeted comparison.
It writes `PREFIX.raw.jsonl`, `PREFIX.raw.csv`, `PREFIX.summary.csv`, and
`PREFIX.summary.json`. Each raw JSONL record is flushed after its run.
Records are summaries, not replayable full basis files, even when
certificate/path recording was enabled for that benchmark computation.
Compare completed runs with identical equations, options, interpreter, and
hardware; report verification and post-processing time separately.

Summary numerical metrics use medians of completed repeats when any completed.
Check `all_completed`, `repeats_completed`, `repeat_statuses`, `all_verified`,
and certification fields. A limited/unsupported run stops the remaining repeats
for that combination. A zero benchmark exit code does not guarantee that every
combination completed; inspect the records. Interrupted benchmarks return `2`;
failed requested verification returns `3`.

### 11.2 Random inputs

```text
python -m gvwzz random --seed 42 --nvars 3 --ngens 4 --degree 3 --terms 5 --coeff-bound 7 --output examples/random-42.json
python -m gvwzz run --input examples/random-42.json --no-show-basis --output results/random-42.json
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--seed` | `1` | Integer random seed for reproducible generation. |
| `--nvars` | `2` | Positive variable count. |
| `--ngens` | `3` | Positive generator count. |
| `--degree` | `2` | Positive maximum sampled total degree. |
| `--terms` | `4` | Positive requested term samples per polynomial; repeated monomials can merge or cancel. |
| `--coeff-bound` | `5` | Positive bound on absolute sampled integer coefficients. |
| `--output FILE` | `examples/random.json` | Input JSON destination. |

`random` also accepts `--reverse-generators` and its aliases. It writes input
only; solving requires a separate `run` command. Unlike `export-case`, it can
overwrite its output file.

## 12. Timing reports and diagnostics

### 12.1 Reading the summary

| Output | Interpretation |
| --- | --- |
| `Status` | Completion/limit/interruption state; check before interpreting the basis. |
| `P`, `S` | Retained labels/signatures, not necessarily minimal polynomial basis sizes. |
| `Generated J/G` | Generated candidates, many of which can be rejected without materialization. |
| `reduced J`, `zero` | J head-reduction calls and zero results, distinct from individual reduction steps. |
| `regular`, `singular`, `Euclidean`, `tail` steps | Types of reduction work; tail counts include source-specific J/G subtotals. |
| `Cover rejected J`, `J dedup` | Rejections by cover and exact candidate deduplication. |
| `materialized J/G` | Candidate polynomials actually expanded from lazy recipes. |
| Queue peak | Maximum live pending queue size. |
| Native work counters | Which eligible native batches/workspaces/queue/cover paths actually ran. |

`Compute` is solver wall-clock time. The exclusive time-breakdown rows partition
that time; `CPU` is a separate process CPU measurement and must not be added.
Aggregate summaries such as total syzygy time and tail subtotals overlap their
breakdown rows, so do not sum both representations.

Setup, independent verification, solver cleanup, JSON writing, polynomial
post-processing, and display are outside `Compute`. `[output]` messages and
`After Compute` identify output work. A large run can therefore spend substantial
time after `Status: completed`. The full source JSON is committed before
polynomial post-processing, so interruption of that stage does not discard the
saved source basis. For solver-only comparisons, leave polynomial views off.

### 12.2 Diagnostics and environment controls

```text
python -m gvwzz doctor
python -m gvwzz doctor --json --require-native
```

`doctor --json` prints machine-readable diagnostics. `doctor --require-native`
exits with status `1` if required accelerators are unavailable. Inspect the
separate arithmetic line for `gmp`.

| Environment variable | Unset behavior | Use |
| --- | --- | --- |
| `GVWZZ_NO_NATIVE` | Native imports permitted | Set to `1` before starting Python to force exact Python fallback for comparisons; unset to restore native imports. Any nonempty value, including `0`, disables them. |
| `GVWZZ_NO_GMP` | Detect bundled GMP during build | Set to `1` before rebuilding to use Python integers inside native reduction; unset and rebuild to restore GMP detection. Any nonempty value disables detection. |
| `CC`, `CXX` | Build-system defaults | On Unix-like systems, select C/C++ compilers for native builds when needed. Windows uses the MSVC toolchain. |

These are environment settings, not CLI parameters. Build with the same Python
used to run the program. No fixed speedup is promised for a particular heuristic,
case, or operating system.

## 13. Python API

### 13.1 Compute a basis

```python
from gvwzz import Options, compute

options = Options(
    reduction_kernel="heap",
    reducer_selection="sparse",
    tail_reduce=True,
    g_tail_reduce=True,
    progress_interval=30.0,
    max_seconds=None,
)
result = compute(
    ["4*x^2", "9*y^4", "6*x*y^2+y"],
    ["x", "y"],
    polynomial_order="grevlex",
    module_order="g2",
    options=options,
    show_basis=False,
    show_u=False,
    show_forest=False,
    show_minimal_lt=True,
)
print(result.status)
```

`compute` returns a `Result`. Its first two arguments are the input polynomial
and variable sequences. Optional arguments are `polynomial_order` (`grevlex`),
`module_order` (`None`, resolved to `g2` or incremental `pot`), `positions`
(`None`, natural order), `block_sizes` (`None`), and `options` (`Options()` when
omitted). **API positions are 0-based**, unlike the CLI and saved JSON. Supply
`module_order="block"` explicitly when using API `block_sizes`.

Display keywords are `show_basis=True`, `show_forest=False`, `show_u=None`,
`print_style="ascii"`, `print_width=100`, and `show_minimal_gb`,
`show_reduced_gb`, `show_minimal_lt` (all false). Explicit `show_u=True` enables
representation tracking on a copy of the options. Display and post-processing
are outside measured solve time. `compute` does not automatically save a file.

### 13.2 `Options` field reference

`Options` is immutable. Use `dataclasses.replace(options, field=value)` for a
modified copy. The CLI rules above apply with these names:

| API field | Default | CLI counterpart |
| --- | --- | --- |
| `cover_mode` | `"indexed"` | `--cover-mode` |
| `j_deduplicate` | `True` | False with `--no-j-dedup` |
| `reducer_policy` | `"regular-first"` | `--reducer-policy` |
| `g_first` | `True` | False with `--j-first` |
| `queue_tie` | `"small-coeff"` | `--queue-tie` |
| `debug` | `False` | `--debug` |
| `track_representations` | `False` | `--certificates` |
| `record_paths` | `False` | `--record-paths` |
| `euclidean` | `True` | `--euclidean` / `--no-euclidean` |
| `euclidean_priority` | `"signature-first"` | `--euclidean-priority` |
| `tail_reduce` | `True` | `--tail-reduce` / `--no-tail-reduce` |
| `g_tail_reduce` | `True` | `--g-tail-reduce` / `--no-g-tail-reduce` |
| `tail_policy` | `"all"` | `--tail-policy` |
| `tail_step_cap` | `0` | `--tail-step-cap` |
| `reduction_kernel` | `"heap"` | `--reduction-kernel` |
| `lazy_candidates` | `True` | `--lazy-candidates` / `--no-lazy-candidates` |
| `g_precheck` | `True` | False with `--no-g-precheck` |
| `j_generation_filter` | `True` | False with `--no-j-generation-filter` |
| `g_policy` | `"discard"` | `--g-policy` |
| `reducer_selection` | `"sparse"` | `--reducer-selection` |
| `max_seconds` | `None` | `--timeout` |
| `max_basis` | `None` | `--max-basis` |
| `max_candidates` | `None` | `--max-candidates` |
| `max_reduction_steps` | `None` | `--max-steps` |
| `max_polynomial_terms` | `None` | `--max-terms` |
| `max_coefficient_bits` | `None` | `--max-coefficient-bits` |
| `cover_cache_size` | `8192` | API only: maximum entries in each cover/top-test query cache; `0` disables these caches. |
| `progress_interval` | `0.0` | `--progress` |
| `incremental_pot` | `False` | `--incremental-pot` |

In Python, disabled `max_*` budgets use **`None`**, not `0`. Positive values
enable them; zero is invalid. `tail_step_cap=0`, `cover_cache_size=0`, and
`progress_interval=0` retain their meanings given above. Boolean fields require
actual booleans. Use an `Options` instance or `None`, not a dictionary.

Input reversal is preparation, not an `Options` field. Use
`gvwzz.reverse_generators(case)` or
`gvwzz.cases.get_case(name, reverse_generators=True)` (import `get_case` from
`gvwzz.cases`);
see [generator order](docs/input_order.md). `python examples/demo.py` demonstrates
the API, path recording, and reconstruction.

## 14. Tests, source layout, and contributing

### 14.1 Release checks

```text
python run_tests.py
python run_tests.py --require-native --differential
```

`--require-native` checks native availability before the suite. `--differential`
also requires native extensions and compares fresh Python/native runs with
32 bundled reference cases. `--quiet` prints a compact test report. No earlier
release is needed.

The GitHub Actions workflow configures Python fallback and native checks on
selected Linux, Windows, and macOS/Python combinations. A configured workflow
does not establish that a platform run passed; consult its actual results.
Local v1 validation used Linux x86-64, Python 3.12.14. See the
[audit report](docs/audit-v1.md) and [release notes](docs/release-notes.md).

### 14.2 Source layout

| Path | Contents |
| --- | --- |
| `gvwzz/` | Python solver, API, CLI, verification, and output. |
| `native/` | Cython sources, shared declarations, and portable C++ header. |
| `native/generated/` | Generated C++ for standalone native builds. |
| `build_native.py` | Build entry point for the current Python interpreter. |
| `scripts/` | Native build implementation. |
| `tests/` | Regression checks and reference fixtures. |
| `examples/` | Small inputs and Python API example. |
| `docs/` | Usage, architecture, input order, incremental mode, and release reports. |

Start with [architecture](docs/architecture.md) for implementation structure and
[usage notes](docs/usage.md) for additional output/performance details.

### 14.3 Publish and contribute

Repository: [xiaopeng-stu/gvwzz](https://github.com/xiaopeng-stu/gvwzz).
See [Publishing on GitHub](docs/github-publishing.md) for updating the repository
and publishing the `v1` release. Bug reports and changes are welcome; see
[CONTRIBUTING.md](CONTRIBUTING.md). Distributed under the [MIT license](LICENSE).

## 15. Troubleshooting

| Symptom | Action |
| --- | --- |
| `git` or `python` is not recognized | Install the prerequisites, reopen the terminal, and check the version commands. |
| A requirements/build file cannot be found | Change into the source directory containing this README and `build_native.py`. |
| `Microsoft Visual C++ 14.0 or greater is required` | Install the Desktop development with C++ workload and rerun the native build. |
| `Python.h` is missing on Linux | Install development headers matching the Python interpreter used for the build. |
| Native entry points are unavailable | Run `doctor` and rebuild with the same interpreter in the current source tree. |
| Arithmetic is `python` after a native build | Install `requirements-native.txt`; check the build messages for bundled GMP headers/libraries, unset `GVWZZ_NO_GMP`, and rebuild. |
| Extensions are disabled | Unset `GVWZZ_NO_NATIVE` and start a fresh Python process. Setting it to `0` still disables extensions. |
| A long pause follows `Status: completed` | Check `[output]` messages; cleanup, serialization, minimal/reduced views, and printing are outside solver time. |
| A result cannot be shown or verified | Use a full result, not `--summary-only` or benchmark summary JSON. |
| `--timeout` seems ignored | Remove `--no-limits`; also remember checks are cooperative rather than hard deadlines. |
| Vectors are missing | Record `--certificates` during computation, or use recorded paths for supported replay. Hiding vectors does not delete them. |
| Output is too large | Hide `P`/`S` with `--no-show-basis`; request only `--show-minimal-lt`, or leave all views off for a solve-only run. |

For slow dependency downloads, retry the same requirements with an accessible
PyPI mirror. For example, the [TUNA PyPI mirror](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)
can be selected for one installation command without changing global settings:

```text
python -m pip install -r requirements-native.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python build_native.py
python -m gvwzz doctor --require-native
```

Continue to build only after dependency installation succeeds. This changes the
download source, not the pinned requirements or algorithm. If a mirror has not
yet synchronized a pinned version, retry the original index or another mirror;
do not silently change dependency versions for a benchmark comparison.
