# GVWZZ v1

GVWZZ computes coefficient-aware GVW-type signature bases over integer polynomial
rings (`ZZ`). It uses one CPU thread and arbitrary-precision exact integers.
Optional C++/GMP extensions accelerate reduction, pair generation, indices,
queues, and cover queries. Full J/G tail reduction is enabled by default.

This standalone CPU source release is based on `0.10.0.post31`. The Python
package version is `1.0.0`; the release name and Git tag are `v1`. Extract it into
a new directory; no earlier release or updater is needed. Existing post31
commands remain available. See the [release notes](docs/release-notes.md) for
the review and validation scope.

## Install and build

Use Python 3.10 or later. Extract the archive and locate `gvwzz-v1`, the
directory containing this README and `build_native.py`.

### Windows

Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
with the **Desktop development with C++** workload. Open PowerShell in
`gvwzz-v1`, then run:

```powershell
python -m pip install -r requirements-native.txt
python build_native.py
python -m gvwzz doctor --require-native
```

### Linux

Linux native builds require a C++ compiler, Python development headers, and
Python's `venv` module. On Ubuntu or Debian, install these prerequisites:

```bash
sudo apt update
sudo apt install -y build-essential python3-dev python3-venv
python3 --version
```

Check that `python3` is version 3.10 or later. For other Linux distributions,
install the equivalent packages with the distribution's package manager. If you
use a separately installed Python, its development headers must match that
interpreter.

From the directory containing the extracted `gvwzz-v1` folder:

```bash
cd gvwzz-v1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-native.txt
python build_native.py
python -m gvwzz doctor --require-native
```

Run Cyclic-7 in the same activated environment:

```bash
python -m gvwzz run --case cyclic7 \
  --poly-order grevlex --module-order g2 --require-native \
  --show-minimal-lt --no-show-basis --no-show-u --no-show-forest \
  --progress 30 --output results/cyclic7.json
```

To run the regression and reference checks:

```bash
python run_tests.py --require-native --differential
```

In a new terminal, return to `gvwzz-v1` and run `source .venv/bin/activate`
before using the program. Run `deactivate` when finished. Install Python packages
inside the virtual environment; `sudo` is only used for system packages above.
This release was checked on Linux x86-64 with Python 3.12.14; see the
[validation report](docs/audit-v1.md) for the tested scope.

### macOS

Install Xcode Command Line Tools with `xcode-select --install` and use Python
3.10 or later. From `gvwzz-v1`, use the virtual-environment, pip, build, and
diagnostic commands in the Linux section. The Ubuntu/Debian `apt` commands
do not apply to macOS.

### Build diagnostics and installation modes

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

## Run a case

Save a complete result and display only minimal leading terms:

```powershell
python -m gvwzz run --case cyclic7 --poly-order grevlex --module-order g2 --require-native --show-minimal-lt --no-show-basis --no-show-u --no-show-forest --progress 30 --output results/cyclic7.json
```

For Noon-9:

```powershell
python -m gvwzz run --case noon9 --poly-order grevlex --module-order g2 --require-native --show-minimal-lt --no-show-basis --no-show-u --no-show-forest --progress 30 --output results/noon9.json
```

Check `Status: completed` before using the result as a completed basis. The output
directory is created automatically. `--progress 30` reports progress at safe
execution boundaries; it is not a hard time limit.

`--no-show-basis` controls console display. The default JSON still contains the
complete P/S and can be loaded later:

```powershell
python -m gvwzz show results/cyclic7.json --show-minimal-gb --show-minimal-lt --no-show-basis --no-show-u --no-show-forest
```

Add `--summary-only` to a `run` command for statistics-only JSON. It reduces saved
output, but the file cannot be used by `show`, verification, or reconstruction.
Requested polynomial views are still computed. `--show-reduced-gb` requests the
reduced strong polynomial basis; this may be expensive.

`Compute` measures the solver. `[output]` messages and `After Compute` separately
report cleanup, JSON writing, polynomial post-processing, and display. Large
results can require substantial time after solver completion. Minimal and reduced
views assume P is already a strong polynomial basis; they are not independent
certificates.

## Scope and performance

All inputs are interpreted over `ZZ`; no primitive-part division or rational
normalization is performed. Noon inputs retain their nonhomogeneous integer
equations and must not be divided by 10. Eco uses the final equation
`x1 + ... + x(n-1) + 1`. Inspect exact equations and provenance with
`python -m gvwzz export-case NAME` before comparing implementations.

Variable count has no fixed limit beyond available resources. Native monomial
encodings have checked bounds; unsupported orders, exponents, or feature
combinations select exact fallback paths. These bounds do not restrict integer
coefficient precision. Certificates, debug checks, and solver budgets can change
kernel eligibility. Runtime depends on the case and hardware; no fixed speedup is
promised.

## Python API

```python
from gvwzz import Options, compute

result = compute(
    ["4*x^2", "9*y^4", "6*x*y^2+y"],
    ["x", "y"],
    polynomial_order="grevlex",
    module_order="g2",
    options=Options(),
    show_basis=False,
    show_u=False,
    show_forest=False,
)
print(result.status)
```

`python examples/demo.py` demonstrates path recording and reconstruction.

## Checks

```powershell
python run_tests.py
python run_tests.py --require-native --differential
```

The second command additionally checks native availability and compares fresh
Python/native runs with bundled reference fixtures. No older release is needed.
See the [release notes](docs/release-notes.md) for checks performed on this release.

The GitHub Actions workflow runs Python fallback and native checks on selected
Linux, Windows, and macOS/Python combinations. A configured workflow is not a
record of a completed platform test; consult its actual run results.

## Layout and documentation

| Path | Contents |
| --- | --- |
| `gvwzz/` | Python solver, API, CLI, verification, and output |
| `native/` | Cython sources, shared declarations, and portable C++ header |
| `native/generated/` | Generated C++ required for standalone native builds |
| `scripts/` | Build implementation used by `build_native.py` |
| `tests/` | Regression checks and reference fixtures |
| `examples/` | Small input and Python API example |
| `docs/` | Usage, architecture, generator-order, incremental-mode, and release notes |

Start with [usage](docs/usage.md), [architecture](docs/architecture.md),
[generator order](docs/input_order.md), or [incremental POT](docs/incremental.md).
Use `python -m gvwzz --help` and each subcommand's `--help` for all options.

## Publish and contribute

See [Publishing on GitHub](docs/github-publishing.md) for creating the repository
and the `v1` release. The repository owner has not been set in this source
archive. `YOUR-USERNAME` in that guide is a placeholder, not an existing project
address. The repository URL will be `https://github.com/YOUR-USERNAME/gvwzz`
after publication under that name.

Bug reports and changes are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Troubleshooting

| Symptom | Action |
| --- | --- |
| A requirements or build file cannot be found | Change into the extracted directory containing this README. |
| `Microsoft Visual C++ 14.0 or greater is required` | Install the C++ workload above and rerun `python build_native.py`. |
| A native extension or queue/cover entry point is unavailable | Run `python -m gvwzz doctor`; rebuild with the same interpreter. |
| Arithmetic is `python` after a native build | Install `requirements-native.txt`, remove any `GVWZZ_NO_GMP` override, and rebuild. |
| Extensions are disabled | Remove `GVWZZ_NO_NATIVE` and start a fresh Python process. |

Distributed under the [MIT license](LICENSE).
