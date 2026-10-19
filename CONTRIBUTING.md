# Contributing

GVWZZ computes over integer polynomial rings with exact arithmetic. Changes must
preserve coefficient divisibility, signature admissibility, and checked native
fallbacks. A numerical approximation or an unproved pair-discard rule is not a
replacement for an exact computation.

## Report a bug

Open an issue in the published repository and include:

- The package version, operating system, and `python --version`.
- The complete command and `python -m gvwzz doctor --json` output.
- The input equations, variable order, polynomial order, and module order.
- The expected result, observed result, and full error traceback, if any.
- A small reproducible case whenever possible.

For performance reports, include solver options, elapsed solver and output times,
work counters, and native usage. Compare matching equations and coefficient
rings. An integer ideal must not be replaced by its rational extension or by
primitive-part normalization when comparing results.

## Develop and test

Work in a virtual environment and run commands from the source root:

```text
python -m pip install -r requirements-native.txt
python build_native.py
python run_tests.py --require-native --differential
```

To check the Python fallback explicitly in PowerShell:

```powershell
$env:GVWZZ_NO_NATIVE = "1"
python run_tests.py
Remove-Item Env:GVWZZ_NO_NATIVE
```

On Linux or macOS:

```sh
GVWZZ_NO_NATIVE=1 python run_tests.py
```

Add a focused regression for a corrected defect. For arithmetic, pair criteria,
or ordering changes, compare Python and native execution on exact small cases.
The test suite also includes a bundled reference snapshot; it does not require
an old installation.

## Native sources

Edit `native/*.pyx`, `native/gvwzz/*.pxd`, and the shared headers. Building with the
pinned Cython dependency regenerates `native/generated/*.cpp`; include those
updated files in a change so users can build from the complete source archive.
Do not commit compiled extensions, build directories, or benchmark results.
Keep comments, documentation, and public help text in English.

Describe what changed, why it is needed, and which checks you ran in each pull
request. Report untested platforms and incomplete benchmark runs explicitly.
Contributions are distributed under the repository's MIT license.
