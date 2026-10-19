# GVWZZ v1 release notes

Git tag: `v1`  
Python package version: `1.0.0`

This release prepares the supplied `0.10.0.post31` CPU source package for public
distribution under its existing MIT license. It keeps the Python package name,
public CLI, source layout, and native source files. It is standalone and does not
need an earlier package or updater.

## Included

- Exact arbitrary-precision integer arithmetic over `ZZ`, with one CPU solver
  thread and optional C++/GMP accelerators.
- The existing J/G pair generation, signature constraints, coefficient-aware
  cover, tail reduction, and checked fallback paths.
- Editable native sources and generated C++ for all five extension modules.
- Regression tests, small reference fixtures, examples, and English guides.
- GitHub publishing instructions, contributor guidance, and a cross-platform
  GitHub Actions workflow.

The release contains no GPU backend and no precompiled extensions. Rebuild native
extensions with the Python interpreter used to run the program. In-place builds
can depend on that environment's gmpy2 libraries, so rebuild after moving or
replacing the environment.

## Corrections in v1

- Validate public solver options at the API boundary, including invalid option
  values and overflowing time settings, before starting the computation.
- Reject malformed metadata in custom input files before solving or writing an
  output file.
- Return exit status `3` when requested benchmark verification fails, while
  still saving the benchmark reports for diagnosis.

These changes address validation and error reporting. They do not introduce a
new pair order or promise a performance improvement.

## Validation

Checked on Linux x86-64 with Python 3.12.14, SymPy 1.14.0, Cython 3.3.0,
and gmpy2 2.3.2:

- Native build: 471 regression tests passed, with no skips.
- Python fallback: 307 tests passed; 164 native-only tests were skipped.
- All five extensions built from an extracted source distribution without
  Cython; the resulting GMP build passed the same 471 tests and the 32-case
  bundled reference comparison.
- Independent installed-wheel smoke checks passed outside the source tree.
- A reproducible audit completed 141 certified computations across six module
  orders and three reduction kernels. Native and Python outputs matched exactly;
  7,026 additional ordinary G-polynomial remainders were zero.
- Cyclic-7 and Katsura-8 completed with exactly the same ordered P/S records
  and all work counters as the supplied post31 package. These were compatibility
  checks, not a statistically controlled performance benchmark.

See [the audit report](audit-v1.md) and [machine-readable validation](release-validation.json)
for scope and results. Historical post31 timings are not used as v1 test results.
The supplied CI workflow targets selected Python 3.10, 3.13, and 3.14
configurations on Linux, Windows, and macOS. Windows/macOS and these additional
Python versions have not been executed in this release review; the workflow
provides those checks when pushed to GitHub.

Reproduce the main checks from the source root:

```text
python -m pip install -r requirements-native.txt
python build_native.py
python run_tests.py --require-native --differential
python -m tests.release_random_audit
```

The differential suite uses bundled fixtures and fresh Python/native processes.
No earlier release is required. Native-only tests are skipped if native
extensions are unavailable, unless native availability is explicitly required.

## Limits

Passing regression checks does not prove that every possible input is bug-free.
Completed solver output, ordinary-basis checks, and full certificates are
different reports; requested minimal/reduced polynomial views do not add an
independent correctness certificate. Large instances can require substantial
time and memory, including post-processing after the solver has completed.
