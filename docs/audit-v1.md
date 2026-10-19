# v1 release audit

Review date: 2026-10-19. Base: the user-supplied `0.10.0.post31` CPU source
archive. Release name and Git tag: `v1`; Python package version: `1.0.0`.

## Confirmed defects fixed

1. **Invalid options at the API boundary.** A falsey value such as `{}` or
   `False` could silently select defaults in the direct solver, while the public
   `compute` API could raise an unrelated `AttributeError`. `compute`,
   `GVWInteger`, and `IncrementalPOT` now require `Options` or `None` consistently.
2. **Oversized time settings.** Extremely large integers or floats could raise
   `OverflowError` during validation or conversion to nanoseconds. Such settings
   now raise a clear `ValueError`; unlimited computation remains available with
   `max_seconds=None`.
3. **Malformed custom-input metadata.** Invalid `name`, `provenance`, or
   `input_order` fields could crash the CLI, sometimes after a solve had finished.
   Metadata is now checked before solving or replacing an output file.
4. **Benchmark verification exit status.** A failed requested verification could
   still return success. The benchmark command now returns status `3` and keeps
   the diagnostic reports, consistent with the other verification commands.

Five new regression test methods cover these boundaries. No reduction loop,
pair-generation policy, cover rule, monomial packing scheme, or native arithmetic
source was changed. The existing MIT license is retained.

## Validation performed

| Check | Result |
| --- | --- |
| Cython-based native build | All five extensions built with GMP |
| Full native regression suite | 471 passed, zero skipped |
| Python fallback suite | 307 passed, 164 native-only skips |
| Generated C++ build without Cython | Passed from extracted source distribution |
| Suite on generated C++ build | 471 passed, zero skipped |
| Bundled reference comparison | 32 native/Python cases matched |
| Optional fixed-seed certificate audit | 141 completed computations, exact native/Python agreement |
| Additional ordinary G-polynomial checks | 7,026 zero remainders |
| Installed Python wheel outside source directory | Version, exact solve/certificate, and options checks passed |
| Cyclic-7 and Katsura-8 compatibility | Ordered results and all work counters matched post31 |

The reference comparison includes deliberately budget-limited cases; it checks
their exact partial states and does not count them as completed basis
computations. The full suite also checks packing and coefficient boundaries,
including sparse 129-variable cases and coefficients exceeding 4,096 bits.

The optional audit is reproducible with `python -m tests.release_random_audit`.
Its seed is `20261019`; it runs eight input systems with six module orders and
three reduction kernels, including zero and duplicate generators. It verifies
exact representation certificates and signature criteria, and additionally
checks ordinary G-polynomial remainders. The native and Python output digest is:

```text
7560c7de91f78fb8f4f3b904c838625f621d4b460ec2f47b36528bb5c1ca6d73
```

The two larger compatibility checks use fresh processes and identical compiled
native sources. They compare the complete ordered labelled bases, syzygies, and
all work counters. Their timings are single-run observations in a shared
environment; they do not establish a speedup or a performance guarantee.

## Scope and limits

Review covered public input validation, integer arithmetic boundaries, signatures
and cover selection, native fallbacks, queue lifetimes, serialization,
post-processing, packaging, and diagnostics. No numerical defect was identified
in the inspected paths and tested cases. Testing is bounded and does not prove
that the implementation is bug-free or certify every possible input.

Local checks used Linux x86-64, Python 3.12.14, SymPy 1.14.0, Cython 3.3.0, and
gmpy2 2.3.2. Windows/macOS execution and other Python versions remain to be
confirmed by the included GitHub Actions workflow. The release contains source
only; each native build must use its target Python and platform.

The source archive excludes compiled modules, environments, temporary compiler
files, run outputs, and previous release archives. Machine-specific paths in
generated C++ metadata comments are removed; generated executable code is
unchanged. The GitHub owner and repository URL remain unspecified until the
maintainer creates the repository using `docs/github-publishing.md`.
