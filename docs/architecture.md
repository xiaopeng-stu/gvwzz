# Architecture

`gvwzz` exposes `Options`, `Result`, and `compute`. Its CLI is `python -m gvwzz`
or the installed `gvwzz` command. Python owns configuration, mathematical
objects, feature selection, verification, and output. Optional C++/GMP extensions
execute eligible inner loops on the same CPU thread.

## Runtime modules

| Module | Responsibility |
| --- | --- |
| `solver.py` | Ordinary signature solver, J/G dispatch, and reduction orchestration |
| `incremental.py` | Separate incremental POT stage lifecycle |
| `core.py`, `polynomial.py`, `orders.py` | Exact helpers, polynomial workspaces, signatures, and orders |
| `structures.py`, `frozen.py`, `monomial_packing.py` | Queues, indices, frozen-prefix queries, and checked encodings |
| `recipes.py` | Compact lazy J/G polynomial records |
| `options.py`, `result.py`, `runtime.py`, `native_status.py` | Configuration, serialization, metadata, and diagnostics |
| `cli.py`, `path_cli.py` | Commands, input preparation, streaming output, and replay interface |
| `cases.py`, `benchmarks.py` | Exact built-in inputs, provenance, and benchmark catalog |
| `certificates.py`, `paths.py`, `reconstruction.py` | Optional vectors, paths, and reconstruction |
| `verification.py`, `incremental_verification.py` | Independent checks for both solver modes |
| `postprocessing.py`, `printing.py`, `forest.py`, `validation.py` | Polynomial views, presentation, forests, and validation |

These modules live in `gvwzz/`. Build sources, checks, small examples, and guides
live separately in `native/`, `scripts/`, `tests/`, `examples/`, and `docs/`.

## Solver flow

The ordinary solver constructs the ring and module order, initializes labelled
polynomials P and syzygy signatures S, then processes J/G candidates in signature
order. It applies generation filters, exact-J admission, queue-time cover, head
reduction, optional tails, and insertion/index updates. Materialization is lazy.
Completed results retain P/S and optional certificates.

Incremental POT completes one input stage at a time and carries a minimal strong
polynomial basis forward. Its result is a polynomial-ideal basis, not a complete
original-module signature or syzygy basis. See [incremental mode](incremental.md).

## Native boundary

| Extension | Responsibility |
| --- | --- |
| `_native_reduction` | Sparse workspaces, GMP updates, reducer selection, and bulk loops |
| `_native_pairs` | Parent rows, exact-J admission, candidate heap/recipes, and frozen-pair kernels |
| `_native_indices` | Divisor/index kernels and versioned syzygy queries |
| `_native_cover` | Packed positive P-cover witness queries |
| `_native_queue_drain` | Bounded queue batches with cover rejection |

Edit maintained `.pyx`, `.pxi`, `.pxd`, or `.hpp` files, then rebuild. Generated
`.cpp` files in `native/generated/` are build inputs, not maintenance sources.
Keep all five units, `native/gvwzz_wide_key.hpp`, and shared declarations with
the source distribution. `build_native.py` delegates to `scripts/setup_native.py`,
forces all extensions to rebuild together, and checks imports in a fresh process.
This keeps the pairs/indices Cython interface consistent.

## Exact fast paths

GMP accumulators use arbitrary-precision integers. Small-coefficient shortcuts
first prove their operands fit; larger values retain exact operations. Monomial
encodings are separate from coefficient arithmetic.

G2/grevlex reducer metadata uses checked one-word or portable two-word encodings.
Signed-ratio bounds are 63/127 bits; guarded-divisibility bounds are 64/128 bits.
The accumulator has its own checked 63-bit monomial-key bound. Larger keys and
unsupported combinations use exact per-step or arbitrary-width paths. No
coordinate is truncated to make a fast path eligible.

Eligible bulk loops select reducers and update terms without entering Python
at each step. They return periodically for progress and signal handling.
Budgets, debug checks, certificates, and unsupported workspace metadata can
select other exact paths.

Parent loops specialize up to eight and 9–16 variables. Larger rings use dynamic
metadata and arbitrary-width Python-integer descriptors. Coordinates must meet
the selected encoding's bounds; otherwise the exact emitter is used. Variable
count has no fixed limit apart from available resources.

Generation top tests seek witnesses rather than selecting actual reducers.
Versioned syzygy caches invalidate stale negative answers. Exact-J admission
uses a Bloom screen only for definite negative answers; possible matches still
undergo exact lookup. Compact recipes retain source owners and materialize only
accepted candidates.

The native heap preserves exact signature/kind/tie/insertion order. Wider keys
retain their full integers. Packed cover is a positive-proof shortcut: an
unsuccessful query falls through to the complete exact criterion. Dynamic
coordinate vectors support arbitrary dimensions. Queue drains are bounded and
return for cooperative progress and interruption.

## Ownership, output, and diagnostics

Encoded GMP reducer payload has a shared 128 MiB LRU budget, including retained
integer limbs. Keys retain source polynomials to prevent identity reuse. Active
reductions own their required encoded terms independently of eviction. The
budget does not bound total process memory or accepted coefficient sizes.
Geometry caches are append-aware and bounded; cached divisor IDs retain the
configured reducer order. Lazy recipe contexts avoid a strong solver/queue cycle.

Availability is reported before solving; dispatch counters report actual usage
afterward. Wide/dynamic parent counts and bulk workspace counts are overlapping
subsets, not extra work. Logical candidate counters describe the algorithm;
heap/cache counters describe implementation activity.

Full source results are streamed to an atomic save before post-processing. A
final atomic save adds requested polynomial views. Timing separates solver work
from cleanup, serialization, post-processing, and display. See [usage](usage.md).

`run_tests.py` discovers regression checks and includes installation smoke tests.
`--differential` also compares fresh Python/native runs with
`tests/fixtures/reduction_reference.json`; no older release is needed. Release
checks and review scope are recorded in [release notes](release-notes.md).
