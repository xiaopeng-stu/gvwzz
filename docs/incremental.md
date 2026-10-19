# Incremental POT mode

```powershell
python -m gvwzz run --case eco3 --incremental-pot --module-order pot --show-minimal-lt --no-show-basis --no-show-u --no-show-forest --progress 30 --output results/eco3_incremental.json
```

`--incremental-pot` is off by default. Its inverse is `--no-incremental-pot`.
Omitting `--module-order` in this mode selects POT. An explicit different module
order, signature blocks, or nontrivial position permutation is rejected. The
input order is unchanged; `lex`, `grlex`, and `grevlex` remain supported.

```python
from gvwzz import Options, compute

result = compute(["2*x", "3*y", "x*y-1"], ["x", "y"],
                 module_order="pot", options=Options(incremental_pot=True),
                 show_basis=False)
```

## Stage invariant

Let `I_i = (f_1,...,f_i)`. Stage i starts with a completed minimal strong basis H
of `I_(i-1)` and activates `f_i`. H is frozen within the stage. Its retained
labels have genuine signatures in lower original-input positions. Natural POT
makes all their polynomial multiples regular reducers for the current position.

The relative stage module is
`M_i = {(u,v) : v - u*f_i belongs to I_(i-1)}`. The signature records the leading
term of u; the zero-signature part is represented by H. This is a relative-stage
computation, not the unchanged global signature criterion after arbitrary row
deletion.

After exhausting the active queue, the stage takes the polynomial projection,
removes leading-term multiples, and normalizes unit signs. It carries that minimal
strong basis and requested representations forward. Tails follow the selected
options; no extra canonical reduced-basis calculation is done at the boundary.
Coefficient divisibility remains essential: `(2*x,3*y)` needs `x*y` in its minimal
strong basis. No rationalization or primitive-part division occurs.

## Work retained and avoided

- Future inputs are inactive, and completed frozen/frozen pairs are not regenerated.
  New rows still pair with frozen and earlier current rows. J and G remain separate.
- The current syzygy component starts with `LT(h)*e_i` for h in H. The identity
  `h*e_i - f_i*u_h` justifies these seeds; their divisibility completeness follows
  from H being a strong basis. New nontrivial coefficient-gcd completion remains.
- Principal current-component syzygies lie in the completed prefix ideal and are
  covered by these coefficient-aware seeds. This shortcut is confined to the
  incremental invariant; ordinary mode still processes principal syzygies.
- Frozen lower-position rows do not enter current cover queries. Current cover
  retains coefficient combinations; it is not replaced by single-divisor cover.
- Completed boundaries retire old S, cover caches, queues, and J-dedup state.

## Output and certification scope

A completed result contains a minimal strong basis of the polynomial ideal.
Surviving labels can be checked against the original inputs, but are not a
complete signature basis of the original graph module. Global syzygies and the
global ratio-layer forest are not returned.

The result reports `solver_mode = incremental-pot`,
`solver_claims_signature_basis = false`, and `signature_basis_certified = false`.
`strong_basis_certified` becomes true only after membership and ideal criteria
pass. Empty saved S means not returned, not a zero original syzygy module.
`--show-forest` and `--reconstruct-syzygies` are rejected in this mode.

Vectors use all original input coordinates. Paths use term-multiplier edges and
retain historical identity nodes across compression, so optional certificate
storage can grow even if the live basis shrinks.

```powershell
python -m gvwzz run --input examples/paper.json --incremental-pot --certificates --verify --verify-mode static --verify-timeout 0 --verify-max-steps 0 --no-show-basis --no-show-u --output results/inc_vectors.json
python -m gvwzz verify results/inc_vectors.json --require-certificates --verify-mode static --timeout 0 --max-steps 0
```

Replace `--certificates` with `--record-paths` to record paths, then use
`verify-paths` instead of `verify`. Incremental verification checks integer
S- and Bezout G-polynomial zero remainders, original inputs, and the minimal
leading-term antichain. `--strong` is redundant here. Without vectors or paths,
ideal criteria alone do not independently prove membership. Verification time
is outside solver time.

## Progress and limits

Progress adds `stage=i/m`, frozen size, and current S size. Stage completion shows
P before/after minimization, retired S, and elapsed time. Candidate serial numbers
are stage-local; use `(stage, serial)` to identify a candidate. Candidate/step
counts are cumulative. Peak P/S/queue sizes are live-stage peaks. Final P is the
compacted basis, not historical insertions. JSON retains per-stage statistics.
Boundary minimization is included in total solver time.

Budgets apply to the entire run and are not reset at stages. `--max-basis` limits
active-stage P, including frozen rows. A limit or interruption returns only the
last completed prefix basis and labels it incomplete for the full input. Saved
JSON cannot restart an in-flight stage.

Compare with ordinary POT on identical inputs and settings. Compression changes
execution, so intermediate P/S and candidate counts need not match; compare final
canonical polynomial bases or independently verify the ideals. No speedup is
promised for every input: minimization and frozen-index rebuilding have costs.
