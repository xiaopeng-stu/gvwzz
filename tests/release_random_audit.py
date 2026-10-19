"""Reproducible bounded certificate audit: python -m tests.release_random_audit.

Run again with GVWZZ_NO_NATIVE=1 to compare exact output digests. The audit is
separate from default unittest discovery and fails on any unfinished solve.
"""
from __future__ import annotations

import hashlib
from itertools import combinations
import json
from random import Random
from time import perf_counter

from gvwzz import Options, compute
from gvwzz.core import lcm_monomials, subtract_monomials
from gvwzz.native_status import native_status
from gvwzz.verification import strong_remainder, verify_result


SEED = 20261019
ORDERS = ('pot', 'top', 'g1', 'g2', 'schreyer', 'block')
KERNELS = ('copy', 'heap', 'packed')


def make_cases():
    rng = Random(SEED)
    cases = []
    for _ in range(5):
        a, b, c = (rng.choice((-6, -4, -3, 2, 4, 6)) for _ in range(3))
        d, e = (rng.choice((-2, -1, 1, 2)) for _ in range(2))
        cases.append([f'({a})*x^2+({d})*y', f'({b})*y^2+({e})',
                      f'({c})*x*y+({d})*x'])
    return cases + [['-6*x', '0', '4*y', '-6*x'], [], ['0', '0']]


def main():
    cases = make_cases()
    checks = g_pairs = 0
    digest = hashlib.sha256()
    start = perf_counter()
    for polynomials in cases:
        for order in ORDERS:
            if not polynomials and order == 'block':
                continue  # A block order requires a nonempty positive partition.
            for kernel in KERNELS:
                context = (polynomials, order, kernel)
                result = compute(polynomials, ['x', 'y'], 'grevlex', order,
                    block_sizes=[len(polynomials)] if order == 'block' else None,
                    options=Options(track_representations=True, reduction_kernel=kernel,
                                    max_seconds=10, max_basis=150, max_candidates=100000),
                    show_basis=False, show_u=False)
                if result.status != 'completed':
                    raise AssertionError((context, result.message))
                report = verify_result(result, strong=True, mode='static', timeout=10,
                                       require_certificates=True)
                if not report['signature_basis_certified']:
                    raise AssertionError((context, report))
                backend = result.backend
                basis = [p.polynomial for p in result.basis]
                # Independently check ordinary G-polynomial remainders as well
                # as the certificate and signature criteria checked above.
                for f, g in combinations(basis, 2):
                    mu, a = backend.leading(f)
                    nu, b = backend.leading(g)
                    common = lcm_monomials(mu, nu)
                    left = subtract_monomials(common, mu)
                    right = subtract_monomials(common, nu)
                    s, t, _ = backend.bezout(a, b)
                    gpoly = (backend.multiply_term(f, s, left)
                             + backend.multiply_term(g, t, right))
                    if strong_remainder(gpoly, basis, backend, deadline=perf_counter()+10):
                        raise AssertionError((context, 'nonzero ordinary G remainder'))
                    g_pairs += 1
                state = {
                    'basis': [(p.signature.to_dict(), backend.serialize(p.polynomial))
                              for p in result.basis],
                    'syzygies': [s.to_dict() for s in result.syzygies],
                    'basis_vectors': [[backend.serialize(u) for u in vector]
                                      for vector in result.basis_representations],
                    'syzygy_vectors': [[backend.serialize(u) for u in vector]
                                       for vector in result.syzygy_representations],
                }
                digest.update(json.dumps(state, sort_keys=True).encode('utf-8'))
                checks += 1
    print(json.dumps(dict(
        status='passed', seed=SEED, cases=len(cases), module_orders=ORDERS,
        reduction_kernels=KERNELS, certified_computations=checks,
        ordinary_g_pairs_checked=g_pairs, exact_outputs_sha256=digest.hexdigest(),
        native_accelerators=native_status()['native_accelerators'],
        seconds=perf_counter()-start), indent=2))


if __name__ == '__main__':
    main()
