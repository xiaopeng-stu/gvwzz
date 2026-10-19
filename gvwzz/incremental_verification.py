"""Independent ordinary strong-basis checks for compacted incremental results.

No original-module signature criterion is asserted on a compressed P or retired
S. Membership vectors or generation paths establish P subset I. Input zero
remainders establish I subset (P). Ordinary integer S and Bezout G remainders
establish that P is a strong basis. All arithmetic is over ZZ.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from math import gcd, isnan
from time import perf_counter

from .core import Signature, divides_monomial, lcm_monomials, subtract_monomials
from .validation import result_structure_errors


def verify_incremental_result(result, *, timeout=60.0, mode='naive',
                              max_reduction_steps=200000, require_certificates=False):
    if mode not in ('naive', 'static'):
        raise ValueError('Verification mode must be naive or static')
    if timeout is not None and (timeout < 0 or isnan(timeout)):
        raise ValueError('Verification timeout must be nonnegative or None')
    if max_reduction_steps is not None and max_reduction_steps < 0:
        raise ValueError('Verification reduction budget must be nonnegative or None')
    start = perf_counter()
    deadline = None if timeout in (None, 0) else start + timeout
    counts = dict(j_pairs_checked=0, g_pairs_checked=0, syzygy_pairs_checked=0,
                  ordinary_s_pairs_checked=0, ordinary_g_pairs_checked=0,
                  inputs_checked=0, basis_vectors_checked=0, syzygy_vectors_checked=0,
                  ordinary_reduction_steps=0, minimal_leading_terms_checked=0)
    report = dict(passed=False, status='invalid_structure', errors=[], counts=counts,
                  mode=mode, seconds=0.0, membership_verified=False,
                  signature_basis_certified=False, strong_basis_certified=False,
                  verification_level='ordinary_ideal_criteria', certificates_required=bool(require_certificates),
                  scope='Strong polynomial ideal basis only; not a global signature or syzygy basis',
                  warnings=[])

    def finish(status, passed=False):
        report['status'], report['passed'] = status, passed
        report['strong_basis_certified'] = passed and report['membership_verified']
        report['seconds'] = perf_counter()-start
        return report

    if result.status != 'completed':
        report['errors'].append(result.message)
        return finish('not_completed')
    report['errors'] = result_structure_errors(result)
    if result.module_order.name != 'pot' or result.module_order.positions != tuple(range(len(result.inputs))):
        report['errors'].append('Incremental result requires natural-order POT')
    if result.syzygies:
        report['errors'].append('Incremental result must not claim retained global syzygy signatures')
    if report['errors']:
        return finish('invalid_structure')
    has_cert = result.basis_representations is not None and result.syzygy_representations is not None
    if require_certificates and not has_cert:
        report['errors'].append('Full representation vectors are required; compute with --certificates, or use verify-paths on --record-paths output')
        return finish('missing_certificate')
    b, order = result.backend, result.module_order
    P = [p.polynomial for p in result.basis]
    leading = [b.leading(f) for f in P]

    def tick():
        if deadline is not None and perf_counter() > deadline:
            raise TimeoutError('Incremental ideal verification timeout')

    def find_monomial_divisors(mu):
        return tuple(i for i, (nu, _) in enumerate(leading) if divides_monomial(nu, mu))
    candidates = lru_cache(maxsize=8192)(find_monomial_divisors) if mode == 'static' else find_monomial_divisors

    def remainder(f):
        g, rem, steps = f.copy(), b.zero, 0
        while g:
            tick()
            steps += 1
            if max_reduction_steps not in (None, 0) and steps > max_reduction_steps:
                raise TimeoutError('Verification reduction budget exceeded (per normal-form call)')
            mu, c = b.leading(g)
            for j in candidates(mu):
                nu, d = leading[j]
                if c % d == 0:
                    g = g - b.multiply_term(P[j], c//d, subtract_monomials(mu, nu))
                    counts['ordinary_reduction_steps'] += 1
                    break
            else:
                t = b.term(c, mu)
                rem = rem + t
                g = g - t
        return rem

    try:
        if has_cert:
            for j, (p, vector) in enumerate(zip(result.basis, result.basis_representations)):
                tick()
                value, signatures = b.zero, []
                for k, (u, f) in enumerate(zip(vector, result.inputs)):
                    tick()
                    value += u*f
                    if u:
                        mu, c = b.leading(u)
                        signatures.append(Signature(c, mu, k))
                actual = max(signatures, key=order.key) if signatures else None
                if value != p.polynomial or actual != p.signature:
                    report['errors'].append(f'P[{j}]: original-input identity or complete signature mismatch')
                counts['basis_vectors_checked'] += 1
            if report['errors']:
                return finish('invalid_certificate')
            report['membership_verified'] = True
            report['verification_level'] = 'ideal_certificate_and_criteria'
        else:
            report['warnings'].append('No vector certificate in this check. Ideal criteria alone do not establish membership; use --certificates or verify-paths.')
        for j, (mu, c) in enumerate(leading):
            tick()
            if c <= 0:
                report['errors'].append(f'P[{j}]: minimal output requires a positive leading coefficient')
            if any(k != j and c % leading[k][1] == 0 for k in candidates(mu)):
                report['errors'].append(f'P[{j}]: complete leading term is divisible by another output leading term')
            counts['minimal_leading_terms_checked'] += 1
        if report['errors']:
            return finish('not_minimal')
        for (i, f), (j, g) in combinations(enumerate(P), 2):
            tick()
            mu, a = leading[i]
            nu, c = leading[j]
            L = lcm_monomials(mu, nu)
            up, uq = subtract_monomials(L, mu), subtract_monomials(L, nu)
            d = gcd(a, c)
            ell = abs((a//d)*c)
            spoly = b.multiply_term(f, ell//a, up) - b.multiply_term(g, ell//c, uq)
            counts['ordinary_s_pairs_checked'] += 1
            if remainder(spoly):
                report['errors'].append(f'Ordinary integer S-polynomial has nonzero remainder ({i},{j})')
                return finish('failed_criteria')
            x, y, _ = b.bezout(a, c)
            gpoly = b.multiply_term(f, x, up) + b.multiply_term(g, y, uq)
            counts['ordinary_g_pairs_checked'] += 1
            counts['g_pairs_checked'] += 1
            if remainder(gpoly):
                report['errors'].append(f'Ordinary integer G-polynomial has nonzero remainder ({i},{j})')
                return finish('failed_criteria')
        for i, f in enumerate(result.inputs):
            tick()
            counts['inputs_checked'] += 1
            if remainder(f):
                report['errors'].append(f'Input {i+1} has nonzero strong remainder')
                return finish('failed_criteria')
        return finish('ideal_certificate_and_criteria' if has_cert else 'ideal_criteria_only', True)
    except TimeoutError as exc:
        report['errors'].append(str(exc))
        return finish('timeout')
    finally:
        if mode == 'static':
            candidates.cache_clear()
