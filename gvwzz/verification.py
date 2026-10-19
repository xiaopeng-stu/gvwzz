"""Independent postcondition checks.

The verifier rebuilds its own flat or static query structures and never uses
the solver's dynamic caches. It checks the label criterion and, when a
representation certificate is present, checks every vector identity and signature.
Without a certificate, a passing label criterion is not a membership certificate.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from itertools import combinations, product
from math import gcd, isnan, prod
from time import perf_counter

from .validation import result_structure_errors

from .core import (
    Signature, add_monomials, divides_monomial, lcm_monomials,
    subtract_monomials,
)

class StaticChecks:
    def __init__(self, result, leading):
        self.b, self.order = result.backend, result.module_order
        self.P, self.S, self.leading = result.basis, result.syzygies, leading
        self.polynomial_buckets = defaultdict(list)
        self.signature_buckets = defaultdict(lambda: defaultdict(list))
        self.syzygy_buckets = defaultdict(lambda: defaultdict(list))
        for i,p in enumerate(self.P):
            self.polynomial_buckets[leading[i][0]].append(i)
            self.signature_buckets[p.signature.index][p.signature.monomial].append(i)
        for s in self.S:self.syzygy_buckets[s.index][s.monomial].append(s.coefficient)
        # Per-checker caches. They are bounded and do not retain solver state.
        self.poly_candidates = lru_cache(maxsize=8192)(self._poly_candidates)
        self.sig_candidates = lru_cache(maxsize=8192)(self._sig_candidates)
        self.syz_gcd = lru_cache(maxsize=8192)(self._syz_gcd)

    @staticmethod
    def _divisor_buckets(monomial, buckets):
        if not buckets:return
        if prod(x+1 for x in monomial) <= max(1, 2*len(buckets)):
            for divisor in product(*(range(x+1) for x in monomial)):
                values=buckets.get(divisor)
                if values is not None:yield values
        else:
            for divisor,values in buckets.items():
                if divides_monomial(divisor,monomial):yield values

    def _poly_candidates(self, W):
        return tuple(i for bucket in self._divisor_buckets(W,self.polynomial_buckets)
                     for i in bucket)

    def _sig_candidates(self, monomial, index):
        return tuple(i for bucket in self._divisor_buckets(monomial,self.signature_buckets[index])
                     for i in bucket)

    def _syz_gcd(self, monomial, index):
        acc=0
        for values in self._divisor_buckets(monomial,self.syzygy_buckets[index]):
            for c in values:acc=gcd(acc,c)
        return acc

    def cover(self, signature, W, cache=None):
        key=(signature.monomial,signature.index,W)
        if cache is not None and key in cache:
            acc=cache[key]
        else:
            acc=self.syz_gcd(signature.monomial,signature.index)
            # A syzygy-only cover is already a complete positive witness.
            # Do not cache an incomplete gcd of all possible contributions.
            if acc and signature.coefficient % acc == 0:
                return True
            Wkey=self.b.order_key(W)
            if acc!=1:
                for i in self.sig_candidates(signature.monomial,signature.index):
                    s=self.P[i].signature;M=self.leading[i][0]
                    scaled=add_monomials(M,subtract_monomials(signature.monomial,s.monomial))
                    if self.b.order_key(scaled)<Wkey:
                        acc=gcd(acc,s.coefficient)
                        if acc==1:break
            if cache is not None:cache[key]=acc
        return bool(acc and signature.coefficient%acc==0)

    def top(self, signature, W, coefficient):
        target=self.order.key(signature)
        for i in self.poly_candidates(W):
            M,c=self.leading[i]
            if coefficient%c:continue
            s=self.P[i].signature
            scaled=add_monomials(s.monomial,subtract_monomials(W,M))
            if self.order.key_monomial(scaled,s.index)<=target:return True
        return False

def naive_cover(signature, W, basis, syzygies, backend, leading=None, cache=None) -> bool:
    cache_key = (signature.monomial, signature.index, W)
    if cache is not None and cache_key in cache:
        acc = cache[cache_key]
        return bool(acc and signature.coefficient % acc == 0)
    acc = 0
    for s in syzygies:
        if s.index == signature.index and divides_monomial(s.monomial, signature.monomial):
            acc = gcd(acc, s.coefficient)
    for j, p in enumerate(basis):
        s = p.signature
        if s.index != signature.index or not divides_monomial(s.monomial, signature.monomial):
            continue
        M, _ = leading[j] if leading is not None else backend.leading(p.polynomial)
        scaled = add_monomials(M, subtract_monomials(signature.monomial, s.monomial))
        if backend.order_key(scaled) < backend.order_key(W):
            acc = gcd(acc, s.coefficient)
    if cache is not None:
        cache[cache_key] = acc
    return bool(acc and signature.coefficient % acc == 0)


def naive_top_exists(signature, W, coefficient, basis, backend, order, leading=None) -> bool:
    for j, p in enumerate(basis):
        M, c = leading[j] if leading is not None else backend.leading(p.polynomial)
        if coefficient % c or not divides_monomial(M, W):
            continue
        scaled = p.signature.multiply(1, subtract_monomials(W, M))
        if order.compare(scaled, signature) <= 0:
            return True
    return False


def strong_remainder(f, polynomials, backend, max_steps: int = 200000, deadline=None):
    """Classical strong reduction over ZZ, WITHOUT primitive/content division."""
    if max_steps is not None and max_steps < 0:
        raise ValueError('Verification max_steps must be nonnegative or None')
    r = backend.zero
    f = f.copy()
    leads = [(g, *backend.leading(g)) for g in polynomials if g]
    steps = 0
    while f:
        steps += 1
        if (max_steps not in (None, 0) and steps > max_steps) or (deadline is not None and perf_counter() > deadline):
            raise TimeoutError('Verification reduction budget exceeded')
        W, c = backend.leading(f)
        for g, M, b in leads:
            if c % b == 0 and divides_monomial(M, W):
                f = f - backend.multiply_term(g, c // b, subtract_monomials(W, M))
                break
        else:
            term = backend.term(c, W)
            r = r + term
            f = f - term
    return r


def verify_result(result, strong: bool = False, timeout: float | None = 60.0, mode: str = 'naive',
                  max_reduction_steps: int | None = 200000,
                  require_certificates: bool = False) -> dict:
    if result.options.incremental_pot:
        from .incremental_verification import verify_incremental_result
        return verify_incremental_result(result, timeout=timeout, mode=mode,
                                         max_reduction_steps=max_reduction_steps,
                                         require_certificates=require_certificates)
    if mode not in ('naive', 'static'):
        raise ValueError('Verification mode must be naive or static')
    if timeout is not None and (timeout < 0 or isnan(timeout)):
        raise ValueError('Verification timeout must be nonnegative or None')
    if max_reduction_steps is not None and max_reduction_steps < 0:
        raise ValueError('Verification reduction budget must be nonnegative or None')
    start = perf_counter()
    deadline = None if timeout in (None, 0) else start + timeout
    errors: list[str] = []
    counts = {'j_pairs_checked': 0, 'g_pairs_checked': 0, 'syzygy_pairs_checked': 0,
              'ordinary_s_pairs_checked': 0, 'inputs_checked': 0,
              'basis_vectors_checked': 0, 'syzygy_vectors_checked': 0}
    membership_verified = False
    common = {"membership_verified": False, "signature_basis_certified": False,
              "verification_level": "unverified", "warnings": [],
              "certificates_required": bool(require_certificates)}
    if result.status != 'completed':
        return {'passed': False, 'status': 'not_completed', 'errors': [result.message],
                'counts': counts, 'mode': mode, 'seconds': perf_counter() - start, **common}
    errors = result_structure_errors(result)
    if errors:
        return {'passed': False, 'status': 'invalid_structure', 'errors': errors,
                'counts': counts, 'mode': mode, 'seconds': perf_counter() - start, **common}
    has_certificate = (result.basis_representations is not None and result.syzygy_representations is not None)
    if require_certificates and not has_certificate:
        return {'passed': False, 'status': 'missing_certificate',
                'errors': ['Exact representation vectors are required. Recompute with track_representations=True.'],
                'counts': counts, 'mode': mode, 'seconds': perf_counter() - start, **common}
    checked_scope = {
        "verification_level": "certificate_and_criteria" if has_certificate else "criteria_only",
        "certificates_required": bool(require_certificates),
        "warnings": [] if has_certificate else [
            "No representation certificate: passing these checks does NOT establish "
            "that the output belongs to the input ideal or graph module. "
            "--strong only adds ordinary integer S checks. Compute with --certificates "
            "and recheck with --require-certificates for membership/signature certification."
        ],
    }
    b, order, P, S = result.backend, result.module_order, result.basis, result.syzygies
    leading = [b.leading(p.polynomial) for p in P]
    # Independent flat scans with per-result memoization, not the solver's
    # tries, dynamic indices, G filter, or cover cache.
    cover_memo, top_memo = {}, {}
    if mode == 'static':
        checker = StaticChecks(result, leading)
    else:
        checker = None
    try:
        if has_certificate:
            def check_vector(vector, polynomial, signature, description):
                value = b.zero
                sigs = []
                for component, (u, f) in enumerate(zip(vector, result.inputs)):
                    if deadline is not None and perf_counter() > deadline:
                        raise TimeoutError('Representation verification timeout')
                    value = value + u * f
                    if u:
                        M, c = b.leading(u)
                        sigs.append(Signature(c, M, component))
                actual = max(sigs, key=order.key) if sigs else None
                if value != polynomial:
                    errors.append(f'{description}: u*f^T does not equal the polynomial part')
                if actual != signature:
                    errors.append(f'{description}: leading module term does not equal the stored signature')
            for i, (p, vector) in enumerate(zip(P, result.basis_representations)):
                check_vector(vector, p.polynomial, p.signature, f'P[{i}]')
                counts['basis_vectors_checked'] += 1
            for i, (sig, vector) in enumerate(zip(S, result.syzygy_representations)):
                check_vector(vector, b.zero, sig, f'S[{i}]')
                counts['syzygy_vectors_checked'] += 1
            membership_verified = not errors
            if errors:
                return {'passed': False, 'status': 'invalid_certificate', 'errors': errors,
                        'counts': counts, 'mode': mode, 'seconds': perf_counter() - start, **common}
        for i in range(len(result.inputs)):
            e = Signature(1, b.zero_monomial, i)
            if not any(s.divides(e) for s in S) and not any(p.signature.divides(e) for p in P):
                errors.append(f'Background hypothesis fails at e{i+1}')
        for i, p in enumerate(P):
            if not p.polynomial:
                errors.append(f'P[{i}] has zero polynomial part')
        for (i, p), (j, q) in combinations(enumerate(P), 2):
            if deadline is not None and perf_counter() > deadline:
                raise TimeoutError('Verification timeout')
            M, c = leading[i]
            N, d = leading[j]
            L = lcm_monomials(M, N)
            u, v = subtract_monomials(L, M), subtract_monomials(L, N)
            s, t = p.signature.multiply(1, u), q.signature.multiply(1, v)
            cmp = order.compare(s, t)
            if cmp:
                ell = abs((c // gcd(c, d)) * d)
                jsig = p.signature.multiply(ell // c, u) if cmp > 0 else q.signature.multiply(ell // d, v)
                counts['j_pairs_checked'] += 1
                covered = (checker.cover(jsig, L, cover_memo) if checker is not None else
                           naive_cover(jsig, L, P, S, b, leading, cover_memo))
                if not covered:
                    errors.append(f'Uncovered J-pair ({i},{j}), sig={jsig}, lm={L}')
            # By the prescribed noncancelling choice, actual G signature monomial
            # is max(s,t); its coefficient is irrelevant to this top test.
            gsig = s if cmp >= 0 else t
            counts['g_pairs_checked'] += 1
            gcoefficient = gcd(c,d)
            if checker is not None and gcoefficient in (abs(c),abs(d)):
                # This is an explicit parent witness, not an omitted condition:
                # LM(parent)|L, LC(parent)|gcd(c,d), scaled signature<=max(s,t).
                has_top = True
            else:
                top_key=(gsig.monomial,gsig.index,L,gcoefficient)
                if top_key not in top_memo:
                    top_memo[top_key] = (checker.top(gsig,L,gcoefficient) if checker is not None else
                                         naive_top_exists(gsig,L,gcoefficient,P,b,order,leading))
                has_top = top_memo[top_key]
            if not has_top:
                errors.append(f'G-polynomial has no P top reducer ({i},{j})')
            if strong:
                ell = abs((c // gcd(c, d)) * d)
                spoly = b.multiply_term(p.polynomial, ell // c, u) - b.multiply_term(q.polynomial, ell // d, v)
                counts['ordinary_s_pairs_checked'] += 1
                if strong_remainder(spoly, [r.polynomial for r in P], b, deadline=deadline, max_steps=max_reduction_steps):
                    errors.append(f'Ordinary integer S-polynomial has nonzero remainder ({i},{j})')
        for i, s in enumerate(S):
            for j, t in enumerate(S[:i]):
                if s.divides(t) or t.divides(s):
                    errors.append(f'S is not divisibility-minimal at ({j},{i})')
                if s.index != t.index:
                    continue
                counts['syzygy_pairs_checked'] += 1
                h = Signature(gcd(s.coefficient, t.coefficient),
                              lcm_monomials(s.monomial, t.monomial), s.index)
                if not any(z.divides(h) for z in S):
                    errors.append(f'S is not gcd-complete at ({j},{i})')
            if deadline is not None and perf_counter() > deadline:
                raise TimeoutError('Verification timeout')
        for f in result.inputs:
            counts['inputs_checked'] += 1
            if strong_remainder(f, [r.polynomial for r in P], b, deadline=deadline, max_steps=max_reduction_steps):
                errors.append('An input polynomial has nonzero strong remainder')
    except TimeoutError as exc:
        return {'passed': False, 'status': 'verification_limit', 'errors': errors + [str(exc)],
                'counts': counts, 'mode': mode, 'seconds': perf_counter() - start,
                'membership_verified': membership_verified, 'signature_basis_certified': False,
                **checked_scope}
    return {'passed': not errors, 'status': 'passed' if not errors else 'failed',
            'errors': errors, 'counts': counts, 'mode': mode, 'seconds': perf_counter() - start,
            'membership_verified': membership_verified,
            'signature_basis_certified': membership_verified and not errors,
            **checked_scope,
            'scope': ('Exact u-vector identities and signatures plus label criterion checks' if has_certificate else
                      'Label criterion only. Membership in the input graph module has NOT been verified')
                     + ('; ordinary integer S checks also run' if strong else '')}
