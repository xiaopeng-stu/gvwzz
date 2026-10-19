"""Replay saved generation paths without repeating polynomial reduction.

For each final S root, propagate scalar polynomial weights backwards through
its ancestor DAG. At an input leaf, accumulate the weight in that input's
coordinate. No full u-vector for any P or internal node is materialized.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from heapq import heappop, heappush
from time import perf_counter
from typing import Callable, Iterable

from .core import Signature, add_monomials, exact_integer
from .paths import GenerationPaths


class ReconstructionError(ValueError):
    pass


def _paths(result, *, allow_incremental_prefix: bool = False) -> GenerationPaths:
    if result.status != "completed" and not (allow_incremental_prefix and result.options.incremental_pot):
        raise ReconstructionError("Finish the signature basis before replay; partial results are not certified")
    graph = getattr(result, "generation_paths", None)
    if not isinstance(graph, GenerationPaths):
        raise ReconstructionError(
            "No saved generation paths. Rerun with --record-paths / Options(record_paths=True). "
            "Legacy signature-only JSON cannot recover the original path; "
            "no re-reduction fallback is used.")
    graph.validate(result)
    return graph


def vector_signature(vector, backend, order):
    terms = []
    for i, p in enumerate(vector):
        if p:
            m, c = backend.leading(p)
            terms.append(Signature(c, m, i))
    return max(terms, key=order.key) if terms else None


def check_syzygy(vector, signature, result, *, identity=True) -> None:
    b = result.backend
    if len(vector) != len(result.inputs):
        raise ReconstructionError("Reconstructed vector has wrong input dimension")
    if vector_signature(vector, b, result.module_order) != signature:
        raise ReconstructionError("Reconstructed vector has the wrong complete signature")
    if identity:
        image = b.zero
        for u, f in zip(vector, result.inputs):
            if u and f:
                image = image + u * f
        if image:
            raise ReconstructionError("Reconstructed vector is not a syzygy: u*f^T != 0")


def _expand_root(result, graph, root, counts, callback=None):
    """Iterative reverse propagation. Equal-node weights cancel before expansion."""
    b = result.backend
    pending = {root: b.one.copy()}
    queue = [-root]
    scheduled = {root}
    pending_terms = 1
    vector = [b.zero.copy() for _ in result.inputs]
    counts['pending_nodes_peak'] = max(counts['pending_nodes_peak'], 1)
    counts['pending_terms_peak'] = max(counts['pending_terms_peak'], 1)
    while queue:
        node_id = -heappop(queue)
        scheduled.remove(node_id)
        weight = pending.pop(node_id, None)
        if weight is None:
            continue
        pending_terms -= len(weight)
        node = graph.nodes[node_id]
        counts['ancestor_nodes_visited'] += 1
        if callback is not None and counts['ancestor_nodes_visited'] % 256 == 0:
            callback(dict(counts))
        if node.kind == 'input':
            i = node.input_index
            vector[i] = vector[i] + weight
            continue
        for e in node.edges:
            contribution = b.multiply_term(weight, e.coefficient, e.monomial)
            if e.polynomial_index is not None:
                contribution = contribution * result.basis[e.polynomial_index].polynomial
                counts['polynomial_reference_products'] += 1
            counts['edge_products'] += 1
            if callback is not None and counts['edge_products'] % 256 == 0:
                callback(dict(counts))
            if not contribution:
                continue
            old = pending.get(e.parent)
            if old is not None:
                pending_terms -= len(old)
                contribution = old + contribution
            if contribution:
                pending[e.parent] = contribution
                pending_terms += len(contribution)
                if e.parent not in scheduled:
                    heappush(queue, -e.parent)
                    scheduled.add(e.parent)
            else:
                pending.pop(e.parent, None)
                counts['cancelled_weights'] += 1
            counts['pending_nodes_peak'] = max(counts['pending_nodes_peak'], len(pending))
            counts['pending_terms_peak'] = max(counts['pending_terms_peak'], pending_terms)
    return tuple(vector)


@dataclass(frozen=True)
class ReplayedSyzygy:
    index: int
    signature: Signature
    vector: tuple


class SyzygyReplay:
    """One-pass iterator. Results can be written one at a time without retaining S."""
    def __init__(self, result, *, indices: Iterable[int] | None = None,
                 verify: bool = False, progress: Callable[[dict], None] | None = None):
        if type(verify) is not bool:
            raise ValueError("verify must be a boolean")
        if result.options.incremental_pot:
            raise ValueError('Incremental POT retires stage syzygies; global syzygy reconstruction is unavailable. '
                             'Use verify-paths for ideal membership certification or run the default signature mode.')
        self.result = result
        self.graph = _paths(result)
        self.indices = tuple(range(len(result.syzygies))) if indices is None else tuple(indices)
        self.indices = tuple(exact_integer(i, "syzygy index") for i in self.indices)
        if any(not 0 <= i < len(result.syzygies) for i in self.indices):
            raise ValueError("Syzygy index outside final S")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("Syzygy indices must not be repeated")
        self.verify, self.progress = verify, progress
        self._started = False
        self.statistics = {
            'method': 'generation-path-backward-propagation',
            'stored_nodes': len(self.graph.nodes),
            'stored_edges': sum(len(n.edges) for n in self.graph.nodes),
            'online_reduction_steps_recorded': self.graph.reduction_steps_recorded,
            'basis_vectors_expanded': 0,
            'internal_u_vectors_expanded': 0,
            'post_reduction_steps': 0,
            'post_reducer_searches': 0,
            'syzygies_reconstructed': 0,
            'output_terms': 0,
            'signature_checks': 0,
            'identity_checks': 0,
            'replay_seconds': 0.0,
            'verification_seconds': 0.0,
        }

    def __iter__(self):
        if self._started:
            raise RuntimeError("SyzygyReplay is single-use; create another instance to replay again")
        self._started = True
        counts = Counter()
        for i in self.indices:
            self.statistics['current_syzygy_index'] = i
            if self.progress:
                self.progress(dict(self.statistics))
            start = perf_counter()
            vector = _expand_root(self.result, self.graph, self.graph.syzygy_roots[i], counts,
                                  (lambda c: self.progress({**self.statistics, **c})) if self.progress else None)
            self.statistics['replay_seconds'] += perf_counter() - start
            start = perf_counter()
            check_syzygy(vector, self.result.syzygies[i], self.result, identity=self.verify)
            self.statistics['verification_seconds'] += perf_counter() - start
            self.statistics.update(counts)
            self.statistics['signature_checks'] += 1
            self.statistics['identity_checks'] += int(self.verify)
            self.statistics['syzygies_reconstructed'] += 1
            self.statistics['output_terms'] += sum(len(f) for f in vector)
            yield ReplayedSyzygy(i, self.result.syzygies[i], vector)
        self.statistics.pop('current_syzygy_index', None)
        if self.progress:
            self.progress(dict(self.statistics))


@dataclass
class PathReconstruction:
    result: object
    records: list[ReplayedSyzygy]
    statistics: dict

    @property
    def syzygies(self):
        return [record.vector for record in self.records]

    def to_dict(self) -> dict:
        b = self.result.backend
        return {
            'format': 'gvwzz-path-syzygies-v1',
            'status': 'completed',
            'variables': list(b.names),
            'coefficient_ring': 'ZZ',
            'polynomial_order': b.order_name,
            **self.result.module_order.metadata(),
            'input_polynomials': [b.serialize(f) for f in self.result.inputs],
            'source_syzygy_count': len(self.result.syzygies),
            'syzygies': [dict(index=r.index + 1, signature=r.signature.to_dict(),
                              vector=[b.serialize(f) for f in r.vector]) for r in self.records],
            'statistics': dict(self.statistics),
            'signature_basis_certified': False,
            'verification_scope': 'Each output complete signature checked; zero identities checked only when identity_checks equals the output count. Completeness requires basis-criterion verification.',
        }


def reconstruct_syzygies(result, *, indices=None, verify: bool = False, progress=None) -> PathReconstruction:
    """Replay final S only; never recompute GVW or construct any intermediate u."""
    replay = SyzygyReplay(result, indices=indices, verify=verify, progress=progress)
    records = list(replay)
    return PathReconstruction(result, records, dict(replay.statistics))


def verify_generation_paths(result, *, criteria: bool = True, strong: bool = False,
                            timeout: float | None = 60.0, mode: str = 'static',
                            max_reduction_steps: int | None = 200000) -> dict:
    """Verify scalar images and complete leading signatures, without full u.

    The noncancelling leading-signature identity is checked at EVERY node.
    The principal identity is shortcut only after verifying its exact shape.
    Other polynomial images are evaluated by the stored linear operations.
    Optional existing static checks establish the final basis criteria.
    """
    start = perf_counter()
    report = dict(passed=False, status='invalid_paths', membership_verified=False,
                  signature_basis_certified=False, errors=[], nodes_checked=0,
                  verification_level='generation_paths', seconds=0.0)
    if result.options.incremental_pot:
        report['strong_basis_certified'] = False
        report['scope'] = 'Membership of compacted ideal-basis records only until ordinary S/G criteria pass'
    try:
        if timeout is not None and (timeout < 0 or timeout != timeout):
            raise ValueError("Verification timeout must be nonnegative or None")
        deadline = start + timeout if timeout else None
        graph = _paths(result, allow_incremental_prefix=not criteria)
        b, order = result.backend, result.module_order
        images, signatures = [], []
        p_roots = {}
        for i, root in enumerate(graph.basis_roots):
            p_roots.setdefault(root, []).append(i)
        s_roots = set(graph.syzygy_roots)
        for j, node in enumerate(graph.nodes):
            if deadline is not None and perf_counter() > deadline:
                raise TimeoutError("Generation-path verification timeout")
            if node.kind == 'input':
                value = result.inputs[node.input_index]
                actual = Signature(1, b.zero_monomial, node.input_index)
            else:
                candidates = []
                for e in node.edges:
                    parent_sig = signatures[e.parent]
                    c, mu = e.coefficient, e.monomial
                    if e.polynomial_index is not None:
                        m, lc = b.leading(result.basis[e.polynomial_index].polynomial)
                        c *= lc
                        mu = add_monomials(mu, m)
                    candidates.append(parent_sig.multiply(c, mu))
                highest = max(candidates, key=order.key)
                c = sum(s.coefficient for s in candidates
                        if s.index == highest.index and s.monomial == highest.monomial)
                if not c:
                    raise ReconstructionError(f"Node {j}: highest signature cancels; path not certified")
                actual = Signature(c, highest.monomial, highest.index)
                # A principal identity needs no large polynomial products.
                principal = False
                if node.kind == 'principal' and len(node.edges) == 2:
                    left, right = node.edges
                    pi, qi = left.polynomial_index, right.polynomial_index
                    principal = (pi is not None and qi is not None and
                        left.coefficient == 1 and right.coefficient == -1 and
                        left.monomial == right.monomial == b.zero_monomial and
                        left.parent == graph.basis_roots[qi] and
                        right.parent == graph.basis_roots[pi])
                value = b.zero
                if not principal:
                    for e in node.edges:
                        if not images[e.parent]:
                            continue
                        term = b.multiply_term(images[e.parent], e.coefficient, e.monomial)
                        if e.polynomial_index is not None:
                            term = term * result.basis[e.polynomial_index].polynomial
                        value = value + term
            if actual != node.signature:
                raise ReconstructionError(f"Node {j}: local complete signature mismatch")
            for i in p_roots.get(j, ()):
                if value != result.basis[i].polynomial:
                    raise ReconstructionError(f"P[{i}]: local path polynomial image mismatch")
            if j in s_roots and value:
                raise ReconstructionError(f"S root {j}: local path image is not zero")
            images.append(value)
            signatures.append(actual)
            report['nodes_checked'] += 1
        report['membership_verified'] = True
        report['passed'] = True
        report['status'] = 'paths_verified'
        if result.options.incremental_pot and result.status != 'completed':
            report['status'] = 'prefix_paths_verified'
            report['scope'] = 'Only membership of saved completed-prefix records; NOT a basis of the full input ideal'
        if criteria:
            from .verification import verify_result
            remaining = None if deadline is None else deadline - perf_counter()
            if remaining is not None and remaining <= 0:
                raise TimeoutError("Verification timeout before final criteria")
            checked = verify_result(result, strong=strong, mode=mode,
                                    timeout=remaining, max_reduction_steps=max_reduction_steps)
            report['criteria'] = checked
            report['passed'] = checked['passed']
            report['signature_basis_certified'] = checked['passed'] and not result.options.incremental_pot
            if result.options.incremental_pot:
                report['strong_basis_certified'] = checked['passed']
                report['scope'] = 'Path membership and ordinary S/G ideal criteria; not a full signature basis'
                checked['membership_supplied_by'] = 'enclosing generation-path verification'
                checked['warnings'] = []
                report['status'] = 'path_certificate_and_ideal_criteria' if checked['passed'] else 'failed_criteria'
            else:
                report['status'] = 'path_certificate_and_criteria' if checked['passed'] else 'failed_criteria' 
            report['errors'].extend(checked['errors'])
    except (ValueError, TypeError, IndexError, KeyError, TimeoutError) as exc:
        report['status'] = 'timeout' if isinstance(exc, TimeoutError) else 'invalid_paths'
        report['passed'] = False
        report['signature_basis_certified'] = False
        if result.options.incremental_pot:
            report['strong_basis_certified'] = False
        report['errors'].append(str(exc))
    report['seconds'] = perf_counter() - start
    return report
