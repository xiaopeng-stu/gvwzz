"""Exact algebra and lifetime checks for the bounded native reducer cache."""
import unittest

from gvwzz.polynomial import IntegerPolynomialRing
import gvwzz.solver as solver_module
from tests.native_bulk_checks import fixture, records, reduce
from tests.native_wide_reduction_checks import lifted_fixture

try:
    from gvwzz import _native_reduction as native_module
except ImportError:
    native_module = None


NATIVE = native_module is not None and solver_module._native_reduce_working is not None
GMP = NATIVE and native_module.COEFFICIENT_BACKEND == "gmp"


@unittest.skipUnless(GMP, "optional GMP native reducer cache is unavailable")
class NativeTermCacheChecks(unittest.TestCase):
    def apply(self, backend, polynomial, reducer, scalar=1, shift=None):
        if shift is None:
            shift = backend.zero_monomial
        retained = reducer.copy()
        work = native_module.NativeWorkspace(backend, polynomial)
        work.subtract_multiple(reducer, scalar, shift)
        self.assertEqual(work.result(), polynomial - backend.multiply_term(reducer, scalar, shift))
        self.assertEqual(reducer, retained)
        info = native_module._compact_cache_info(backend)
        self.assertLessEqual(info["cached_bytes"], info["budget_bytes"])
        return work, info

    def test_degree_contexts_share_one_budget_and_keep_distinct_encodings(self):
        backend = IntegerPolynomialRing(["x", "y"], "grevlex")
        x, y = backend.gens
        huge = (1 << 4096) + 71
        q = huge*x**2 - (huge - 3)*y + 1
        first, a = self.apply(backend, x**3 + y**2 + 7, q, -huge, (1, 0))
        frozen = first.result()
        _, b = self.apply(backend, x**7 + y + 5, q, huge)
        self.assertEqual((a["entries"], b["entries"]), (1, 2))
        self.assertEqual(b["misses"], a["misses"] + 1)
        _, c = self.apply(backend, x**3 + y**2 + 7, q, 3)
        self.assertEqual(c["hits"], b["hits"] + 1)
        self.assertEqual(c["entries"], 2)
        # An existing context must observe a changed shared budget, and an
        # independent workspace must keep its own exact coefficients alive.
        native_module._set_compact_cache_budget(backend, a["cached_bytes"])
        trimmed = native_module._compact_cache_info(backend)
        self.assertEqual(trimmed["entries"], 1)
        self.assertLessEqual(trimmed["cached_bytes"], trimmed["budget_bytes"])
        _, d = self.apply(backend, x**7 + y + 5, q, -huge)
        self.assertEqual(d["misses"], c["misses"] + 1)
        self.assertGreater(d["evictions"], c["evictions"])
        self.assertEqual(first.result(), frozen)

    def test_lru_eviction_and_oversized_entry_preserve_useful_reducers(self):
        backend = IntegerPolynomialRing(["x", "y"], "grevlex")
        x, y = backend.gens
        p = x**7 + y**2 + 19
        reducers = [x**2 + y + 1, -x**2 + 2*y + 3, 5*x**2 - 7*y + 11]
        _, first = self.apply(backend, p, reducers[0])
        native_module._set_compact_cache_budget(backend, 2*first["cached_bytes"])
        self.apply(backend, p, reducers[1])
        self.apply(backend, p, reducers[0])  # Keep the first reducer recent.
        _, before = self.apply(backend, p, reducers[2])
        self.assertEqual(before["entries"], 2)
        self.assertGreater(before["evictions"], 0)
        _, hit = self.apply(backend, p, reducers[0])
        self.assertEqual(hit["hits"], before["hits"] + 1)
        huge = (1 << 4096) + 31
        oversized = huge*x**2 - (huge + 1)*y + huge - 7
        _, after = self.apply(backend, p, oversized, -huge)
        self.assertEqual(after["entries"], hit["entries"])
        self.assertEqual(after["cached_bytes"], hit["cached_bytes"])
        self.assertEqual(after["evictions"], hit["evictions"])
        _, retained = self.apply(backend, p, reducers[0])
        self.assertEqual(retained["hits"], after["hits"] + 1)

    def test_large_integer_payload_is_charged_and_backend_orders_are_isolated(self):
        payloads = []
        for order in ("grevlex", "grlex"):
            for coefficient in (3, (1 << 4096) + 51):
                backend = IntegerPolynomialRing(["x", "y"], order)
                x, y = backend.gens
                self.assertIsNone(native_module._compact_cache_info(backend))
                _, info = self.apply(backend, x**5 + y**3 + 1,
                                     coefficient*x**2 - (coefficient + 2)*y + 7, -coefficient)
                self.assertEqual((info["entries"], info["hits"], info["misses"]), (1, 0, 1))
                payloads.append(info["cached_bytes"])
        self.assertGreater(payloads[1] - payloads[0], 512)
        self.assertGreater(payloads[3] - payloads[2], 512)

    def test_bulk_owners_survive_uncached_terms_eviction_and_reuse(self):
        for make in (fixture, lifted_fixture):
            for tail in (False, True):
                answers = []
                for native in (True, False):
                    engine, label = make()
                    frozen = records(row.polynomial for row in engine.P)
                    if native:
                        # Every encoded reducer exceeds this budget. During
                        # one bulk call only its active owners retain it.
                        native_module._set_compact_cache_budget(engine.backend, 1)
                    for _ in range(2):
                        result, counters = reduce(engine, label, native=native, tail=tail)
                        self.assertEqual(records(row.polynomial for row in engine.P), frozen)
                    answers.append((result.signature, records([result.polynomial]), counters))
                    if native:
                        info = native_module._compact_cache_info(engine.backend)
                        self.assertEqual((info["entries"], info["cached_bytes"]), (0, 0))
                        self.assertGreater(engine.stats.counters["native_bulk_terms_loaded"], 1)
                        self.assertEqual(engine.stats.counters["native_bulk_reduction_workspaces"], 2)
                self.assertEqual(answers[0], answers[1])


@unittest.skipUnless(NATIVE, "optional native reducer is unavailable")
class NativeTermCacheFallbackChecks(unittest.TestCase):
    def test_large_monomial_keys_keep_exact_fallback_without_compact_entries(self):
        huge = (1 << 4096) + 17
        for order in ("grevlex", "grlex"):
            backend = IntegerPolynomialRing([f"x{i}" for i in range(80)], order)
            x, y = backend.gens[0], backend.gens[-1]
            p = x**5 - huge*y**3 + 1
            q = huge*x**2 + 3*y - 7
            work = native_module.NativeWorkspace(backend, p)
            for scalar in (huge, -huge):
                work.subtract_multiple(q, scalar, backend.zero_monomial)
            self.assertEqual(work.result(), p)
            info = native_module._compact_cache_info(backend)
            if GMP:
                self.assertEqual((info["entries"], info["cached_bytes"], info["misses"]), (0, 0, 0))
            else:
                self.assertIsNone(info)


if __name__ == "__main__":
    unittest.main(verbosity=2)
