"""Report measured kernel dispatch separately from installed extensions."""
from contextlib import redirect_stdout
from io import StringIO
import json
import unittest
from unittest.mock import patch

from gvwzz import Options, compute
from gvwzz.native_status import print_native_start, status_line, usage_lines
from gvwzz.runtime import environment, native_usage


class NativeUsageReportingChecks(unittest.TestCase):
    def test_available_extensions_do_not_claim_a_batch_was_used(self):
        available = {
            "native_accelerators": dict.fromkeys(("reduction", "pairs", "indices"), True),
            "native_entry_points": {"parent_batch": True},
            "native_reduction_coefficient_backend": "gmp",
            "all_native_available": True,
            "GVWZZ_NO_NATIVE": None,
        }
        counters = {"parent_pairs": 12, "parent_python_rows": 4,
                    "parent_fallback_certificate_rows": 4}
        usage = native_usage(counters)
        with patch("gvwzz.native_status.native_status", return_value=available), \
                redirect_stdout(StringIO()) as out:
            # --require-native remains an availability check, not a promise
            # that every case can use every bulk kernel.
            print_native_start(require_native=True)
        self.assertIn("Native extensions available: reduction=yes", out.getvalue())
        self.assertIn("actual usage is reported after computation", out.getvalue())
        lines = "\n".join(usage_lines(usage))
        self.assertIn("rows=0 pairs=0", lines)
        self.assertIn("Python parent dispatch: rows=4 pairs=12", lines)
        self.assertIn("representation certificates are enabled", lines)
        self.assertEqual(usage["reduction"]["native_workspaces"], 0)

    def test_mixed_parent_usage_counts_fallback_pairs_once(self):
        usage = native_usage({
            "parent_pairs": 110, "native_parent_rows": 8,
            "native_parent_pairs": 100, "native_parent_wide_rows": 6,
            "native_parent_wide_pairs": 75, "native_parent_dynamic_rows": 2,
            "native_parent_dynamic_pairs": 20, "parent_python_rows": 2,
            "native_parent_fallback_pairs": 3, "parent_fallback_exponent_rows": 2,
        })["parent_generation"]
        self.assertEqual(usage["python_pairs"], 10)
        self.assertEqual(usage["within_batch_python_pairs"], 3)
        self.assertEqual(usage["native_wide_pairs"], 75)
        self.assertEqual(usage["native_dynamic_rows"], 2)
        self.assertEqual(usage["native_dynamic_pairs"], 20)
        self.assertEqual(usage["fallback_reasons_rows"]["exponent"]["rows"], 2)

    def test_queue_drain_reports_observed_subsets_without_extra_rejections(self):
        usage = native_usage({"native_queue_drain_calls": 7,
                              "native_queue_drain_pops": 900,
                              "native_queue_drain_covered": 880})
        self.assertEqual(usage["queue_drain"], {
            "calls": 7, "popped": 900, "covered": 880,
        })
        self.assertIn("calls=7 popped=900 covered=880", "\n".join(usage_lines(usage)))

    def test_queue_heap_and_cover_counts_retain_their_distinct_scopes(self):
        usage = native_usage({"native_queue_pushes": 50, "native_queue_pops": 47,
                              "native_queue_dedup_checks": 80, "native_cover_queries": 30,
                              "native_cover_proofs": 20, "native_cover_fallbacks": 10})
        self.assertEqual(usage["candidate_queue"], {"pushes": 50, "pops": 47, "dedup_checks": 80})
        self.assertEqual(usage["packed_cover"], {"queries": 30, "proofs": 20, "fallbacks": 10})
        rendered = "\n".join(usage_lines(usage))
        self.assertIn("pushes=50 pops=47 dedup checks=80", rendered)
        self.assertIn("queries=30 proofs=20 fallbacks=10", rendered)

    def test_dynamic_parent_count_is_a_wide_subset_not_an_extra_total(self):
        usage = native_usage({"parent_pairs": 90, "native_parent_pairs": 90,
                              "native_parent_wide_pairs": 70,
                              "native_parent_dynamic_pairs": 30})
        parent = usage["parent_generation"]
        self.assertEqual(parent["native_batch_pairs"], 90)
        self.assertEqual(parent["python_pairs"], 0)
        self.assertIn("dynamic rows/pairs=0/30", "\n".join(usage_lines(usage)))

    def test_native_nonbulk_reduction_is_not_labeled_python(self):
        usage = native_usage({"native_reduction_workspaces": 20,
                              "native_bulk_reduction_workspaces": 16,
                              "native_wide_bulk_reduction_workspaces": 7})
        reduction = usage["reduction"]
        self.assertEqual(reduction["native_nonbulk_workspaces"], 4)
        self.assertIsNone(reduction["python_workspaces"])
        self.assertIsNone(reduction["bulk_fallback_reasons"])
        rendered = "\n".join(usage_lines(usage))
        self.assertIn("total=20 bulk=16 wide=7 non-bulk=4", rendered)
        self.assertIn("still use native reduction", rendered)

    def test_unrecorded_python_rows_are_not_reported_as_zero(self):
        usage = native_usage({"parent_pairs": 15})
        self.assertIsNone(usage["parent_generation"]["python_rows"])
        self.assertIn("rows=not recorded", "\n".join(usage_lines(usage)))

    def test_observed_native_only_dispatch_reports_zero_python_rows(self):
        usage = native_usage({"parent_dispatch_rows": 3, "native_parent_rows": 3,
                              "parent_pairs": 3, "native_parent_pairs": 3})
        self.assertEqual(usage["parent_generation"]["python_rows"], 0)
        self.assertEqual(usage["parent_generation"]["python_pairs"], 0)

    def test_empty_counters_are_reported_without_invented_fallbacks(self):
        usage = native_usage({})
        self.assertIsNone(usage["parent_generation"]["python_rows"])
        self.assertIsNone(usage["parent_generation"]["python_pairs"])
        self.assertEqual(usage["parent_generation"]["fallback_reasons_rows"], {})
        self.assertTrue(usage_lines(usage))

    def test_incremental_skipped_pairs_are_not_counted_as_python(self):
        usage = native_usage({"parent_pairs": 100, "native_parent_pairs": 12},
                             incremental_pot=True)
        self.assertIsNone(usage["parent_generation"]["python_pairs"])
        self.assertEqual(usage["parent_generation"]["native_batch_pairs"], 12)

    def test_failed_import_availability_is_displayed_without_on_claim(self):
        line = status_line({"native_accelerators": {
            "reduction": True, "pairs": False, "indices": True,
        }})
        self.assertEqual(line, "Native extensions available: reduction=yes | pairs=no | indices=yes")

    def test_ordinary_parent_import_is_reported_separately(self):
        with patch("gvwzz.solver._native_emit_parent_pairs", None), \
                patch("gvwzz.incremental._native_emit_frozen_pairs", object()):
            runtime = environment()
        # Retain the historical accelerator mapping while exposing the missing
        # ordinary-parent entry point for the stricter post28 build guard.
        self.assertTrue(runtime["native_accelerators"]["pairs"])
        self.assertFalse(runtime["native_entry_points"]["parent_batch"])
        self.assertTrue(runtime["native_entry_points"]["frozen_parent_batch"])

    def test_result_json_contains_actual_usage_without_requiring_native(self):
        with patch("gvwzz.solver._native_emit_parent_pairs", None):
            result = compute(["2*x", "3*x"], ("x",), "grevlex", "g2",
                             options=Options(), show_basis=False, show_u=False)
        data = json.loads(json.dumps(result.to_dict(include_basis=False)))
        self.assertEqual(data["status"], "completed")
        parent = data["native_usage"]["parent_generation"]
        self.assertEqual(parent["native_batch_pairs"], 0)
        self.assertGreater(parent["python_pairs"], 0)
        # Native counters describe the completed solve, independently of the
        # environment queried after the temporary entry-point override ends.
        self.assertEqual(parent["python_pairs"], data["counters"]["parent_pairs"])

    def test_partial_result_usage_does_not_count_unvisited_parents(self):
        result = compute(["2*x", "3*x", "5*x"], ("x",), "grevlex", "g2",
                         options=Options(max_candidates=1),
                         show_basis=False, show_u=False)
        data = result.to_dict(include_basis=False)
        self.assertEqual(data["status"], "limit")
        parent, counters = data["native_usage"]["parent_generation"], data["counters"]
        self.assertEqual(parent["native_batch_pairs"] + parent["python_pairs"],
                         counters["parent_pairs"])
        self.assertEqual(parent["native_batch_pairs"], 0)
        self.assertLess(counters["parent_pairs"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
