"""Regression checks for release output and saved-file validation boundaries."""
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from gvwzz import Options, compute
from gvwzz.cli import load_result, main, write_csv, write_json
from gvwzz.path_cli import replay_to_file, verify_syzygy_file
from gvwzz.printing import print_result


class ReleaseIOChecks(unittest.TestCase):
    def test_bad_input_annotations_fail_before_solving_or_output(self):
        invalid = [
            ("name", None), ("name", 3),
            ("provenance", True), ("provenance", "source"), ("provenance", [1]),
            ("input_order", None), ("input_order", []), ("input_order", {}),
            ("input_order", {"direction": "reversed", "effective_to_source": [True]}),
            ("input_order", {"direction": "reversed", "effective_to_source": [2]}),
            ("input_order", {"direction": 1, "effective_to_source": [1]}),
        ]
        with TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.json", Path(directory) / "result.json"
            original = b'{"previous_result":true}\n'
            output.write_bytes(original)
            for field, value in invalid:
                data = {"variables": ["x"], "polynomials": ["x"], field: value}
                source.write_text(json.dumps(data), encoding="utf-8")
                with self.subTest(field=field, value=value), \
                        patch("gvwzz.cli.execute_case") as solve, \
                        redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                    self.assertEqual(main(["run", "--input", str(source),
                                           "--output", str(output)]), 1)
                    solve.assert_not_called()
                self.assertEqual(output.read_bytes(), original)
                self.assertFalse(output.with_suffix(".csv").exists())

    def test_saved_timings_survive_loading_resaving_and_show(self):
        result = compute(["x"], ["x"], show_basis=False, show_u=False)
        result.setup_seconds = 0.125
        result.stats.total_ns = 12345678901
        result.stats.cpu_ns = 11000000007
        result.stats.times_ns.update({"cover": 1300000001, "syzygy": 3000000000,
                                      "syzygy_check": 2000000000,
                                      "syzygy_complete": 1000000000,
                                      "tail_reduction": 2500000000,
                                      "j_tail_reduction": 1500000000,
                                      "g_tail_reduction": 1000000000,
                                      "generation": 2000000000,
                                      "frozen_seed_screen": 250000000,
                                      "frozen_multiplier_minimal": 125000000,
                                      "incremental_minimization": 1000000000})
        with TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_json(path, result.to_dict())
            restored = load_result(path)
            self.assertEqual(restored.setup_seconds, result.setup_seconds)
            self.assertEqual(restored.stats.total_ns, result.stats.total_ns)
            self.assertEqual(restored.stats.cpu_ns, result.stats.cpu_ns)
            self.assertEqual(restored.stats.times_ns, result.stats.times_ns)
            # The derived remainder is recomputed from the trusted timer fields.
            expected = result.stats.to_dict()["timings_seconds"]
            actual = restored.to_dict()["timings_seconds"]
            for key in expected:
                self.assertAlmostEqual(actual[key], expected[key], places=8)
            write_json(path, restored.to_dict())
            self.assertEqual(load_result(path).stats.total_ns, result.stats.total_ns)
            with patch("gvwzz.cli.print_result", wraps=print_result) as printer, \
                    redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(main(["show", str(path), "--no-show-u"]), 0)
            self.assertEqual(printer.call_args.args[0].stats.total_ns, result.stats.total_ns)

    def test_legacy_and_malformed_saved_timings(self):
        result = compute(["x"], ["x"], show_basis=False, show_u=False)
        data = result.to_dict()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            legacy = deepcopy(data)
            legacy.pop("setup_seconds")
            legacy.pop("timings_seconds")
            write_json(path, legacy)
            restored = load_result(path)
            self.assertEqual((restored.setup_seconds, restored.stats.total_ns,
                              restored.stats.cpu_ns), (0.0, 0, 0))
            self.assertNotIn("j_tail_reduction", restored.stats.times_ns)
            for field in ("setup_seconds", "total", "process_cpu", "cover",
                          "j_tail_reduction", "other"):
                for value in (True, "1.0", None, float("nan"), float("inf")):
                    broken = deepcopy(data)
                    if field == "setup_seconds":
                        broken[field] = value
                    else:
                        broken["timings_seconds"][field] = value
                    write_json(path, broken)
                    with self.subTest(field=field, value=value), self.assertRaisesRegex(
                            ValueError, "finite"):
                        load_result(path)
            broken = deepcopy(data)
            broken["timings_seconds"] = []
            write_json(path, broken)
            with self.assertRaisesRegex(ValueError, "timings_seconds must be an object"):
                load_result(path)

    def test_csv_failure_preserves_previous_file(self):
        class BrokenCell:
            def __str__(self):
                raise ValueError("cell conversion failed")

        with TemporaryDirectory() as directory:
            path = Path(directory) / "result.csv"
            original = b"old,data\r\n1,2\r\n"
            path.write_bytes(original)
            with self.assertRaisesRegex(ValueError, "cell conversion failed"):
                write_csv(path, [{"value": BrokenCell()}])
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_benchmark_medians_use_completed_repeats_only(self):
        # A later budget limit must not change a completed timing or P size.
        records = [
            {"case": "cyclic3", "polynomial_order": "grevlex", "module_order": "pot",
             "status": "completed", "message": "", "options": {},
             "basis_size": 5, "syzygy_size": 2, "timings_seconds": {"total": 2.0}},
            {"case": "cyclic3", "polynomial_order": "grevlex", "module_order": "pot",
             "status": "limit", "message": "time limit", "options": {},
             "basis_size": 1, "syzygy_size": 0, "timings_seconds": {"total": 100.0}},
        ]
        with TemporaryDirectory() as directory:
            prefix = Path(directory) / "benchmark"
            with patch("gvwzz.cli.execute_case", side_effect=deepcopy(records)), \
                    redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                status = main(["benchmark", "--cases", "cyclic3", "--poly-orders", "grevlex",
                               "--module-orders", "pot", "--repeat", "3", "--output", str(prefix)])
            self.assertEqual(status, 0)
            row = json.loads(Path(str(prefix) + ".summary.json").read_text())["results"][0]
            self.assertEqual(row["total_seconds"], 2.0)
            self.assertEqual(row["basis_size"], 5)
            self.assertEqual(row["repeats_actual"], 2)
            self.assertEqual(row["repeats_completed"], 1)
            self.assertEqual(row["repeat_statuses"], ["completed", "limit"])

    def test_benchmark_failed_verification_has_nonzero_exit_and_saved_report(self):
        record = {"case": "cyclic3", "polynomial_order": "grevlex", "module_order": "pot",
                  "status": "completed", "message": "", "options": {},
                  "basis_size": 5, "syzygy_size": 2, "timings_seconds": {"total": 2.0},
                  "verification": {"passed": False, "status": "failed"}}
        with TemporaryDirectory() as directory:
            prefix = Path(directory) / "benchmark"
            with patch("gvwzz.cli.execute_case", return_value=record), \
                    redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                status = main(["benchmark", "--cases", "cyclic3", "--poly-orders", "grevlex",
                               "--module-orders", "pot", "--repeat", "1", "--verify",
                               "--output", str(prefix)])
            self.assertEqual(status, 3)
            report = json.loads(Path(str(prefix) + ".raw.jsonl").read_text())
            self.assertFalse(report["verification"]["passed"])
            summary = json.loads(Path(str(prefix) + ".summary.json").read_text())["results"][0]
            self.assertFalse(summary["all_verified"])

    def test_stream_rejects_noncanonical_selection_and_footer_counts(self):
        result = compute(["x", "y"], ["x", "y"], options=Options(record_paths=True),
                         show_basis=False, show_u=False)
        self.assertTrue(result.syzygies)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "syzygies.jsonl"
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                replay_to_file(result, path, indices=[0], stream=True,
                               verify=True, progress_interval=0)
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertTrue(verify_syzygy_file(path)["passed"])
            for selected in ([True], [1, 1]):
                broken = deepcopy(records)
                broken[0]["selected_indices"] = selected
                path.write_text("".join(json.dumps(row) + "\n" for row in broken))
                self.assertFalse(verify_syzygy_file(path)["passed"], selected)
            broken = deepcopy(records)
            broken[-1]["statistics"]["syzygies_reconstructed"] = True
            path.write_text("".join(json.dumps(row) + "\n" for row in broken))
            self.assertFalse(verify_syzygy_file(path)["passed"])


if __name__ == "__main__":
    unittest.main()
