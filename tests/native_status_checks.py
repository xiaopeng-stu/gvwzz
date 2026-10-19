"""Early native diagnostics and explicit opt-in failure guard checks."""
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import build_native
from gvwzz.cli import main, timing_report
from gvwzz.native_status import native_status, entry_status_lines


def missing_status():
    return {
        "python_executable": sys.executable, "python_version": "test",
        "platform": "test", "package_directory": "test",
        "native_accelerators": dict.fromkeys(("reduction", "pairs", "indices"), False),
        "all_native_available": False, "GVWZZ_NO_NATIVE": None, "problems": {},
    }


class NativeStatusChecks(unittest.TestCase):
    def test_require_native_run_stops_before_solver_or_output(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "unwritten.json"
            with patch("gvwzz.native_status.native_status", return_value=missing_status()), \
                 patch("gvwzz.cli.execute_case") as solve, \
                 redirect_stdout(StringIO()) as out, redirect_stderr(StringIO()) as err:
                code = main(["run", "--case", "gcd_line", "--require-native",
                             "--output", str(output), "--no-print-ideal"])
            self.assertEqual(code, 1)
            solve.assert_not_called()
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".csv").exists())
            self.assertIn("Native extensions available: reduction=no", out.getvalue())
            self.assertIn("--require-native", err.getvalue())

    def test_require_native_benchmark_stops_before_solver_or_output(self):
        with TemporaryDirectory() as directory:
            with patch("gvwzz.native_status.native_status", return_value=missing_status()), \
                 patch("gvwzz.cli.execute_case") as solve, \
                 redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                code = main(["benchmark", "--cases", "gcd_line", "--require-native",
                             "--output", str(Path(directory) / "benchmark")])
            self.assertEqual(code, 1)
            solve.assert_not_called()
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_status_is_printed_before_computation_and_fallback_still_runs(self):
        observed = []
        output = StringIO()

        class SolverReached(Exception):
            pass

        def reached(*args, **kwargs):
            observed.append(output.getvalue())
            raise SolverReached

        with patch("gvwzz.native_status.native_status", return_value=missing_status()), \
             patch("gvwzz.cli.execute_case", side_effect=reached), redirect_stdout(output):
            with self.assertRaises(SolverReached):
                main(["run", "--case", "gcd_line", "--no-print-ideal"])
        self.assertEqual(len(observed), 1)
        self.assertIn("Native extensions available: reduction=no", observed[0])
        self.assertIn("python build_native.py", observed[0])

    def test_doctor_import_error_is_actionable_without_solving(self):
        with patch("gvwzz.solver.environment", return_value={
                "native_accelerators": missing_status()["native_accelerators"]}), \
             patch.dict(os.environ, {"GVWZZ_NO_NATIVE": ""}), \
             patch("gvwzz.native_status.importlib.import_module",
                   side_effect=ModuleNotFoundError("extension is missing")):
            data = native_status(diagnose=True)
        self.assertFalse(data["all_native_available"])
        self.assertEqual(set(data["problems"]), {
            "reduction", "pairs", "indices", "parent_batch", "candidate_heap",
            "packed_cover", "queue_drain",
        })
        self.assertTrue(all("extension is missing" in reason for reason in data["problems"].values()))

    def test_old_three_extensions_do_not_hide_a_missing_post28_entry_point(self):
        accelerators = dict.fromkeys(("reduction", "pairs", "indices"), True)
        entries = dict.fromkeys(("parent_batch", "candidate_heap", "packed_cover",
                                 "queue_drain"), True)
        entries["queue_drain"] = False
        with patch("gvwzz.solver.environment", return_value={
                "native_accelerators": accelerators, "native_entry_points": entries}):
            data = native_status()
        self.assertFalse(data["all_native_available"])
        self.assertIn("queue_drain=no", "\n".join(entry_status_lines(data)))

    def test_current_entry_points_satisfy_availability_without_running_a_kernel(self):
        with patch("gvwzz.solver.environment", return_value={
                "native_accelerators": dict.fromkeys(("reduction", "pairs", "indices"), True),
                "native_entry_points": dict.fromkeys(("parent_batch", "candidate_heap",
                                                       "packed_cover", "queue_drain"), True)}):
            data = native_status()
        self.assertTrue(data["all_native_available"])

    def test_imported_old_pairs_module_is_diagnosed_as_stale(self):
        from types import SimpleNamespace
        with patch("gvwzz.solver.environment", return_value={
                "native_accelerators": dict.fromkeys(("reduction", "pairs", "indices"), True),
                "native_entry_points": {"parent_batch": True, "candidate_heap": False,
                                         "packed_cover": True, "queue_drain": True}}), \
             patch.dict(os.environ, {"GVWZZ_NO_NATIVE": ""}), \
             patch("gvwzz.native_status.importlib.import_module", return_value=SimpleNamespace()):
            data = native_status(diagnose=True)
        self.assertEqual(set(data["problems"]), {"candidate_heap"})
        self.assertIn("lacks NativeCandidateHeap", data["problems"]["candidate_heap"])

    def test_fresh_interpreter_disable_flag_and_doctor_json(self):
        env = dict(os.environ, GVWZZ_NO_NATIVE="1")
        result = subprocess.run(
            [sys.executable, "-m", "gvwzz", "doctor", "--json", "--require-native"],
            cwd=Path(__file__).resolve().parents[1], env=env, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["native_accelerators"], missing_status()["native_accelerators"])
        self.assertEqual(data["native_reduction_coefficient_backend"], "python")
        self.assertTrue(all("GVWZZ_NO_NATIVE=1" in value for value in data["problems"].values()))

    def test_import_does_not_start_a_compiler_or_other_subprocess(self):
        script = """
import subprocess
def forbidden(*args, **kwargs):
    raise AssertionError('Package import must not start a subprocess')
subprocess.Popen = forbidden
subprocess.run = forbidden
import gvwzz
from gvwzz.native_status import native_status
from gvwzz import solver, incremental, structures
assert solver._native_reduce_working is None
assert incremental._native_emit_frozen_pairs is None
assert structures._native_has_divisor_packed is None
assert structures._NativeCandidateHeap is None
assert structures._NativeCoverQuery is None
assert solver._native_select_candidate is None
"""
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
            env=dict(os.environ, GVWZZ_NO_NATIVE="1"), capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_build_helper_uses_same_interpreter_and_checks_fresh_process(self):
        with patch("build_native.subprocess.run", return_value=subprocess.CompletedProcess([], 0)) as run, \
             redirect_stdout(StringIO()):
            self.assertEqual(build_native.main(), 0)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(run.call_args_list[0].args[0][1:],
                         [str(Path(build_native.__file__).resolve().parent /
                              "scripts" / "setup_native.py"),
                          "build_ext", "--inplace", "--force"])
        for call in run.call_args_list:
            self.assertEqual(call.args[0][0], sys.executable)
            self.assertEqual(call.kwargs["cwd"], Path(build_native.__file__).resolve().parent)
        self.assertEqual(run.call_args_list[-1].args[0][1:],
                         ["-m", "gvwzz", "doctor", "--require-native"])

    def test_build_failure_does_not_claim_success_or_run_doctor(self):
        with patch("build_native.subprocess.run", return_value=subprocess.CompletedProcess([], 2)) as run, \
             redirect_stdout(StringIO()), redirect_stderr(StringIO()) as err:
            self.assertEqual(build_native.main(), 2)
        self.assertEqual(run.call_count, 1)
        self.assertIn("Native build failed", err.getvalue())

    def test_nested_generation_timings_are_not_added_to_exclusive_sum(self):
        timings = {"total": 10.0, "generation": 10.0,
                   "frozen_seed_screen": 3.0, "frozen_multiplier_minimal": 4.0}
        with redirect_stdout(StringIO()) as out:
            timing_report(timings)
        text = out.getvalue()
        sum_line = next(line for line in text.splitlines() if "SUM / measured wall" in line)
        self.assertIn("10.000000s / 10.000000s", sum_line)
        self.assertIn("frozen seed screen 3.000000s", text)
        self.assertIn("multiplier minimization 4.000000s (included above)", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
