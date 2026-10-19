"""Small release smoke test. Does not replace the development regression suite."""
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from gvwzz import (
    Options, compute, format_forest, reconstruct_syzygies,
    verify_generation_paths,
)
from gvwzz.cli import load_result, main, write_json
from gvwzz.core import decode_integer, encode_integer
from gvwzz.path_cli import replay_to_file, verify_syzygy_file
from gvwzz.verification import verify_result


class ReleaseSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.online = compute(
            ["4*x^2", "9*y^4", "6*x*y^2+y"], ["x", "y"], "lex", "pot",
            options=Options(track_representations=True, g_first=False),
            show_basis=False, show_u=False,
        )
        cls.paths = compute(
            ["4*x^2", "9*y^4", "6*x*y^2+y"], ["x", "y"], "lex", "pot",
            options=Options(record_paths=True, g_first=False),
            show_basis=False, show_u=False,
        )

    def test_01_defaults(self) -> None:
        opt = Options()
        opt.validate()
        self.assertTrue(opt.euclidean and opt.tail_reduce and opt.lazy_candidates)
        self.assertEqual((opt.reducer_selection, opt.queue_tie), ("sparse", "small-coeff"))
        for name in ("max_seconds", "max_basis", "max_candidates", "max_reduction_steps",
                     "max_polynomial_terms", "max_coefficient_bits"):
            self.assertIsNone(getattr(opt, name))

    def test_02_online_certificate(self) -> None:
        self.assertEqual(self.online.status, "completed")
        report = verify_result(self.online, mode="static", strong=True,
                               require_certificates=True)
        self.assertTrue(report["signature_basis_certified"], report)

    def test_03_path_certificate_and_same_history(self) -> None:
        self.assertEqual(self.paths.status, "completed")
        self.assertIsNone(self.paths.basis_representations)
        self.assertEqual(self.paths.basis, self.online.basis)
        self.assertEqual(self.paths.syzygies, self.online.syzygies)
        report = verify_generation_paths(self.paths)
        self.assertTrue(report["signature_basis_certified"], report)

    def test_04_syzygy_replay(self) -> None:
        post = reconstruct_syzygies(self.paths, verify=True)
        self.assertEqual(post.syzygies, self.online.syzygy_representations)
        self.assertEqual(post.statistics["basis_vectors_expanded"], 0)

    def test_05_roundtrip_and_stream(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "paths.json"
            write_json(path, self.paths.to_dict())
            loaded = load_result(path)
            self.assertTrue(verify_generation_paths(loaded)["passed"])
            target = Path(directory) / "syzygies.jsonl"
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                replay_to_file(loaded, target, stream=True, verify=True, progress_interval=0)
            self.assertTrue(verify_syzygy_file(target)["passed"])

    def test_06_mutually_exclusive_modes(self) -> None:
        with self.assertRaises(ValueError):
            Options(record_paths=True, track_representations=True).validate()

    def test_07_forest_and_integer_codec(self) -> None:
        self.assertIn("Ratio-layer forests", format_forest(self.paths))
        for n in (0, 1, -1, 2**3000, -(10**5000)):
            self.assertEqual(decode_integer(encode_integer(n)), n)

    def test_08_builtin_cli(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cyclic3.json"
            args = ["run", "--case", "cyclic3", "--poly-order", "grevlex",
                    "--module-order", "pot", "--record-paths", "--verify",
                    "--no-print-ideal", "--no-show-basis", "--no-show-u",
                    "--output", str(path)]
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(main(args), 0)
            self.assertEqual(load_result(path).status, "completed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
