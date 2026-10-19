"""Run the release checks from any current working directory.

The default suite includes regression checks and installation smoke tests.
Use --differential to also compare native and Python execution against the
bundled reference snapshot; no earlier source release is needed.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-native", action="store_true",
        help="Check all required native entry points before running the suite.",
    )
    parser.add_argument(
        "--differential", action="store_true",
        help="Also run the 32 bundled reference comparisons (requires native extensions).",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Print a compact unittest report.",
    )
    args = parser.parse_args()

    if args.require_native or args.differential:
        status = subprocess.run(
            [sys.executable, "-m", "gvwzz", "doctor", "--require-native"],
            cwd=ROOT,
        )
        if status.returncode:
            return status.returncode

    # An explicit top-level directory keeps tests.* imports independent of cwd.
    loader = unittest.TestLoader()
    suite = loader.discover(
        str(ROOT / "tests"), pattern="*_checks.py", top_level_dir=str(ROOT),
    )
    suite.addTests(loader.loadTestsFromName("tests.selftest"))
    result = unittest.TextTestRunner(verbosity=1 if args.quiet else 2).run(suite)
    if not result.wasSuccessful():
        return 1

    if args.differential:
        return subprocess.run(
            [sys.executable, "-m", "tests.reduction_differential_checks"],
            cwd=ROOT,
        ).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
