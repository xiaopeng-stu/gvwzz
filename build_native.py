"""Build and check all optional extensions using this exact Python interpreter.

Run explicitly as ``python build_native.py`` from an extracted source release.
This script does not install Python packages or operating-system compilers.
"""
from pathlib import Path
import subprocess
import sys


def main() -> int:
    root = Path(__file__).resolve().parent
    print("Building native accelerators with:", sys.executable, flush=True)
    # Re-extracted source timestamps can be older than an existing binary.
    # An explicit upgrade build must not silently reuse the previous release.
    command = [sys.executable, str(root / "scripts" / "setup_native.py"),
               "build_ext", "--inplace", "--force"]
    result = subprocess.run(command, cwd=root)
    if result.returncode:
        print("Native build failed. Read the compiler error above.", file=sys.stderr)
        print("Install Python build dependencies with this interpreter:", file=sys.stderr)
        print("  python -m pip install -r requirements-native.txt", file=sys.stderr)
        print("On Windows, install Microsoft C++ Build Tools with the C++ workload.", file=sys.stderr)
        return result.returncode
    # A fresh interpreter avoids reporting stale import state from a prior run.
    return subprocess.run(
        [sys.executable, "-m", "gvwzz", "doctor", "--require-native"], cwd=root,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
