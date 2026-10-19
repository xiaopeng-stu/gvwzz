"""Configure optional native extensions for the selected Python interpreter.

The public entry point is ``python build_native.py``. This module can also be
run directly as ``python scripts/setup_native.py build_ext --inplace``.
Importing it only defines helpers; it never starts a compiler or installs files.
"""
from pathlib import Path
import os
import shlex
import shutil
import sys
import sysconfig

from setuptools import Extension, setup


ROOT = Path(__file__).resolve().parents[1]
NATIVE_SOURCES = ROOT / "native"
GENERATED_SOURCES = NATIVE_SOURCES / "generated"


def configure_compiler() -> None:
    """Respect user compiler choices and repair unavailable portable defaults."""
    if sys.platform == "win32":
        return  # setuptools selects the MSVC toolchain for this Python.
    for variable, fallback in (("CC", "gcc"), ("CXX", "g++")):
        configured = sysconfig.get_config_var(variable) or ""
        executable = shlex.split(configured)[0] if configured else ""
        if (variable not in os.environ and not shutil.which(executable)
                and shutil.which(fallback)):
            os.environ[variable] = fallback


def bundled_gmp_options() -> dict:
    """Use public headers and libraries bundled in the installed gmpy2 wheel.

    No system GMP installation or global compiler-path changes are needed.
    Source installations without these optional files retain the exact
    Python-integer native workspace.
    """
    if os.environ.get("GVWZZ_NO_GMP"):
        print("Native reduction arithmetic: Python integers (GVWZZ_NO_GMP).")
        return {}
    try:
        import gmpy2
    except (ImportError, OSError):
        print("Native reduction arithmetic: Python integers; gmpy2 is unavailable.")
        return {}
    package = Path(gmpy2.__file__).resolve().parent
    headers = ("gmp.h", "mpfr.h", "mpc.h", "gmpy2.h")
    if not all((package / name).is_file() for name in headers):
        print("Native reduction arithmetic: Python integers; gmpy2 headers are unavailable.")
        return {}
    directories = [package.parent / "gmpy2.libs", package / ".dylibs",
                   package.parent / ".dylibs"]
    if sys.platform == "win32":
        patterns = ["gmp.lib"]
    elif sys.platform == "darwin":
        patterns = ["libgmp*.dylib"]
    else:
        patterns = ["libgmp*.so*"]
    libraries = [path for directory in directories for pattern in patterns
                 for path in sorted(directory.glob(pattern)) if path.is_file()]
    if not libraries:
        print("Native reduction arithmetic: Python integers; bundled GMP library is unavailable.")
        return {}
    library = libraries[0].resolve()
    options = dict(include_dirs=[str(package)], extra_objects=[str(library)],
                   define_macros=[("GVWZZ_HAVE_GMP", "1")])
    if sys.platform != "win32":
        # This is an in-place build for the selected interpreter. Rebuild
        # after moving the environment or changing its gmpy2 installation.
        options["runtime_library_dirs"] = [str(library.parent)]
    print("Native reduction arithmetic: GMP exact integers from gmpy2", gmpy2.__version__)
    return options


def make_extensions() -> list[Extension]:
    """Keep module names stable although their editable sources live elsewhere."""
    sources = sorted(NATIVE_SOURCES.glob("_native_*.pyx"))
    if not sources:
        raise SystemExit("No native accelerator sources were found in native/.")
    gmp_options = bundled_gmp_options()
    # Editable and generated C++ sources share portable native helper headers.
    # Resolve from the release root so an extracted source build is standalone.
    gmp_options["include_dirs"] = [str(NATIVE_SOURCES), *gmp_options.get("include_dirs", [])]
    compile_flags = ["/O2"] if sys.platform == "win32" else ["-O3"]
    return [Extension(
        "gvwzz." + path.stem,
        [str(path.relative_to(ROOT))],
        language="c++",
        extra_compile_args=compile_flags,
        **(gmp_options if path.stem == "_native_reduction"
           else {"include_dirs": [str(NATIVE_SOURCES)]}),
    ) for path in sources]


def prepare_sources(extensions: list[Extension]) -> list[Extension]:
    """Regenerate C++ when Cython is present; otherwise use the shipped copy."""
    try:
        from Cython.Build import cythonize
    except ImportError:
        for extension in extensions:
            generated = GENERATED_SOURCES / (extension.name.rsplit(".", 1)[1] + ".cpp")
            if not generated.is_file():
                raise SystemExit(
                    "Install build dependencies: python -m pip install -r requirements-native.txt"
                )
            extension.sources = [str(generated.relative_to(ROOT))]
        return extensions

    extensions = cythonize(extensions, build_dir=str(Path("build") / "cython"),
                           include_path=[str(NATIVE_SOURCES)],
                           compiler_directives={
                               "language_level": 3, "annotation_typing": False,
                               "binding": True, "embedsignature": True,
                           })
    # Keep generated output separate from editable .pyx files. The release
    # includes these copies so building without Cython remains possible.
    GENERATED_SOURCES.mkdir(parents=True, exist_ok=True)
    for extension in extensions:
        name = extension.name.rsplit(".", 1)[1] + ".cpp"
        generated = next(Path(source) for source in extension.sources
                         if Path(source).name == name)
        if not generated.is_absolute():
            generated = ROOT / generated
        shutil.copy2(generated, GENERATED_SOURCES / name)
    return extensions


def main() -> None:
    # setuptools resolves pyproject.toml and relative source paths from cwd.
    # Resolve from this file so direct invocation also works outside the tree.
    os.chdir(ROOT)
    configure_compiler()
    setup(ext_modules=prepare_sources(make_extensions()))


if __name__ == "__main__":
    main()
