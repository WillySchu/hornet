"""Builds a runnable executable from a Hornet source file: compiles
the source through the ordinary lex -> parse -> analyze -> codegen
pipeline (compile_to_asm), compiles runtime/runtime.c (fresh, every
time -- see runtime.c's own module docstring for why this is
deliberately not yet a precompiled, cached artifact), and links both
together with the platform's own C compiler.

This is the one piece of the toolchain that didn't exist at all
before runtime.c did: previously, "run a Hornet program" meant
compile_to_asm's own assembly text, assembled and linked directly (no
second object file ever entered the picture, since hornet_stringify
used to be emitted as literal assembly text inline in that same .s
file). Bringing in a real C runtime means there are now genuinely two
separate compilation units that need to be produced and linked
together, not one.
"""
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from codegen.codegen import compile_to_asm

REPO_ROOT = Path(__file__).resolve().parent
RUNTIME_C_PATH = REPO_ROOT / "runtime" / "runtime.c"

CC = "gcc"  # matches the compiler used throughout this project's own test suite


class BuildError(Exception):
    """Raised when any step of the build (codegen, compiling
    runtime.c, or the final link) fails -- carries the underlying
    subprocess's own captured stderr, not just a bare non-zero exit
    code, so a failure here is actionable without re-running the
    failing command by hand to see what it actually said."""


def _run(args: list[str], step_name: str) -> None:
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise BuildError(
            f"{step_name} failed (exit code {result.returncode}):\n"
            f"  command: {' '.join(args)}\n"
            f"  stderr:\n{result.stderr}"
        )


def build_executable(source_path: str, output_path: str, platform: str = "linux") -> None:
    """Compiles `source_path` (a .ht file) and links it, together with
    a freshly-compiled runtime.c, into a single executable at
    `output_path`.

    Raises BuildError (wrapping the underlying failure's own stderr)
    if lexing/parsing/semantic analysis fails (propagated directly,
    not wrapped, since those already raise their own clear,
    Hornet-specific exceptions), or if compiling runtime.c or the
    final link step fails.
    """
    asm = compile_to_asm(source_path, platform=platform)

    with tempfile.TemporaryDirectory() as tmpdir:
        asm_path = os.path.join(tmpdir, "program.s")
        with open(asm_path, "w") as f:
            f.write(asm)

        runtime_o_path = os.path.join(tmpdir, "runtime.o")
        _run([CC, "-c", str(RUNTIME_C_PATH), "-o", runtime_o_path], "compiling runtime.c")

        _run([CC, asm_path, runtime_o_path, "-o", output_path], "linking")


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="Builds a runnable executable from a Hornet source file.")
    arg_parser.add_argument("file", type=str, help="Source file to compile.")
    arg_parser.add_argument(
        "--platform", choices=["macos", "linux"], default="linux",
        help="Target platform; affects symbol naming. Default: linux",
    )
    arg_parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Path to write the resulting executable to.",
    )
    args = arg_parser.parse_args()

    try:
        build_executable(args.file, args.output, platform=args.platform)
    except BuildError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
