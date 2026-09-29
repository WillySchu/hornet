"""Build an executable: compile .ht to assembly, compile runtime.c, link."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from compile import compile_to_asm
from diagnostics import run_cli

REPO_ROOT = Path(__file__).resolve().parent
RUNTIME_C_PATH = REPO_ROOT / "runtime" / "runtime.c"

CC = "gcc"

# Without -arch x86_64, gcc on Apple Silicon targets arm64 and rejects our x86-64 output.
HOST_IS_MACOS = sys.platform == "darwin"
DEFAULT_PLATFORM = "macos" if HOST_IS_MACOS else "linux"


class BuildError(Exception):
    """Build step failed; carries the subprocess's stderr."""


def _run(args: list[str], step_name: str) -> None:
    if HOST_IS_MACOS:
        args = args[:1] + ["-arch", "x86_64"] + args[1:]
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise BuildError(
            f"{step_name} failed (exit code {result.returncode}):\n"
            f"  command: {' '.join(args)}\n"
            f"  stderr:\n{result.stderr}"
        )


def build_executable(source_path: str, output_path: str, platform: str = DEFAULT_PLATFORM) -> None:
    """Compile `source_path` and link it with runtime.c into `output_path`."""
    asm = compile_to_asm(source_path, platform=platform)

    with tempfile.TemporaryDirectory() as tmpdir:
        asm_path = os.path.join(tmpdir, "program.s")
        # Latin-1: see compile.py.
        with open(asm_path, "w", encoding="latin-1") as f:
            f.write(asm)

        runtime_o_path = os.path.join(tmpdir, "runtime.o")
        _run([CC, "-c", str(RUNTIME_C_PATH), "-o", runtime_o_path], "compiling runtime.c")

        _run([CC, asm_path, runtime_o_path, "-o", output_path], "linking")


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="Builds a runnable executable from a Hornet source file.")
    arg_parser.add_argument("file", type=str, help="Source file to compile.")
    arg_parser.add_argument(
        "--platform", choices=["macos", "linux"], default=DEFAULT_PLATFORM,
        help=f"Target platform; affects symbol naming. Default: {DEFAULT_PLATFORM} (this host)",
    )
    arg_parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Path to write the resulting executable to.",
    )
    arg_parser.add_argument("--traceback", action="store_true", help="Show Python tracebacks for all errors")
    args = arg_parser.parse_args()

    def action():
        try:
            build_executable(args.file, args.output, platform=args.platform)
        except BuildError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)

    run_cli(action, args.traceback)

if __name__ == "__main__":
    main()
