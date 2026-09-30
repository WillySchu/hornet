"""Build an executable: compile .ht to assembly, compile runtime.c, link."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from compile import add_target_argument, compile_to_asm
from diagnostics import run_cli
from target import Target, as_target, host_target

REPO_ROOT = Path(__file__).resolve().parent
RUNTIME_C_PATH = REPO_ROOT / "runtime" / "runtime.c"


def c_compiler(target: Target) -> list[str]:
    """Command prefix for compiling and linking C and assembly for `target`."""
    if target.os == 'macos':
        # Apple's gcc picks the architecture with -arch.
        return ["gcc", "-arch", "arm64" if target.arch == "aarch64" else "x86_64"]
    if target.arch == host_target().arch:
        return ["gcc"]
    return [f"{target.arch}-linux-gnu-gcc"]  # Debian/Ubuntu cross toolchain naming


def run_prefix(target: Target) -> Optional[list[str]]:
    """Command prefix for running a `target` binary on this machine ([] if it runs directly), or
    None if it can't run here. Foreign Linux architectures run under qemu-user; x86-64 macOS
    binaries run under Rosetta 2 on Apple Silicon."""
    host = host_target()
    if target.os != host.os:
        return None
    if target.arch == host.arch or (target.os == 'macos' and target.arch == 'x86_64'):
        return []
    if target.os == 'linux':
        return [f"qemu-{target.arch}", "-L", f"/usr/{target.arch}-linux-gnu"]
    return None


def can_build(target: Target) -> bool:
    """Whether this machine has a C toolchain that produces `target` binaries."""
    return target.os == host_target().os and shutil.which(c_compiler(target)[0]) is not None


def can_run(target: Target) -> bool:
    """Whether `target` binaries can be both built and run here."""
    prefix = run_prefix(target)
    return can_build(target) and prefix is not None and (not prefix or shutil.which(prefix[0]) is not None)


class BuildError(Exception):
    """Build step failed; carries the subprocess's stderr."""


def _run(args: list[str], step_name: str) -> None:
    try:
        result = subprocess.run(args, capture_output=True, text=True)
    except FileNotFoundError:
        raise BuildError(f"{step_name} failed: '{args[0]}' not found (is the toolchain for this target installed?)")
    if result.returncode != 0:
        raise BuildError(
            f"{step_name} failed (exit code {result.returncode}):\n"
            f"  command: {' '.join(args)}\n"
            f"  stderr:\n{result.stderr}"
        )


def build_executable(source_path: str, output_path: str, target=None) -> None:
    """Compile `source_path` for `target` (Target, `arch-os`, or None for the default) and link it
    with runtime.c into `output_path`."""
    target = as_target(target)
    asm = compile_to_asm(source_path, target)
    cc = c_compiler(target)

    with tempfile.TemporaryDirectory() as tmpdir:
        asm_path = os.path.join(tmpdir, "program.s")
        # Latin-1: see compile.py.
        with open(asm_path, "w", encoding="latin-1") as f:
            f.write(asm)

        runtime_o_path = os.path.join(tmpdir, "runtime.o")
        _run(cc + ["-c", str(RUNTIME_C_PATH), "-o", runtime_o_path], "compiling runtime.c")

        _run(cc + [asm_path, runtime_o_path, "-o", output_path], "linking")


def main() -> None:
    arg_parser = argparse.ArgumentParser(description="Builds a runnable executable from a Hornet source file.")
    arg_parser.add_argument("file", type=str, help="Source file to compile.")
    add_target_argument(arg_parser)
    arg_parser.add_argument(
        "-o", "--output", type=str, required=True,
        help="Path to write the resulting executable to.",
    )
    arg_parser.add_argument("--traceback", action="store_true", help="Show Python tracebacks for all errors")
    args = arg_parser.parse_args()

    def action():
        try:
            build_executable(args.file, args.output, target=args.target)
        except BuildError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)

    run_cli(action, args.traceback)

if __name__ == "__main__":
    main()
