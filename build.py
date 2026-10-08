"""Build an executable: compile .ht to assembly, compile runtime.c, link."""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from compile import add_target_argument, compile_to_asm
from diagnostics import file_error, run_cli, write_output
from target import Target, as_target, host_target

REPO_ROOT = Path(__file__).resolve().parent
RUNTIME_C_PATH = REPO_ROOT / "runtime" / "runtime.c"


def c_compiler(target: Target) -> list[str]:
    """Command prefix for compiling and linking C and assembly for `target`."""
    if target.os == 'macos':
        # Apple's gcc picks the architecture with -arch.
        return ["gcc", "-arch", "arm64" if target.arch == "aarch64" else "x86_64"]
    if target.os == 'windows':  # MinGW-w64: itself on Windows, its cross compiler elsewhere
        return ["gcc"] if host_target().os == 'windows' else [f"{target.arch}-w64-mingw32-gcc"]
    if target.arch == host_target().arch:
        return ["gcc"]
    return [f"{target.arch}-linux-gnu-gcc"]  # Debian/Ubuntu cross toolchain naming


# The main thread's stack on Windows is fixed when a program is linked, at 1 MB unless asked
# otherwise; elsewhere it is the system's to give, and usually 8 MB. Hornet programs get that much
# everywhere, so a recursion that fits on one system fits on the others.
WINDOWS_STACK_BYTES = 8 * 1024 * 1024


def link_flags(target: Target) -> list[str]:
    """What the C compiler is told when linking a `target` program, besides its inputs and output.
    (Windows keeps its sockets, which the runtime's hornet_tcp_* use, in a library of their own.)"""
    return [f"-Wl,--stack,{WINDOWS_STACK_BYTES}", "-lws2_32"] if target.os == 'windows' else []


def executable_name(name: str, target: Target) -> str:
    """`name` as the file a linker writes for `target`: MinGW's adds `.exe` to a name without one."""
    return f"{name}.exe" if target.os == 'windows' and not name.endswith('.exe') else name


def run_prefix(target: Target) -> Optional[list[str]]:
    """Command prefix for running a `target` binary on this machine ([] if it runs directly), or
    None if it can't run here. Foreign Linux architectures run under qemu-user; x86-64 macOS
    binaries run under Rosetta 2 on Apple Silicon; Windows binaries run under Wine."""
    host = host_target()
    if target.os == 'windows' and host.os != 'windows':
        wine = next((w for w in ("wine64", "wine", "/usr/lib/wine/wine64") if shutil.which(w)), None)
        # WINEDEBUG=-all: without Wine's own diagnostics, which would mix into the program's stderr.
        return ["env", "WINEDEBUG=-all", wine] if wine and target.arch == host.arch else None
    if target.os != host.os:
        return None
    if target.arch == host.arch or (target.os == 'macos' and target.arch == 'x86_64'):
        return []
    if target.os == 'linux':
        return [f"qemu-{target.arch}", "-L", f"/usr/{target.arch}-linux-gnu"]
    return None


def can_build(target: Target) -> bool:
    """Whether this machine has a C toolchain that produces `target` binaries."""
    native_or_mingw = target.os in (host_target().os, 'windows')
    return native_or_mingw and shutil.which(c_compiler(target)[0]) is not None


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


def cache_dir() -> Path:
    """Where compiled artifacts are kept between runs: $HORNET_CACHE_DIR, else the user cache."""
    if os.environ.get("HORNET_CACHE_DIR"):
        return Path(os.environ["HORNET_CACHE_DIR"])
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "hornet"


def runtime_object(target: Target) -> Path:
    """runtime.c compiled for `target`, built once and reused: the file name carries a hash of the
    runtime's sources and the compile command, so any change builds a new one. Written atomically,
    so concurrent builds can share it."""
    command = c_compiler(target) + ["-c"]
    digest = hashlib.sha256(" ".join(command).encode())
    for path in [RUNTIME_C_PATH, *sorted(RUNTIME_C_PATH.parent.glob("*.h"))]:
        digest.update(path.read_bytes())
    obj = cache_dir() / f"runtime-{target}-{digest.hexdigest()[:16]}.o"
    if not obj.exists():
        obj.parent.mkdir(parents=True, exist_ok=True)
        partial = obj.with_name(f"{obj.stem}.{os.getpid()}.partial.o")
        _run(command + [str(RUNTIME_C_PATH), "-o", str(partial)], "compiling runtime.c")
        os.replace(partial, obj)
    return obj


def build_executable(source_path: str, output_path: str, target=None) -> None:
    """Compile `source_path` for `target` (Target, `arch-os`, or None for the default) and link it
    with runtime.c into `output_path`."""
    target = as_target(target)
    asm = compile_to_asm(source_path, target, require_main=True)
    cc = c_compiler(target)

    with tempfile.TemporaryDirectory() as tmpdir:
        asm_path = os.path.join(tmpdir, "program.s")
        write_output(asm_path, asm)  # (as bytes: see compile.py)

        # The output is written exactly where asked, whatever name the linker would choose.
        linked = os.path.join(tmpdir, executable_name("program", target))
        _run(cc + [asm_path, str(runtime_object(target)), "-o", linked] + link_flags(target), "linking")
        try:
            shutil.move(linked, output_path)
        except OSError as problem:
            raise file_error("write", output_path, problem) from None


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
