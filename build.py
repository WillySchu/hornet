"""Build an executable: compile .ht to assembly, build the runtime, link."""

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
RUNTIME_DIR = REPO_ROOT / "runtime"
# The runtime's C files: a `_windows.c` one is built only for Windows, a `_posix.c` one only elsewhere.
# By name, so that a file left behind in runtime/ isn't built into it.
RUNTIME_FILES = (
    "start.c", "stack.c", "stack_posix.c", "stack_windows.c", "memory.c", "heap.c", "panic.c", "format.c", "slice.c",
    "dict.c", "io.c", "fs.c", "net.c", "system.c", "process_posix.c", "process_windows.c",
)
# The runtime's file that every program links whole: what runs before main is in it, and nothing
# calls into it, so the linker would leave it out if it were taken from the archive with the rest.
RUNTIME_START = "start.c"


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


def archiver(target: Target) -> str:
    """The `ar` that goes with c_compiler(target): a cross toolchain's has its prefix."""
    compiler = c_compiler(target)[0]
    return compiler[:-len("gcc")] + "ar" if compiler.endswith("-gcc") else "ar"


def runtime_sources(target: Target) -> list[Path]:
    """The runtime's C files for `target` (see RUNTIME_FILES)."""
    left_out = "_posix.c" if target.os == 'windows' else "_windows.c"
    return [RUNTIME_DIR / name for name in RUNTIME_FILES if not name.endswith(left_out)]


# How the runtime's C is compiled, wherever it is (the tests build it the same way): optimized, as C11,
# with memory read as whatever type it is wanted as (the compiled code writes much of what the runtime
# reads), and signed arithmetic wrapping, as Hornet's does.
RUNTIME_CFLAGS = ["-O2", "-std=c11", "-fno-strict-aliasing", "-fwrapv"]


def runtime_objects(target: Target) -> list[Path]:
    """The runtime built for `target`, as what a program links with it: start.c's object, then an
    archive of the rest, from which the linker takes only what the program uses. Built once and
    reused: the names carry a hash of the runtime's sources and the commands, so any change builds
    them anew. Written atomically, so concurrent builds can share them."""
    command = c_compiler(target) + RUNTIME_CFLAGS + ["-c"]
    digest = hashlib.sha256(" ".join(command + [archiver(target)]).encode())
    sources = runtime_sources(target)
    for path in [*sources, *sorted(RUNTIME_DIR.glob("*.h"))]:
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    stem = cache_dir() / f"runtime-{target}-{digest.hexdigest()[:16]}"
    start, archive = stem.with_name(f"{stem.name}-start.o"), stem.with_name(f"{stem.name}.a")
    if not (start.exists() and archive.exists()):
        start.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=start.parent) as tmpdir:
            objects = []
            for source in sources:
                obj = Path(tmpdir) / f"{source.stem}.o"
                _run(command + [str(source), "-o", str(obj)], f"compiling runtime/{source.name}")
                objects.append(obj)
            built_start = Path(tmpdir) / f"{Path(RUNTIME_START).stem}.o"
            built_archive = Path(tmpdir) / "runtime.a"
            _run([archiver(target), "rcs", str(built_archive)] + [str(o) for o in objects if o != built_start],
                 "archiving the runtime")
            os.replace(built_start, start)
            os.replace(built_archive, archive)
    return [start, archive]


def build_executable(source_path: str, output_path: str, target=None) -> None:
    """Compile `source_path` for `target` (Target, `arch-os`, or None for the default) and link it
    with the runtime into `output_path`."""
    target = as_target(target)
    asm = compile_to_asm(source_path, target, require_main=True)
    cc = c_compiler(target)

    with tempfile.TemporaryDirectory() as tmpdir:
        asm_path = os.path.join(tmpdir, "program.s")
        write_output(asm_path, asm)  # (as bytes: see compile.py)

        # The output is written exactly where asked, whatever name the linker would choose.
        linked = os.path.join(tmpdir, executable_name("program", target))
        runtime = [str(path) for path in runtime_objects(target)]
        _run(cc + [asm_path, *runtime, "-o", linked] + link_flags(target), "linking")
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
