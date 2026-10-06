"""Targets whose binaries this machine can build and run, for parametrizing tests over targets."""

import atexit
import os
import shutil
import signal
import subprocess
from pathlib import Path

import pytest

from build import can_run, run_prefix
from target import IMPLEMENTED_ARCHES, TARGET_NAMES, Target, default_target

RUNNABLE_TARGETS = [t for t in map(Target.parse, TARGET_NAMES) if can_run(t)]

# Parametrize a test over RUNNABLE_TARGETS as `target`.
each_runnable_target = pytest.mark.parametrize('target', RUNNABLE_TARGETS, ids=str)


def is_native(target: Target) -> bool:
    """Runs directly, not under an emulator (so tools like sanitizers work)."""
    return run_prefix(target) == []


# How `abort()` ends a process on Windows: exit code 3, where other systems report SIGABRT.
WINDOWS_ABORT_EXIT_CODE = 3


def _keep_a_wine_server_running() -> None:
    """Without a Wine server already running, each Windows program starts one, and with it Wine's
    services, which inherit the program's output pipes and hold them open for seconds after it has
    exited: whoever reads its output waits that long. So one server is started here, and its
    services by a first program that has no pipes to inherit; every later program then starts and
    finishes in milliseconds. The server stays until the tests end."""
    prefix = next((run_prefix(t) for t in RUNNABLE_TARGETS if t.os == 'windows'), None)
    if not prefix:  # no Windows target here, or Windows itself
        return
    wine = Path(shutil.which(prefix[-1]))
    server = next(
        (str(path) for path in (wine.with_name('wineserver64'), wine.with_name('wineserver')) if path.exists()),
        shutil.which('wineserver')
    )
    if server is None:
        return
    detached = dict(env={**os.environ, 'WINEDEBUG': '-all'}, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
    subprocess.run([server, '-p'], **detached)
    try:
        subprocess.run([str(wine), 'cmd', '/c', 'exit'], timeout=300, **detached)  # starts the services
    except (OSError, subprocess.TimeoutExpired):
        pass
    atexit.register(lambda: subprocess.run([server, '-k'], **detached))


_keep_a_wine_server_running()


def run_binary(target: Target, argv: list, **kwargs) -> subprocess.CompletedProcess:
    """Run a binary built for `target`: under qemu-user if it's a foreign architecture, under Wine
    if it's for Windows. The result compares equal across targets: qemu's own report of a fatal
    signal is dropped from stderr, and a Windows panic is reported as the SIGABRT it is elsewhere."""
    prefix = run_prefix(target)
    result = subprocess.run(prefix + [str(a) for a in argv], **kwargs)
    if prefix and isinstance(result.stderr, str):
        result.stderr = ''.join(line for line in result.stderr.splitlines(keepends=True)
                                if not line.startswith('qemu: uncaught target signal'))
    if (
            target.os == 'windows'
            and result.returncode == WINDOWS_ABORT_EXIT_CODE
            and b'panic: ' in (
                result.stderr.encode('latin-1') if isinstance(result.stderr, str) else result.stderr or b''
            )
    ):
        result.returncode = -signal.SIGABRT
    return result


# The test tier (see conftest.py): 'quick', 'standard', or 'full'.
TIER = os.environ.get('HORNET_TEST_TIER', 'full')


def _e2e_targets() -> list:
    if os.environ.get('HORNET_E2E_TARGETS'):
        return [Target.parse(n) for n in os.environ['HORNET_E2E_TARGETS'].split(',')]
    # Another system's binaries (Windows under Wine) take much longer to start: only on request.
    targets = [t for t in RUNNABLE_TARGETS if t.arch in IMPLEMENTED_ARCHES and t.os == default_target().os]
    if TIER == 'full':
        return targets
    native = default_target()
    return [native] if native in targets else targets[:1]


# Targets every end-to-end program is built and run for: this machine's own target, or with
# --full every runnable target of this machine's system with a complete backend.
# HORNET_E2E_TARGETS (comma-separated arch-os names) overrides this, e.g. to test a backend that is
# still in progress, or `x86_64-windows` under Wine.
E2E_TARGETS = _e2e_targets()

# Parametrize a test over E2E_TARGETS as `target`.
each_e2e_target = pytest.mark.parametrize('target', E2E_TARGETS, ids=str)


def on_every_target(build_and_run, label: str = '', agree: bool = True) -> subprocess.CompletedProcess:
    """Call build_and_run(target) for each E2E target; unless `agree` is False (output that
    legitimately varies, such as addresses), they must agree on exit status and output.
    Returns the first target's result."""
    results = [(t, build_and_run(t)) for t in E2E_TARGETS]
    first_target, first = results[0]
    for target, r in results[1:] if agree else []:
        assert (r.returncode, r.stdout, r.stderr) == (first.returncode, first.stdout, first.stderr), (
            f"{target} and {first_target} disagree{': ' + label if label else ''}\n"
            f"{first_target}: {first.returncode} {first.stdout!r} {first.stderr!r}\n"
            f"{target}: {r.returncode} {r.stdout!r} {r.stderr!r}")
    return first


def build_and_run(source: str, target: Target, stdin: str = '', timeout: int = 20) -> subprocess.CompletedProcess:
    """Build `source` (full driver) for one target and run it."""
    import tempfile
    from pathlib import Path
    from build import build_executable
    with tempfile.TemporaryDirectory() as tmp:
        src, exe = Path(tmp) / 'p.ht', Path(tmp) / 'p'
        src.write_text(source)
        build_executable(str(src), str(exe), target=target)
        return run_binary(target, [exe], input=stdin, capture_output=True, text=True, timeout=timeout)
