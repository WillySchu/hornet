"""Targets whose binaries this machine can build and run, for parametrizing tests over targets."""

import os
import subprocess

import pytest

from build import can_run, run_prefix
from target import IMPLEMENTED_ARCHES, TARGET_NAMES, Target

RUNNABLE_TARGETS = [t for t in map(Target.parse, TARGET_NAMES) if can_run(t)]

# Parametrize a test over RUNNABLE_TARGETS as `target`.
each_runnable_target = pytest.mark.parametrize('target', RUNNABLE_TARGETS, ids=str)


def is_native(target: Target) -> bool:
    """Runs directly, not under an emulator (so tools like sanitizers work)."""
    return run_prefix(target) == []


def run_binary(target: Target, argv: list, **kwargs) -> subprocess.CompletedProcess:
    """Run a binary built for `target`, under qemu-user if it's a foreign architecture (without
    qemu's own report of a fatal signal in stderr, so output compares equal across targets)."""
    prefix = run_prefix(target)
    result = subprocess.run(prefix + [str(a) for a in argv], **kwargs)
    if prefix and isinstance(result.stderr, str):
        result.stderr = ''.join(line for line in result.stderr.splitlines(keepends=True)
                                if not line.startswith('qemu: uncaught target signal'))
    return result


# Runnable targets that have a complete backend: every end-to-end program is built and run for each.
# HORNET_E2E_TARGETS (comma-separated arch-os names) overrides this, e.g. to run the whole suite
# against a backend that is still in progress.
E2E_TARGETS = ([Target.parse(n) for n in os.environ['HORNET_E2E_TARGETS'].split(',')]
               if os.environ.get('HORNET_E2E_TARGETS')
               else [t for t in RUNNABLE_TARGETS if t.arch in IMPLEMENTED_ARCHES])

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
