"""Targets whose binaries this machine can build and run, for parametrizing tests over targets."""

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
    """Run a binary built for `target`, under qemu-user if it's a foreign architecture."""
    return subprocess.run(run_prefix(target) + [str(a) for a in argv], **kwargs)


# Runnable targets that have a backend: every end-to-end program is built and run for each.
E2E_TARGETS = [t for t in RUNNABLE_TARGETS if t.arch in IMPLEMENTED_ARCHES]

# Parametrize a test over E2E_TARGETS as `target`.
each_e2e_target = pytest.mark.parametrize('target', E2E_TARGETS, ids=str)


def on_every_target(build_and_run, label: str = '') -> subprocess.CompletedProcess:
    """Call build_and_run(target) for each E2E target; they must agree on exit status and output.
    Returns the first target's result."""
    results = [(t, build_and_run(t)) for t in E2E_TARGETS]
    first_target, first = results[0]
    for target, r in results[1:]:
        assert (r.returncode, r.stdout, r.stderr) == (first.returncode, first.stdout, first.stderr), (
            f"{target} and {first_target} disagree{': ' + label if label else ''}\n"
            f"{first_target}: {first.returncode} {first.stdout!r} {first.stderr!r}\n"
            f"{target}: {r.returncode} {r.stdout!r} {r.stderr!r}")
    return first
