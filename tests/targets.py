"""Targets whose binaries this machine can build and run, for parametrizing tests over targets."""

import subprocess

import pytest

from build import can_run, run_prefix
from target import TARGET_NAMES, Target

RUNNABLE_TARGETS = [t for t in map(Target.parse, TARGET_NAMES) if can_run(t)]

# Parametrize a test over RUNNABLE_TARGETS as `target`.
each_runnable_target = pytest.mark.parametrize('target', RUNNABLE_TARGETS, ids=str)


def is_native(target: Target) -> bool:
    """Runs directly, not under an emulator (so tools like sanitizers work)."""
    return run_prefix(target) == []


def run_binary(target: Target, argv: list, **kwargs) -> subprocess.CompletedProcess:
    """Run a binary built for `target`, under qemu-user if it's a foreign architecture."""
    return subprocess.run(run_prefix(target) + [str(a) for a in argv], **kwargs)
