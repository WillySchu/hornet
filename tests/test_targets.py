"""tests/targets.py: running end-to-end programs on every target."""

import subprocess

import pytest

import tests.targets as targets
from target import Target


def test_on_every_target_requires_agreement(monkeypatch):
    a, b = Target('x86_64', 'linux'), Target('aarch64', 'linux')
    monkeypatch.setattr(targets, 'E2E_TARGETS', [a, b])
    same = lambda t: subprocess.CompletedProcess([], 0, 'ok\n', '')
    assert targets.on_every_target(same).stdout == 'ok\n'
    differ = lambda t: subprocess.CompletedProcess([], 0, 'ok\n' if t == a else 'no\n', '')
    with pytest.raises(AssertionError, match='aarch64-linux and x86_64-linux disagree'):
        targets.on_every_target(differ)


def test_e2e_targets_are_runnable_and_implemented():
    from target import IMPLEMENTED_ARCHES
    assert targets.E2E_TARGETS, 'no end-to-end target can run on this machine'
    assert all(t.arch in IMPLEMENTED_ARCHES and t in targets.RUNNABLE_TARGETS for t in targets.E2E_TARGETS)
