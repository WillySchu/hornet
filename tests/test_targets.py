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


def test_e2e_targets_follow_the_tier(monkeypatch):
    from target import default_target
    monkeypatch.delenv('HORNET_E2E_TARGETS', raising=False)
    runnable = [t for t in targets.RUNNABLE_TARGETS if t.arch in targets.IMPLEMENTED_ARCHES]
    monkeypatch.setattr(targets, 'TIER', 'full')
    assert targets._e2e_targets() == runnable
    monkeypatch.setattr(targets, 'TIER', 'standard')
    assert targets._e2e_targets() == [default_target()]
    monkeypatch.setenv('HORNET_E2E_TARGETS', 'aarch64-linux')
    assert targets._e2e_targets() == [Target('aarch64', 'linux')]


def test_tests_using_build_helpers_are_classified_end_to_end(tmp_path):
    import importlib.util
    import conftest
    path = tmp_path / 'fake_tests.py'
    path.write_text("def helper(src):\n    return compile_and_run(src)\n"
                    "def uses_helper():\n    helper('x')\n"
                    "def pure():\n    return 1 + 1\n")
    spec = importlib.util.spec_from_file_location('fake_tests', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = conftest._e2e_names(module)
    assert {'helper', 'uses_helper'} <= names and 'pure' not in names
