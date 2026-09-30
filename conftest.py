"""Test tiers. `pytest --quick`: unit tests only, nothing is built or run. `pytest`: adds end-to-end
tests on this machine's own target. `pytest --full`: adds every runnable target (qemu-user,
Rosetta 2) and the slow tests (benchmark reruns and runner, scrambled formatter programs,
sanitizers). HORNET_E2E_TARGETS overrides the targets in any tier."""

import inspect
import os
import re

import pytest

# Slow tests, run only with --full: node id fragments.
SLOW = (
    'tests/test_benchmarks.py::test_benchmark_runner',
    'tests/test_hfmt.py::test_formatting_scrambled_programs_keeps_meaning',
    'tests/test_hfmt.py::test_formatting_repo_files_is_stable_and_keeps_meaning',
    'tests/runtime/test_runtime_isolated_c.py::test_isolated_c_tests_pass_under_sanitizers',
)

# End-to-end tests the rules in _is_e2e can't see (they call a C compiler directly).
E2E_FILES = ('tests/runtime/', 'tests/backend/aarch64/test_programs.py', 'tests/test_examples.py')


def pytest_addoption(parser):
    group = parser.addmutuallyexclusivegroup() if hasattr(parser, 'addmutuallyexclusivegroup') else parser
    group.addoption('--quick', action='store_true', help='Unit tests only: build and run nothing')
    group.addoption('--full', action='store_true', help='Every runnable target, and the slow tests')


def pytest_configure(config):
    if config.getoption('quick') and config.getoption('full'):
        raise pytest.UsageError('--quick and --full are exclusive')
    tier = 'quick' if config.getoption('quick') else 'full' if config.getoption('full') else 'standard'
    os.environ['HORNET_TEST_TIER'] = tier  # read by tests/targets.py; inherited by xdist workers
    config.addinivalue_line('markers', 'e2e: builds and runs programs (skipped by --quick)')
    config.addinivalue_line('markers', 'slow: run only by --full')
    if tier == 'quick':
        _forbid_building()


# Helpers that build or run programs; a test (or module helper) that mentions one is end-to-end.
E2E_HELPERS = {
    'compile_and_run', 'assert_exit_code', 'assert_program_exit_code', 'assert_stdout', 'assert_program_stdout',
    'assert_panics', 'assert_crashes_with_sigabrt', '_compile_to_binary', '_run_binary', 'build_and_run',
    'on_every_target', 'build_executable', 'run_binary', 'runtime_object', 'hfmt',
}
_module_e2e_names: dict = {}


def _source(obj) -> str:
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        return ''


def _mentions(source: str, names) -> bool:
    return any(re.search(rf'\b{re.escape(n)}\b', source) for n in names)


def _e2e_names(module) -> set:
    """E2E_HELPERS plus this module's own functions that use them, transitively."""
    if module not in _module_e2e_names:
        names = set(E2E_HELPERS)
        functions = [(n, _source(f)) for n, f in inspect.getmembers(module, inspect.isfunction)
                     if getattr(f, '__module__', None) == module.__name__]
        changed = True
        while changed:
            changed = False
            for name, source in functions:
                if name not in names and _mentions(source, names):
                    names.add(name)
                    changed = True
        _module_e2e_names[module] = names
    return _module_e2e_names[module]


def _is_e2e(item) -> bool:
    if any(item.nodeid.startswith(f) for f in E2E_FILES) or item.get_closest_marker('e2e'):
        return True
    if any(m.name == 'skipif' and 'gcc' in str(m.kwargs.get('reason', '')) for m in item.iter_markers()):
        return True
    function = getattr(item, 'function', None)
    return function is not None and _mentions(_source(function), _e2e_names(item.module))


def pytest_collection_modifyitems(config, items):
    tier = os.environ['HORNET_TEST_TIER']
    keep, drop = [], []
    for item in items:
        slow = any(item.nodeid.startswith(s) for s in SLOW) or item.get_closest_marker('slow')
        if slow and tier != 'full' or tier == 'quick' and _is_e2e(item):
            drop.append(item)
        else:
            keep.append(item)
    if drop:
        config.hook.pytest_deselected(items=drop)
        items[:] = keep


def _forbid_building():
    """In --quick, anything that builds a program fails, so a test missing from the e2e rules is
    noticed instead of silently slowing the quick tier down."""
    import build

    def refuse(*args, **kwargs):
        raise AssertionError('this test builds a program: mark it e2e (see conftest.py) or run without --quick')
    build.runtime_object = refuse
    build.build_executable = refuse
