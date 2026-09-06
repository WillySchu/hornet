"""Sanity checks for benchmarks/programs/*.ht.

These programs exist to be MEASURED (see benchmarks/run_benchmarks.py),
not to test correctness directly -- but they still need to keep
working as the language and compiler evolve. This file only confirms
each one still compiles and produces its known-correct result; it
says nothing about performance and takes no timing measurements
itself, so it stays fast and deterministic enough to run alongside
the rest of the suite. If a benchmark's own expected value ever needs
to change here, benchmarks/baseline.json almost certainly needs a
fresh --save-baseline run too, since the program itself changed.
"""

from pathlib import Path

import pytest

from tests.test_compiler import compile_and_run, GCC_SKIP

PROGRAMS_DIR = Path(__file__).parent.parent / 'benchmarks' / 'programs'

# Each expected value was independently verified against a pure-Python
# reference implementation (not just "whatever the compiler happened
# to produce") before being recorded here -- see the exit-code-is-
# truncated-to-a-byte note on struct_heavy below for the one subtlety
# that came up doing that.
EXPECTED_EXIT_CODES = {
    'arithmetic_heavy': 217,
    'recursive_fibonacci': 231,
    'loop_accumulator': 192,
    'array_heavy': 185,
    'struct_heavy': 19,
    'string_heavy': 1,  # r1 == r2 -- two independently-built copies of the same repeated string
}


@GCC_SKIP
@pytest.mark.parametrize('name', sorted(EXPECTED_EXIT_CODES))
def test_benchmark_still_compiles_and_runs_correctly(name):
    source = (PROGRAMS_DIR / f'{name}.ht').read_text()
    result = compile_and_run(source)
    assert result.returncode == EXPECTED_EXIT_CODES[name]


def test_every_benchmark_program_has_an_expected_exit_code():
    """Catches the other direction of bit-rot: a new .ht file dropped
    into benchmarks/programs/ without a matching entry above would
    otherwise just never get checked here at all."""
    on_disk = {p.stem for p in PROGRAMS_DIR.glob('*.ht')}
    assert on_disk == set(EXPECTED_EXIT_CODES)
