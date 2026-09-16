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
    'array_heavy': 158,  # bubble sort, plus indexing directly into a bare bracketed-list literal (`[10, 20, 30][j % 3]`, no variable in between), in the innermost loop
    'struct_heavy': 13,  # per-field struct comparison (distSquared), plus struct equality itself: a genuine mismatch and a self-comparison, in a hot loop
    'string_heavy': 1,  # r1 == r2 -- two independently-built copies of the same repeated string
    'copy_heavy': 223,  # whole-array/whole-struct/whole-slice copy, slice production, indexing through a slice-typed struct field/array element, chained slice production, repeated append, a no-initializer slice, slice-vs-none comparison, and a bare bracketed-list literal resolved to SLICE by context, in a hot loop -- verified against a Go-style aliasing simulation, not independent-copy semantics
    'calling_convention_heavy': 99,  # array-typed, slice-typed, and array-typed-struct-field function arguments (the last exercising the Field-argument fix), plus composite-returning function calls including forwarding one level deep, a bare local Variable return, an all-scalar positional struct-literal return, a nested struct-literal return, and a composite-returning call used directly as an addressable base (indexed, field-accessed, and sliced -- the last exercising the always-heap-allocate-for-slice-production path), in a hot loop
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
