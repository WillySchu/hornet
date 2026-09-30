"""Every composite type x context x value shape either fails semantic analysis cleanly or
compiles; a quarter of the compiling programs are also run and their output checked."""

import pytest

from compile import compile_to_asm
from diagnostics import CompileError
from tests.shape_matrix import programs
from tests.test_compiler import GCC_SKIP, compile_and_run

PROGRAMS = list(programs())


@pytest.mark.parametrize('name,source,expected', PROGRAMS, ids=[p[0] for p in PROGRAMS])
def test_no_internal_compiler_error(name, source, expected, tmp_path):
    path = tmp_path / 'p.ht'
    path.write_text(source)
    try:
        compile_to_asm(str(path), platform='linux')
    except CompileError:
        pass  # a clean rejection is fine; anything else fails the test


RUN = [p for i, p in enumerate(PROGRAMS) if i % 4 == 0]


@GCC_SKIP
@pytest.mark.parametrize('name,source,expected', RUN, ids=[p[0] for p in RUN])
def test_sampled_programs_run_correctly(name, source, expected, tmp_path):
    path = tmp_path / 'p.ht'
    path.write_text(source)
    try:
        compile_to_asm(str(path), platform='linux')
    except CompileError:
        pytest.skip('rejected by semantic analysis')
    result = compile_and_run(source)
    assert result.returncode == 0, result.stderr
    if expected is not None:
        assert result.stdout == expected
