"""Tests for build.py's own build_executable: the one entry point
that compiles a Hornet source file AND links it against a freshly-
compiled runtime.c into a single, runnable executable -- the piece of
the toolchain that didn't exist at all before runtime.c did (see
build.py's own module docstring).
"""
import shutil
import subprocess
import tempfile

import pytest

import build

GCC_AVAILABLE = shutil.which("gcc") is not None
pytestmark = pytest.mark.skipif(not GCC_AVAILABLE, reason="gcc not available")


def _write(tmpdir, name, content):
    path = f"{tmpdir}/{name}"
    with open(path, "w") as f:
        f.write(content)
    return path


def test_build_executable_compiles_and_runs_correctly():
    with tempfile.TemporaryDirectory() as tmpdir:
        source = _write(tmpdir, "program.ht", "def int main():\n    int x = 5\n    int y = 10\n    return x + y\n")
        binary = f"{tmpdir}/program"
        build.build_executable(source, binary)
        result = subprocess.run([binary], capture_output=True)
        assert result.returncode == 15


def test_build_executable_propagates_semantic_errors_directly():
    """A SemanticError (or SyntaxError) from the ordinary Hornet
    pipeline is a lexer/parser/semantic-analysis failure, not a build
    failure -- it should propagate as itself, not get wrapped into a
    BuildError alongside actual compiling/linking failures."""
    with tempfile.TemporaryDirectory() as tmpdir:
        source = _write(tmpdir, "program.ht", "def int main():\n    return undeclaredVariable\n")
        binary = f"{tmpdir}/program"
        with pytest.raises(Exception) as exc_info:
            build.build_executable(source, binary)
        assert not isinstance(exc_info.value, build.BuildError)


def test_build_executable_raises_build_error_with_actionable_message_on_bad_runtime_c(monkeypatch, tmp_path):
    broken_runtime_c = tmp_path / "broken_runtime.c"
    broken_runtime_c.write_text("this is not valid C at all !!!\n")
    monkeypatch.setattr(build, "RUNTIME_C_PATH", broken_runtime_c)

    source = tmp_path / "program.ht"
    source.write_text("def int main():\n    return 0\n")
    binary = tmp_path / "program"

    with pytest.raises(build.BuildError) as exc_info:
        build.build_executable(str(source), str(binary))
    message = str(exc_info.value)
    assert "compiling runtime.c" in message
    assert "gcc" in message
