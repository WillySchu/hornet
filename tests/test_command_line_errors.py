"""What the command line does with what isn't a compile error or a compiler bug: a file it can't
read or write is one line and exit status 1; a program that nests very deeply compiles, and one
that nests beyond even that is one line too."""

import subprocess
import sys
from pathlib import Path

import pytest

import diagnostics
from diagnostics import CompileError, _with_room_to_recurse, file_error, run_cli
from tests.test_compiler import GCC_SKIP

ROOT = Path(__file__).resolve().parent.parent
OK = "def int main():\n    return 0\n"


def _run(tool: str, *args, cwd) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / tool), *args], capture_output=True, text=True, cwd=cwd,
                          timeout=240)


@pytest.mark.parametrize("tool,arguments,message", [
    ("compile.py", ["missing.ht", "-o", "out.s"], "error: can't read 'missing.ht': No such file or directory\n"),
    ("compile.py", ["missing.ht", "--dump", "tokens"], "error: can't read 'missing.ht': No such file or directory\n"),
    ("build.py", ["missing.ht", "-o", "out"], "error: can't read 'missing.ht': No such file or directory\n"),
    ("compile.py", ["a_directory", "-o", "out.s"], "error: can't read 'a_directory': Is a directory\n"),
    ("compile.py", ["ok.ht", "-o", "nowhere/out.s"], "error: can't write 'nowhere/out.s': No such file or directory\n"),
    ("compile.py", ["ok.ht", "--dump", "ir", "-o", "nowhere/out.ir"],
     "error: can't write 'nowhere/out.ir': No such file or directory\n"),
    ("build.py", ["ok.ht", "-o", "nowhere/out"], "error: can't write 'nowhere/out': No such file or directory\n"),
])
def test_a_file_that_cant_be_read_or_written_is_an_ordinary_error(tmp_path, tool, arguments, message):
    if sys.platform == 'win32':
        pytest.skip("the system's wording for these differs on Windows")
    (tmp_path / "ok.ht").write_text(OK)
    (tmp_path / "a_directory").mkdir()
    result = _run(tool, *arguments, cwd=tmp_path)
    assert (result.returncode, result.stdout, result.stderr) == (1, "", message)


def test_file_error_names_the_file_as_it_was_given(tmp_path):
    error = file_error("read", tmp_path / "x.ht", OSError(2, "No such file or directory"))
    assert isinstance(error, CompileError) and error.message.startswith("can't read '")
    assert error.message.endswith("x.ht': No such file or directory")


def _program(expression: str, declared: str = "int") -> str:
    return f"def int main():\n    int one = 1\n    {declared} n = {expression}\n    return 0\n"


DEEP = {
    "3,000 terms added": _program(" + ".join(["one"] * 3000)),
    "1,500 strings joined": _program(" + ".join(["'a'"] * 1500), "str"),
    "400 nested parentheses": _program("(" * 400 + "one" + ")" * 400),
    "500 nested ifs": "def int main():\n    int n = 1\n" + "".join(
        "    " * (i + 1) + "if n > 0:\n" for i in range(500)) + "    " * 501 + "n += 1\n    return 0\n",
    "an array literal 300 levels deep, indexed": (
        f"def int main():\n    {'[1]' * 300}int a = {'[' * 300}7{']' * 300}\n    return a{'[0]' * 300}\n"),
}


@pytest.mark.parametrize("source", DEEP.values(), ids=DEEP.keys())
def test_a_program_that_nests_deeply_compiles(tmp_path, source):
    # Each is far past Python's usual recursion limit, which the compiler's own recursion follows.
    (tmp_path / "p.ht").write_text(source)
    result = _run("compile.py", "p.ht", "-o", "p.s", cwd=tmp_path)
    assert (result.returncode, result.stderr) == (0, "")


@GCC_SKIP
def test_and_computes_what_it_says(tmp_path):
    (tmp_path / "p.ht").write_text(
        "def int main():\n    int one = 1\n    print(" + " + ".join(["one"] * 2000) + ")\n"
        "    print(len(" + " + ".join(["'ab'"] * 600) + "))\n    return 0\n")
    assert _run("build.py", "p.ht", "-o", "p", cwd=tmp_path).returncode == 0
    assert subprocess.run([str(tmp_path / "p")], capture_output=True, text=True).stdout == "2000\n1200\n"


def test_an_index_chain_costs_what_its_length_does(tmp_path):
    # Each level's base was once checked twice: 2**64 checks for this one.
    (tmp_path / "p.ht").write_text(
        f"def int main():\n    {'[1]' * 64}int a\n    a{'[0]' * 64} = 3\n    return a{'[0]' * 64}\n")
    assert _run("compile.py", "p.ht", "-o", "p.s", cwd=tmp_path).returncode == 0


def test_room_to_recurse_hands_back_results_and_errors():
    def depth(n: int) -> int:
        return 0 if n == 0 else 1 + depth(n - 1)
    assert _with_room_to_recurse(lambda: depth(50_000)) == 50_000
    with pytest.raises(ValueError, match="handed on"):
        _with_room_to_recurse(lambda: (_ for _ in ()).throw(ValueError("handed on")))
    assert sys.getrecursionlimit() < diagnostics._RECURSION_LIMIT  # ... and leaves the limit as it found it


def test_nesting_beyond_even_that_is_one_line(capfd):
    def forever():
        return forever()
    with pytest.raises(SystemExit) as exit_info:
        run_cli(forever)
    assert exit_info.value.code == 1
    assert capfd.readouterr().err == (
        "error: this program nests more deeply than the compiler can follow (an expression or a block many "
        "thousands of levels deep)\n")
