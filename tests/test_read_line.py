"""stdlib/os.ht's read_line (standard input a line at a time) and is_terminal."""

from tests.targets import build_and_run, on_every_target
from tests.test_compiler import GCC_SKIP

ECHO = (
    "from 'os' import read_line, read_stdin, is_terminal\n"
    "from 'errors' import Error\n"
    "def int main():\n"
    "    print(is_terminal(0))\n"
    "    int lines = 0\n"
    "    while true:\n"
    "        match read_line() as line:\n"
    "            is str:\n"
    "                lines += 1\n"
    "                print('[' + line + ']')\n"
    "                if line == 'rest:':\n"         # read_stdin continues from where the lines stopped
    "                    match read_stdin() as rest:\n"
    "                        is str:\n"
    "                            print(len(rest))\n"
    "                        is Error:\n"
    "                            return 2\n"
    "            is Error:\n"
    "                return 1\n"
    "            is none:\n"
    "                break\n"
    "    return lines\n"
)


@GCC_SKIP
def test_read_line():
    def run(stdin):
        return on_every_target(lambda target: build_and_run(ECHO, target, stdin=stdin))
    # Line endings are dropped ("\\n" or "\\r\\n"), an empty line is a line, and a last line needs no ending.
    result = run("one\r\n\ntwo words\nlast")
    assert (result.returncode, result.stdout) == (4, "false\n[one]\n[]\n[two words]\n[last]\n")
    assert run("").returncode == 0
    assert run("\n").stdout == "false\n[]\n"
    long_line = "x" * 5000
    assert run(long_line + "\n").stdout == f"false\n[{long_line}]\n"
    result = run("a\nrest:\n1234\n5678\n")
    assert (result.returncode, result.stdout) == (2, "false\n[a]\n[rest:]\n10\n")
