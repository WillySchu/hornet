"""A source file is read as bytes, one to a character: a string literal holds the bytes the file
holds, whatever the file's encoding, and columns count bytes. Outside strings and comments source is
ASCII. A diagnostic is written back as those bytes, with its caret placed for UTF-8."""

import subprocess
import sys
from pathlib import Path

import pytest

from build import build_executable, executable_name
from diagnostics import CompileError, _columns_shown, format_error
from dump import dump
from lexer import LexError, lex
from tests.targets import on_every_target, run_binary
from tests.test_compiler import GCC_SKIP

ROOT = Path(__file__).resolve().parent.parent

E_ACUTE, U_ACUTE, CJK = "\u00e9".encode(), "\u00fa".encode(), "\u4e2d".encode()   # 2, 2, and 3 bytes of UTF-8


def _file(tmp_path, source: bytes, name: str = "p.ht") -> str:
    (tmp_path / name).write_bytes(source)
    return str(tmp_path / name)


def _compile(path: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / 'compile.py'), path, '-o', path + '.s'], capture_output=True,
                          cwd=Path(path).parent)


@GCC_SKIP
def test_a_string_literal_holds_the_bytes_the_file_holds(tmp_path):
    source = (
        b"# a comment may hold anything: " + CJK + b" \xff\n"
        b"def int main():\n"
        b"    str s = 'caf" + E_ACUTE + b" " + CJK + b"'\n"
        b"    str raw = 'not UTF-8: \xe9\xff'\n"                    # nor need a file be UTF-8 at all
        b"    print(len(s))\n"
        b"    print(s)\n"
        b"    print(raw)\n"
        b"    print(s == 'caf\\xc3\\xa9 \\xe4\\xb8\\xad')\n"        # the same bytes, spelled out
        b"    print('" + E_ACUTE + b"' < '" + U_ACUTE + b"')\n"
        b"    [3]int xs = [1, 2, 3]\n"
        b"    int i = len('" + CJK + b"') + 4\n"
        b"    print('" + CJK + b"' + '!')\n"
        b"    return len('" + CJK + b"') + xs[i]\n"
    )
    path = _file(tmp_path, source)

    def run(target):
        exe = tmp_path / executable_name(f"p-{target}", target)
        build_executable(path, str(exe), target=target)
        return run_binary(target, [exe], capture_output=True)
    result = on_every_target(run)
    assert result.stdout == (b"9\ncaf" + E_ACUTE + b" " + CJK + b"\nnot UTF-8: \xe9\xff\ntrue\ntrue\n" + CJK + b"!\n")
    # A position counts bytes, at run time as when compiling: `xs` is the 25th byte of its line (the 23rd character).
    assert result.stderr == b"p.ht:13:25: panic: index out of bounds: index 7, length 3\n"


@pytest.mark.parametrize("line,col,byte", [
    (b"    int caf" + E_ACUTE + b" = 1\n", 12, "0xC3"),
    (b"    int caf" + U_ACUTE + b" = 1\n", 12, "0xC3"),               # (both bytes of this one are Latin-1 letters)
    (b"    int " + CJK + b" = 1\n", 9, "0xE4"),
    (b"    int x = 1\xc2\xa0+ 2\n", 14, "0xC2"),                       # a no-break space
    (b"    int x = 1 \xe9 2\n", 15, "0xE9"),
])
def test_outside_strings_and_comments_source_is_ascii(tmp_path, line, col, byte):
    path = _file(tmp_path, b"def int main():\n" + line + b"    return 0\n")
    with pytest.raises(LexError, match=f"Unexpected byte {byte} -- outside strings and comments, source is ASCII") as e:
        lex(path)
    assert (e.value.line, e.value.col) == (2, col)


def test_a_byte_literal_is_one_byte(tmp_path):
    path = _file(tmp_path, b'def int main():\n    byte b = "' + E_ACUTE + b'"\n    return int(b)\n')
    result = _compile(path)
    assert result.returncode == 1
    assert result.stderr.startswith(b"p.ht:2:14: error: A byte literal must resolve to exactly one byte (0-255), got '"
                                    + E_ACUTE + b"'\n")


def test_a_diagnostic_is_the_file_s_bytes_with_the_caret_under_what_it_names(tmp_path):
    line = b"    str s = 'caf" + E_ACUTE + b" " + CJK + b"' + missing"
    result = _compile(_file(tmp_path, b"def int main():\n" + line + b"\n    return 0\n"))
    column = len(line) - len(b"missing") + 1            # in bytes: 27
    shown = column - 1 - 3                              # three of those bytes continue a character
    assert (result.returncode, result.stdout) == (1, b"")
    assert result.stderr == (
        b"p.ht:2:%d: error: Reference to undeclared variable 'missing'\n" % column
        + b"    " + line + b"\n"
        + b"    " + b" " * shown + b"^\n")


def test_the_caret_s_column():
    assert _columns_shown("plain") == 5
    assert _columns_shown((b"caf" + E_ACUTE + b" " + CJK).decode('latin-1')) == 6       # c a f e-acute space CJK
    assert _columns_shown(b"\xe9\xff".decode('latin-1')) == 2                            # not UTF-8: a column each
    error = CompileError("m", None, 0, 0)
    assert format_error(error) == "error: m"


def test_a_byte_order_mark_is_named(tmp_path):
    path = _file(tmp_path, b"\xef\xbb\xbfdef int main():\n    return 0\n")
    with pytest.raises(LexError, match="starts with a UTF-8 byte-order mark, which Hornet source doesn't use") as e:
        lex(path)
    assert (e.value.line, e.value.col) == (1, 1)
    assert _compile(path).stderr.startswith(b"p.ht:1:1: error: This file starts with a UTF-8 byte-order mark")


def test_the_dumps_keep_the_bytes(tmp_path):
    path = _file(tmp_path, b"def int main():\n    print('" + CJK + b"')\n    return 0\n")
    assert "2:11 STRING '" + CJK.decode('latin-1') + "'\n" in dump(path, 'tokens')
    command = [sys.executable, str(ROOT / 'compile.py'), path, '--dump', 'tokens']
    assert b"2:11 STRING '" + CJK + b"'\n" in subprocess.run(command, capture_output=True).stdout
    assert "StrLit value='\\xe4\\xb8\\xad' : str" in dump(path, 'typed')      # (a literal is shown escaped)
