"""tools/hfmt, the Hornet formatter written in Hornet."""

import subprocess
from pathlib import Path

import pytest

from build import build_executable
from lexer import Lexer, TokenType
from tests.test_compiler import ASM_PLATFORM, GCC_SKIP

ROOT = Path(__file__).resolve().parent.parent
SOURCES = sorted(
    p for d in ('stdlib', 'examples', 'benchmarks/programs', 'tools/hfmt') for p in (ROOT / d).glob('*.ht'))


@pytest.fixture(scope='session')
def hfmt(tmp_path_factory):
    exe = tmp_path_factory.mktemp('hfmt') / 'hfmt'
    build_executable(str(ROOT / 'tools' / 'hfmt' / 'main.ht'), str(exe), platform=ASM_PLATFORM)
    return exe


def _run(hfmt, *args, stdin=None):
    return subprocess.run([str(hfmt), *args], input=stdin, capture_output=True, text=True, timeout=20)


_KEYWORD_TYPES = set(Lexer('').keywords.values())


def _python_tokens(src: str) -> list:
    """The compiler lexer's tokens, in hfmt's --tokens vocabulary (no INDENT/DEDENT/EOF)."""
    out = []
    for t in Lexer(src).tokenize():
        if t.type in (TokenType.INDENT, TokenType.DEDENT, TokenType.EOF):
            continue
        if t.type == TokenType.NEWLINE:
            if t.val:  # a synthesized final NEWLINE has no text
                out.append(('NEWLINE', t.line, t.col, ''))
            continue
        kind = {TokenType.IDENTIFIER: 'IDENT', TokenType.NUMBER: 'NUMBER', TokenType.STRING: 'STRING',
                TokenType.BYTE: 'BYTE'}.get(t.type, 'KEYWORD' if t.type in _KEYWORD_TYPES else 'OP')
        out.append((kind, t.line, t.col, t.val))
    return out


def _hfmt_tokens(dump: str) -> list:
    """hfmt --tokens output without comments, EOF, or newlines inside brackets."""
    out = []
    for line in dump.splitlines():
        kind, lnum, col, text = (line.split(' ', 3) + [''])[:4]
        if kind in ('COMMENT', 'EOF') or (kind == 'NEWLINE' and text != '0'):
            continue
        out.append((kind, int(lnum), int(col), '' if kind == 'NEWLINE' else text))
    return out


@GCC_SKIP
@pytest.mark.parametrize('path', SOURCES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_tokens_match_the_compiler_lexer(hfmt, path):
    r = _run(hfmt, '--tokens', str(path))
    assert r.returncode == 0, r.stderr
    assert _hfmt_tokens(r.stdout) == _python_tokens(path.read_text())


_EVERY_TOKEN = (
    "a <<= b >>= c == d != e >= f <= g << h >> i += j -= k *= l /= m %= n &= o |= p ^= q\n"
    "( ) [ ] { } > < : ; , = + - * / % ~ & | ^ . x.y a[0]b(c){d}\n"
    "1 23 4.5 6. 7.x _a9 'it\\'s' '' \"\\n\" \"\\x41\" 'a\\\\'\n"
    "def int int8 uint8 int64 int32 byte str return and or not bool true false if else elif for while break\n"
    "continue none struct type is in match as import from extern intrinsic dict const deff intx\n"
    "\tx\r\n"
)


@GCC_SKIP
def test_every_token_kind_matches_the_compiler_lexer(hfmt):
    r = _run(hfmt, '--tokens', stdin=_EVERY_TOKEN)
    assert r.returncode == 0, r.stderr
    assert _hfmt_tokens(r.stdout) == _python_tokens(_EVERY_TOKEN)


@GCC_SKIP
def test_tokens_keep_comments_and_bracket_depth(hfmt):
    r = _run(hfmt, '--tokens', stdin="x = (1 +  # c\n    2)\n")
    assert r.stdout.splitlines() == [
        'IDENT 1 1 x', 'OP 1 3 =', 'OP 1 5 (', 'NUMBER 1 6 1', 'OP 1 8 +', 'COMMENT 1 11 # c', 'NEWLINE 1 14 1',
        'NUMBER 2 5 2', 'OP 2 6 )', 'NEWLINE 2 7 0', 'EOF 3 1 ']


@GCC_SKIP
@pytest.mark.parametrize('src,message', [
    ("x = 'abc\n", "<stdin>:1:5: error: unterminated literal"),
    ("x = \"a", "<stdin>:1:5: error: unterminated literal"),
    ("f(1))\n", "<stdin>:1:5: error: unmatched ')'"),
    ("f(1]\n", "<stdin>:1:4: error: unmatched ']'"),
    ("x = [1,\n  2\n", "<stdin>:1:5: error: unclosed '['"),
    ("x = a ! b\n", "<stdin>:1:7: error: unexpected character '!'"),
])
def test_lex_errors_are_reported_with_position(hfmt, src, message):
    r = _run(hfmt, '--tokens', stdin=src)
    assert r.returncode == 1
    assert r.stderr == message + '\n'


@GCC_SKIP
def test_usage_error(hfmt):
    assert _run(hfmt).returncode == 2
