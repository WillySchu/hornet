"""tools/hfmt, the Hornet formatter written in Hornet."""

import re
import subprocess
from pathlib import Path

import pytest

from build import build_executable, run_prefix
from lexer import Lexer, TokenType
from tests.test_compiler import GCC_SKIP
from tests.targets import E2E_TARGETS

ROOT = Path(__file__).resolve().parent.parent
SOURCES = sorted(
    p for d in ('stdlib', 'examples', 'benchmarks/programs', 'tools/hfmt') for p in (ROOT / d).glob('*.ht'))


@pytest.fixture(scope='session', params=E2E_TARGETS, ids=str)
def hfmt(request, tmp_path_factory):
    """Command running hfmt built for each E2E target (under qemu for a foreign architecture)."""
    exe = tmp_path_factory.mktemp(f'hfmt-{request.param}') / 'hfmt'
    build_executable(str(ROOT / 'tools' / 'hfmt' / 'main.ht'), str(exe), target=request.param)
    return run_prefix(request.param) + [str(exe)]


def _run(hfmt, *args, stdin=None):
    return subprocess.run([*hfmt, *args], input=stdin, capture_output=True, text=True, timeout=20)


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
    ("x = 'abc", "<stdin>:1:5: error: unterminated literal"),
    ("x = \"a", "<stdin>:1:5: error: unterminated literal"),
    ("x = 'a\nb'\n", "<stdin>:1:5: error: newline in literal (write \\n)"),
    ("x = \"a\\\nb\"\n", "<stdin>:1:5: error: newline in literal (write \\n)"),
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
    assert _run(hfmt, '--tokens', 'a', 'b').returncode == 2
    assert _run(hfmt, '--bogus').returncode == 2
    assert _run(hfmt, '--tokens', '--check').returncode == 2
    assert _run(hfmt, '-', '-', stdin='').returncode == 2


def test_flags_may_follow_files_and_dash_names_stdin(hfmt, tmp_path):
    messy, tidy = "def int main():\n  return 0\n", "def int main():\n    return 0\n"
    path = tmp_path / 'a.ht'
    path.write_text(messy)
    r = _run(hfmt, str(path), '--check')
    assert (r.returncode, r.stdout) == (1, f"{path}\n") and path.read_text() == messy
    assert _run(hfmt, '-', stdin=messy).stdout == tidy
    r = _run(hfmt, '--check', '-', stdin=messy)
    assert (r.returncode, r.stdout) == (1, "<stdin>\n")
    assert _run(hfmt, '-', '--check', stdin=tidy).returncode == 0
    assert _run(hfmt, '-', '--tokens', stdin=tidy).stdout == _run(hfmt, '--tokens', stdin=tidy).stdout
    assert _run(hfmt, str(path), '--tokens').stdout.startswith('KEYWORD 1 1 def')
    # stdin alongside files: the files are formatted in place, stdin goes to stdout
    r = _run(hfmt, str(path), '-', stdin=messy)
    assert (r.returncode, r.stdout, path.read_text()) == (0, tidy, tidy)


# -- formatting ---------------------------------------------------------------

GOLDEN = sorted((ROOT / 'tests' / 'formatter').glob('*.in.ht'))


@GCC_SKIP
@pytest.mark.parametrize('case', GOLDEN, ids=lambda p: p.name[:-len('.in.ht')])
def test_golden(hfmt, case):
    expected = case.with_name(case.name.replace('.in.ht', '.out.ht')).read_text()
    r = _run(hfmt, stdin=case.read_text())
    assert r.returncode == 0, r.stderr
    assert r.stdout == expected
    assert _run(hfmt, stdin=expected).stdout == expected  # formatting twice changes nothing


@GCC_SKIP
def test_inconsistent_dedent_is_an_error(hfmt):
    r = _run(hfmt, stdin="def f():\n    if x:\n        y = 1\n      z = 2\n")
    assert (r.returncode, r.stderr) == (1, "<stdin>:4:1: error: unindent does not match any outer indentation level\n")


def _meaning(path) -> str:
    """The program at `path` as assembly, without the source positions in its panic messages:
    formatting moves those and nothing else."""
    from compile import compile_to_asm
    return re.sub(r'(\.asciz ")[^":\n]+:\d+:\d+: (panic: )', r'\1\2', compile_to_asm(str(path), 'x86_64-linux'))


@GCC_SKIP
@pytest.mark.parametrize('path', SOURCES, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_formatting_repo_files_is_stable_and_keeps_meaning(hfmt, path, tmp_path):
    """Formatting twice changes nothing, and the formatted file compiles to identical assembly."""
    formatted = _run(hfmt, stdin=path.read_text())
    assert formatted.returncode == 0, formatted.stderr
    assert _run(hfmt, stdin=formatted.stdout).stdout == formatted.stdout
    # Sibling modules are formatted too, so relative imports resolve to formatted code.
    for sibling in path.parent.glob('*.ht'):
        (tmp_path / sibling.name).write_text(_run(hfmt, stdin=sibling.read_text()).stdout)
    assert _meaning(tmp_path / path.name) == _meaning(path)


# -- command line ---------------------------------------------------------------

@GCC_SKIP
def test_in_place_formats_changed_files_only_and_continues_past_errors(hfmt, tmp_path):
    import os
    messy, clean, broken = tmp_path / 'messy.ht', tmp_path / 'clean.ht', tmp_path / 'broken.ht'
    messy.write_text("x=1\n")
    clean.write_text("y = 2\n")
    broken.write_text("z = 'oops")
    os.utime(clean, (1_000_000, 1_000_000))
    r = _run(hfmt, str(messy), str(broken), str(clean))
    assert r.returncode == 1
    assert r.stderr == f"{broken}:1:5: error: unterminated literal\n"
    assert messy.read_text() == "x = 1\n"
    assert clean.stat().st_mtime == 1_000_000  # unchanged files aren't rewritten
    assert broken.read_text() == "z = 'oops"


@GCC_SKIP
def test_check_lists_files_that_would_change(hfmt, tmp_path):
    messy, clean = tmp_path / 'messy.ht', tmp_path / 'clean.ht'
    messy.write_text("x=1\n")
    clean.write_text("y = 2\n")
    r = _run(hfmt, '--check', str(messy), str(clean))
    assert (r.returncode, r.stdout) == (1, f"{messy}\n")
    assert messy.read_text() == "x=1\n"
    assert _run(hfmt, '--check', str(clean)).returncode == 0
    assert _run(hfmt, '--check', stdin="x=1\n").returncode == 1
    assert _run(hfmt, '--check', stdin="x = 1\n").returncode == 0


@GCC_SKIP
def test_missing_file_is_reported(hfmt, tmp_path):
    r = _run(hfmt, str(tmp_path / 'nope.ht'))
    assert r.returncode == 1
    assert 'could not open' in r.stderr and 'No such file or directory' in r.stderr


# -- scrambled generated programs -------------------------------------------------

_OPERATORS = {'<<=', '>>=', '==', '!=', '>=', '<=', '<<', '>>', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^='}


def _must_separate(prev: str, cur: str) -> bool:
    """Whether writing prev and cur with no space would lex differently."""
    wordy = lambda c: c.isalnum() or c == '_'
    if wordy(prev[-1]) and wordy(cur[0]):
        return True
    return any(op.startswith(prev[-1] + cur[0]) for op in _OPERATORS) or (prev[-1] == '.' and cur[0].isdigit())


def scramble(source: str, seed: int) -> str:
    """Same tokens, different whitespace: random indentation width, random spacing between tokens,
    extra blank lines, and line breaks inside brackets."""
    import random
    r = random.Random(seed)
    unit = r.choice(['  ', '   ', '\t', '        ', ' '])
    lines, cur, widths, depth = [], [], [0], 0
    toks = [t for t in Lexer(source).tokenize() if t.type not in (TokenType.INDENT, TokenType.DEDENT, TokenType.EOF)]
    for t in toks:
        if t.type == TokenType.NEWLINE:
            if cur:
                lines.append(cur)
            cur = []
            continue
        cur.append(t)
    out = []
    for line in lines:
        width = line[0].col - 1
        while width < widths[-1]:
            widths.pop()
        if width > widths[-1]:
            widths.append(width)
        text = unit * (len(widths) - 1)
        prev = None
        for t in line:
            if prev is not None:
                if prev.val == ',' and depth > 0 and r.random() < 0.3:
                    text += '\n' + ' ' * r.randint(0, 9)
                else:
                    text += ' ' * (r.choice([0, 1, 1, 2]) or (1 if _must_separate(prev.val, t.val) else 0))
            text += t.val
            depth += t.val in '([{' and t.type != TokenType.STRING
            depth -= t.val in ')]}' and t.type != TokenType.STRING
            prev = t
        out.append(text + ' ' * r.choice([0, 0, 2]))
        if r.random() < 0.2:
            out.append(' ' * r.randint(0, 3))
    return '\n'.join(out) + '\n'


def _generated_programs():
    from tests.shape_matrix import programs
    from tests.test_random_programs import Gen
    progs = [(f"random-{seed}", Gen(seed).program()[0]) for seed in range(0, 40, 2)]
    progs += [(name, src) for i, (name, src, _) in enumerate(programs()) if i % 6 == 0]
    return progs


GENERATED = _generated_programs()


@GCC_SKIP
@pytest.mark.parametrize('name,source', GENERATED, ids=[g[0] for g in GENERATED])
def test_formatting_scrambled_programs_keeps_meaning(hfmt, name, source, tmp_path):
    from diagnostics import CompileError
    original = tmp_path / 'original.ht'
    original.write_text(source)
    try:
        expected = _meaning(original)
    except CompileError:
        pytest.skip('rejected by semantic analysis')
    import zlib
    scrambled = scramble(source, zlib.crc32(name.encode()))
    assert scrambled != source
    (tmp_path / 'scrambled.ht').write_text(scrambled)
    assert _meaning(tmp_path / 'scrambled.ht') == expected  # the scrambler itself is sound
    r = _run(hfmt, stdin=scrambled)
    assert r.returncode == 0, r.stderr + scrambled
    assert _run(hfmt, stdin=r.stdout).stdout == r.stdout
    formatted = tmp_path / 'formatted.ht'
    formatted.write_text(r.stdout)
    assert _meaning(formatted) == expected


@GCC_SKIP
def test_repository_sources_are_formatted(hfmt):
    r = _run(hfmt, '--check', *map(str, SOURCES))
    assert (r.returncode, r.stdout, r.stderr) == (0, '', ''), 'run tools/hfmt on these files:\n' + r.stdout


def test_trailing_comma_changes_keep_meaning(hfmt, tmp_path):
    source = "type P struct:\n    int x\n" + (ROOT / 'tests' / 'formatter' / 'trailing_commas.in.ht').read_text()
    formatted = _run(hfmt, stdin=source).stdout
    assert formatted != source
    (tmp_path / 'a.ht').write_text(source)
    (tmp_path / 'b.ht').write_text(formatted)
    assert _meaning(tmp_path / 'a.ht') == _meaning(tmp_path / 'b.ht')


def test_pointer_element_type_commas_keep_meaning(hfmt, tmp_path):
    source = (
        "type P struct:\n    int x\n"
        "def int main():\n"
        "    P p = P(1)\n"
        "    []*P ps = []*P[\n        &p\n    ]\n"
        "    [1]*P pa = [1]*P[\n        &p\n    ]\n"
        "    [][]*P pp = [][]*P[\n        ps\n    ]\n"
        "    [2]int xs = [3, 4]\n"
        "    int z = xs[0] * xs[\n        1\n    ]\n"
        "    [2][1]*P grid = [\n        [1]*P[&p],\n        [1]*P[&p]\n    ]\n"
        "    print(len(ps) + len(pa) + len(pp) + z + len(grid))\n"
        "    return 0\n")
    formatted = _run(hfmt, stdin=source).stdout
    assert "        &p,\n" in formatted and "        1\n    ]" in formatted and "[1]*P[&p],\n    ]" in formatted
    (tmp_path / 'a.ht').write_text(source)
    (tmp_path / 'b.ht').write_text(formatted)
    assert _meaning(tmp_path / 'a.ht') == _meaning(tmp_path / 'b.ht')


def test_star_spacing_follows_the_compilers_reading(hfmt, tmp_path):
    """Where hfmt spaces a `*` (a multiplication) or keeps it tight (a pointer type), the compiler
    reads it the same way: both spellings compile to the same program."""
    source = (
        "type P struct:\n    int x\n"
        "def [1]*P keep([1]*P a):\n    return a\n"
        "def int main():\n"
        "    int x = 3\n    [2]int ys = [4, 5]\n    P p = P(6)\n"
        "    int a = [x][0]*2\n"
        "    int b = [x][0]*ys[1]\n"
        "    [2][1]*P g = [[1]*P[&p], [1]*P[&p]]\n"
        "    [1]*P one = keep([1]*P[&p])\n"
        "    print(a + b + g[1][0].x + one[0].x)\n"
        "    return 0\n")
    formatted = _run(hfmt, stdin=source).stdout
    assert "[x][0] * 2" in formatted and "[x][0] * ys[1]" in formatted and "keep([1]*P[&p])" in formatted
    (tmp_path / 'a.ht').write_text(source)
    (tmp_path / 'b.ht').write_text(formatted)
    assert _meaning(tmp_path / 'a.ht') == _meaning(tmp_path / 'b.ht')


def test_a_file_keeps_its_line_endings(hfmt, tmp_path):
    """The lexer skips the "\\r" of a "\\r\\n"; a source that uses them gets them back."""
    source = b"def int main():\n    int x=1   # note\n    return x\n"
    formatted = b"def int main():\n    int x = 1  # note\n    return x\n"
    crlf, lf = tmp_path / "crlf.ht", tmp_path / "lf.ht"
    crlf.write_bytes(source.replace(b"\n", b"\r\n"))
    lf.write_bytes(source)
    assert subprocess.run([*hfmt, str(crlf), str(lf)], capture_output=True, timeout=20).returncode == 0
    assert crlf.read_bytes() == formatted.replace(b"\n", b"\r\n") and lf.read_bytes() == formatted
    assert subprocess.run([*hfmt, '--check', str(crlf), str(lf)], capture_output=True, timeout=20).returncode == 0
    piped = subprocess.run(hfmt, input=source.replace(b"\n", b"\r\n"), capture_output=True, timeout=20)
    assert piped.stdout == formatted.replace(b"\n", b"\r\n")
