"""examples/json, the JSON reader and formatter: JSONTestSuite's files (tests/json/test_parsing,
each named for whether a parser must accept it, must reject it, or may do either), and its output
compared with Python's own for random documents."""

import json
import random
import subprocess
from pathlib import Path

import pytest

from build import build_executable, run_prefix
from tests.targets import E2E_TARGETS
from tests.test_compiler import GCC_SKIP

ROOT = Path(__file__).resolve().parent.parent
SUITE = sorted((ROOT / 'tests' / 'json' / 'test_parsing').glob('*.json'))

pytestmark = GCC_SKIP


@pytest.fixture(scope='session', params=E2E_TARGETS, ids=str)
def jsonfmt(request, tmp_path_factory):
    """Command running the formatter built for each E2E target."""
    exe = tmp_path_factory.mktemp(f'jsonfmt-{request.param}') / 'jsonfmt'
    build_executable(str(ROOT / 'examples' / 'json' / 'main.ht'), str(exe), target=request.param)
    return run_prefix(request.param) + [str(exe)]


def _run(jsonfmt, *args, stdin: bytes = None):
    return subprocess.run([*jsonfmt, *args], input=stdin, capture_output=True, timeout=120)


def _format(jsonfmt, document, *options) -> str:
    result = _run(jsonfmt, *options, stdin=json.dumps(document, ensure_ascii=False).encode())
    assert result.returncode == 0, result.stderr
    return result.stdout.decode()


def test_the_suite_is_all_here():
    assert len(SUITE) == 318 and {path.name[:2] for path in SUITE} == {'y_', 'n_', 'i_'}


def test_what_must_be_accepted_is_and_means_the_same(jsonfmt):
    for path in SUITE:
        if path.name.startswith('y_'):
            result = _run(jsonfmt, '--compact', str(path))
            assert result.returncode == 0, (path.name, result.stderr)
            assert json.loads(result.stdout.decode()) == json.loads(path.read_bytes().decode()), path.name


def test_what_must_be_rejected_is_with_a_position(jsonfmt):
    for path in SUITE:
        if path.name.startswith('n_'):
            result = _run(jsonfmt, str(path))
            assert result.returncode == 1 and result.stdout == b'', path.name
            located, _, message = result.stderr.decode('latin-1').rpartition(': error: ')
            assert located.startswith(str(path) + ':') and message.endswith('\n'), path.name


def test_what_a_parser_may_do_either_with_is_never_a_crash(jsonfmt):
    for path in SUITE:
        if path.name.startswith('i_'):
            assert _run(jsonfmt, str(path)).returncode in (0, 1), path.name


def _random_document(rng, depth):
    choice = rng.random()
    if depth == 0 or choice < 0.3:
        return rng.choice(
            [
                None,
                True,
                False,
                rng.randrange(-10 ** 12, 10 ** 12),
                0,
                "",
                "plain",
                "quote \" slash \\ and /", "line\nbreak\ttab\r",
                "\x00\x01\x1f\x7f",
                "caf\u00e9 \u4e2d \U0001f600",
                ''.join(chr(rng.randrange(1, 0x250)) for _ in range(rng.randrange(8)))
            ]
        )
    if choice < 0.65:
        return [_random_document(rng, depth - 1) for _ in range(rng.randrange(5))]
    return {rng.choice(["a", "b", "key", "", "\u00e9", "with \"quotes\"", "z" * 5]) + str(rng.randrange(40)):
            _random_document(rng, depth - 1) for _ in range(rng.randrange(5))}


def test_random_documents_are_laid_out_as_python_lays_them_out(jsonfmt):
    rng = random.Random(20261004)
    for _ in range(60):
        document = _random_document(rng, depth=rng.randrange(1, 6))
        assert _format(jsonfmt, document) == json.dumps(document, indent=2, ensure_ascii=False) + "\n"
        assert _format(jsonfmt, document, '--compact') == \
            json.dumps(document, separators=(',', ':'), ensure_ascii=False) + "\n"
        assert _format(jsonfmt, document, '--sort-keys', '--indent', '3') == \
            json.dumps(document, indent=3, ensure_ascii=False, sort_keys=True) + "\n"


def test_numbers_and_member_order_are_kept_as_written(jsonfmt):
    source = b'{"b": 1.50, "a": -0, "b": 1E+2, "big": 123456789012345678901234567890, "tiny": 1e-999}'
    result = _run(jsonfmt, '--compact', stdin=source)
    assert result.stdout == b'{"b":1.50,"a":-0,"b":1E+2,"big":123456789012345678901234567890,"tiny":1e-999}\n'
    # Sorting keeps equal keys in their order.
    assert _run(jsonfmt, '--compact', '--sort-keys', stdin=source).stdout == \
        b'{"a":-0,"b":1.50,"b":1E+2,"big":123456789012345678901234567890,"tiny":1e-999}\n'


def test_formatting_its_own_output_changes_nothing(jsonfmt):
    document = _random_document(random.Random(7), depth=6)
    pretty = _format(jsonfmt, document)
    assert _run(jsonfmt, stdin=pretty.encode()).stdout.decode() == pretty
    compact = _run(jsonfmt, '--compact', stdin=pretty.encode()).stdout
    assert _run(jsonfmt, stdin=compact).stdout.decode() == pretty


@pytest.mark.parametrize("source,where,message", [
    (b'{"a": 1,}', "1:9", "expected a string for a member's name"),
    (b'[1 2]', "1:4", "expected ',' or ']'"),
    (b'{"a" 1}', "1:6", "expected ':'"),
    (b'[\n  1,\n  "abc\n]', "3:7", "a control character in a string must be escaped"),
    (b'01', "1:2", "a number can't start with 0"),
    (b'1.', "1:3", "expected a digit after '.'"),
    (b'"\\x"', "1:3", "unknown escape"),
    (b'"\\u12g4"', "1:4", "expected four hexadecimal digits after \\u"),
    (b'[1]x', "1:4", "unexpected text after the value"),
    (b'', "1:1", "unexpected end of input"),
    (b'[' * 2000, "1:1001", "nested too deeply"),
])
def test_errors(jsonfmt, source, where, message):
    result = _run(jsonfmt, stdin=source)
    assert (result.returncode, result.stdout) == (1, b'')
    assert result.stderr.decode() == f"<stdin>:{where}: error: {message}\n"


def test_usage(jsonfmt):
    for arguments in (['--bogus'], ['a.json', 'b.json'], ['--indent', 'two'], ['--indent', '-1']):
        result = _run(jsonfmt, *arguments)
        assert (result.returncode, result.stderr) == \
            (2, b"usage: jsonfmt [--compact] [--indent N] [--sort-keys] [file]\n")
    missing = _run(jsonfmt, 'no-such-file.json')
    assert missing.returncode == 1 and missing.stderr.startswith(b"jsonfmt: could not open 'no-such-file.json': ")
