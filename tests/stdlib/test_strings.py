"""stdlib/strings.ht (Builder and helpers) and stdlib/fmt.ht formatting."""

import random

import pytest

from tests.targets import on_every_target, build_and_run
from tests.test_compiler import GCC_SKIP
from tests.test_os_output import compile_and_run

_IMPORTS = (
    "from 'std:strings' import Builder, split, join, trim, repeat, starts_with, ends_with, index_of, index_of_byte\n"
    "from 'std:fmt' import int_to_str, int_to_hex, pad_left\n"
)

PAIRS = [("", ""), ("a", "a"), ("a", "b"), ("b", "a"), ("ab", "a"), ("a", "ab"), ("", "a"), ("Z", "a"),
         ("abc", "abd"), ("key10", "key9"), ("caf\\xc3\\xa9", "cafz")]


def _run(body: str) -> str:
    r = compile_and_run(_IMPORTS + "def int main():\n" + ''.join(f"    {l}\n" for l in body.strip('\n').split('\n')))
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def _hex(n: int) -> str:
    return ('-' if n < 0 else '') + format(abs(n), 'x')


@GCC_SKIP
def test_int_to_str_and_hex_match_python():
    r = random.Random(3)
    values = [0, 1, -1, 9, 10, -10, 2 ** 63 - 1, -(2 ** 63), 255, -255]
    values += [r.randint(-(2 ** 63), 2 ** 63 - 1) for _ in range(30)] + [r.randint(-1000, 1000) for _ in range(10)]
    lits = [str(v) if v != -(2 ** 63) else "-9223372036854775807 - 1" for v in values]
    body = '\n'.join(f"print(int_to_str({l}))\nprint(int_to_hex({l}))" for l in lits) + "\nreturn 0"
    assert _run(body) == ''.join(f"{v}\n{_hex(v)}\n" for v in values)


@GCC_SKIP
def test_pad_left():
    assert _run(
        "print(pad_left('7', 3, \"0\"))\nprint(pad_left('1234', 3, \" \"))\nprint(pad_left('', 2, \"x\"))\nreturn 0") \
           == "007\n1234\nxx\n"


@GCC_SKIP
def test_builder():
    out = _run(
        "Builder b\n"
        "b.write('n=')\n"
        "b.write_int(-42)\n"
        "b.write_byte(\"!\")\n"
        "print(b.to_str())\n"
        "b.write('?')\n"
        "print(b.to_str())\n"
        "print(b.length())\n"
        "b.reset()\n"
        "print(b.length())\n"
        "for int i = 0; i < 100000; i += 1:\n"
        "    b.write('abcdefghij')\n"
        "str s = b.to_str()\n"
        "print(len(s))\n"
        "print(s[999990:1000000])\n"
        "return 0")
    assert out == "n=-42!\nn=-42!?\n7\n0\n1000000\nabcdefghij\n"


@GCC_SKIP
def test_split_and_join():
    cases = [(',a,,b,', ','), ('abc', ','), ('', ','), ('a::b::c', '::'), ('::', '::'), ('aaa', 'aa')]
    body, expected = [], []
    for k, (s, sep) in enumerate(cases):
        body += [f"[]str p{k} = split('{s}', '{sep}')", f"print(len(p{k}))", f"print(join(p{k}, '|'))"]
        parts = s.split(sep)
        expected += [str(len(parts)), '|'.join(parts)]
    assert _run('\n'.join(body) + "\nreturn 0") == '\n'.join(expected) + '\n'


@GCC_SKIP
def test_search_trim_repeat():
    out = _run(
        "print(starts_with('hello', 'he'))\n"
        "print(starts_with('he', 'hello'))\n"
        "print(ends_with('hello', 'lo'))\n"
        "print(ends_with('hello', ''))\n"
        "print(index_of('hello', 'll'))\n"
        "print(index_of('hello', 'z'))\n"
        "print(index_of('hello', ''))\n"
        "print(index_of_byte('hello', \"o\"))\n"
        "print(index_of_byte('', \"o\"))\n"
        "print(trim('  hi there \\n'))\n"
        "print(len(trim(' \\t\\r\\n ')))\n"
        "print(repeat('ab', 3))\n"
        "print(len(repeat('ab', 0)))\n"
        "return 0")
    assert out == "true\nfalse\ntrue\ntrue\n2\n-1\n0\n4\n-1\nhi there\n0\nababab\n0\n"


@GCC_SKIP
def test_split_with_empty_separator_panics():
    r = compile_and_run(_IMPORTS + "def int main():\n    split('a', '')\n    return 0\n")
    assert r.returncode != 0 and 'empty separator' in r.stderr


@GCC_SKIP
def test_quoted():
    source = (
        "from 'std:strings' import quoted\n"
        "def int main():\n"
        "    print(quoted('plain'))\n"
        "    print(quoted(''))\n"
        "    print(quoted('it\\'s a \\\\ and\\ttab\\nline\\r'))\n"
        "    print(quoted(str(byte(1)) + str(byte(31)) + str(byte(127)) + str(byte(0))))\n"
        "    print(quoted('caf' + str(byte(195)) + str(byte(169))))\n"      # UTF-8 is left as it is
        "    print(format('unknown name {}', quoted('x y')))\n"
        "    return 0\n")
    result = on_every_target(lambda target: build_and_run(source, target))
    assert result.stdout.split("\n")[:4] == [
        "'plain'", "''", "'it\\'s a \\\\ and\\ttab\\nline\\r'", "'\\x01\\x1f\\x7f\\x00'"]
    assert result.stdout.split("\n")[5] == "unknown name 'x y'"


@GCC_SKIP
def test_compare_orders_like_bytes():
    calls = "".join(f"    print(sign(compare('{a}', '{b}')))\n" for a, b in PAIRS)
    source = (
            "from 'std:strings' import compare\n"
            "def int sign(int n):\n"
            "    if n < 0:\n"
            "        return -1\n"
            "    if n > 0:\n"
            "        return 1\n"
            "    return 0\n"
            "def int main():\n" + calls + "    return 0\n")
    result = on_every_target(lambda target: build_and_run(source, target))

    def as_bytes(text: str) -> bytes:
        return text.encode().decode('unicode_escape').encode('latin-1')
    expected = [(as_bytes(a) > as_bytes(b)) - (as_bytes(a) < as_bytes(b)) for a, b in PAIRS]
    assert result.stdout.split() == [str(sign) for sign in expected]


@GCC_SKIP
@pytest.mark.parametrize("source,expected", [
    ("from 'std:strings' import lower\n"
     "def int main():\n"
     "    str a = 'x'\n"
     "    str b = lower(a)\n"
     "    print(b)\n"
     "    return 0",
     "x\n"),
    ("from 'std:strings' import lower\n"
     "def int main():\n"
     "    str a = 'X'\n"
     "    str b = lower(a)\n"
     "    print(b)\n"
     "    return 0",
     "x\n"),
    ("from 'std:strings' import lower\n"
     "def int main():\n"
     "    str a = 'XflkNjdkQk'\n"
     "    str b = lower(a)\n"
     "    print(b)\n"
     "    return 0",
     "xflknjdkqk\n"),
    ("from 'std:strings' import lower\n"
     "def int main():\n"
     "    str a = 'Hello World!'\n"
     "    str b = lower(a)\n"
     "    print(b)\n"
     "    return 0",
     "hello world!\n"),
])
def test_lower(source, expected):
    result = on_every_target(lambda target: build_and_run(source, target))
    assert result.stdout == expected
