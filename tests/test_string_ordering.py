"""`<`, `>`, `<=`, and `>=` on strs: byte by byte, bytes unsigned, and a str before the longer strs
it begins. The order strings.ht's `compare` gives, and Python's for bytes."""

import pytest

from compile import generate_asm
from tests.test_compiler import GCC_SKIP, _parse, analyze, assert_program_semantic_error, assert_program_stdout
from typed_ast import dump

# Hornet expressions for strs, and the bytes each is.
STRS = [
    ("''", b""), ("'a'", b"a"), ("'b'", b"b"), ("'ab'", b"ab"), ("'abc'", b"abc"), ("'abd'", b"abd"),
    ("'Z'", b"Z"), ("'key10'", b"key10"), ("'key9'", b"key9"), ("'a\\0'", b"a\0"), ("'a b'", b"a b"),
    ("str(byte(200))", b"\xc8"), ("'a' + str(byte(128))", b"a\x80"), ("str(byte(127))", b"\x7f"),
]
OPS = {'<': bytes.__lt__, '>': bytes.__gt__, '<=': bytes.__le__, '>=': bytes.__ge__}


@GCC_SKIP
def test_every_pair_orders_as_its_bytes_do():
    lines, expected = [], []
    for i, (_, left) in enumerate(STRS):
        for j, (_, right) in enumerate(STRS):
            for op, holds in OPS.items():
                lines.append(f"    print(s[{i}] {op} s[{j}])\n")
                expected.append("true\n" if holds(left, right) else "false\n")
    assert_program_stdout(
        "def int main():\n"
        f"    [{len(STRS)}]str s = [{', '.join(text for text, _ in STRS)}]\n" + "".join(lines) + "    return 0\n",
        "".join(expected),
    )


@GCC_SKIP
def test_in_conditions_constants_and_longer_expressions():
    assert_program_stdout(
        "const bool FIRST = 'apple' < 'pear'\n"
        "const bool SAME = 'pear' >= 'pear' and not 'pear' > 'pear'\n"
        "const int COUNT = 3\n"
        "def str smallest([3]str names):\n"
        "    str best = names[0]\n"
        "    for name in names:\n"
        "        if name < best:\n"
        "            best = name\n"
        "    return best\n"
        "def int main():\n"
        "    [COUNT]str names = ['pear', 'apple', 'fig']\n"
        "    print(smallest(names))\n"
        "    print(FIRST)\n"
        "    print(SAME)\n"
        "    str a = 'x'\n"
        "    print(a + 'y' > a and a <= a + '')\n"       # concatenations as operands
        "    print(a < 'y' == true)\n"                    # binds like `==`
        "    return 0\n",
        "apple\ntrue\ntrue\ntrue\ntrue\n",
    )


@pytest.mark.parametrize("expression,got", [
    ("'a' < 1", "str and int"),
    ("1 >= 'a'", "int and str"),
    ("'a' <= \"b\"", "str and uint8"),      # a byte literal is a number
    ("'a' > true", "str and bool"),
])
def test_a_str_orders_only_against_a_str(expression, got):
    assert_program_semantic_error(
        f"def bool main():\n    return {expression}\n",
        match=rf"requires two operands of the same integer type \(int, int8, uint8, or int32\) or two str operands, "
              rf"got {got}")


def test_the_typed_tree_and_the_code():
    program = analyze(_parse("def bool f(str a, str b):\n    return a < b\ndef int main():\n    return 0\n"))
    assert "StrCompare op=LESS_THAN : bool" in dump(program)
    assert "memcmp" in generate_asm(program)  # the bytes they share, then their lengths
