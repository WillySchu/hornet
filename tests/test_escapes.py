"""A backslash in a string or byte literal starts one of the escapes, or it is an error: `'\\q'`
once quietly meant `q`, and a mistyped escape became some other text."""

import pytest

from tests.test_compiler import GCC_SKIP, _parse, assert_program_stdout
from parser import ParseError

ESCAPES = "the escapes are \\n, \\t, \\r, \\0, \\\\, \\', \\\", and \\xNN (a byte, as two hexadecimal digits)"


@GCC_SKIP
def test_every_escape():
    assert_program_stdout(
        "def int main():\n"
        "    str s = 'a\\nb\\tc\\rd\\0e\\\\f\\'g\\\"h\\x41\\xff'\n"
        "    print(len(s))\n"
        "    print(int(s[1]) + int(s[3]) + int(s[5]) + int(s[7]) + int(s[9]) + int(s[16]))\n"
        "    print(s[10:15])\n"
        "    byte b = \"\\x7e\"\n"
        "    byte q = \"\\\"\"\n"
        "    print(int(b) + int(q))\n"
        "    return 0\n",
        "17\n" + str(10 + 9 + 13 + 0 + 92 + 255) + "\nf'g\"h\n160\n",
    )


@pytest.mark.parametrize("line,col,message", [
    ("    str s = 'a\\qb'", 15, "Unknown escape '\\q' -- " + ESCAPES),
    ("    str s = 'digits: \\d+'", 22, "Unknown escape '\\d' -- " + ESCAPES),
    ("    str s = '\\N'", 14, "Unknown escape '\\N' -- " + ESCAPES),                 # the case of an escape matters
    ("    byte b = \"\\e\"", 15, "Unknown escape '\\e' -- " + ESCAPES),
    ("    str s = 'ok \\x4'", 17, "'\\x' must be followed by two hexadecimal digits, as in '\\x41'"),
    ("    str s = '\\xZZ'", 14, "'\\x' must be followed by two hexadecimal digits, as in '\\x41'"),
    ("    str s = '\\x'", 14, "'\\x' must be followed by two hexadecimal digits, as in '\\x41'"),
    ("    print(format('{}\\p', 1))", 21, "Unknown escape '\\p' -- " + ESCAPES),
])
def test_a_backslash_that_starts_no_escape_is_an_error(line, col, message):
    with pytest.raises(ParseError) as e:
        _parse("def int main():\n" + line + "\n    return 0\n")
    assert (e.value.message, e.value.line, e.value.col) == (message, 2, col)


def test_in_an_import_s_path_too():
    with pytest.raises(ParseError, match="Unknown escape") as e:
        _parse("from 'li\\b' import f\n\ndef int main():\n    return 0\n")
    assert (e.value.line, e.value.col) == (1, 9)
