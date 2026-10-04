"""stdlib/fmt.ht's parse_int: a decimal string to an int, or an Error."""

from tests.targets import build_and_run, on_every_target
from tests.test_compiler import GCC_SKIP

CASES = [
    ("0", "0"), ("7", "7"), ("-7", "-7"), ("007", "7"), ("-0", "0"),
    ("9223372036854775807", "9223372036854775807"), ("-9223372036854775808", "-9223372036854775808"),
    ("9223372036854775808", "number too large: '9223372036854775808'"),
    ("-9223372036854775809", "number too large: '-9223372036854775809'"),
    ("99999999999999999999999", "number too large: '99999999999999999999999'"),
    ("", "not a number: ''"), ("-", "not a number: '-'"), ("12a", "not a number: '12a'"),
    ("+5", "not a number: '+5'"), (" 5", "not a number: ' 5'"), ("--5", "not a number: '--5'"),
]


@GCC_SKIP
def test_parse_int():
    calls = "".join(f"    show(parse_int('{text}'))\n" for text, _ in CASES)
    source = (
        "from 'fmt' import parse_int\n"
        "from 'errors' import Error, IntResult\n"
        "def show(IntResult r):\n"
        "    match r as v:\n"
        "        is int:\n"
        "            print(v)\n"
        "        is Error:\n"
        "            print(v.message)\n"
        "def int main():\n" + calls + "    return 0\n")
    result = on_every_target(lambda target: build_and_run(source, target))
    assert (result.returncode, result.stdout) == (0, "".join(expected + "\n" for _, expected in CASES))
