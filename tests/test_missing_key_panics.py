"""A dict key that isn't there is named in the panic: `dict lookup: key not found: 'apple'`. The
key is shown as print shows one inside a value, for reading `d[k]`, updating it (`d[k] += 1`), and
`del(d, k)`; a long str key is cut short."""

import pytest

from tests.test_compiler import GCC_SKIP, assert_program_panics

DECLS = (
    "type Color enum:\n"
    "    Red\n"
    "    Green\n"
    "def int main():\n"
    "    dict[str]int names = dict[str]int{'pear': 1, 'fig': 2}\n"
    "    dict[str]int nothing\n"                    # no table yet: a different path in the runtime
    "    dict[int]int numbers = dict[int]int{1: 1, 17: 2, 33: 3}\n"
    "    dict[int32]int small\n"
    "    dict[int8]int tiny = dict[int8]int{5: 1}\n"
    "    dict[byte]int bytes_seen = dict[byte]int{\"a\": 1}\n"
    "    dict[bool]int flags = dict[bool]int{true: 1}\n"
    "    dict[Color]int colours = dict[Color]int{Color.Red: 1}\n"
    "    str long_key = 'x'\n"
    "    for int i = 0; i < 7; i += 1:\n"
    "        long_key = long_key + long_key\n"      # 128 bytes
    "    int n = len(names)\n"
)


@GCC_SKIP
@pytest.mark.parametrize("statement,message", [
    ("return names['apple']", "dict lookup: key not found: 'apple'"),
    ("return nothing['apple']", "dict lookup: key not found: 'apple'"),
    ("return names['']", "dict lookup: key not found: ''"),
    ("return names['it is' + ' two words']", "dict lookup: key not found: 'it is two words'"),
    ("return numbers[n * 21]", "dict lookup: key not found: 42"),
    ("return numbers[n - 9]", "dict lookup: key not found: -7"),
    ("return numbers[9223372036854775807]", "dict lookup: key not found: 9223372036854775807"),
    ("return small[int32(n) - 100000]", "dict lookup: key not found: -99998"),
    ("return tiny[int8(n) - 100]", "dict lookup: key not found: -98"),
    ("return bytes_seen[\"z\"]", "dict lookup: key not found: 122"),          # a byte is a number
    ("return flags[n > 5]", "dict lookup: key not found: false"),
    ("return colours[Color.Green]", "dict lookup: key not found: Color.Green"),
    # Updating an entry reads it first; deleting has a message of its own.
    ("names['plum'] += 1\n    return 0", "dict lookup: key not found: 'plum'"),
    ("numbers[n] *= 2\n    return 0", "dict lookup: key not found: 2"),
    ("del(names, 'plum')\n    return 0", "dict delete: key not found: 'plum'"),
    ("del(nothing, 'plum')\n    return 0", "dict delete: key not found: 'plum'"),
    ("del(numbers, 49)\n    return 0", "dict delete: key not found: 49"),
    ("del(colours, Color.Green)\n    return 0", "dict delete: key not found: Color.Green"),
    # A str key is shown whole up to 100 bytes.
    ("return names[long_key[:100]]", "dict lookup: key not found: '" + "x" * 100 + "'"),
    ("return names[long_key[:101]]", "dict lookup: key not found: '" + "x" * 100 + "...' (101 bytes)"),
    ("return names[long_key]", "dict lookup: key not found: '" + "x" * 100 + "...' (128 bytes)"),
    ("del(names, long_key)\n    return 0", "dict delete: key not found: '" + "x" * 100 + "...' (128 bytes)"),
])
def test_the_panic_names_the_key(statement, message):
    assert_program_panics(DECLS + f"    {statement}\n", message)


@GCC_SKIP
def test_a_key_that_is_found_is_unaffected_by_an_earlier_miss_elsewhere():
    # `in` misses without panicking, and the next panic still shows its own key.
    assert_program_panics(
        DECLS +
        "    print('zzz' in names)\n"
        "    print(names['pear'] + numbers[17])\n"
        "    return numbers[n + 5]\n",
        "dict lookup: key not found: 7",
        expected_stdout="false\n3\n",
    )
