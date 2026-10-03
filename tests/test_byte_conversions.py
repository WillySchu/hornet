"""str(byte), str([]byte), and bytes(str)."""

import pytest

from tests.test_compiler import GCC_SKIP, assert_program_semantic_error, assert_program_stdout


@GCC_SKIP
def test_conversions_and_copy_semantics():
    assert_program_stdout(
        "def int main():\n"
        "    str s = 'hello'\n"
        "    print(str(s[1]))\n"
        "    []byte bs = bytes(s)\n"
        "    bs[0] = \"J\"\n"
        "    print(s)\n"
        "    str t = str(bs)\n"
        "    bs[1] = \"E\"\n"
        "    print(t)\n"
        "    print(str(bs) + str(\"!\"))\n"
        "    print(len(bytes('')))\n"
        "    print(str(bytes('')) == '')\n"
        "    print(str(bytes(s)) == s)\n"
        "    bs = append(bs, \"\\xff\")\n"
        "    print(len(str(bs)))\n"
        "    print(str(bs)[1:3])\n"
        "    return 0\n",
        "e\nhello\nJello\nJEllo!\n0\ntrue\ntrue\n6\nEl\n",
    )


@GCC_SKIP
def test_every_byte_value_round_trips():
    assert_program_stdout(
        "def int main():\n"
        "    []byte all\n"
        "    for int i = 0; i < 256; i += 1:\n"
        "        all = append(all, uint8(i))\n"
        "    str s = str(all)\n"
        "    int ok = 0\n"
        "    for int i = 0; i < 256; i += 1:\n"
        "        if str(uint8(i)) == s[i:i + 1] and int(bytes(s)[i]) == i:\n"
        "            ok += 1\n"
        "    print(ok)\n"
        "    print(len(s))\n"
        "    return 0\n",
        "256\n256\n",
    )


@GCC_SKIP
def test_large_round_trip():
    assert_program_stdout(
        "def int main():\n"
        "    str s = 'abcdefghijklmnop'\n"
        "    for int i = 0; i < 16; i += 1:\n"
        "        s = s + s\n"
        "    print(str(bytes(s)) == s)\n"
        "    return 0\n",
        "true\n",
    )


@GCC_SKIP
def test_bytes_in_argument_return_append_index_and_dict_positions():
    assert_program_stdout(
        "def int total([]byte b):\n"
        "    int t = 0\n"
        "    for x in b:\n"
        "        t += int(x)\n"
        "    return t\n"
        "def []byte twice(str s):\n"
        "    return bytes(s + s)\n"
        "def int main():\n"
        "    print(total(bytes('ab')))\n"
        "    []byte a = append(bytes('xy'), \"z\")\n"
        "    print(str(a))\n"
        "    print(str(twice('q')))\n"
        "    print(bytes('abc')[2])\n"
        "    dict[int][]byte d = dict[int][]byte{}\n"
        "    d[1] = bytes('k')\n"
        "    print(str(d[1]))\n"
        "    return 0\n",
        "195\nxyz\nqq\n99\nk\n",
    )


@pytest.mark.parametrize('expr,match', [
    ("str(5)", "str\\(...\\) takes a byte, a \\[\\]byte, or an enum .*got int"),
    ("str(int8(5))", "got int8"),
    ("str([]int[1])", "got \\[\\]int"),
    ("bytes(5)", "bytes\\(\\) takes a str, got int"),
    ("bytes('a', 'b')", "exactly one argument"),
])
def test_bad_conversions_are_rejected(expr, match):
    assert_program_semantic_error(f"def int main():\n    print({expr})\n    return 0\n", match=match)
