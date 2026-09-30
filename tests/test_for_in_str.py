"""`for ... in` over a str yields its bytes."""

from tests.test_compiler import GCC_SKIP, assert_program_stdout


@GCC_SKIP
def test_for_in_over_strings():
    assert_program_stdout(
        "const str W = 'xy'\n"
        "type H struct:\n"
        "    str s\n"
        "def int main():\n"
        "    str s = 'abc'\n"
        "    for c in s:\n"
        "        print(c)\n"
        "    for i, c in 'hi':\n"
        "        print(i)\n"
        "        print(str(c))\n"
        "    for c in W:\n"
        "        print(str(c))\n"
        "    for c in s[1:3]:\n"
        "        print(str(c))\n"
        "    H h = H('q')\n"
        "    for c in h.s:\n"
        "        print(str(c))\n"
        "    int n = 0\n"
        "    for c in '':\n"
        "        n += 1\n"
        "    print(n)\n"
        "    return 0\n",
        "97\n98\n99\n0\nh\n1\ni\nx\ny\nb\nc\nq\n0\n",
    )
