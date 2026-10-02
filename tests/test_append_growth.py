"""append across every growth boundary: empty -> 1, doubling below 256, cap/4 growth above."""

from tests.test_compiler import GCC_SKIP, assert_program_stdout


@GCC_SKIP
def test_append_across_growth_boundaries():
    assert_program_stdout(
        "type P struct:\n"
        "    int8 a\n"
        "    int b\n"
        "\n"
        "def int main():\n"
        "    []int s\n"
        "    []P ps\n"
        "    []str names\n"
        "    for int i = 0; i < 600; i += 1:\n"
        "        s = append(s, i * 3)\n"
        "        ps = append(ps, P(int8(i), i * 7))\n"
        "        names = append(names, 'n')\n"
        "        int n = len(s)\n"
        "        if n == 1 or n == 2 or n == 255 or n == 256 or n == 257 or n == 600:\n"
        "            print(s[n - 1] + ps[n - 1].b + int(ps[n - 1].a))\n"
        "    int total = 0\n"
        "    for v in s:\n"
        "        total += v\n"
        "    for p in ps:\n"
        "        total += p.b + int(p.a)\n"
        "    print(total)\n"
        "    print(len(names))\n"
        "    return 0\n",
        "".join(f"{3 * (n - 1) + 7 * (n - 1) + ((n - 1 + 128) % 256 - 128)}\n" for n in (1, 2, 255, 256, 257, 600))
        + f"{sum(3 * i + 7 * i + ((i + 128) % 256 - 128) for i in range(600))}\n600\n",
    )
