"""`*p = v` for every composite pointee kind and value shape, observed through the pointee."""

from tests.test_compiler import GCC_SKIP, assert_program_stdout

_DECLS = (
    "type P struct:\n"
    "    int x\n"
    "    []int s\n"
    "type C struct:\n"
    "    int r\n"
    "type D struct:\n"
    "    int k\n"
    "type U is C | D\n"
    "def P mkp(int v):\n"
    "    return P(v, []int[v])\n"
    "def [2]int mka():\n"
    "    return [7, 8]\n"
    "def []int mks():\n"
    "    return []int[4, 5]\n"
    "def dict[int]int mkd():\n"
    "    return dict[int]int{3: 4}\n"
    "def U mku():\n"
    "    return D(9)\n"
)


@GCC_SKIP
def test_deref_assign_all_composite_kinds():
    assert_program_stdout(
        _DECLS +
        "def int main():\n"
        "    []int s = none\n"
        "    *[]int ps = &s\n"
        "    [3]int arr = [1, 2, 3]\n"
        "    *ps = arr[0:2]\n"
        "    print(s)\n"
        "    *ps = []int[9, 9, 9]\n"
        "    print(s)\n"
        "    *ps = append(s, 10)\n"
        "    print(s)\n"
        "    *ps = mks()\n"
        "    print(s)\n"
        "    []int other = []int[6]\n"
        "    *ps = other\n"
        "    print(s)\n"
        "    *ps = none\n"
        "    print(s)\n"
        "    [2]int a = [0, 0]\n"
        "    *[2]int pa = &a\n"
        "    *pa = [5, 6]\n"
        "    print(a)\n"
        "    *pa = mka()\n"
        "    print(a)\n"
        "    dict[int]int d = dict[int]int{1: 2}\n"
        "    *dict[int]int pd = &d\n"
        "    *pd = mkd()\n"
        "    print(d)\n"
        "    *pd = dict[int]int{5: 6}\n"
        "    d[7] = 8\n"
        "    print(len(d))\n"
        "    P p = P(0, none)\n"
        "    *P pp = &p\n"
        "    *pp = mkp(3)\n"
        "    print(p)\n"
        "    *pp = P(4, none)\n"
        "    print(p)\n"
        "    U u = C(1)\n"
        "    *U pu = &u\n"
        "    *pu = mku()\n"
        "    print(u)\n"
        "    *pu = C(2)\n"
        "    print(u)\n"
        "    str t = 'a'\n"
        "    *str pt = &t\n"
        "    *pt = t + 'b'\n"
        "    print(t)\n"
        "    return 0\n",
        "[]int[1, 2]\n[]int[9, 9, 9]\n[]int[9, 9, 9, 10]\n[]int[4, 5]\n[]int[6]\n[]int[]\n"
        "[2]int[5, 6]\n[2]int[7, 8]\ndict[int]int{3: 4}\n2\n"
        "P(x: 3, s: []int[3])\nP(x: 4, s: []int[])\nD(k: 9)\nC(r: 2)\nab\n",
    )


@GCC_SKIP
def test_deref_assign_through_parameter_survives_return():
    assert_program_stdout(
        "def put(*[]int p):\n"
        "    [3]int b = [1, 2, 3]\n"
        "    *p = b[:]\n"
        "def int noise(int a):\n"
        "    [16]int z = [a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a]\n"
        "    return z[3]\n"
        "def int main():\n"
        "    []int r = none\n"
        "    put(&r)\n"
        "    noise(9)\n"
        "    print(r)\n"
        "    return 0\n",
        "[]int[1, 2, 3]\n",
    )
