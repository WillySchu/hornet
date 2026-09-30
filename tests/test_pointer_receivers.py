"""Pointer receivers: `def m(*self, ...)` methods mutate their receiver."""

import pytest

from tests.test_compiler import GCC_SKIP, assert_program_semantic_error, assert_program_stdout

_COUNTER = (
    "type Counter struct:\n"
    "    int n\n"
    "    []int log\n"
    "\n"
    "    def inc(*c, int by):\n"
    "        c.n += by\n"
    "        c.log = append(c.log, c.n)\n"
    "\n"
    "    def int get(c):\n"
    "        return c.n\n"
    "\n"
    "    def twice(*c):\n"
    "        c.inc(1)\n"
    "        c.inc(1)\n"
    "\n"
    "    def bump_copy(c):\n"
    "        c.inc(1000)\n"
    "\n"
    "type Holder struct:\n"
    "    Counter c\n"
    "\n"
    "def Counter mk():\n"
    "    return Counter(0, none)\n"
)


@GCC_SKIP
def test_pointer_receivers_mutate_every_addressable_receiver_shape():
    assert_program_stdout(
        _COUNTER +
        "def bump(*Counter p):\n"
        "    p.inc(10)\n"
        "def int main():\n"
        "    Counter a = Counter(0, none)\n"
        "    a.inc(5)\n"
        "    a.twice()\n"
        "    print(a.get())\n"
        "    print(a.log)\n"
        "    a.bump_copy()\n"
        "    print(a.n)\n"
        "    Holder h = Holder(Counter(1, none))\n"
        "    h.c.inc(2)\n"
        "    print(h.c.n)\n"
        "    [2]Counter arr = [mk(), mk()]\n"
        "    arr[1].inc(7)\n"
        "    print(arr[1].n)\n"
        "    []Counter sl = arr[:]\n"
        "    sl[1].inc(1)\n"
        "    print(arr[1].n)\n"
        "    *Counter pa = &a\n"
        "    pa.inc(100)\n"
        "    (*pa).inc(1)\n"
        "    bump(&a)\n"
        "    print(a.n)\n"
        "    print(pa.get())\n"
        "    return 0\n",
        "7\n[]int[5, 6, 7]\n7\n3\n7\n8\n118\n118\n",
    )


@pytest.mark.parametrize('receiver', ["mk()", "Counter(0, none)"])
def test_pointer_receiver_on_a_temporary_is_rejected(receiver):
    assert_program_semantic_error(
        _COUNTER + f"def int main():\n    {receiver}.inc(1)\n    return 0\n",
        match="has a pointer receiver, so it needs an addressable receiver")
