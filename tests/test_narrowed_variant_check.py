"""A narrowed variable whose address is taken (`&u`) can change variant through the pointer; each
read or write of it as the narrowed variant checks that it still holds that variant, and panics
otherwise. Other variables carry no check."""

import pytest

from ir.program_builder import build_ir_program
from tests.test_compiler import GCC_SKIP, _parse, analyze, assert_program_panics, assert_program_stdout

DECLS = (
    "type A struct:\n"
    "    str s\n"
    "type B struct:\n"
    "    int n\n"
    "type C struct:\n"
    "    int k\n"
    "type U is A | B\n"
    "type U3 is A | B | C\n"
)
MESSAGE = "'u' changed variant while narrowed"


@GCC_SKIP
def test_a_read_after_the_variant_changed_panics():
    assert_program_panics(
        DECLS +
        "def int main():\n"
        "    U u = A('hello')\n"
        "    *U p = &u\n"
        "    if u is A:\n"
        "        print(u.s)\n"
        "        *p = A('again')\n"  # the same variant: fine, and seen
        "        print(u.s)\n"
        "        *p = B(12345)\n"
        "        print(len(u.s))\n"
        "    return 0\n",
        MESSAGE, expected_stdout="hello\nagain\n",
    )


@GCC_SKIP
def test_a_write_after_the_variant_changed_panics():
    assert_program_panics(
        DECLS +
        "def int main():\n"
        "    U u = B(1)\n"
        "    *U p = &u\n"
        "    if u is B:\n"
        "        *p = A('text')\n"
        "        u.n = 5\n"
        "    return 0\n",
        MESSAGE,
    )


@GCC_SKIP
def test_a_match_arm_checks_too():
    assert_program_panics(
        DECLS +
        "def int main():\n"
        "    U u = A('hello')\n"
        "    *U p = &u\n"
        "    match u:\n"
        "        is A:\n"
        "            *p = B(1)\n"
        "            print(u.s)\n"
        "        is B:\n"
        "            print(u.n)\n"
        "    return 0\n",
        MESSAGE,
    )


@GCC_SKIP
def test_a_callee_putting_back_an_excluded_variant_panics():
    assert_program_panics(
        DECLS +
        "def reset(*U3 p):\n"
        "    *p = A('x')\n"
        "def int last(U3 u):\n"
        "    if u is A:\n"
        "        return 0\n"
        "    reset(&u)\n"  # u is still a U3 here: two variants are left
        "    if u is B:\n"
        "        return 1\n"
        "    return u.k\n"
        "def int main():\n"
        "    return last(C(7))\n",
        MESSAGE,
    )


@GCC_SKIP
def test_an_aliased_variable_that_keeps_its_variant_runs_normally():
    assert_program_stdout(
        DECLS +
        "def bump(*U p):\n"
        "    if *p is B as b:\n"
        "        *p = B(b.n + 1)\n"
        "def int main():\n"
        "    U u = B(1)\n"
        "    *U p = &u\n"
        "    if u is A:\n"
        "        return 1\n"
        "    for int i = 0; i < 3; i += 1:\n"
        "        bump(p)\n"
        "        u.n *= 2\n"
        "    print(u.n)\n"
        "    return 0\n",
        "22\n",
    )


@pytest.mark.parametrize("body,checked", [
    ("def int f(U u):\n    if u is A:\n        return len(u.s)\n    return u.n\n", False),
    # A pointer into the payload, or to the narrowed variable, can't change the variant.
    ("def int f(U u):\n    if u is A:\n        *str q = &u.s\n        *A a = &u\n        return len(*q) + len(a.s)\n"
     "    return u.n\n", False),
    ("def int f(U u):\n    *U p = &u\n    if u is A:\n        return len(u.s)\n    return u.n\n", True),
])
def test_only_variables_with_their_address_taken_are_checked(body, checked):
    program = build_ir_program(analyze(_parse(DECLS + body + "def int main():\n    return 0\n")))
    assert any(text.endswith(MESSAGE) for _, text in program.string_literals) == checked
