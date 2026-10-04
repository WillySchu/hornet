"""A sum can't be another sum's variant, but a pointer to one, or a slice or dict of them, can: a
result type that holds a tree is `*Expr | ParseError`. `is` and `match` name such a variant."""

import pytest

from tests.test_compiler import GCC_SKIP, assert_program_semantic_error, assert_program_stdout

DECLS = (
    "type Num struct:\n"
    "    int value\n"
    "type Pair struct:\n"
    "    *Expr left\n"
    "    *Expr right\n"
    "type Expr is Num | Pair\n"
    "type Failure struct:\n"
    "    str message\n"
    "type Result is *Expr | Failure\n"
    "type Many is []Expr | dict[str]Expr | none\n"
)


@GCC_SKIP
def test_a_result_that_holds_a_pointer_to_a_sum():
    assert_program_stdout(
        DECLS +
        "def *Expr num(int v):\n"
        "    return &Num(v)\n"
        "def Result build(int v):\n"
        "    if v < 0:\n"
        "        return Failure('negative')\n"
        "    *Expr tree = &Num(v)\n"
        "    tree = &Pair(tree, num(v + 1))\n"
        "    return tree\n"
        "def int total(*Expr e):\n"
        "    match *e as node:\n"
        "        is Num:\n"
        "            return node.value\n"
        "        is Pair:\n"
        "            return total(node.left) + total(node.right)\n"
        "def int count(Many many):\n"
        "    match many as m:\n"
        "        is []Expr:\n"
        "            return len(m)\n"
        "        is dict[str]Expr:\n"
        "            return len(m) * 10\n"
        "        is none:\n"
        "            return -1\n"
        "def int main():\n"
        "    Result r = build(3)\n"
        "    if r is Failure:\n"
        "        return 1\n"
        "    print(total(r))\n"                    # r is a *Expr here
        "    match build(-1) as bad:\n"
        "        is *Expr:\n"
        "            return 2\n"
        "        is Failure:\n"
        "            print(bad.message)\n"
        "    print(build(1) is *Expr)\n"
        "    []Expr list = [Num(1), Num(2)]\n"
        "    dict[str]Expr named = dict[str]Expr{'a': Num(1)}\n"
        "    print(count(list) + count(named) + count(none))\n"
        "    return 0\n",
        "7\nnegative\ntrue\n11\n",
    )


@pytest.mark.parametrize("declaration,match", [
    ("type Wrapped is Expr | none\n", "'Expr' is itself a sum type"),
    ("type Loop is [2]Loop | int\n", r"'Loop' contains itself by value \(Loop's \[2\]Loop variant\)"),
    ("type Other is *Missing | int\n", "Unknown type 'Missing'"),
])
def test_what_is_still_an_error(declaration, match):
    assert_program_semantic_error(DECLS + declaration + "def int main():\n    return 0\n", match=match)
