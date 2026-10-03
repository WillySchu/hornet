"""What the `==` errors say about `none`: only a pointer, or a sum with a `none` variant, compares
to it. A slice or a dict never does."""

import pytest

from tests.test_compiler import assert_program_semantic_error

DECLS = (
    "type Holder struct:\n"
    "    []int items\n"
)


@pytest.mark.parametrize("body,match", [
    ("    []int a = [1]\n    []int b = [1]\n    return a == b\n",
     r"does not support slice, void, sum type, dict, or none operands, except comparing a pointer, or a sum "
     r"type with a `none` variant, to none at line"),
    ("    [2]Holder a\n    [2]Holder b\n    return a == b\n",
     r"array equality isn't defined yet when the elements are \(or contain\) a slice, sum type, or dict, none "
     r"of which has '==' defined yet at line"),
    ("    Holder a\n    Holder b\n    return a == b\n",
     r"is a slice, sum type, or dict, none of which has '==' defined yet at line"),
    # What a slice or dict compared to none is told instead.
    ("    []int a = [1]\n    return a == none\n", "A slice is never none"),
    ("    dict[str]int d\n    return d == none\n", "A dict is never none"),
])
def test_equality_errors(body, match):
    assert_program_semantic_error(DECLS + "def bool main():\n" + body, match=match)
