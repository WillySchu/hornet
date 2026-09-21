"""Tests for semantic.py"""

import pytest

import parser
import semantic


def test_type_from_name_str():
    assert semantic.Type.STR == semantic.type_from_name('str', {}, {})


def test_type_from_name_int():
    assert semantic.Type.INT == semantic.type_from_name('int', {}, {})


def test_type_from_name_bool():
    assert semantic.Type.BOOL == semantic.type_from_name('bool', {}, {})


def test_type_from_name_array():
    type_expr = parser.ArrayTypeExpr(size=5, element_type='int')
    expected = semantic.Type(
        kind=semantic.TypeKind.ARRAY,
        element_type=semantic.Type(kind=semantic.TypeKind.INT),
        size=5,
    )
    assert expected == semantic.type_from_name(type_expr, {}, {})


def test_type_from_name_slice():
    type_expr = parser.SliceTypeExpr(element_type='int')
    expected = semantic.Type(kind=semantic.TypeKind.SLICE, element_type=semantic.Type(kind=semantic.TypeKind.INT))
    assert expected == semantic.type_from_name(type_expr, {}, {})


def test_type_from_name_array_nested():
    type_expr = parser.ArrayTypeExpr(size=5, element_type=parser.ArrayTypeExpr(size=3, element_type='str'))
    expected = semantic.Type(
        kind=semantic.TypeKind.ARRAY,
        element_type=semantic.Type(
            kind=semantic.TypeKind.ARRAY,
            element_type=semantic.Type(kind=semantic.TypeKind.STR),
            size=3,
        ),
        size=5,
    )
    assert expected == semantic.type_from_name(type_expr, {}, {})


def test_type_from_name_slice_nested():
    type_expr = parser.SliceTypeExpr(
        element_type=parser.SliceTypeExpr(
            element_type=parser.SliceTypeExpr(element_type='bool'),
        ),
    )
    expected = semantic.Type(
        kind=semantic.TypeKind.SLICE,
        element_type=semantic.Type(
            kind=semantic.TypeKind.SLICE,
            element_type=semantic.Type(
                kind=semantic.TypeKind.SLICE,
                element_type=semantic.Type(kind=semantic.TypeKind.BOOL),
            )
        )
    )
    assert expected == semantic.type_from_name(type_expr, {}, {})


def test_type_from_name_unknown():
    with pytest.raises(semantic.SemanticError, match='Unknown type \'unknown\''):
        semantic.type_from_name('unknown', {}, {})


# ---------------------------------------------------------------------------
# type_from_name's own sum_types parameter -- deliberately opt-in
# (defaults to disallowed), unlike structs/aliases, which are required.
# See type_from_name's own docstring for why. TestSumTypes in test_
# compiler.py covers the full semantic-analysis picture (SemanticAnalyzer's
# own registry, name collisions, variant validation, widening); these
# stay scoped to type_from_name in isolation.
# ---------------------------------------------------------------------------

def test_type_from_name_sum_type_resolves_when_passed():
    sum_types = {'Shape': semantic.SumTypeInfo(name='Shape', variants=['Circle', 'Square'])}
    expected = semantic.Type(kind=semantic.TypeKind.SUM, sum_type_name='Shape')
    assert expected == semantic.type_from_name('Shape', {}, {}, sum_types=sum_types)


def test_type_from_name_sum_type_unknown_when_omitted():
    """The deliberately safe default: a sum type name reports as
    unknown -- not silently resolved -- at a call site that doesn't
    pass sum_types at all, exactly like _resolve_struct_fields's own
    call site (struct fields can't be sum-typed yet)."""
    sum_types = {'Shape': semantic.SumTypeInfo(name='Shape', variants=['Circle', 'Square'])}
    with pytest.raises(semantic.SemanticError, match="Unknown type 'Shape'"):
        semantic.type_from_name('Shape', {}, {})  # sum_types omitted


def test_type_from_name_sum_type_array_element():
    """A sum type as an array's own element type, recursing through
    ArrayTypeExpr -- exercises sum_types being threaded through the
    recursive call, not just the top-level one."""
    sum_types = {'Shape': semantic.SumTypeInfo(name='Shape', variants=['Circle', 'Square'])}
    type_expr = parser.ArrayTypeExpr(size=3, element_type='Shape')
    expected = semantic.Type(
        kind=semantic.TypeKind.ARRAY,
        element_type=semantic.Type(kind=semantic.TypeKind.SUM, sum_type_name='Shape'),
        size=3,
    )
    assert expected == semantic.type_from_name(type_expr, {}, {}, sum_types=sum_types)


# TODO(will): Test always_returns

# TODO(will): Test SemanticAnalyzer


# ---------------------------------------------------------------------------
# SemanticError's own node -> "at line L, column C" formatting (the
# mechanism every one of this file's ~80 raise sites relies on --
# tested once here, directly, rather than per call site).
# ---------------------------------------------------------------------------

def test_semantic_error_without_a_node_has_a_plain_message():
    err = semantic.SemanticError("something went wrong")
    assert str(err) == "something went wrong"


def test_semantic_error_with_a_positioned_node_appends_location():
    node = parser.Constant(value=1, line=4, col=9)
    err = semantic.SemanticError("something went wrong", node)
    assert str(err) == "something went wrong at line 4, column 9"


def test_semantic_error_with_an_unset_node_has_a_plain_message():
    """A node with no real token behind it (line=col=0, the default
    for one built by hand) must not print a misleading "line 0,
    column 0" -- silently falls back to the plain message instead."""
    node = parser.Constant(value=1)  # line/col both default to 0
    err = semantic.SemanticError("something went wrong", node)
    assert str(err) == "something went wrong"
