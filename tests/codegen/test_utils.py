"""Tests for type_byte_width.py, likely to be merged into another file."""

import parser
import semantic
from codegen.utils import type_byte_width


def test_type_byte_width_int():
    t = semantic.Type(kind=semantic.TypeKind.INT)
    assert 4 == type_byte_width(t, {})


def test_type_byte_width_bool():
    t = semantic.Type(kind=semantic.TypeKind.BOOL)
    assert 4 == type_byte_width(t, {})


def test_type_byte_width_str():
    t = semantic.Type(kind=semantic.TypeKind.STR)
    assert 8 == type_byte_width(t, {})


def test_type_byte_width_array_int():
    t = semantic.Type(kind=semantic.TypeKind.ARRAY, element_type=semantic.Type(kind=semantic.TypeKind.INT), size=7)
    expected = 28  # 4 * 7
    assert expected == type_byte_width(t, {})


def test_type_byte_width_array_str():
    t = semantic.Type(kind=semantic.TypeKind.ARRAY, element_type=semantic.Type(kind=semantic.TypeKind.STR), size=11)
    expected = 88  # 8 * 11
    assert expected == type_byte_width(t, {})


def test_type_byte_width_nested_array_str():
    t = semantic.Type(
        kind=semantic.TypeKind.ARRAY,
        element_type=semantic.Type(
            kind=semantic.TypeKind.ARRAY,
            element_type=semantic.Type(kind=semantic.TypeKind.STR),
            size=5,
        ),
        size=11,
    )
    expected = 440  # 11 * 5 * 8
    assert expected == type_byte_width(t, {})


def test_type_byte_width_doubly_nested_array_int():
    t = semantic.Type(
        kind=semantic.TypeKind.ARRAY,
        element_type=semantic.Type(
            kind=semantic.TypeKind.ARRAY,
            element_type=semantic.Type(
                kind=semantic.TypeKind.ARRAY,
                element_type=semantic.Type(kind=semantic.TypeKind.INT),
                size=7,
            ),
            size=5,
        ),
        size=11,
    )
    expected = 1540  # 11 * 5 * 7 * 4
    assert expected == type_byte_width(t, {})


# TODO(will): Structs.
