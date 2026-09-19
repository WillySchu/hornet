"""Tests for ir/utils.py's own semantic-level helpers (type_byte_
width, leaf_type, type_of) -- split out of tests/codegen/test_utils.py
alongside the module split itself (see ir/utils.py's own module
docstring for why): codegen/utils.py's own remaining, machine-level
half (as_byte_register/as_qword_register/escape_for_asciz) keeps its
own tests there."""

import re

import pytest

import parser
import semantic
from ir.errors import IRError
from ir.utils import COMPOSITE_KINDS, leaf_type, type_byte_width, type_of


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


def test_type_byte_width_basic_struct():
    structs = {
        'A': semantic.StructInfo(
            name='A',
            fields={
                'x': semantic.Type(kind=semantic.TypeKind.STR),
                'y': semantic.Type(kind=semantic.TypeKind.INT),
            },
        ),
    }
    t = semantic.Type(kind=semantic.TypeKind.STRUCT, struct_name='A')
    expected = 12  # 8 + 4
    assert expected == type_byte_width(t, structs)


def test_type_byte_width_struct_with_array_field():
    structs = {
        'A': semantic.StructInfo(
            name='A',
            fields={
                'x': semantic.Type(kind=semantic.TypeKind.STR),
                'y': semantic.Type(kind=semantic.TypeKind.INT),
                'arr': semantic.Type(
                    kind=semantic.TypeKind.ARRAY,
                    element_type=semantic.Type(kind=semantic.TypeKind.STR),
                    size=5,
                ),
            },
        ),
    }
    t = semantic.Type(kind=semantic.TypeKind.STRUCT, struct_name='A')
    expected = 52  # 8 + 4 + (8 * 5)
    assert expected == type_byte_width(t, structs)


def test_type_byte_width_struct_with_slice_field():
    structs = {
        'A': semantic.StructInfo(
            name='A',
            fields={
                'x': semantic.Type(kind=semantic.TypeKind.STR),
                'y': semantic.Type(kind=semantic.TypeKind.INT),
                'arr': semantic.Type(
                    kind=semantic.TypeKind.SLICE,
                    element_type=semantic.Type(kind=semantic.TypeKind.STR),
                ),
            },
        ),
    }
    t = semantic.Type(kind=semantic.TypeKind.STRUCT, struct_name='A')
    expected = 36  # 8 + 4 + 24
    assert expected == type_byte_width(t, structs)


def test_type_byte_width_struct_with_struct_array_field():
    structs = {
        'A': semantic.StructInfo(
            name='A',
            fields={
                'x': semantic.Type(kind=semantic.TypeKind.STR),
                'y': semantic.Type(kind=semantic.TypeKind.INT),
                'arr': semantic.Type(
                    kind=semantic.TypeKind.ARRAY,
                    element_type=semantic.Type(kind=semantic.TypeKind.STRUCT, struct_name='B'),
                    size=2,
                ),
            },
        ),
        'B': semantic.StructInfo(
            name='B',
            fields={
                'i': semantic.Type(kind=semantic.TypeKind.INT),
                'j': semantic.Type(kind=semantic.TypeKind.INT),
            }
        ),
    }
    t = semantic.Type(kind=semantic.TypeKind.STRUCT, struct_name='A')
    expected = 28  # 8 + 4 + ((4 + 4) * 2)
    assert expected == type_byte_width(t, structs)


def test_type_byte_width_struct_with_struct_slice_field():
    structs = {
        'A': semantic.StructInfo(
            name='A',
            fields={
                'x': semantic.Type(kind=semantic.TypeKind.STR),
                'y': semantic.Type(kind=semantic.TypeKind.INT),
                'arr': semantic.Type(
                    kind=semantic.TypeKind.SLICE,
                    element_type=semantic.Type(kind=semantic.TypeKind.STRUCT, struct_name='B'),
                ),
            },
        ),
        'B': semantic.StructInfo(
            name='B',
            fields={
                'i': semantic.Type(kind=semantic.TypeKind.INT),
                'j': semantic.Type(kind=semantic.TypeKind.INT),
            }
        ),
    }
    t = semantic.Type(kind=semantic.TypeKind.STRUCT, struct_name='A')
    expected = 36  # 8 + 4 + 24
    assert expected == type_byte_width(t, structs)


def test_type_byte_width_struct_with_self_referential_struct_array_field():
    structs = {
        'A': semantic.StructInfo(
            name='A',
            fields={
                'x': semantic.Type(kind=semantic.TypeKind.STR),
                'y': semantic.Type(kind=semantic.TypeKind.INT),
                'arr': semantic.Type(
                    kind=semantic.TypeKind.ARRAY,
                    element_type=semantic.Type(kind=semantic.TypeKind.STRUCT, struct_name='A'),
                    size=4,
                ),
            },
        ),
    }
    t = semantic.Type(kind=semantic.TypeKind.STRUCT, struct_name='A')

    with pytest.raises(RecursionError):
        type_byte_width(t, structs)


def test_leaf_type_no_array():
    t = semantic.Type(kind=semantic.TypeKind.INT)
    assert t == leaf_type(t)


def test_leaf_type_single_array():
    t = semantic.Type(kind=semantic.TypeKind.ARRAY, element_type=semantic.Type(kind=semantic.TypeKind.INT))
    assert semantic.Type(kind=semantic.TypeKind.INT) == leaf_type(t)


def test_leaf_type_multiple_array():
    t = semantic.Type(
        kind=semantic.TypeKind.ARRAY,
        element_type=semantic.Type(
            kind=semantic.TypeKind.ARRAY,
            element_type=semantic.Type(
                kind=semantic.TypeKind.ARRAY,
                element_type=semantic.Type(kind=semantic.TypeKind.INT),
            ),
        )
    )
    assert semantic.Type(kind=semantic.TypeKind.INT) == leaf_type(t)


def test_leaf_type_stops_at_slices():
    t = semantic.Type(kind=semantic.TypeKind.ARRAY, element_type=semantic.Type(kind=semantic.TypeKind.SLICE))
    assert semantic.Type(kind=semantic.TypeKind.SLICE) == leaf_type(t)


def test_type_of_no_type():
    node = parser.Constant(value=1)

    with pytest.raises(
            IRError,
            match=re.escape('Constant(value=1, resolved_type=None) has no resolved type -- semantic.analyze() must run'
                            ' before codegen (see compile_to_asm)')
    ):
        type_of(node)


def test_type_of_int():
    node = parser.Constant(value=1, resolved_type=semantic.Type(kind=semantic.TypeKind.INT))

    assert semantic.Type(kind=semantic.TypeKind.INT) == type_of(node)


def test_type_of_array():
    array_type = semantic.Type(
        kind=semantic.TypeKind.ARRAY,
        element_type=semantic.Type(
            kind=semantic.TypeKind.SLICE,
            element_type=semantic.Type(kind=semantic.TypeKind.INT),
        ),
        size=5,
    )
    node = parser.ArrayLiteral(
        resolved_type=array_type
    )
    assert array_type == type_of(node)


def test_composite_kinds_is_exactly_array_slice_struct():
    """Pinned exactly, not just checked for a subset/superset: every
    call site that switched to this shared constant (ir/statements.py,
    ir/arrays_slices.py, ir/builder.py, ir/dispatch.py, ir/structs.py)
    relies on it meaning precisely "composite, address-based value
    type" -- no more, no less."""
    assert COMPOSITE_KINDS == {semantic.TypeKind.ARRAY, semantic.TypeKind.SLICE, semantic.TypeKind.STRUCT}
