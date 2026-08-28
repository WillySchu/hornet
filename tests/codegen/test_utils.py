"""Tests for type_byte_width.py, likely to be merged into another file."""

import pytest

import semantic
from codegen.utils import escape_for_asciz, leaf_type, type_byte_width


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


def test_escape_for_asciz():
    tcs = [
        {
            'name': 'No escape.',
            'in': 'asdf',
            'expected': 'asdf',
        },
        {
            'name': 'Escape backslashes.',
            'in': 'as\df',
            'expected': 'as\\\\df',
        },
        {
            'name': 'Escape double quotes.',
            'in': 'as"df',
            'expected': 'as\\"df',
        },
        {
            'name': 'Escape newline.',
            'in': 'as\ndf',
            'expected': 'as\\ndf',
        },
        {
            'name': 'Escape tab.',
            'in': 'as\tdf',
            'expected': 'as\\tdf',
        },
        {
            'name': 'Escape carriage return.',
            'in': 'as\rdf',
            'expected': 'as\\rdf',
        },
    ]

    for tc in tcs:
        assert tc['expected'] == escape_for_asciz(tc['in']), tc['name']
