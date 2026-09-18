"""Tests for codegen/utils.py's own machine-level helpers -- register-
width aliasing and GAS string escaping. See tests/codegen/test_ir_
utils.py for the semantic-level half (type_byte_width/leaf_type/
type_of) split out alongside ir/utils.py itself."""

from codegen.utils import escape_for_asciz

# TODO(will): Test as_qword_register


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


