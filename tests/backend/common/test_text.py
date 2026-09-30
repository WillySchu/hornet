"""backend/common/text.py."""

from backend.common.text import escape_for_asciz


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


