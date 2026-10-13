"""Hornet lays a value out as C lays out the same fields (typesys.py): checked by compiling C structs
equivalent to a program's structs and sums, on each target, and comparing what C's sizeof, alignof,
and offsetof give with typesys's widths, alignments, and field offsets."""

import re
import subprocess

from build import c_compiler, executable_name
from ir.program_builder import build_ir_program
from modules import discover_modules
from semantic import analyze
from tests.targets import each_runnable_target, run_binary
from typesys import SUM_PAYLOAD_OFFSET, Type, TypeKind, dict_bucket_layout, field_offsets, type_alignment, \
    type_byte_width

# Declared before use: C needs a struct defined before it is held by value.
PROGRAM = """\
type Color enum:
    Red
    Green

type Small is int32 | bool
type Wide is str | int8 | none
type Tiny is int8 | uint8

type Inner struct:
    int8 a
    int32 b

type Tail struct:
    int x
    int8 y

type Mixed struct:
    int8 a
    *Mixed next
    int32 b
    str s
    bool flag
    []int xs
    Inner inner
    [3]int8 bytes
    dict[str]int d
    Color c
    Small small
    Wide wide
    Tiny tiny
    [2]Tail tails
    uint8 last

type Bytes struct:
    int8 a
    uint8 b
    int8 c

type Holder is Mixed | Tail | Bytes

def int main():
    return 0
"""


def _c_name(name: str) -> str:
    return re.sub(r'\W', '_', name)


def _declare(t: Type, name: str) -> str:
    """A C declaration of `name` as Hornet's `t`."""
    if t.kind == TypeKind.ARRAY:
        dims = ''
        while t.kind == TypeKind.ARRAY:
            dims, t = dims + f'[{t.size}]', t.element_type
        return f'{_declare(t, name)}{dims}'
    simple = {Type.INT8: 'int8_t', Type.UINT8: 'uint8_t', Type.INT32: 'int32_t', Type.INT: 'int64_t'}
    if t in simple:
        return f'{simple[t]} {name}'
    if t.kind in (TypeKind.BOOL, TypeKind.ENUM):
        return f'int32_t {name}'
    if t.kind in (TypeKind.POINTER, TypeKind.DICT):
        return f'void *{name}'
    if t.kind == TypeKind.STR:
        return f'struct {{ void *ptr; int64_t len; }} {name}'
    if t.kind == TypeKind.SLICE:
        return f'struct {{ void *ptr; int64_t len; int64_t cap; }} {name}'
    if t.kind == TypeKind.STRUCT:
        return f'struct s_{_c_name(t.struct_name)} {name}'
    if t.kind == TypeKind.SUM:
        return f'struct u_{_c_name(t.sum_type_name)} {name}'
    raise AssertionError(f'no C for {t}')


def _c_program(structs: dict, sums: dict) -> str:
    """C that defines each struct and sum, then prints, one per line: `name size alignment` and
    `name.field offset`."""
    lines = ['#include <stdalign.h>', '#include <stddef.h>', '#include <stdint.h>', '#include <stdio.h>']
    order = [('sum', name) for name in sums if all(v.kind != TypeKind.STRUCT for v in sums[name].variants)]
    order += [('struct', name) for name in structs]
    order += [('sum', name) for name in sums if ('sum', name) not in order]
    for kind, name in order:
        if kind == 'struct':
            fields = ' '.join(f'{_declare(t, "f_" + f)};' for f, t in structs[name].fields.items())
            lines.append(f'struct s_{_c_name(name)} {{ {fields} }};')
        else:  # the tag, then the payload at SUM_PAYLOAD_OFFSET
            variants = [v for v in sums[name].variants if v.kind != TypeKind.NONE]
            members = ' '.join(f'{_declare(v, f"v{i}")};' for i, v in enumerate(variants))
            lines.append(f'struct u_{_c_name(name)} {{ int32_t tag; char pad[{SUM_PAYLOAD_OFFSET - 4}]; '
                         f'union {{ {members} }} payload; }};')
    lines.append('int main(void) {')
    for kind, name in order:
        c_type = f'struct {"s" if kind == "struct" else "u"}_{_c_name(name)}'
        lines.append(f'    printf("{name} %zu %zu\\n", sizeof({c_type}), alignof({c_type}));')
        if kind == 'struct':
            for field in structs[name].fields:
                lines.append(f'    printf("{name}.{field} %zu\\n", offsetof({c_type}, f_{field}));')
        else:
            lines.append(f'    printf("{name}.payload %zu\\n", offsetof({c_type}, payload));')
    lines += ['    return 0;', '}']
    return '\n'.join(lines) + '\n'


def _registries(tmp_path):
    (tmp_path / 'p.ht').write_text(PROGRAM)
    program = build_ir_program(analyze(*discover_modules(str(tmp_path / 'p.ht'))))
    return program.struct_registry, program.sum_type_registry


def test_the_layout_is_c_s(tmp_path):
    structs, sums = _registries(tmp_path)
    assert set(structs) >= {'Mixed', 'Inner', 'Tail', 'Bytes'} and set(sums) >= {'Small', 'Wide', 'Holder'}
    assert field_offsets('Mixed', structs, sums)['next'] == 8  # (after an int8: no pointer is split)
    assert type_alignment(Type(TypeKind.SUM, sum_type_name='Tiny'), structs, sums) == 4  # (its tag's)


@each_runnable_target
def test_sizes_alignments_and_offsets_are_what_c_gives(target, tmp_path):
    structs, sums = _registries(tmp_path)
    (tmp_path / 'layout.c').write_text(_c_program(structs, sums))
    exe = tmp_path / executable_name('layout', target)
    subprocess.run([*c_compiler(target), '-std=c11', str(tmp_path / 'layout.c'), '-o', str(exe)],
                   check=True, capture_output=True, text=True)
    c = run_binary(target, [exe], capture_output=True, text=True).stdout
    expected = []
    for kind, registry in (('struct', structs), ('sum', sums)):
        for name in registry:
            t = Type(TypeKind.STRUCT, struct_name=name) if kind == 'struct' else Type(TypeKind.SUM, sum_type_name=name)
            expected.append(f'{name} {type_byte_width(t, structs, sums)} {type_alignment(t, structs, sums)}')
            if kind == 'struct':
                expected += [f'{name}.{field} {offset}' for field, offset in field_offsets(name, structs, sums).items()]
            else:
                expected.append(f'{name}.payload {SUM_PAYLOAD_OFFSET}')
    assert sorted(c.splitlines()) == sorted(expected)


def test_a_dict_bucket_holds_its_key_and_value_where_their_alignments_put_them():
    structs, sums = {}, {}
    assert dict_bucket_layout(Type.INT8, Type.INT, structs, sums) == (1, 8, 16)
    assert dict_bucket_layout(Type.STR, Type.INT8, structs, sums) == (8, 24, 32)
    assert dict_bucket_layout(Type.INT32, Type.BOOL, structs, sums) == (4, 8, 12)
    assert dict_bucket_layout(Type.UINT8, Type.UINT8, structs, sums) == (1, 2, 3)
