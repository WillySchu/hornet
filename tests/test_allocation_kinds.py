"""Each allocation says whether its memory may hold a pointer (typesys.holds_pointers), so that a
collector need look for pointers only where there may be some: what holds none is `_atomic`."""

import ast
from pathlib import Path

import pytest

from ir.ir import IRCall, IRStore
from ir.program_builder import build_ir_program
from tests.test_compiler import _parse, analyze
from typesys import DICT_HOLDS_POINTERS_FIELD, Type, TypeKind, holds_pointers

ROOT = Path(__file__).resolve().parent.parent

DECLS = """\
type Color enum:
    Red
    Green

type Plain struct:
    int8 a
    [3]int b
    Color c

type Named struct:
    int id
    str name

type Numbers is int | bool | none
type Mixed is int | str
type Nested is Plain | Numbers

"""


def _program(functions: str = ""):
    return build_ir_program(analyze(_parse(DECLS + functions + "def int main():\n    return 0\n")),
                            keep_unreachable=True)


def _calls(functions: str, name: str) -> list:
    """The allocation calls (hornet_alloc*, hornet_slice_grow*) in function `name`, by name."""
    fn = next(f for f in _program(functions).functions if f.name == name)
    return [i.name for i in fn.body if isinstance(i, IRCall) and i.name.startswith(('hornet_alloc', 'hornet_slice'))]


def _type(text: str, program) -> Type:
    kinds = {'int': Type.INT, 'int8': Type.INT8, 'bool': Type.BOOL, 'str': Type.STR}
    if text in kinds:
        return kinds[text]
    if text.startswith('[]'):
        return Type(TypeKind.SLICE, element_type=_type(text[2:], program))
    if text.startswith('*'):
        return Type(TypeKind.POINTER, element_type=_type(text[1:], program))
    if text in program.struct_registry:
        return Type(TypeKind.STRUCT, struct_name=text)
    if text in program.sum_type_registry:
        return Type(TypeKind.SUM, sum_type_name=text)
    return Type(TypeKind.ENUM, enum_name=text)


@pytest.mark.parametrize("text, holds", [
    ("int", False), ("int8", False), ("bool", False), ("Color", False),
    ("Plain", False), ("Numbers", False), ("Nested", False),
    ("str", True), ("[]int", True), ("*int", True), ("Named", True), ("Mixed", True),
])
def test_what_may_hold_a_pointer(text, holds):
    program = _program()
    t = _type(text, program)
    assert holds_pointers(t, program.struct_registry, program.sum_type_registry) == holds
    arrays = Type(TypeKind.ARRAY, element_type=Type(TypeKind.ARRAY, element_type=t, size=2), size=3)
    assert holds_pointers(arrays, program.struct_registry, program.sum_type_registry) == holds
    dict_type = Type(TypeKind.DICT, key_type=Type.INT, element_type=t)
    assert holds_pointers(dict_type, program.struct_registry, program.sum_type_registry)  # (its header)


@pytest.mark.parametrize("element, grow", [
    ("byte", "hornet_slice_grow_atomic"), ("int", "hornet_slice_grow_atomic"), ("Plain", "hornet_slice_grow_atomic"),
    ("Numbers", "hornet_slice_grow_atomic"), ("str", "hornet_slice_grow"), ("*int", "hornet_slice_grow"),
    ("Mixed", "hornet_slice_grow"), ("[]int", "hornet_slice_grow"), ("Named", "hornet_slice_grow"),
])
def test_a_slice_grows_as_its_elements_hold(element, grow):
    one = f"def []{element} one([]{element} xs, {element} x):\n    return append(xs, x)\n\n"
    every = f"def []{element} every([]{element} xs, []{element} ys):\n    return append(xs, ys...)\n\n"
    assert _calls(one + every, 'one$') == [grow] and _calls(one + every, 'every$') == [grow]


def test_a_str_s_bytes_hold_no_pointer():
    joined = "def str joined(str a, str b):\n    return a + b\n\n"
    copied = "def str copied([]byte b):\n    return str(b)\n\n"
    assert _calls(joined, 'joined$') == ['hornet_alloc_atomic']
    assert _calls(copied, 'copied$') == ['hornet_alloc_atomic']


def test_a_variable_on_the_heap_is_allocated_as_it_holds():
    number = "def *int number():\n    int x = 1\n    return &x\n\n"
    text = "def *str text():\n    str s = 'a'\n    return &s\n\n"
    plain = "def *Plain plain():\n    return &Plain(1, [3]int[1, 2, 3], Color.Red)\n\n"
    named = "def *Named named():\n    return &Named(1, 'a')\n\n"
    program = number + text + plain + named
    assert _calls(program, 'number$') == ['hornet_alloc_atomic']
    assert _calls(program, 'text$') == ['hornet_alloc']
    assert _calls(program, 'plain$') == ['hornet_alloc_atomic']
    assert _calls(program, 'named$') == ['hornet_alloc']


@pytest.mark.parametrize("dict_type, holds", [
    ("dict[int]int", 0), ("dict[int]Plain", 0), ("dict[Color]Numbers", 0),
    ("dict[str]int", 1), ("dict[int]str", 1), ("dict[int]Mixed", 1), ("dict[int]*int", 1),
])
def test_a_dict_s_buckets_are_allocated_as_they_hold(dict_type, holds):
    key = "'k'" if "[str]" in dict_type else ("Color.Red" if "[Color]" in dict_type else "1")
    value = {"int": "2", "Plain": "Plain(1, [3]int[1, 2, 3], Color.Red)", "Numbers": "true", "str": "'v'",
             "Mixed": "3", "*int": "none"}[dict_type[dict_type.index(']') + 1:]]
    functions = (f"def {dict_type} literal():\n    return {dict_type}{{{key}: {value}}}\n\n"
                 f"def {dict_type} empty():\n    {dict_type} d\n    return d\n\n")
    buckets = 'hornet_alloc_zeroed' if holds else 'hornet_alloc_zeroed_atomic'
    assert _calls(functions, 'literal$') == [buckets, 'hornet_alloc_zeroed']  # (the header points to them)
    assert _calls(functions, 'empty$') == ['hornet_alloc_zeroed']
    for name in ('literal$', 'empty$'):  # what the runtime allocates them as when the dict grows
        fn = next(f for f in _program(functions).functions if f.name == name)
        offsets = {i.dst: i.right.value for i in fn.body if hasattr(i, 'right') and hasattr(i.right, 'value')}
        flags = [i.value.value for i in fn.body
                 if isinstance(i, IRStore) and offsets.get(i.address) == DICT_HOLDS_POINTERS_FIELD]
        assert flags == [holds]


def _allocation_names(tree) -> list:
    """(function, name) for each string constant naming a runtime allocation."""
    found = []
    for function in ast.walk(tree):
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for node in ast.walk(function):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                        and node.value.startswith(('hornet_alloc', 'hornet_slice_grow')):
                    found.append((function.name, node.value))
    return found


def test_the_compiler_allocates_only_through_the_two_calls_that_say_what_memory_holds():
    sources = [p for d in ('ir', 'optimize', 'backend', 'semantic') for p in (ROOT / d).rglob('*.py')]
    sources += [ROOT / name for name in ('compile.py', 'escape_analysis.py')]
    places = {(path.relative_to(ROOT).as_posix(), function) for path in sources
              for function, _ in _allocation_names(ast.parse(path.read_text()))}
    assert places == {('ir/typed_builder.py', 'allocation'), ('ir/typed_builder.py', 'slice_growth')}
