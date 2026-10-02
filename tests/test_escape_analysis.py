"""Tests for escape_analysis.py"""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import desugar
import escape_analysis as ea
import parser
import semantic
from lexer import lex


def parse_and_analyze(source: str):
    """The typed program (escape analysis runs on the typed tree), plus the registries it needs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / 'program.ht'
        src_path.write_text(source)
        tokens = lex(str(src_path))
        ast = parser.Parser(tokens).parse_program()
        desugar.desugar_methods(ast)
        typed = semantic.analyze(ast)
        return SimpleNamespace(functions=list(typed.functions), struct_registry=ast.struct_registry,
                               symbols=ast.symbols)


def parse_expression(source: str) -> parser.Node:
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / 'program.ht'
        src_path.write_text(source)
        tokens = lex(str(src_path))
        ast = parser.Parser(tokens).parse_expression()
        return ast


def test_is_heap_allocated_int():
    t = semantic.Type(kind=semantic.TypeKind.INT)
    assert not ea.is_heap_allocated(t, {}, {})


def test_is_heap_allocated_str():
    t = semantic.Type(kind=semantic.TypeKind.STR)
    assert not ea.is_heap_allocated(t, {}, {})


def test_is_heap_allocated_bool():
    t = semantic.Type(kind=semantic.TypeKind.BOOL)
    assert not ea.is_heap_allocated(t, {}, {})


def test_is_heap_allocated_none():
    t = semantic.Type(kind=semantic.TypeKind.NONE)
    assert not ea.is_heap_allocated(t, {}, {})


def test_is_heap_allocated_array_int_stack():
    size = 2048  # int = 8, 8 * 2048 = 16384
    t = semantic.Type(kind=semantic.TypeKind.ARRAY, element_type=semantic.Type(kind=semantic.TypeKind.INT), size=size)
    assert not ea.is_heap_allocated(t, {}, {})


def test_is_heap_allocated_array_int_heap():
    size = 2049  # int = 8, 8 * 2049 = 16392
    t = semantic.Type(kind=semantic.TypeKind.ARRAY, element_type=semantic.Type(kind=semantic.TypeKind.INT), size=size)
    assert ea.is_heap_allocated(t, {}, {})


def test_is_heap_allocated_array_str_stack():
    size = 1024  # str = 16, 1024 * 16 = 16384
    t = semantic.Type(kind=semantic.TypeKind.ARRAY, element_type=semantic.Type(kind=semantic.TypeKind.STR), size=size)
    assert not ea.is_heap_allocated(t, {}, {})


def test_is_heap_allocated_array_str_heap():
    size = 1025  # str = 16, 1025 * 16 = 16400
    t = semantic.Type(kind=semantic.TypeKind.ARRAY, element_type=semantic.Type(kind=semantic.TypeKind.STR), size=size)
    assert ea.is_heap_allocated(t, {}, {})


# TODO(will): Test Structs.


def test_analyze_array_escapes_empty():
    ast = parse_and_analyze("def f():\n    return\n")
    assert ea.analyze_array_escapes(ast.functions[0], {}) == set()


def test_analyze_array_escapes_fn_on_uninitialized_slice():
    ast = parse_and_analyze("def f():\n    []int sl\n    fn(sl)\n\ndef fn([]int x):\n    print(len(x))\n")
    assert ea.analyze_array_escapes(ast.functions[0], {}) == set()


# TODO(will): I feel like this should escape?
def test_analyze_array_escapes_fn_on_initialized_slice():
    ast = parse_and_analyze("def f():\n    []int sl = [1, 2, 3]\n    fn(sl)\n\ndef fn([]int x):\n    print(len(x))\n")
    assert ea.analyze_array_escapes(ast.functions[0], {}) == set()


# TODO(will): I feel like this should escape?
def test_analyze_array_escapes_return_initialized_slice():
    ast = parse_and_analyze("def []int f():\n    []int sl = [1, 2, 3]\n    return sl\n")
    assert ea.analyze_array_escapes(ast.functions[0], {}) == set()


def test_analyze_array_escapes_return_sliced_array():
    ast = parse_and_analyze("def []int f():\n    [5]int arr = [1, 2, 3, 4, 5]\n    []int sl = arr[arr[1]:arr[3]]\n"
                            "    return sl\n")
    assert len(ea.analyze_array_escapes(ast.functions[0], {})) == 1


# TODO(will): I feel like this should escape?
def test_test():
    source = '''
def int main():
    []int ints = [1, 2, 3]
    print_ints(ints)
    return 0


def print_ints([]int ints):
    print(ints)
'''
    ast = parse_and_analyze(source)
    main_res = ea.analyze_array_escapes(ast.functions[0], {})
    assert main_res == set()
    print_ints_res = ea.analyze_array_escapes(ast.functions[1], {})
    assert print_ints_res == set()


def test_test2():
    source = (
        "def []int sliceints([5]int arr):\n"
        "    []int a = arr[:]\n"
        "    return a\n"
        "\n"
        "def int helper(int x):\n"
        "    int a = x + 1\n"
        "    int b = a + 1\n"
        "    return a + b\n"
        "\n"
        "def int main():\n"
        "    [5]int arr = [1, 2, 3, 4, 5]\n"
        "    []int sl = sliceints(arr)\n"
        "    int junk = helper(1)\n"
        "    junk = helper(2)\n"
        "    junk = helper(3)\n"
        "    print(sl)\n"
        "    return 0\n"
    )

    ast = parse_and_analyze(source)
    sliceints_res = ea.analyze_array_escapes(ast.functions[0], {})
    assert len(sliceints_res) == 1
    assert set() == ea.analyze_array_escapes(ast.functions[1], {})
    assert set() == ea.analyze_array_escapes(ast.functions[2], {})


def _escapes(source: str, fn_index: int = 0) -> tuple:
    ast = parse_and_analyze(source)
    fn = ast.functions[fn_index]
    return fn, ea.analyze_array_escapes(fn, ast.struct_registry)


def test_store_through_pointer_param_escapes():
    fn, res = _escapes(
        "type S struct:\n"
        "    []int s\n"
        "def fill(*S out):\n"
        "    [3]int a = [1, 2, 3]\n"
        "    out.s = a[:]\n"
    )
    assert fn.body[0].symbol.id in res


def test_store_into_dict_param_escapes():
    fn, res = _escapes(
        "def put(dict[int][]int d):\n"
        "    [3]int a = [1, 2, 3]\n"
        "    d[0] = a[:]\n"
    )
    assert fn.body[0].symbol.id in res


def test_store_into_slice_param_escapes():
    fn, res = _escapes(
        "def put([][]int rows):\n"
        "    [3]int a = [1, 2, 3]\n"
        "    rows[0] = a[:]\n"
    )
    assert fn.body[0].symbol.id in res


def test_append_element_escapes_with_result():
    fn, res = _escapes(
        "def [][]int f():\n"
        "    [3]int a = [1, 2, 3]\n"
        "    [][]int s\n"
        "    s = append(s, a[:])\n"
        "    return s\n"
    )
    assert fn.body[0].symbol.id in res


def test_local_only_aggregate_stays_on_stack():
    fn, res = _escapes(
        "def int f():\n"
        "    [3]int a = [1, 2, 3]\n"
        "    [][]int rows = [][]int[a[:]]\n"
        "    return len(rows[0])\n"
    )
    assert fn.body[0].symbol.id not in res


def test_store_into_local_via_pointer_then_return_escapes():
    fn, res = _escapes(
        "type S struct:\n"
        "    []int s\n"
        "def S f():\n"
        "    [3]int a = [1, 2, 3]\n"
        "    S v = S([])\n"
        "    *S q = &v\n"
        "    q.s = a[:]\n"
        "    return v\n"
    )
    assert fn.body[0].symbol.id in res
    assert fn.body[1].symbol.id not in res

# ---------------------------------------------------------------------------
# Pointers: `&x` as a second way to produce a direct_backing edge,
# alongside array-slicing -- the generalization discussed at length before
# any of this was written. Two real bugs were found and fixed while
# building it, both by actually running programs rather than trusting the
# analysis alone:
#   1. _ir_address_of never had any concept of a heap-allocated variable
#      at all -- it always computed &x as the slot's own address, but a
#      heap-allocated variable's slot holds a POINTER to the real data,
#      not the data itself. Fixed by mirroring _ir_struct_address's own
#      "if heap-allocated, load through one more indirection" pattern.
#   2. A scalar whose address escapes now gets real heap-promotion
#      machinery too (see _ir_finish_scalar_var_decl, ir/statements.py):
#      is_heap_allocated's own escape check was already written
#      generically for any type, but nothing in a scalar VarDecl's own
#      construction mallocs one until now. Silently treating an escaping
#      scalar like any other escaping declaration WITHOUT that machinery
#      would have read its raw VALUE as if it were a pointer, corrupting
#      it -- rejected outright at first (a deliberate, narrower v1 scope
#      decision, not a soundness gap left open by accident), with the
#      heap-promotion machinery itself following as its own, later stage.
# See tests/test_compiler.py's own TestPointerEscapeAnalysis for the
# compile-and-run counterparts, including the dangling-pointer program
# that motivated this whole stage.
# ---------------------------------------------------------------------------

def test_address_of_a_struct_local_returned_directly_escapes():
    ast = parse_and_analyze(
        "type Circle struct:\n"
        "    int radius\n"
        "\n"
        "def *Circle makeCircle():\n"
        "    Circle c = Circle(5)\n"
        "    return &c\n"
    )
    fn = ast.functions[0]
    c_decl_id = fn.body[0].symbol.id
    result = ea.analyze_array_escapes(fn, ast.struct_registry)
    assert c_decl_id in result


def test_address_of_a_struct_local_never_escapes_stays_out_of_the_result():
    ast = parse_and_analyze(
        "type Circle struct:\n"
        "    int radius\n"
        "\n"
        "def int main():\n"
        "    Circle c = Circle(5)\n"
        "    *Circle p = &c\n"
        "    return p.radius\n"
    )
    fn = ast.functions[0]
    c_decl_id = fn.body[0].symbol.id
    result = ea.analyze_array_escapes(fn, ast.struct_registry)
    assert c_decl_id not in result


def test_address_of_a_scalar_local_returned_directly_is_now_heap_promoted():
    """Was rejected outright before scalar heap-promotion existed --
    now a scalar's decl_id, exactly like an array/struct/sum one, just
    lands in the returned escaping set."""
    ast = parse_and_analyze(
        "def *int makeDangling():\n"
        "    int x = 42\n"
        "    return &x\n"
    )
    fn = ast.functions[0]
    x_decl_id = fn.body[0].symbol.id
    result = ea.analyze_array_escapes(fn, ast.struct_registry)
    assert x_decl_id in result


def test_address_of_a_scalar_local_wrapped_in_a_returned_struct_is_now_heap_promoted():
    """Confirms the same holds when &x is passed as a struct
    CONSTRUCTOR argument (Holder(&x)) rather than returned bare --
    reached via scan_expr_for_escaping_calls's own conservative "any
    call's arguments might escape" treatment, which a struct
    constructor call falls under too (contribution() has no dedicated
    case for a struct literal itself, a separate, pre-existing gap for
    slices too -- deliberately out of scope here), not via
    contribution() being given the whole struct literal directly."""
    ast = parse_and_analyze(
        "type Holder struct:\n"
        "    *int p\n"
        "\n"
        "def Holder makeDangling():\n"
        "    int x = 42\n"
        "    return Holder(&x)\n"
    )
    fn = ast.functions[0]
    x_decl_id = fn.body[0].symbol.id
    result = ea.analyze_array_escapes(fn, ast.struct_registry)
    assert x_decl_id in result


def test_address_of_a_scalar_local_passed_as_a_call_argument_is_now_heap_promoted():
    """The existing, pre-pointer conservatism (any call argument might
    escape, intraprocedurally) already covers this -- &x passed to
    ANY user-defined function is treated the same as returning it
    directly, matching how a slice argument already works."""
    ast = parse_and_analyze(
        "def int useIt(*int p):\n"
        "    return *p\n"
        "\n"
        "def int caller():\n"
        "    int x = 7\n"
        "    return useIt(&x)\n"
    )
    fn = ast.functions[1]  # caller, not useIt
    x_decl_id = fn.body[0].symbol.id
    result = ea.analyze_array_escapes(fn, ast.struct_registry)
    assert x_decl_id in result


def test_address_of_a_field_resolves_to_the_containing_struct():
    """`&s.field` escaping doesn't back the pointer with the field
    itself -- there's no separate box for one field to escape into --
    it backs it with s's own, WHOLE containing declaration, which the
    existing composite heap-promotion machinery already knows how to
    heap-allocate entirely once it's marked escaping this way."""
    ast = parse_and_analyze(
        "type Circle struct:\n"
        "    int radius\n"
        "\n"
        "def *int getFieldAddr():\n"
        "    Circle s = Circle(5)\n"
        "    return &s.radius\n"
    )
    fn = ast.functions[0]
    s_decl_id = fn.body[0].symbol.id
    result = ea.analyze_array_escapes(fn, ast.struct_registry)
    assert s_decl_id in result


def test_address_of_an_element_resolves_to_the_containing_array():
    """The Index counterpart to the Field case just above."""
    ast = parse_and_analyze(
        "def *int getElemAddr():\n"
        "    [3]int arr = [1, 2, 3]\n"
        "    return &arr[1]\n"
    )
    fn = ast.functions[0]
    arr_decl_id = fn.body[0].symbol.id
    result = ea.analyze_array_escapes(fn, ast.struct_registry)
    assert arr_decl_id in result


def test_address_of_a_field_through_an_auto_dereferenced_pointer_does_not_escape_the_pointer():
    """The critical guard: `&p.field` where p itself is POINTER-typed
    (auto-deref) must NOT mark p's own decl_id as escaping. p's own
    pointee is already, independently safe -- heap-allocated (or
    otherwise valid) wherever p first came from, entirely outside this
    function's own control -- and &p.field is just that already-valid
    address plus an offset. Marking p itself as escaping here would be
    actively wrong: p is an ordinary, freely-copied pointer VALUE, and
    would incorrectly trigger scalar heap-promotion (Temp boxing) on
    the pointer variable itself, which has nothing to do with what's
    actually happening."""
    ast = parse_and_analyze(
        "type Circle struct:\n"
        "    int radius\n"
        "\n"
        "def *int getFieldThroughPointer(*Circle p):\n"
        "    return &p.radius\n"
    )
    fn = ast.functions[0]
    p_decl_id = fn.params[0].id
    result = ea.analyze_array_escapes(fn, ast.struct_registry)
    assert p_decl_id not in result


def test_address_of_via_reassignment_not_just_var_decl_init():
    """`p = &x` (Assign, an EXISTING pointer variable reassigned) needs
    the identical treatment `*int p = &x` (VarDecl's own init) already
    gets -- walk_statements' own Assign case mirrors its VarDecl case
    exactly, but this exercises that path directly rather than only
    ever through a VarDecl's own initializer."""
    ast = parse_and_analyze(
        "type Circle struct:\n"
        "    int radius\n"
        "\n"
        "def *Circle makeCircle():\n"
        "    Circle c = Circle(5)\n"
        "    *Circle p = none\n"
        "    p = &c\n"
        "    return p\n"
    )
    fn = ast.functions[0]
    c_decl_id = fn.body[0].symbol.id
    result = ea.analyze_array_escapes(fn, ast.struct_registry)
    assert c_decl_id in result


def test_pointer_aliasing_through_a_struct_field_assign_propagates():
    """s.field = someOtherPointer (FieldAssign, a pointer-VALUED RHS
    that's itself a Variable, not a bare &x) needs to propagate the
    aliasing through slice_deps -- if the whole struct later escapes,
    whatever someOtherPointer itself pointed at must be found too."""
    ast = parse_and_analyze(
        "type Circle struct:\n"
        "    int radius\n"
        "\n"
        "type Holder struct:\n"
        "    *Circle p\n"
        "\n"
        "def Holder makeHolder():\n"
        "    Circle c = Circle(5)\n"
        "    *Circle q = &c\n"
        "    Holder h = Holder(none)\n"
        "    h.p = q\n"
        "    return h\n"
    )
    fn = ast.functions[0]
    c_decl_id = fn.body[0].symbol.id
    result = ea.analyze_array_escapes(fn, ast.struct_registry)
    assert c_decl_id in result



def _heap_names(source: str, fn_index: int = 0) -> set:
    """Names of the VarDecls (and for-in bindings) in one function that need the heap."""
    ast = parse_and_analyze(source)
    fn = ast.functions[fn_index]
    heap = ea.analyze_array_escapes(fn, {}, ea.compute_escape_summaries(ast.functions, {}))
    return {ast.symbols[i].name for i in heap if isinstance(i, int) and ast.symbols[i].kind in ('local', 'binding')}


@pytest.mark.parametrize("loop,decl", [
    ("for int i = 0; i < 3; i += 1:", "        int x = i"),
    ("while len(ps) < 3:", "        int x = 1"),
    ("for v in [1, 2]:", "        int x = v"),
])
def test_body_declaration_held_outside_its_loop_needs_the_heap(loop, decl):
    source = f"def int main():\n    []*int ps\n    {loop}\n{decl}\n        ps = append(ps, &x)\n    print(len(ps))\n    return 0\n"
    assert 'x' in _heap_names(source)


def test_loop_variable_and_binding_held_outside_the_loop_need_the_heap():
    source = ("def int main():\n    []*int ps\n    for int i = 0; i < 3; i += 1:\n        ps = append(ps, &i)\n"
              "    for v in [1, 2]:\n        ps = append(ps, &v)\n    print(len(ps))\n    return 0\n")
    assert _heap_names(source) == {'i', 'v'}


def test_address_used_only_within_the_iteration_stays_on_the_stack():
    source = ("def bump(*int p):\n    *p += 1\n"
              "def int main():\n    int t = 0\n    for int i = 0; i < 3; i += 1:\n        int x = i\n"
              "        *int p = &x\n        *p += 1\n        bump(&x)\n        t += x\n    return t\n")
    assert _heap_names(source, 1) == set()


def test_inner_loop_declaration_held_by_the_outer_loop_body_needs_the_heap():
    source = ("def int main():\n    for int i = 0; i < 2; i += 1:\n        *int last = none\n"
              "        for int j = 0; j < 2; j += 1:\n            int x = j\n            last = &x\n        print(*last)\n    return 0\n")
    assert _heap_names(source) == {'x'}
