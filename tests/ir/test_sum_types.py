"""Tests for sum-type IR construction (ir/sum_types.py, and every
other place a struct value widening into a sum type had to be taught
about -- ir/statements.py's VarDecl/Return handling, ir/builder.py's
argument-temp reservation, ir/scalars.py's argument marshaling).

These are deliberately INTEGRATION-level, building a real Program
through the full lex -> parse -> desugar -> analyze -> build_ir_
program pipeline, not unit tests of _ir_write_sum_type_value_into in
isolation -- every bug actually found while building this feature was
about how separate parts of the pipeline disagreed about a value's own
type (the argument's own type vs. the parameter's declared type, the
value's own type vs. the function's declared return type), which an
isolated unit test of one function alone would never have caught."""

import tempfile
from pathlib import Path

import desugar
import parser
import semantic
from lexer import lex
from ir.ir import IRBinOp, IRConst, IRCopy, IRSliceGrow, IRStore
from ir.program_builder import build_ir_program
from semantic import Type, TypeKind


def _build(source: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "program.ht"
        src_path.write_text(source)
        tokens = lex(str(src_path))
        ast = parser.Parser(tokens).parse_program()
        desugar.desugar_methods(ast)
        semantic.analyze(ast)
        return build_ir_program(ast)


def _fn(ir_program, name):
    return [f for f in ir_program.functions if f.name == name][0]


def _discriminant_writes(body):
    """Every IRStore writing a plain INT IRConst -- both the tag
    itself and (coincidentally, for these small test structs) a
    scalar field's own value have this exact shape, so callers filter
    by expected VALUE, not just presence, to tell them apart."""
    return [instr for instr in body if isinstance(instr, IRStore) and isinstance(instr.value, IRConst) and instr.value.type == Type.INT]


# Circle first, Square second, throughout -- so Circle's own
# discriminant is always 0 and Square's is always 1.
_SHAPE_DECLS = (
    "type Circle struct:\n"
    "    int radius\n"
    "\n"
    "type Square struct:\n"
    "    int64 side\n"  # int64, deliberately -- makes Circle's own width (4) and
    "\n"                # Shape's own width (4 tag + 8) unambiguously different,
    "type Shape is Circle | Square\n"  # unlike if both variants were plain int.
    "\n"
)


def test_var_decl_literal_widening_writes_circles_own_discriminant():
    ir_program = _build(
        _SHAPE_DECLS +
        "def int main():\n"
        "    Shape s = Circle(5)\n"
        "    return 0\n"
    )
    fn = _fn(ir_program, 'main')
    tag_writes = [w for w in _discriminant_writes(fn.body) if w.value.value == 0]
    assert len(tag_writes) == 1


def test_var_decl_literal_widening_writes_squares_own_discriminant():
    ir_program = _build(
        _SHAPE_DECLS +
        "def int main():\n"
        "    Shape s = Square(9)\n"
        "    return 0\n"
    )
    fn = _fn(ir_program, 'main')
    tag_writes = [w for w in _discriminant_writes(fn.body) if w.value.value == 1]
    assert len(tag_writes) == 1


def test_var_decl_literal_widening_reserves_shapes_own_width_not_circles():
    """Circle alone is 4 bytes; Shape is 4 (tag) + 8 (Square, the
    wider variant) = 12. `Shape s` must reserve a Shape-shaped slot,
    not a Circle-shaped one -- this is exactly the class of bug found
    while building this feature (a slot sized from the wrong side of
    the widening)."""
    ir_program = _build(
        _SHAPE_DECLS +
        "def int main():\n"
        "    Shape s = Circle(5)\n"
        "    return 0\n"
    )
    fn = _fn(ir_program, 'main')
    assert 12 in fn.slot_widths.values()
    assert 4 not in fn.slot_widths.values()


def test_var_decl_from_already_typed_variable_copies_at_circles_own_width():
    """`Shape s = c`, c already Circle-typed -- not a struct literal.
    The IRCopy writing the payload must be tagged with Circle's own
    (narrower) type, not Shape's -- copying Shape's own width from
    Circle's own (shorter) address would read past its real bounds."""
    ir_program = _build(
        _SHAPE_DECLS +
        "def int main():\n"
        "    Circle c = Circle(5)\n"
        "    Shape s = c\n"
        "    return 0\n"
    )
    fn = _fn(ir_program, 'main')
    copies = [instr for instr in fn.body if isinstance(instr, IRCopy)]
    assert len(copies) == 1
    assert copies[0].value_type == Type(TypeKind.STRUCT, struct_name='Circle')


def test_assign_widening_writes_discriminant():
    ir_program = _build(
        _SHAPE_DECLS +
        "def int main():\n"
        "    Shape s = Circle(5)\n"
        "    s = Square(9)\n"
        "    return 0\n"
    )
    fn = _fn(ir_program, 'main')
    tag_writes = [w for w in _discriminant_writes(fn.body) if w.value.value == 1]
    assert len(tag_writes) == 1


def test_return_widening_writes_discriminant():
    """`return Circle(5)` from a function declared to return Shape --
    the hidden-pointer write must include the tag, not just Circle's
    own raw field bytes (the actual bug found: this path originally
    used the VALUE's own type, Circle, instead of the function's
    DECLARED return type, Shape)."""
    ir_program = _build(
        _SHAPE_DECLS +
        "def Shape makeShape():\n"
        "    return Circle(5)\n"
        "\n"
        "def int main():\n"
        "    return 0\n"
    )
    fn = _fn(ir_program, 'makeShape')
    tag_writes = [w for w in _discriminant_writes(fn.body) if w.value.value == 0]
    assert len(tag_writes) == 1


def test_function_argument_widening_reserves_target_sized_slot():
    """takesShape(Circle(5)) -- the reserved argument-temp slot must
    be sized for Shape (12 bytes), not Circle (4 bytes): the actual
    bug found, since argument-temp reservation had no way to see the
    callee's own declared parameter type at all before function_
    registry existed."""
    ir_program = _build(
        _SHAPE_DECLS +
        "def int takesShape(Shape s):\n"
        "    return 0\n"
        "\n"
        "def int main():\n"
        "    return takesShape(Circle(5))\n"
    )
    fn = _fn(ir_program, 'main')
    assert 12 in fn.slot_widths.values()
    assert 4 not in fn.slot_widths.values()


def test_function_argument_widening_writes_discriminant():
    ir_program = _build(
        _SHAPE_DECLS +
        "def int takesShape(Shape s):\n"
        "    return 0\n"
        "\n"
        "def int main():\n"
        "    return takesShape(Square(9))\n"
    )
    fn = _fn(ir_program, 'main')
    tag_writes = [w for w in _discriminant_writes(fn.body) if w.value.value == 1]
    assert len(tag_writes) == 1


def test_already_sum_typed_argument_needs_no_widening_but_still_a_real_address():
    """takesShape(s), s already Shape-typed -- no widening involved at
    all, but still needs SOME real address-based argument passing, not
    the scalar gen_expr_ir path a 12-byte value can't fit through (the
    other bug this same investigation found: arg_type.kind == SUM had
    no case at all in _ir_call_arguments before this)."""
    ir_program = _build(
        _SHAPE_DECLS +
        "def int takesShape(Shape s):\n"
        "    return 0\n"
        "\n"
        "def int main():\n"
        "    Shape s = Circle(5)\n"
        "    return takesShape(s)\n"
    )
    fn = _fn(ir_program, 'main')
    # Doesn't crash building the IR at all -- that's the actual bug
    # being guarded against here; build_ir_program's own verify_
    # program call (see ir/program_builder.py) already re-confirms
    # the result is structurally well-formed.
    assert fn is not None


def test_append_into_sum_typed_slice_uses_shapes_own_element_width():
    """append(shapes, Circle(5)) into a []Shape -- IRSliceGrow's own
    element_width must be Shape's width (12), not Circle's (4): the
    one position checked, not fixed, since it turned out to already
    work correctly (the target element type comes from the slice's
    own declared type, known locally, not from a function signature
    lookup like the argument-passing case needed)."""
    ir_program = _build(
        _SHAPE_DECLS +
        "def int main():\n"
        "    []Shape shapes\n"
        "    shapes = append(shapes, Circle(5))\n"
        "    return 0\n"
    )
    fn = _fn(ir_program, 'main')
    grows = [instr for instr in fn.body if isinstance(instr, IRSliceGrow)]
    assert len(grows) == 1
    assert grows[0].element_width == 12


def test_array_literal_of_shapes_widens_each_element():
    """[Circle(5), Square(9)] into a [2]Shape -- each element widens
    independently through _ir_write_composite_value_into's own SUM
    check, which already covers array-literal elements (no separate
    fix needed for THIS case specifically) since _ir_write_array_
    literal_into recurses through that same dispatcher for each of its
    own composite elements. The already-sum-typed sibling case right
    below DID need its own fix, though -- see that test's own
    docstring."""
    ir_program = _build(
        _SHAPE_DECLS +
        "def int main():\n"
        "    [2]Shape shapes = [Circle(5), Square(9)]\n"
        "    return 0\n"
    )
    fn = _fn(ir_program, 'main')
    tag_writes = _discriminant_writes(fn.body)
    # Subset, not exact equality -- Circle's own radius=5 field write
    # is ALSO an IRStore of a plain INT IRConst, coincidentally
    # matching this same broad filter (see _discriminant_writes' own
    # docstring); {0, 1} both appearing somewhere is what actually
    # matters here.
    assert {0, 1} <= {w.value.value for w in tag_writes}


def test_array_literal_of_already_sum_typed_elements_does_not_crash():
    """[a, b] into a [2]Shape, a and b ALREADY Shape-typed variables --
    NOT widening at all (both sides already the same type). The actual
    bug: _ir_write_composite_value_into's own SUM check originally
    fired unconditionally on value_type.kind == SUM alone, without
    checking whether value_expr itself was genuinely narrower --
    crashing here with "None is not in list" (source_struct_type.
    struct_name is None for an already-sum-typed source, not a real
    struct name to look up). Found only by testing this exact
    combination -- element widening (tested above) and an already-
    typed source (tested elsewhere for VarDecl/Assign) had each been
    tested separately, but not together, inside an array literal."""
    ir_program = _build(
        _SHAPE_DECLS +
        "def int main():\n"
        "    Shape a = Circle(5)\n"
        "    Shape b = Square(9)\n"
        "    [2]Shape shapes = [a, b]\n"
        "    return 0\n"
    )
    fn = _fn(ir_program, 'main')
    assert fn is not None


def test_index_assign_into_sum_typed_array_element_writes_discriminant():
    """shapes[0] = Square(3) -- the actual bug found: element_type.
    kind not in (SLICE, STRUCT) is true for SUM too (SUM is neither),
    so this fell into the scalar IndexAssign path, which tried to
    gen_expr_ir a struct-literal Call -- not even a wrong-width bug
    like the others, straight to IRError with no real IR produced at
    all."""
    ir_program = _build(
        _SHAPE_DECLS +
        "def int main():\n"
        "    [2]Shape shapes = [Circle(5), Square(9)]\n"
        "    shapes[0] = Square(3)\n"
        "    return 0\n"
    )
    fn = _fn(ir_program, 'main')
    tag_writes = [w for w in _discriminant_writes(fn.body) if w.value.value == 1]
    # Two Square tags now: the array literal's own second element,
    # and the IndexAssign's -- just confirming the IndexAssign's own
    # write happened at all (it wouldn't have, before the fix) rather
    # than trying to disentangle which of the two is which.
    assert len(tag_writes) == 2


def test_widening_argument_too_large_for_a_stack_slot_mallocs_instead():
    """A sum type wide enough to cross is_heap_allocated's own
    threshold gets malloc'd fresh at the call site, exactly like an
    oversized struct argument already does -- _reserve_argument_temp
    skips reservation (no "argument_temp" slot at all, unlike the
    small-Shape case above), and _ir_materialize_sum_type_value's own
    malloc branch sizes the allocation for THE SUM TYPE (tag + Big's
    own [5000]int -- 4 + 20000 = 20004), not Circle's own tiny size."""
    ir_program = _build(
        "type Circle struct:\n"
        "    int radius\n"
        "\n"
        "type Big struct:\n"
        "    [5000]int data\n"
        "\n"
        "type Shape is Circle | Big\n"
        "\n"
        "def int takesShape(Shape s):\n"
        "    return 0\n"
        "\n"
        "def int main():\n"
        "    return takesShape(Circle(5))\n"
    )
    fn = _fn(ir_program, 'main')
    assert 'argument_temp' not in fn.slot_labels.values()
    mallocs = [
        instr for instr in fn.body
        if hasattr(instr, 'name') and getattr(instr, 'name', None) == 'malloc'
    ]
    assert len(mallocs) == 1
    assert mallocs[0].args[0].value == 20004
