"""Tests for IRFunction/IRProgram (see ir/ir.py) -- the thin
aggregation objects introduced by this compiler's own IR/codegen
decoupling work, build_ir_program's own construction of a complete
IRProgram, and generate()'s own population of self.ir_program from
whatever IRProgram it's given.

No consumer of self.ir_program exists yet (see IRProgram's own
docstring for why it's built anyway) -- these tests exist specifically
because nothing else in the suite inspects it directly: every other
test only checks a compiled program's own runtime behavior, which
would stay green even if self.ir_program were silently wrong or never
populated at all.
"""

import tempfile
from pathlib import Path
from typing import get_args

import parser
import semantic
from lexer import lex
from backend.x86_64.codegen import CodeGenerator
from ir.ir import IRFunction, IRInstr, IRProgram
from ir.program_builder import build_ir_program
from ir.typed_builder import link_name


def _parse_and_analyze(source: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "program.lang"
        src_path.write_text(source)
        tokens = lex(str(src_path))
        ast = parser.Parser(tokens).parse_program()
        ast = semantic.analyze(ast)
        return ast


_IR_INSTR_TYPES = get_args(IRInstr)


def test_generate_populates_ir_program():
    ast = _parse_and_analyze(
        "def int add(int a, int b):\n"
        "    return a + b\n"
        "\n"
        "def int main():\n"
        "    return add(1, 2)\n"
    )
    ir_program = build_ir_program(ast)
    gen = CodeGenerator()
    gen.generate(ir_program)
    assert isinstance(gen.ir_program, IRProgram)
    assert [fn.name for fn in gen.ir_program.functions] == [link_name('add'), 'main']
    assert all(isinstance(fn, IRFunction) for fn in gen.ir_program.functions)


def test_ir_program_is_none_before_generate_runs():
    """Declared defensively in __init__ (see its own comment) --
    confirmed here so a bug that reads this before generate() ever
    ran fails on an assertion, not silently."""
    gen = CodeGenerator()
    assert gen.ir_program is None


def test_ir_function_body_is_nonempty_for_a_nontrivial_function():
    ast = _parse_and_analyze(
        "def int main():\n"
        "    int x = 1\n"
        "    int y = 2\n"
        "    return x + y\n"
    )
    ir_program = build_ir_program(ast)
    gen = CodeGenerator()
    gen.generate(ir_program)
    main_fn = gen.ir_program.functions[0]
    assert main_fn.name == 'main'
    assert len(main_fn.body) > 0


def test_ir_function_body_contains_only_real_ir_ops():
    """A regression this exists specifically to catch: an old-style
    Instruction (see assembly_ast.py) leaking into body would mean
    something fell back to non-IR construction without raising --
    exactly the failure mode IRRaw's own removal (see ir.py's own
    module docstring) was supposed to make impossible. Every element
    of body must be one of IRInstr's own listed types, no exceptions."""
    ast = _parse_and_analyze(
        "type Point struct:\n"
        "    int x\n"
        "    int y\n"
        "\n"
        "def int sumSlice([]int s):\n"
        "    return s[0] + s[1]\n"
        "\n"
        "def int main():\n"
        "    Point p = Point(1, 2)\n"
        "    []int s = [1, 2, 3]\n"
        "    int total = p.x + sumSlice(s)\n"
        "    if total > 0:\n"
        "        total = total + 1\n"
        "    else:\n"
        "        total = total - 1\n"
        "    return total\n"
    )
    ir_program = build_ir_program(ast)
    gen = CodeGenerator()
    gen.generate(ir_program)
    for fn in gen.ir_program.functions:
        for instr in fn.body:
            assert isinstance(instr, _IR_INSTR_TYPES), (
                f"{fn.name}'s own body contains a non-IRInstr op: {instr!r}"
            )


def test_ir_program_shares_string_and_typedesc_data_with_asm_program():
    """string_literals/type_descriptors are the same underlying data
    AsmProgram itself carries (see IRProgram's own docstring) -- not a
    separate copy that could silently drift from what actually gets
    emitted."""
    ast = _parse_and_analyze(
        "def int main():\n"
        "    str s = 'hello'\n"
        "    print(s)\n"
        "    return 0\n"
    )
    ir_program = build_ir_program(ast)
    gen = CodeGenerator()
    asm_program = gen.generate(ir_program)
    assert gen.ir_program.string_literals == asm_program.string_literals
    assert gen.ir_program.type_descriptors == asm_program.type_descriptors
    assert len(gen.ir_program.string_literals) > 0


def test_ir_program_carries_the_layout_registries():
    """IRProgram keeps the struct and sum-type registries (layout for the backends), taken from the
    typed program; aliases are resolved away before IR, so there's no alias registry."""
    program = _parse_and_analyze(
        "type Point struct:\n"
        "    int x\n"
        "    int y\n"
        "\n"
        "type Coord = Point\n"
        "\n"
        "def int main():\n"
        "    Coord p = Point(1, 2)\n"
        "    return p.x\n"
    )
    ir_program = build_ir_program(program)
    assert ir_program.struct_registry is program.structs and ir_program.sum_type_registry is program.sum_types
    assert 'Point' in ir_program.struct_registry and not hasattr(ir_program, 'type_alias_registry')


def test_ir_program_carries_its_own_ids():
    """ids (an IdAllocator) is the third piece of IRProgram's own
    self-containment, alongside the two registries above -- see its
    own docstring for why building needs a live one, not just a
    default placeholder."""
    ast = _parse_and_analyze("def int main():\n    return 1\n")
    ir_program = build_ir_program(ast)
    assert ir_program.ids is not None


def test_lowering_does_not_modify_the_ir():
    """Spill and outgoing-argument slots belong to the backend, not the IRFunction."""
    import copy
    from pathlib import Path
    from backend.x86_64.codegen import CodeGenerator
    from ir.program_builder import build_ir_program
    from modules import discover_modules
    from optimize.optimizer import optimize
    from semantic import analyze
    path = Path(__file__).resolve().parents[3] / 'benchmarks' / 'programs' / 'register_pressure.ht'
    entry, modules = discover_modules(str(path))
    program = analyze(entry, modules)
    ir_program = optimize(build_ir_program(program))
    before = copy.deepcopy([(f.slot_widths, f.slot_labels, f.temp_homes, f.body) for f in ir_program.functions])
    first = CodeGenerator().generate(ir_program)
    assert [(f.slot_widths, f.slot_labels, f.temp_homes, f.body) for f in ir_program.functions] == before
    assert CodeGenerator().generate(ir_program) == first
