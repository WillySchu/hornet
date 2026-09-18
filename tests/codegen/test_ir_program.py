"""Tests for IRFunction/IRProgram (see ir/ir.py) -- the thin
aggregation objects introduced by this compiler's own IR/codegen
decoupling work, and generate()'s own population of self.ir_program
alongside the AsmProgram it already returns.

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
from codegen.codegen import CodeGenerator
from ir.ir import IRFunction, IRInstr, IRProgram


def _parse_and_analyze(source: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "program.lang"
        src_path.write_text(source)
        tokens = lex(str(src_path))
        ast = parser.Parser(tokens).parse_program()
        semantic.analyze(ast)
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
    gen = CodeGenerator()
    gen.generate(ast)
    assert isinstance(gen.ir_program, IRProgram)
    assert [fn.name for fn in gen.ir_program.functions] == ['add', 'main']
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
    gen = CodeGenerator()
    gen.generate(ast)
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
        "struct Point:\n"
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
    gen = CodeGenerator()
    gen.generate(ast)
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
    gen = CodeGenerator()
    asm_program = gen.generate(ast)
    assert gen.ir_program.string_literals == asm_program.string_literals
    assert gen.ir_program.type_descriptors == asm_program.type_descriptors
    assert len(gen.ir_program.string_literals) > 0


def test_gen_function_wrapper_matches_generate_output():
    """gen_function (the thin gen_function_ir + lower_function
    wrapper) must still produce an AsmFunction identical to what
    generate() itself produces for the same function -- confirmed
    directly since generate() no longer calls gen_function at all
    (see its own updated body)."""
    ast = _parse_and_analyze(
        "def int main():\n"
        "    int x = 5\n"
        "    return x * 2\n"
    )
    gen_a = CodeGenerator()
    gen_a.struct_registry = ast.struct_registry
    gen_a.type_alias_registry = ast.type_alias_registry
    via_wrapper = gen_a.gen_function(ast.functions[0])

    gen_b = CodeGenerator()
    asm_program = gen_b.generate(ast)
    via_generate = asm_program.functions[0]

    assert via_wrapper == via_generate
