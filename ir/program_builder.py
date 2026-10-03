"""Build and verify an IRProgram from an analyzed Program."""

from escape_analysis import compute_escape_summaries
from ir.errors import IRError
from ir.id_allocator import IdAllocator
from ir.ir import IRProgram
from ir.division_checks import insert_division_checks
from ir.null_checks import expand_null_checks
from ir.typed_builder import NotYetPorted, TypedFunctionBuilder
from ir.verify import verify_program
import typed_ast as typed


def build_ir_program(program: typed.Program) -> IRProgram:
    """IR for a typed program (what semantic.analyze() returns); nothing from the parser's tree."""
    if not isinstance(program, typed.Program):
        raise IRError(f"build_ir_program takes semantic.analyze()'s typed program, not a {type(program).__name__}")
    ir_program = IRProgram(struct_registry=program.structs, sum_type_registry=program.sum_types,
                           enum_registry=program.enums, ids=IdAllocator())
    ir_program.escape_summaries = compute_escape_summaries(program.functions, program.structs)
    ir_program.functions = []
    for typed_fn in program.functions:
        try:
            ir_program.functions.append(TypedFunctionBuilder(ir_program).build(typed_fn))
        except NotYetPorted as e:
            raise IRError(f"No IR for {typed_fn.name}: {e}") from e
    for ir_fn in ir_program.functions:
        expand_null_checks(ir_fn, ir_program)
        insert_division_checks(ir_fn, ir_program)
    verify_program(ir_program)
    return ir_program
