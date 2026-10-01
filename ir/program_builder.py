"""Build and verify an IRProgram from an analyzed Program."""

from escape_analysis import compute_escape_summaries
from ir.errors import IRError
from ir.id_allocator import IdAllocator
from ir.ir import IRProgram
from ir.builder import IRFunctionBuilder
from ir.division_checks import insert_division_checks
from ir.null_checks import expand_null_checks
from ir.verify import verify_program
from parser import Program


def build_ir_program(program: Program) -> IRProgram:
    """IRError if `program` hasn't been through semantic.analyze()."""
    if not hasattr(program, 'struct_registry'):
        raise IRError(
            "Program has no struct registry -- semantic.analyze() "
            "must run before codegen (see compile_to_asm)"
        )
    if not hasattr(program, 'type_alias_registry'):
        raise IRError(
            "Program has no type alias registry -- semantic.analyze() "
            "must run before codegen (see compile_to_asm)"
        )
    if not hasattr(program, 'sum_type_registry'):
        raise IRError(
            "Program has no sum type registry -- semantic.analyze() "
            "must run before codegen (see compile_to_asm)"
        )
    if not hasattr(program, 'function_registry'):
        raise IRError(
            "Program has no function registry -- semantic.analyze() "
            "must run before codegen (see compile_to_asm)"
        )
    ir_program = IRProgram(
        struct_registry=program.struct_registry,
        type_alias_registry=program.type_alias_registry,
        sum_type_registry=program.sum_type_registry,
        function_registry=program.function_registry,
        intrinsic_original_names=getattr(program, 'intrinsic_original_names', {}),
        ids=IdAllocator(),
    )
    ir_program.escape_summaries = compute_escape_summaries(program.functions, program.struct_registry)
    ir_program.functions = [IRFunctionBuilder(ir_program).gen_function_ir(fn) for fn in program.functions]
    for ir_fn in ir_program.functions:
        expand_null_checks(ir_fn, ir_program)
        insert_division_checks(ir_fn, ir_program)
    verify_program(ir_program)
    return ir_program
