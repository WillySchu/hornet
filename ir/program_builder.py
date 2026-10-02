"""Build and verify an IRProgram from an analyzed Program."""

import os
from typing import Optional

from elaborate import elaborate
from escape_analysis import compute_escape_summaries
from ir.errors import IRError
from ir.id_allocator import IdAllocator
from ir.ir import IRProgram
from ir.builder import IRFunctionBuilder
from ir.division_checks import insert_division_checks
from ir.null_checks import expand_null_checks
from ir.typed_builder import NotYetPorted, TypedFunctionBuilder
from ir.verify import verify_program
from parser import Program


def typed_ir_enabled() -> bool:
    """Whether to build IR from the typed tree where that builder supports the function
    (HORNET_TYPED_IR=1), while it is being completed."""
    return os.environ.get('HORNET_TYPED_IR') == '1'


def build_ir_program(program: Program, typed: Optional[bool] = None) -> IRProgram:
    """IRError if `program` hasn't been through semantic.analyze(). With `typed` (default: the
    HORNET_TYPED_IR setting), each function is built from the typed tree when the typed builder
    supports everything it uses; `ir_program.typed_functions` names those."""
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
    ir_program.typed_functions = []
    if typed if typed is not None else typed_ir_enabled():
        typed_program = elaborate(program)
        typedescs = IRFunctionBuilder(ir_program)
        ir_program.functions = []
        for fn, typed_fn in zip(program.functions, typed_program.functions):
            try:
                ir_fn = TypedFunctionBuilder(ir_program, typedescs).build(typed_fn, fn)
                ir_program.typed_functions.append(fn.name)
            except NotYetPorted:
                ir_fn = IRFunctionBuilder(ir_program).gen_function_ir(fn)
            ir_program.functions.append(ir_fn)
    else:
        ir_program.functions = [IRFunctionBuilder(ir_program).gen_function_ir(fn) for fn in program.functions]
    for ir_fn in ir_program.functions:
        expand_null_checks(ir_fn, ir_program)
        insert_division_checks(ir_fn, ir_program)
    verify_program(ir_program)
    return ir_program
