"""Builds the whole program's own IR: validates program's two
registries, builds one IRFunction per function via IRFunctionBuilder,
and returns a self-contained IRProgram (struct_registry/type_alias_
registry/ids included).

A plain function, not a class -- no per-build state to hold across
multiple methods, unlike IRFunctionBuilder.

Separate from CodeGenerator.generate() so optimize() can run between
them (see compile_to_asm)."""

from codegen.id_allocator import IdAllocator
from ir.errors import IRError
from ir.ir import IRProgram
from ir.builder import IRFunctionBuilder
from parser import Program


def build_ir_program(program: Program) -> IRProgram:
    """Raises IRError (rather than a bare AttributeError) if `program`
    hasn't been through semantic.analyze() yet -- struct_registry/
    type_alias_registry are stamped on there, not declared on the
    dataclass itself."""
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
    ir_program = IRProgram(
        struct_registry=program.struct_registry,
        type_alias_registry=program.type_alias_registry,
        ids=IdAllocator(),
    )
    ir_program.functions = [IRFunctionBuilder(ir_program).gen_function_ir(fn) for fn in program.functions]
    return ir_program
