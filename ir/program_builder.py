"""Builds the whole program's own real IR: validates program's two
registries, builds one IRFunction per function (via a fresh
IRFunctionBuilder each -- see its own module docstring), and returns
the result as a genuinely self-contained IRProgram (struct_registry/
type_alias_registry/ids included -- see IRProgram's own docstring for
why that matters).

A plain function, not a class: unlike IRFunctionBuilder, which is
inherently a class because it holds real per-function state (scopes,
loop_labels) across its own several methods, there is no equivalent
per-build state here to hold -- everything this needs is either an
argument or lives on the IRProgram it constructs and returns. Making
it a class anyway, just for symmetry with IRFunctionBuilder, would be
manufacturing state that doesn't exist.

Takes no CodeGenerator at all, unlike this arc's own earlier shape:
building IR needs nothing lowering-specific (frame layout, register
assignment -- see codegen.py's own lower_function), so nothing here
should have needed a CodeGenerator instance just to run. generate()
now takes an already-built IRProgram directly rather than building one
itself, so build_ir_program and generate() are genuinely separate
steps a caller can run independently -- with a real seam, right where
this function returns, for whatever IR-to-IR transform (optimization
passes, eventually) wants to sit between them. See compile_to_asm's
own body for the shape this now takes."""

from codegen.errors import CodegenError
from codegen.id_allocator import IdAllocator
from ir.ir import IRProgram
from ir.builder import IRFunctionBuilder
from parser import Program


def build_ir_program(program: Program) -> IRProgram:
    """The two hasattr checks match type_of's own "has no resolved
    type" defensive check one level up: fail with a clear, actionable
    CodegenError right here rather than a bare AttributeError from
    whatever the first registry lookup happens to be, since Program.
    struct_registry/type_alias_registry are stamped on by semantic.
    analyze(), not fields the dataclass itself declares -- an AST that
    skipped analyze() entirely simply won't have them."""
    if not hasattr(program, 'struct_registry'):
        raise CodegenError(
            "Program has no struct registry -- semantic.analyze() "
            "must run before codegen (see compile_to_asm)"
        )
    if not hasattr(program, 'type_alias_registry'):
        raise CodegenError(
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
