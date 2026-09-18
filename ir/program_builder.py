"""Builds the whole program's own real IR: validates program's two
registries, builds one IRFunction per function (via a fresh
IRFunctionBuilder each -- see its own module docstring), and collects
the results into an IRProgram -- struct_registry/type_alias_registry
included, so the object it returns is a genuinely self-contained
artifact (see IRProgram's own docstring for why that matters). generate()
used to do this inline before this arc pulled gen_function_ir out as
its own step; this is the identical move one level up, so generate()
itself can now just build, then lower.

`host` (the CodeGenerator instance) is still needed here, and still
needs its OWN struct_registry/type_alias_registry set on it (not just
read off the IRProgram this returns): lowering reads them from host,
not from this object (see arrays_slices_lowering.py/ir_lowering.py's
own `self.host.X` references) -- one program, one host, across both
building and lowering."""

from codegen.errors import CodegenError
from ir.ir import IRProgram
from ir.builder import IRFunctionBuilder
from parser import Program


class IRProgramBuilder:
    def __init__(self, host):
        self.host = host

    def build(self, program: Program) -> IRProgram:
        """The two hasattr checks match type_of's own "has no resolved
        type" defensive check one level up: fail with a clear,
        actionable CodegenError right here rather than a bare
        AttributeError from whatever the first registry lookup happens
        to be, since Program.struct_registry/type_alias_registry are
        stamped on by semantic.analyze(), not fields the dataclass
        itself declares -- an AST that skipped analyze() entirely
        simply won't have them."""
        if not hasattr(program, 'struct_registry'):
            raise CodegenError(
                "Program has no struct registry -- semantic.analyze() "
                "must run before codegen (see compile_to_asm)"
            )
        self.host.struct_registry = program.struct_registry
        if not hasattr(program, 'type_alias_registry'):
            raise CodegenError(
                "Program has no type alias registry -- semantic.analyze() "
                "must run before codegen (see compile_to_asm)"
            )
        self.host.type_alias_registry = program.type_alias_registry
        ir_functions = [IRFunctionBuilder(self.host).gen_function_ir(fn) for fn in program.functions]
        return IRProgram(
            functions=ir_functions,
            string_literals=self.host.string_literals,
            type_descriptors=self.host.type_descriptors,
            struct_registry=self.host.struct_registry,
            type_alias_registry=self.host.type_alias_registry,
        )
