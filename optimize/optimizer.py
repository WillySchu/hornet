"""optimize(ir_program) is the one entrypoint this whole package
exposes -- see generate_asm's own body (codegen/codegen.py) for the
seam it runs in, right between build_ir_program and CodeGenerator().
generate: building needs nothing from lowering, and now, symmetrically,
optimizing needs nothing from either -- just the IRProgram itself.

A plain function, not a class: there's no state a composing entrypoint
would need to hold across the passes it runs (each pass here is
itself a plain, stateless function taking one IRFunction -- see
constant_folding.py's own fold_constants), the identical reasoning
that turned ir.program_builder's own IRProgramBuilder into a plain
build_ir_program function once IT lost its own reason to be a class.

Mutates every function in ir_program.functions in place and returns
the same object, rather than building a new IRProgram -- matching how
each individual pass already works (see fold_constants' own
docstring): nothing here needs a fresh copy to reason about, and
returning the identical object rather than a new one costs nothing
while keeping `ir_program = optimize(ir_program)` and a bare
`optimize(ir_program)` equally correct for a caller to write.

One pass today. Adding a second (identity reduction, say) means
importing its own module's own entrypoint here and calling it
alongside fold_constants for each function -- in whatever order makes
each pass see the other's own output where that matters (constant
folding before identity reduction lets `x + (2 + 3)` reduce in one
optimize() call rather than needing two), not necessarily the order
listed."""

from ir.ir import IRProgram
from optimize.constant_folding import fold_constants


def optimize(ir_program: IRProgram) -> IRProgram:
    for ir_fn in ir_program.functions:
        fold_constants(ir_fn)
    return ir_program
