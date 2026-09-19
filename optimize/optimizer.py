"""optimize(ir_program) is the one entrypoint this whole package
exposes -- see generate_asm's own body (codegen/codegen.py) for the
seam it runs in, right between build_ir_program and CodeGenerator().
generate: building needs nothing from lowering, and optimizing needs
nothing from either -- just the IRProgram itself.

A plain function, not a class: there's no state a composing entrypoint
would need to hold across the passes it runs (each pass here is
itself a plain, stateless function taking one IRFunction -- see
constant_folding.py's own fold_constants and identity_reduction.py's
own reduce_identities).

Mutates every function in ir_program.functions in place and returns
the same object, rather than building a new IRProgram -- matching how
each individual pass already works (see fold_constants' own
docstring): nothing here needs a fresh copy to reason about, and
returning the identical object rather than a new one costs nothing
while keeping `ir_program = optimize(ir_program)` and a bare
`optimize(ir_program)` equally correct for a caller to write.

The order fold_constants/reduce_identities run in genuinely doesn't
matter: neither one propagates a computed value INTO a later
instruction's own operand -- `int y = (2 + 3) + 1` builds as two
separate IRBinOps, the second one's own left operand a Temp
referencing the first one's own dst, never the constant 5 itself,
regardless of whether the first has already been folded by the time
the second runs. The two passes only ever interact within a SINGLE
instruction, where an op can qualify for both (`5 + 0` is both fully
constant and has an identity operand) -- and there, both independently
compute the identical correct replacement, so whichever runs first
simply leaves nothing for the second to do. Real constant PROPAGATION
across instructions would be a genuinely different, more involved
pass than either of these."""

from ir.ir import IRProgram
from optimize.constant_folding import fold_constants
from optimize.identity_reduction import reduce_identities


def optimize(ir_program: IRProgram) -> IRProgram:
    for ir_fn in ir_program.functions:
        fold_constants(ir_fn)
        reduce_identities(ir_fn)
    return ir_program
