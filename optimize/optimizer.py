"""IR optimization entry point: per-function local passes, then re-verify."""

from ir.ir import IRProgram
from ir.verify import verify_program
from optimize.constant_folding import fold_constants
from optimize.identity_reduction import reduce_identities


def optimize(ir_program: IRProgram) -> IRProgram:
    for ir_fn in ir_program.functions:
        fold_constants(ir_fn)
        reduce_identities(ir_fn)
    verify_program(ir_program)  # a pass may break an invariant
    return ir_program
