"""IR optimization entry point: per-function passes repeated until nothing changes, loops inverted and
the passes repeated, then re-verify."""

from ir.ir import IRFunction, IRLocalAddress, IRProgram
from ir.verify import verify_program
from optimize.address_folding import fold_addresses
from optimize.branch_simplification import simplify_branches
from optimize.constant_folding import fold_constants
from optimize.copy_coalescing import coalesce_copies
from optimize.copy_propagation import propagate_copies
from optimize.dead_code import remove_dead_code
from optimize.identity_reduction import reduce_identities
from optimize.jump_threading import merge_blocks, thread_jumps
from optimize.loop_inversion import invert_loops
from optimize.strength_reduction import reduce_strength

MAX_ROUNDS = 10


def pinned_temps(ir_fn: IRFunction) -> set:
    """Temps whose home slot has its address taken: they can change through memory."""
    addressed = {instr.slot for instr in ir_fn.body if isinstance(instr, IRLocalAddress)}
    return {tid for tid, slot in ir_fn.temp_homes.items() if slot in addressed}


def optimize_function(ir_fn: IRFunction) -> None:
    pinned = pinned_temps(ir_fn)
    for _ in range(MAX_ROUNDS):
        before = list(ir_fn.body)
        simplify_branches(ir_fn)
        thread_jumps(ir_fn)
        merge_blocks(ir_fn)
        propagate_copies(ir_fn, pinned)
        fold_constants(ir_fn)
        reduce_identities(ir_fn)
        reduce_strength(ir_fn)
        coalesce_copies(ir_fn, pinned)
        fold_addresses(ir_fn, pinned)
        remove_dead_code(ir_fn, pinned)
        if ir_fn.body == before:
            break


def optimize(ir_program: IRProgram) -> IRProgram:
    for ir_fn in ir_program.functions:
        optimize_function(ir_fn)
        if invert_loops(ir_fn, ir_program.ids):  # once, on settled code: see its module
            optimize_function(ir_fn)
    verify_program(ir_program)  # a pass may break an invariant
    return ir_program
