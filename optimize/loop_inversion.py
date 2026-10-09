"""Loop inversion: a loop's jump back to its test becomes a copy of the test, so each turn ends in
one conditional branch instead of a jump to one. The test stays where it was, guarding the way in.

Run once, when the other passes have settled (see optimize): a copy propagated into before the
loop's step has been coalesced would keep the step's temporary alive. And run after any pass that
goes by a block having one way in, which a loop's body no longer has.
"""

from dataclasses import replace

from ir.cfg import build_blocks, reads, replace_reads, reverse_postorder, uses, writes
from ir.ir import IRBranch, IRCall, IRFunction, IRJump

# A test of more instructions than this is left where it is (few come near it).
MAX_TEST = 16


def invert_loops(ir_fn: IRFunction, ids) -> bool:
    """Invert ir_fn.body's loops in place, taking fresh Temps from `ids`; whether any was."""
    blocks = build_blocks(ir_fn.body)
    position = {b: n for n, b in enumerate(reverse_postorder(blocks))}
    read_at, written_at = uses(ir_fn.body), _writes(ir_fn.body)
    body, inverted = [], False
    for i, block in enumerate(blocks):
        test = blocks[block.successors[0]] if isinstance(block.instructions[-1], IRJump) else None
        # A jump is a loop's way back if it goes to a block that comes no later (control flow is
        # structured). Were it not one, a copy of the block it goes to would still do what it did.
        if (test is None or test is block or not isinstance(test.instructions[-1], IRBranch)
                or len(test.instructions) - 2 > MAX_TEST or i not in position
                or position[blocks.index(test)] > position[i]):
            body += block.instructions
            continue
        body += block.instructions[:-1] + _copy(test, _private(test, ir_fn, read_at, written_at), ids)
        inverted = True
    ir_fn.body = body
    return inverted


def _writes(body: list) -> dict:
    """temp.id -> indices of the instructions in `body` that write it."""
    out: dict = {}
    for i, instr in enumerate(body):
        for t in writes(instr):
            out.setdefault(t.id, []).append(i)
    return out


def _private(test, ir_fn: IRFunction, read_at: dict, written_at: dict) -> set:
    """Ids of the temps that are `test`'s alone: written there before they are read there, touched
    nowhere else, and not a variable's (whose slot may be read instead)."""
    inside = range(test.start, test.start + len(test.instructions))
    private, exposed = set(), set()
    for instr in test.instructions:
        exposed |= {t.id for t in reads(instr)} - private
        for t in writes(instr):
            if (t.id not in exposed and t.id not in ir_fn.temp_homes
                    and all(n in inside for n in read_at.get(t.id, []) + written_at[t.id])):
                private.add(t.id)
    return private


def _copy(test, private: set, ids) -> list:
    """`test`'s instructions again, without its label. Its private temps get new names: a backend
    makes `cmp; jcc` of a comparison only where the branch after it is the one reader it has."""
    names, out = {}, []
    for instr in test.instructions[1:]:
        instr = replace_reads(instr, names)
        instr = replace(instr, args=list(instr.args)) if isinstance(instr, IRCall) else replace(instr)
        for t in writes(instr):
            if t.id in private:
                names[t.id] = ids.new_temp(t.type)
                instr = replace(instr, dst=names[t.id])
        out.append(instr)
    return out
