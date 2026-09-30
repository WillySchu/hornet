"""Assembly-level cleanups on one function's finished instruction list: the common jump rules
(backend.common.jumps), and

- `movq a, b; movq b, a` -> `movq a, b`
- `movq r, r` -> removed

32-bit moves are never removed: `movl` clears the upper half of its destination.
"""

from backend.common.jumps import drop_jumps_to_next, invert_branches
from backend.x86_64.assembly_ast import Imm, Instruction, JCC, Ja, Jae, Je, Jg, Jle, Jmp, Jne, Label, MovQ

INVERSE_CC = {
    'e': 'ne', 'ne': 'e',
    'l': 'ge', 'ge': 'l',
    'le': 'g', 'g': 'le',
    'b': 'ae', 'ae': 'b',
    'be': 'a', 'a': 'be',
}

_FIXED_CC = {Je: 'e', Jne: 'ne', Jae: 'ae', Ja: 'a', Jle: 'le', Jg: 'g'}


def _cond(instr: Instruction):
    """(cc, target) of a conditional jump, else None."""
    if isinstance(instr, JCC):
        return instr.cc, instr.target
    cc = _FIXED_CC.get(type(instr))
    return (cc, instr.target) if cc else None


class _Jumps:
    """x86 spelling of jumps and labels for backend.common.jumps."""

    def label(self, instr):
        return instr.name if isinstance(instr, Label) else None

    def jump(self, instr):
        return instr.target if isinstance(instr, Jmp) else None

    def branch(self, instr):
        return _cond(instr)

    def make_branch(self, cc, target):
        return JCC(cc, target)

    def invert(self, cc):
        return INVERSE_CC[cc]


_JUMPS = _Jumps()


def _drop_redundant_moves(instrs: list) -> list:
    out = []
    for instr in instrs:
        if isinstance(instr, MovQ) and not isinstance(instr.src, Imm):
            if instr.src == instr.dst:
                continue
            prev = out[-1] if out else None
            if isinstance(prev, MovQ) and prev.src == instr.dst and prev.dst == instr.src:
                continue
        out.append(instr)
    return out


def optimize_asm(instrs: list) -> list:
    """Apply every rule until nothing changes."""
    while True:
        new = _drop_redundant_moves(drop_jumps_to_next(invert_branches(instrs, _JUMPS), _JUMPS))
        if new == instrs:
            return new
        instrs = new
