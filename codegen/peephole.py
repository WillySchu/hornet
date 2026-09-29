"""Assembly-level cleanups on one function's finished instruction list.

- `jCC A; jmp B; A:` -> `jNCC B; A:`
- `jmp L; L:` -> `L:`
- `movq a, b; movq b, a` -> `movq a, b`
- `movq r, r` -> removed

32-bit moves are never removed: `movl` clears the upper half of its destination.
"""

from codegen.assembly_ast import Imm, Instruction, JCC, Ja, Jae, Je, Jg, Jle, Jmp, Jne, Label, MovQ

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


def _invert_branches(instrs: list) -> list:
    out = []
    i = 0
    while i < len(instrs):
        cond = _cond(instrs[i])
        if (cond and i + 2 < len(instrs) and isinstance(instrs[i + 1], Jmp)
                and isinstance(instrs[i + 2], Label) and instrs[i + 2].name == cond[1]):
            out.append(JCC(INVERSE_CC[cond[0]], instrs[i + 1].target))
            i += 2
            continue
        out.append(instrs[i])
        i += 1
    return out


def _drop_jumps_to_next(instrs: list) -> list:
    return [
        instr for i, instr in enumerate(instrs)
        if not (isinstance(instr, Jmp) and i + 1 < len(instrs)
                and isinstance(instrs[i + 1], Label) and instrs[i + 1].name == instr.target)
    ]


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
        new = _drop_redundant_moves(_drop_jumps_to_next(_invert_branches(instrs)))
        if new == instrs:
            return new
        instrs = new
