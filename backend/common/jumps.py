"""Jump cleanups that hold for any target, over its assembly instructions via a JumpSyntax:

- `bCOND A; jump B; A:` -> `bNOTCOND B; A:`
- `jump L; L:` -> `L:`
"""

from typing import Optional, Protocol


class JumpSyntax(Protocol):
    def label(self, instr) -> Optional[str]: ...
    def jump(self, instr) -> Optional[str]: ...  # target of an unconditional jump
    def branch(self, instr) -> Optional[tuple]: ...  # (condition, target) of a conditional jump
    def make_branch(self, condition, target: str): ...
    def invert(self, condition): ...


def invert_branches(instrs: list, syntax: JumpSyntax) -> list:
    out = []
    i = 0
    while i < len(instrs):
        branch = syntax.branch(instrs[i])
        if branch and i + 2 < len(instrs):
            over = syntax.jump(instrs[i + 1])
            if over is not None and syntax.label(instrs[i + 2]) == branch[1]:
                out.append(syntax.make_branch(syntax.invert(branch[0]), over))
                i += 2
                continue
        out.append(instrs[i])
        i += 1
    return out


def drop_jumps_to_next(instrs: list, syntax: JumpSyntax) -> list:
    return [
        instr for i, instr in enumerate(instrs)
        if not (syntax.jump(instr) is not None and i + 1 < len(instrs)
                and syntax.label(instrs[i + 1]) == syntax.jump(instr))
    ]
