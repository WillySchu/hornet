"""AArch64 assembly: one generic instruction type with typed operands."""

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class Reg:
    name: str  # x0..x30, w0..w30, sp, xzr, wzr

    @property
    def x(self) -> 'Reg':
        return Reg('x' + self.name[1:]) if self.name[0] in 'wx' and self.name[1:].isdigit() else self

    @property
    def w(self) -> 'Reg':
        return Reg('w' + self.name[1:]) if self.name[0] in 'wx' and self.name[1:].isdigit() else self


@dataclass(frozen=True)
class Imm:
    value: int


@dataclass(frozen=True)
class Mem:
    """[base, #offset]; `pre`/`post` for pre- and post-indexed forms."""
    base: Reg
    offset: int = 0
    mode: str = ''  # '', 'pre', or 'post'


@dataclass(frozen=True)
class FrameSlot:
    """A frame slot's memory; resolved to a Mem after frame layout."""
    slot: object
    extra: int = 0


@dataclass(frozen=True)
class LabelRef:
    name: str


@dataclass(frozen=True)
class SymPage:
    """The 4 KB page of a data symbol (adrp operand)."""
    name: str


@dataclass(frozen=True)
class SymPageOffset:
    """A data symbol's offset within its page (add operand)."""
    name: str


@dataclass(frozen=True)
class Cond:
    """A condition code operand, e.g. `eq` in `cset w0, eq`."""
    name: str


@dataclass(frozen=True)
class Shift:
    """A shifted operand modifier, e.g. `lsl #16`."""
    kind: str
    amount: int


Operand = Union[Reg, Imm, Mem, FrameSlot, LabelRef, SymPage, SymPageOffset, Cond, Shift]


@dataclass
class Instr:
    mnemonic: str
    operands: tuple = ()


@dataclass
class LabelDef:
    name: str


@dataclass
class Call:
    """`bl symbol`; the emitter applies the target's symbol naming."""
    target: str


@dataclass
class AddrOf:
    """dst = address of a frame slot; resolved after frame layout."""
    dst: Reg
    slot: FrameSlot


@dataclass
class AsmFunction:
    name: str
    instructions: list


@dataclass
class AsmProgram:
    functions: list
    string_literals: list
    type_descriptors: list


def X(n: int) -> Reg:
    return Reg(f'x{n}')


def W(n: int) -> Reg:
    return Reg(f'w{n}')


FP, LR, SP = Reg('x29'), Reg('x30'), Reg('sp')
