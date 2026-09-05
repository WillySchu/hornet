"""A small, generic intermediate representation sitting between the
semantically-analyzed AST and assembly_ast.py's own, machine-level
instruction set. Every value lives in a Temp -- a virtual register,
unlimited in supply, carrying a real semantic.Type -- rather than a
concrete x86 register, so a construct's codegen no longer has to
reason about which physical register a sub-expression's value happens
to be sitting in; that's lower_ir's job (ir_lowering.py).

Every block ends in exactly one terminator (IRJump, IRBranch, or
IRReturn) -- no implicit fallthrough, even where the eventual assembly
will fall through naturally. Once anything ever reorders blocks (a
future optimization pass), implicit fallthrough would silently break;
an explicit terminator can't. IRBranch always carries both target
labels for the same reason.

IRRaw is the escape hatch that makes incremental, construct-by-
construct migration possible: it splices in a not-yet-migrated
gen_X_into method's existing output verbatim. By convention, those
instructions leave their result in Register('eax') (or its 64-bit
view, for int64) -- if `dst` is given, lower_ir appends one store from
there into dst's slot. IRRaw is self-eliminating: once a construct
builds real IR instead, nothing constructs one for it again.
"""

from dataclasses import dataclass
from typing import Optional, Union

from codegen.assembly_ast import Instruction
from parser import BinaryOp, UnaryOp
from semantic import Type


@dataclass(frozen=True)
class Temp:
    """A virtual register: unlimited supply, identified by `id` alone
    (two Temps are the same iff their ids match -- `type` is carried
    for convenience, not part of identity, so this stays a safe dict
    key without needing Type's own equality involved at all)."""
    id: int
    type: Type

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Temp) and self.id == other.id


@dataclass(frozen=True)
class IRConst:
    """A compile-time constant operand -- the IR-level counterpart to
    assembly_ast.py's own Imm, kept separate (rather than reusing Imm
    directly) so an IR operand is always exactly a Temp or an IRConst,
    never a raw machine-level Operand a lowering rule hasn't chosen
    yet."""
    value: int
    type: Type


IRValue = Union[Temp, IRConst]


@dataclass
class IRMove:
    """dst = src."""
    dst: Temp
    src: IRValue


@dataclass
class IRBinOp:
    """dst = left OP right."""
    dst: Temp
    op: BinaryOp
    left: IRValue
    right: IRValue


@dataclass
class IRUnOp:
    """dst = OP operand."""
    dst: Temp
    op: UnaryOp
    operand: IRValue


@dataclass
class IRCall:
    """dst = call name(args). dst is None for a void call."""
    dst: Optional[Temp]
    name: str
    args: list[IRValue]


@dataclass
class IRReturn:
    """return value. value is None for a bare return."""
    value: Optional[IRValue]


@dataclass
class IRLabel:
    """A jump target."""
    name: str


@dataclass
class IRJump:
    """Unconditional jump."""
    label: str


@dataclass
class IRBranch:
    """Conditional jump -- always both targets, never an implied
    fallthrough."""
    cond: IRValue
    true_label: str
    false_label: str


@dataclass
class IRRaw:
    """Splices `instructions` -- real assembly_ast.py Instructions,
    exactly as an existing gen_X_into method already returns them --
    in verbatim. If `dst` is given, those instructions are assumed (by
    the caller's own construction) to leave their result in
    Register('eax') or its 64-bit view, and lower_ir appends a store
    from there into dst's slot."""
    instructions: list[Instruction]
    dst: Optional[Temp] = None


IRInstr = Union[IRMove, IRBinOp, IRUnOp, IRCall, IRReturn, IRLabel, IRJump, IRBranch, IRRaw]
