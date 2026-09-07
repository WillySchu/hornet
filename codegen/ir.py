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
    key without needing Type's own equality involved at all).

    `is_named_local`, set only by CodeGenerator._temp_at_offset, marks
    a Temp that backs a source-level variable rather than an anonymous
    compiler-generated value. register_allocator.py excludes these
    unconditionally: a named variable's memory slot can still be read
    or written directly (via _local_offset), bypassing the Temp
    entirely, by any not-yet-migrated construct -- promoting one to a
    register would risk exactly that code reading a stale value."""
    id: int
    type: Type
    is_named_local: bool = False

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
class IRLoad:
    """dst = *address, reading dst.type's own width from that
    location. `address` is an ordinary INT64-typed IRValue -- an
    address needs no dedicated representation of its own, since it's
    already just a 64-bit value like any other; whatever computed it
    (bounds-checked array indexing, a struct field's byte offset) is
    typically still old-style code, spliced in via IRRaw producing
    this same INT64 Temp, not rewritten to build IR itself. See
    gen_expr_ir's Index/Field cases."""
    dst: Temp
    address: IRValue


@dataclass
class IRStore:
    """*address = value, writing at `value_type`'s own width to that
    location -- the DECLARED type of the destination, deliberately
    not necessarily value.type: an untyped literal or expression
    flowing into a differently (but compatibly) typed slot can
    disagree, the same reason gen_index_assign/gen_field_assign
    already use the element's/field's own declared type rather than
    the value's for exactly this decision. IRMove doesn't need this
    same explicit field only because its own destination is a Temp,
    which already carries its own correct, declared type on dst.type
    -- an address carries none, so there's nowhere else for it to
    live. See gen_statement_ir's IndexAssign/FieldAssign cases."""
    address: IRValue
    value: IRValue
    value_type: Type


@dataclass
class IRCopy:
    """Copies value_type's own byte width from the address src_
    address to the address dst_address -- both ordinary INT64-typed
    IRValues, the same address-as-a-Temp pattern IRLoad/IRStore
    already use. Whole-array/whole-struct assignment between two
    already-addressable locations (a Variable/Field/Index on both
    sides -- see gen_statement_ir's own VarDecl/Assign/IndexAssign/
    FieldAssign cases, and _ir_copy_assign, for exactly which shapes
    reach this and which still don't): `b = a`, `s.inner = a`, `rows[i]
    = a`, or any mix, but not an ArrayLiteral/struct-literal Call
    (construction, not a copy from an existing address) or an
    ordinary composite-returning Call (writes through a hidden output
    pointer instead, never through two already-existing addresses).

    dst_address/src_address are named for what they hold, not `dst`/
    `src` alone, specifically to avoid reading like a result-Temp the
    way every other op's own `dst` field is -- neither one is a
    result here; both are addresses being READ, and this op writes to
    memory, not to any Temp at all. lower_ir hands both, plus value_
    type, straight to the existing gen_array_copy unchanged -- this
    only makes the address capture and the copy itself real IR, not a
    redesign of the underlying byte-copying mechanism."""
    dst_address: IRValue
    src_address: IRValue
    value_type: Type


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


IRInstr = Union[IRMove, IRBinOp, IRUnOp, IRCall, IRReturn, IRLabel, IRJump, IRBranch, IRRaw, IRLoad, IRStore]
