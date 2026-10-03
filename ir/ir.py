"""Architecture-agnostic IR. Values live in typed Temps (virtual registers).
Every block ends in exactly one terminator (IRJump, IRBranch, IRReturn); no fallthrough.
"""

from dataclasses import dataclass, field
from typing import Optional, Union

from ops import BinaryOp, UnaryOp
from typesys import Type


@dataclass(frozen=True)
class Temp:
    """Virtual register; identity is `id` alone."""
    id: int
    type: Type

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Temp) and self.id == other.id


@dataclass(frozen=True)
class IRConst:
    """Constant operand."""
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
class IRCast:
    """dst = src converted to dst.type."""
    dst: Temp
    src: IRValue


@dataclass
class IRCall:
    """dst = name(args); dst None for void."""
    dst: Optional[Temp]
    name: str
    args: list[IRValue]


@dataclass
class IRReturn:
    """Return value; None for bare return."""
    value: Optional[IRValue]


@dataclass
class IRLoad:
    """dst = *(address + offset) at dst.type's width; `offset` is a constant byte count."""
    dst: Temp
    address: IRValue
    offset: int = 0


@dataclass
class IRStore:
    """*(address + offset) = value at value_type's width (the destination's declared type)."""
    address: IRValue
    value: IRValue
    value_type: Type
    offset: int = 0


@dataclass
class IRLocalAddress:
    """dst = address of frame slot `slot` (logical id; offsets resolved at lowering)."""
    dst: Temp
    slot: int


@dataclass
class IRStaticDataAddress:
    """dst = address of a static data label."""
    dst: Temp
    label: str


@dataclass
class IRCopy:
    """Copy value_type's width from src_address + src_offset to dst_address + dst_offset."""
    dst_address: IRValue
    src_address: IRValue
    value_type: Type
    dst_offset: int = 0
    src_offset: int = 0


@dataclass
class IRNullCheck:
    """Panic if pointer is none. Built by the IR builder at each dereference and expanded into a
    branch to a panic block right after the function is built (ir/null_checks.py), so passes and
    backends never see it."""
    pointer: IRValue


@dataclass
class IRBoundsCheck:
    """Panic if unsigned index >= length."""
    index: IRValue
    length: IRValue


@dataclass
class IRSliceBoundsCheck:
    """Panic if unsigned value > bound."""
    value: IRValue
    bound: IRValue


@dataclass
class IRLabel:
    """Jump target."""
    name: str


@dataclass
class IRJump:
    """Unconditional jump."""
    label: str


@dataclass
class IRBranch:
    """Conditional jump; both targets explicit."""
    cond: IRValue
    true_label: str
    false_label: str


IRInstr = Union[
    IRBinOp,
    IRBoundsCheck,
    IRBranch,
    IRCall,
    IRCast,
    IRCopy,
    IRJump,
    IRLabel,
    IRLoad,
    IRLocalAddress,
    IRMove,
    IRReturn,
    IRSliceBoundsCheck,
    IRStaticDataAddress,
    IRStore,
    IRUnOp,
]


@dataclass
class IRFunction:
    """One function's IR plus its frame-slot registry (slot id -> width, label).

    `params` are the Temps holding the function's incoming argument words, in order. Hornet's
    calling convention, independent of any target: a composite return value's destination
    address comes first, then each parameter as one word, except str (ptr, len) and slices
    (ptr, len, cap); arrays, structs, sum types, and dicts are passed by address. Where each
    word arrives is the backend's business."""
    name: str
    body: list = field(default_factory=list)
    return_type: Optional[Type] = None
    params: list = field(default_factory=list)
    slot_widths: dict = field(default_factory=dict)
    slot_labels: dict = field(default_factory=dict)
    hidden_return_ptr_slot: Optional[int] = None
    var_slots: dict = field(default_factory=dict)
    temp_homes: dict = field(default_factory=dict)  # named-variable Temp id -> its variable's slot


@dataclass
class IRProgram:
    """Whole-program IR: functions, static data, semantic registries, and the shared IdAllocator."""
    functions: list = field(default_factory=list)
    string_literals: list = field(default_factory=list)
    type_descriptors: list = field(default_factory=list)
    struct_registry: dict = field(default_factory=dict)
    sum_type_registry: dict = field(default_factory=dict)
    escape_summaries: dict = field(default_factory=dict)  # function name -> per-param escape flags
    ids: object = None
    _empty_str_label: object = None
