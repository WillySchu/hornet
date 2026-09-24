"""A small, generic intermediate representation sitting between the
semantically-analyzed AST and assembly_ast.py's own, machine-level
instruction set. Every value lives in a Temp -- a virtual register,
unlimited in supply, carrying a real semantic.Type -- rather than a
concrete x86 register, so a construct's codegen no longer has to
reason about which physical register a sub-expression's value happens
to be sitting in; that's lower_ir's job (ir_lowering.py).

Deliberately architecture-agnostic, not just x86-64-convenient: no op
defined here references a concrete register, an x86-64 instruction, or
any other machine-level detail -- IRLocalAddress/IRStaticDataAddress
included, which represent INTENT ("the address of this frame slot,"
"the address of this label") rather than the mechanism x86-64 happens
to use for it (LeaQFrame/LeaQ). A hypothetical retarget to a different
architecture would need only a new lower_ir, with every op defined in
this file, and everything built from them, entirely unchanged.

Every block ends in exactly one terminator (IRJump, IRBranch, or
IRReturn) -- no implicit fallthrough, even where the eventual assembly
will fall through naturally. IRBranch always carries both target
labels for the same reason."""

from dataclasses import dataclass, field
from typing import Optional, Union

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
    assembly_ast.py's own Imm, kept separate so an IR operand is
    always exactly a Temp or an IRConst, never a raw machine-level
    Operand a lowering rule hasn't chosen yet."""
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
    """dst = src, re-narrowed/re-widened to dst's own declared type --
    the actual work behind an explicit `type(expr)` cast. dst.type
    alone carries the cast's own target type, the same convention
    IRLoad uses for its own read width.

    Deliberately its own op, not an ordinary IRMove into a
    differently-typed Temp: IRMove's own narrowing (via _gen_write_
    temp_from) only happens implicitly ON WRITE to storage, but a
    cast's result must be correctly narrowed in the VALUE itself,
    immediately -- `int8(300) + int8(5)` needs 300 already wrapped to
    44 BEFORE the addition runs, since every later int8/uint8
    operation assumes its own operands already represent a correctly-
    narrowed value. See gen_cast_narrowing_into's own docstring, which
    this op's own lowering reuses unchanged."""
    dst: Temp
    src: IRValue


@dataclass
class IRCall:
    """dst = call name(args). dst is None for a void call."""
    dst: Optional[Temp]
    name: str
    args: list[IRValue]


@dataclass
class IRReadArgument:
    """dst = whatever value the SysV calling convention placed in
    argument slot `index` (0-based, matching codegen/utils.py's own
    ARG_REGISTERS_64/32 indexing) when THIS function was called -- the
    one place real IR needs to know a specific physical register's own
    identity, since that's a fact about the calling convention, not
    about the program.

    `index` counts every argument SLOT the ABI reserves for THIS
    function, not source-level parameters: a slice-typed parameter
    consumes three consecutive indices (ptr, len, cap), and a hidden
    array/slice/struct return pointer, when this function has one, is
    always index 0, shifting every real parameter's own index by one.

    dst's own declared type decides whether this reads the 32- or
    64-bit view of that slot's own register (INT64/STR wide, every
    other scalar type narrow) -- never a narrowing or widening
    read/write, since the value already arrived correctly represented
    for dst's own type. Once a parameter is an ordinary Temp from the
    moment it arrives, every general mechanism that already exists for
    any other Temp -- register allocation, and surviving live across
    an IRCall it doesn't own -- applies to it unconditionally."""
    dst: Temp
    index: int


@dataclass
class IRReturn:
    """return value. value is None for a bare return."""
    value: Optional[IRValue]


@dataclass
class IRLoad:
    """dst = *address, reading dst.type's own width from that
    location. `address` is an ordinary INT64-typed IRValue -- general
    address ARITHMETIC (bounds-checked array indexing, a struct
    field's byte offset) needs no dedicated representation, since it's
    already expressible as ordinary IRBinOp/IRBoundsCheck composition
    against a Temp. The one thing that genuinely needs its own
    representation is the irreducible BASE case that arithmetic like
    this ultimately starts from -- the address of a frame-relative
    local slot, or of a static data label, with no arithmetic to
    express via IRBinOp at all. See IRLocalAddress/IRStaticDataAddress
    for exactly that."""
    dst: Temp
    address: IRValue


@dataclass
class IRStore:
    """*address = value, writing at `value_type`'s own width to that
    location -- the DECLARED type of the destination, deliberately not
    necessarily value.type: an untyped literal or expression flowing
    into a differently (but compatibly) typed slot can disagree.
    IRMove doesn't need this same explicit field only because its own
    destination is a Temp, which already carries its own declared type
    on dst.type -- an address carries none. See gen_statement_ir's
    IndexAssign/FieldAssign cases."""
    address: IRValue
    value: IRValue
    value_type: Type


@dataclass
class IRLocalAddress:
    """dst = the address of the current frame's own local slot
    identified by `slot` -- x86-64's own LeaQFrame, expressed
    architecture-agnostically.

    `slot` is a purely logical identifier (see IdAllocator's own
    new_slot), carrying no physical byte offset: this op never needs
    to know where in the frame a slot actually lives, only which slot
    it means. _resolve_frame_layout is the one place that decides what
    offset a slot gets.

    Always means "the address of this slot," never "the value stored
    there," even for a slot holding a POINTER to heap-allocated data:
    a caller needing the latter composes this with an ordinary IRLoad
    rather than this op having two different meanings depending on
    context.

    Covers a named user variable's own slot, a compiler-reserved
    scratch slot, and the hidden-return-pointer slot alike -- all the
    same underlying concept regardless of what put it there."""
    dst: Temp
    slot: int


@dataclass
class IRStaticDataAddress:
    """dst = the address of a static, read-only data label (e.g. a
    string literal's own backing bytes, or the shared empty-string
    constant) -- x86-64's own LeaQ against a label, expressed
    architecture-agnostically the same way IRLocalAddress is for
    frame-relative addresses."""
    dst: Temp
    label: str


@dataclass
class IRCopy:
    """Copies value_type's own byte width from the address src_address
    to the address dst_address -- both ordinary INT64-typed IRValues,
    the same address-as-a-Temp pattern IRLoad/IRStore use. Whole-
    array/whole-struct/whole-slice assignment between two already-
    addressable locations: `b = a`, `s.inner = a`, `rows[i] = a`, but
    not an ArrayLiteral/struct-literal Call/Slice (construction, not a
    copy from an existing address) or an ordinary composite-returning
    Call (writes through a hidden output pointer instead).

    A slice value_type needs no special handling: a slice's own
    24-byte descriptor is exactly one leaf-sized value as far as
    gen_array_copy is concerned.

    dst_address/src_address are named for what they hold, not dst/src
    alone, to avoid reading like a result-Temp the way every other
    op's dst field is -- neither is a result here; both are addresses
    being READ, and this op writes to memory, not to any Temp."""
    dst_address: IRValue
    src_address: IRValue
    value_type: Type


@dataclass
class IRBoundsCheck:
    """Traps (via a shared, per-function/per-message panic label) if
    `index`, treated as unsigned, is >= `length` -- catching a
    negative index and a too-large one in one check (Hornet has no
    unsigned-comparison operator a program could write, so this is
    purely an internal codegen concept, not a source-level one).

    No result Temp -- like IRCopy, this writes nothing to a Temp; it
    only conditionally jumps elsewhere in the function. `index`/
    `length` are both ordinary INT-typed (32-bit) IRValues."""
    index: IRValue
    length: IRValue


@dataclass
class IRSliceBoundsCheck:
    """Traps (via the same shared panic-label mechanism IRBoundsCheck
    uses, with its own "slice bounds out of range" message) if
    `value`, treated as unsigned, is strictly greater than `bound`.

    A deliberately separate op from IRBoundsCheck, not a generalized
    version with a mode flag: slice production needs three of these
    (low <= cap, high <= cap, low <= high), and unlike indexing's
    single `>=` check, `value == bound` is VALID here (`arr[5:5]` on a
    5-element array is a valid, empty-slice-producing expression) -- a
    real, different comparison, not just a different message.

    No result Temp, for the identical reason IRBoundsCheck has none."""
    value: IRValue
    bound: IRValue


@dataclass
class IRSliceGrow:
    """The growth-ONLY half of append(s, value): grows to a fresh,
    larger backing array (via hornet_slice_grow -- see runtime.c --
    which mallocs the new backing and copies the existing `length`
    elements over) -- correct for ANY element type, since copying an
    already-valid element is always just "copy element_width bytes."
    Produces the new backing's own {ptr, cap} -- NOT length, and NOT
    the newly-appended value itself: growth doesn't change how many
    elements currently exist, only how much room there is, and writing
    the new element (via _ir_write_composite_value_into or an ordinary
    IRStore) is a genuinely separate step the caller takes immediately
    after this op returns.

    Reached only after a real-IR bounds-style check (IRBinOp comparing
    length against cap, plus IRBranch) has already determined there's
    no spare room -- this op itself makes no such check. new_cap is
    computed by the caller (_gen_new_cap_into, pure cap-doubling
    policy with no allocation of its own) before this op's own
    lowering runs, not decided here.

    element_width is plain metadata, not an IRValue -- an element's
    own byte width is always known at IR-construction time.

    Reads ptr/length/cap; writes dst_ptr/dst_cap. Must be treated as
    an unsafe position for register allocation, unlike every other op
    in this file: its own lowering calls hornet_slice_grow, an
    ordinary external function call that's free to clobber any
    caller-saved register -- including %r10d/%r11d, two of this
    compiler's own three allocator-pool registers -- so a Temp
    allocated to the pool cannot safely survive across it, the same
    reasoning that already makes IRCall an unsafe position."""
    dst_ptr: Temp
    dst_cap: Temp
    ptr: IRValue
    length: IRValue
    cap: IRValue
    element_width: int


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
    IRReadArgument,
    IRReturn,
    IRSliceBoundsCheck,
    IRSliceGrow,
    IRStaticDataAddress,
    IRStore,
    IRUnOp,
]


@dataclass
class IRFunction:
    """One Hornet function's own codegen artifacts, gathered into a
    single object by gen_function_ir.

    `body` is real IR (see IRInstr above): this function's own
    parameters (see _ir_param_setup) and its own statements (see gen_
    statement_ir), concatenated into one list, in that order.

    `return_type` is this function's own declared return type (Type.
    VOID for a function with none) -- needed by lowering to decide
    whether a trailing epilogue is required.

    `slot_widths`/`slot_labels` are this function's own logical-slot
    registry (see IdAllocator's own new_slot): every slot's id ->
    (byte width, debug label), in creation order. Living on this
    object (rather than on CodeGenerator) is what lets generate()
    build every function's own IRFunction completely, independent of
    the others, before lowering any of them.

    `hidden_return_ptr_slot`/`var_slots` are two more pieces of per-
    function state read only from gen_statement_ir's own top-level
    Return/VarDecl cases (via _ir_hidden_return_ptr/_bind_local).

    `scopes`, `_argument_temp_slots`, and `_escaping_decl_ids` stay on
    CodeGenerator's own self instead, reset per function: each is read
    from deep inside gen_expr_ir's own call tree across several files,
    so threading them through explicitly would touch many call sites
    for no benefit."""
    name: str
    body: list = field(default_factory=list)
    return_type: Optional[Type] = None
    slot_widths: dict = field(default_factory=dict)
    slot_labels: dict = field(default_factory=dict)
    hidden_return_ptr_slot: Optional[int] = None
    var_slots: dict = field(default_factory=dict)
    outgoing_stack_args_slot: Optional[int] = None  # set by lower_function,
    # not gen_function_ir, before lower_ir runs (see its own comment there)
    # -- the one slot _resolve_frame_layout gives special, always-last
    # treatment, regardless of where in slot_widths' own insertion order
    # it actually landed. None for a function whose own calls never need
    # more than the 6 register-passed argument slots.


@dataclass
class IRProgram:
    """The whole program's own real IR: one IRFunction per Hornet
    function, plus the same two pieces of static, whole-program data
    AsmProgram itself carries one level down (see assembly_ast.py) --
    string_literals/type_descriptors aren't tied to any one function's
    frame, so they live here.

    struct_registry/type_alias_registry/sum_type_registry/function_
    registry (name -> StructInfo/Type/SumTypeInfo/(param types, return
    type), stamped onto Program by semantic.analyze() -- see build_ir_
    program's own defensive check) are copied here too: an
    IRLocalAddress/IRStructAddress referencing a struct field, or any
    type this IR's own Temps carry, is only fully interpretable
    alongside these registries.

    intrinsic_original_names (mangled name -> original_name, also
    stamped onto Program by semantic.analyze() -- see check_intrinsic_
    decl's own docstring) is copied here for the identical reason: a
    Call site's own gen_expr_ir only ever sees its own, already-
    mangled name string, with no way back to the IntrinsicDecl that
    produced it, so this is what lets it recognize "this mangled name
    is one of the compiler's own, fixed intrinsics, specifically this
    one" instead.

    ids (an IdAllocator -- see its own module docstring) is the
    single, live counter every Temp/label/slot this program's own
    build, lowering, and any optimization pass in between draws from.
    No typed default here (unlike every field above): IdAllocator
    lives in codegen/, which itself imports from this module, so
    importing it back here would be circular -- build_ir_program
    constructs the real instance and passes it in explicitly."""
    functions: list = field(default_factory=list)
    string_literals: list = field(default_factory=list)
    type_descriptors: list = field(default_factory=list)
    struct_registry: dict = field(default_factory=dict)
    type_alias_registry: dict = field(default_factory=dict)
    sum_type_registry: dict = field(default_factory=dict)
    function_registry: dict = field(default_factory=dict)
    intrinsic_original_names: dict = field(default_factory=dict)
    ids: object = None
    _empty_str_label: object = None
