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
will fall through naturally. Once anything ever reorders blocks (a
future optimization pass), implicit fallthrough would silently break;
an explicit terminator can't. IRBranch always carries both target
labels for the same reason.

IRRaw used to be the escape hatch that made incremental,
construct-by-construct migration possible: it spliced in a not-yet-
migrated gen_X_into method's existing output verbatim, and was
deliberately, explicitly documented as NOT architecture-agnostic --
a temporary exception to this file's own goal, not a counterexample
to it. It was also self-eliminating by design: once a construct built
real IR instead, nothing constructed one for it again. That
prediction played out completely -- print() (calling the runtime's
own hornet_print) was the last remaining construct still routing
through it, and once that migrated, IRRaw had zero remaining call
sites anywhere, confirmed directly by re-running this arc's own full
audit. It has been removed entirely as a result, not just left
unused: every op below is architecture-agnostic by construction now,
with no remaining exception.
"""

from dataclasses import dataclass, field
from typing import Optional, Union

from parser import BinaryOp, UnaryOp
from semantic import Type


@dataclass(frozen=True)
class Temp:
    """A virtual register: unlimited supply, identified by `id` alone
    (two Temps are the same iff their ids match -- `type` is carried
    for convenience, not part of identity, so this stays a safe dict
    key without needing Type's own equality involved at all).

    Used to also carry is_named_local, marking a Temp that backs a
    source-level variable (set only by IdAllocator.temp_at_offset)
    rather than an anonymous compiler-generated value:
    register_allocator.py excluded these from allocation by default,
    historically because old-style code, and later parameter
    marshaling, could write to a named local's own slot directly,
    bypassing this Temp entirely, which would make promoting it to a
    register unsafe (nothing would ever load the real value into it).
    Both hazards are gone now (old-style code removed entirely;
    parameter marshaling migrated to real IR -- see _ir_param_setup),
    confirmed directly rather than assumed (see this arc's own
    removal of legacy access tracking, CodeGenerator._escaped_
    offsets, once its own exclusion set was confirmed permanently
    empty), so every named-local Temp is read/written exclusively
    through itself now, the identical discipline every other Temp
    already follows -- with nothing left to mark or exclude, this
    field, and the machinery built on it, were removed entirely
    rather than kept as permanently-inert scaffolding."""
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
class IRCast:
    """dst = src, re-narrowed/re-widened to dst's own declared type --
    the actual work behind an explicit `TYPE(expr)` cast. dst.type
    alone carries the cast's own target type, the same convention
    IRLoad already uses for its own read width -- no separate field
    needed for it.

    Deliberately its own op, not an ordinary IRMove into a
    differently-typed Temp: IRMove's own narrowing (via _gen_write_
    temp_from) only ever happens implicitly ON WRITE to storage, but a
    cast's result has to be correctly narrowed in the VALUE itself,
    immediately -- `int8(300) + int8(5)` needs 300 already wrapped to
    44 BEFORE the addition runs, since every later int8/uint8
    operation assumes its own operands already represent a correctly-
    narrowed value, not just "correct once eventually stored." See
    gen_cast_narrowing_into's own docstring, which this op's own
    lowering reuses completely unchanged, for the full account."""
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
    one place real IR ever needs to know a specific physical
    register's own identity, since that's a fact about the calling
    convention itself, not about the program. Every other real-IR op
    is already free of this; this one exists specifically so gen_
    function_ir's own parameter-reading logic doesn't have to build
    plain Instructions directly to get one.

    `index` counts every argument SLOT the ABI reserves for THIS
    function, not source-level parameters: a slice-typed parameter
    consumes three consecutive indices (ptr, len, cap), and a hidden
    array/slice/struct return pointer, when this function has one, is
    always index 0, shifting every real parameter's own index by one
    -- the identical arg_shift accounting gen_function_ir already does
    today, just fed into this instead of an argument register name
    directly.

    dst's own declared type decides whether this reads the 32- or
    64-bit view of that slot's own register (INT64/STR wide, every
    other scalar type narrow) -- never a narrowing or widening
    read/write of any kind, since the value already arrived correctly
    represented for dst's own type: the caller's own IRCall lowering
    already placed it there via the identical, ordinary register-to-
    register move every OTHER Temp-to-Temp copy in this compiler
    relies on (see ir_lowering.py's own IRCall case), which never
    narrows either -- a value is only ever narrowed/widened when it
    crosses a MEMORY boundary (see _gen_read_scalar_into/_gen_write_
    scalar_from's own docstrings), never between two registers already
    holding it correctly. This is exactly what removes the need for
    this compiler to treat a parameter's own arrival specially at all:
    once it's an ordinary Temp from this op onward, every general
    mechanism that already exists for any other Temp -- register
    allocation, and specifically surviving live across an IRCall it
    doesn't own (see register_allocator.py's own module docstring) --
    applies to it unconditionally, with nothing parameter-specific left
    to reimplement."""
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
    field's byte offset) needs no dedicated representation of its own,
    since it's already expressible as ordinary IRBinOp/IRBoundsCheck
    composition against a Temp, and that composition is itself
    already architecture-agnostic (no assembly-level instruction
    embedded anywhere in it). The one thing that genuinely DOES need
    its own representation is the irreducible BASE case that
    arithmetic like this ultimately starts from -- the address of a
    frame-relative local slot, or of a static data label -- since
    those have no arithmetic to express via IRBinOp at all, just a
    fixed, compile-time-known location. See IRLocalAddress/
    IRStaticDataAddress for exactly that, and their own docstrings for
    why they exist at all despite this file's long-standing "an
    address is just a 64-bit value, nothing special" position: that
    position is still correct for the arithmetic, just not for these
    two remaining leaves, which used to be spliced in via IRRaw
    wrapping a raw x86-64 instruction (LeaQFrame/LeaQ) directly --
    architecture-specific machinery embedded in what's meant to be an
    architecture-agnostic IR, unlike IRBinOp/IRLoad/IRStore
    themselves, none of which reference anything x86-64-specific."""
    dst: Temp
    address: IRValue


@dataclass
class IRStore:
    """*address = value, writing at `value_type`'s own width to that
    location -- the DECLARED type of the destination, deliberately
    not necessarily value.type: an untyped literal or expression
    flowing into a differently (but compatibly) typed slot can
    disagree, the same reason _ir_index_assign/_ir_field_assign
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
class IRLocalAddress:
    """dst = the address of the current frame's own local slot
    identified by `slot` -- x86-64's own LeaQFrame, expressed
    architecture-agnostically: a hypothetical ARM64 lowering would
    emit whatever ITS OWN frame-relative addressing looks like, with
    zero change needed to this op, or to anything built on top of it
    (every other real IR op already references nothing x86-64-
    specific at all).

    `slot` is a purely logical identifier -- see IdAllocator's own
    new_slot -- carrying no physical byte offset of its own at all:
    this op, and everything built on top of it, never needs to know
    where in the frame a slot actually lives, only which slot it
    means. _resolve_frame_layout is the one place that ever decides
    what offset a slot gets, once every slot a function needs is
    known; this op is deliberately never the place that decision gets
    made, the same reason it doesn't carry a raw offset directly.

    Always means "the address of this slot," never "the value stored
    there," even for a slot that happens to hold a POINTER to heap-
    allocated data rather than a value directly: a caller needing the
    latter composes this with an ordinary IRLoad (reading the pointer
    stored at that address) rather than this op having two different
    meanings depending on context -- the same "one op, one meaning;
    compose for the rest" discipline this whole arc has followed
    throughout (see _ir_write_slice_descriptor_into_address's own
    +8/+16 IRBinOp composition for an identical example).

    Covers a named user variable's own slot, a compiler-reserved
    scratch slot (_unnamed_slice_temp_slot), and the hidden-return-
    pointer slot alike -- every one of these is "a value at some
    slot, wherever frame layout ultimately puts it," the same
    underlying concept regardless of what put it there or who
    reserved it, so one op covers all three with no special-casing."""
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
    """Copies value_type's own byte width from the address src_
    address to the address dst_address -- both ordinary INT64-typed
    IRValues, the same address-as-a-Temp pattern IRLoad/IRStore
    already use. Whole-array/whole-struct/whole-slice assignment
    between two already-addressable locations (a Variable/Field/Index
    on both sides -- see gen_statement_ir's own VarDecl/Assign/
    IndexAssign/FieldAssign cases, and _ir_copy_assign, for exactly
    which shapes reach this and which still don't): `b = a`, `s.inner
    = a`, `rows[i] = a`, or any mix, but not an ArrayLiteral/struct-
    literal Call/Slice (construction, not a copy from an existing
    address) or an ordinary composite-returning Call (writes through
    a hidden output pointer instead, never through two already-
    existing addresses).

    A slice value_type needs no special handling here at all: a
    slice's own 24-byte descriptor is exactly one leaf-sized value as
    far as gen_array_copy is concerned (see its own docstring -- this
    is the identical flat-byte-copy mechanism an array-of-slices'
    own slice-typed elements already relied on before this op existed
    at all), so lower_ir's own call to it is unchanged regardless of
    which of the three kinds value_type actually is.

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
class IRBoundsCheck:
    """Traps (via the same shared, already-existing per-function/per-
    message panic label _get_bounds_check_fail_label already manages
    -- see _ir_index_address's own docstring for the full "why"
    of the check itself) if `index`, treated as unsigned, is >=
    `length` -- catching a negative index and a too-large one in the
    identical single check the old-style bounds check already does,
    for the identical reason: no ordinary BinaryOp can express this
    (COMPARISON_CONDITION_CODES only has signed condition codes, and
    Hornet itself has no unsigned-comparison operator a program could
    ever write -- this is purely an internal codegen concept, not a
    source-level one, so it doesn't belong in BinaryOp).

    No result Temp -- like IRCopy, this writes nothing to a Temp; it
    only conditionally jumps elsewhere in the function (never falls
    through to something that reads a result), so eligible_intervals'
    own start/end reasoning for a Temp's own def/use never applies to
    it. `index`/`length` are both ordinary INT-typed IRValues (32-bit
    -- an array/slice's own length always fits, exactly like the old-
    style check's own len_reg_32 already assumes)."""
    index: IRValue
    length: IRValue


@dataclass
class IRSliceBoundsCheck:
    """Traps (via the same shared per-function/per-message panic
    label mechanism IRBoundsCheck's own lowering uses, just its own
    "slice bounds out of range" message rather than "array index out
    of bounds" -- see _ir_slice_into's own docstring for the full
    "why" of the check itself) if `value`, treated as unsigned, is
    strictly greater than `bound`.

    A deliberately separate op from IRBoundsCheck, not a generalized
    version of it with a mode flag: slice production needs three of
    these (low <= cap, high <= cap, low <= high -- each an ordinary
    (value, bound) pair, just with different operands), and unlike
    indexing's own single `>=` check, `value == bound` is VALID here
    (`arr[5:5]` on a 5-element array is a valid, empty-slice-producing
    expression) -- a real, different comparison, not just a different
    message, so it earns a real, separate op rather than overloading
    IRBoundsCheck's own meaning.

    No result Temp, for the identical reason IRBoundsCheck has none:
    this only conditionally jumps elsewhere in the function, never
    produces a value anything reads back."""
    value: IRValue
    bound: IRValue


@dataclass
class IRSliceGrow:
    """The growth-ONLY half of append(s, value): mallocs a fresh,
    larger backing array and copies the existing `length` elements
    over from the old one, via an ordinary, generic byte-for-byte
    copy loop -- correct for ANY element type (int/bool/str/array/
    struct/slice alike), since copying an ALREADY-existing, already-
    valid element is always just "copy element_width bytes," with no
    type-specific construction logic needed at all. Produces the new
    backing's own {ptr, cap} -- NOT length, and NOT the newly-
    appended value itself: growth doesn't change how many elements
    currently exist, only how much room there is, and writing the
    new element is a genuinely separate concern, deliberately left to
    the caller (see below).

    Reached only after a real-IR bounds-style check (ordinary IRBinOp
    comparing length against cap, plus IRBranch) has already
    determined there's no spare room in the existing backing array --
    this op itself makes no such check, unlike the old-style _gen_
    grow_and_append_one_into it shares its own growth arithmetic with
    (via _gen_new_cap_into specifically, extracted so both share the
    identical cap-doubling rule without duplicating it).

    Deliberately narrower than this compiler's own former IRAppendGrow,
    which fused growth together with writing the newly-appended value,
    and was scoped to a scalar element type only as a direct
    consequence (its own write step used an ordinary scalar store,
    with no way to write an arbitrary composite value through it).
    Splitting the two apart is what let _ir_append_call support a
    composite element type at all: writing the new value, whatever its
    own type, now happens as an ordinary, SEPARATE step immediately
    after this op returns, reusing _ir_write_composite_value_into (for
    array/slice/struct) or an ordinary IRStore (for everything else)
    completely unchanged -- the identical dispatcher every other
    "write a composite value into a known address" site in this arc
    already uses. This split also means a future runtime-call lowering
    for growth specifically (mirroring hornet_stringify's own "one
    hand-built function, shared across every call site" pattern, since
    growth -- like stringify -- never needs to know anything about the
    VALUE being handled, only its own byte width) would only ever need
    to replace THIS op's own lowering, with zero change needed to how
    the new value gets written afterward, for any element type.

    element_width is plain metadata, not an IRValue -- an element's
    own byte width is always known at IR-construction time.

    Reads ptr/length/cap; writes dst_ptr/dst_cap. Must be treated as
    an unsafe position for register allocation, unlike every other op
    in this file: its own lowering calls malloc, an ordinary external
    function call that's free to clobber any caller-saved register --
    including %r10d/%r11d, two of this compiler's own three allocator-
    pool registers -- so a Temp allocated to the pool cannot safely
    survive across it, the same reasoning that already makes IRCall
    an unsafe position."""
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
    single object by gen_function_ir -- the boundary this compiler's
    own IR/codegen decoupling starts from.

    `body` is real IR (see IRInstr above): this function's own
    parameters (see _ir_param_setup) AND its own statements (see gen_
    statement_ir), concatenated into one list, in that order. Used to
    be two separate things -- a hidden `param_setup` field, still
    plain old-style Instructions, sitting alongside this one -- until
    parameter marshaling itself migrated to real IR (see _ir_param_
    setup's own docstring for why that migration also deleted the old
    two-pass "stash every argument register, then process" structure
    entirely, not just changed its representation): once a parameter's
    own value is an ordinary Temp from the moment it arrives (via
    IRReadArgument), nothing distinguishes it from any statement's own
    Temp at all, so there's no reason left to keep them in separate
    lists, lowered separately, either.

    Used to also carry a `prologue` field -- the callee-saved-register
    pushes and the frame-pointer setup, still plain old-style
    Instructions, not real IR -- removed entirely once it became clear
    it never belonged on this object at all: unlike body, prologue has
    zero dependence on program semantics -- the identical shape for
    every function, parameterized only by frame_size, which isn't even
    known until body is fully lowered anyway -- so there's no VarDecl-
    shaped decision in it for gen_function_ir's own build phase to ever
    have a reason to compute early. lower_function builds it directly
    now, in one place, right where frame_size itself already becomes
    known, rather than gen_function_ir handing back a piece of pure
    calling-convention bookkeeping it never actually needed to touch.

    `return_type` is this function's own declared return type (Type.
    VOID for a function with none) -- gen_function's own lowering half
    needs it (to decide whether a trailing epilogue is required), and
    it's already resolved during the build half, so it travels along
    on this object rather than being recomputed or re-read off self
    afterward.

    `slot_widths`/`slot_labels` are this function's own logical-slot
    registry (see IdAllocator's own new_slot): every slot a name/id ->
    (byte width, debug label) pair, in the exact order each was
    created. Used to live on CodeGenerator's own self, reset at the
    start of every gen_function_ir call -- which meant it could only
    ever be trusted for the ONE function currently being built or
    lowered, since building the NEXT function would silently wipe out
    a not-yet-lowered one's own registry. Living here instead is what
    lets generate() build every function's own IRFunction completely,
    independent of the others, before lowering any of them: each
    carries its own slot registry with it, genuinely surviving from
    new_slot's own first call (during gen_function_ir) through to
    _resolve_frame_layout's own single call (during lower_function),
    no matter how many OTHER functions get built or lowered in
    between. Passed explicitly wherever it's needed (new_slot,
    _collect_params, _collect_locals, _reserve_argument_temp, _resolve_
    frame_layout, and ir_lowering.py's own InstructionSelector, which
    holds the ir_fn it was constructed for as a genuine field) rather
    than read implicitly off self, the same reason IRLocalAddress
    itself carries a slot id explicitly rather than assuming ambient
    knowledge of which function is "current."

    `hidden_return_ptr_slot`/`var_slots` are two more pieces of gen_
    function_ir's own per-function state that used to live on self,
    for the identical reason and with the identical risk the slot
    registry itself had (see above) -- moved here alongside it. Both
    are read only from gen_statement_ir's own top-level Return/VarDecl
    cases (via _ir_hidden_return_ptr/_bind_local), never from gen_
    expr_ir or anything it calls, which is what makes this a small,
    contained move: gen_statement_ir needed `self` threaded through
    its own recursive If/While calls either way, and that's as far as
    either of these ever needs to reach. return_type (above) already
    covers what a separate `_current_return_type` used to duplicate --
    _ir_hidden_return_ptr reads THIS field directly now, once gen_
    function_ir assigns it early enough (right after computing it,
    not at this method's own end) for that to work.

    `scopes`, `_argument_temp_slots`, and `_escaping_array_ids` are
    the ones that deliberately did NOT make this same move, despite
    being reset alongside these on self today: each is read from deep
    inside gen_expr_ir's own call tree (_local_slot/_is_heap_allocated/
    _ir_materialize_composite_call and friends, called from arrays_
    slices.py, structs.py, scalars.py, and dispatch.py alike) --
    threading a context explicitly that deep would mean touching
    gen_expr_ir's own 27 call sites across 6 files, not a handful in
    one method. That's a distinctly bigger, separately-scoped piece of
    work, not attempted here."""
    name: str
    body: list = field(default_factory=list)
    return_type: Optional[Type] = None
    slot_widths: dict = field(default_factory=dict)
    slot_labels: dict = field(default_factory=dict)
    hidden_return_ptr_slot: Optional[int] = None
    var_slots: dict = field(default_factory=dict)


@dataclass
class IRProgram:
    """The whole program's own real IR: one IRFunction per Hornet
    function that generate() actually built through gen_function_ir
    (see IRFunction's own docstring), plus the same two pieces of
    static, whole-program data AsmProgram itself carries one level
    down (see assembly_ast.py) -- string_literals/type_descriptors
    aren't tied to any one function's frame, so they live here rather
    than being duplicated per function.

    struct_registry/type_alias_registry (name -> StructInfo/Type,
    stamped onto Program by semantic.analyze() -- see build_ir_
    program's own defensive check) are copied here too, for the same
    reason: an IRLocalAddress/IRStructAddress referencing a struct
    field, or any type this IR's own Temps carry, is only fully
    interpretable alongside the registries that gave those names and
    types meaning in the first place. Without them, this object would
    be IR that LOOKS self-contained but silently depends on whatever
    CodeGenerator instance happened to build it still being alive and
    unchanged -- exactly the kind of hidden coupling this whole arc has
    been removing one piece at a time.

    ids/_empty_str_label complete that same self-containment: ids (an
    IdAllocator -- see its own module docstring) is THE single, live
    counter every Temp/label/slot this program's own build, and
    everything downstream of it (lowering, and eventually whatever
    optimization passes run in between), draws from -- never a fresh
    one, since two different counters each handing out their own
    "slot 0" would collide. No typed default here (unlike every field
    above): IdAllocator lives in codegen/, and codegen/id_allocator.py
    itself imports IRFunction/Temp from this same module, so importing
    IdAllocator back in here would be circular. build_ir_program
    constructs the one real instance and passes it in explicitly;
    `None` is purely a placeholder for code that hasn't gone through a
    real build yet, the same defensive-default reasoning CodeGenerator
    itself uses for struct_registry and the rest.

    Used to also deliberately exclude the print-stringify helper
    function old-style code conditionally added to AsmProgram's own
    function list -- that gate (generate()'s own `if self._print_used`
    check) and the hand-built AsmFunction it guarded (build_
    stringify_function) have both been removed entirely since, as part
    of this same arc's old-style dead-code cleanup (see this module's
    own top docstring): nothing ever set _print_used to True once
    print() itself migrated to a real IRCall against the runtime, so
    the branch was already permanently unreachable before it was
    deleted. Worth noting for history, since it's exactly the shape of
    thing this object is FOR excluding -- generate() building
    something that was never real IR at all -- even though there's
    nothing left to exclude here anymore.

    No consumer of this object exists yet -- generate() builds and
    keeps it (see self.ir_program) purely so it exists as a real,
    inspectable artifact for this arc's own next steps to build on,
    the identical "make the object real before anything depends on it"
    order IRFunction itself was introduced in."""
    functions: list = field(default_factory=list)
    string_literals: list = field(default_factory=list)
    type_descriptors: list = field(default_factory=list)
    struct_registry: dict = field(default_factory=dict)
    type_alias_registry: dict = field(default_factory=dict)
    ids: object = None
    _empty_str_label: object = None
