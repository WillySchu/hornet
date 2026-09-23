"""Small, stateless, semantic-level helpers shared across every IR-
building mixin: type-width and leaf-type computation, and the
resolved-type accessor every expression's own IR-building reads from.
Deliberately free of anything machine-level (a register, an
assembly_ast operand, an x86 condition code) -- see codegen/utils.py
for that half."""

from parser import Node, Field, Index, Unary, UnaryOp, Variable
from semantic import Type, TypeKind, StructInfo

from ir.errors import IRError

# Every composite (non-scalar, address-based) value type this
# compiler has: assigned/passed/returned by copying the whole value
# through an address rather than living directly in one Temp the way
# int/bool/str/int8/uint8/int64 do -- see e.g. VarDecl/Assign/
# FieldAssign's own composite-value branches in ir/statements.py, all
# of which dispatch on exactly this set. Shared here so admitting a
# new composite kind (SUM, below -- a sum type) was a one-line
# addition instead of an audit of every call site below -- NOT a
# stand-in for every other (ARRAY, STRUCT) or (ARRAY, SLICE) pairing
# elsewhere in this codebase, several of which mean something narrower
# (which two kinds share a bare literal syntax with its own address
# function; which two are passed as a single pointer argument in the
# calling convention) and would be wrong to fold into this one.
COMPOSITE_KINDS = {TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT, TypeKind.SUM, TypeKind.STR}


def is_composite_addressable(expr: Node) -> bool:
    """Whether `expr` already has a real address an array/struct/
    slice-typed value can be copied FROM directly, with no
    construction (a literal) or hidden-output-pointer convention
    (an ordinary composite-returning Call) involved: a Variable,
    Field, or Index (see _ir_array_address/_ir_struct_address/_ir_
    slice_address, whose own dispatch chains this exactly mirrors),
    or, now, `*p` -- a pointer dereferenced for its own pointee's
    whole value (Unary(DEREFERENCE, ...); semantic.py's own check_
    unary still rejects a SUM-typed pointee, so this can only ever
    be ARRAY/SLICE/STRUCT in practice).

    Shared by every VarDecl/Assign/IndexAssign/FieldAssign/Return/
    call-argument/equality-operand site that used to check `isinstance
    (expr, (Variable, Field, Index))` directly before `*p` as a value
    existed -- one shared predicate rather than repeating the same
    added Unary/DEREFERENCE clause at each one individually. Each of
    those three address functions already knows how to compute *p's
    own address on its own (a one-line dispatch to gen_expr_ir(expr.
    operand), added alongside this function -- the pointer's own value
    already IS its pointee's address); this predicate exists purely so
    each CALL SITE's own gating check recognizes the shape at all,
    before ever reaching one of those three functions."""
    if isinstance(expr, (Variable, Field, Index)):
        return True
    return isinstance(expr, Unary) and expr.op == UnaryOp.DEREFERENCE

# A sum type's own memory shape: this many bytes of discriminant tag
# (a plain int -- no variant count in reach anytime soon needs more
# range than that), then, starting immediately after with no padding
# (matching every other type here -- x86-64 doesn't require aligned
# access), room for whichever variant needs the most space. Shared as
# a constant, not recomputed, since BOTH sides of a sum-typed value's
# life need to agree on it exactly: type_byte_width's own SUM case
# (sizing the whole thing) and wherever a value is actually widened
# into one or read back out (locating the tag, then the payload).
SUM_TYPE_TAG_WIDTH = 4


def type_byte_width(t: Type, structs: dict[str, StructInfo], sum_types: dict) -> int:
    """Total bytes needed to store a value of type `t`: 1 for
    int8/uint8, 4 for int/bool, 16 for str (ptr, len -- see ir/
    strings.py's own module docstring), 24 for a slice
    (ptr, len, cap), recursively `size * type_byte_width(element_type)`
    for an array, the sum of type_byte_width over each field, in
    declaration order, for a struct, or -- for a sum type -- a fixed
    4-byte discriminant tag (see codegen's own SUM_TYPE_TAG_WIDTH; a
    plain int-width tag needs no more range than a small variant count
    ever will) plus whichever VARIANT needs the most room, since only
    one is ever live at once and every variant shares that same
    payload space starting right after the tag -- not the SUM of every
    variant's own width the way a struct's fields are, and not padded
    or aligned to the payload's own start, matching every other type
    here (x86-64 doesn't require aligned access). `structs` is this
    program's struct registry, used to look up a struct type's field
    list by name; `sum_types` its sum-type counterpart, used to look
    up a sum type's own variant list.

    `sum_types` is required (like `structs`), not defaulted to an
    empty dict, for the identical reason type_from_name's own required
    parameters are: a call site that forgets it fails loudly (KeyError,
    from the dict lookup below) rather than silently computing a wrong
    width for a sum-typed value -- unlike type_from_name's OWN
    sum_types parameter, which deliberately defaults to disallowed,
    because unlike there, there's no position here where "sum types
    aren't allowed" is a meaningful answer: by the time this is asked
    to size a real Type, semantic.py has already decided whether a sum
    type was legal wherever it came from.

    int8/uint8 returning 1 (not the 4-byte default other scalars get)
    is what makes an array of them genuinely dense; every scalar
    read/write site elsewhere in codegen must be width-aware rather
    than unconditionally moving 4 bytes, since an address computed
    from this width points at a slot of exactly this size."""
    if t == Type.INT8 or t == Type.UINT8:
        return 1
    if t == Type.INT64:
        return 8
    if t.kind == TypeKind.ARRAY:
        return t.size * type_byte_width(t.element_type, structs, sum_types)
    if t.kind == TypeKind.SLICE:
        return 24
    if t.kind == TypeKind.STR:
        return 16  # {ptr, len} -- see this file's own module docstring
    if t.kind == TypeKind.STRUCT:
        return sum(type_byte_width(field_type, structs, sum_types) for field_type in structs[t.struct_name].fields.values())
    if t.kind == TypeKind.SUM:
        variant_widths = (
            type_byte_width(Type(TypeKind.STRUCT, struct_name=variant_name), structs, sum_types)
            for variant_name in sum_types[t.sum_type_name].variants
        )
        return SUM_TYPE_TAG_WIDTH + max(variant_widths)
    if t.kind == TypeKind.POINTER:
        return 8  # one machine address, regardless of the pointee's own width
    return 4  # INT, BOOL


def is_wide_type(t: Type) -> bool:
    """True for the two scalar-shaped types that need a FULL 8-byte
    register/memory move rather than codegen's own ordinary 4-byte
    default: int64 (already established -- see, e.g., _gen_read_
    scalar_into's own docstring in codegen/scalars_lowering.py) and
    POINTER, for the identical reason -- a pointer IS just a raw
    8-byte address, and an ordinary 4-byte Mov would silently truncate
    it, corrupting the address rather than merely losing precision the
    way it would for an oversized int. Every "wide" check throughout
    codegen/ir_lowering.py and codegen/scalars_lowering.py goes
    through this one predicate, so a third wide-needing type, if one
    is ever added, only needs updating here.

    str is deliberately NOT here, despite once being -- see ir/
    strings.py's own module docstring for its current, 16-byte {ptr,
    len} representation: no longer one 8-byte value that merely needs
    a full-width move, but two, needing the multi-value treatment
    slice already gets (a descriptor read/written through an address,
    flattened into two separate call-argument slots, never a single
    Temp) rather than is_wide_type's own single-Temp-but-wide one."""
    return t in (Type.INT64,) or t.kind == TypeKind.POINTER


def leaf_type(t: Type) -> Type:
    """Recursively unwraps array types to find the innermost, non-array
    element type -- e.g. for [2][3]int, the leaf type is int. Used to
    decide whether a flat element-by-element copy should move 1, 4, or
    8 bytes at a time. Stops at SLICE and STR without unwrapping
    further -- both are copied as one fixed-size unit."""
    while t.kind == TypeKind.ARRAY:
        t = t.element_type
    return t


def type_of(expr: Node) -> Type:
    """Reads the type semantic.py already resolved and annotated onto
    this node (expr.resolved_type) rather than re-deriving it.

    Raises IRError, not a bare AttributeError, if resolved_type is
    None -- the one legitimate cause is codegen running on an AST that
    skipped semantic analysis."""
    if expr.resolved_type is None:
        raise IRError(
            f"{expr!r} has no resolved type -- semantic.analyze() "
            f"must run before codegen (see compile_to_asm)"
        )
    return expr.resolved_type
