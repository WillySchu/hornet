"""Small, stateless, semantic-level helpers shared across every IR-
building mixin: type-width and leaf-type computation, and the
resolved-type accessor every expression's own IR-building reads from.
Deliberately free of anything machine-level (a register, an
assembly_ast operand, an x86 condition code) -- see codegen/utils.py
for that half."""

from parser import Node
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
COMPOSITE_KINDS = {TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT, TypeKind.SUM}

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
    int8/uint8, 4 for int/bool, 8 for str (a pointer), 24 for a slice
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
        return 8
    if t.kind == TypeKind.STRUCT:
        return sum(type_byte_width(field_type, structs, sum_types) for field_type in structs[t.struct_name].fields.values())
    if t.kind == TypeKind.SUM:
        variant_widths = (
            type_byte_width(Type(TypeKind.STRUCT, struct_name=variant_name), structs, sum_types)
            for variant_name in sum_types[t.sum_type_name].variants
        )
        return SUM_TYPE_TAG_WIDTH + max(variant_widths)
    return 4  # INT, BOOL


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
