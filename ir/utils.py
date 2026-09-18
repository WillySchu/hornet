"""Small, stateless, semantic-level helpers shared across every IR-
building mixin: type-width and leaf-type computation, and the
resolved-type accessor every expression's own IR-building reads from.
Deliberately free of anything machine-level (a register, an
assembly_ast operand, an x86 condition code) -- see codegen/utils.py's
own module docstring for that half, split out from this one for the
identical reason arrays_slices.py/scalars.py were split into a
building half and a lowering half back when this arc's own IR/codegen
package boundary was first drawn: something that only ever reasons
about semantic.Type/parser.Node has no business living in a module
that also reasons about concrete x86-64 registers, even before
anything actually depended on the distinction."""

from parser import Node
from semantic import Type, TypeKind, StructInfo

from codegen.errors import CodegenError


def type_byte_width(t: Type, structs: dict[str, StructInfo]) -> int:
    """Total bytes needed to store a value of type `t`: 1 for
    int8/uint8, 4 for int/bool, 8 for str (a pointer), 24 for a slice
    (ptr, len, cap), recursively `size * type_byte_width(element_type)`
    for an array, and, for a struct, the SUM of type_byte_width over
    each field's type, in declaration order. `structs` is this
    program's struct registry, needed to look up a struct type's field
    list by name. This is the one place that recursion lives; every
    caller that needs an array's or struct's total size, or the shift-
    per-index/per-field for address computation, goes through this or
    leaf_type below rather than re-deriving either.

    int8/uint8 returning 1 (rather than the 4-byte default other
    scalars get) is what makes an array of them genuinely dense -- a
    [1000000]int8 buffer is 1MB, not 4MB -- and is why every scalar
    read/write site elsewhere in codegen needs to be width-aware
    instead of unconditionally moving 4 bytes: an address computed
    from this width now genuinely points at a 1-byte slot. int64
    returning 8 is the same change in the opposite direction, for the
    same reason: an address computed from that width needs a full
    8-byte read/write to reach every byte the value occupies."""
    if t == Type.INT8 or t == Type.UINT8:
        return 1
    if t == Type.INT64:
        return 8
    if t.kind == TypeKind.ARRAY:
        return t.size * type_byte_width(t.element_type, structs)
    if t.kind == TypeKind.SLICE:
        return 24
    if t.kind == TypeKind.STR:
        return 8
    if t.kind == TypeKind.STRUCT:
        return sum(type_byte_width(field_type, structs) for field_type in structs[t.struct_name].fields.values())
    return 4  # INT, BOOL


def leaf_type(t: Type) -> Type:
    """Recursively unwraps array types to find the innermost, non-array
    element type -- e.g. for [2][3]int, the leaf type is int. Used
    wherever codegen needs to know the actual SCALAR type stored at
    the bottom of a (possibly multi-dimensional) array, e.g. to decide
    whether a flat element-by-element copy should move 1, 4, or 8
    bytes at a time -- a multi-dimensional array is just one
    contiguous block of leaf values for copying purposes, with no
    per-dimension logic needed once this is known.

    Stops at a SLICE the same way it stops at STR -- neither is
    unwrapped further, since both are copied as one fixed-size unit
    rather than recursed into element by element."""
    while t.kind == TypeKind.ARRAY:
        t = t.element_type
    return t


def type_of(expr: Node) -> Type:
    """Reads the type semantic.py already resolved and annotated onto
    this node (expr.resolved_type) rather than re-deriving it
    independently.

    Raises a clear, defensive CodegenError rather than a bare
    AttributeError if resolved_type is somehow None -- the one
    legitimate way that happens is codegen being invoked on an AST
    that skipped semantic analysis entirely.

    Returns a full semantic.Type object. Callers can compare against
    Type.INT/Type.BOOL/Type.STR directly, or inspect
    .kind/.element_type/.size for an array."""
    if expr.resolved_type is None:
        raise CodegenError(
            f"{expr!r} has no resolved type -- semantic.analyze() "
            f"must run before codegen (see compile_to_asm)"
        )
    return expr.resolved_type
