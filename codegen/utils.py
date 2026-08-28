"""Utility functions for codegen."""

from semantic import Type, TypeKind, StructInfo


def type_byte_width(t: Type, structs: dict[str, StructInfo]) -> int:
    """Total bytes needed to store a value of type `t`: 4 for int/bool,
    8 for str (a pointer), 24 for a slice (ptr, len, cap), recursively `size *
    type_byte_width(element_type)` for an array, and, for a struct, the SUM of type_byte_width
    over each of its own fields' types, in declaration order (see the
    STRUCTS section). This is the same "flatten it and add up the
    pieces" idea the array case already uses, just over a
    heterogeneous field list instead of N copies of one element type.
    `structs` is this program's own struct registry (see StructInfo in
    semantic.py), needed to look up a struct type's own field list by
    name; threaded through every recursive call the same way type_
    from_name's own registry parameter is. This is the one place that
    recursion lives; every caller that needs an array's or struct's
    total size (stack allocation, whole-value copies) or the shift-
    per-index/per-field (address computation) goes through this or
    leaf_type below rather than re-deriving either."""
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
    whether a flat element-by-element copy should move 4 or 8 bytes at
    a time -- a multi-dimensional array is just one contiguous block
    of leaf values for copying purposes, with no per-dimension logic
    needed once this is known.

    Stops at a SLICE the same way it already stops at STR -- neither
    is unwrapped further, since both are copied as one fixed-size
    unit (a pointer, or a {pointer, length} pair) rather than
    recursed into element by element."""
    while t.kind == TypeKind.ARRAY:
        t = t.element_type
    return t


def escape_for_asciz(s: str) -> str:
    """Escapes `s` (an already-unescaped Hornet string value -- see
    parser.py's _unescape_string_literal) for embedding in a GAS
    `.asciz "..."` directive. Backslash has to be escaped *first*, or
    the escapes added for the other characters would themselves get
    re-escaped; double-quote needs escaping since that's the
    directive's own delimiter; the rest are the common control
    characters getting their standard short escape so the emitted
    assembly stays readable text rather than raw control bytes."""
    s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"')
    s = s.replace('\n', '\\n')
    s = s.replace('\t', '\\t')
    s = s.replace('\r', '\\r')
    return s
