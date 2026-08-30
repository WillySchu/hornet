"""Utility functions for codegen."""

from codegen.assembly_ast import Operand, Register
from codegen.errors import CodegenError
from parser import Node
from semantic import Type, TypeKind, StructInfo


# 32-bit register name -> its 8-bit low-byte alias (e.g. %eax -> %al).
# `sete` (and friends) can only target an 8-bit operand, so codegen needs
# to be able to get from "the register I'm working in" to "its byte
# alias". Only registers actually in use are listed; extend this table
# alongside whatever new registers the code generator starts using.
_BYTE_REGISTER_ALIASES = {
    'eax': 'al',
    'edx': 'dl',  # needed by the print machinery's own int-to-decimal
    # conversion, which computes each digit as a remainder in %edx and
    # needs to write it out as a single byte.
}


def as_byte_register(reg: Operand) -> Register:
    if not isinstance(reg, Register) or reg.name not in _BYTE_REGISTER_ALIASES:
        raise CodegenError(f"No 8-bit alias known for register operand: {reg!r}")
    return Register(_BYTE_REGISTER_ALIASES[reg.name])


# 32-bit register name -> its 64-bit alias (e.g. %eax -> %rax). Needed
# because Push/Pop can't operate on a 32-bit operand size in long mode.
_QWORD_REGISTER_ALIASES = {
    'eax': 'rax',
    'ecx': 'rcx',
}


def as_qword_register(reg: Operand) -> Register:
    if not isinstance(reg, Register) or reg.name not in _QWORD_REGISTER_ALIASES:
        raise CodegenError(f"No 64-bit alias known for register operand: {reg!r}")
    return Register(_QWORD_REGISTER_ALIASES[reg.name])


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


def type_of(expr: Node) -> Type:
    """Reads the type semantic.py already resolved and annotated onto this
    exact node (expr.resolved_type -- see semantic.py's check_expr) rather
    than re-deriving it independently.

    Raises a clear, defensive CodegenError (matching _local_offset's own
    posture) rather than a bare AttributeError if resolved_type is somehow
    None. The one legitimate way that happens is codegen being invoked on an
    AST that skipped semantic analysis entirely.

    Returns a full semantic.Type object. Callers can compare against
    Type.INT/Type.BOOL/Type.STR directly, or inspect
    .kind/.element_type/.size for an array."""
    if expr.resolved_type is None:
        raise CodegenError(
            f"{expr!r} has no resolved type -- semantic.analyze() "
            f"must run before codegen (see compile_to_asm)"
        )
    return expr.resolved_type
