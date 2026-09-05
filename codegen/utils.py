"""Small, stateless helpers shared across every mixin: register-width
aliasing (a 32-bit name to its 8-bit or 64-bit alias), type-width and
leaf-type computation, the resolved-type accessor every expression's
codegen reads from, and the register-protection wrapper used wherever
code that might use a destination's base register as scratch has to
run before that destination is finally written to."""

from codegen.assembly_ast import Operand, Register, Memory, Instruction, Pop, Push
from codegen.errors import CodegenError
from parser import Node, BinaryOp
from semantic import Type, TypeKind, StructInfo


# BinaryOp -> the x86 condition-code suffix that implements it, given
# that Cmp(src=right, dst=left) computes (left - right) and sets flags
# accordingly. All six comparisons share one codegen path (see
# gen_binary_op) that just plugs the relevant cc into SetCC.
COMPARISON_CONDITION_CODES = {
    BinaryOp.EQUAL: 'e',
    BinaryOp.NOT_EQUAL: 'ne',
    BinaryOp.LESS_THAN: 'l',
    BinaryOp.GREATER_THAN: 'g',
    BinaryOp.LESS_THAN_OR_EQUAL: 'le',
    BinaryOp.GREATER_THAN_OR_EQUAL: 'ge',
}


# SysV ABI integer/pointer argument registers, in order, 64-bit and
# 32-bit forms. Only the first 6 arguments of a call are supported --
# beyond that the ABI moves to stack-passed arguments, which this
# compiler doesn't implement. The 32-bit names don't follow one
# consistent pattern: rdi/rsi/rdx/rcx are "legacy" registers with
# their own historical e-prefixed names, while r8/r9 are x86-64-only
# and use a d-suffix instead -- hence two explicit parallel lists
# rather than a derived mapping.
ARG_REGISTERS_64 = ['rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9']
ARG_REGISTERS_32 = ['edi', 'esi', 'edx', 'ecx', 'r8d', 'r9d']


# 32-bit register name -> its 8-bit low-byte alias (e.g. %eax -> %al).
# `sete` (and friends) can only target an 8-bit operand, so codegen
# needs to get from "the register I'm working in" to "its byte alias".
# Covers all 16 general-purpose registers, not just the ones already
# in active use: a scalar write can be asked to truncate-and-store
# from whatever register a call site's value happens to be sitting in
# (%eax most of the time, but also a protected register like %r8d),
# so this needs to be complete up front rather than extended
# reactively call site by call site.
_BYTE_REGISTER_ALIASES = {
    'eax': 'al', 'ebx': 'bl', 'ecx': 'cl', 'edx': 'dl',
    'esi': 'sil', 'edi': 'dil', 'ebp': 'bpl', 'esp': 'spl',
    'r8d': 'r8b', 'r9d': 'r9b', 'r10d': 'r10b', 'r11d': 'r11b',
    'r12d': 'r12b', 'r13d': 'r13b', 'r14d': 'r14b', 'r15d': 'r15b',
}


def as_byte_register(reg: Operand) -> Register:
    if not isinstance(reg, Register) or reg.name not in _BYTE_REGISTER_ALIASES:
        raise CodegenError(f"No 8-bit alias known for register operand: {reg!r}")
    return Register(_BYTE_REGISTER_ALIASES[reg.name])


# 32-bit register name -> its 64-bit alias (e.g. %eax -> %rax). Needed
# because Push/Pop can't operate on a 32-bit operand size in long mode,
# and because int64 needs a full-width read/write from whatever
# register a call site's computed value happens to be sitting in --
# covers all 16 general-purpose registers up front for the same reason
# _BYTE_REGISTER_ALIASES above does.
_QWORD_REGISTER_ALIASES = {
    'eax': 'rax', 'ebx': 'rbx', 'ecx': 'rcx', 'edx': 'rdx',
    'esi': 'rsi', 'edi': 'rdi', 'ebp': 'rbp', 'esp': 'rsp',
    'r8d': 'r8', 'r9d': 'r9', 'r10d': 'r10', 'r11d': 'r11',
    'r12d': 'r12', 'r13d': 'r13', 'r14d': 'r14', 'r15d': 'r15',
}


def as_qword_register(reg: Operand) -> Register:
    if not isinstance(reg, Register) or reg.name not in _QWORD_REGISTER_ALIASES:
        raise CodegenError(f"No 64-bit alias known for register operand: {reg!r}")
    return Register(_QWORD_REGISTER_ALIASES[reg.name])


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


def escape_for_asciz(s: str) -> str:
    """Escapes `s` (an already-unescaped Hornet string value) for
    embedding in a GAS `.asciz "..."` directive. Backslash is escaped
    *first*, or the escapes added for the other characters would
    themselves get re-escaped; double-quote needs escaping since
    that's the directive's own delimiter; the rest are the common
    control characters getting their standard short escape so the
    emitted assembly stays readable text."""
    s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"')
    s = s.replace('\n', '\\n')
    s = s.replace('\t', '\\t')
    s = s.replace('\r', '\\r')
    return s


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


def gen_protecting_dst_across(dst_mem: Memory, inner: list[Instruction]) -> list[Instruction]:
    """Wraps `inner` with a push/pop protecting dst_mem's base register
    across it, but only when that base isn't 'rbp' -- the frame
    pointer, never clobbered by anything in this file, so wrapping
    would be wasted instructions. Used wherever code that might use
    dst_mem.base as scratch internally (bounds-checking, evaluating an
    arbitrary expression, computing another address) has to run before
    dst_mem is finally read from or written to -- e.g. a hidden return
    pointer received in %rax could otherwise be silently destroyed by
    an inner computation's own use of %rax/%rcx as scratch before it
    was ever used. Found necessary by a real bug during development (a
    segfault on `return matrix[i]` from a function returning an
    array), not assumed defensively."""
    if dst_mem.base == 'rbp':
        return inner
    return [Push(Register(dst_mem.base))] + inner + [Pop(Register(dst_mem.base))]
