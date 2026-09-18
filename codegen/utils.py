"""Small, stateless, machine-level helpers shared across every
lowering mixin: register-width aliasing (a 32-bit name to its 8-bit
or 64-bit alias), the SysV ABI's own condition-code and argument-
register tables. See ir/utils.py's own module docstring for this
file's semantic-level counterpart -- type-width/leaf-type computation
and the resolved-type accessor -- split out from here since neither
reasons about a concrete x86-64 register at all."""

from codegen.assembly_ast import Operand, Register
from codegen.errors import CodegenError
from parser import BinaryOp


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
