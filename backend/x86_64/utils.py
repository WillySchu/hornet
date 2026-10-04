"""Machine-level helpers: register aliases, condition codes, argument registers, string escaping."""

from backend.x86_64.assembly_ast import Operand, Register
from backend.errors import CodegenError
from ops import BinaryOp


# BinaryOp -> setcc suffix for Cmp(src=right, dst=left).
COMPARISON_CONDITION_CODES = {
    BinaryOp.EQUAL: 'e',
    BinaryOp.NOT_EQUAL: 'ne',
    BinaryOp.LESS_THAN: 'l',
    BinaryOp.GREATER_THAN: 'g',
    BinaryOp.LESS_THAN_OR_EQUAL: 'le',
    BinaryOp.GREATER_THAN_OR_EQUAL: 'ge',
}


# 32-bit -> 8-bit register name.
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


# 32-bit -> 64-bit register name.
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
