"""x86-64 register roles under the two calling conventions: SysV (Linux, macOS) and Windows x64."""

from dataclasses import dataclass

from target import Target

# Allocatable registers: every general-purpose register but the scratch ones (%rax, %rcx, %rdx,
# which also have implicit roles in division and shifts) and %rsp/%rbp. %rdi, %rsi, %r8, and %r9
# also pass arguments, so incoming parameters and outgoing arguments are moved in parallel
# (ir_lowering._parallel_moves); being caller-saved, they never hold a value live across a call.
# Caller-saved come first so values not live across a call leave the callee-saved ones free.
# Callee-saved registers are saved by the prologue when used.
CALLER_SAVED_POOL = ['r10d', 'r11d', 'edi', 'esi', 'r8d', 'r9d']
CALLEE_SAVED_POOL = ['ebx', 'r12d', 'r13d', 'r14d', 'r15d']
ALLOCATABLE_REGISTERS = CALLER_SAVED_POOL + CALLEE_SAVED_POOL

# SysV callee-saved registers the allocator may use (%rbp is the frame pointer).
CALLEE_SAVED_REGISTERS = ['rbx', 'r12', 'r13', 'r14', 'r15']


@dataclass(frozen=True)
class Abi:
    """What a calling convention fixes. Under both, %rax, %rcx, and %rdx are scratch (never
    allocated), integer results return in %rax, and the stack is 16-byte aligned at a call."""
    arg_registers_64: tuple  # integer arguments, in order; later ones go on the stack
    arg_registers_32: tuple
    shadow_space: int  # bytes the caller keeps free at the bottom of its frame, below stack arguments
    caller_saved_pool: tuple  # allocatable, lost across a call
    callee_saved_pool: tuple  # allocatable, saved by the prologue when used
    callee_saved_registers: tuple  # callee_saved_pool by 64-bit name (%rbp is the frame pointer)
    probes_stack: bool  # a frame of a page or more must touch each page in order as it grows
    unwind_tables: bool  # each function describes its prologue, and ends in the form the unwinder knows

    @property
    def allocatable(self) -> list:
        return list(self.caller_saved_pool + self.callee_saved_pool)

    def outgoing_bytes(self, argument_words: int) -> int:
        """The space a call with this many argument words needs at the bottom of the frame."""
        return self.shadow_space + 8 * max(0, argument_words - len(self.arg_registers_64))


SYSV = Abi(
    arg_registers_64=('rdi', 'rsi', 'rdx', 'rcx', 'r8', 'r9'),
    arg_registers_32=('edi', 'esi', 'edx', 'ecx', 'r8d', 'r9d'),
    shadow_space=0,
    caller_saved_pool=tuple(CALLER_SAVED_POOL), callee_saved_pool=tuple(CALLEE_SAVED_POOL),
    callee_saved_registers=tuple(CALLEE_SAVED_REGISTERS),
    probes_stack=False, unwind_tables=False,
)

# Windows x64: four argument registers; the caller leaves 32 bytes ("shadow space") above the return
# address for the callee's use on every call; %rdi and %rsi are callee-saved; the stack grows a
# guard page at a time, so a large frame is probed (___chkstk_ms) before %rsp moves; and the system
# walks the stack (for debuggers, profilers, and exceptions) from tables that describe each
# function's prologue, expecting an epilogue of `lea`, `pop`s, and `ret`.
WIN64 = Abi(
    arg_registers_64=('rcx', 'rdx', 'r8', 'r9'),
    arg_registers_32=('ecx', 'edx', 'r8d', 'r9d'),
    shadow_space=32,
    caller_saved_pool=('r10d', 'r11d', 'r8d', 'r9d'),
    callee_saved_pool=('ebx', 'r12d', 'r13d', 'r14d', 'r15d', 'edi', 'esi'),
    callee_saved_registers=('rbx', 'r12', 'r13', 'r14', 'r15', 'rdi', 'rsi'),
    probes_stack=True, unwind_tables=True,
)

STACK_PROBE_FROM = 4096  # a frame at least this large is probed, where the ABI asks for it


def abi_for(target: Target) -> Abi:
    return WIN64 if target.os == 'windows' else SYSV
