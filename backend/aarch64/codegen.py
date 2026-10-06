"""AArch64 backend driver: register allocation, instruction selection, frame layout, prologue/epilogue.

Frame: x29 points at the saved x29/x30 pair; below it the callee-saved registers this function
uses, then slots, then outgoing stack arguments at sp. sp stays 16-byte aligned.
"""

from backend.aarch64.assembly import (
    AddrOf,
    AsmFunction,
    AsmProgram,
    Call,
    FP,
    FrameSlot,
    Imm,
    Instr,
    LabelDef,
    LabelRef,
    LR,
    Mem,
    Reg,
    SP,
    SymPage,
    SymPageOffset,
)
from backend.aarch64.calling_convention import ALLOCATABLE_REGISTERS, CALLEE_SAVED_POOL, MAX_REGISTER_ARGS
from backend.aarch64.emitter import Emitter
from backend.aarch64.legalize import legalize, materialize
from backend.aarch64.lowering import INVERSE, Selector
from backend.common.frame import Frame
from backend.common.jumps import drop_jumps_to_next, invert_branches
from backend.common.regalloc import allocate_registers
from ir.ir import IRCall, IRProgram
from target import Target


class _Jumps:
    """AArch64 spelling of jumps and labels for backend.common.jumps."""

    def label(self, instr):
        return instr.name if isinstance(instr, LabelDef) else None

    def jump(self, instr):
        return instr.operands[0].name if isinstance(instr, Instr) and instr.mnemonic == 'b' else None

    def branch(self, instr):
        if isinstance(instr, Instr) and instr.mnemonic.startswith('b.'):
            return instr.mnemonic[2:], instr.operands[0].name
        if isinstance(instr, Instr) and instr.mnemonic in ('cbz', 'cbnz'):
            return (instr.mnemonic, instr.operands[0]), instr.operands[1].name
        return None

    def make_branch(self, cond, target):
        if isinstance(cond, tuple):
            return Instr(cond[0], (cond[1], LabelRef(target)))
        return Instr(f'b.{cond}', (LabelRef(target),))

    def invert(self, cond):
        if isinstance(cond, tuple):
            return ('cbnz' if cond[0] == 'cbz' else 'cbz', cond[1])
        return INVERSE[cond]


_JUMPS = _Jumps()


def _referenced_slots(instrs: list) -> set:
    used = set()
    for i in instrs:
        if isinstance(i, AddrOf):
            used.add(i.slot.slot)
        elif isinstance(i, Instr):
            used.update(o.slot for o in i.operands if isinstance(o, FrameSlot))
    return used


def _peephole(instrs: list) -> list:
    while True:
        new = drop_jumps_to_next(invert_branches(instrs, _JUMPS), _JUMPS)
        # `mov xN, xN` does nothing (`mov wN, wN` clears the top half, so it stays).
        new = [i for i in new if not (isinstance(i, Instr) and i.mnemonic == 'mov' and len(i.operands) == 2
                                      and i.operands[0] == i.operands[1] and i.operands[0].name.startswith('x'))]
        if new == instrs:
            return new
        instrs = new


class CodeGenerator:
    def __init__(self):
        self.ir_program = None
        self.frame = None
        self.assignment = {}
        self.saved = []  # callee-saved registers the current function uses, as x names
        self.fail_blocks = []  # per function
        self._message_labels = {}  # message -> static string label, per program
        self.allocation_log = None  # set to a list to record (ir, temp homes, params, assignment) per function

    def generate(self, ir_program: IRProgram) -> AsmProgram:
        self.ir_program = ir_program
        return AsmProgram([self.lower_function(fn) for fn in ir_program.functions],
                          ir_program.string_literals, ir_program.type_descriptors)

    def lower_function(self, ir_fn) -> AsmFunction:
        prog = self.ir_program
        self.frame = Frame(ir_fn, prog.struct_registry, prog.sum_type_registry)
        body = ir_fn.body
        overflow = max((len(i.args) - MAX_REGISTER_ARGS for i in body if isinstance(i, IRCall)), default=0)
        self.frame.reserve_outgoing(8 * overflow)
        self.assignment = allocate_registers(
            body, ALLOCATABLE_REGISTERS, CALLEE_SAVED_POOL, ir_fn.temp_homes, ir_fn.params)
        if self.allocation_log is not None:
            self.allocation_log.append((body, ir_fn.temp_homes, ir_fn.params, dict(self.assignment)))
        used = set(self.assignment.values())
        self.saved = [r for r in CALLEE_SAVED_POOL if r in used]
        selector = Selector(self, ir_fn)
        selector.lower_params(ir_fn.params, body)
        self.fail_blocks = []
        instrs = selector.lower(body) + self._panic_blocks()
        save_area = 16 * ((len(self.saved) + 1) // 2)
        self.frame.layout(save_area=save_area, used=_referenced_slots(instrs))
        instrs = _peephole(legalize(instrs, self.frame))
        return AsmFunction(ir_fn.name, self._prologue(save_area + self.frame.size) + instrs)

    def message_label(self, message: str) -> str:
        """The static string holding a panic's `message`, one per program."""
        if message not in self._message_labels:
            self._message_labels[message] = self.ir_program.ids.new_label("bounds_msg")
            self.ir_program.string_literals.append((self._message_labels[message], message))
        return self._message_labels[message]

    def _panic_blocks(self) -> list:
        """The function's failed-bounds-check blocks, one per check (the Selector's _bounds_fail)."""
        return [instr for block in self.fail_blocks for instr in block]

    def _save_pairs(self) -> list:
        """(registers, offset below x29) for saving/restoring callee-saved registers in pairs."""
        regs = [Reg(r) for r in self.saved]
        return [(regs[k:k + 2], -16 * (k // 2 + 1)) for k in range(0, len(regs), 2)]

    def _prologue(self, total: int) -> list:
        out = [Instr('stp', (FP, LR, Mem(SP, -16, 'pre'))), Instr('mov', (FP, SP))]
        if total:
            if total <= 4095:
                out.append(Instr('sub', (SP, SP, Imm(total))))
            else:
                out += materialize(Reg('x9'), total) + [Instr('sub', (SP, SP, Reg('x9')))]
        for regs, offset in self._save_pairs():
            out.append(Instr('stp', (regs[0], regs[1], Mem(FP, offset))) if len(regs) == 2
                       else Instr('str', (regs[0], Mem(FP, offset))))
        return out

    def epilogue(self) -> list:
        out = []
        for regs, offset in self._save_pairs():
            out.append(Instr('ldp', (regs[0], regs[1], Mem(FP, offset))) if len(regs) == 2
                       else Instr('ldr', (regs[0], Mem(FP, offset))))
        return out + [Instr('mov', (SP, FP)), Instr('ldp', (FP, LR, Mem(SP, 16, 'post'))), Instr('ret')]


def lower_to_asm(ir_program: IRProgram, target: Target) -> str:
    return Emitter(target).emit(CodeGenerator().generate(ir_program))
