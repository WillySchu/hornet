"""x86-64 backend driver: per-function register allocation, instruction selection, frame layout, prologue/epilogue.

Frame: saved rbp, then the callee-saved registers this function uses, then slots, then
outgoing stack arguments at %rsp. Saved registers plus slots are a multiple of 16 bytes.
"""


import dataclasses
from typing import Dict, List, Optional

from backend.x86_64.arrays_slices_lowering import ArraysSlicesLoweringMixin
from backend.x86_64.assembly_ast import (
    AsmFunction,
    AsmProgram,
    FrameSlot,
    Imm,
    Instruction,
    LeaQFrame,
    LeaQFrameSlot,
    Leave,
    Memory,
    MovQ,
    Pop,
    Push,
    Register,
    Ret,
    SubQ,
)
from backend.x86_64.calling_convention import ALLOCATABLE_REGISTERS, CALLEE_SAVED_POOL, CALLEE_SAVED_REGISTERS
from backend.x86_64.emitter import Emitter
from backend.x86_64.peephole import optimize_asm
from ir.ir import IRCall, IRFunction, IRProgram
from backend.x86_64.ir_lowering import InstructionSelector
from backend.common.frame import Frame
from backend.common.regalloc import allocate_registers
from backend.x86_64.scalars_lowering import ScalarsLoweringMixin
from backend.x86_64.utils import as_qword_register
from target import Target


# IR -> assembly AST

class CodeGenerator(
        ArraysSlicesLoweringMixin,
        ScalarsLoweringMixin):
    """Lowers an IRProgram to an AsmProgram."""

    def __init__(self):
        self._saved_registers: List[str] = []
        self._slot_offsets: Dict[int, int] = {}  # slot id -> %rbp offset; set by _resolve_frame_layout
        self._register_assignment: Dict[int, str] = {}
        self.allocation_log = None  # set to a list to record (ir, temp homes, params, assignment) per function
        self.frame = None  # Frame of the function being lowered
        # fail labels reset per function; message labels cached per program
        self._bounds_check_fail_labels = {}
        self._bounds_check_message_labels = {}
        self.ir_program: Optional[IRProgram] = None

    @staticmethod
    def _referenced_slots(instructions: List[Instruction]) -> set:
        used = set()
        for instr in instructions:
            if isinstance(instr, LeaQFrameSlot):
                used.add(instr.slot)
                continue
            for f in dataclasses.fields(instr):
                value = getattr(instr, f.name)
                if isinstance(value, FrameSlot):
                    used.add(value.slot)
        return used

    def _patch_frame_slots(self, instructions: List[Instruction]) -> None:
        """Replace logical slot placeholders with resolved frame offsets."""
        for i, instr in enumerate(instructions):
            if isinstance(instr, LeaQFrameSlot):
                instructions[i] = LeaQFrame(offset=self._slot_offsets[instr.slot], dst=instr.dst)
                continue
            for f in dataclasses.fields(instr):
                value = getattr(instr, f.name)
                if isinstance(value, FrameSlot):
                    setattr(instr, f.name, Memory('rbp', self._slot_offsets[value.slot] + value.extra_offset))

    def generate(self, ir_program: IRProgram) -> AsmProgram:
        """Lower every function in `ir_program`."""
        self.ir_program = ir_program
        asm_functions = [self.lower_function(ir_fn, ir_program) for ir_fn in ir_program.functions]
        return AsmProgram(
            functions=asm_functions,
            string_literals=ir_program.string_literals,
            type_descriptors=ir_program.type_descriptors,
        )

    def lower_function(self, ir_fn: IRFunction, ir_program: IRProgram) -> AsmFunction:
        """Allocate registers, lower, lay out the frame, and add prologue/epilogue."""
        self.ir_program = ir_program
        self._bounds_check_fail_labels = {}
        self._slot_offsets = {}
        # The IR's slots plus this backend's own; kept here so lowering never modifies the IR.
        self.frame = Frame(ir_fn, ir_program.struct_registry, ir_program.sum_type_registry)
        ir = ir_fn.body

        # Reserve outgoing stack-argument space before lower_ir needs it.
        max_overflow_slots = max(
            (len(instr.args) - 6 for instr in ir if isinstance(instr, IRCall)),
            default=0,
        )
        self.frame.reserve_outgoing(8 * max_overflow_slots)

        self._register_assignment = allocate_registers(
            ir, ALLOCATABLE_REGISTERS, CALLEE_SAVED_POOL, ir_fn.temp_homes, ir_fn.params)
        if self.allocation_log is not None:
            self.allocation_log.append((ir, ir_fn.temp_homes, ir_fn.params, dict(self._register_assignment)))
        used = {as_qword_register(Register(r)).name for r in self._register_assignment.values()}
        self._saved_registers = [r for r in CALLEE_SAVED_REGISTERS if r in used]
        selector = InstructionSelector(self, ir_fn)
        instructions = selector.lower_params(ir_fn.params, ir)
        instructions.extend(selector.lower_ir(ir))
        self.frame.layout(save_area=8 * len(self._saved_registers), used=self._referenced_slots(instructions))
        self._slot_offsets = self.frame.offsets
        self._patch_frame_slots(instructions)
        self._register_assignment = {}
        instructions.extend(self._gen_bounds_check_panic_block())
        instructions = optimize_asm(instructions)

        prologue: List[Instruction] = [
            Push(Register('rbp')),
            MovQ(src=Register('rsp'), dst=Register('rbp')),
        ]
        for reg in self._saved_registers:
            prologue.append(Push(Register(reg)))
        if self.frame.size:
            prologue.append(SubQ(src=Imm(self.frame.size), dst=Register('rsp')))

        return AsmFunction(name=ir_fn.name, instructions=prologue + instructions)

    def _gen_epilogue(self) -> List[Instruction]:
        """Restore saved callee-saved registers (from just below %rbp), then leave/ret."""
        instructions = []
        if self._saved_registers:
            instructions.append(LeaQFrame(offset=-8 * len(self._saved_registers), dst=Register('rsp')))
        for reg in reversed(self._saved_registers):
            instructions.append(Pop(Register(reg)))
        instructions.append(Leave())
        instructions.append(Ret())
        return instructions


# Entry points

def lower_to_asm(ir_program: IRProgram, target: Target) -> str:
    """Lower an optimized IRProgram to assembly text."""
    return Emitter(target).emit(CodeGenerator().generate(ir_program))
