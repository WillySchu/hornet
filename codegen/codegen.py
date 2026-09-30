"""x86-64 backend driver: per-function register allocation, instruction selection, frame layout, prologue/epilogue.

Frame: saved rbp, then the callee-saved registers this function uses, then slots, then
outgoing stack arguments at %rsp. Saved registers plus slots are a multiple of 16 bytes.
"""


import dataclasses
from typing import Dict, List, Optional

from codegen.arrays_slices_lowering import ArraysSlicesLoweringMixin
from codegen.assembly_ast import (
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
from codegen.calling_convention import CALLEE_SAVED_REGISTERS
from codegen.emitter import Emitter
from codegen.peephole import optimize_asm
from ir.ir import IRCall, IRFunction, IRProgram, Temp
from typesys import type_byte_width
from codegen.ir_lowering import InstructionSelector
from codegen.register_allocator import allocate_registers
from codegen.scalars_lowering import ScalarsLoweringMixin
from codegen.utils import as_qword_register


# IR -> assembly AST

class CodeGenerator(
        ArraysSlicesLoweringMixin,
        ScalarsLoweringMixin):
    """Lowers an IRProgram to an AsmProgram."""

    def __init__(self):
        self._frame_bytes = 0
        self._saved_registers: List[str] = []
        self._slot_offsets: Dict[int, int] = {}  # slot id -> %rbp offset; set by _resolve_frame_layout
        self._register_assignment: Dict[int, str] = {}
        self._frame_slots: Dict[object, int] = {}  # slot key -> width, for the function being lowered
        self._spill_slots: Dict[int, object] = {}  # temp id -> slot key
        self._outgoing_slot = None
        # fail labels reset per function; message labels cached per program
        self._bounds_check_fail_labels = {}
        self._bounds_check_message_labels = {}
        self.ir_program: Optional[IRProgram] = None

    def _resolve_frame_layout(self, ir_fn: IRFunction) -> None:
        """Assign %rbp offsets to every logical slot, in creation order, below the saved registers.
        Outgoing stack arguments go at the bottom (%rsp). Runs once per function after lower_ir."""
        saved_bytes = 8 * len(self._saved_registers)
        next_offset = -saved_bytes
        for slot_id, width in self._frame_slots.items():
            if slot_id == self._outgoing_slot:
                continue
            next_offset -= width
            self._slot_offsets[slot_id] = next_offset
        used = -next_offset
        if self._outgoing_slot is not None:
            used += self._frame_slots[self._outgoing_slot]
        # %rsp must be 16-byte aligned at calls: saved registers plus frame is a multiple of 16.
        self._frame_bytes = (used + 15) // 16 * 16 - saved_bytes
        if self._outgoing_slot is not None:
            self._slot_offsets[self._outgoing_slot] = -(saved_bytes + self._frame_bytes)

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
        # The frame: the IR's slots plus this backend's own (outgoing arguments, spills), which
        # are kept here so lowering never modifies the IR.
        self._frame_slots = dict(ir_fn.slot_widths)
        self._spill_slots = {}
        ir = ir_fn.body

        # Reserve outgoing stack-argument space before lower_ir needs it.
        max_overflow_slots = max(
            (len(instr.args) - 6 for instr in ir if isinstance(instr, IRCall)),
            default=0,
        )
        self._outgoing_slot = self.new_frame_slot(8 * max_overflow_slots) if max_overflow_slots > 0 else None

        self._register_assignment = allocate_registers(ir, ir_fn.temp_homes, ir_fn.params)
        used = {as_qword_register(Register(r)).name for r in self._register_assignment.values()}
        self._saved_registers = [r for r in CALLEE_SAVED_REGISTERS if r in used]
        selector = InstructionSelector(self, ir_fn)
        instructions = selector.lower_params(ir_fn.params, ir)
        instructions.extend(selector.lower_ir(ir))
        self._resolve_frame_layout(ir_fn)
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
        if self._frame_bytes:
            prologue.append(SubQ(src=Imm(self._frame_bytes), dst=Register('rsp')))

        return AsmFunction(name=ir_fn.name, instructions=prologue + instructions)

    def new_frame_slot(self, width: int):
        """A backend-owned frame slot (its key never collides with the IR's integer slot ids)."""
        key = ('backend', len(self._frame_slots))
        self._frame_slots[key] = width
        return key

    def spill_slot(self, temp: Temp):
        """The frame slot holding `temp` when it has no register."""
        if temp.id not in self._spill_slots:
            width = type_byte_width(temp.type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
            self._spill_slots[temp.id] = self.new_frame_slot(width)
        return self._spill_slots[temp.id]

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

def lower_to_asm(ir_program: IRProgram, platform: str = 'macos') -> str:
    """Lower an optimized IRProgram to assembly text."""
    return Emitter(platform=platform).emit(CodeGenerator().generate(ir_program))
