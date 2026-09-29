"""x86-64 backend driver: per-function register allocation, instruction selection, frame layout, prologue/epilogue."""


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
from codegen.calling_convention import CALLEE_SAVED_SCRATCH_REGISTERS
from codegen.emitter import Emitter
from ir.ir import IRCall, IRFunction, IRProgram
from codegen.ir_lowering import InstructionSelector
from codegen.register_allocator import allocate_registers
from codegen.scalars_lowering import ScalarsLoweringMixin


# IR -> assembly AST

class CodeGenerator(
        ArraysSlicesLoweringMixin,
        ScalarsLoweringMixin):
    """Lowers an IRProgram to an AsmProgram."""

    def __init__(self):
        self._next_offset = 0
        self._slot_offsets: Dict[int, int] = {}  # slot id -> %rbp offset; set by _resolve_frame_layout
        self._register_assignment: Dict[int, str] = {}
        # fail labels reset per function; message labels cached per program
        self._bounds_check_fail_labels = {}
        self._bounds_check_message_labels = {}
        self.ir_program: Optional[IRProgram] = None

    def _resolve_frame_layout(self, ir_fn: IRFunction) -> None:
        """Assign %rbp offsets to every logical slot, in creation order. Runs once per function after lower_ir, when all slots are known."""
        next_offset = 0
        for slot_id, width in ir_fn.slot_widths.items():
            if slot_id == ir_fn.outgoing_stack_args_slot:
                continue
            next_offset -= width
            self._slot_offsets[slot_id] = next_offset

        if ir_fn.outgoing_stack_args_slot is not None:
            width = ir_fn.slot_widths[ir_fn.outgoing_stack_args_slot]
            raw_before = -next_offset
            pad = (16 - (raw_before + width) % 16) % 16
            next_offset -= pad
            next_offset -= width
            pushed_bytes = 8 * len(CALLEE_SAVED_SCRATCH_REGISTERS)
            self._slot_offsets[ir_fn.outgoing_stack_args_slot] = next_offset - pushed_bytes

        self._next_offset = next_offset

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
        ir = ir_fn.body

        # Reserve outgoing stack-argument space before lower_ir needs it.
        max_overflow_slots = max(
            (len(instr.args) - 6 for instr in ir if isinstance(instr, IRCall)),
            default=0,
        )
        if max_overflow_slots > 0:
            ir_fn.outgoing_stack_args_slot = self.ir_program.ids.new_slot(
                8 * max_overflow_slots, "outgoing_stack_args", ir_fn,
            )

        self._register_assignment = allocate_registers(ir, self.ir_program.ids._temp_offsets)
        instructions = []
        instructions.extend(InstructionSelector(self, ir_fn).lower_ir(ir))
        self._resolve_frame_layout(ir_fn)
        self._patch_frame_slots(instructions)
        self._register_assignment = {}
        instructions.extend(self._gen_bounds_check_panic_block())

        # Identical for every function.
        prologue: List[Instruction] = [
            Push(Register('rbp')),
            MovQ(src=Register('rsp'), dst=Register('rbp')),
        ]
        # Callee-saved scratch registers are saved unconditionally.
        for reg in CALLEE_SAVED_SCRATCH_REGISTERS:
            prologue.append(Push(Register(reg)))

        frame_size = self._frame_size()
        if frame_size:
            prologue.append(SubQ(src=Imm(frame_size), dst=Register('rsp')))

        return AsmFunction(name=ir_fn.name, instructions=prologue + instructions)

    def _frame_size(self) -> int:
        # Frame rounded up to 16 bytes for call alignment.
        raw = -self._next_offset
        return ((raw + 15) // 16) * 16 if raw > 0 else 0

    def _gen_epilogue(self) -> List[Instruction]:
        """Restore callee-saved scratch registers, then leave/ret."""
        instructions = []
        for reg in reversed(CALLEE_SAVED_SCRATCH_REGISTERS):
            instructions.append(Pop(Register(reg)))
        instructions.append(Leave())
        instructions.append(Ret())
        return instructions


# Entry points

def lower_to_asm(ir_program: IRProgram, platform: str = 'macos') -> str:
    """Lower an optimized IRProgram to assembly text."""
    return Emitter(platform=platform).emit(CodeGenerator().generate(ir_program))
