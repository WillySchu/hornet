"""Array/slice instruction selection: copies, slice growth, bounds-check panics."""

from codegen.assembly_ast import (
    Add,
    AddQ,
    CallInstr,
    Cmp,
    Imm,
    IMul,
    Instruction,
    Jae,
    Je,
    Jmp,
    Label,
    LeaQ,
    Memory,
    Mov,
    MovB,
    MovQ,
    Register,
    ShiftRightArithmetic,
)
from ir.utils import type_byte_width, leaf_type
from codegen.utils import as_byte_register
from semantic import Type


class ArraysSlicesLoweringMixin:
    def gen_array_copy(self, dst_mem: Memory, src_mem: Memory, array_type: Type) -> list[Instruction]:
        """Copy array_type's bytes from src_mem to dst_mem, leaf by leaf."""
        leaf = leaf_type(array_type)
        used_bases = {src_mem.base, dst_mem.base}
        scratch_64, scratch_32 = next(
            (r64, r32) for r64, r32 in [('rax', 'eax'), ('rcx', 'ecx'), ('rdx', 'edx')]
            if r64 not in used_bases
        )
        leaf_width = type_byte_width(leaf, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        total = type_byte_width(array_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        instructions = []
        off = 0
        while off < total:
            # 8-byte chunks, then 4, then bytes.
            chunk_off = 0
            while leaf_width - chunk_off >= 8:
                field_src = Memory(src_mem.base, src_mem.offset + off + chunk_off)
                field_dst = Memory(dst_mem.base, dst_mem.offset + off + chunk_off)
                instructions.append(MovQ(src=field_src, dst=Register(scratch_64)))
                instructions.append(MovQ(src=Register(scratch_64), dst=field_dst))
                chunk_off += 8
            if leaf_width - chunk_off >= 4:
                field_src = Memory(src_mem.base, src_mem.offset + off + chunk_off)
                field_dst = Memory(dst_mem.base, dst_mem.offset + off + chunk_off)
                instructions.append(Mov(src=field_src, dst=Register(scratch_32)))
                instructions.append(Mov(src=Register(scratch_32), dst=field_dst))
                chunk_off += 4
            scratch_8 = None
            while leaf_width - chunk_off >= 1:
                if scratch_8 is None:
                    scratch_8 = as_byte_register(Register(scratch_32))
                field_src = Memory(src_mem.base, src_mem.offset + off + chunk_off)
                field_dst = Memory(dst_mem.base, dst_mem.offset + off + chunk_off)
                instructions.append(MovB(src=field_src, dst=scratch_8))
                instructions.append(MovB(src=scratch_8, dst=field_dst))
                chunk_off += 1
            off += leaf_width
        return instructions

    def _gen_new_cap_into(self, r_cap_32: Register) -> list[Instruction]:
        """New capacity in place: cap*2 below 256, else cap + cap/4."""
        instructions = []
        zero_label = self.ir_program.ids.new_label("append_cap_zero")
        quarter_label = self.ir_program.ids.new_label("append_cap_quarter")
        growth_done_label = self.ir_program.ids.new_label("append_growth_done")

        instructions.append(Cmp(src=Imm(0), dst=r_cap_32))
        instructions.append(Je(zero_label))
        instructions.append(Cmp(src=Imm(256), dst=r_cap_32))
        instructions.append(Jae(quarter_label))
        instructions.append(IMul(src=Imm(2), dst=r_cap_32))
        instructions.append(Jmp(growth_done_label))
        instructions.append(Label(zero_label))
        instructions.append(Mov(src=Imm(1), dst=r_cap_32))
        instructions.append(Jmp(growth_done_label))
        instructions.append(Label(quarter_label))
        instructions.append(Mov(src=r_cap_32, dst=Register('eax')))
        instructions.append(Mov(src=Imm(2), dst=Register('ecx')))
        instructions.append(ShiftRightArithmetic(dst=r_cap_32))
        instructions.append(Add(src=Register('eax'), dst=r_cap_32))
        instructions.append(Label(growth_done_label))
        return instructions

    def _gen_slice_grow_into(
            self,
            r_ptr: Register,
            r_len_32: Register,
            r_cap: Register,
            r_cap_32: Register,
            element_width: int,
    ) -> list[Instruction]:
        """IRSliceGrow lowering: compute new cap, call hornet_slice_grow."""
        instructions = self._gen_new_cap_into(r_cap_32)

        instructions.append(MovQ(src=r_ptr, dst=Register('rdi')))
        instructions.append(Mov(src=r_len_32, dst=Register('esi')))
        instructions.append(Mov(src=r_cap_32, dst=Register('edx')))
        instructions.append(Mov(src=Imm(element_width), dst=Register('ecx')))
        instructions.append(CallInstr('hornet_slice_grow'))
        instructions.append(MovQ(src=Register('rax'), dst=r_ptr))
        return instructions

    def _get_bounds_check_fail_label(self, message: str) -> str:
        """Per-function fail label for `message`."""
        if message not in self._bounds_check_fail_labels:
            self._bounds_check_fail_labels[message] = self.ir_program.ids.new_label("bounds_check_fail")
        return self._bounds_check_fail_labels[message]

    def _get_bounds_check_message_label(self, message: str) -> str:
        """Program-wide static label for `message`."""
        if message not in self._bounds_check_message_labels:
            label = self.ir_program.ids.new_label("bounds_msg")
            self._bounds_check_message_labels[message] = label
            self.ir_program.string_literals.append((label, message))
        return self._bounds_check_message_labels[message]

    def _gen_bounds_check_panic_block(self) -> list[Instruction]:
        """Panic block for one bounds-check message."""
        instructions = []
        for message, fail_label in self._bounds_check_fail_labels.items():
            msg_label = self._get_bounds_check_message_label(message)
            instructions.extend([
                Label(fail_label),
                LeaQ(label=msg_label, dst=Register('rdi')),
                CallInstr('hornet_panic'),
            ])
        return instructions
