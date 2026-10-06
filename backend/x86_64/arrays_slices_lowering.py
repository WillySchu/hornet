"""Array/slice instruction selection: copies and bounds-check panics."""

from backend.x86_64.assembly_ast import (
    CallInstr,
    Instruction,
    Label,
    LeaQ,
    Memory,
    Mov,
    MovB,
    MovQ,
    Register,
)
from typesys import leaf_type, type_byte_width
from backend.x86_64.utils import as_byte_register
from typesys import Type


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

    def _get_bounds_check_message_label(self, message: str) -> str:
        """Program-wide static label for `message`."""
        if message not in self._bounds_check_message_labels:
            label = self.ir_program.ids.new_label("bounds_msg")
            self._bounds_check_message_labels[message] = label
            self.ir_program.string_literals.append((label, message))
        return self._bounds_check_message_labels[message]

    def _gen_bounds_check_panic_block(self) -> list[Instruction]:
        """The function's failed-bounds-check blocks, one per check (ir_lowering's _bounds_fail)."""
        return [instr for block in self._bounds_check_fail_blocks for instr in block]
