"""The instruction-selection half of array/slice codegen -- everything
here takes already-decided registers or memory locations and builds
plain assembly_ast Instructions directly, the same role gen_binary_op/
gen_unary_op/_gen_read_scalar_into/_gen_write_scalar_from play for
scalars (see scalars_lowering.py's own module docstring).

Entry points reached from elsewhere: gen_array_copy, _gen_slice_grow_
into, _get_bounds_check_fail_label (ir_lowering.py's own three direct
calls into this file), and _gen_bounds_check_panic_block (lower_
function's own direct call). Everything else here (_gen_new_cap_into,
_get_bounds_check_message_label) is purely internal to those."""

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
        """Copies array_type's worth of data from src_mem to dst_mem
        -- both arbitrary Memory operands -- via a flat sequence of
        movq/movl/movb instructions. A multi-dimensional array is just
        one contiguous block of leaf values in row-major order for
        copying purposes, so no per-dimension logic is needed, just
        the total byte width and the leaf element's own width.

        Each leaf-sized chunk is copied as a flat run of 8-byte movqs,
        then one trailing 4-byte movl if at least 4 bytes remain, then
        a trailing run of 1-byte movbs for whatever's left (0 to 3
        bytes) -- correct for ANY leaf width, not just a multiple of 4.

        A raw, flat byte copy is always semantically identical to
        copying a value "as" whatever logical type or fields those
        bytes represent, given this language's value semantics: there's
        no reference counting, copy-constructor, or write barrier
        anywhere that a flat byte copy could get wrong. This is why a
        slice element (24 bytes: ptr, len, cap) already worked before
        struct existed, and why a struct containing a nested
        array/slice/struct needs nothing more than this same flat copy.

        The scratch register shuttling each chunk is picked dynamically
        to differ from BOTH src_mem's and dst_mem's own base register
        -- otherwise loading a value into it would destroy the address
        a later iteration still needs."""
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
            # Copy exactly leaf_width bytes starting at offset `off`:
            # as many 8-byte movq chunks as fit, then one 4-byte movl
            # if at least 4 bytes remain, then a trailing run of
            # 1-byte movbs for whatever's left (0 to 3 bytes, so at
            # most three movb pairs, never a real loop of its own).
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
        """Computes append's own growth rule in place, overwriting
        r_cap_32 with the new capacity: cap*2 while cap < 256, else
        cap + cap//4 (a right shift by 1, matching integer division by
        2, then added back), with a cap==0 floor of 1 -- doubling zero
        forever stays zero, so that case needs its own explicit floor.
        Uses %eax/%ecx as scratch (the shift's own fixed operand
        register); r_cap_32 itself is never one of those two, by every
        one of this method's own callers' own convention."""
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
        """The growth-ONLY half of append -- IRSliceGrow's own
        lowering. Computes the new capacity inline first, via _gen_
        new_cap_into (pure arithmetic/policy with no allocation of its
        own, so it stays here rather than moving into runtime.c),
        then calls hornet_slice_grow (see runtime.c) to malloc the
        new, larger backing and copy the existing r_len_32 elements
        over.

        Leaves r_len_32 itself untouched: growth doesn't change how
        many elements currently exist, only how much room there is.
        r_ptr is overwritten with the new backing's own address on
        return; r_cap/r_cap_32 already hold the new capacity by the
        time the call happens, and -- being fixed, callee-saved
        registers (%rbx/%r12/%r13) -- simply survive the call
        unchanged.

        Assumes the caller has ALREADY determined reallocation is
        needed -- no internal check of any kind here."""
        instructions = self._gen_new_cap_into(r_cap_32)
        # r_cap_32 (and, via the zero-extension a 32-bit write always
        # gives its own 64-bit register, r_cap itself) now holds
        # new_cap.

        instructions.append(MovQ(src=r_ptr, dst=Register('rdi')))
        instructions.append(Mov(src=r_len_32, dst=Register('esi')))
        instructions.append(Mov(src=r_cap_32, dst=Register('edx')))
        instructions.append(Mov(src=Imm(element_width), dst=Register('ecx')))
        instructions.append(CallInstr('hornet_slice_grow'))
        instructions.append(MovQ(src=Register('rax'), dst=r_ptr))
        return instructions

    def _get_bounds_check_fail_label(self, message: str) -> str:
        """Lazily creates a per-function, per-message label that every
        bounds check using this exact `message` jumps to on failure --
        reused across however many checks in this function share the
        same message, rather than duplicating the panic sequence (see
        _gen_bounds_check_panic_block) at every check site. A single
        function can use more than one message (e.g. "array index out
        of bounds" vs "slice bounds out of range"), each getting its
        own fail label, all reset together at the start of every
        function -- unlike the message labels below, these are purely
        LOCAL jump targets, meaningless outside the function they're
        generated for."""
        if message not in self._bounds_check_fail_labels:
            self._bounds_check_fail_labels[message] = self.ir_program.ids.new_label("bounds_check_fail")
        return self._bounds_check_fail_labels[message]

    def _get_bounds_check_message_label(self, message: str) -> str:
        """Lazily creates and caches (for the rest of the whole
        compilation, unlike the per-function fail labels above -- a
        plain static string, safely shared by every function that
        needs it) a label for this exact `message` string."""
        if message not in self._bounds_check_message_labels:
            label = self.ir_program.ids.new_label("bounds_msg")
            self._bounds_check_message_labels[message] = label
            self.ir_program.string_literals.append((label, message))
        return self._bounds_check_message_labels[message]

    def _gen_bounds_check_panic_block(self) -> list[Instruction]:
        """Appended once at the end of a function's instructions (see
        gen_function) for every distinct message that function's
        bounds checks actually used -- none at all if it never
        triggered any. Each block calls hornet_panic(msg) (see
        runtime.c), which prints the message then aborts (SIGABRT)
        rather than a plain exit() -- an out-of-bounds access is a
        genuine program bug, not a normal termination condition, the
        same "abnormal termination" character division by zero's
        hardware-trapped SIGFPE already has. Never reached via
        ordinary fall-through from the function's body -- every return
        already leaves via `leave; ret`, and hornet_panic itself never
        returns -- so appending these at the very end is always
        safe."""
        instructions = []
        for message, fail_label in self._bounds_check_fail_labels.items():
            msg_label = self._get_bounds_check_message_label(message)
            instructions.extend([
                Label(fail_label),
                LeaQ(label=msg_label, dst=Register('rdi')),
                CallInstr('hornet_panic'),
            ])
        return instructions
