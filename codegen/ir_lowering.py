"""Instruction selection: IR -> assembly AST. Temps not assigned a register get a lazily allocated frame slot,
emitted as a FrameSlot placeholder and resolved after lowering.
"""

from codegen.assembly_ast import (
    CmpQ,
    CallInstr,
    Cmp,
    FrameSlot,
    Imm,
    Instruction,
    Ja,
    Jae,
    Je,
    Jmp,
    Label,
    LeaQ,
    LeaQFrameSlot,
    Memory,
    Mov,
    MovQ,
    Operand,
    Register,
)
from ir.ir import (
    IRBinOp,
    IRBoundsCheck,
    IRBranch,
    IRCall,
    IRCast,
    IRConst,
    IRCopy,
    IRJump,
    IRLabel,
    IRLoad,
    IRLocalAddress,
    IRMove,
    IRReadArgument,
    IRReturn,
    IRSliceBoundsCheck,
    IRSliceGrow,
    IRStaticDataAddress,
    IRStore,
    IRUnOp,
    IRValue,
    Temp,
)
from typesys import is_wide_type, type_byte_width
from codegen.utils import as_qword_register, ARG_REGISTERS_32, ARG_REGISTERS_64
from typesys import Type


class InstructionSelector:
    """Lowers one function's IR."""

    def __init__(self, host, ir_fn):
        self.host = host
        self.ir_fn = ir_fn

    def _temp_mem(self, temp: Temp) -> Operand:
        """Frame slot for `temp`, allocated on first use."""
        if temp.id in self.host.ir_program.ids._temp_offsets:
            return FrameSlot(slot=self.host.ir_program.ids._temp_offsets[temp.id])
        if temp.id not in self.host.ir_program.ids._temp_slots:
            width = type_byte_width(temp.type, self.host.ir_program.struct_registry, self.host.ir_program.sum_type_registry)
            self.host.ir_program.ids._temp_slots[temp.id] = self.host.ir_program.ids.new_slot(width, f"temp:{temp.id}", self.ir_fn)
        return FrameSlot(slot=self.host.ir_program.ids._temp_slots[temp.id])

    def _gen_load_value(self, value: IRValue, dst: Register) -> list[Instruction]:
        """Load an IRValue into `dst` (32-bit name; widened by type)."""
        if isinstance(value, IRConst):
            if value.type == Type.INT64:
                return [MovQ(src=Imm(value.value), dst=as_qword_register(dst))]
            return [Mov(src=Imm(value.value), dst=dst)]
        return self._gen_read_temp_into(value, dst)

    def _gen_read_temp_into(self, temp: Temp, dst: Register) -> list[Instruction]:
        """Read a Temp from its register or slot into `dst`."""
        reg_name = self.host._register_assignment.get(temp.id)
        if reg_name is not None:
            src = Register(reg_name)
            wide = is_wide_type(temp.type)
            if wide:
                src, dst = as_qword_register(src), as_qword_register(dst)
            if src == dst:
                return []
            return [MovQ(src=src, dst=dst)] if wide else [Mov(src=src, dst=dst)]
        return self.host._gen_read_scalar_into(self._temp_mem(temp), temp.type, dst)

    def _gen_write_temp_from(self, src: Register, temp: Temp) -> list[Instruction]:
        """Write `src` to a Temp's register or slot."""
        reg_name = self.host._register_assignment.get(temp.id)
        if reg_name is not None:
            dst = Register(reg_name)
            wide = is_wide_type(temp.type)
            if wide:
                src, dst = as_qword_register(src), as_qword_register(dst)
            if src == dst:
                return []
            return [MovQ(src=src, dst=dst)] if wide else [Mov(src=src, dst=dst)]
        return self.host._gen_write_scalar_from(src, temp.type, self._temp_mem(temp))

    @staticmethod
    def _cmp(src: IRValue, dst: IRValue):
        """Compare %eax (dst) against %ecx (src), 64-bit if either operand is wide."""
        if is_wide_type(src.type) or is_wide_type(dst.type):
            return CmpQ(src=Register('rcx'), dst=Register('rax'))
        return Cmp(src=Register('ecx'), dst=Register('eax'))

    def lower_ir(self, instructions: list) -> list[Instruction]:
        """Lower an IR fragment."""
        out: list[Instruction] = []
        for instr in instructions:
            if isinstance(instr, IRMove):
                out.extend(self._gen_load_value(instr.src, Register('eax')))
                out.extend(self._gen_write_temp_from(Register('eax'), instr.dst))
            elif isinstance(instr, IRBinOp):
                out.extend(self._gen_load_value(instr.left, Register('eax')))
                out.extend(self._gen_load_value(instr.right, Register('ecx')))
                out.extend(self.host.gen_binary_op(
                    instr.op, src=Register('ecx'), dst=Register('eax'), operand_type=instr.left.type))
                out.extend(self._gen_write_temp_from(Register('eax'), instr.dst))
            elif isinstance(instr, IRUnOp):
                out.extend(self._gen_load_value(instr.operand, Register('eax')))
                out.extend(self.host.gen_unary_op(instr.op, Register('eax'), operand_type=instr.operand.type))
                out.extend(self._gen_write_temp_from(Register('eax'), instr.dst))
            elif isinstance(instr, IRCast):
                # src.type avoids truncating an already-int64 source.
                out.extend(self._gen_load_value(instr.src, Register('eax')))
                out.extend(self.host.gen_cast_narrowing_into(instr.dst.type, Register('eax'), instr.src.type))
                out.extend(self._gen_write_temp_from(Register('eax'), instr.dst))
            elif isinstance(instr, IRLabel):
                out.append(Label(instr.name))
            elif isinstance(instr, IRJump):
                out.append(Jmp(instr.label))
            elif isinstance(instr, IRBranch):
                # No fallthrough peephole yet: always emits both jumps.
                out.extend(self._gen_load_value(instr.cond, Register('eax')))
                out.append(Cmp(src=Imm(0), dst=Register('eax')))
                out.append(Je(instr.false_label))
                out.append(Jmp(instr.true_label))
            elif isinstance(instr, IRCall):
                for i, arg_value in enumerate(instr.args):
                    if i < 6:
                        out.extend(self._gen_load_value(arg_value, Register(ARG_REGISTERS_32[i])))
                        continue
                    wide = is_wide_type(arg_value.type)
                    scratch = as_qword_register(Register('eax')) if wide else Register('eax')
                    out.extend(self._gen_load_value(arg_value, Register('eax')))
                    dst = FrameSlot(self.ir_fn.outgoing_stack_args_slot, 8 * (i - 6))
                    out.append(MovQ(src=scratch, dst=dst) if wide else Mov(src=scratch, dst=dst))
                out.append(CallInstr(instr.name))
                if instr.dst is not None:
                    out.extend(self._gen_write_temp_from(Register('eax'), instr.dst))
            elif isinstance(instr, IRReadArgument):
                wide = is_wide_type(instr.dst.type)
                if instr.index < 6:
                    src = Register((ARG_REGISTERS_64 if wide else ARG_REGISTERS_32)[instr.index])
                else:
                    src = Memory('rbp', 16 + 8 * (instr.index - 6))
                out.append(MovQ(src=src, dst=Register('rax')) if wide else Mov(src=src, dst=Register('eax')))
                out.extend(self._gen_write_temp_from(Register('eax'), instr.dst))
            elif isinstance(instr, IRReturn):
                if instr.value is not None:
                    out.extend(self._gen_load_value(instr.value, Register('eax')))
                out.extend(self.host._gen_epilogue())
            elif isinstance(instr, IRLoad):
                out.extend(self._gen_load_value(instr.address, Register('eax')))
                if instr.dst.type == Type.STR:
                    out.append(MovQ(src=Memory('rax', 0), dst=Register('rax')))
                else:
                    out.extend(self.host._gen_read_scalar_into(Memory('rax', 0), instr.dst.type, Register('eax')))
                out.extend(self._gen_write_temp_from(Register('eax'), instr.dst))
            elif isinstance(instr, IRStore):
                # address in %r9, value in %eax: both live at the store
                out.extend(self._gen_load_value(instr.address, Register('r9d')))
                out.extend(self._gen_load_value(instr.value, Register('eax')))
                if instr.value_type == Type.STR:
                    out.append(MovQ(src=Register('rax'), dst=Memory('r9', 0)))
                else:
                    out.extend(self.host._gen_write_scalar_from(Register('eax'), instr.value_type, Memory('r9', 0)))
            elif isinstance(instr, IRLocalAddress):
                out.append(LeaQFrameSlot(slot=instr.slot, dst=Register('rax')))
                out.extend(self._gen_write_temp_from(Register('eax'), instr.dst))
            elif isinstance(instr, IRStaticDataAddress):
                out.append(LeaQ(label=instr.label, dst=Register('rax')))
                out.extend(self._gen_write_temp_from(Register('eax'), instr.dst))
            elif isinstance(instr, IRCopy):
                # %r9/%r8: both addresses live at once
                out.extend(self._gen_load_value(instr.dst_address, Register('r9d')))
                out.extend(self._gen_load_value(instr.src_address, Register('r8d')))
                out.extend(self.host.gen_array_copy(Memory('r9', 0), Memory('r8', 0), instr.value_type))
            elif isinstance(instr, IRBoundsCheck):
                # Cmp is unsigned; signedness lives in the jump.
                out.extend(self._gen_load_value(instr.length, Register('ecx')))
                out.extend(self._gen_load_value(instr.index, Register('eax')))
                out.append(self._cmp(instr.length, instr.index))
                out.append(Jae(self.host._get_bounds_check_fail_label("array index out of bounds")))
            elif isinstance(instr, IRSliceBoundsCheck):
                out.extend(self._gen_load_value(instr.bound, Register('ecx')))
                out.extend(self._gen_load_value(instr.value, Register('eax')))
                out.append(self._cmp(instr.bound, instr.value))
                out.append(Ja(self.host._get_bounds_check_fail_label("slice bounds out of range")))
            elif isinstance(instr, IRSliceGrow):
                # Fixed callee-saved %rbx/%r12/%r13 survive the malloc inside grow.
                out.extend(self._gen_load_value(instr.ptr, Register('ebx')))
                out.extend(self._gen_load_value(instr.length, Register('r12d')))
                out.extend(self._gen_load_value(instr.cap, Register('r13d')))
                out.extend(self.host._gen_slice_grow_into(
                    Register('rbx'), Register('r12'), Register('r13'), instr.element_width))
                # Order the writes out of %ebx/%r13d so neither clobbers the other's source; swap via %r12 (length is dead).
                dst_ptr_reg = self.host._register_assignment.get(instr.dst_ptr.id)
                dst_cap_reg = self.host._register_assignment.get(instr.dst_cap.id)
                if dst_ptr_reg == 'r13d' and dst_cap_reg == 'ebx':
                    out.append(MovQ(src=Register('rbx'), dst=Register('r12')))
                    out.extend(self._gen_write_temp_from(Register('r13d'), instr.dst_cap))
                    out.extend(self._gen_write_temp_from(Register('r12d'), instr.dst_ptr))
                elif dst_ptr_reg == 'r13d':
                    out.extend(self._gen_write_temp_from(Register('r13d'), instr.dst_cap))
                    out.extend(self._gen_write_temp_from(Register('ebx'), instr.dst_ptr))
                else:
                    out.extend(self._gen_write_temp_from(Register('ebx'), instr.dst_ptr))
                    out.extend(self._gen_write_temp_from(Register('r13d'), instr.dst_cap))
            else:
                raise NotImplementedError(f"lower_ir has no rule for: {instr!r}")
        return out
