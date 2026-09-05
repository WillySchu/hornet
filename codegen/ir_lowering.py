"""Lowers a list of ir.py instructions into assembly_ast.py
Instructions -- the "instruction selection" step of this pipeline.
v1, deliberately: every Temp gets its own permanent stack slot,
assigned the first time it's referenced here (see _temp_mem; the same
self._next_offset allocator _collect_locals/_reserve_argument_temp
already share) -- allocating a Temp itself (_new_temp, in codegen.py)
makes no storage decision at all. An op that combines two values
loads them into scratch registers, then hands off to the existing
gen_binary_op/gen_unary_op (ScalarsMixin) as this pass's own
instruction-selection rule -- that arithmetic isn't reimplemented
here. Real register allocation -- a Temp getting a physical register
instead of a permanent slot when possible -- is a separate, later
step: only _temp_mem's own policy should need to change when that
arrives.
"""

from codegen.assembly_ast import Instruction, Register, Memory, Imm, Mov, MovQ, Cmp, Je, Jmp, Label, CallInstr
from codegen.ir import (
    IRBinOp,
    IRBranch,
    IRCall,
    IRConst,
    IRJump,
    IRLabel,
    IRMove,
    IRRaw,
    IRReturn,
    IRUnOp,
    IRValue,
    Temp,
)
from codegen.utils import as_qword_register, type_byte_width
from semantic import Type


class IRLoweringMixin:
    def _temp_mem(self, temp: Temp) -> Memory:
        """Returns temp's Memory location, assigning it a permanent
        stack slot the first time it's referenced (memoized in
        _temp_offsets) rather than at temp-creation time -- see
        _new_temp's own docstring for why storage assignment is kept
        separate from allocating the temp itself: this method is v1's
        entire lowering policy for where a Temp lives, and a future
        register allocator only needs to replace this one method, not
        anything upstream that creates Temps."""
        if temp.id not in self._temp_offsets:
            self._next_offset -= type_byte_width(temp.type, self.struct_registry)
            self._temp_offsets[temp.id] = self._next_offset
        return Memory('rbp', self._temp_offsets[temp.id])

    def _gen_load_value(self, value: IRValue, dst: Register) -> list[Instruction]:
        """Loads an IRValue (a Temp's current value, or a compile-time
        IRConst) into `dst`, a 32-bit-named register -- widened to
        `dst`'s own 64-bit view internally wherever the value's type
        needs it, matching every other scalar read/write site in this
        compiler. IRConst is never str-typed (a string literal needs a
        label address, not an immediate value), so only the Temp path
        needs str's own special case -- see _gen_read_temp_into."""
        if isinstance(value, IRConst):
            if value.type == Type.INT64:
                return [MovQ(src=Imm(value.value), dst=as_qword_register(dst))]
            return [Mov(src=Imm(value.value), dst=dst)]
        return self._gen_read_temp_into(value, dst)

    def _gen_read_temp_into(self, temp: Temp, dst: Register) -> list[Instruction]:
        """Reads a Temp's current value from its slot into `dst`.
        str needs its own case (a full 8-byte pointer read via MovQ):
        _gen_read_scalar_into (ScalarsMixin) only special-cases
        int8/uint8/int64, falling through to a plain 4-byte Mov for
        everything else -- which would truncate a pointer. Every
        existing caller of that method already special-cases str
        itself first; this is one more such caller, not a gap in it."""
        if temp.type == Type.STR:
            return [MovQ(src=self._temp_mem(temp), dst=as_qword_register(dst))]
        return self._gen_read_scalar_into(self._temp_mem(temp), temp.type, dst)

    def _gen_write_temp_from(self, src: Register, temp: Temp) -> list[Instruction]:
        """The write-side counterpart to _gen_read_temp_into -- same
        str special case, for the same reason."""
        if temp.type == Type.STR:
            return [MovQ(src=as_qword_register(src), dst=self._temp_mem(temp))]
        return self._gen_write_scalar_from(src, temp.type, self._temp_mem(temp))

    def lower_ir(self, instructions: list) -> list[Instruction]:
        """Translates one self-contained IR fragment into real
        Instructions -- every op ir.py defines now has a rule here."""
        out: list[Instruction] = []
        for instr in instructions:
            if isinstance(instr, IRRaw):
                out.extend(instr.instructions)
                if instr.dst is not None:
                    out.extend(self._gen_write_temp_from(Register('eax'), instr.dst))
            elif isinstance(instr, IRMove):
                out.extend(self._gen_load_value(instr.src, Register('eax')))
                out.extend(self._gen_write_temp_from(Register('eax'), instr.dst))
            elif isinstance(instr, IRBinOp):
                out.extend(self._gen_load_value(instr.left, Register('eax')))
                out.extend(self._gen_load_value(instr.right, Register('ecx')))
                out.extend(self.gen_binary_op(instr.op, src=Register('ecx'), dst=Register('eax'), operand_type=instr.left.type))
                out.extend(self._gen_write_temp_from(Register('eax'), instr.dst))
            elif isinstance(instr, IRUnOp):
                out.extend(self._gen_load_value(instr.operand, Register('eax')))
                out.extend(self.gen_unary_op(instr.op, Register('eax'), operand_type=instr.operand.type))
                out.extend(self._gen_write_temp_from(Register('eax'), instr.dst))
            elif isinstance(instr, IRLabel):
                out.append(Label(instr.name))
            elif isinstance(instr, IRJump):
                out.append(Jmp(instr.label))
            elif isinstance(instr, IRBranch):
                # No peephole yet for when true_label happens to be
                # whatever immediately follows -- always emits both
                # the conditional and the unconditional jump. Fixing
                # that (once it's worth it) is a lowering-only change;
                # nothing that builds IRBranch needs to know or care.
                out.extend(self._gen_load_value(instr.cond, Register('eax')))
                out.append(Cmp(src=Imm(0), dst=Register('eax')))
                out.append(Je(instr.false_label))
                out.append(Jmp(instr.true_label))
            elif isinstance(instr, IRCall):
                # `args` is unused for now -- see gen_call_into's own
                # docstring for why argument marshaling still happens
                # entirely through the pre-existing calling-convention
                # code, spliced in as an IRRaw immediately before this
                # op runs, rather than through this field.
                out.append(CallInstr(instr.name))
                if instr.dst is not None:
                    out.extend(self._gen_write_temp_from(Register('eax'), instr.dst))
            elif isinstance(instr, IRReturn):
                # The one terminator that leaves the function entirely
                # rather than jumping to a label. Load the value if
                # there is one, then the ordinary epilogue either way.
                # Never used for an array/slice/struct return -- those
                # write through the hidden output pointer instead (see
                # gen_return), a mechanism this doesn't touch.
                if instr.value is not None:
                    out.extend(self._gen_load_value(instr.value, Register('eax')))
                out.extend(self._gen_epilogue())
            else:
                raise NotImplementedError(f"lower_ir has no rule for: {instr!r}")
        return out
