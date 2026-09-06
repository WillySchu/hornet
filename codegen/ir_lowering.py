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
here. Real register allocation now exists (register_allocator.py) and
hooks in at _gen_read_temp_into/_gen_write_temp_from: _temp_mem's own
permanent-stack-slot policy remains exactly this, used as the
fallback for whichever Temps the allocator didn't promote.
"""

from codegen.assembly_ast import Instruction, Register, Memory, Imm, Mov, MovQ, Cmp, Je, Jmp, Label, CallInstr
from codegen.ir import (
    IRBinOp,
    IRBranch,
    IRCall,
    IRConst,
    IRJump,
    IRLabel,
    IRLoad,
    IRMove,
    IRRaw,
    IRReturn,
    IRStore,
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
        separate from allocating the temp itself. This is the
        fallback every Temp used to rely on unconditionally; now it's
        only reached for one that register_allocator.py didn't (or
        couldn't -- see its own module docstring) promote to a
        physical register -- see _gen_read_temp_into/_gen_write_temp_
        from, the two places that actually decide which applies."""
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
        """Reads a Temp's current value into `dst`.

        If register_allocator.py assigned it a register (see
        self._register_assignment, set once per function by
        gen_function), the value is already sitting there, not in
        memory -- this is just a register-to-register move (or, if it
        already happens to BE `dst`, no instruction at all). Otherwise,
        falls back to reading its memory slot: str needs its own case
        (a full 8-byte pointer read via MovQ) since _gen_read_scalar_
        into (ScalarsMixin) only special-cases int8/uint8/int64,
        falling through to a plain 4-byte Mov for everything else --
        which would truncate a pointer. Every existing caller of that
        method already special-cases str itself first; this is one
        more such caller, not a gap in it."""
        reg_name = self._register_assignment.get(temp.id)
        if reg_name is not None:
            src = Register(reg_name)
            wide = temp.type in (Type.INT64, Type.STR)
            if wide:
                src, dst = as_qword_register(src), as_qword_register(dst)
            if src == dst:
                return []
            return [MovQ(src=src, dst=dst)] if wide else [Mov(src=src, dst=dst)]
        if temp.type == Type.STR:
            return [MovQ(src=self._temp_mem(temp), dst=as_qword_register(dst))]
        return self._gen_read_scalar_into(self._temp_mem(temp), temp.type, dst)

    def _gen_write_temp_from(self, src: Register, temp: Temp) -> list[Instruction]:
        """The write-side counterpart to _gen_read_temp_into -- same
        register-assignment check, same str special case, for the
        same reasons."""
        reg_name = self._register_assignment.get(temp.id)
        if reg_name is not None:
            dst = Register(reg_name)
            wide = temp.type in (Type.INT64, Type.STR)
            if wide:
                src, dst = as_qword_register(src), as_qword_register(dst)
            if src == dst:
                return []
            return [MovQ(src=src, dst=dst)] if wide else [Mov(src=src, dst=dst)]
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
            elif isinstance(instr, IRLoad):
                # `address` is always INT64-typed, so this already
                # widens to %rax internally -- reading through it
                # right back into %eax (its own 32-bit alias) is safe
                # even though that clobbers the address: the read
                # happens before the overwrite, in the same
                # instruction, and nothing here needs the address
                # again afterward.
                out.extend(self._gen_load_value(instr.address, Register('eax')))
                if instr.dst.type == Type.STR:
                    out.append(MovQ(src=Memory('rax', 0), dst=Register('rax')))
                else:
                    out.extend(self._gen_read_scalar_into(Memory('rax', 0), instr.dst.type, Register('eax')))
                out.extend(self._gen_write_temp_from(Register('eax'), instr.dst))
            elif isinstance(instr, IRStore):
                # The address and the value need to be alive
                # SIMULTANEOUSLY for the final write, unlike IRLoad --
                # loaded into %r9 and %eax respectively so neither
                # step can clobber the other. %r9, not one of
                # register_allocator.py's own pool registers
                # (%r10d/%r11d/%r15d), deliberately: those CAN be a
                # Temp's actual home, and this runs with no visibility
                # into whether one is live across this exact point --
                # %r9 never persistently holds a value outside a
                # call's own narrow argument-placement window, so
                # nothing else could be relying on it surviving here.
                # instr.value_type -- not instr.value.type -- decides
                # the write's own width, per IRStore's own docstring.
                out.extend(self._gen_load_value(instr.address, Register('r9d')))
                out.extend(self._gen_load_value(instr.value, Register('eax')))
                if instr.value_type == Type.STR:
                    out.append(MovQ(src=Register('rax'), dst=Memory('r9', 0)))
                else:
                    out.extend(self._gen_write_scalar_from(Register('eax'), instr.value_type, Memory('r9', 0)))
            else:
                raise NotImplementedError(f"lower_ir has no rule for: {instr!r}")
        return out
