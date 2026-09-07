"""Lowers a list of ir.py instructions into assembly_ast.py
Instructions -- the "instruction selection" step of this pipeline.
v1, deliberately: every Temp gets its own permanent stack slot,
assigned the first time it's referenced here (see _temp_mem; the same
host._next_offset allocator _collect_locals/_reserve_argument_temp
already share) -- allocating a Temp itself (_new_temp, in codegen.py)
makes no storage decision at all. An op that combines two values
loads them into scratch registers, then hands off to the host's own
gen_binary_op/gen_unary_op (ScalarsMixin) as this pass's own
instruction-selection rule -- that arithmetic isn't reimplemented
here. Real register allocation now exists (register_allocator.py) and
hooks in at _gen_read_temp_into/_gen_write_temp_from: _temp_mem's own
permanent-stack-slot policy remains exactly this, used as the
fallback for whichever Temps the allocator didn't promote.

InstructionSelector is a standalone class, not a CodeGenerator mixin
(it was one -- IRLoweringMixin -- until every dependency below was
made an explicit constructor argument instead of an implicit
assumption about whatever else happened to be mixed into a shared
self). `host` is still held and read/written directly for two pieces
of state -- _next_offset and _temp_offsets -- because those are
genuinely shared with CodeGenerator's own stack-slot allocation for
named locals/parameters (see _temp_at_offset): fully separating them
would mean also restructuring that side, a distinct, larger piece of
work this doesn't attempt. Every OTHER dependency (which leaf codegen
methods get called, which read-only program-level data is needed) is
listed here, in one place, rather than discovered by grepping for
self. across the rest of the codebase.
"""

from codegen.assembly_ast import Instruction, Register, Memory, Imm, Mov, MovQ, Cmp, Je, Jmp, Label, CallInstr
from codegen.ir import (
    IRBinOp,
    IRBranch,
    IRCall,
    IRConst,
    IRCopy,
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
from codegen.utils import as_qword_register, type_byte_width, ARG_REGISTERS_32
from semantic import Type


class InstructionSelector:
    """See this module's own docstring for what `host` is and isn't
    used for."""

    def __init__(self, host):
        self.host = host

    def _temp_mem(self, temp: Temp) -> Memory:
        """Returns temp's Memory location, assigning it a permanent
        stack slot the first time it's referenced (memoized in
        host._temp_offsets) rather than at temp-creation time -- see
        _new_temp's own docstring for why storage assignment is kept
        separate from allocating the temp itself. This is the
        fallback every Temp used to rely on unconditionally; now it's
        only reached for one that register_allocator.py didn't (or
        couldn't -- see its own module docstring) promote to a
        physical register -- see _gen_read_temp_into/_gen_write_temp_
        from, the two places that actually decide which applies."""
        if temp.id not in self.host._temp_offsets:
            self.host._next_offset -= type_byte_width(temp.type, self.host.struct_registry)
            self.host._temp_offsets[temp.id] = self.host._next_offset
        return Memory('rbp', self.host._temp_offsets[temp.id])

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
        host._register_assignment, set once per function by
        gen_function), the value is already sitting there, not in
        memory -- this is just a register-to-register move (or, if it
        already happens to BE `dst`, no instruction at all). Otherwise,
        falls back to reading its memory slot: str needs its own case
        (a full 8-byte pointer read via MovQ) since host._gen_read_
        scalar_into (ScalarsMixin) only special-cases int8/uint8/int64,
        falling through to a plain 4-byte Mov for everything else --
        which would truncate a pointer. Every existing caller of that
        method already special-cases str itself first; this is one
        more such caller, not a gap in it.

        A named-local Temp falling back to memory HERE, before host.
        _allocation_finalized is set (see its own docstring), is
        itself a legacy-access hazard, not just the ordinary safe
        case: several old-style codegen methods (gen_binary_into
        among them) build a small, self-contained IR fragment and
        lower it immediately, via this same method, well before this
        function's own whole-body allocate_registers call has run --
        at that point host._register_assignment is still empty for
        THIS function, so any named-local Temp referenced from inside
        one of those fragments falls back to memory unconditionally,
        regardless of what the real, whole-function allocation
        decision later turns out to be. A named-local's own Temp
        identity is shared and reused across every reference to that
        variable, unlike an anonymous Temp (always confined to the
        one IR-building-and-lowering call that created it), so this
        is the one case where an earlier, premature memory fallback
        can go stale the moment the real decision is made -- caught
        empirically (`arr[i + 1] = 42` reading a stale i from memory
        after i's own Temp was allocated a register elsewhere), not by
        reasoning about this method in isolation."""
        reg_name = self.host._register_assignment.get(temp.id)
        if reg_name is not None:
            src = Register(reg_name)
            wide = temp.type in (Type.INT64, Type.STR)
            if wide:
                src, dst = as_qword_register(src), as_qword_register(dst)
            if src == dst:
                return []
            return [MovQ(src=src, dst=dst)] if wide else [Mov(src=src, dst=dst)]
        if temp.is_named_local and not self.host._allocation_finalized:
            self.host._escaped_offsets.add(self.host._temp_offsets[temp.id])
        if temp.type == Type.STR:
            return [MovQ(src=self._temp_mem(temp), dst=as_qword_register(dst))]
        return self.host._gen_read_scalar_into(self._temp_mem(temp), temp.type, dst)

    def _gen_write_temp_from(self, src: Register, temp: Temp) -> list[Instruction]:
        """The write-side counterpart to _gen_read_temp_into -- same
        register-assignment check, same str special case, same legacy-
        access recording for the same reasons."""
        reg_name = self.host._register_assignment.get(temp.id)
        if reg_name is not None:
            dst = Register(reg_name)
            wide = temp.type in (Type.INT64, Type.STR)
            if wide:
                src, dst = as_qword_register(src), as_qword_register(dst)
            if src == dst:
                return []
            return [MovQ(src=src, dst=dst)] if wide else [Mov(src=src, dst=dst)]
        if temp.is_named_local and not self.host._allocation_finalized:
            self.host._escaped_offsets.add(self.host._temp_offsets[temp.id])
        if temp.type == Type.STR:
            return [MovQ(src=as_qword_register(src), dst=self._temp_mem(temp))]
        return self.host._gen_write_scalar_from(src, temp.type, self._temp_mem(temp))

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
                out.extend(self.host.gen_binary_op(
                    instr.op, src=Register('ecx'), dst=Register('eax'), operand_type=instr.left.type))
                out.extend(self._gen_write_temp_from(Register('eax'), instr.dst))
            elif isinstance(instr, IRUnOp):
                out.extend(self._gen_load_value(instr.operand, Register('eax')))
                out.extend(self.host.gen_unary_op(instr.op, Register('eax'), operand_type=instr.operand.type))
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
                # `args` is empty for a call with any slice argument
                # (see _ir_call's own docstring for why) -- marshaled
                # entirely through the pre-existing calling-convention
                # code, spliced in as an IRRaw immediately before this
                # op runs, in that case. Otherwise, each one is placed
                # directly into its own argument register, in any
                # order: no push/pop dance needed, since nothing a
                # Temp could ever be assigned to (memory, or the
                # allocator's own pool) overlaps an argument register,
                # so placing one can never clobber another's source.
                for i, arg_value in enumerate(instr.args):
                    out.extend(self._gen_load_value(arg_value, Register(ARG_REGISTERS_32[i])))
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
                out.extend(self.host._gen_epilogue())
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
                    out.extend(self.host._gen_read_scalar_into(Memory('rax', 0), instr.dst.type, Register('eax')))
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
                    out.extend(self.host._gen_write_scalar_from(Register('eax'), instr.value_type, Memory('r9', 0)))
            elif isinstance(instr, IRCopy):
                # dst_address/src_address both need to be alive
                # simultaneously, so both get their own fixed,
                # never-a-Temp's-actual-home register -- %r9/%r8,
                # extending IRStore's own reasoning for %r9 to a
                # second register here. Pinning both is also what
                # lets gen_array_copy's own dynamic "pick a scratch
                # register that isn't either base" logic collapse to
                # always just %rax: %r9/%r8 can never be an allocated
                # Temp's home, so %rax can never collide with either
                # base the way it could when gen_array_copy is handed
                # two arbitrary, possibly-overlapping Memory operands.
                out.extend(self._gen_load_value(instr.dst_address, Register('r9d')))
                out.extend(self._gen_load_value(instr.src_address, Register('r8d')))
                out.extend(self.host.gen_array_copy(Memory('r9', 0), Memory('r8', 0), instr.value_type))
            else:
                raise NotImplementedError(f"lower_ir has no rule for: {instr!r}")
        return out
