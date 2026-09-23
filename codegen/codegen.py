"""Composes CodeGenerator from the two lowering-side mixins
(arrays_slices_lowering, scalars_lowering -- see their own module
docstrings) and owns everything specific to LOWERING an already-built
IRProgram into an AsmProgram: per-function frame layout (locals,
parameters, escape analysis, and every function's own set of
unconditionally-reserved scratch slots), the prologue/epilogue --
including the callee-saved register save/restore required since
Hornet functions can call each other and each other's string/print/
array machinery -- and the CLI wrappers that chain lexing, parsing,
semantic analysis, IR building, and lowering together.

Building a function's own real IR (arrays_slices, scalars, structs,
strings, statements, dispatch) is ir.builder.IRFunctionBuilder's job;
building the whole program's IR from every function's own is
ir.program_builder.build_ir_program's -- see their own module
docstrings for why neither needs a CodeGenerator at all. generate()
itself takes an already-built IRProgram directly: building and
lowering are genuinely separate steps a caller can run independently,
with a real seam in between for optimize() -- see compile_to_asm's own
body for the shape this takes.
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
from codegen.calling_convention import CALLEE_SAVED_SCRATCH_REGISTERS
from codegen.emitter import Emitter
from ir.ir import IRCall, IRFunction, IRProgram
from ir.builder import IRFunctionBuilder
from ir.program_builder import build_ir_program
from optimize.optimizer import optimize
from codegen.ir_lowering import InstructionSelector
from codegen.register_allocator import allocate_registers
from codegen.scalars_lowering import ScalarsLoweringMixin
from parser import Function, Parser, Program


# ---------------------------------------------------------------------------
# AST -> Assembly AST
# ---------------------------------------------------------------------------

class CodeGenerator(
        ArraysSlicesLoweringMixin,
        ScalarsLoweringMixin):
    """Walks the IR and produces an equivalent AsmProgram."""

    def __init__(self):
        self._next_offset = 0
        self._slot_offsets: Dict[int, int] = {}  # slot id -> its final, physical offset; see _resolve_frame_layout
        # Populated once per function, by lower_function, from
        # register_allocator.allocate_registers -- maps a (necessarily
        # anonymous, necessarily IRCall-free) Temp's id to the
        # physical register it lives in instead of a memory slot. See
        # ir_lowering.py's _gen_read_temp_into/_gen_write_temp_from,
        # the only two places that consult it.
        self._register_assignment: Dict[int, str] = {}
        # Lazily created, but with different lifetimes from each other:
        # the fail labels are reset per function (lower_function); the
        # message labels are cached for the whole compilation. Both
        # are dicts keyed by message text, since a function can
        # trigger more than one distinct bounds-check message.
        self._bounds_check_fail_labels = {}
        self._bounds_check_message_labels = {}
        # Set by generate()/gen_function/lower_function, from whatever
        # IRProgram the caller passes in -- never built here (see this
        # module's own updated docstring for why). Declared here
        # defensively (None, not left unset) so a bug that reads this
        # before any of those has ever run fails with a clear
        # AttributeError-from-None-access rather than one from
        # nowhere.
        self.ir_program: Optional[IRProgram] = None

    def _resolve_frame_layout(self, ir_fn: IRFunction) -> None:
        """Assigns a final, physical %rbp-relative byte offset to
        every logical slot _new_slot has handed out for `ir_fn`, in
        the exact order they were created. Stores the result in self.
        _slot_offsets (slot id -> offset), and leaves self._next_
        offset at its own final value too, for _frame_size to read --
        both stay on self, unlike ir_fn.slot_widths/slot_labels, since
        neither is ever read outside the ONE lower_function call that
        computes them.

        Called exactly once per function, by lower_function, right
        after lower_ir returns -- by which point EVERY slot this
        function will ever need is known: the up-front ones (scratch
        slots, parameters, locals, argument-temps), reserved before any
        body IR was built, AND whatever _temp_mem discovered lazily,
        mid-lowering, for anonymous Temps register_allocator.py didn't
        promote to a register. Nothing anywhere in this compiler ever
        reads a slot's own physical offset before this single call
        resolves it: gen_function_ir's own build phase only ever
        carries a slot's own LOGICAL id around, never its offset;
        ir_lowering.py's own IRLocalAddress case builds a LeaQFrameSlot
        placeholder rather than a resolved LeaQFrame for the identical
        reason _temp_mem builds a FrameSlot rather than a resolved
        Memory for an anonymous Temp -- both wait for _patch_frame_
        slots, right below, to resolve them, once this call has run.

        ir_fn.outgoing_stack_args_slot, when this function reserved
        one (lower_function does this itself, before lower_ir runs --
        see its own comment), gets special, always-LAST treatment
        here, skipped in the ordinary loop below and placed explicitly
        afterward -- regardless of where it actually falls in slot_
        widths' own insertion order, which is EARLIER than whatever
        _temp_mem discovers for anonymous Temps during lower_ir. This
        one slot has to be the physically last thing in the frame,
        immediately adjacent to %rsp: a callee reads its own first
        overflow argument at a fixed 16(%rbp), which is only correct
        if THIS function's own %rsp, at the moment of the call, points
        exactly at this region's own first byte -- one byte off (from
        ordinary slots landing below it instead of above) and every
        overflow argument the callee reads is simply wrong, silently.

        One more offset this region alone needs that no other slot
        does: TRUE %rsp, at the moment of any call this function
        makes, is NOT %rbp - frame_size -- it's %rbp - frame_size
        minus the 8*len(CALLEE_SAVED_SCRATCH_REGISTERS) bytes the
        prologue's own unconditional callee-saved pushes already
        consumed, between %rbp being set and %rsp being adjusted by
        frame_size (see gen_function's own prologue-building code).
        Every ORDINARY slot's own %rbp-relative address is correct
        regardless of this gap -- nothing else is ever read via %rsp
        directly -- so this region alone needs its own recorded offset
        shifted that much deeper than frame_size's own computation
        would otherwise place it. This shift needs no alignment
        accounting of its own: %rbp is always 16-byte aligned (a
        standard SysV fact, given the caller's own 16-byte-aligned
        %rsp before its own call instruction), and the callee-saved
        push total is itself already a multiple of 16, so frame_size's
        own alignment (via the padding below) already guarantees TRUE
        %rsp lands 16-byte aligned too.

        The explicit padding below exists for the identical reason
        (frame_size's own 16-byte rounding) discussed above:
        _frame_size's own 16-byte rounding adds any padding it needs
        BELOW the last slot resolved here, which would otherwise land
        between this region and %rsp -- so the padding needed is
        computed and inserted HERE, ABOVE this region, guaranteeing
        the running total is already a multiple of 16 by the time this
        region itself is added, leaving nothing for _frame_size's own
        rounding to add."""
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
        """Walks every instruction in `instructions`, resolving every
        remaining reference to a logical slot into its own concrete,
        physical one, now that self._slot_offsets covers every slot
        this function will ever need -- named locals, parameters,
        compiler scratch slots, AND whatever _temp_mem discovered
        lazily during lowering itself -- all at once, resolved
        together by the single _resolve_frame_layout call immediately
        before this runs.

        Two different shapes of "not yet resolved" exist, needing two
        different kinds of replacement here:

        A FrameSlot (see its own docstring) is an Operand, substituted
        directly into whatever field held it -- typically a Mov/MovQ's
        own src or dst -- with an equivalent, concrete Memory('rbp',
        ...) operand. Found via a plain, generic field-by-field scan
        (dataclasses.fields), not by checking specific instruction
        shapes: whatever field happens to hold one gets replaced, and a
        field that never does is simply never touched.

        A LeaQFrameSlot (see its own docstring) is not an Operand at
        all -- it stands in for a WHOLE instruction, since LeaQFrame's
        own `offset` field is a plain int, with nowhere a FrameSlot
        could be substituted into without lying about its own declared
        type. This is resolved by replacing the WHOLE instruction, in
        place in the list, with an equivalent, concrete LeaQFrame.

        Mutates in place (an ordinary FrameSlot substitution) or
        replaces in place (a LeaQFrameSlot's own whole-instruction
        substitution) rather than rebuilding the list. Safe to run
        over parameter-marshaling instructions too, even though none of
        them ever contain either shape in practice (a parameter's own
        addressing goes through ordinary real IR now, resolved the
        identical way as everything else in body) -- this doesn't need
        to know that in advance, since a scan that finds nothing to
        replace is just a no-op.

        Never needs to look inside the prologue, the epilogue, or the
        bounds-check panic block: none of those ever reference a frame
        slot of any kind (the prologue itself is built entirely after
        this runs, as a local variable in lower_function -- see its
        own ordering)."""
        for i, instr in enumerate(instructions):
            if isinstance(instr, LeaQFrameSlot):
                instructions[i] = LeaQFrame(offset=self._slot_offsets[instr.slot], dst=instr.dst)
                continue
            for f in dataclasses.fields(instr):
                value = getattr(instr, f.name)
                if isinstance(value, FrameSlot):
                    setattr(instr, f.name, Memory('rbp', self._slot_offsets[value.slot] + value.extra_offset))

    def generate(self, ir_program: IRProgram) -> AsmProgram:
        """Purely a lowering step now: takes an already-built
        IRProgram (see ir.program_builder's own module docstring for
        why building it is a genuinely separate call, not something
        this method does itself anymore) and lowers every function in
        it. Two genuinely separate passes even within that: every
        function's own IRFunction already exists, complete, before ANY
        of them is lowered -- nothing here interleaves building and
        lowering the way old-style code once had to."""
        self.ir_program = ir_program
        asm_functions = [self.lower_function(ir_fn, ir_program) for ir_fn in ir_program.functions]
        return AsmProgram(
            functions=asm_functions,
            string_literals=ir_program.string_literals,
            type_descriptors=ir_program.type_descriptors,
        )

    def lower_function(self, ir_fn: IRFunction, ir_program: IRProgram) -> AsmFunction:
        """Allocates registers over ir_fn's own body, lowers it, and
        assembles the final AsmFunction: frame layout (_frame_size,
        computed only now, since lowering can still grow it), the
        epilogue, and the bounds-check panic block. Takes ir_program
        too, not just ir_fn: struct_registry/ids -- read below and by
        InstructionSelector/the lowering mixins, via self.ir_program
        -- live there, not on self, and ir_fn's own IRProgram is
        generally a DIFFERENT object than whatever's currently on self
        if this is being called standalone (see gen_function), so it
        can't be assumed to already be self.ir_program. Does not take
        the original Function AST node: ir_fn.name already carries
        fn.name by construction, and nothing else here ever needed fn
        itself.

        Resets self._bounds_check_fail_labels/_slot_offsets here, not
        in gen_function_ir: this state is written and read entirely
        during lowering, never touched during the build phase at all
        -- since generate() builds every IRFunction first, then lowers
        all of them, resetting this per-function, in lower_function,
        is what keeps two different functions' own lowering from
        sharing the same, never-reset dict."""
        self.ir_program = ir_program
        self._bounds_check_fail_labels = {}
        self._slot_offsets = {}
        ir = ir_fn.body

        # Reserve outgoing-stack-argument space, if this function makes
        # any call needing more than 6 argument slots, BEFORE lower_ir
        # runs: IRCall's own lowering (inside lower_ir, right below)
        # needs a real slot id to build a FrameSlot placeholder around
        # for each overflow argument, and every IRCall this function
        # will ever make -- with its own final, flat args list, a slice
        # argument's 3 components already among them -- already exists
        # in `ir` right now, unlowered. Sized to the WORST call this
        # function makes, not the sum across all of them: calls happen
        # one after another, never concurrently, so one call's own
        # overflow arguments can safely reuse the same bytes a later
        # (or earlier) call's own overflow arguments used.
        #
        # _resolve_frame_layout gives ir_fn.outgoing_stack_args_slot
        # special, always-LAST treatment regardless of where this
        # reservation actually lands in slot_widths' own insertion
        # order -- which matters, because it's earlier than whatever
        # _temp_mem discovers lazily, mid-lowering, for anonymous Temps
        # register_allocator.py doesn't promote to a register. This
        # slot must be the one physically closest to %rsp (so its own
        # first 8 bytes are exactly what a callee reads at 16(%rbp)),
        # and insertion order alone can't guarantee that here.
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
        # A fresh InstructionSelector per function: _temp_mem's own
        # call to _new_slot needs ir_fn to write onto, and constructing
        # this object fresh, right here, is what lets it hold ir_fn as
        # a genuine field, never repointed at a different function's
        # own ir_fn.
        instructions.extend(InstructionSelector(self, ir_fn).lower_ir(ir))
        # Every slot this function will EVER need is now known -- named
        # locals, parameters, scratch slots, and argument-temps, alike
        # with whatever anonymous Temps lower_ir just discovered,
        # lazily, via _temp_mem. None of them were resolved before
        # this point -- see _resolve_frame_layout's own docstring for
        # why this single call is the only place any of them ever get
        # a real, physical offset at all.
        self._resolve_frame_layout(ir_fn)
        self._patch_frame_slots(instructions)
        self._register_assignment = {}  # never valid past this function's own body
        # Every function's own body ends in a real IRReturn somewhere
        # reachable (gen_function_ir guarantees this, appending an
        # explicit IRReturn(None) to a void function with no explicit
        # one), and ir_lowering.py's own IRReturn case already emits
        # the epilogue -- so there's no separate "fell off the end"
        # case left for anything else to handle here.
        instructions.extend(self._gen_bounds_check_panic_block())

        # Built here, not in gen_function_ir, since this needs nothing
        # from that build phase at all: identical for every function
        # regardless of what it computes, parameterized only by
        # frame_size -- itself only known now, once body is fully
        # lowered -- so there was never a real reason for the build
        # phase to hand this back as part of IRFunction in the first
        # place.
        prologue: List[Instruction] = [
            Push(Register('rbp')),
            MovQ(src=Register('rsp'), dst=Register('rbp')),
        ]
        # Save every callee-saved scratch register unconditionally, not
        # just in functions that happen to do string work themselves --
        # required now that functions can call each other.
        for reg in CALLEE_SAVED_SCRATCH_REGISTERS:
            prologue.append(Push(Register(reg)))

        frame_size = self._frame_size()
        if frame_size:
            prologue.append(SubQ(src=Imm(frame_size), dst=Register('rsp')))

        return AsmFunction(name=ir_fn.name, instructions=prologue + instructions)

    def _frame_size(self) -> int:
        # Total bytes used by locals and parameters, rounded up to a
        # 16-byte boundary. Genuinely required: gen_string_*/gen_
        # call_into both emit real `call` instructions, and the SysV
        # ABI requires %rsp to be 16-byte-aligned at every one of them.
        # (The 4 callee-saved register pushes in the prologue don't
        # need accounting for here -- an already-even number of 8-byte
        # pushes never changes whether %rsp ends up aligned.)
        raw = -self._next_offset
        return ((raw + 15) // 16) * 16 if raw > 0 else 0

    def _gen_epilogue(self) -> List[Instruction]:
        """The ordinary function epilogue: restore every callee-saved
        scratch register (in reverse of the prologue's push order),
        then leave/ret. Called from exactly one place now -- IRReturn's
        own lowering (see ir_lowering.py), on every return this
        compiler's own IR ever produces, bare or with a value alike:
        gen_function_ir guarantees every function's own body ends in a
        real IRReturn somewhere reachable, void or not (see its own
        docstring), so there's no separate "fell off the end with no
        IRReturn at all" case left for anything else to handle
        anymore. Leave resets %rsp straight to %rbp, which was
        captured before the callee-saved registers were pushed in the
        prologue, so anything pushed after that point has to be popped
        explicitly first or it's silently discarded rather than
        restored."""
        instructions = []
        for reg in reversed(CALLEE_SAVED_SCRATCH_REGISTERS):
            instructions.append(Pop(Register(reg)))
        instructions.append(Leave())
        instructions.append(Ret())
        return instructions


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------

def generate_asm(program: Program, platform: str = 'macos') -> str:
    """Build, optimize, then lower -- three genuinely separate calls
    now (see codegen.py's own module docstring, and ir.program_
    builder's/optimize.optimizer's own): build_ir_program needs
    nothing from CodeGenerator, and optimize needs nothing from either
    build_ir_program or CodeGenerator -- just the IRProgram itself."""
    ir_program = build_ir_program(program)
    ir_program = optimize(ir_program)
    asm_program = CodeGenerator().generate(ir_program)
    return Emitter(platform=platform).emit(asm_program)
