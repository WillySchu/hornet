"""Composes CodeGenerator from the two lowering-side mixins
(arrays_slices_lowering, scalars_lowering -- see their own module
docstrings) and owns everything that's specific to LOWERING an
already-built IRProgram into an AsmProgram: per-function frame layout
(locals, parameters, escape analysis, and every function's own set of
unconditionally-reserved scratch slots), the prologue/epilogue --
including the callee-saved register save/restore required since
Hornet functions can call each other and each other's string/print/
array machinery -- and the CLI wrappers that chain lexing, parsing,
semantic analysis, IR building, and lowering together.

Building a function's own real IR (arrays_slices, scalars, structs,
strings, statements, dispatch -- what used to be mixed directly into
this class) is ir.builder.IRFunctionBuilder's job now; building the
whole program's IR from every function's own is ir.program_builder.
build_ir_program's -- see their own module docstrings for why, and for
why neither needs a CodeGenerator at all. generate() itself now takes
an already-built IRProgram directly, rather than building one from a
Program AST the way it used to: building and lowering are genuinely
separate steps a caller can run independently now, with a real seam in
between for whatever IR-to-IR transform (optimization passes,
eventually) wants to sit there -- see compile_to_asm's own body for
the shape this takes.
"""


import argparse
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
from codegen.errors import CodegenError
from ir.ir import IRFunction, IRProgram
from ir.builder import IRFunctionBuilder
from ir.program_builder import build_ir_program
from codegen.ir_lowering import InstructionSelector
from codegen.register_allocator import allocate_registers
from codegen.scalars_lowering import ScalarsLoweringMixin
from lexer import lex
from parser import Function, Parser, Program
from semantic import analyze


# ---------------------------------------------------------------------------
# AST -> Assembly AST
# ---------------------------------------------------------------------------

class CodeGenerator(
        ArraysSlicesLoweringMixin,
        ScalarsLoweringMixin):
    """Walks the source AST (Program/Function/Return/Constant/...) and
    produces an equivalent AsmProgram."""

    def __init__(self):
        self._next_offset = 0
        self._slot_offsets: Dict[int, int] = {}  # slot id -> its final, physical offset; see _resolve_frame_layout
        # Legacy access tracking: every offset written to directly, by
        # this function's own parameter-marshaling code (see gen_
        # function_ir's own scalar/str parameter cases below), bypassing
        # the Temp mechanism entirely. Reset in lower_function,
        # alongside _allocation_finalized -- these are frame-relative,
        # so a raw offset value means nothing across a function
        # boundary.
        #
        # Used to also be populated by _local_offset, on the theory
        # that old-style code might read a variable's memory directly,
        # bypassing its own Temp -- moot now that old-style code is
        # gone entirely (confirmed directly: a composite-typed
        # variable's own named-local Temp, the only kind _local_offset
        # -- now _local_slot -- ever concerned itself with, never
        # appears as an operand anywhere in the IR this compiler
        # builds today, so excluding it from register allocation was
        # already excluding something that could never have been
        # eligible in the first place). Parameter marshaling is a
        # narrower but still genuine reason this needs to exist: it
        # writes a scalar/str parameter's own initial value directly
        # into its permanent slot, never through _gen_write_temp_from,
        # since register_allocator.py's own decision for that Temp
        # isn't made yet at that point in the pipeline -- if that
        # Temp were later allocated a register anyway, nothing would
        # ever actually load the real parameter value into it. This
        # will have nothing left to track once parameter marshaling
        # itself is migrated off plain Instructions the same way
        # everything else in this arc has been.
        self._escaped_offsets: set[int] = set()
        # False from the start of lower_function until this function's
        # OWN allocate_registers call has returned -- see ir_lowering.
        # py's _gen_read_temp_into/_gen_write_temp_from for why a
        # named-local Temp's memory fallback needs to know this, not
        # just whether register_allocator.py assigned it a register.
        self._allocation_finalized: bool = False
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
        the exact order they were created -- reproducing what used to
        be a single, uninterrupted "self._next_offset -= width"
        running counter, just computed as one explicit step instead of
        scattered across every individual reservation site. Stores the
        result in self._slot_offsets (slot id -> offset), and leaves
        self._next_offset at its own final value too, for _frame_size
        to read -- both stay on self, unlike ir_fn.slot_widths/slot_
        labels, since neither is ever read outside the ONE lower_
        function call that computes them (see IRFunction's own
        docstring for the distinction: what must survive across
        function boundaries lives on ir_fn; what's purely local to one
        call is fine staying on self).

        Called exactly once per function, by lower_function, right
        after lower_ir returns -- by which point EVERY slot this
        function will ever need is known: the up-front ones (scratch
        slots, parameters, locals, argument-temps), reserved before any
        body IR was built, AND whatever _temp_mem discovered lazily,
        mid-lowering, for anonymous Temps register_allocator.py didn't
        promote to a register (see its own docstring). This IS what
        "deciding layout once, after the entire function -- including
        whatever lowering itself discovers -- is fully built" means:
        nothing anywhere in this compiler ever reads a slot's own
        physical offset before this single call resolves it. gen_
        function_ir's own build phase -- including a named local's own
        Temp (_bind_local/_bind_param, via _temp_at_offset) and every
        IRLocalAddress the body's own statements construct -- only
        ever carries a slot's own LOGICAL id around, never its offset;
        ir_lowering.py's own IRLocalAddress case builds a LeaQFrameSlot
        placeholder rather than a resolved LeaQFrame for the identical
        reason _temp_mem builds a FrameSlot rather than a resolved
        Memory for an anonymous Temp -- both wait for _patch_frame_
        slots, right below, to resolve them, once this call has run."""
        next_offset = 0
        for slot_id, width in ir_fn.slot_widths.items():
            next_offset -= width
            self._slot_offsets[slot_id] = next_offset
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
                    setattr(instr, f.name, Memory('rbp', self._slot_offsets[value.slot]))

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

    def gen_function(self, fn: Function, ir_program: IRProgram) -> AsmFunction:
        """Thin convenience wrapper: builds fn's own IRFunction, then
        immediately lowers it -- see gen_function_ir/lower_function for
        what each half does. Takes ir_program explicitly, same as
        lower_function does, rather than building its own: this
        function alone has no Program AST to build one from (only fn,
        a single Function), so the caller must already have one -- via
        ir.program_builder.build_ir_program, or hand-built directly,
        with at least struct_registry/type_alias_registry/ids set to
        whatever fn's own body needs. generate() itself no longer goes
        through this method at all (see its own updated body): it
        lowers every function in an already-built IRProgram in one
        pass, into the AsmProgram it returns. This method still builds
        and lowers, back to back, for one function at a time, since
        that's still a perfectly correct way to compile a single
        function end to end (nothing about gen_function_ir/lower_
        function's own contract requires the whole-program shape
        generate() now uses) -- it exists for anything that wants
        exactly that, without caring about the IR in between. Every
        existing test that calls compile_to_asm/generate_asm goes
        through generate(), not this method, so it has exactly one
        caller left: itself, from outside this class, if anything ever
        wants it directly."""
        ir_fn = IRFunctionBuilder(ir_program).gen_function_ir(fn)
        return self.lower_function(ir_fn, ir_program)

    def lower_function(self, ir_fn: IRFunction, ir_program: IRProgram) -> AsmFunction:
        """Allocates registers over ir_fn's own body, lowers it, and
        assembles the final AsmFunction -- everything gen_function_ir's
        own docstring says is deliberately NOT captured on IRFunction
        yet: frame layout (_frame_size, computed only now, since
        lowering can still grow it -- see this method's own comment
        below), the epilogue, and the bounds-check panic block. Takes
        ir_program too, not just ir_fn: struct_registry/ids -- read
        below and by InstructionSelector/the lowering mixins, via
        self.ir_program -- live there now, not on self (see this
        module's own updated docstring), and ir_fn's own IRProgram is
        generally a DIFFERENT object than whatever's currently on self
        if this is being called standalone (see gen_function), so it
        can't be assumed to already be self.ir_program the way it
        would be if generate() were this method's only caller. Does
        not take the original Function AST node at all -- ir_fn.name
        already carries fn.name by construction (see gen_function_ir),
        and nothing else here ever needed fn itself, only what gen_
        function_ir already extracted from it. Nothing about HOW any
        of this is computed has changed from before this split
        existed.

        Resets self._bounds_check_fail_labels/_slot_offsets/_escaped_
        offsets/_allocation_finalized here, not in gen_function_ir:
        found as a real bug for the first of these -- this state is
        written and read ENTIRELY during lowering (_get_bounds_check_
        fail_label/_gen_bounds_check_panic_block, both in arrays_
        slices_lowering.py, never touched anywhere during the build
        phase at all), so resetting it in gen_function_ir only ever
        worked by historical accident, back when build and lower ran
        back-to-back for the same function -- generate() building
        every IRFunction first, THEN lowering all of them, breaks that
        accident: every function's own build call would reset this
        before ANY function's own lower_function call ever ran,
        leaving two different functions' own lower_function calls
        sharing the same, never-reset dict in between, each thinking
        it owns whichever fail label the OTHER one already claimed --
        caught by a real duplicate-symbol assembler error, two
        different functions each emitting a `.Lbounds_check_fail_0:`
        label of their own. The other three moved here for the
        identical reason, once building stopped needing a
        CodeGenerator at all (see ir.builder's own module docstring):
        building never read any of them, only reset them, so there was
        never a genuine reason for the build phase to touch them in
        the first place, just historical accident again. Resetting
        state where it's actually used, not wherever it historically
        happened to line up, is the general fix."""
        self.ir_program = ir_program
        self._bounds_check_fail_labels = {}
        self._slot_offsets = {}
        self._escaped_offsets = set()
        self._allocation_finalized = False
        ir = ir_fn.body
        # A named-local Temp is safe to allocate despite is_named_local
        # (see eligible_intervals' own docstring) exactly when this
        # function's own parameter-marshaling code (the only remaining
        # source of _escaped_offsets -- see its own docstring) never
        # wrote to its own offset directly, bypassing this Temp.
        # Harmless to compute over every Temp ever created so far, not
        # just this function's own: allocate_registers below only ever
        # looks a Temp id up if it's already present in THIS function's
        # own intervals, so an unrelated, earlier function's Temp id
        # appearing here too changes nothing.
        safe_named_locals = frozenset(
            temp_id for temp_id, offset in ir_program.ids._temp_offsets.items()
            if offset not in self._escaped_offsets
        )
        self._register_assignment = allocate_registers(ir, safe_named_locals)
        self._allocation_finalized = True
        instructions = []
        # A fresh InstructionSelector per function, not the single,
        # whole-compilation-lifetime one this used to be: _temp_mem's
        # own call to _new_slot needs ir_fn to write onto (see
        # IRFunction's own docstring), and constructing this object
        # fresh, right here, is what lets it hold ir_fn as a genuine,
        # honest field -- set once, at construction, for this one
        # function's own lowering, never repointed at a DIFFERENT
        # function's own ir_fn the way a shared, whole-lifetime
        # attribute on self would risk.
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
        # A void function's own body used to be allowed to fall off
        # the end with no IRReturn on some path, relying on a trailing
        # epilogue appended right here for exactly that case (see
        # ir_lowering.py's own IRReturn case for the epilogue every
        # OTHER path already gets this same way). gen_function_ir now
        # appends an explicit, real IRReturn(None) to the end of every
        # void function's own body unconditionally (see its own
        # docstring), closing that gap at the IR level instead --
        # ir_lowering.py's own IRReturn case already emits the
        # identical epilogue this used to append directly, making this
        # permanently redundant, the same "always_returns already
        # guarantees it" reasoning that already applied to every other
        # function.
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
    """Build, then lower -- two genuinely separate calls now, not one
    (see codegen.py's own module docstring, and ir.program_builder's):
    build_ir_program needs nothing from CodeGenerator at all, so this
    is exactly where a future IR-to-IR transform (an optimization
    pass, say) would run, given ir_program and returning a
    (presumably different) IRProgram of its own, right between these
    two lines."""
    ir_program = build_ir_program(program)
    asm_program = CodeGenerator().generate(ir_program)
    return Emitter(platform=platform).emit(asm_program)


def compile_to_asm(filename: str, platform: str = 'macos') -> str:
    tokens = lex(filename)
    ast = Parser(tokens).parse_program()
    analyze(ast)  # raises SemanticError before any code is generated
    return generate_asm(ast, platform=platform)


def main():
    arg_parser = argparse.ArgumentParser(description='Assembly generator')
    arg_parser.add_argument('file', type=str, help='Source file to compile.')
    arg_parser.add_argument(
        '--platform', choices=['macos', 'linux'], default='macos',
        help="Target platform; affects symbol naming. Default: macos",
    )
    arg_parser.add_argument(
        '-o', '--output', type=str, default=None,
        help='Write assembly to this file instead of stdout.',
    )
    args = arg_parser.parse_args()

    asm = compile_to_asm(args.file, platform=args.platform)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(asm)
    else:
        print(asm, end='')


if __name__ == '__main__':
    main()
