"""Composes CodeGenerator from one mixin per language feature
(arrays_slices, calling_convention, dispatch, scalars, statements,
strings, structs -- see their own module docstrings) and owns
everything that spans all of them: the AST-to-AsmProgram entry point,
per-function frame layout (locals, parameters, escape analysis, and
every function's own set of unconditionally-reserved scratch slots),
the prologue/epilogue -- including the callee-saved register save/
restore now required since Hornet functions can call each other and
each other's string/print/array machinery -- and the CLI wrappers
that chain lexing, parsing, semantic analysis, and codegen together.
"""


import argparse
import dataclasses
from typing import Dict, List, Optional

from codegen.arrays_slices import ArraysSlicesMixin
from codegen.assembly_ast import (
    AsmFunction,
    AsmProgram,
    FrameSlot,
    Imm,
    Instruction,
    Leave,
    Memory,
    MovQ,
    Pop,
    Push,
    Register,
    Ret,
    SubQ,
)
from codegen.calling_convention import CALLEE_SAVED_SCRATCH_REGISTERS, CallingConventionMixin
from codegen.dispatch import DispatchMixin
from codegen.emitter import Emitter
from codegen.errors import CodegenError
from codegen.escape_analysis import analyze_array_escapes, is_heap_allocated
from codegen.ir import IRCall, IRConst, IRCopy, IRFunction, IRLocalAddress, IRProgram, IRReadArgument, IRStore, Temp
from codegen.ir_lowering import InstructionSelector
from codegen.register_allocator import allocate_registers
from codegen.scalars import ScalarsMixin
from codegen.statements import StatementsMixin
from codegen.strings import StringsMixin
from codegen.structs import StructsMixin
from codegen.utils import type_byte_width, type_of, ARG_REGISTERS_64
from lexer import lex
from parser import (
    ArrayLiteral,
    Assign,
    Binary,
    Call,
    ExprStmt,
    Field,
    FieldAssign,
    Function,
    If,
    Index,
    IndexAssign,
    Node,
    Param,
    Parser,
    Program,
    Return,
    Slice,
    Unary,
    VarDecl,
    Variable,
    While,
)
from semantic import analyze, type_from_name, Type, TypeKind, StructInfo


# ---------------------------------------------------------------------------
# AST -> Assembly AST
# ---------------------------------------------------------------------------

class CodeGenerator(
        ArraysSlicesMixin,
        CallingConventionMixin,
        DispatchMixin,
        ScalarsMixin,
        StatementsMixin,
        StringsMixin,
        StructsMixin):
    """Walks the source AST (Program/Function/Return/Constant/...) and
    produces an equivalent AsmProgram."""

    def __init__(self):
        self._instruction_selector = InstructionSelector(self)
        self._label_count = 0
        self._var_slots: Dict[int, int] = {}  # id(VarDecl node) -> its permanent logical slot
        self._next_offset = 0
        # Fresh-id counter for logical frame slots (see _new_slot),
        # parallel to _temp_count below -- globally unique across the
        # whole compilation, the same convention Temp.id already uses,
        # even though a slot's own physical offset is only ever
        # meaningful within the one function that reserved it.
        self._slot_count = 0
        self._slot_widths: Dict[int, int] = {}  # slot id -> its own byte width; reset per function
        self._slot_labels: Dict[int, str] = {}  # slot id -> a debug label; reset per function
        self._slot_offsets: Dict[int, int] = {}  # slot id -> its final, physical offset; see _resolve_frame_layout
        # IR temps (see ir.py): _temp_count is a fresh-id counter for
        # _new_temp, mirroring _label_count. _temp_offsets maps a
        # NAMED-local Temp's id to its permanent, already-resolved
        # physical offset (set eagerly by _temp_at_offset, when the
        # Temp itself is created). _temp_slots maps an ANONYMOUS
        # Temp's id to its own logical slot instead -- populated
        # lazily by ir_lowering.py's _temp_mem, the first time a Temp
        # is actually referenced during lowering, not when it's
        # created; see _new_temp's own docstring for why that split
        # matters, and _temp_mem's own docstring for why an anonymous
        # Temp's slot needs a placeholder (FrameSlot) rather than an
        # immediately-resolved offset the way every other slot in this
        # compiler gets one.
        self._temp_count = 0
        self._temp_offsets: Dict[int, int] = {}
        self._temp_slots: Dict[int, int] = {}
        # Legacy access tracking: every offset written to directly, by
        # this function's own parameter-marshaling code (see gen_
        # function_ir's own scalar/str parameter cases below), bypassing
        # the Temp mechanism entirely. Reset in gen_function_ir,
        # alongside _next_offset -- these are frame-relative, so a raw
        # offset value means nothing across a function boundary.
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
        # False from the start of gen_function until this function's
        # OWN allocate_registers call has returned -- see ir_lowering.
        # py's _gen_read_temp_into/_gen_write_temp_from for why a
        # named-local Temp's memory fallback needs to know this, not
        # just whether register_allocator.py assigned it a register.
        self._allocation_finalized: bool = False
        # Populated once per function, by gen_function, from
        # register_allocator.allocate_registers -- maps a (necessarily
        # anonymous, necessarily IRCall-free) Temp's id to the
        # physical register it lives in instead of a memory slot. See
        # ir_lowering.py's _gen_read_temp_into/_gen_write_temp_from,
        # the only two places that consult it.
        self._register_assignment: Dict[int, str] = {}
        self.scopes: List[Dict[str, tuple]] = []  # name -> (slot, Type), generation-time
        self.loop_labels: List[tuple] = []  # stack of (start_label, end_label), innermost last
        self.string_literals: List[tuple] = []  # (label, content) pairs
        self.type_descriptors: List[tuple] = []  # (label, fields) pairs
        # Set once, at the very start of generate(), from
        # Program.struct_registry (stashed there by semantic.analyze).
        # Declared here defensively (an empty dict, not left unset) so
        # a bug that calls a method needing this before generate() runs
        # fails with a clear "unknown struct" error rather than an
        # AttributeError from nowhere.
        self.struct_registry: Dict[str, StructInfo] = {}
        # Set the same way, from Program.type_alias_registry. Every
        # entry is already a fully-resolved Type by this point --
        # type_from_name just does a single dict lookup with this,
        # never a recursive re-resolution of an alias's own target.
        self.type_alias_registry: Dict[str, Type] = {}
        self._empty_str_label = None  # "" -- str's own zero value; see _get_empty_str_label
        # Lazily created, but with different lifetimes from each other:
        # the fail labels are reset per function (gen_function); the
        # message labels, like the print-related ones above, are cached
        # for the whole compilation. Both are dicts keyed by message
        # text, since a function can trigger more than one distinct
        # bounds-check message.
        self._bounds_check_fail_labels = {}
        self._bounds_check_message_labels = {}
        # Lazily created and cached for the whole compilation, keyed by
        # content -- a general-purpose version of the print-related
        # caches above, for the punctuation/prefix pieces printing an
        # array or slice needs.
        self._static_string_labels = {}
        # Set fresh at the start of every gen_function call, to either
        # None (this function's return type isn't an array) or the
        # Logical slot of the hidden output pointer the caller passed
        # in, if this function has one. Declared here too, defensively,
        # so referencing it before any function has been generated
        # fails with a clear AttributeError rather than silently
        # reading a stale value from a previous instance.
        self._hidden_return_ptr_slot = None
        # This function's own DECLARED return type -- needed by
        # gen_statement_ir's own Return case specifically to
        # disambiguate an ArrayLiteral return value's own dispatch
        # (ARRAY vs SLICE): type_of(stmt.value) is unusable for this,
        # since semantic.py always annotates an ArrayLiteral node by
        # its own literal shape ("N elements of type X"), regardless
        # of what the surrounding context (here, this function's own
        # signature) resolves the overall expression to. See gen_
        # statement_ir's own Return/ArrayLiteral case for the bug this
        # fixed: `return [1, 2, 3]` from a slice-returning function
        # used to segfault because of exactly this ambiguity.
        self._current_return_type = None
        # Set once, at the end of generate() -- see IRProgram's own
        # docstring for why it's built at all despite having no
        # consumer yet. Declared here defensively, same reason as
        # struct_registry/type_alias_registry above: referencing it
        # before generate() has ever run fails with a clear
        # AttributeError-from-None-access rather than one from
        # nowhere.
        self.ir_program: Optional[IRProgram] = None

    def new_label(self, prefix: str) -> str:
        """Returns a fresh, uniquely-numbered local label like
        `.Land_short_0`. Needed because AND/OR/if codegen all emit real
        jump targets, and a program can contain any number of them --
        each one needs a name the assembler won't collide with any
        other."""
        label = f".L{prefix}_{self._label_count}"
        self._label_count += 1
        return label

    def _new_temp(self, t: Type) -> Temp:
        """Allocates a fresh virtual register (see ir.py) of type `t`
        -- id only, no storage decision yet. Where a Temp actually
        lives is v1's own lowering policy, decided lazily on first
        reference by ir_lowering.py's _temp_mem, not here -- this
        method's only job is handing out an identity, the same way
        new_label's only job is handing out a name."""
        temp_id = self._temp_count
        self._temp_count += 1
        return Temp(id=temp_id, type=t)

    def _temp_at_offset(self, t: Type, offset: int) -> Temp:
        """Allocates a fresh virtual register that already has a
        known home -- `offset`, not a freshly-carved one -- registered
        immediately rather than left for _temp_mem to decide lazily.
        Used exactly once per named scalar local/parameter (see
        _bind_local/_bind_param): a variable's own slot is already
        fixed by _collect_locals/_collect_params before this ever
        runs, so there's no lazy decision left to make, and reusing
        that slot -- rather than allocating a second, redundant one --
        is what lets every read and write of that variable, for the
        rest of the function, share one Temp identity. is_named_local
        marks it as such -- see Temp's own docstring for why."""
        temp_id = self._temp_count
        self._temp_count += 1
        self._temp_offsets[temp_id] = offset
        return Temp(id=temp_id, type=t, is_named_local=True)

    def _new_slot(self, width: int, label: str) -> int:
        """Allocates a fresh, logical frame-slot identifier -- the
        _new_temp of frame slots, but for IRLocalAddress's own `slot`
        field rather than a virtual register. Carries no physical
        offset of its own at all yet: every caller here used to
        compute `self._next_offset -= width` and use the result
        directly (as an offset, immediately) -- now it records the
        slot's own width and a human-readable label (for debugging;
        never read by anything else) and returns an opaque id instead,
        deferring the actual offset decision to _resolve_frame_layout.

        Order matters here: _resolve_frame_layout assigns offsets in
        the exact order slots were created in (self._slot_widths is a
        plain dict, so insertion order is preserved), reproducing
        today's own running-counter layout exactly. Every caller below
        that used to reserve a slot with `self._next_offset -= width`
        directly now calls this instead, in the identical order those
        subtractions used to happen in."""
        slot_id = self._slot_count
        self._slot_count += 1
        self._slot_widths[slot_id] = width
        self._slot_labels[slot_id] = label
        return slot_id

    def _resolve_frame_layout(self) -> None:
        """Assigns a final, physical %rbp-relative byte offset to
        every logical slot _new_slot has handed out so far for the
        CURRENT function, in the exact order they were created --
        reproducing what used to be a single, uninterrupted
        "self._next_offset -= width" running counter, just computed as
        one explicit step instead of scattered across every individual
        reservation site. Stores the result in self._slot_offsets
        (slot id -> offset) for gen_function_ir's own parameter-
        marshaling code and ir_lowering.py's own IRLocalAddress case to
        read from, and leaves self._next_offset at its own final value
        here too, for whichever caller needs it next.

        Called TWICE per function, by two different callers, not once:
        first by gen_function_ir, right after every up-front
        reservation (scratch slots, parameters, locals, argument-
        temps) is done, before any body IR is built -- so by the time
        gen_statement_ir's own walk ever constructs an IRLocalAddress,
        every slot it could reference already has a real offset
        sitting in self._slot_offsets. Then again by lower_function,
        right after lower_ir returns -- by which point _temp_mem may
        have called self._new_slot some more, lazily, for anonymous
        Temps register_allocator.py didn't promote to a register (see
        its own docstring), slots the first call couldn't have known
        about yet. Safe to simply call again, not just harmless:
        self._slot_widths is the SAME insertion-ordered dict both
        times, just with more entries appended before the second call
        -- so every slot the first call already resolved gets the
        identical offset back (same width, same position, same
        starting point), and only the newly-appended ones get a real
        offset for the first time. This IS what "deciding layout once,
        after the entire function (including whatever Phase C
        discovers) is built" would look like, functionally -- what's
        still deliberately NOT true yet is that IRLocalAddress waits
        for that second call before it's ever resolved: the first
        call's own result is already used immediately, by parameter
        marshaling and by ir_lowering.py's IRLocalAddress case alike,
        well before lower_ir (and therefore the second call) ever
        runs. Only _temp_mem's own FrameSlot placeholders (see
        _patch_frame_slots, right below) actually wait for the second
        call to mean anything."""
        next_offset = 0
        for slot_id, width in self._slot_widths.items():
            next_offset -= width
            self._slot_offsets[slot_id] = next_offset
        self._next_offset = next_offset

    def _patch_frame_slots(self, instructions: List[Instruction]) -> None:
        """Walks every instruction in `instructions`, replacing any
        FrameSlot operand (see its own docstring) with the equivalent,
        concrete Memory('rbp', ...) operand, now that self._slot_
        offsets covers every slot this function ever needed -- named
        locals, parameters, and compiler scratch slots (resolved
        before body IR was even built), AND whatever _temp_mem
        discovered lazily during lowering itself (resolved by the
        second _resolve_frame_layout call, immediately before this
        runs) -- all at once. Mutates each instruction in place (these
        are ordinary, non-frozen dataclasses) rather than rebuilding
        the list; safe to run over parameter-marshaling instructions
        too, even though none of them ever contain a FrameSlot in
        practice (they're built from an already-resolved offset
        directly, from the very first _resolve_frame_layout call) --
        this doesn't need to know that in advance, since a plain field-
        by-field scan that finds nothing to replace is just a no-op.

        Never needs to look inside the prologue, the epilogue, or the
        bounds-check panic block: none of those ever reference a frame
        slot of any kind (the prologue itself is built entirely after
        this runs, as a local variable in lower_function -- see its
        own ordering)."""
        for instr in instructions:
            for f in dataclasses.fields(instr):
                value = getattr(instr, f.name)
                if isinstance(value, FrameSlot):
                    setattr(instr, f.name, Memory('rbp', self._slot_offsets[value.slot]))

    def generate(self, program: Program) -> AsmProgram:
        # getattr, not direct attribute access: Program.struct_registry
        # is stamped on by semantic.analyze(), not a field the
        # dataclass itself declares -- an AST that skipped analyze()
        # entirely simply won't have it. Matching type_of's own "has no
        # resolved type" defensive check one level up: fail with a
        # clear, actionable CodegenError right here, at the very first
        # thing generate() does, rather than a bare AttributeError from
        # whatever the first struct-registry lookup happens to be.
        if not hasattr(program, 'struct_registry'):
            raise CodegenError(
                "Program has no struct registry -- semantic.analyze() "
                "must run before codegen (see compile_to_asm)"
            )
        self.struct_registry = program.struct_registry
        # Same defensive check, same reason, one registry over.
        if not hasattr(program, 'type_alias_registry'):
            raise CodegenError(
                "Program has no type alias registry -- semantic.analyze() "
                "must run before codegen (see compile_to_asm)"
            )
        self.type_alias_registry = program.type_alias_registry
        # One function at a time, build-then-lower immediately, exactly
        # as gen_function's own two calls already did internally --
        # see lower_function's own docstring for why this can't yet be
        # two separate whole-program passes (build every IRFunction,
        # THEN lower all of them): a lot of per-function state gen_
        # function_ir sets on self (_hidden_return_ptr_slot, _next_
        # offset, and others) gets reset by the NEXT function's own
        # gen_function_ir call, and lower_function still depends on
        # the CURRENT function's own values of it. Calling both here,
        # explicitly, in generate() itself -- rather than via gen_
        # function, which still exists as a convenience wrapper around
        # the identical two calls -- is what makes collecting every
        # IRFunction into a real IRProgram possible at all, without
        # changing this ordering constraint or anything about what
        # gets computed.
        ir_functions = []
        asm_functions = []
        for fn in program.functions:
            ir_fn = self.gen_function_ir(fn)
            ir_functions.append(ir_fn)
            asm_functions.append(self.lower_function(ir_fn))
        # No consumer reads this yet -- see IRProgram's own docstring
        # for why it's built and kept anyway.
        self.ir_program = IRProgram(
            functions=ir_functions, string_literals=self.string_literals, type_descriptors=self.type_descriptors)
        return AsmProgram(
            functions=asm_functions, string_literals=self.string_literals, type_descriptors=self.type_descriptors)

    def gen_function_ir(self, fn: Function) -> IRFunction:
        """Builds this function's own codegen artifacts up through its
        body's real IR -- see IRFunction's own docstring for exactly
        what's gathered here and why, and for what's deliberately NOT
        moved onto it yet (frame layout, register assignment, the
        epilogue). gen_function (this method's own caller, and the
        only one) does everything past that: allocating registers over
        this function's own body, lowering it, and assembling the
        final AsmFunction.

        Nothing about HOW any of this is computed has changed from
        before this split existed -- every line below is identical to
        what gen_function's own first half used to do directly; this
        method exists so that half has a name and a return value of
        its own, not to change its behavior."""
        # Fresh allocator state per function -- offsets are relative to
        # *this* function's own %rbp.
        self._var_slots = {}
        self._argument_temp_slots = {}  # id(ArrayLiteral or Call) -> its permanent logical slot; see _collect_argument_temps
        self._next_offset = 0
        self._slot_widths = {}
        self._slot_labels = {}
        self._slot_offsets = {}
        self._escaped_offsets = set()
        self._allocation_finalized = False
        # No declared return type means Type.VOID, the same internal-
        # only sentinel semantic.py's analyze_function uses.
        return_type = Type.VOID if fn.return_type is None else type_from_name(
            fn.return_type, self.struct_registry, self.type_alias_registry)
        param_types = [type_from_name(p.type, self.struct_registry, self.type_alias_registry) for p in fn.params]

        # Which of this function's array declarations need to be heap-
        # allocated because a slice backed by them might outlive this
        # function's return, regardless of size (see
        # analyze_array_escapes). Computed once, up front, since
        # _collect_params/_collect_locals (below) need to know this to
        # decide how much stack space each declaration's slot takes (8
        # bytes for a heap pointer vs. the array's full width).
        self._escaping_array_ids = analyze_array_escapes(
            fn, param_types, self.struct_registry, self.type_alias_registry)

        # An array- OR slice-typed return needs a hidden pointer -- the
        # caller passes the address to write the result into, as an
        # extra, FIRST argument, shifting every real parameter one
        # register position later. Rather than dedicate a register to
        # it for the whole function (which would need its own save/
        # restore discipline, and would break the callee-saved-register
        # prologue's even-push-count alignment invariant), it just gets
        # its own ordinary stack slot, via the same "reserve a slot,
        # then store the incoming register into it" mechanism every
        # real parameter uses.
        #
        # A slice-typed return uses this SAME mechanism, not a separate
        # one -- a slice's {ptr, len, cap} descriptor is 24 bytes, too
        # wide for any register-return shape this compiler has
        # precedent for, so a Return's Slice case (see gen_statement_
        # ir) is structurally identical to its Array one. This is also
        # what makes forwarding one slice-returning call's result
        # straight out of another free (`return otherFn()`): the same
        # address just gets
        # passed one level deeper, with no intermediate copy.
        self._hidden_return_ptr_slot = None
        self._current_return_type = return_type
        arg_shift = 0
        if return_type.kind in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT):
            self._hidden_return_ptr_slot = self._new_slot(8, "hidden_return_ptr")
            arg_shift = 1

        # A second, 24-byte slot -- reserved unconditionally for EVERY
        # function -- used by _ir_print_call to materialize a slice-
        # typed print() argument's own {ptr, len, cap} triple (an
        # existing slice Variable/Field/Index, or a freshly-produced
        # one -- a Slice production, append, an ordinary Call) so
        # hornet_print always has a real address to read from. Reusing
        # a single shared slot is safe even under arbitrarily deep
        # nesting, since each materialization is fully consumed before
        # any subsequent one can write to it again -- the same way a
        # call stack's frames nest.
        self._unnamed_slice_temp_slot = self._new_slot(24, "unnamed_slice_temp")

        # A third, small (8-byte) scratch slot -- also reserved
        # unconditionally -- used by _ir_print_call to materialize
        # a non-Variable int/bool/str argument (e.g. `print(x + 1)`) so
        # hornet_print always has a real address to read from, the
        # same "one shared slot, safe because each use is fully
        # consumed before the next can start" reasoning as the unnamed-
        # slice slot above. Array/slice/struct print arguments don't
        # need (and can't safely share) a slot like this: those can be
        # arbitrarily large, so print requires a Variable or Index for
        # them instead.
        self._print_scalar_temp_slot = self._new_slot(8, "print_scalar_temp")

        self._collect_params(fn.params)
        self._collect_locals(fn.body)
        # A THIRD pre-pass, alongside the two above: finds every array-
        # or struct-typed function-call argument that has no address of
        # its own -- an ArrayLiteral, a struct literal, or an ordinary
        # array/struct-returning Call used directly as an argument --
        # anywhere in this function's body, however deeply nested, and
        # reserves each its own permanent stack slot up front, sized to
        # fit. See _collect_argument_temps for why this can't reuse the
        # single-shared-slot trick _unnamed_slice_temp_slot relies on.
        self._collect_argument_temps(fn.body)
        # Every slot this function will ever need, up front, is now
        # known -- assign each its own final, physical offset in one
        # shot (see _resolve_frame_layout's own docstring). Everything
        # from here on (parameter marshaling below, and the body's own
        # IR, built further down) reads a slot's own offset out of
        # self._slot_offsets rather than deciding one itself.
        self._resolve_frame_layout()
        self.scopes = [{}]

        # A slice parameter needs THREE consecutive argument-register
        # slots (ptr, len, then cap), not one -- matching how a real C
        # compiler would pass a `struct{void*,long,long}` parameter,
        # the same running-slot-count accounting _gen_call_arguments_
        # into needs on the CALLER side for the same reason.
        param_slots = sum(3 if pt.kind == TypeKind.SLICE else 1 for pt in param_types)
        total_slots = arg_shift + param_slots
        if total_slots > 6:
            raise CodegenError(
                f"Function '{fn.name}' needs {total_slots} argument "
                f"register(s) for its parameters (a slice-typed "
                f"parameter needs 3)"
                + (" plus the hidden array/slice-return pointer" if arg_shift else "")
                + " -- this compiler only supports up to 6 (passed via "
                "registers per the SysV ABI -- stack-passed parameters "
                "aren't implemented)"
            )

        # Every parameter's own initial value -- and, first, the
        # hidden return pointer's own, if this function has one -- is
        # read straight into an ordinary Temp (via IRReadArgument) and
        # processed from there, exactly like a VarDecl's own
        # initializer -- see _ir_param_setup's own docstring for why
        # the old two-pass "stash everything, then process" structure
        # (and its own %rbx-specific trick for a heap-allocated
        # parameter's own pointer) is entirely unnecessary now:
        # register_allocator.py's own general "a Temp surviving live
        # across an IRCall it doesn't own" protection already covers
        # this for free, the same way it already covers everything
        # else.
        param_setup_ir = self._ir_param_setup(fn, param_types, arg_shift)

        self._bounds_check_fail_labels = {}  # fresh, per-function jump targets
        # Accumulated as one IR list for the whole body -- see
        # gen_statement_ir -- and lowered exactly once, by gen_
        # function (this method's own caller) right after this
        # returns, seeing this entire function's Temps and their live
        # ranges together, which is what allocate_registers itself
        # needs: it can only decide which Temps are safe to keep in a
        # register (and for how long) by looking at the whole function
        # at once, not one already-resolved statement at a time.
        statement_ir = []
        for stmt in fn.body:
            statement_ir.extend(self.gen_statement_ir(stmt))
        body = param_setup_ir + statement_ir
        return IRFunction(name=fn.name, body=body, return_type=return_type)

    def _ir_param_setup(self, fn: Function, param_types: List[Type], arg_shift: int) -> list:
        """Builds (without lowering) this function's own real
        parameters -- AND its own hidden return pointer, if it has one
        (always argument slot 0 when present -- see arg_shift's own
        comment above) -- as real IR, in two passes.

        FIRST, every argument slot -- the hidden return pointer's own,
        if present, then every parameter's -- is read into its own
        fresh Temp via IRReadArgument, as ONE uninterrupted block,
        nothing else running in between. This is the one piece of the
        old two-pass "stash everything, then process" structure that's
        still genuinely necessary, for a DIFFERENT reason than the
        original: a physical argument register holds nothing register_
        allocator.py can protect until IRReadArgument actually captures
        it into a Temp -- an ORDINARY scratch-register-using op (an
        IRBinOp computing a field offset for an EARLIER parameter's own
        slice-descriptor write, or IRStore's own %r9d scratch when
        writing the hidden pointer into its own slot, say) can clobber
        a LATER parameter's own still-unread argument register just as
        easily as an IRCall can, since neither is a Temp yet at that
        point. Found as two real, separate bugs, both fixed the same
        way: two slice parameters, the second one's own values
        silently corrupted by the first one's own descriptor-writing
        arithmetic (which uses %rcx as ordinary scratch -- exactly
        argument slot 3's own register); and a hidden-return-pointer
        function with 5 scalar parameters, the fifth one's own value
        silently corrupted by the hidden pointer's own IRStore (which
        uses %r9d as scratch -- exactly argument slot 5's own
        register, the one holding this function's own fifth
        parameter). Reading every argument first, as one uninterrupted
        block, before any processing begins, avoids both regardless of
        how many parameters there are or what any of them need.
        IRReadArgument's own lowering itself only ever uses %eax/%rax
        as scratch -- never any of the six SysV argument registers --
        so this pass is safe internally, for the identical reason.

        SECOND, each parameter is processed using its own Temp(s) from
        the first pass -- exactly like a VarDecl's own initializer
        would be, via whichever existing real-IR building block
        already matches its shape (IRCopy for a stack-allocated array/
        struct's own value copy, an ordinary IRCall to malloc plus
        IRCopy for a heap-allocated one, _ir_write_slice_descriptor_
        into_address for a slice's own three-value alias) -- and,
        FIRST, before any of them, the hidden return pointer, if
        present, is written into its own slot the identical way, via
        IRLocalAddress and an ordinary IRStore. A scalar or str
        parameter has nothing left to do here at all: the first pass's
        own IRReadArgument already targeted its permanent Temp
        directly.

        Once a value is safely captured into a Temp at all (by the
        first pass), register_allocator.py's own ordinary live-range
        tracking -- including surviving live across an IRCall it
        doesn't own (see its own module docstring) -- covers
        everything from there exactly like any other Temp in this
        compiler, with nothing parameter-specific left to reimplement.
        This is what makes the old %rbx-specific trick for holding a
        heap-allocated parameter's own caller-pointer across its own
        malloc call entirely unnecessary: caller_ptr below is an
        ordinary Temp by the time any malloc call could ever run."""
        ir = []
        reg_index = arg_shift
        hidden_ptr = None
        if self._hidden_return_ptr_slot is not None:
            hidden_ptr = self._new_temp(Type.INT64)
            ir.append(IRReadArgument(dst=hidden_ptr, index=0))
        captured = []
        for p, p_type in zip(fn.params, param_types):
            if p_type.kind == TypeKind.SLICE:
                ptr_value = self._new_temp(Type.INT64)
                len_value = self._new_temp(Type.INT)
                cap_value = self._new_temp(Type.INT)
                ir.append(IRReadArgument(dst=ptr_value, index=reg_index))
                ir.append(IRReadArgument(dst=len_value, index=reg_index + 1))
                ir.append(IRReadArgument(dst=cap_value, index=reg_index + 2))
                reg_index += 3
                captured.append((ptr_value, len_value, cap_value))
            elif p_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT):
                caller_ptr = self._new_temp(Type.INT64)
                ir.append(IRReadArgument(dst=caller_ptr, index=reg_index))
                reg_index += 1
                captured.append(caller_ptr)
            else:
                # str and every other scalar type alike: IRReadArgument
                # targets this parameter's own permanent Temp directly
                # -- _bind_param creates it, anchored at this
                # parameter's own resolved slot -- so the second pass
                # below has nothing left to do at all for this one.
                # register_allocator.py treats it exactly like any
                # other Temp-producing op from here on: no more
                # writing directly into memory, bypassing the Temp,
                # the way this used to (see _escaped_offsets' own
                # docstring for why that mattered before, and why it
                # no longer applies to a parameter at all now).
                self._bind_param(p)
                ir.append(IRReadArgument(dst=self._local_temp(p.name), index=reg_index))
                reg_index += 1
                captured.append(None)

        if hidden_ptr is not None:
            hidden_ptr_addr = self._new_temp(Type.INT64)
            ir.append(IRLocalAddress(dst=hidden_ptr_addr, slot=self._hidden_return_ptr_slot))
            ir.append(IRStore(address=hidden_ptr_addr, value=hidden_ptr, value_type=Type.INT64))

        for p, p_type, cap in zip(fn.params, param_types, captured):
            if p_type.kind == TypeKind.SLICE:
                ptr_value, len_value, cap_value = cap
                slot = self._bind_param(p)
                param_addr = self._new_temp(Type.INT64)
                ir.append(IRLocalAddress(dst=param_addr, slot=slot))
                ir.extend(self._ir_write_slice_descriptor_into_address(param_addr, ptr_value, len_value, cap_value))
            elif p_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT):
                caller_ptr = cap
                slot = self._bind_param(p)
                param_addr = self._new_temp(Type.INT64)
                ir.append(IRLocalAddress(dst=param_addr, slot=slot))
                if self._is_heap_allocated(id(p), p_type):
                    size = type_byte_width(p_type, self.struct_registry)
                    new_ptr = self._new_temp(Type.INT64)
                    ir.append(IRCall(dst=new_ptr, name='malloc', args=[IRConst(size, Type.INT64)]))
                    ir.append(IRStore(address=param_addr, value=new_ptr, value_type=Type.INT64))
                    ir.append(IRCopy(dst_address=new_ptr, src_address=caller_ptr, value_type=p_type))
                else:
                    ir.append(IRCopy(dst_address=param_addr, src_address=caller_ptr, value_type=p_type))
            # scalar/str: the first pass already did everything.
        return ir

    def gen_function(self, fn: Function) -> AsmFunction:
        """Thin convenience wrapper: builds fn's own IRFunction, then
        immediately lowers it -- see gen_function_ir/lower_function for
        what each half does. generate() itself no longer goes through
        this method at all (see its own updated body): it calls gen_
        function_ir/lower_function itself, one function at a time, so
        it can also collect the resulting IRFunctions into a real
        IRProgram alongside the AsmProgram it already built. This
        method still exists for anything that wants "just compile one
        function end to end" without caring about the IR in between --
        every existing test that calls compile_to_asm/generate_asm
        goes through generate(), not this method, so it has exactly
        one caller left: itself, from outside this class, if anything
        ever wants it directly."""
        ir_fn = self.gen_function_ir(fn)
        return self.lower_function(ir_fn)

    def lower_function(self, ir_fn: IRFunction) -> AsmFunction:
        """Allocates registers over ir_fn's own body, lowers it, and
        assembles the final AsmFunction -- everything gen_function_ir's
        own docstring says is deliberately NOT captured on IRFunction
        yet: frame layout (_frame_size, computed only now, since
        lowering can still grow it -- see this method's own comment
        below), the epilogue, and the bounds-check panic block. Takes
        only ir_fn, not the original Function AST node at all --
        ir_fn.name already carries fn.name by construction (see gen_
        function_ir), and nothing else here ever needed fn itself,
        only what gen_function_ir already extracted from it. Nothing
        about HOW any of this is computed has changed from before this
        split existed."""
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
            temp_id for temp_id, offset in self._temp_offsets.items()
            if offset not in self._escaped_offsets
        )
        self._register_assignment = allocate_registers(ir, safe_named_locals)
        self._allocation_finalized = True
        instructions = []
        instructions.extend(self._instruction_selector.lower_ir(ir))
        # Every slot this function will EVER need is now known --
        # named locals/parameters/scratch slots/argument-temps
        # (already resolved once, before any body IR was built -- see
        # gen_function_ir's own ordering), plus whatever anonymous
        # Temps lower_ir just discovered, lazily, via _temp_mem, and
        # left as unresolved FrameSlot placeholders (see its own
        # docstring). Calling this again is safe and correct, not
        # merely harmless: self._slot_widths is the SAME insertion-
        # ordered dict as before, just with more entries appended
        # since the first call, so every slot already resolved gets
        # the identical offset back, and only the newly-appended ones
        # get a real offset for the first time.
        self._resolve_frame_layout()
        self._patch_frame_slots(instructions)
        self._register_assignment = {}  # never valid past this function's own body
        if ir_fn.return_type == Type.VOID:
            # A function with no declared return type never has to
            # guarantee every path returns explicitly (see
            # analyze_function's always_returns skip for this case) --
            # its body can legitimately fall off the end, relying on
            # THIS trailing epilogue rather than an IRReturn-emitted
            # one (see ir_lowering.py's own IRReturn case) on every
            # path. Every OTHER function never needs this:
            # always_returns already guarantees some IRReturn-emitted
            # epilogue executes on every path, making a trailing one
            # here permanently unreachable. Without this, a void
            # function that fell off the end would fall straight
            # through into whatever comes next in the generated
            # assembly -- the bounds-check panic block, or the next
            # function's prologue -- a real, silent crash.
            #
            # Appended unconditionally, even when this body already
            # returns explicitly on every path: there's no cheap way to
            # know that without effectively re-running always_returns,
            # and an extra, unreachable epilogue costs nothing but a
            # few bytes.
            instructions.extend(self._gen_epilogue())
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

    def _collect_params(self, params: List[Param]) -> None:
        """Gives each parameter its own logical frame slot, exactly
        like _collect_locals does for VarDecls (same node-identity
        keying) -- kept as a separate method since Param and VarDecl
        are different AST node types, not because parameters need
        fundamentally different treatment. Each slot's width is the
        parameter's actual type width -- 1 byte for int8/uint8, 4 for
        int/bool, 8 for str, and an array's full flattened footprint
        for a stack-allocated array parameter -- except for an array
        parameter over _STACK_ARRAY_LIMIT_BYTES, which only needs 8
        bytes here: its slot holds a pointer to a heap block
        gen_function's parameter loop allocates, not the array's data
        directly. Called after gen_function_ir has already reserved
        the hidden-return-pointer slot, if this function needs one --
        _new_slot's own creation-order guarantee is what keeps this
        placed right after it, matching today's own layout exactly."""
        for p in params:
            p_type = type_from_name(p.type, self.struct_registry, self.type_alias_registry)
            width = 8 if self._is_heap_allocated(id(p), p_type) else type_byte_width(p_type, self.struct_registry)
            self._var_slots[id(p)] = self._new_slot(width, f"param:{p.name}")

    def _bind_param(self, p: Param) -> int:
        """The Param counterpart to _bind_local -- registers `p`'s name
        and declared type (as a real semantic.Type, via type_from_name,
        not the raw parser-level string/ArrayTypeExpr), plus id(p)
        itself, in the current scope, pointing at the logical slot
        _collect_params already assigned it. Also creates `p`'s own
        Temp (see _bind_local's own docstring for why), anchored at
        this slot's own physical offset -- already resolved, since
        _resolve_frame_layout has already run by the time this method
        is ever called (see gen_function_ir's own ordering)."""
        slot = self._var_slots[id(p)]
        p_type = type_from_name(p.type, self.struct_registry, self.type_alias_registry)
        self.scopes[-1][p.name] = (slot, p_type, id(p), self._temp_at_offset(p_type, self._slot_offsets[slot]))
        return slot

    def _collect_locals(self, statements: List[Node]) -> None:
        """Recursively walks `statements`, including into every If's
        then_body/else_body and every While's body, and gives each
        VarDecl found its own permanent stack slot, keyed by the AST
        node's identity rather than its name.

        Each slot's width is the variable's actual type width -- 1
        byte for int8/uint8, 4 for int/bool, 8 for str, and an array's
        full flattened footprint (e.g. 24 bytes for [2][3]int) for a
        stack-allocated array local. An array whose footprint exceeds
        _STACK_ARRAY_LIMIT_BYTES only needs 8 bytes here regardless of
        its real size: its slot holds a pointer to a heap block,
        allocated via _ir_malloc_and_store (see gen_statement_ir's own
        VarDecl case), not the array's data directly -- this is the
        one place that decision changes how much stack space gets
        reserved. No alignment padding is added between
        slots: x86-64 doesn't require aligned access, and %rsp's own
        16-byte alignment requirement is satisfied purely by
        _frame_size rounding the TOTAL frame size up at the end,
        regardless of how the space within it is subdivided."""
        for stmt in statements:
            if isinstance(stmt, VarDecl):
                var_type = type_from_name(stmt.var_type, self.struct_registry, self.type_alias_registry)
                width = 8 if self._is_heap_allocated(
                    id(stmt), var_type) else type_byte_width(var_type, self.struct_registry)
                self._var_slots[id(stmt)] = self._new_slot(width, f"local:{stmt.name}")
            elif isinstance(stmt, If):
                self._collect_locals(stmt.then_body)
                if stmt.else_body is not None:
                    self._collect_locals(stmt.else_body)
            elif isinstance(stmt, While):
                self._collect_locals(stmt.body)

    def _collect_argument_temps(self, statements: List[Node]) -> None:
        """Recursively walks `statements` -- including into every If's
        then_body/else_body and every While's body, like
        _collect_locals -- looking for THREE kinds of array-/struct-
        typed expression with no address of its own: a function-call
        argument (an ArrayLiteral, a struct literal, or an ordinary
        array/struct-returning Call used DIRECTLY as an argument), an
        ordinary composite-returning Call sitting directly at an
        Index.array/Field.base position (`makeArray()[i]`,
        `makePoint().x`), and a bare bracketed-list literal sitting
        directly at an Index.array position (`[1, 2, 3][i]`) -- NOT a
        Slice.array position, for any of these three, and no Field.
        base equivalent for the third (a struct literal at a Field.
        base position, `Point(1,2).x`, is already rejected outright by
        semantic.py, wherever it would appear) -- see _collect_
        argument_temps_in_expr's own Slice case for why that position
        never reserves a slot at all here, unlike Index/Field. Neither
        of the first two is a Variable, Index, or Field, each of which
        already has a real address via _ir_array_address/_ir_struct_
        address.

        Not just ORDINARY function-call arguments, despite the name:
        the walk finds a qualifying argument inside ANY Call node, with
        no check on `expr.name` -- print, len, and append are all
        ordinary Call nodes as far as this pass is concerned, so an
        array-typed literal or returning-call passed to print() already
        gets a slot reserved here. (len's and append's own array/
        struct-typed arguments, if ever a literal or returning-call,
        also get a slot reserved that neither currently reads back out
        -- harmless, just a few unused bytes of frame space.)

        WHY THIS CAN'T REUSE THE SHARED-SLOT TRICK: _unnamed_slice_
        temp_offset gets away with ONE shared, per-function scratch
        slot because a slice's 24-byte descriptor is written and then
        immediately drained into registers. An array or struct argument
        is different in kind: it's passed BY ADDRESS, and that address
        has to keep pointing at valid data right up until the `call`
        instruction executes, since the callee only reads through it
        after control transfer. A single call can have MORE THAN ONE
        such argument at once (`foo([1,2], [3,4])`), and both need to
        be alive simultaneously through the call -- a shared slot would
        let the second one's write clobber the first's before `call`
        ever runs. So each occurrence needs its OWN distinct backing
        storage, discovered ahead of time here, the same way every
        named local already is.

        SIZE THRESHOLD, MATCHING EVERY OTHER ARRAY/STRUCT VALUE: not
        every occurrence found here gets a stack slot -- _reserve_
        argument_temp applies the same is_heap_allocated size check
        every named local/parameter goes through. A small literal or
        returning-call's result gets a real, permanent slot, read back
        out by its own real-IR materialization case (_ir_materialize_
        composite_call/_ir_materialize_array_literal/_ir_materialize_
        struct_literal). A large one gets NO
        slot at all -- it's heap-allocated fresh at the point of the
        call instead, needing no space reserved in this function's
        frame: an argument-temp's pointer is read exactly once, by the
        callee's own entry-time copy, and never again.

        WHY THIS WALKS EXPRESSIONS, NOT JUST STATEMENTS: unlike
        _collect_locals, a literal-or-returning-call-as-argument can be
        buried arbitrarily deep inside another expression entirely
        unrelated to the call itself -- `int x = foo(1) + bar([1, 2,
        3])` -- so this needs a real, general expression walk
        (_collect_argument_temps_in_expr) rather than only inspecting a
        statement's top-level shape."""
        for stmt in statements:
            if isinstance(stmt, VarDecl):
                if stmt.init is not None:
                    self._collect_argument_temps_in_expr(stmt.init)
            elif isinstance(stmt, Assign):
                self._collect_argument_temps_in_expr(stmt.value)
            elif isinstance(stmt, IndexAssign):
                self._collect_argument_temps_in_expr(stmt.array)
                self._collect_argument_temps_in_expr(stmt.index)
                self._collect_argument_temps_in_expr(stmt.value)
            elif isinstance(stmt, FieldAssign):
                self._collect_argument_temps_in_expr(stmt.base)
                self._collect_argument_temps_in_expr(stmt.value)
            elif isinstance(stmt, Return):
                if stmt.value is not None:
                    self._collect_argument_temps_in_expr(stmt.value)
            elif isinstance(stmt, If):
                self._collect_argument_temps_in_expr(stmt.condition)
                self._collect_argument_temps(stmt.then_body)
                if stmt.else_body is not None:
                    self._collect_argument_temps(stmt.else_body)
            elif isinstance(stmt, While):
                self._collect_argument_temps_in_expr(stmt.condition)
                self._collect_argument_temps(stmt.body)
            elif isinstance(stmt, ExprStmt):
                self._collect_argument_temps_in_expr(stmt.expr)
            # Break/Continue carry no expressions at all.

    def _is_ordinary_composite_call(self, expr: Node) -> bool:
        """True for a Call that goes through the ordinary hidden-
        pointer convention (see _ir_composite_call) -- excludes a
        struct-literal Call (construction, not an ordinary call at
        all -- Point(1,2).x needs its own, separate treatment, not
        attempted here) and append (a builtin, never compiled as an
        ordinary function -- calling it via the hidden-pointer
        convention would try to call a symbol literally named
        'append' that was never compiled), matching the identical
        exclusion this arc has applied everywhere else a composite-
        returning Call is distinguished from these two shapes."""
        return isinstance(expr, Call) and expr.name != 'append' and expr.name not in self.struct_registry

    def _collect_argument_temps_in_expr(self, expr: Optional[Node]) -> None:
        """The general expression-tree walk _collect_argument_temps
        needs but _collect_locals never did -- recurses into every
        expression node that can contain another expression (Binary,
        Unary, Index, Field, Slice, ArrayLiteral's elements, a Call's
        arguments), with a leaf case for everything else.

        The actual DECISION -- does this specific Call argument need
        its own reserved slot -- is made only at a Call node: after
        recursing into each of ITS OWN arguments first (so a nested
        call, `foo(bar([1,2,3]))`, is discovered on the way back up),
        any argument that's array- or struct-typed and isn't a
        Variable, Index, or Field gets handed to
        _reserve_argument_temp. A Variable/Index/Field argument is
        skipped -- it already has a real address of its own, so it was
        never a candidate for one of these slots."""
        if expr is None:
            return
        if isinstance(expr, Call):
            for arg in expr.args:
                self._collect_argument_temps_in_expr(arg)
                arg_type = type_of(arg)
                if arg_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT) and not isinstance(arg, (Variable, Index, Field)):
                    self._reserve_argument_temp(arg, arg_type)
        elif isinstance(expr, Binary):
            self._collect_argument_temps_in_expr(expr.left)
            self._collect_argument_temps_in_expr(expr.right)
        elif isinstance(expr, Unary):
            self._collect_argument_temps_in_expr(expr.operand)
        elif isinstance(expr, Index):
            self._collect_argument_temps_in_expr(expr.array)
            self._collect_argument_temps_in_expr(expr.index)
            array_type = type_of(expr.array)
            if array_type.kind in (TypeKind.ARRAY, TypeKind.SLICE) and (
                    self._is_ordinary_composite_call(expr.array) or isinstance(expr.array, ArrayLiteral)):
                self._reserve_argument_temp(expr.array, array_type)
        elif isinstance(expr, Field):
            self._collect_argument_temps_in_expr(expr.base)
            base_type = type_of(expr.base)
            if base_type.kind == TypeKind.STRUCT and self._is_ordinary_composite_call(expr.base):
                self._reserve_argument_temp(expr.base, base_type)
        elif isinstance(expr, Slice):
            self._collect_argument_temps_in_expr(expr.array)
            self._collect_argument_temps_in_expr(expr.low)
            self._collect_argument_temps_in_expr(expr.high)
            # Deliberately NEVER reserves a slot here, unlike the
            # Index/Field cases just above: a slice PRODUCED from this
            # base escapes -- its own ptr aliases whatever backs the
            # base for as long as the slice itself is alive, which can
            # far outlive this statement (assigned to a variable,
            # returned, stored). A stack slot would be unsafe here
            # regardless of size, the same reasoning _is_heap_
            # allocated's own escape-analysis consultation already
            # applies to a NAMED variable that's ever sliced -- except
            # here there's no name to track at all (this base has none
            # of its own), so the decision is made unconditionally
            # rather than via that machinery. See _ir_materialize_
            # composite_call's own docstring for the codegen-time half
            # of this: no reservation found for id(expr.array) is
            # exactly what tells it to malloc instead of using a slot.
        elif isinstance(expr, ArrayLiteral):
            for element in expr.elements:
                self._collect_argument_temps_in_expr(element)
        # Constant/BoolLiteral/StringLiteral/NoneLiteral/Variable: leaves,
        # nothing further to recurse into.

    def _reserve_argument_temp(self, expr: Node, t: Type) -> None:
        """Reserves a logical frame slot for `expr` -- an ArrayLiteral,
        a struct literal, or an ordinary array/struct-returning Call
        used directly as a function-call argument -- keyed by id(expr)
        exactly like _var_slots keys a VarDecl/Param, just for a
        synthetic, unnamed "declaration" with no actual source-level
        variable.

        Skips reservation entirely when `t` is over the same
        is_heap_allocated size threshold every named local/parameter
        uses: a large value gets heap-allocated fresh at the point of
        the call instead, needing no space in this function's frame --
        unlike a large NAMED local's heap pointer, which needs a
        permanent 8-byte slot to survive as long as the variable stays
        in scope, an argument-temp's pointer is read exactly once, by
        the callee's own entry-time copy, and never again.

        Deliberately not routed through _is_heap_allocated (which also
        consults self._escaping_array_ids): an argument-temp is never a
        candidate for escape-driven promotion -- it's never sliced by
        the caller, it flows into the callee as a whole value copied on
        entry -- so only the plain size check ever applies, via
        is_heap_allocated directly."""
        if is_heap_allocated(t, self.struct_registry):
            return
        width = type_byte_width(t, self.struct_registry)
        self._argument_temp_slots[id(expr)] = self._new_slot(width, "argument_temp")

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

    def _push_scope(self) -> None:
        self.scopes.append({})

    def _pop_scope(self) -> None:
        self.scopes.pop()

    def _bind_local(self, stmt: VarDecl) -> int:
        """Registers `stmt`'s name -- its declared type, needed by
        _local_type, and id(stmt) itself, needed by _local_decl_id --
        in the current (innermost) generation-time scope, pointing at
        the logical slot _collect_locals already assigned this exact
        VarDecl node, and returns that slot. Also creates
        `stmt`'s own Temp (see _local_temp), pointed at that same
        slot's own physical offset via _temp_at_offset (already
        resolved by _resolve_frame_layout by the time this runs)
        rather than a freshly-carved one --
        this is what lets a scalar VarDecl-with-initializer or Assign
        (see gen_statement_ir) target this variable with a genuine
        IRMove, and every later read of it (see gen_expr_ir's own
        Variable case) resolve to the identical Temp, rather than each
        read re-emitting its own throwaway copy: the same Temp
        identity, reused for this variable's entire lifetime, is
        exactly what would let a future register allocator consider
        keeping it in a register instead of memory. Composite-typed
        (array/struct/slice) locals get a Temp here too, for
        uniformity, but never actually use it -- nothing in
        arrays_slices.py/structs.py reads or writes through a Temp;
        they address this variable's slot directly, exactly as
        before."""
        slot = self._var_slots[id(stmt)]
        var_type = type_from_name(stmt.var_type, self.struct_registry, self.type_alias_registry)
        self.scopes[-1][stmt.name] = (slot, var_type, id(stmt), self._temp_at_offset(var_type, self._slot_offsets[slot]))
        return slot

    def _local_slot(self, name: str) -> int:
        """Resolves `name` to its own logical frame slot -- for a
        composite-typed (array/struct/slice) variable's own address,
        the only case this is ever called for (see _ir_array_address/
        _ir_struct_address/_ir_slice_address/_ir_indexable_base, its
        four live callers). Used to also record every call here into
        _escaped_offsets, on the theory that old-style code might read
        this variable's memory directly, bypassing its own Temp --
        dropped entirely now that old-style code is gone: confirmed
        directly that a composite-typed variable's own named-local
        Temp never appears as an operand anywhere in the IR this
        compiler builds (every read/write of one goes through
        IRLocalAddress instead, which references this slot directly,
        never the Temp), so excluding it from register allocation was
        already excluding something that could never have been
        eligible in the first place."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name][0]
        raise CodegenError(f"Reference to undeclared variable '{name}'")

    def _local_type(self, name: str) -> Type:
        """Used specifically where a Variable's *slot* is also being
        looked up right alongside it (see _ir_array_address's own
        Variable case) -- both come from the same (slot, Type,
        decl_id, Temp) tuple
        in the same scope-stack entry, which codegen has to maintain
        regardless of type_of's existence, since resolved_type has no
        way to encode *which* stack slot a name refers to. This is
        deliberately not replaced by type_of, even though it gives the
        same answer for a Variable node.

        Returns a real semantic.Type (via type_from_name, called once
        up front in _bind_local/_bind_param) -- not the raw parser-level
        string/ArrayTypeExpr -- so callers can uniformly inspect
        .kind/.element_type/.size exactly like they can on whatever
        type_of returns."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name][1]
        raise CodegenError(f"Reference to undeclared variable '{name}'")

    def _local_decl_id(self, name: str) -> int:
        """Returns id(the VarDecl or Param node) that `name` currently
        resolves to -- the third element of the same (slot, Type,
        decl_id, Temp) tuple _local_slot/_local_type/_local_temp
        read the others of, kept in the SAME scope-stack lookup
        (rather than a separate, parallel name-to-id table)
        specifically so this respects shadowing correctly: Hornet
        allows re-declaring a name in a nested if/while block, so a
        plain name doesn't uniquely identify a declaration the way
        id() of the actual AST node does. Used by _is_heap_allocated
        to look up whether THIS SPECIFIC declaration (not just any
        variable sharing its name) was found to escape by
        analyze_array_escapes."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name][2]
        raise CodegenError(f"Reference to undeclared variable '{name}'")

    def _local_temp(self, name: str) -> Temp:
        """Returns `name`'s own persistent Temp -- the fourth element
        of the same scope-stack tuple, created once by _bind_local/
        _bind_param and reused for every read and write of this
        variable for the rest of its scope (see _bind_local's own
        docstring)."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name][3]
        raise CodegenError(f"Reference to undeclared variable '{name}'")

    def _is_heap_allocated(self, decl_id: int, t: Type) -> bool:
        """Whether the SPECIFIC array- or struct-typed declaration
        identified by decl_id (id() of its own VarDecl or Param node)
        needs to be heap-allocated -- combining is_heap_allocated's
        pure size check (covering both array and struct) with
        analyze_array_escapes's independent result (computed once per
        function, in gen_function, cached in self._escaping_array_ids
        -- an array-specific trigger only, since the terminal backing
        storage a slice descriptor ever points at is always a real
        array, never a struct directly, regardless of whether the
        slice was reached through an array-of-slices container or a
        struct's slice-typed field: either kind of container might
        itself need promoting for size, but never merely because a
        slice somewhere within it escapes): either reason alone is
        sufficient. This is the actual decision point every call site
        that used to call is_heap_allocated directly now goes through
        instead, each passing whichever decl_id it has on hand."""
        return is_heap_allocated(t, self.struct_registry) or decl_id in self._escaping_array_ids

    def _gen_epilogue(self) -> List[Instruction]:
        """The ordinary function epilogue: restore every callee-saved
        scratch register (in reverse of the prologue's push order),
        then leave/ret. Shared by IRReturn's own bare-return lowering
        (see ir_lowering.py -- no value to compute) and lower_function's
        own trailing fall-through case: both are "there's no value to
        compute, just exit the function cleanly" situations. Leave
        resets %rsp straight to %rbp, which was captured before the
        callee-saved registers were pushed in the prologue, so
        anything pushed after that point has to be popped explicitly
        first or it's silently discarded rather than restored."""
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
    asm_program = CodeGenerator().generate(program)
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
