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
from typing import Dict, List, Optional, Tuple, Union

from codegen.arrays_slices import ArraysSlicesMixin
from codegen.assembly_ast import (
    Add,
    AddQ,
    AsmFunction,
    AsmProgram,
    CallInstr,
    Cmp,
    CmpQ,
    Imm,
    IMul,
    Instruction,
    Ja,
    Jae,
    Je,
    Jle,
    Jmp,
    Jne,
    Label,
    LeaQ,
    LeaQFrame,
    Leave,
    Memory,
    Mov,
    MovB,
    MovQ,
    MovZX,
    Operand,
    Pop,
    Push,
    Register,
    Ret,
    SetCC,
    ShiftRightArithmetic,
    Sub,
    SubQ,
)
from codegen.calling_convention import CALLEE_SAVED_SCRATCH_REGISTERS, CallingConventionMixin
from codegen.dispatch import DispatchMixin
from codegen.emitter import Emitter
from codegen.errors import CodegenError
from codegen.escape_analysis import analyze_array_escapes, is_heap_allocated
from codegen.scalars import ScalarsMixin
from codegen.statements import StatementsMixin
from codegen.strings import StringsMixin
from codegen.structs import StructsMixin
from codegen.utils import as_byte_register, leaf_type, type_byte_width, type_of, ARG_REGISTERS_64
from lexer import lex
from parser import (
    ArrayLiteral,
    Assign,
    Binary,
    BinaryOp,
    Call,
    ExprStmt,
    Field,
    FieldAssign,
    Function,
    If,
    Index,
    IndexAssign,
    Node,
    NoneLiteral,
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
        self._label_count = 0
        self._var_offsets: Dict[int, int] = {}  # id(VarDecl node) -> its permanent Memory offset
        self._next_offset = 0
        self.scopes: List[Dict[str, tuple]] = []  # name -> (offset, Type), generation-time
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
        # Lazily created, then cached and reused for the rest of this
        # compilation -- see gen_print_call_into for why these
        # specifically get a small dedicated cache rather than
        # following string_literals' usual "every occurrence gets its
        # own label, no dedup" policy.
        self._true_str_label = None
        self._false_str_label = None
        self._comma_space_label = None  # ", " -- the print machinery's own element/field separator
        self._colon_space_label = None  # ": " -- between a struct field's own name and its value
        self._empty_str_label = None  # "" -- str's own zero value; see _get_empty_str_label
        # Set the first time gen_print_call_into actually runs; checked
        # in generate() to decide whether hornet_stringify needs to be
        # added to the program's function list at all -- a program that
        # never calls print() shouldn't pay for it.
        self._print_used = False
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
        # %rbp offset of the stack slot holding the hidden output
        # pointer the caller passed in. Declared here too, defensively,
        # so referencing it before any function has been generated
        # fails with a clear AttributeError rather than silently
        # reading a stale value from a previous instance.
        self._hidden_return_ptr_offset = None

    def new_label(self, prefix: str) -> str:
        """Returns a fresh, uniquely-numbered local label like
        `.Land_short_0`. Needed because AND/OR/if codegen all emit real
        jump targets, and a program can contain any number of them --
        each one needs a name the assembler won't collide with any
        other."""
        label = f".L{prefix}_{self._label_count}"
        self._label_count += 1
        return label

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
        functions = [self.gen_function(fn) for fn in program.functions]
        if self._print_used:
            functions.append(self.build_stringify_function())
        return AsmProgram(functions=functions, string_literals=self.string_literals, type_descriptors=self.type_descriptors)

    def gen_function(self, fn: Function) -> AsmFunction:
        # Fresh allocator state per function -- offsets are relative to
        # *this* function's own %rbp.
        self._var_offsets = {}
        self._argument_temp_offsets = {}  # id(ArrayLiteral or Call) -> its permanent slot; see _collect_argument_temps
        self._next_offset = 0
        # No declared return type means Type.VOID, the same internal-
        # only sentinel semantic.py's analyze_function uses.
        return_type = Type.VOID if fn.return_type is None else type_from_name(fn.return_type, self.struct_registry, self.type_alias_registry)
        param_types = [type_from_name(p.type, self.struct_registry, self.type_alias_registry) for p in fn.params]

        # Which of this function's array declarations need to be heap-
        # allocated because a slice backed by them might outlive this
        # function's return, regardless of size (see
        # analyze_array_escapes). Computed once, up front, since
        # _collect_params/_collect_locals (below) need to know this to
        # decide how much stack space each declaration's slot takes (8
        # bytes for a heap pointer vs. the array's full width).
        self._escaping_array_ids = analyze_array_escapes(fn, param_types, self.struct_registry, self.type_alias_registry)

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
        # precedent for, so gen_return's Slice case is structurally
        # identical to its Array one. This is also what makes
        # forwarding one slice-returning call's result straight out of
        # another free (`return otherFn()`): the same address just gets
        # passed one level deeper, with no intermediate copy.
        self._hidden_return_ptr_offset = None
        arg_shift = 0
        if return_type.kind in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT):
            self._next_offset -= 8
            self._hidden_return_ptr_offset = self._next_offset
            arg_shift = 1

        # A second, 24-byte slot -- reserved unconditionally for EVERY
        # function -- used by gen_indexable_base_into to materialize an
        # unnamed Slice or slice-returning Call expression's descriptor
        # when it's used directly as the base of a `[...]` chain (e.g.
        # `arr[:][0]`), rather than requiring it be assigned to a named
        # variable first, and by gen_expr_stmt for a bare Slice-
        # expression statement. Reusing a single shared slot is safe
        # even under arbitrarily deep nesting, since each
        # materialization is fully consumed before any subsequent one
        # can write to it again -- the same way a call stack's frames
        # nest.
        self._next_offset -= 24
        self._unnamed_slice_temp_offset = self._next_offset

        # A third, small (8-byte) scratch slot -- also reserved
        # unconditionally -- used by gen_print_call_into to materialize
        # a non-Variable int/bool/str argument (e.g. `print(x + 1)`) so
        # hornet_stringify always has a real address to read from, the
        # same "one shared slot, safe because each use is fully
        # consumed before the next can start" reasoning as the unnamed-
        # slice slot above. Array/slice/struct print arguments don't
        # need (and can't safely share) a slot like this: those can be
        # arbitrarily large, so print requires a Variable or Index for
        # them instead.
        self._next_offset -= 8
        self._print_scalar_temp_offset = self._next_offset

        # A fourth scratch slot (24 bytes) -- the {ptr, len, cap}
        # triple gen_print_call_into's growable buffer lives in.
        # Reserved unconditionally for the same reason: a print()
        # call's buffer setup needs somewhere real to live, and reusing
        # one shared slot is safe by the same non-overlapping-lifetime
        # argument.
        self._next_offset -= 24
        self._print_buf_state_temp_offset = self._next_offset

        # One extra, purely internal temp slot per parameter, used to
        # stash its incoming register value(s) immediately, before any
        # parameter is processed -- see the loop below for why this has
        # to happen up front rather than processing each parameter
        # directly out of its own argument register. 24 bytes for a
        # slice parameter (ptr, len, AND cap each need stashing), 8 for
        # everything else.
        param_temp_offsets = []
        for p_type in param_types:
            width = 24 if p_type.kind == TypeKind.SLICE else 8
            self._next_offset -= width
            param_temp_offsets.append(self._next_offset)

        self._collect_params(fn.params)
        self._collect_locals(fn.body)
        # A THIRD pre-pass, alongside the two above: finds every array-
        # or struct-typed function-call argument that has no address of
        # its own -- an ArrayLiteral, a struct literal, or an ordinary
        # array/struct-returning Call used directly as an argument --
        # anywhere in this function's body, however deeply nested, and
        # reserves each its own permanent stack slot up front, sized to
        # fit. See _collect_argument_temps for why this can't reuse the
        # single-shared-slot trick _unnamed_slice_temp_offset relies on.
        self._collect_argument_temps(fn.body)
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

        instructions: List[Instruction] = [
            Push(Register('rbp')),
            MovQ(src=Register('rsp'), dst=Register('rbp')),
        ]
        # Save every callee-saved scratch register unconditionally, not
        # just in functions that happen to do string work themselves --
        # required now that functions can call each other.
        for reg in CALLEE_SAVED_SCRATCH_REGISTERS:
            instructions.append(Push(Register(reg)))

        frame_size = self._frame_size()
        if frame_size:
            instructions.append(SubQ(src=Imm(frame_size), dst=Register('rsp')))

        if self._hidden_return_ptr_offset is not None:
            instructions.append(MovQ(src=Register('rdi'), dst=Memory('rbp', self._hidden_return_ptr_offset)))

        # Parameters arrive in registers per the SysV ABI (shifted one
        # position later if this function itself returns an array or
        # slice -- see arg_shift above). Handled in two passes rather
        # than reading each one directly out of its own argument
        # register in turn:
        #
        # FIRST, every incoming register is stashed into its own
        # temporary slot (param_temp_offsets, reserved above) via a
        # plain %rbp-relative store -- these never touch %rsp, so
        # there's no stack-alignment concern regardless of parameter
        # count. A slice parameter stashes THREE consecutive registers
        # into its own 24-byte temp slot, advancing the running
        # register-index counter by 3.
        #
        # SECOND, each parameter is processed using its safely-stashed
        # value(s) rather than its original argument register(s). This
        # two-pass structure exists specifically because a heap-
        # allocated array parameter needs its own malloc call to build
        # an independent copy -- and malloc, like any real call, can
        # clobber every caller-saved register, including OTHER, not-
        # yet-processed parameters' incoming values still sitting in
        # their argument registers. Stashing everything first, before
        # any malloc call can run, avoids that regardless of which
        # parameters (if any) end up needing one. (An earlier version
        # tried protecting registers with ordinary push/pop instead --
        # which works for a single value, but breaks down here: popping
        # one parameter's value immediately before processing it leaves
        # a DIFFERENT number of not-yet-popped values on the stack ahead
        # of each parameter's malloc call, misaligning %rsp for roughly
        # half of them. Plain %rbp-relative stores sidestep that
        # failure mode entirely, since they never move %rsp.)
        reg_index = arg_shift
        for i, p_type in enumerate(param_types):
            if p_type.kind == TypeKind.SLICE:
                instructions.append(MovQ(src=Register(ARG_REGISTERS_64[reg_index]), dst=Memory('rbp', param_temp_offsets[i])))
                instructions.append(MovQ(src=Register(ARG_REGISTERS_64[reg_index + 1]), dst=Memory('rbp', param_temp_offsets[i] + 8)))
                instructions.append(MovQ(src=Register(ARG_REGISTERS_64[reg_index + 2]), dst=Memory('rbp', param_temp_offsets[i] + 16)))
                reg_index += 3
            else:
                instructions.append(MovQ(src=Register(ARG_REGISTERS_64[reg_index]), dst=Memory('rbp', param_temp_offsets[i])))
                reg_index += 1

        for i, p in enumerate(fn.params):
            offset = self._bind_param(p)
            p_type = param_types[i]
            temp_offset = param_temp_offsets[i]
            if p_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT):
                if self._is_heap_allocated(id(p), p_type):
                    # Needs its own, independent heap copy -- like the
                    # stack-allocated case below, just backed by
                    # malloc'd memory -- to preserve value semantics:
                    # mutating this parameter must never affect the
                    # caller's own array or struct. %rbx holds the
                    # caller's pointer across the malloc call: it's
                    # callee-saved, so malloc is obligated to preserve
                    # it.
                    instructions.append(MovQ(src=Memory('rbp', temp_offset), dst=Register('rbx')))
                    instructions.extend(self._gen_malloc_array(p_type))
                    instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', offset)))
                    instructions.append(MovQ(src=Register('rax'), dst=Register('r10')))
                    instructions.extend(self.gen_array_copy(Memory('r10', 0), Memory('rbx', 0), p_type))
                else:
                    instructions.append(MovQ(src=Memory('rbp', temp_offset), dst=Register('rbx')))
                    instructions.extend(self.gen_array_copy(Memory('rbp', offset), Memory('rbx', 0), p_type))
            elif p_type.kind == TypeKind.SLICE:
                # A slice parameter is never heap-promoted or copied
                # the way an array is -- it's just an alias, so this
                # only needs to copy the three already-stashed values
                # into its own permanent slot; no malloc, no is_heap_
                # allocated check. The underlying array it points to
                # (if any) is already guaranteed to outlive this call:
                # analyze_array_escapes treats passing a slice as an
                # argument to a user-defined call as escaping, so
                # whatever backs it in the CALLER is already heap-
                # allocated by the time this function starts.
                instructions.append(MovQ(src=Memory('rbp', temp_offset), dst=Register('rax')))
                instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', offset)))
                instructions.append(MovQ(src=Memory('rbp', temp_offset + 8), dst=Register('rax')))
                instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', offset + 8)))
                instructions.append(MovQ(src=Memory('rbp', temp_offset + 16), dst=Register('rax')))
                instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', offset + 16)))
            elif p_type == Type.STR:
                instructions.append(MovQ(src=Memory('rbp', temp_offset), dst=Register('rax')))
                instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', offset)))
            else:
                instructions.extend(self._gen_read_scalar_into(Memory('rbp', temp_offset), p_type, Register('eax')))
                instructions.extend(self._gen_write_scalar_from(Register('eax'), p_type, Memory('rbp', offset)))

        self._bounds_check_fail_labels = {}  # fresh, per-function jump targets
        for stmt in fn.body:
            instructions.extend(self.gen_statement(stmt))
        if return_type == Type.VOID:
            # A function with no declared return type never has to
            # guarantee every path returns explicitly (see
            # analyze_function's always_returns skip for this case) --
            # its body can legitimately fall off the end, relying on
            # THIS trailing epilogue rather than a gen_return-emitted
            # one on every path. Every OTHER function never needs this:
            # always_returns already guarantees some gen_return-emitted
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

        return AsmFunction(name=fn.name, instructions=instructions)

    def _collect_params(self, params: List[Param]) -> None:
        """Gives each parameter its own permanent stack slot, exactly
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
        directly. Called after gen_function has already reserved the
        hidden-return-pointer slot, if this function needs one --
        _next_offset just keeps counting down from wherever it already
        is."""
        for p in params:
            p_type = type_from_name(p.type, self.struct_registry, self.type_alias_registry)
            width = 8 if self._is_heap_allocated(id(p), p_type) else type_byte_width(p_type, self.struct_registry)
            self._next_offset -= width
            self._var_offsets[id(p)] = self._next_offset

    def _bind_param(self, p: Param) -> int:
        """The Param counterpart to _bind_local -- registers `p`'s name
        and declared type (as a real semantic.Type, via type_from_name,
        not the raw parser-level string/ArrayTypeExpr), plus id(p)
        itself, in the current scope, pointing at the permanent offset
        _collect_params already assigned it."""
        offset = self._var_offsets[id(p)]
        self.scopes[-1][p.name] = (offset, type_from_name(p.type, self.struct_registry, self.type_alias_registry), id(p))
        return offset

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
        allocated by gen_var_decl, not the array's data directly --
        this is the one place that decision changes how much stack
        space gets reserved. No alignment padding is added between
        slots: x86-64 doesn't require aligned access, and %rsp's own
        16-byte alignment requirement is satisfied purely by
        _frame_size rounding the TOTAL frame size up at the end,
        regardless of how the space within it is subdivided."""
        for stmt in statements:
            if isinstance(stmt, VarDecl):
                var_type = type_from_name(stmt.var_type, self.struct_registry, self.type_alias_registry)
                width = 8 if self._is_heap_allocated(id(stmt), var_type) else type_byte_width(var_type, self.struct_registry)
                self._next_offset -= width
                self._var_offsets[id(stmt)] = self._next_offset
            elif isinstance(stmt, If):
                self._collect_locals(stmt.then_body)
                if stmt.else_body is not None:
                    self._collect_locals(stmt.else_body)
            elif isinstance(stmt, While):
                self._collect_locals(stmt.body)

    def _collect_argument_temps(self, statements: List[Node]) -> None:
        """Recursively walks `statements` -- including into every If's
        then_body/else_body and every While's body, like
        _collect_locals -- looking for a function-call argument that is
        array- or struct-typed but has no address of its own: an
        ArrayLiteral, a struct literal, or an ordinary array/struct-
        returning Call used DIRECTLY as an argument -- as opposed to a
        Variable, Index, or Field, each of which already has a real
        address via gen_array_address_into/gen_struct_address_into.

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
        out by _gen_materialize_argument_temp_into. A large one gets NO
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
        elif isinstance(expr, Field):
            self._collect_argument_temps_in_expr(expr.base)
        elif isinstance(expr, Slice):
            self._collect_argument_temps_in_expr(expr.array)
            self._collect_argument_temps_in_expr(expr.low)
            self._collect_argument_temps_in_expr(expr.high)
        elif isinstance(expr, ArrayLiteral):
            for element in expr.elements:
                self._collect_argument_temps_in_expr(element)
        # Constant/BoolLiteral/StringLiteral/NoneLiteral/Variable: leaves,
        # nothing further to recurse into.

    def _reserve_argument_temp(self, expr: Node, t: Type) -> None:
        """Reserves a permanent stack slot for `expr` -- an ArrayLiteral,
        a struct literal, or an ordinary array/struct-returning Call
        used directly as a function-call argument -- keyed by id(expr)
        exactly like _var_offsets keys a VarDecl/Param, just for a
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
        self._next_offset -= width
        self._argument_temp_offsets[id(expr)] = self._next_offset

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
        the permanent offset _collect_locals already assigned this
        exact VarDecl node, and returns that offset."""
        offset = self._var_offsets[id(stmt)]
        self.scopes[-1][stmt.name] = (offset, type_from_name(stmt.var_type, self.struct_registry, self.type_alias_registry), id(stmt))
        return offset

    def _local_offset(self, name: str) -> int:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name][0]
        raise CodegenError(f"Reference to undeclared variable '{name}'")

    def _local_type(self, name: str) -> Type:
        """Used specifically where a Variable's *offset* is also being
        looked up right alongside it (see gen_expr_into's Variable case)
        -- both come from the same (offset, Type, decl_id) tuple in the
        same scope-stack entry, which codegen has to maintain
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
        resolves to -- the third element of the same (offset, Type,
        decl_id) tuple _local_offset/_local_type read the first two of,
        kept in the SAME scope-stack lookup (rather than a separate,
        parallel name-to-id table) specifically so this respects
        shadowing correctly: Hornet allows re-declaring a name in a
        nested if/while block, so a plain name doesn't uniquely
        identify a declaration the way id() of the actual AST node
        does. Used by _is_heap_allocated to look up whether THIS
        SPECIFIC declaration (not just any variable sharing its name)
        was found to escape by analyze_array_escapes."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name][2]
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
        then leave/ret. Shared by gen_return's bare-return case (no
        value to compute) and gen_function's trailing fall-through
        case: both are "there's no value to compute, just exit the
        function cleanly" situations. Leave resets %rsp straight to
        %rbp, which was captured before the callee-saved registers were
        pushed in the prologue, so anything pushed after that point has
        to be popped explicitly first or it's silently discarded rather
        than restored."""
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
