"""IRFunctionBuilder builds one function's own real IR -- everything
gen_function_ir/gen_statement_ir/gen_expr_ir and the rest of this
arc's own IR-building mixins (arrays_slices, scalars, structs,
strings, statements, dispatch -- see their own module docstrings) used
to do as methods directly on CodeGenerator. Constructed fresh per
function: `ir_program` is a genuine field, set once at construction,
used to reach whole-program state this class doesn't own itself
(ir_program.struct_registry, ir_program.ids, ir_program.string_
literals, and the rest -- see each mixin's own docstring for which,
and IRProgram's own docstring for why it holds them). No CodeGenerator
involved at all, unlike this arc's own earlier shape: building a
function's own IR needs nothing lowering-specific (frame layout,
register assignment), so nothing here ever needed one -- see ir.
program_builder's own module docstring for the build/lower split this
enables.

scopes -- and everything else gen_function_ir used to reset at the top
of every call (loop_labels, _argument_temp_slots, _escaping_array_ids,
the two unconditionally-reserved scratch slots) -- is a genuine field
here instead, for the same reason ir_fn became one on InstructionSelector:
a fresh instance starting blank is a stronger guarantee than a shared,
whole-compilation-lifetime attribute manually reset at the right
moment, and there is no other object left un-shared enough to hold it
faithfully. generate()'s own docstring already confirms none of this
state is ever read past its own function's own build call -- a fresh
instance per function was already exactly how it behaved, just
enforced by discipline rather than by construction.

Composes the six IR-building mixins directly, the same way CodeGenerator
itself used to -- see codegen.py's own updated docstring for what it
keeps instead (frame layout, register allocation, lowering, and every
mixin instruction-selection produces real Instructions from)."""

from typing import List, Optional

from codegen.errors import CodegenError
from codegen.escape_analysis import analyze_array_escapes, is_heap_allocated
from codegen.utils import type_byte_width, type_of
from ir.ir import (
    IRBranch, IRCall, IRConst, IRCopy, IRFunction, IRJump, IRLocalAddress, IRReadArgument, IRReturn, IRStore, Temp,
)
from ir.arrays_slices import ArraysSlicesMixin
from ir.dispatch import DispatchMixin
from ir.scalars import ScalarsMixin
from ir.statements import StatementsMixin
from ir.strings import StringsMixin
from ir.structs import StructsMixin
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
    Return,
    Slice,
    Unary,
    VarDecl,
    Variable,
    While,
)
from semantic import type_from_name, Type, TypeKind


class IRFunctionBuilder(
        ArraysSlicesMixin,
        DispatchMixin,
        ScalarsMixin,
        StatementsMixin,
        StringsMixin,
        StructsMixin):
    """See this module's own docstring for what `host` is and isn't
    used for, and why scopes/loop_labels live here as genuine fields
    rather than on host."""

    def __init__(self, ir_program):
        self.ir_program = ir_program
        self.loop_labels: List[tuple] = []  # stack of (start_label, end_label), innermost last

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
        # id(ArrayLiteral or Call) -> its permanent logical slot,
        # fresh per function; see _collect_argument_temps.
        self._argument_temp_slots = {}
        # Constructed here, early, rather than at the very end the way
        # it used to be built up from local variables -- ir_fn.slot_
        # widths/slot_labels need somewhere to live from _new_slot's
        # own very first call onward (see IRFunction's own docstring
        # for why self can't be that place anymore), and body is
        # simply assigned onto it once it's known, right before this
        # method's own return.
        ir_fn = IRFunction(name=fn.name)
        # No declared return type means Type.VOID, the same internal-
        # only sentinel semantic.py's analyze_function uses. Assigned
        # onto ir_fn immediately, not at this method's own end the way
        # body is -- _ir_hidden_return_ptr (see gen_statement_ir's own
        # Return case) needs to read this back well before this method
        # returns (see IRFunction's own docstring for why that's a
        # small, contained move, unlike scopes/_escaping_array_ids/
        # _argument_temp_slots).
        ir_fn.return_type = Type.VOID if fn.return_type is None else type_from_name(
            fn.return_type, self.ir_program.struct_registry, self.ir_program.type_alias_registry)
        return_type = ir_fn.return_type
        param_types = [type_from_name(p.type, self.ir_program.struct_registry, self.ir_program.type_alias_registry) for p in fn.params]

        # Which of this function's array declarations need to be heap-
        # allocated because a slice backed by them might outlive this
        # function's return, regardless of size (see
        # analyze_array_escapes). Computed once, up front, since
        # _collect_params/_collect_locals (below) need to know this to
        # decide how much stack space each declaration's slot takes (8
        # bytes for a heap pointer vs. the array's full width).
        self._escaping_array_ids = analyze_array_escapes(
            fn, param_types, self.ir_program.struct_registry, self.ir_program.type_alias_registry)

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
        arg_shift = 0
        if return_type.kind in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT):
            ir_fn.hidden_return_ptr_slot = self.ir_program.ids.new_slot(8, "hidden_return_ptr", ir_fn)
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
        self._unnamed_slice_temp_slot = self.ir_program.ids.new_slot(24, "unnamed_slice_temp", ir_fn)

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
        self._print_scalar_temp_slot = self.ir_program.ids.new_slot(8, "print_scalar_temp", ir_fn)

        self._collect_params(fn.params, ir_fn)
        self._collect_locals(fn.body, ir_fn)
        # A THIRD pre-pass, alongside the two above: finds every array-
        # or struct-typed function-call argument that has no address of
        # its own -- an ArrayLiteral, a struct literal, or an ordinary
        # array/struct-returning Call used directly as an argument --
        # anywhere in this function's body, however deeply nested, and
        # reserves each its own permanent stack slot up front, sized to
        # fit. See _collect_argument_temps for why this can't reuse the
        # single-shared-slot trick _unnamed_slice_temp_slot relies on.
        self._collect_argument_temps(fn.body, ir_fn)
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
        param_setup_ir = self._ir_param_setup(fn, param_types, arg_shift, ir_fn)

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
            statement_ir.extend(self.gen_statement_ir(stmt, ir_fn))
        ir_fn.body = param_setup_ir + statement_ir
        ir_fn.return_type = return_type
        if not ir_fn.body or not isinstance(ir_fn.body[-1], (IRBranch, IRJump, IRReturn)):
            # Two genuinely different reasons this can be true, both
            # closed the identical way:
            #
            # A function with no declared return type never has to
            # guarantee every path returns explicitly (see
            # analyze_function's own always_returns skip for this
            # case), so unlike every other function, this one's own
            # body can genuinely fall off the end with no IRReturn
            # anywhere on some path -- an ordinary VarDecl or Assign,
            # say, as the very last real statement.
            #
            # Every OTHER function DOES have that guarantee -- but
            # guaranteeing a RETURN happens somewhere on every path is
            # not the same as guaranteeing body's own last op is one:
            # an If/While as the very last statement builds its own
            # trailing IRLabel (if_end/while_end) as its own last op,
            # regardless of whether every branch inside it already
            # returns -- that label is only ever REACHED by jumping
            # into it from inside, never by falling out the bottom of
            # the function, but it still needs SOME terminator
            # syntactically following it, the identical reason
            # _ir_while_head's own leading IRJump exists: the IR itself
            # has to be structurally valid regardless of which paths
            # are reachable at runtime, not just whichever paths this
            # particular check happens to reason about.
            #
            # Appended unconditionally rather than only where actually
            # reachable -- there's no cheap way to know a given path
            # doesn't need this without effectively re-running always_
            # returns, and an extra, unreachable IRReturn (whatever
            # this function's own real return type -- ir_lowering.py's
            # own IRReturn case tolerates a None value regardless,
            # loading nothing before the epilogue either way) costs
            # nothing but a few bytes once lowered. Closes the last way
            # this compiler's own real IR could still fall off the end
            # of a function with no terminator at all -- see ir.verify's
            # own module docstring for why every other such gap (this
            # one's own sibling cases, if/while's own interior labels)
            # was closed the identical way.
            ir_fn.body.append(IRReturn(value=None))
        return ir_fn

    def _ir_param_setup(self, fn: Function, param_types: List[Type], arg_shift: int, ir_fn: IRFunction) -> list:
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
        if ir_fn.hidden_return_ptr_slot is not None:
            hidden_ptr = self.ir_program.ids.new_temp(Type.INT64)
            ir.append(IRReadArgument(dst=hidden_ptr, index=0))
        captured = []
        for p, p_type in zip(fn.params, param_types):
            if p_type.kind == TypeKind.SLICE:
                ptr_value = self.ir_program.ids.new_temp(Type.INT64)
                len_value = self.ir_program.ids.new_temp(Type.INT)
                cap_value = self.ir_program.ids.new_temp(Type.INT)
                ir.append(IRReadArgument(dst=ptr_value, index=reg_index))
                ir.append(IRReadArgument(dst=len_value, index=reg_index + 1))
                ir.append(IRReadArgument(dst=cap_value, index=reg_index + 2))
                reg_index += 3
                captured.append((ptr_value, len_value, cap_value))
            elif p_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT):
                caller_ptr = self.ir_program.ids.new_temp(Type.INT64)
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
                self._bind_param(p, ir_fn)
                ir.append(IRReadArgument(dst=self._local_temp(p.name), index=reg_index))
                reg_index += 1
                captured.append(None)

        if hidden_ptr is not None:
            hidden_ptr_addr = self.ir_program.ids.new_temp(Type.INT64)
            ir.append(IRLocalAddress(dst=hidden_ptr_addr, slot=ir_fn.hidden_return_ptr_slot))
            ir.append(IRStore(address=hidden_ptr_addr, value=hidden_ptr, value_type=Type.INT64))

        for p, p_type, cap in zip(fn.params, param_types, captured):
            if p_type.kind == TypeKind.SLICE:
                ptr_value, len_value, cap_value = cap
                slot = self._bind_param(p, ir_fn)
                param_addr = self.ir_program.ids.new_temp(Type.INT64)
                ir.append(IRLocalAddress(dst=param_addr, slot=slot))
                ir.extend(self._ir_write_slice_descriptor_into_address(param_addr, ptr_value, len_value, cap_value))
            elif p_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT):
                caller_ptr = cap
                slot = self._bind_param(p, ir_fn)
                param_addr = self.ir_program.ids.new_temp(Type.INT64)
                ir.append(IRLocalAddress(dst=param_addr, slot=slot))
                if self._is_heap_allocated(id(p), p_type):
                    size = type_byte_width(p_type, self.ir_program.struct_registry)
                    new_ptr = self.ir_program.ids.new_temp(Type.INT64)
                    ir.append(IRCall(dst=new_ptr, name='malloc', args=[IRConst(size, Type.INT64)]))
                    ir.append(IRStore(address=param_addr, value=new_ptr, value_type=Type.INT64))
                    ir.append(IRCopy(dst_address=new_ptr, src_address=caller_ptr, value_type=p_type))
                else:
                    ir.append(IRCopy(dst_address=param_addr, src_address=caller_ptr, value_type=p_type))
            # scalar/str: the first pass already did everything.
        return ir


    def _collect_params(self, params: List[Param], ir_fn: IRFunction) -> None:
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
        placed right after it, matching today's own layout exactly.
        `ir_fn` is threaded through purely to hand to _new_slot -- see
        IRFunction's own docstring for why that's explicit now rather
        than implicit self state."""
        for p in params:
            p_type = type_from_name(p.type, self.ir_program.struct_registry, self.ir_program.type_alias_registry)
            width = 8 if self._is_heap_allocated(id(p), p_type) else type_byte_width(p_type, self.ir_program.struct_registry)
            ir_fn.var_slots[id(p)] = self.ir_program.ids.new_slot(width, f"param:{p.name}", ir_fn)

    def _bind_param(self, p: Param, ir_fn: IRFunction) -> int:
        """The Param counterpart to _bind_local -- registers `p`'s name
        and declared type (as a real semantic.Type, via type_from_name,
        not the raw parser-level string/ArrayTypeExpr), plus id(p)
        itself, in the current scope, pointing at the logical slot
        _collect_params already assigned it. Also creates `p`'s own
        Temp (see _bind_local's own docstring for why), anchored at
        that SAME logical slot -- not yet a resolved offset at all:
        _resolve_frame_layout doesn't run until lower_function, well
        after this method (and every other piece of this function's
        own body IR) is done being built."""
        slot = ir_fn.var_slots[id(p)]
        p_type = type_from_name(p.type, self.ir_program.struct_registry, self.ir_program.type_alias_registry)
        self.scopes[-1][p.name] = (slot, p_type, id(p), self.ir_program.ids.temp_at_offset(p_type, slot))
        return slot

    def _collect_locals(self, statements: List[Node], ir_fn: IRFunction) -> None:
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
        regardless of how the space within it is subdivided. `ir_fn`
        is threaded through purely to hand to _new_slot, including on
        every recursive call here -- see IRFunction's own docstring
        for why that's explicit now rather than implicit self state."""
        for stmt in statements:
            if isinstance(stmt, VarDecl):
                var_type = type_from_name(stmt.var_type, self.ir_program.struct_registry, self.ir_program.type_alias_registry)
                width = 8 if self._is_heap_allocated(
                    id(stmt), var_type) else type_byte_width(var_type, self.ir_program.struct_registry)
                ir_fn.var_slots[id(stmt)] = self.ir_program.ids.new_slot(width, f"local:{stmt.name}", ir_fn)
            elif isinstance(stmt, If):
                self._collect_locals(stmt.then_body, ir_fn)
                if stmt.else_body is not None:
                    self._collect_locals(stmt.else_body, ir_fn)
            elif isinstance(stmt, While):
                self._collect_locals(stmt.body, ir_fn)

    def _collect_argument_temps(self, statements: List[Node], ir_fn: IRFunction) -> None:
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
                    self._collect_argument_temps_in_expr(stmt.init, ir_fn)
            elif isinstance(stmt, Assign):
                self._collect_argument_temps_in_expr(stmt.value, ir_fn)
            elif isinstance(stmt, IndexAssign):
                self._collect_argument_temps_in_expr(stmt.array, ir_fn)
                self._collect_argument_temps_in_expr(stmt.index, ir_fn)
                self._collect_argument_temps_in_expr(stmt.value, ir_fn)
            elif isinstance(stmt, FieldAssign):
                self._collect_argument_temps_in_expr(stmt.base, ir_fn)
                self._collect_argument_temps_in_expr(stmt.value, ir_fn)
            elif isinstance(stmt, Return):
                if stmt.value is not None:
                    self._collect_argument_temps_in_expr(stmt.value, ir_fn)
            elif isinstance(stmt, If):
                self._collect_argument_temps_in_expr(stmt.condition, ir_fn)
                self._collect_argument_temps(stmt.then_body, ir_fn)
                if stmt.else_body is not None:
                    self._collect_argument_temps(stmt.else_body, ir_fn)
            elif isinstance(stmt, While):
                self._collect_argument_temps_in_expr(stmt.condition, ir_fn)
                self._collect_argument_temps(stmt.body, ir_fn)
            elif isinstance(stmt, ExprStmt):
                self._collect_argument_temps_in_expr(stmt.expr, ir_fn)
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
        return isinstance(expr, Call) and expr.name != 'append' and expr.name not in self.ir_program.struct_registry

    def _collect_argument_temps_in_expr(self, expr: Optional[Node], ir_fn: IRFunction) -> None:
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
                self._collect_argument_temps_in_expr(arg, ir_fn)
                arg_type = type_of(arg)
                if arg_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT) and not isinstance(arg, (Variable, Index, Field)):
                    self._reserve_argument_temp(arg, arg_type, ir_fn)
        elif isinstance(expr, Binary):
            self._collect_argument_temps_in_expr(expr.left, ir_fn)
            self._collect_argument_temps_in_expr(expr.right, ir_fn)
        elif isinstance(expr, Unary):
            self._collect_argument_temps_in_expr(expr.operand, ir_fn)
        elif isinstance(expr, Index):
            self._collect_argument_temps_in_expr(expr.array, ir_fn)
            self._collect_argument_temps_in_expr(expr.index, ir_fn)
            array_type = type_of(expr.array)
            if array_type.kind in (TypeKind.ARRAY, TypeKind.SLICE) and (
                    self._is_ordinary_composite_call(expr.array) or isinstance(expr.array, ArrayLiteral)):
                self._reserve_argument_temp(expr.array, array_type, ir_fn)
        elif isinstance(expr, Field):
            self._collect_argument_temps_in_expr(expr.base, ir_fn)
            base_type = type_of(expr.base)
            if base_type.kind == TypeKind.STRUCT and self._is_ordinary_composite_call(expr.base):
                self._reserve_argument_temp(expr.base, base_type, ir_fn)
        elif isinstance(expr, Slice):
            self._collect_argument_temps_in_expr(expr.array, ir_fn)
            self._collect_argument_temps_in_expr(expr.low, ir_fn)
            self._collect_argument_temps_in_expr(expr.high, ir_fn)
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
                self._collect_argument_temps_in_expr(element, ir_fn)
        # Constant/BoolLiteral/StringLiteral/NoneLiteral/Variable: leaves,
        # nothing further to recurse into.

    def _reserve_argument_temp(self, expr: Node, t: Type, ir_fn: IRFunction) -> None:
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
        if is_heap_allocated(t, self.ir_program.struct_registry):
            return
        width = type_byte_width(t, self.ir_program.struct_registry)
        self._argument_temp_slots[id(expr)] = self.ir_program.ids.new_slot(width, "argument_temp", ir_fn)


    def _push_scope(self) -> None:
        self.scopes.append({})

    def _pop_scope(self) -> None:
        self.scopes.pop()

    def _bind_local(self, stmt: VarDecl, ir_fn: IRFunction) -> int:
        """Registers `stmt`'s name -- its declared type, needed by
        _local_type, and id(stmt) itself, needed by _local_decl_id --
        in the current (innermost) generation-time scope, pointing at
        the logical slot _collect_locals already assigned this exact
        VarDecl node, and returns that slot. Also creates
        `stmt`'s own Temp (see _local_temp), pointed at that same
        logical slot via _temp_at_offset -- not yet a resolved offset
        at all: _resolve_frame_layout doesn't run until lower_function,
        well after this VarDecl (and every other one in this function's
        own body) is done being built -- rather than a freshly-carved
        one --
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
        slot = ir_fn.var_slots[id(stmt)]
        var_type = type_from_name(stmt.var_type, self.ir_program.struct_registry, self.ir_program.type_alias_registry)
        self.scopes[-1][stmt.name] = (slot, var_type, id(stmt), self.ir_program.ids.temp_at_offset(var_type, slot))
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
        return is_heap_allocated(t, self.ir_program.struct_registry) or decl_id in self._escaping_array_ids

