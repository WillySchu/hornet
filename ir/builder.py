"""IRFunctionBuilder builds one function's own real IR -- everything
gen_function_ir/gen_statement_ir/gen_expr_ir and the rest of this
package's own IR-building mixins (arrays_slices, scalars, structs,
strings, statements, dispatch -- see their own module docstrings) live
on. Constructed fresh per function: `ir_program` is a field, set once
at construction, used to reach whole-program state this class doesn't
own itself (ir_program.struct_registry, ir_program.ids, ir_program.
string_literals, and the rest -- see each mixin's own docstring for
which). Building a function's own IR needs nothing lowering-specific
(frame layout, register assignment), so this class has no
CodeGenerator dependency at all.

scopes, loop_labels, _argument_temp_slots, _escaping_decl_ids, and the
two unconditionally-reserved scratch slots are genuine fields here,
reset fresh per function -- none of this state is ever read past its
own function's own build call.

Composes the six IR-building mixins directly -- see codegen.py's own
docstring for what CodeGenerator keeps instead (frame layout, register
allocation, lowering)."""

from typing import List, Optional

from codegen.escape_analysis import analyze_array_escapes, is_heap_allocated
from ir.errors import IRError
from ir.utils import COMPOSITE_KINDS, type_byte_width, type_of
from ir.ir import (
    IRBranch, IRCall, IRConst, IRCopy, IRFunction, IRJump, IRLocalAddress, IRReadArgument, IRReturn, IRStore, Temp,
)
from ir.arrays_slices import ArraysSlicesMixin
from ir.dispatch import DispatchMixin
from ir.pointers import PointersMixin
from ir.scalars import ScalarsMixin
from ir.statements import StatementsMixin
from ir.strings import StringsMixin
from ir.structs import StructsMixin
from ir.sum_types import SumTypesMixin
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
    UnaryOp,
    VarDecl,
    Variable,
    While,
)
from semantic import type_from_name, Type, TypeKind


class IRFunctionBuilder(
        ArraysSlicesMixin,
        DispatchMixin,
        PointersMixin,
        ScalarsMixin,
        StatementsMixin,
        StringsMixin,
        StructsMixin,
        SumTypesMixin):

    def __init__(self, ir_program):
        self.ir_program = ir_program
        self.loop_labels: List[tuple] = []  # stack of (start_label, end_label), innermost last

    def gen_function_ir(self, fn: Function) -> IRFunction:
        """Builds this function's own codegen artifacts up through its
        body's real IR -- see IRFunction's own docstring for what's
        gathered here and what's deliberately not (frame layout,
        register assignment, the epilogue). gen_function (this
        method's own caller) does everything past that."""
        # id(ArrayLiteral or Call) -> its permanent logical slot,
        # fresh per function; see _collect_argument_temps.
        self._argument_temp_slots = {}
        ir_fn = IRFunction(name=fn.name)
        ir_fn.return_type = Type.VOID if fn.return_type is None else type_from_name(
            fn.return_type, self.ir_program.struct_registry, self.ir_program.type_alias_registry, sum_types=self.ir_program.sum_type_registry)
        return_type = ir_fn.return_type
        param_types = [type_from_name(p.type, self.ir_program.struct_registry, self.ir_program.type_alias_registry, sum_types=self.ir_program.sum_type_registry) for p in fn.params]

        # Which of this function's declarations (of ANY type -- a
        # pointer's own target isn't restricted to arrays the way a
        # slice's own backing storage always is) need to be heap-
        # allocated because a slice or pointer backed by them might
        # outlive this function's return, regardless of size (see
        # analyze_array_escapes) -- needed before _collect_params/
        # _collect_locals decide each slot's own byte width.
        self._escaping_decl_ids = analyze_array_escapes(
            fn, param_types, self.ir_program.struct_registry, self.ir_program.type_alias_registry, self.ir_program.sum_type_registry)

        # An array/slice/struct-typed return needs a hidden pointer --
        # the caller passes the address to write the result into, as
        # an extra, FIRST argument, shifting every real parameter one
        # register position later. Given its own ordinary stack slot
        # rather than a dedicated register, via the same "reserve a
        # slot, then store the incoming register into it" mechanism
        # every real parameter uses -- a dedicated register would need
        # its own save/restore discipline and would break the
        # callee-saved-register prologue's even-push-count alignment.
        arg_shift = 0
        if return_type.kind in COMPOSITE_KINDS:
            ir_fn.hidden_return_ptr_slot = self.ir_program.ids.new_slot(8, "hidden_return_ptr", ir_fn)
            arg_shift = 1

        # A second, 24-byte slot, reserved unconditionally for every
        # function -- used by _ir_print_call to materialize a slice-
        # typed print() argument's own {ptr, len, cap} triple so
        # hornet_print always has a real address to read from. Safe to
        # share across arbitrarily deep nesting: each materialization
        # is fully consumed before any subsequent one writes to it
        # again, the same way a call stack's frames nest.
        self._unnamed_slice_temp_slot = self.ir_program.ids.new_slot(24, "unnamed_slice_temp", ir_fn)

        # A third, 8-byte scratch slot, also reserved unconditionally
        # -- used by _ir_print_call for a non-Variable scalar argument
        # (`print(x + 1)`), the same shared-slot reasoning as above.
        # Array/slice/struct arguments can't safely share a slot like
        # this (arbitrarily large), so print requires a Variable or
        # Index for those instead.
        self._print_scalar_temp_slot = self.ir_program.ids.new_slot(8, "print_scalar_temp", ir_fn)

        self._collect_params(fn.params, ir_fn)
        self._collect_locals(fn.body, ir_fn)
        self._collect_argument_temps(fn.body, ir_fn)
        self.scopes = [{}]

        # No slot-count limit enforced here anymore: a parameter (or
        # the hidden array/slice-return pointer) beyond the 6th slot
        # is read from the caller's own stack region by IRReadArgument's
        # own lowering (index >= 6 there), not a register -- see its
        # own docstring. _ir_param_setup below needs no change either
        # way: it just keeps counting register/slot indices upward
        # regardless of how many there are.
        param_setup_ir = self._ir_param_setup(fn, param_types, arg_shift, ir_fn)

        statement_ir = []
        for stmt in fn.body:
            statement_ir.extend(self.gen_statement_ir(stmt, ir_fn))
        ir_fn.body = param_setup_ir + statement_ir
        ir_fn.return_type = return_type
        if not ir_fn.body or not isinstance(ir_fn.body[-1], (IRBranch, IRJump, IRReturn)):
            # Two reasons this can be true, both closed the same way:
            # a function with no declared return type can genuinely
            # fall off the end with no IRReturn on some path (an
            # ordinary VarDecl/Assign as the last real statement); and
            # an If/While as the very last statement builds its own
            # trailing IRLabel as its own last op regardless of
            # whether every branch inside it already returns -- that
            # label is only ever reached by jumping in, never by
            # falling out the bottom, but still needs a terminator
            # syntactically following it. Appended unconditionally
            # rather than only where reachable -- an extra,
            # unreachable IRReturn costs a few bytes once lowered and
            # nothing else.
            ir_fn.body.append(IRReturn(value=None))
        return ir_fn

    def _ir_param_setup(self, fn: Function, param_types: List[Type], arg_shift: int, ir_fn: IRFunction) -> list:
        """Builds (without lowering) this function's own real
        parameters -- and its own hidden return pointer, if it has one
        (always argument slot 0 when present) -- as real IR, in two
        passes.

        FIRST, every argument slot is read into its own fresh Temp via
        IRReadArgument, as one uninterrupted block, nothing else
        running in between: a physical argument register holds nothing
        register_allocator.py can protect until IRReadArgument actually
        captures it into a Temp, so an ordinary scratch-register-using
        op elsewhere in this setup (an IRBinOp computing an earlier
        parameter's own slice-descriptor offset, say) could otherwise
        clobber a LATER parameter's own still-unread argument register.
        IRReadArgument's own lowering only ever uses %eax/%rax as
        scratch, never any of the six SysV argument registers, so this
        pass is safe internally too.

        SECOND, each parameter is processed using its own Temp(s) from
        the first pass -- exactly like a VarDecl's own initializer --
        via whichever real-IR building block matches its shape (IRCopy
        for a stack-allocated array/struct/sum-typed copy, malloc plus
        IRCopy for a heap-allocated one, _ir_write_slice_descriptor_
        into_address for a slice's three-value alias), and, first, the
        hidden return
        pointer, if present, is written into its own slot via
        IRLocalAddress and IRStore. A scalar or str parameter has
        nothing left to do here: the first pass already targeted its
        permanent Temp directly."""
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
            elif p_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT, TypeKind.SUM):
                caller_ptr = self.ir_program.ids.new_temp(Type.INT64)
                ir.append(IRReadArgument(dst=caller_ptr, index=reg_index))
                reg_index += 1
                captured.append(caller_ptr)
            else:
                # str and every other scalar type alike: IRReadArgument
                # reads the incoming value into a fresh Temp first,
                # then _ir_finish_scalar_var_decl (identical helper
                # VarDecl's own initializer uses -- a parameter's own
                # incoming value is exactly a first write, the same
                # kind VarDecl's own init is) decides where it actually
                # ends up: straight into this parameter's own permanent
                # Temp for the ordinary case, or malloc'd into a fresh
                # box (with the permanent Temp repointed at that box's
                # own address) for one whose own address escapes --
                # never IRReadArgument targeting the permanent Temp
                # directly for that second case, since it now holds a
                # pointer, not p_type's own raw value.
                self._bind_param(p, ir_fn)
                incoming = self.ir_program.ids.new_temp(p_type)
                ir.append(IRReadArgument(dst=incoming, index=reg_index))
                ir.extend(self._ir_finish_scalar_var_decl(p.name, id(p), p_type, incoming))
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
            elif p_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT, TypeKind.SUM):
                caller_ptr = cap
                slot = self._bind_param(p, ir_fn)
                param_addr = self.ir_program.ids.new_temp(Type.INT64)
                ir.append(IRLocalAddress(dst=param_addr, slot=slot))
                if self._is_heap_allocated(id(p), p_type):
                    size = type_byte_width(p_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
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
        like _collect_locals does for VarDecls. Each slot's width is
        the parameter's actual type width, except an array parameter
        over _STACK_ARRAY_LIMIT_BYTES, which only needs 8 bytes here:
        its slot holds a pointer to a heap block allocated elsewhere,
        not the array's data directly. Called after gen_function_ir
        has already reserved the hidden-return-pointer slot, if
        needed."""
        for p in params:
            p_type = type_from_name(p.type, self.ir_program.struct_registry, self.ir_program.type_alias_registry, sum_types=self.ir_program.sum_type_registry)
            width = 8 if self._is_heap_allocated(id(p), p_type) else type_byte_width(p_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
            ir_fn.var_slots[id(p)] = self.ir_program.ids.new_slot(width, f"param:{p.name}", ir_fn)

    def _bind_param(self, p: Param, ir_fn: IRFunction) -> int:
        """The Param counterpart to _bind_local -- registers `p`'s
        name and declared type, plus id(p), in the current scope,
        pointing at the logical slot _collect_params already assigned
        it. Also creates `p`'s own Temp, anchored at that same logical
        slot -- not yet a resolved offset, which _resolve_frame_layout
        computes later, in lower_function.

        The Temp's own type is p_type itself UNLESS p's own address
        escapes (_is_heap_allocated), in which case it's a pointer to
        p_type instead: this Temp now holds the address of a malloc'd
        box holding the real value (see _ir_param_setup's own scalar
        case), not the value directly, so every later is_wide_type/
        register-allocation decision needs to see it as 8-byte pointer-
        shaped, not p_type's own natural width. scope[p.name][1]
        (p_type itself, read by _local_type) is deliberately NOT
        changed here -- callers checking "is this variable composite,
        does it need widening, ..." need Hornet's own declared type,
        never this storage-representation detail."""
        slot = ir_fn.var_slots[id(p)]
        p_type = type_from_name(p.type, self.ir_program.struct_registry, self.ir_program.type_alias_registry, sum_types=self.ir_program.sum_type_registry)
        temp_type = Type(TypeKind.POINTER, element_type=p_type) if self._is_heap_allocated(id(p), p_type) else p_type
        self.scopes[-1][p.name] = (slot, p_type, id(p), self.ir_program.ids.temp_at_offset(temp_type, slot))
        return slot

    def _collect_locals(self, statements: List[Node], ir_fn: IRFunction) -> None:
        """Recursively walks `statements`, including into every If's
        then_body/else_body and every While's body, and gives each
        VarDecl found its own permanent stack slot, keyed by the AST
        node's identity rather than its name.

        Each slot's width is the variable's actual type width, except
        an array over _STACK_ARRAY_LIMIT_BYTES, which only needs 8
        bytes here: its slot holds a pointer to a heap block, allocated
        via _ir_malloc_and_store (see gen_statement_ir's own VarDecl
        case), not the array's data directly. No alignment padding
        between slots -- x86-64 doesn't require it, and %rsp's own
        16-byte alignment is satisfied by _frame_size rounding the
        total frame size up at the end."""
        for stmt in statements:
            if isinstance(stmt, VarDecl):
                var_type = type_from_name(stmt.var_type, self.ir_program.struct_registry, self.ir_program.type_alias_registry, sum_types=self.ir_program.sum_type_registry)
                width = 8 if self._is_heap_allocated(
                    id(stmt), var_type) else type_byte_width(var_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
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
        _collect_locals -- looking for FOUR kinds of array-/struct-
        typed expression with no address of its own: a function-call
        argument (an ArrayLiteral, a struct literal, or an ordinary
        array/struct-returning Call used directly as an argument), an
        ordinary composite-returning Call sitting directly at an
        Index.array/Field.base position, a bare bracketed-list literal
        at an Index.array position -- not Slice.array, for any of
        these, and no Field.base equivalent for the third (a struct
        literal there is already rejected by semantic.py) -- and a
        struct literal used directly as a bare ExprStmt, reserved on
        the way back up from this method's own ExprStmt case rather
        than inside _collect_argument_temps_in_expr, since it's the
        top-level statement shape itself that qualifies. None of the
        first three is a Variable, Index, or Field, each of which
        already has a real address via _ir_array_address/_ir_struct_
        address.

        Not just ORDINARY function-call arguments: the walk finds a
        qualifying argument inside ANY Call node, with no check on
        `expr.name` -- print, len, and append are ordinary Call nodes
        as far as this pass is concerned.

        WHY THIS CAN'T REUSE THE SHARED-SLOT TRICK: an array/struct
        argument is passed BY ADDRESS, and that address has to keep
        pointing at valid data right up until the `call` instruction
        executes. A single call can have more than one such argument
        at once (`foo([1,2], [3,4])`), both alive simultaneously
        through the call -- a shared slot would let the second write
        clobber the first before `call` ever runs. Each occurrence
        needs its own distinct backing storage, discovered ahead of
        time here.

        SIZE THRESHOLD, matching every other array/struct value: not
        every occurrence found here gets a stack slot -- _reserve_
        argument_temp applies the same is_heap_allocated size check
        every named local/parameter goes through. A large one gets no
        slot at all -- heap-allocated fresh at the point of the call
        instead, since an argument-temp's pointer is read exactly once,
        by the callee's own entry-time copy, and never again.

        WHY THIS WALKS EXPRESSIONS, not just statements: a literal-or-
        returning-call-as-argument can be buried arbitrarily deep
        inside another expression (`int x = foo(1) + bar([1,2,3])`),
        so this needs a real expression walk (_collect_argument_temps_
        in_expr)."""
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
                if isinstance(stmt.expr, Call) and stmt.expr.name in self.ir_program.struct_registry:
                    # A struct literal used directly as a bare
                    # statement has no address of its own to write
                    # through, same as one used as a function-call
                    # argument.
                    self._reserve_argument_temp(stmt.expr, type_of(stmt.expr), ir_fn)
            # Break/Continue carry no expressions at all.

    def _is_ordinary_composite_call(self, expr: Node) -> bool:
        """True for a Call that goes through the ordinary hidden-
        pointer convention (see _ir_composite_call) -- excludes a
        struct-literal Call (construction, not an ordinary call) and
        append (a builtin, never compiled as an ordinary function)."""
        return isinstance(expr, Call) and expr.name != 'append' and expr.name not in self.ir_program.struct_registry

    def _collect_argument_temps_in_expr(self, expr: Optional[Node], ir_fn: IRFunction) -> None:
        """The general expression-tree walk _collect_argument_temps
        needs -- recurses into every expression node that can contain
        another (Binary, Unary, Index, Field, Slice, ArrayLiteral's
        elements, a Call's arguments), with a leaf case for everything
        else.

        The actual decision -- does this Call argument need its own
        reserved slot -- is made only at a Call node: after recursing
        into each of its own arguments first (so a nested call is
        discovered on the way back up), any argument that's array- or
        struct-typed and isn't a Variable, Index, or Field gets handed
        to _reserve_argument_temp."""
        if expr is None:
            return
        if isinstance(expr, Call):
            # The callee's own declared parameter types (None for a
            # builtin -- print/len/append are never in function_
            # registry, and none of them has a sum-typed parameter to
            # widen an argument into regardless). Looked up ONCE per
            # Call, not per argument -- a single dict lookup, reused
            # for however many arguments this call actually has.
            param_types = self.ir_program.function_registry[expr.name][0] if expr.name in self.ir_program.function_registry else None
            for i, arg in enumerate(expr.args):
                self._collect_argument_temps_in_expr(arg, ir_fn)
                arg_type = type_of(arg)
                # A struct-typed argument flowing into a sum-typed
                # parameter needs a slot sized for the WIDER sum type
                # (tag + largest variant), not the narrower struct it
                # actually is -- see _ir_materialize_sum_type_value
                # (ir/sum_types.py), which is what actually writes
                # into a slot reserved this way. Every other case
                # (including an ALREADY sum-typed argument, needing no
                # widening at all -- e.g. `takesShape(makeShape())`)
                # keeps using arg_type unchanged, exactly as before.
                reserve_type = arg_type
                if param_types is not None and arg_type.kind == TypeKind.STRUCT and param_types[i].kind == TypeKind.SUM:
                    reserve_type = param_types[i]
                if reserve_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT, TypeKind.SUM) and not isinstance(arg, (Variable, Index, Field)):
                    self._reserve_argument_temp(arg, reserve_type, ir_fn)
        elif isinstance(expr, Binary):
            self._collect_argument_temps_in_expr(expr.left, ir_fn)
            self._collect_argument_temps_in_expr(expr.right, ir_fn)
        elif isinstance(expr, Unary):
            self._collect_argument_temps_in_expr(expr.operand, ir_fn)
            if (expr.op == UnaryOp.ADDRESS_OF and isinstance(expr.operand, Call)
                    and expr.operand.name in self.ir_program.struct_registry):
                # `&Circle(5)` -- semantic.py's own check_unary is the
                # only place that allows a struct-literal ADDRESS_OF
                # operand at all, so reaching this shape here already
                # means the literal needs SOME real address, one way
                # or another. Unlike an ordinary argument-temp (see
                # _reserve_argument_temp's own docstring for why THAT
                # one is never escape-driven -- it's always read
                # exactly once, by a callee's own entry-time copy),
                # this address might genuinely outlive this function
                # (returned, stored, passed on) -- exactly the same
                # question a named local's own address already answers
                # via the full, escape-aware _is_heap_allocated, not
                # the size-only is_heap_allocated function that lets
                # every other argument-temp skip reservation for a
                # large value instead of ever asking. id(expr.operand)
                # is this literal's own synthetic decl_id -- the same
                # identity scheme every other declaration already
                # uses, and the exact one escape_analysis.py's own
                # contribution() registers into _escaping_decl_ids
                # under, via declare(), the first time this same node
                # is seen there.
                struct_type = type_of(expr.operand)
                if not self._is_heap_allocated(id(expr.operand), struct_type):
                    width = type_byte_width(struct_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
                    self._argument_temp_slots[id(expr.operand)] = self.ir_program.ids.new_slot(width, "struct_literal_address", ir_fn)
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
            # Never reserves a slot here, unlike Index/Field: a slice
            # PRODUCED from this base escapes -- its own ptr aliases
            # whatever backs the base for as long as the slice is
            # alive, which can far outlive this statement. A stack
            # slot would be unsafe regardless of size. See _ir_
            # materialize_composite_call's own docstring for the
            # codegen-time half: no reservation found is exactly what
            # tells it to malloc instead.
        elif isinstance(expr, ArrayLiteral):
            for element in expr.elements:
                self._collect_argument_temps_in_expr(element, ir_fn)
        # Constant/BoolLiteral/StringLiteral/NoneLiteral/Variable: leaves,
        # nothing further to recurse into.

    def _reserve_argument_temp(self, expr: Node, t: Type, ir_fn: IRFunction) -> None:
        """Reserves a logical frame slot for `expr` -- an ArrayLiteral,
        a struct literal, or an ordinary array/struct-returning Call
        used directly as a function-call argument -- keyed by id(expr)
        exactly like var_slots keys a VarDecl/Param, for a synthetic,
        unnamed "declaration" with no actual source-level variable.

        Skips reservation when `t` is over the same is_heap_allocated
        size threshold every named local/parameter uses: a large value
        is heap-allocated fresh at the call site instead. Deliberately
        not routed through _is_heap_allocated (which also consults
        self._escaping_decl_ids): an argument-temp is never a
        candidate for escape-driven promotion, since it flows into the
        callee as a whole value copied on entry, never sliced by the
        caller."""
        if is_heap_allocated(t, self.ir_program.struct_registry, self.ir_program.sum_type_registry):
            return
        width = type_byte_width(t, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        self._argument_temp_slots[id(expr)] = self.ir_program.ids.new_slot(width, "argument_temp", ir_fn)

    def _push_scope(self) -> None:
        self.scopes.append({})

    def _pop_scope(self) -> None:
        self.scopes.pop()

    def _bind_local(self, stmt: VarDecl, ir_fn: IRFunction) -> int:
        """Registers `stmt`'s name, declared type, and id(stmt) in the
        current (innermost) scope, pointing at the logical slot
        _collect_locals already assigned this VarDecl node, and
        returns that slot. Also creates `stmt`'s own Temp, anchored at
        that same logical slot -- the same Temp identity reused for
        this variable's entire lifetime, so every later read (gen_
        expr_ir's own Variable case) resolves to it rather than
        re-emitting a fresh copy each time. Composite-typed (array/
        struct/slice) locals get a Temp here too, for uniformity, but
        never actually use it -- those address this variable's slot
        directly instead.

        The Temp's own type is var_type itself UNLESS stmt's own
        address escapes (_is_heap_allocated), in which case it's a
        pointer to var_type instead -- see _bind_param's own,
        identical reasoning for why (this Temp now holds a malloc'd
        box's own address, not the value directly) and for why
        scope[stmt.name][1] (var_type itself, read by _local_type)
        stays untouched regardless."""
        slot = ir_fn.var_slots[id(stmt)]
        var_type = type_from_name(stmt.var_type, self.ir_program.struct_registry, self.ir_program.type_alias_registry, sum_types=self.ir_program.sum_type_registry)
        temp_type = Type(TypeKind.POINTER, element_type=var_type) if self._is_heap_allocated(id(stmt), var_type) else var_type
        self.scopes[-1][stmt.name] = (slot, var_type, id(stmt), self.ir_program.ids.temp_at_offset(temp_type, slot))
        return slot

    def _local_slot(self, name: str) -> int:
        """Resolves `name` to its own logical frame slot -- for a
        composite-typed (array/struct/slice) variable's own address
        (_ir_array_address/_ir_struct_address/_ir_slice_address/_ir_
        indexable_base), or, now, for `&name` on a SCALAR too (see
        ir/pointers.py's own _ir_address_of) -- the identical slot
        lookup either way, since every variable gets one at _bind_
        local time regardless of its own type."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name][0]
        raise IRError(f"Reference to undeclared variable '{name}'")

    def _local_type(self, name: str) -> Type:
        """Used where a Variable's *slot* is also being looked up
        right alongside it (_ir_array_address's own Variable case) --
        both come from the same (slot, Type, decl_id, Temp) tuple in
        the same scope-stack entry. Deliberately not replaced by
        type_of, even though it gives the same answer for a Variable
        node: codegen has to maintain this mapping regardless, since
        resolved_type has no way to encode which stack slot a name
        refers to.

        Returns a real semantic.Type (via type_from_name, resolved
        once in _bind_local/_bind_param), not the raw parser-level
        string/ArrayTypeExpr."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name][1]
        raise IRError(f"Reference to undeclared variable '{name}'")

    def _local_decl_id(self, name: str) -> int:
        """Returns id(the VarDecl or Param node) that `name` currently
        resolves to -- kept in the same scope-stack lookup so this
        respects shadowing correctly: Hornet allows re-declaring a name
        in a nested if/while block, so a plain name doesn't uniquely
        identify a declaration the way id() of the AST node does. Used
        by _is_heap_allocated to check whether THIS SPECIFIC
        declaration was found to escape by analyze_array_escapes."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name][2]
        raise IRError(f"Reference to undeclared variable '{name}'")

    def _local_temp(self, name: str) -> Temp:
        """Returns `name`'s own persistent Temp, created once by
        _bind_local/_bind_param and reused for every read and write of
        this variable for the rest of its scope."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name][3]
        raise IRError(f"Reference to undeclared variable '{name}'")

    def _is_heap_allocated(self, decl_id: int, t: Type) -> bool:
        """Whether the specific array- or struct-typed declaration
        identified by decl_id needs to be heap-allocated: is_heap_
        allocated's pure size check, OR analyze_array_escapes's result
        (cached in self._escaping_decl_ids -- a declaration of ANY
        type can appear here now, not just array: a pointer's own
        target isn't restricted to arrays the way a slice's own
        backing storage always is, since `&x` can target a scalar,
        struct, array, or sum-typed x. This method itself is still
        only ever CALLED for array/struct/sum-typed decl_ids -- a
        scalar whose address escapes is rejected outright at semantic
        analysis instead of reaching here at all; see check_unary's
        own ADDRESS_OF case)."""
        return is_heap_allocated(t, self.ir_program.struct_registry, self.ir_program.sum_type_registry) or decl_id in self._escaping_decl_ids
