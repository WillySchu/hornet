"""Statement-level codegen: declarations, assignment, control flow
(if/while/break/continue), and return. Establishes that an
uninitialized declaration gets its type's real zero value rather than
leaving memory untouched, and the label-pair shape (start/end, or
else/end) every branching or looping construct here builds on."""

from codegen.assembly_ast import Instruction, MovQ, Register, Memory, Imm, Push, Pop, Mov, Jmp, LeaQ, MovB
from codegen.errors import CodegenError
from codegen.ir import IRRaw, IRReturn, IRBranch, IRLabel, IRJump, IRMove, IRStore, IRCopy, IRConst, IRLoad, IRLocalAddress, IRCall
from codegen.utils import type_of, type_byte_width
from parser import (
    ArrayLiteral,
    Assign,
    Break,
    Call,
    Continue,
    ExprStmt,
    Field,
    FieldAssign,
    If,
    Index,
    IndexAssign,
    Node,
    NoneLiteral,
    Return,
    Slice,
    VarDecl,
    Variable,
    While,
)
from semantic import TypeKind, Type, type_from_name


class StatementsMixin:
    def gen_statement(self, stmt: Node) -> list[Instruction]:
        if isinstance(stmt, VarDecl):
            return self.gen_var_decl(stmt)
        if isinstance(stmt, Assign):
            return self.gen_assign(stmt)
        if isinstance(stmt, IndexAssign):
            return self.gen_index_assign(stmt)
        if isinstance(stmt, FieldAssign):
            return self.gen_field_assign(stmt)
        if isinstance(stmt, Return):
            return self.gen_return(stmt)
        if isinstance(stmt, If):
            return self.gen_if(stmt)
        if isinstance(stmt, While):
            return self.gen_while(stmt)
        if isinstance(stmt, Break):
            return self.gen_break(stmt)
        if isinstance(stmt, Continue):
            return self.gen_continue(stmt)
        if isinstance(stmt, ExprStmt):
            return self.gen_expr_stmt(stmt)
        raise CodegenError(f"No codegen rule for statement: {stmt!r}")

    def gen_statement_ir(self, stmt: Node) -> list:
        """The IR-native counterpart to gen_statement: builds real IR
        for Return/If/While (recursing into itself, not gen_statement,
        for If/While bodies -- so nested control flow stays real IR
        all the way down), a scalar VarDecl-with-initializer, Assign,
        IndexAssign, or FieldAssign (an IRMove into a variable's own
        persistent Temp, or an IRStore through an address -- see
        _ir_index_assign/_ir_field_assign -- for the latter two), and
        a bare expression statement (via gen_expr_ir, discarding
        whatever value comes back).

        An array/struct/slice-typed VarDecl/Assign/IndexAssign/
        FieldAssign is also real IR now, across several distinct
        value shapes, for all four statement kinds alike: a Variable/
        Field/Index value (an IRCopy between two captured addresses --
        see _ir_copy_assign); a Slice production, for a slice-typed
        target (see _ir_slice_into/_ir_write_slice_descriptor); an
        append(...) call, likewise slice-typed-target only (see _ir_
        append_call/_ir_write_slice_descriptor); an ordinary (non-
        struct-literal, non-append) function call returning a
        composite value, via the hidden-pointer convention (see _ir_
        composite_call) -- writing directly through this target's own
        address, with no intermediate copy ever materialized; and now
        an array literal or struct literal too (positional or named/
        partial alike), via the exact same _ir_write_array_literal_
        into/_ir_write_struct_literal_into/_ir_write_composite_value_
        into machinery Return's own identical case already used --
        both writers take an arbitrary destination address, never
        assumed to be Return's own hidden pointer specifically, so
        this was pure wiring: compute this target's own address the
        same way every other case here already does, then hand it to
        the same writer. (IndexAssign's own case is narrower here,
        covering only struct literals, not array ones -- ARRAY never
        occurs as an IndexAssign element type in the first place; see
        that case's own opening comment.) A VarDecl with NO initializer
        at all is real IR too, for all three composite kinds now, not
        just slice: a slice's own nil zero value needs no computation
        at all (three IRConst(0, ...)s through _ir_write_slice_
        descriptor), while an array/struct's own zero value -- capable
        of recursing through further nested composites, with a real,
        bounded-count loop (not per-element unrolling) for an array
        leaf -- goes through _ir_write_zero_value_into/_ir_zero_
        array_loop.

        A REAL BUG surfaced, and was fixed, only once this VarDecl/
        Assign/IndexAssign/FieldAssign wiring finally exercised an
        array-of-slices literal (`[2][]int[[1, 2], [3, 4]]`) through
        _ir_write_composite_value_into for the first time: the same
        bracketed-list AST shape parses identically for an array
        literal and a slice literal, and the dispatcher's own
        ArrayLiteral case wasn't checking value_type.kind == ARRAY
        before routing there, silently corrupting every element's
        computed address once value_type.kind was actually SLICE. See
        that method's own docstring for the full account -- fixed by
        an explicit kind check; genuine slice-LITERAL construction (as
        opposed to slice PRODUCTION via `arr[a:b]`, already real IR)
        remains unbuilt, an honest, separate gap, not something this
        fix incidentally also solved.

        Return's own composite case now covers all but ONE genuinely
        hard shape: forwarding another composite-returning call's
        result (`return someFn()`, via _ir_composite_call), a
        Variable/Field/Index value (via _ir_copy_into_address), and an
        array literal or struct literal -- positional or named/partial
        (kwargs) alike, an omitted field's own zero value written via
        _ir_write_zero_value_into -- (via _ir_write_array_literal_
        into/_ir_write_struct_literal_into) -- all reuse the current
        function's own received hidden pointer, read via an ordinary
        address-as-a-Temp leaf (_ir_hidden_return_ptr). A literal's own
        elements/fields need not be scalar either: a nested composite
        element/field (another literal, an existing value, a Slice
        production, an append call, `none`, or an ordinary composite-
        returning Call) is handled by _ir_write_composite_value_into, a
        general-purpose dispatcher the two literal-writing methods call
        back into for their own composite elements/fields -- mutual
        recursion, the same shape address computation's own Field/
        Index handling already relies on elsewhere in this arc.

        Break and Continue are real IR too (_ir_break/_ir_continue --
        an exact, zero-behavioral-difference IRJump-for-Jmp swap, see
        their own docstrings), as is an ArrayLiteral- or Slice-valued
        ExprStmt (_ir_array_literal_side_effects_only/_ir_slice_into,
        evaluating for side effects/the bounds check alone, with
        nothing ever materialized or read back -- see gen_expr_stmt's
        own docstring for exactly why an ArrayLiteral in particular
        needs this narrower treatment rather than an ordinary
        gen_expr_ir call).

        Falls back to wrapping gen_statement itself, as a single
        opaque IRRaw, for everything else (a Return whose value is an
        ArrayLiteral/struct-literal Call with some field/element out of
        scope for _ir_write_composite_value_into, or bare `none`; a
        Slice-valued ExprStmt whose own base is out of scope for _ir_
        slice_into) -- a real, deliberate scope boundary, not an
        oversight: those still need their own IR-native handling as a
        follow-up.
        """
        if isinstance(stmt, Return):
            is_composite_return = isinstance(stmt.value, NoneLiteral) or (
                stmt.value is not None and type_of(stmt.value).kind in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT)
            )
            if not is_composite_return:
                return self._ir_return(stmt.value)
            # A composite return whose own value is an ordinary
            # function call (`return someFn()`) -- forwards the
            # CURRENT function's own received hidden pointer straight
            # through as the destination for the INNER call, via the
            # exact same _ir_composite_call every other composite-
            # returning-call case above uses, with no intermediate
            # copy ever materialized -- the identical "just pass the
            # same address one level deeper" forwarding gen_return's
            # own old-style version already does, now built from an
            # ordinary address-as-a-Temp leaf (reading this function's
            # own fixed, known hidden_return_ptr_offset) rather than a
            # raw %rax reload. IRReturn(value=None) still needs to run
            # -- there's no scalar result to pass back, but the
            # epilogue still has to execute -- matching the bare-
            # return shape exactly, since the result is already fully
            # written through the pointer by the time control reaches
            # it.
            if isinstance(stmt.value, Call) and stmt.value.name != 'append' and stmt.value.name not in self.struct_registry:
                value_type = type_of(stmt.value)
                hidden_ptr_ir, hidden_ptr = self._ir_hidden_return_ptr()
                call_ir = self._ir_composite_call(hidden_ptr, stmt.value, value_type)
                return hidden_ptr_ir + call_ir + [IRReturn(value=None)]
            # A composite return whose own value is a Variable/Field/
            # Index (an existing value with a real address to copy
            # from) -- the exact same "given an already-computed
            # destination address, copy src_expr's value there" core
            # _ir_copy_assign's own VarDecl/Assign/IndexAssign/
            # FieldAssign callers already use, via _ir_copy_into_
            # address directly (this function's own hidden pointer
            # standing in for a freshly-computed destination address,
            # since it's already exactly that: an address, however
            # it was obtained).
            if isinstance(stmt.value, (Variable, Field, Index)):
                value_type = type_of(stmt.value)
                hidden_ptr_ir, hidden_ptr = self._ir_hidden_return_ptr()
                copy_ir = self._ir_copy_into_address(hidden_ptr, stmt.value, value_type)
                return hidden_ptr_ir + copy_ir + [IRReturn(value=None)]
            # A composite return whose own value is an array literal,
            # a bare bracketed-list literal resolved to SLICE by this
            # function's own declared return type, or a POSITIONAL
            # struct literal, with every element/field scalar -- writes
            # directly through the hidden pointer via _ir_write_array_
            # literal_into/_ir_slice_literal/_ir_write_struct_literal_
            # into. All return None (no binding-style concern here at
            # all, unlike VarDecl's own analogous cases -- there's
            # nothing to bind for a Return) when out of scope: a
            # nested composite element/field, or named/partial
            # (kwargs) struct construction -- both still need their
            # own recursive real-IR treatment, a genuinely separate,
            # later step, not attempted here.
            #
            # A REAL BUG, found and fixed here: this ARRAY case used to
            # call _ir_write_array_literal_into unconditionally for
            # ANY ArrayLiteral return value, with no value_type.kind ==
            # ARRAY check first -- the identical AST-shape ambiguity
            # (an ArrayLiteral node can resolve to either ARRAY or
            # SLICE, depending on context) _ir_write_composite_value_
            # into's own dispatcher was fixed for, just never
            # backported to this, a separate call site that bypasses
            # that dispatcher entirely. `return [1, 2, 3]` from a
            # slice-returning function used to segfault: the hidden
            # pointer for a SLICE return points at a three-field
            # descriptor slot, not a bare array's own backing, so
            # writing element bytes directly through it (as if it were
            # one) corrupted memory immediately after the elements'
            # own extent. See _ir_slice_literal's own docstring for
            # the SLICE case's own, now-correct treatment just below.
            if isinstance(stmt.value, ArrayLiteral):
                # Dispatches on self._current_return_type (this
                # function's own DECLARED return type), NOT type_of(
                # stmt.value): the latter is always ARRAY-kind for an
                # ArrayLiteral, correctly sized to its own element
                # count, regardless of what the surrounding context
                # (here, this function's own signature) resolves the
                # overall expression to -- the identical ambiguity
                # _ir_slice_literal's own docstring documents fixing
                # for its own malloc-size computation. Using type_of(
                # stmt.value) here instead used to make `return [1, 2,
                # 3]` from a slice-returning function segfault: it
                # always took the ARRAY branch, writing raw element
                # bytes directly through the hidden pointer as if it
                # addressed the array's own backing, when a SLICE
                # return's hidden pointer actually addresses a three-
                # field descriptor slot instead.
                if self._current_return_type.kind == TypeKind.ARRAY:
                    hidden_ptr_ir, hidden_ptr = self._ir_hidden_return_ptr()
                    write_ir = self._ir_write_array_literal_into(hidden_ptr, stmt.value, self._current_return_type)
                    if write_ir is not None:
                        return hidden_ptr_ir + write_ir + [IRReturn(value=None)]
                elif self._current_return_type.kind == TypeKind.SLICE:
                    hidden_ptr_ir, hidden_ptr = self._ir_hidden_return_ptr()
                    production = self._ir_slice_literal(stmt.value)
                    if production is not None:
                        slice_ir, ptr_value, len_value, cap_value = production
                        write_ir = self._ir_write_slice_descriptor_into_address(hidden_ptr, ptr_value, len_value, cap_value)
                        return hidden_ptr_ir + slice_ir + write_ir + [IRReturn(value=None)]
            if isinstance(stmt.value, Call) and stmt.value.name in self.struct_registry:
                value_type = type_of(stmt.value)
                hidden_ptr_ir, hidden_ptr = self._ir_hidden_return_ptr()
                write_ir = self._ir_write_struct_literal_into(hidden_ptr, stmt.value, value_type)
                if write_ir is not None:
                    return hidden_ptr_ir + write_ir + [IRReturn(value=None)]
            # Every OTHER composite return shape (an array literal or
            # struct literal with a nested composite element/field, a
            # named/partial struct literal, or `return none`) still
            # falls through to the old-style catch-all below -- a
            # real, deliberate scope boundary, not an oversight.
        elif isinstance(stmt, If):
            then_label = self.new_label("if_then")
            else_label = self.new_label("if_else")
            end_label = self.new_label("if_end")
            ir = self._ir_if_head(stmt, then_label, else_label)
            self._push_scope()
            for s in stmt.then_body:
                ir.extend(self.gen_statement_ir(s))
            self._pop_scope()
            ir.append(IRJump(end_label))
            ir.append(IRLabel(else_label))
            if stmt.else_body is not None:
                self._push_scope()
                for s in stmt.else_body:
                    ir.extend(self.gen_statement_ir(s))
                self._pop_scope()
            ir.append(IRLabel(end_label))
            return ir
        elif isinstance(stmt, While):
            start_label = self.new_label("while_start")
            body_label = self.new_label("while_body")
            end_label = self.new_label("while_end")
            ir = self._ir_while_head(stmt, start_label, body_label, end_label)
            self.loop_labels.append((start_label, end_label))
            self._push_scope()
            for s in stmt.body:
                ir.extend(self.gen_statement_ir(s))
            self._pop_scope()
            self.loop_labels.pop()
            ir.append(IRJump(start_label))
            ir.append(IRLabel(end_label))
            return ir
        elif isinstance(stmt, VarDecl):
            # A scalar VarDecl WITH an initializer: bind the variable
            # (creating its own persistent Temp -- see _bind_local),
            # build the initializer's IR, and IRMove the result
            # straight into that Temp. A no-initializer VarDecl (needs
            # composite-aware zero-init) or a slice-typed one falls to
            # the catch-all below WITHOUT binding here -- gen_var_decl
            # does its own single _bind_local call, and binding twice
            # would just orphan a Temp id, harmlessly but pointlessly.
            var_type = type_from_name(stmt.var_type, self.struct_registry, self.type_alias_registry)
            if stmt.init is not None and not isinstance(stmt.init, NoneLiteral) and var_type.kind not in (
                    TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT):
                self._bind_local(stmt)
                ir, value = self.gen_expr_ir(stmt.init)
                return ir + [IRMove(dst=self._local_temp(stmt.name), src=value)]
            # An array/struct/slice-typed initializer that's itself a
            # Variable/Field/Index (an existing value with a real
            # address to copy from, not a literal or a call producing
            # a fresh one) -- see _ir_copy_assign. Checked, and bound,
            # only once the shape is already known to qualify, so an
            # ArrayLiteral/Slice/Call initializer still falls to the
            # catch-all with no binding done here either.
            if var_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT, TypeKind.SLICE) and isinstance(stmt.init, (Variable, Field, Index)):
                offset = self._bind_local(stmt)
                ir = []
                # A slice variable is never heap-allocated (see gen_
                # assign's own heap-allocation check, scoped to ARRAY/
                # STRUCT only -- a slice's own descriptor is always a
                # small, fixed-size, stack-resident value), so this
                # malloc step is explicitly skipped for SLICE, not
                # left to _is_heap_allocated's own implicit False.
                if var_type.kind != TypeKind.SLICE and self._is_heap_allocated(id(stmt), var_type):
                    # Unlike Assign/IndexAssign/FieldAssign, this
                    # destination is BRAND NEW -- its slot holds
                    # nothing yet, heap-allocated or not. A heap-
                    # allocated one needs its own fresh backing
                    # allocation made here, exactly like gen_var_decl's
                    # own identical branch does, BEFORE _ir_copy_assign
                    # ever tries to read an address out of this slot --
                    # otherwise it would read whatever pointer-sized
                    # garbage was already sitting there.
                    ir.extend(self._ir_malloc_and_store(var_type, offset))
                return ir + self._ir_copy_assign(Variable(name=stmt.name), stmt.init, var_type)
            # A slice-typed initializer that's itself a Slice
            # production (`arr[a:b]`, not an alias of an existing
            # slice -- see _ir_slice_into for exactly which shapes of
            # `arr` it covers). _ir_slice_into is tried FIRST, before
            # any binding, since whether it succeeds depends only on
            # stmt.init's own array/base, never on the destination --
            # binding only happens once it's already known to
            # succeed, so an out-of-scope base still falls to the
            # catch-all with no binding done here either, matching
            # every other case in this method.
            if var_type.kind == TypeKind.SLICE and isinstance(stmt.init, Slice):
                production = self._ir_slice_into(stmt.init)
                if production is not None:
                    slice_ir, ptr_value, len_value, cap_value = production
                    self._bind_local(stmt)
                    return slice_ir + self._ir_write_slice_descriptor(Variable(name=stmt.name), ptr_value, len_value, cap_value)
            # A slice-typed initializer that's a bare bracketed-list
            # literal, resolved to SLICE by this VarDecl's own
            # declared type (`[]int s = [1, 2, 3]`) -- see _ir_slice_
            # literal's own docstring for why this is a genuinely
            # different AST shape from the Slice-production case just
            # above (an ArrayLiteral, not a Slice node) and from the
            # already-real-IR ARRAY-typed VarDecl case (same
            # ArrayLiteral AST shape, disambiguated by var_type.kind,
            # the exact fix _ir_write_composite_value_into's own
            # identical disambiguation already applies for a nested
            # element/field). Same "try first, bind only on success"
            # discipline as Slice production/append above.
            if var_type.kind == TypeKind.SLICE and isinstance(stmt.init, ArrayLiteral):
                production = self._ir_slice_literal(stmt.init)
                if production is not None:
                    slice_ir, ptr_value, len_value, cap_value = production
                    self._bind_local(stmt)
                    return slice_ir + self._ir_write_slice_descriptor(Variable(name=stmt.name), ptr_value, len_value, cap_value)
            # A slice-typed initializer that's a call to the append
            # builtin specifically (`append(s, value)`) -- same "try
            # first, bind only on success" discipline as the Slice
            # production case just above, for the identical reason.
            # ANY element type is in scope now, including array/slice/
            # struct -- see _ir_append_call's own docstring. What still
            # falls back here: s's own base out of scope (see _ir_
            # indexable_base), or value itself out of scope for _ir_
            # write_append_value_at when the element type is composite
            # (a named/partial struct literal nested inside it,
            # chiefly).
            if var_type.kind == TypeKind.SLICE and isinstance(stmt.init, Call) and stmt.init.name == 'append':
                production = self._ir_append_call(stmt.init)
                if production is not None:
                    append_ir, ptr_value, len_value, cap_value = production
                    self._bind_local(stmt)
                    return append_ir + self._ir_write_slice_descriptor(Variable(name=stmt.name), ptr_value, len_value, cap_value)
            # An array/struct/slice-typed initializer that's an
            # ordinary function call (not append -- already handled,
            # more specifically, just above; not a struct-literal Call
            # either -- those are construction, disambiguated the same
            # way semantic.py's own check_call does, by registry
            # membership, not by shape) -- writes directly through
            # this variable's own address via the hidden-pointer
            # convention (see _ir_composite_call's own docstring),
            # with no intermediate copy at all. Needs the SAME heap-
            # allocation-first step the Variable/Field/Index case
            # above needs, and for the identical reason: this
            # destination is BRAND NEW, so a heap-allocated one needs
            # its own fresh backing allocation made BEFORE this
            # variable's own address is ever computed.
            if var_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT, TypeKind.SLICE) and (
                    isinstance(stmt.init, Call) and stmt.init.name != 'append' and stmt.init.name not in self.struct_registry):
                offset = self._bind_local(stmt)
                ir = []
                if var_type.kind != TypeKind.SLICE and self._is_heap_allocated(id(stmt), var_type):
                    ir.extend(self._ir_malloc_and_store(var_type, offset))
                address_fn = {
                    TypeKind.ARRAY: self._ir_array_address,
                    TypeKind.STRUCT: self._ir_struct_address,
                    TypeKind.SLICE: self._ir_slice_address,
                }[var_type.kind]
                dst_ir, dst_address = address_fn(Variable(name=stmt.name))
                ir.extend(dst_ir)
                return ir + self._ir_composite_call(dst_address, stmt.init, var_type)
            # An array/struct-typed initializer that's an array
            # literal or a struct literal (positional or named/
            # partial alike) -- writes directly through this
            # variable's own address via the exact same _ir_write_
            # array_literal_into/_ir_write_struct_literal_into
            # machinery Return's own identical case already uses
            # (both take an arbitrary destination address, never
            # assumed to be Return's own hidden pointer specifically).
            # Needs the SAME heap-allocation-first step every other
            # fresh-value-producing case above needs.
            #
            # Binds FIRST, unlike Slice production/append above:
            # producing this value needs the destination's own
            # address as an INPUT (not just its own source shape), so
            # there's no way to know success without it. This is safe
            # regardless of outcome, though: _bind_local's own Temp
            # always points at this VarDecl's own pre-computed,
            # permanent offset (see its own docstring), so a second
            # _bind_local call from gen_var_decl's own old-style
            # fallback, if this one fails, resolves to the identical
            # underlying storage location, not a different one -- the
            # same "orphans a Temp id, harmlessly" waste already
            # accepted elsewhere in this arc, not a genuine hazard.
            if var_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT) and (
                    isinstance(stmt.init, ArrayLiteral) or (isinstance(stmt.init, Call) and stmt.init.name in self.struct_registry)):
                offset = self._bind_local(stmt)
                ir = []
                if self._is_heap_allocated(id(stmt), var_type):
                    ir.extend(self._ir_malloc_and_store(var_type, offset))
                address_fn = self._ir_array_address if var_type.kind == TypeKind.ARRAY else self._ir_struct_address
                dst_ir, dst_address = address_fn(Variable(name=stmt.name))
                ir.extend(dst_ir)
                writer = self._ir_write_array_literal_into if isinstance(stmt.init, ArrayLiteral) else self._ir_write_struct_literal_into
                write_ir = writer(dst_address, stmt.init, var_type)
                if write_ir is not None:
                    return ir + write_ir
            # A slice-typed VarDecl with NO initializer at all -- its
            # implicit zero value is the nil slice (ptr=0, len=0,
            # cap=0), needing no computation whatsoever: three
            # compile-time IRConst(0, ...) values handed straight to
            # _ir_write_slice_descriptor, the same helper every other
            # slice-producing case above already uses to write its own
            # result -- a slice's own zero value is uniquely trivial
            # among the three, with nothing to recurse into at all.
            if var_type.kind == TypeKind.SLICE and stmt.init is None:
                self._bind_local(stmt)
                zero_ptr = IRConst(0, Type.INT64)
                zero_int = IRConst(0, Type.INT)
                return self._ir_write_slice_descriptor(Variable(name=stmt.name), zero_ptr, zero_int, zero_int)
            # An array- or struct-typed VarDecl with NO initializer at
            # all -- its implicit zero value is now real IR too, via
            # _ir_write_zero_value_into (a TOTAL function -- see its
            # own docstring for why it never needs to fall back).
            # Needs the SAME heap-allocation-first step the composite-
            # call case just above needs, and for the identical
            # reason: this destination is BRAND NEW, so a heap-
            # allocated one needs its own fresh backing allocation
            # made BEFORE this variable's own address is ever
            # computed.
            if var_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT) and stmt.init is None:
                offset = self._bind_local(stmt)
                ir = []
                if self._is_heap_allocated(id(stmt), var_type):
                    ir.extend(self._ir_malloc_and_store(var_type, offset))
                address_fn = self._ir_array_address if var_type.kind == TypeKind.ARRAY else self._ir_struct_address
                dst_ir, dst_address = address_fn(Variable(name=stmt.name))
                ir.extend(dst_ir)
                return ir + self._ir_write_zero_value_into(dst_address, var_type)
        elif isinstance(stmt, Assign):
            # Same shape as the VarDecl case above, for an existing
            # scalar variable -- `var_type` here is necessarily
            # already scalar in any well-typed program whenever it
            # isn't ARRAY/SLICE/STRUCT (none of those, including
            # NoneLiteral flowing into a slice target, can reach a
            # non-composite variable at all), so no separate NoneLiteral
            # check is needed the way VarDecl's own case above needs
            # one.
            var_type = self._local_type(stmt.name)
            if var_type.kind not in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT):
                ir, value = self.gen_expr_ir(stmt.value)
                return ir + [IRMove(dst=self._local_temp(stmt.name), src=value)]
            # Same Variable/Field/Index-shaped copy as VarDecl's own
            # case just above -- no heap-allocation branch needed here
            # at all (unlike VarDecl's own): an existing slice
            # variable is never heap-allocated in the first place, and
            # an existing array/struct one already has its own real
            # allocation from declaration time, reused in place.
            if var_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT, TypeKind.SLICE) and isinstance(stmt.value, (Variable, Field, Index)):
                return self._ir_copy_assign(Variable(name=stmt.name), stmt.value, var_type)
            # Same Slice-production case as VarDecl's own, just above
            # -- no binding concern here at all, unlike VarDecl's own
            # (the destination already exists).
            if var_type.kind == TypeKind.SLICE and isinstance(stmt.value, Slice):
                production = self._ir_slice_into(stmt.value)
                if production is not None:
                    slice_ir, ptr_value, len_value, cap_value = production
                    return slice_ir + self._ir_write_slice_descriptor(Variable(name=stmt.name), ptr_value, len_value, cap_value)
            # Same slice-literal case as VarDecl's own, just above --
            # no binding concern here at all, unlike VarDecl's own.
            if var_type.kind == TypeKind.SLICE and isinstance(stmt.value, ArrayLiteral):
                production = self._ir_slice_literal(stmt.value)
                if production is not None:
                    slice_ir, ptr_value, len_value, cap_value = production
                    return slice_ir + self._ir_write_slice_descriptor(Variable(name=stmt.name), ptr_value, len_value, cap_value)
            # Same append-call case as VarDecl's own, just above -- no
            # binding concern here at all, unlike VarDecl's own.
            if var_type.kind == TypeKind.SLICE and isinstance(stmt.value, Call) and stmt.value.name == 'append':
                production = self._ir_append_call(stmt.value)
                if production is not None:
                    append_ir, ptr_value, len_value, cap_value = production
                    return append_ir + self._ir_write_slice_descriptor(Variable(name=stmt.name), ptr_value, len_value, cap_value)
            # Same ordinary-function-call case as VarDecl's own --
            # append already handled, more specifically, just above.
            if var_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT, TypeKind.SLICE) and (
                    isinstance(stmt.value, Call) and stmt.value.name != 'append' and stmt.value.name not in self.struct_registry):
                dst_ir, dst_address = {
                    TypeKind.ARRAY: self._ir_array_address,
                    TypeKind.STRUCT: self._ir_struct_address,
                    TypeKind.SLICE: self._ir_slice_address,
                }[var_type.kind](Variable(name=stmt.name))
                return dst_ir + self._ir_composite_call(dst_address, stmt.value, var_type)
            # Same array-literal/struct-literal case as VarDecl's own,
            # just above -- no binding concern here at all, unlike
            # VarDecl's own (the destination already exists).
            if var_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT) and (
                    isinstance(stmt.value, ArrayLiteral) or (isinstance(stmt.value, Call) and stmt.value.name in self.struct_registry)):
                address_fn = self._ir_array_address if var_type.kind == TypeKind.ARRAY else self._ir_struct_address
                dst_ir, dst_address = address_fn(Variable(name=stmt.name))
                writer = self._ir_write_array_literal_into if isinstance(stmt.value, ArrayLiteral) else self._ir_write_struct_literal_into
                write_ir = writer(dst_address, stmt.value, var_type)
                if write_ir is not None:
                    return dst_ir + write_ir
        elif isinstance(stmt, IndexAssign):
            # Same scope boundary as gen_index_assign itself. ARRAY
            # never occurs here at all (IndexAssign's own grammar
            # can't produce an array-typed element -- see gen_field_
            # assign's own docstring for the contrast with
            # FieldAssign, which can).
            element_type = type_of(stmt.array).element_type
            if element_type.kind not in (TypeKind.SLICE, TypeKind.STRUCT):
                return self._ir_index_assign(stmt, element_type)
            if element_type.kind in (TypeKind.STRUCT, TypeKind.SLICE) and isinstance(stmt.value, (Variable, Field, Index)):
                dst_expr = Index(array=stmt.array, index=stmt.index)
                return self._ir_copy_assign(dst_expr, stmt.value, element_type)
            if element_type.kind == TypeKind.SLICE and isinstance(stmt.value, Slice):
                production = self._ir_slice_into(stmt.value)
                if production is not None:
                    slice_ir, ptr_value, len_value, cap_value = production
                    dst_expr = Index(array=stmt.array, index=stmt.index)
                    return slice_ir + self._ir_write_slice_descriptor(dst_expr, ptr_value, len_value, cap_value)
            if element_type.kind == TypeKind.SLICE and isinstance(stmt.value, ArrayLiteral):
                production = self._ir_slice_literal(stmt.value)
                if production is not None:
                    slice_ir, ptr_value, len_value, cap_value = production
                    dst_expr = Index(array=stmt.array, index=stmt.index)
                    return slice_ir + self._ir_write_slice_descriptor(dst_expr, ptr_value, len_value, cap_value)
            if element_type.kind == TypeKind.SLICE and isinstance(stmt.value, Call) and stmt.value.name == 'append':
                production = self._ir_append_call(stmt.value)
                if production is not None:
                    append_ir, ptr_value, len_value, cap_value = production
                    dst_expr = Index(array=stmt.array, index=stmt.index)
                    return append_ir + self._ir_write_slice_descriptor(dst_expr, ptr_value, len_value, cap_value)
            if element_type.kind in (TypeKind.STRUCT, TypeKind.SLICE) and (
                    isinstance(stmt.value, Call) and stmt.value.name != 'append' and stmt.value.name not in self.struct_registry):
                dst_expr = Index(array=stmt.array, index=stmt.index)
                address_fn = self._ir_struct_address if element_type.kind == TypeKind.STRUCT else self._ir_slice_address
                dst_ir, dst_address = address_fn(dst_expr)
                return dst_ir + self._ir_composite_call(dst_address, stmt.value, element_type)
            # A struct-literal (positional or named/partial) value --
            # no ArrayLiteral case needed here at all, unlike VarDecl/
            # Assign/FieldAssign's own: ARRAY never occurs as an
            # IndexAssign element type in the first place (see this
            # case's own opening comment).
            if element_type.kind == TypeKind.STRUCT and isinstance(stmt.value, Call) and stmt.value.name in self.struct_registry:
                dst_expr = Index(array=stmt.array, index=stmt.index)
                dst_ir, dst_address = self._ir_struct_address(dst_expr)
                write_ir = self._ir_write_struct_literal_into(dst_address, stmt.value, element_type)
                if write_ir is not None:
                    return dst_ir + write_ir
        elif isinstance(stmt, FieldAssign):
            # Same idea one level over -- FieldAssign's grammar can
            # ALSO produce an ARRAY-typed field (unlike IndexAssign).
            field_type = self._check_struct_and_field_type(stmt.base, stmt.name)
            if field_type.kind not in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT):
                return self._ir_field_assign(stmt, field_type)
            if field_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT, TypeKind.SLICE) and isinstance(stmt.value, (Variable, Field, Index)):
                dst_expr = Field(base=stmt.base, name=stmt.name)
                return self._ir_copy_assign(dst_expr, stmt.value, field_type)
            if field_type.kind == TypeKind.SLICE and isinstance(stmt.value, Slice):
                production = self._ir_slice_into(stmt.value)
                if production is not None:
                    slice_ir, ptr_value, len_value, cap_value = production
                    dst_expr = Field(base=stmt.base, name=stmt.name)
                    return slice_ir + self._ir_write_slice_descriptor(dst_expr, ptr_value, len_value, cap_value)
            if field_type.kind == TypeKind.SLICE and isinstance(stmt.value, ArrayLiteral):
                production = self._ir_slice_literal(stmt.value)
                if production is not None:
                    slice_ir, ptr_value, len_value, cap_value = production
                    dst_expr = Field(base=stmt.base, name=stmt.name)
                    return slice_ir + self._ir_write_slice_descriptor(dst_expr, ptr_value, len_value, cap_value)
            if field_type.kind == TypeKind.SLICE and isinstance(stmt.value, Call) and stmt.value.name == 'append':
                production = self._ir_append_call(stmt.value)
                if production is not None:
                    append_ir, ptr_value, len_value, cap_value = production
                    dst_expr = Field(base=stmt.base, name=stmt.name)
                    return append_ir + self._ir_write_slice_descriptor(dst_expr, ptr_value, len_value, cap_value)
            if field_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT, TypeKind.SLICE) and (
                    isinstance(stmt.value, Call) and stmt.value.name != 'append' and stmt.value.name not in self.struct_registry):
                dst_expr = Field(base=stmt.base, name=stmt.name)
                address_fn = {
                    TypeKind.ARRAY: self._ir_array_address,
                    TypeKind.STRUCT: self._ir_struct_address,
                    TypeKind.SLICE: self._ir_slice_address,
                }[field_type.kind]
                dst_ir, dst_address = address_fn(dst_expr)
                return dst_ir + self._ir_composite_call(dst_address, stmt.value, field_type)
            # Same array-literal/struct-literal case as VarDecl/
            # Assign's own -- FieldAssign's grammar can ALSO produce
            # an array-typed field (unlike IndexAssign, per this
            # case's own opening comment), so both apply here, just
            # like VarDecl/Assign's own.
            if field_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT) and (
                    isinstance(stmt.value, ArrayLiteral) or (isinstance(stmt.value, Call) and stmt.value.name in self.struct_registry)):
                dst_expr = Field(base=stmt.base, name=stmt.name)
                address_fn = self._ir_array_address if field_type.kind == TypeKind.ARRAY else self._ir_struct_address
                dst_ir, dst_address = address_fn(dst_expr)
                writer = self._ir_write_array_literal_into if isinstance(stmt.value, ArrayLiteral) else self._ir_write_struct_literal_into
                write_ir = writer(dst_address, stmt.value, field_type)
                if write_ir is not None:
                    return dst_ir + write_ir
        elif isinstance(stmt, Break):
            return self._ir_break()
        elif isinstance(stmt, Continue):
            return self._ir_continue()
        elif isinstance(stmt, ExprStmt) and isinstance(stmt.expr, ArrayLiteral):
            return self._ir_array_literal_side_effects_only(stmt.expr)
        elif isinstance(stmt, ExprStmt) and isinstance(stmt.expr, Slice):
            production = self._ir_slice_into(stmt.expr)
            if production is not None:
                slice_ir, _, _, _ = production
                return slice_ir
        elif isinstance(stmt, ExprStmt) and isinstance(stmt.expr, NoneLiteral):
            # A REAL BUG, found and fixed here: a bare `none` used to
            # crash (CodegenError, "Cannot compute 'none' via gen_expr_
            # into"), since gen_expr_ir's own catch-all delegates to
            # gen_expr_into for anything it doesn't already have a
            # case for -- but gen_expr_into itself defensively REJECTS
            # none rather than handling it (its own docstring's claim
            # that this fallback "covers" every shape gen_expr_into
            # rejects was simply wrong for this one). A bare `none`
            # has no side effect of any kind -- it's a pure literal,
            # nothing to compute or discard -- so the correct real-IR
            # treatment is zero instructions, the same as a bare
            # Constant/BoolLiteral ExprStmt already produces via gen_
            # expr_ir's own first two cases.
            return []
        elif isinstance(stmt, ExprStmt) and isinstance(stmt.expr, Call) and stmt.expr.name == 'append':
            # append(s, value) used directly as a bare statement,
            # its own resulting {ptr, len, cap} triple entirely
            # discarded -- only the growth/write side effect matters.
            # Checked here, before the ordinary composite-Call case
            # just below, for the identical reason append is always
            # checked before an ordinary Call everywhere else in this
            # arc: it's a builtin, never a compiled function, so
            # falling through to that case instead would try to call
            # a symbol literally named 'append' that doesn't exist.
            production = self._ir_append_call(stmt.expr)
            if production is not None:
                append_ir, _, _, _ = production
                return append_ir
        elif isinstance(stmt, ExprStmt) and type_of(stmt.expr).kind in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT) and self._is_ordinary_composite_call(stmt.expr):
            # A REAL BUG, found and fixed here: an array/struct/slice-
            # returning ordinary Call used directly as a bare statement
            # (`makeArray()` alone on a line, its own result entirely
            # discarded) used to crash the identical way `none` did --
            # gen_expr_into defensively rejects a composite-returning
            # Call too (it can't fit the hidden-pointer convention's
            # own result into a single register). The call's own side
            # effect still needs to happen, so this materializes it via
            # _ir_materialize_composite_call (never out of scope, unlike
            # _ir_append_call above -- it has no base/value argument of
            # its own that could be) and simply discards the resulting
            # address -- the identical "malloc when nothing reserved a
            # slot" fallback _ir_materialize_composite_call's own
            # docstring already documents for the addressable-base
            # case, reused here unchanged, with the result never read.
            ir, _ = self._ir_materialize_composite_call(stmt.expr, type_of(stmt.expr))
            return ir
        elif isinstance(stmt, ExprStmt):
            ir, _ = self.gen_expr_ir(stmt.expr)
            return ir
        return [IRRaw(self.gen_statement(stmt))]

    def _ir_malloc_and_store(self, var_type, offset: int) -> list:
        """Builds (without lowering) a heap-allocated local's own
        fresh backing allocation as real IR: an ordinary IRCall to
        malloc -- IRCall's own lowering is already fully generic over
        any callee name, external C library functions included, so
        this needs no new IR concept at all, the identical realization
        _ir_string_concat already relies on for strlen/strcpy/etc. --
        then an ordinary IRStore writing the returned pointer into
        this variable's own frame slot, via IRLocalAddress for the
        slot's own address. IRLocalAddress always means "the address
        of this slot," never "the value stored there" (see its own
        docstring), so writing to it via IRStore, rather than reading
        through it via IRLoad, is exactly the composition every other
        caller of a heap-allocated slot's own address already needs.

        Used wherever a fresh, heap-allocated array/struct local needs
        its own backing allocation made before its address is ever
        computed -- see this method's own four call sites in
        gen_statement_ir's own VarDecl case, one per fresh-value-
        producing shape (Variable/Field/Index copy, Slice production,
        append, an ordinary composite-returning call)."""
        size = type_byte_width(var_type, self.struct_registry)
        ptr = self._new_temp(Type.INT64)
        slot_addr = self._new_temp(Type.INT64)
        return [
            IRCall(dst=ptr, name='malloc', args=[IRConst(size, Type.INT64)]),
            IRLocalAddress(dst=slot_addr, offset=offset),
            IRStore(address=slot_addr, value=ptr, value_type=Type.INT64),
        ]

    def gen_var_decl(self, stmt: VarDecl) -> list[Instruction]:
        # _collect_locals already reserved this VarDecl's slot;
        # _bind_local just makes its name resolvable in the current
        # scope and returns where to store the initializer, if there
        # is one. `int a` with no initializer gets its type's implicit
        # zero value (see _gen_zero_value_into) rather than genuinely
        # uninitialized memory -- the same holds for a heap-allocated
        # array/struct's malloc'd memory below: always written through,
        # never left as raw malloc garbage.
        offset = self._bind_local(stmt)
        var_type = self._local_type(stmt.name)
        if self._is_heap_allocated(id(stmt), var_type):
            # A fresh backing allocation, made exactly once here at
            # declaration time (see gen_assign's array case for why a
            # later assignment reuses this allocation instead of
            # mallocing again). %rax still holds the pointer right
            # after storing it into the slot, so it's safe to use
            # directly as the initializer's destination.
            instructions = self._gen_malloc_array(var_type)
            instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', offset)))
            if stmt.init is not None:
                if var_type.kind == TypeKind.STRUCT:
                    instructions.extend(self.gen_struct_value_into(stmt.init, Memory('rax', 0), var_type))
                else:
                    instructions.extend(self.gen_array_value_into(stmt.init, Memory('rax', 0), var_type))
            else:
                instructions.extend(self._gen_zero_value_into(var_type, Memory('rax', 0)))
            return instructions
        if stmt.init is None:
            # A no-initializer variable's zero value is always written
            # directly to its permanent slot, never through _gen_write_
            # temp_from -- same reasoning as the parameter-initialization
            # case in gen_function's own prologue loop (see its own
            # comment for why this is recorded rather than migrated).
            self._escaped_offsets.add(offset)
            return self._gen_zero_value_into(var_type, Memory('rbp', offset))
        if isinstance(stmt.init, NoneLiteral):
            # none's resolved type (Type.NONE) never equals var_type --
            # semantic.py's _types_compatible is what lets this
            # declaration through despite that -- so this needs
            # var_type, the TARGET type, passed explicitly, rather than
            # going through _gen_store's ordinary dispatch, which only
            # needs the value expression since every other kind of
            # value's resolved type already matches what's being stored.
            return self.gen_none_into(Memory('rbp', offset), var_type)
        if isinstance(stmt.init, ArrayLiteral) and var_type.kind == TypeKind.SLICE:
            # `[]int s = [1, 2, 3]` -- an untyped array literal used
            # directly as a slice's initializer, treated like the
            # general, explicitly-typed form (`[]int s = []int[1, 2,
            # 3]`): construct a new, heap-allocated backing array and
            # produce a descriptor for the whole thing. Needed here
            # separately because stmt.init's resolved type
            # (Type(ARRAY,...)) never equals var_type (Type(SLICE,...)),
            # so _gen_store's ordinary dispatch, which trusts the
            # value's own resolved type, would never route this to
            # slice-producing codegen on its own.
            instructions = self.gen_array_literal_heap_alloc_into(stmt.init)
            instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', offset)))
            instructions.append(MovQ(src=Imm(len(stmt.init.elements)), dst=Memory('rbp', offset + 8)))
            return instructions
        return self._gen_store(offset, stmt.init)

    def gen_assign(self, stmt: Assign) -> list[Instruction]:
        offset = self._local_offset(stmt.name)
        if isinstance(stmt.value, NoneLiteral):
            # See gen_var_decl's identical case above: needs the
            # TARGET type (the variable's declared type), not
            # stmt.value's resolved type (Type.NONE).
            var_type = self._local_type(stmt.name)
            return self.gen_none_into(Memory('rbp', offset), var_type)
        var_type = self._local_type(stmt.name)
        if isinstance(stmt.value, ArrayLiteral) and var_type.kind == TypeKind.SLICE:
            # See gen_var_decl's identical case for the full reasoning
            # -- unlike an array's own Assign below, this always
            # mallocs a FRESH allocation rather than reusing an
            # existing one: an assigned-to slice variable might
            # currently point at a different array (or none at all) of
            # a different size, so there's no existing allocation here
            # that could be safe to reuse in place.
            instructions = self.gen_array_literal_heap_alloc_into(stmt.value)
            instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', offset)))
            instructions.append(MovQ(src=Imm(len(stmt.value.elements)), dst=Memory('rbp', offset + 8)))
            return instructions
        value_type = type_of(stmt.value)
        if value_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT) and self._is_heap_allocated(
                self._local_decl_id(stmt.name), value_type):
            # Reuses the EXISTING allocation from this variable's
            # declaration -- a fixed-size array's (or struct's) own
            # footprint never changes across its lifetime, so there's
            # nothing to reallocate, only to load the existing pointer
            # and write the new value through it.
            instructions = [MovQ(src=Memory('rbp', offset), dst=Register('rax'))]
            if value_type.kind == TypeKind.STRUCT:
                instructions.extend(self.gen_struct_value_into(stmt.value, Memory('rax', 0), value_type))
            else:
                instructions.extend(self.gen_array_value_into(stmt.value, Memory('rax', 0), value_type))
            return instructions
        return self._gen_store(offset, stmt.value)

    def gen_index_assign(self, stmt: IndexAssign) -> list[Instruction]:
        """`array[index] = value` -- computes the target element's
        address (via gen_index_address_into, which includes the
        runtime bounds check). A SLICE element (`rows[i] = someSlice`)
        needs its own 24-byte descriptor write via gen_slice_value_
        into, which already protects an arbitrary dst_mem.base
        internally, so this can hand it Memory('rax', 0) directly
        without its own push/pop dance. The scalar case is built as
        IR -- see _ir_index_assign, which is also what actually
        decides the store's width from the element's own DECLARED
        type, not stmt.value's.

        Deliberately NOT stmt.value's resolved type: an untyped array
        literal flowing into a SLICE-typed element (`rows[0] = [9, 9,
        9]`) has its resolved type set to the ARRAY it actually builds,
        not the slice it's being treated as -- dispatching on the
        VALUE's type would miss this case and fall through to the
        scalar path below (the same bug-class already fixed in
        gen_var_decl/gen_assign, just at a third call site).

        An ARRAY-typed element (`matrix[i] = other_row`) isn't
        reachable here: IndexAssign's grammar only ever produces a
        single leaf-level element write.
        """
        base_type = type_of(stmt.array)
        element_type = base_type.element_type
        addr_reg = Register('rax')
        instructions = self.gen_index_address_into(Index(array=stmt.array, index=stmt.index), addr_reg)
        if element_type.kind == TypeKind.SLICE:
            instructions.extend(self.gen_slice_value_into(stmt.value, Memory('rax', 0)))
            return instructions
        if element_type.kind == TypeKind.STRUCT:
            return instructions + self.gen_struct_value_into(stmt.value, Memory('rax', 0), element_type)
        return self._instruction_selector.lower_ir(self._ir_index_assign(stmt, element_type))

    def _ir_index_assign(self, stmt: IndexAssign, element_type) -> list:
        """Builds (without lowering) the scalar-element case of
        gen_index_assign -- the caller (gen_index_assign or
        gen_statement_ir) is responsible for already having ruled out
        SLICE/STRUCT. Captures the address via _ir_index_address_or_
        fallback (real IR wherever expr.array's own base allows it --
        see its own docstring for the genuinely-deferred cases that
        still fall back to an opaque leaf), builds the value's own IR
        (already works, via gen_expr_ir), then IRStores it through
        the address, at the ELEMENT's own declared width -- not
        necessarily the value's own, per IRStore's own docstring."""
        addr_ir, addr_value = self._ir_index_address_or_fallback(Index(array=stmt.array, index=stmt.index))
        value_ir, value = self.gen_expr_ir(stmt.value)
        return addr_ir + value_ir + [IRStore(address=addr_value, value=value, value_type=element_type)]

    def _ir_copy_into_address(self, dst_address, src_expr: Node, value_type) -> list:
        """The shared core of _ir_copy_assign: given an ALREADY-
        computed destination address (an ordinary IRValue -- however
        the caller has it: a freshly-computed _ir_array_address/_ir_
        struct_address/_ir_slice_address for the existing VarDecl/
        Assign/IndexAssign/FieldAssign callers, or the current
        function's own received hidden pointer for a forwarding
        `return someVar` -- see gen_statement_ir's own Return case),
        captures src_expr's own address the same way _ir_copy_assign
        always has, then IRCopies between them.

        Callers are responsible for already having confirmed src_expr
        is a Variable/Field/Index -- see _ir_copy_assign's own
        docstring for why an ArrayLiteral/struct-literal Call/Slice/
        composite-returning Call is a genuinely different case."""
        address_of = {
            TypeKind.ARRAY: self._ir_array_address,
            TypeKind.STRUCT: self._ir_struct_address,
            TypeKind.SLICE: self._ir_slice_address,
        }[value_type.kind]
        src_ir, src_addr = address_of(src_expr)
        return src_ir + [IRCopy(dst_address=dst_address, src_address=src_addr, value_type=value_type)]

    def _ir_copy_assign(self, dst_expr: Node, src_expr: Node, value_type) -> list:
        """Builds (without lowering) a whole-array/whole-struct/whole-
        slice copy as real IR: captures the destination's own address
        via _ir_array_address/_ir_struct_address/_ir_slice_address --
        real IR themselves (see their own docstrings), so a heap-
        allocated variable or a nested field/index destination is
        handled identically, with the address computation itself
        inspectable rather than opaque -- then delegates to _ir_copy_
        into_address for the rest (capturing src_expr's own address
        the same way, then IRCopy between them).

        Dispatches on value_type.kind, not on dst_expr/src_expr's own
        shape: SLICE has no old-style equivalent to fall back to at
        all (a slice's own address is only ever computed inline,
        inside gen_slice_value_into's own Variable/Index/Field cases,
        never as a standalone reusable method) -- so all three kinds
        go through the real-IR builders uniformly here, not a mix.

        Callers are responsible for already having confirmed src_expr
        is a Variable/Field/Index -- an ArrayLiteral, a struct-literal
        Call, a Slice (slice production), or an ordinary composite-
        returning Call is a genuinely different case (construction,
        or the hidden-output-pointer convention) with no existing
        address to capture at all, and stays on the old-style path
        instead; see gen_statement_ir's own VarDecl/Assign/
        IndexAssign/FieldAssign cases for where that check happens."""
        address_of = {
            TypeKind.ARRAY: self._ir_array_address,
            TypeKind.STRUCT: self._ir_struct_address,
            TypeKind.SLICE: self._ir_slice_address,
        }[value_type.kind]
        dst_ir, dst_addr = address_of(dst_expr)
        return dst_ir + self._ir_copy_into_address(dst_addr, src_expr, value_type)

    def gen_field_assign(self, stmt: FieldAssign) -> list[Instruction]:
        """`base.name = value` -- mirrors gen_index_assign one level
        over: computes the target field's address (via
        gen_field_address_into). The scalar case is built as IR -- see
        _ir_field_assign, which is also what actually decides the
        store's width from the field's own DECLARED type, not
        stmt.value's -- for the same reason gen_index_assign uses the
        element's declared type.

        A STRUCT-typed field (`s.inner = otherInner`) is handled via
        gen_struct_value_into's flat copy, since a field write of a
        whole struct value is exactly as much "copy N bytes" as any
        other struct value production. An array-typed field (`s.arr =
        otherArr`) works the same way via gen_array_value_into --
        unlike IndexAssign, FieldAssign's grammar CAN produce this
        shape (a struct field can itself be a whole array), so this
        needs a real case for it.

        A NoneLiteral value flowing into a slice-typed field
        (`s.values = none`) needs the identical short-circuit gen_var_
        decl/gen_assign already have, checked BEFORE the SLICE
        dispatch: none's resolved type (Type.NONE) never equals the
        field's declared type, so gen_slice_value_into's ordinary
        dispatch has no case for it. This was a real gap found by
        testing: FieldAssign wasn't a reachable path for a slice-typed
        value until slice-typed fields existed at all."""
        field_type = self._check_struct_and_field_type(stmt.base, stmt.name)
        addr_reg = Register('rax')
        instructions = self.gen_field_address_into(stmt, addr_reg)
        if field_type.kind == TypeKind.SLICE:
            if isinstance(stmt.value, NoneLiteral):
                return instructions + self.gen_none_into(Memory('rax', 0), field_type)
            instructions.extend(self.gen_slice_value_into(stmt.value, Memory('rax', 0)))
            return instructions
        if field_type.kind == TypeKind.STRUCT:
            return instructions + self.gen_struct_value_into(stmt.value, Memory('rax', 0), field_type)
        if field_type.kind == TypeKind.ARRAY:
            return instructions + self.gen_array_value_into(stmt.value, Memory('rax', 0), field_type)
        return self._instruction_selector.lower_ir(self._ir_field_assign(stmt, field_type))

    def _ir_field_assign(self, stmt: FieldAssign, field_type) -> list:
        """Builds (without lowering) the scalar-field case of
        gen_field_assign -- the caller (gen_field_assign or
        gen_statement_ir) is responsible for already having ruled out
        SLICE/STRUCT/ARRAY (and the slice-typed NoneLiteral case).
        Same shape as _ir_index_assign one level over: captures the
        address via _ir_field_address (real IR -- a struct-typed
        base's own shape is always Variable/Field/Index, never
        needing a fallback the way an array/slice base can; see its
        own docstring), builds the value's own IR, then IRStores it
        through the address at the FIELD's own declared width."""
        addr_ir, addr_value = self._ir_field_address(Field(base=stmt.base, name=stmt.name))
        value_ir, value = self.gen_expr_ir(stmt.value)
        return addr_ir + value_ir + [IRStore(address=addr_value, value=value, value_type=field_type)]

    def _gen_store(self, offset: int, value_expr: Node) -> list[Instruction]:
        """Shared by VarDecl-with-initializer and Assign: both are just
        "compute this expression, then write the result into that
        variable's slot". Which store instruction depends on the
        value's type: an array or struct can't fit in a single
        register, so each is dispatched to gen_array_value_into or
        gen_struct_value_into separately; a slice is a fixed-size
        24-byte descriptor, dispatched to gen_slice_value_into the same
        way; a str is an 8-byte pointer sitting in %rax and needs
        `movq`; int/bool/int8/uint8 all compute the same way (via
        gen_expr_into, oblivious to which of the four it actually is)
        and then write out via _gen_write_scalar_from, the one place
        that distinguishes a narrow 1-byte store (int8/uint8) from an
        ordinary 4-byte one -- only this call site needs to ask "which
        width, or which entirely different mechanism, am I storing"."""
        value_type = type_of(value_expr)
        if value_type.kind == TypeKind.ARRAY:
            return self.gen_array_value_into(value_expr, Memory('rbp', offset), value_type)
        if value_type.kind == TypeKind.SLICE:
            return self.gen_slice_value_into(value_expr, Memory('rbp', offset))
        if value_type.kind == TypeKind.STRUCT:
            return self.gen_struct_value_into(value_expr, Memory('rbp', offset), value_type)
        instructions = self.gen_expr_into(value_expr, Register('eax'))
        if value_type == Type.STR:
            instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', offset)))
        else:
            instructions.extend(self._gen_write_scalar_from(Register('eax'), value_type, Memory('rbp', offset)))
        return instructions

    def gen_return(self, stmt: Return) -> list[Instruction]:
        # A bare `return` (no value -- valid exactly when this function
        # has no declared return type) needs nothing computed, just
        # the ordinary epilogue -- IRReturn with no value, same as the
        # scalar case below minus the load.
        if stmt.value is None:
            return self._instruction_selector.lower_ir(self._ir_return(None))

        if isinstance(stmt.value, NoneLiteral):
            # none's resolved type (Type.NONE) never equals SLICE --
            # semantic.py's _types_compatible is what lets `return
            # none` through despite that, and already guarantees it's
            # only valid when this function's declared return type IS
            # a slice. Written directly (not via gen_none_into, which
            # needs a real target_type to check against, not readily
            # available here) through the hidden return pointer, like
            # every other slice-typed return value.
            ptr_reg = Register('rax')
            instructions = [MovQ(src=Memory('rbp', self._hidden_return_ptr_offset), dst=ptr_reg)]
            instructions.append(MovQ(src=Imm(0), dst=Memory('rax', 0)))
            instructions.append(MovQ(src=Imm(0), dst=Memory('rax', 8)))
            instructions.append(MovQ(src=Imm(0), dst=Memory('rax', 16)))
            instructions.extend(self._gen_epilogue())
            return instructions

        # An array- OR slice-typed return writes directly through the
        # hidden pointer this function received instead of ever
        # putting anything in %eax/%rax -- nothing reads a return
        # value that way for an array- or slice-returning call.
        # Loading the pointer back out of its slot and handing it to
        # gen_array_value_into/gen_slice_value_into as an ordinary
        # Memory destination is also what makes `return bar()`
        # (forwarding another array/slice-returning call's result)
        # free: the Call case just passes that same address one level
        # deeper, with no intermediate copy ever materialized.
        #
        # Real IR now instead, for the Call-forwarding case (via _ir_
        # composite_call), a Variable/Field/Index value (via _ir_copy_
        # into_address), and an array literal or positional struct
        # literal (via _ir_write_array_literal_into/_ir_write_struct_
        # literal_into) alike -- including a NESTED composite element/
        # field within either, via _ir_write_composite_value_into,
        # which those two now call back into for their own composite
        # elements/fields, mutually recursively. A named/partial
        # (kwargs) struct literal, or `return none`, still reach this
        # old-style path. See gen_statement_ir's own Return case,
        # which reads the identical hidden_return_ptr_offset via an
        # ordinary address-as-a-Temp leaf (_ir_hidden_return_ptr),
        # then reuses whichever real-IR builder applies, rather than
        # calling this method at all.
        value_type = type_of(stmt.value)
        if value_type.kind == TypeKind.ARRAY:
            ptr_reg = Register('rax')
            instructions = [MovQ(src=Memory('rbp', self._hidden_return_ptr_offset), dst=ptr_reg)]
            instructions.extend(self.gen_array_value_into(stmt.value, Memory('rax', 0), value_type))
        elif value_type.kind == TypeKind.SLICE:
            ptr_reg = Register('rax')
            instructions = [MovQ(src=Memory('rbp', self._hidden_return_ptr_offset), dst=ptr_reg)]
            instructions.extend(self.gen_slice_value_into(stmt.value, Memory('rax', 0)))
        elif value_type.kind == TypeKind.STRUCT:
            # Same hidden-pointer mechanism -- unchanged for struct:
            # gen_struct_value_into already knows how to write into an
            # arbitrary Memory destination, so `return bar()` is
            # exactly as free here as it is for arrays and slices.
            ptr_reg = Register('rax')
            instructions = [MovQ(src=Memory('rbp', self._hidden_return_ptr_offset), dst=ptr_reg)]
            instructions.extend(self.gen_struct_value_into(stmt.value, Memory('rax', 0), value_type))
        else:
            # A scalar return: build its IR (see _ir_return) and lower
            # it -- which loads the value into %eax (or %rax, per its
            # type) and emits the ordinary epilogue. None of the
            # epilogue touches %eax/%rax/%rdx, so this is unaffected by
            # whatever those registers held during the body.
            return self._instruction_selector.lower_ir(self._ir_return(stmt.value))
        instructions.extend(self._gen_epilogue())
        return instructions

    def _ir_return(self, value_expr) -> list:
        """Builds (without lowering) IRReturn for a bare return
        (value_expr=None) or a scalar return -- the two cases
        gen_return doesn't route through the hidden-pointer
        convention."""
        if value_expr is None:
            return [IRReturn(value=None)]
        ir, value = self.gen_expr_ir(value_expr)
        return ir + [IRReturn(value=value)]

    def _ir_hidden_return_ptr(self) -> tuple:
        """Reads the current function's own received hidden return
        pointer (for a composite-returning function, passed by ITS
        OWN caller, at this function's own fixed, known hidden_
        return_ptr_offset) -- returns (ir, address). IRLocalAddress
        for the slot's own address, then an ordinary IRLoad reading
        the pointer stored there (IRLocalAddress always means "the
        address of this slot," never "the value stored there" --
        see its own docstring). Shared by every gen_statement_ir
        Return case that forwards into it (an ordinary function call,
        or a Variable/Field/Index value) -- neither cares how the
        address was obtained, only that it's an ordinary IRValue."""
        slot_addr = self._new_temp(Type.INT64)
        hidden_ptr = self._new_temp(Type.INT64)
        ir = [
            IRLocalAddress(dst=slot_addr, offset=self._hidden_return_ptr_offset),
            IRLoad(dst=hidden_ptr, address=slot_addr),
        ]
        return ir, hidden_ptr

    def gen_if(self, stmt: If) -> list[Instruction]:
        """Computes the condition into a Temp and branches on it via
        IRBranch:

            <condition>          ; -> t_cond
            branch t_cond, .then, .else
        .then:
            <then_body>
            jump .end
        .else:
            <else_body>          ; only emitted if else_body is present
        .end:

        (lower_ir's own IRBranch rule always emits both jumps, so the
        actual assembly has one redundant jmp right before .then -- no
        peephole yet.)

        then_body and else_body each get their own pushed/popped scope
        (see _push_scope), matching semantic.py's independent-branch
        scoping -- and since an elif is just a nested If inside
        else_body, gen_statement's ordinary recursion handles a whole
        elif/else chain of any length with no extra logic here.
        """
        then_label = self.new_label("if_then")
        else_label = self.new_label("if_else")
        end_label = self.new_label("if_end")

        instructions = self._instruction_selector.lower_ir(self._ir_if_head(stmt, then_label, else_label))

        self._push_scope()
        for s in stmt.then_body:
            instructions.extend(self.gen_statement(s))
        self._pop_scope()

        instructions.extend(self._instruction_selector.lower_ir([IRJump(end_label), IRLabel(else_label)]))
        if stmt.else_body is not None:
            self._push_scope()
            for s in stmt.else_body:
                instructions.extend(self.gen_statement(s))
            self._pop_scope()
        instructions.extend(self._instruction_selector.lower_ir([IRLabel(end_label)]))
        return instructions

    def _ir_if_head(self, stmt: If, then_label: str, else_label: str) -> list:
        """Builds (without lowering) the condition-and-branch IR
        landing at the given then/else labels -- the caller (gen_if)
        is responsible for the bodies and the trailing jump/labels
        around them, since those still go through gen_statement, not
        native IR."""
        cond_ir, cond_value = self.gen_expr_ir(stmt.condition)
        return cond_ir + [
            IRBranch(cond=cond_value, true_label=then_label, false_label=else_label),
            IRLabel(then_label),
        ]

    def gen_while(self, stmt: While) -> list[Instruction]:
        """Computes the condition, re-checked before every iteration
        (including the first), with the body sitting between the
        start label (break/continue's own re-check target) and the
        end label (loop exit):

        .start:
            <condition>          ; -> t_cond
            branch t_cond, .body, .end
        .body:
            <body>
            jump .start
        .end:

        Both labels get pushed onto self.loop_labels for the body's
        duration, so any Break/Continue inside it -- including nested
        inside an If -- finds its way back via gen_break/gen_continue
        with no need to know where inside the body it is. Popped again
        once the body's done, so a Break/Continue after this while
        can't resolve to this loop's labels.

        The body gets its own pushed/popped scope, same as an If's
        then/else bodies, even though the same physical stack slots
        are reused on every iteration -- this is purely about name
        resolution during codegen, not anything that happens at
        runtime.
        """
        start_label = self.new_label("while_start")
        body_label = self.new_label("while_body")
        end_label = self.new_label("while_end")

        instructions = self._instruction_selector.lower_ir(self._ir_while_head(stmt, start_label, body_label, end_label))

        self.loop_labels.append((start_label, end_label))
        self._push_scope()
        for s in stmt.body:
            instructions.extend(self.gen_statement(s))
        self._pop_scope()
        self.loop_labels.pop()

        instructions.extend(self._instruction_selector.lower_ir([IRJump(start_label), IRLabel(end_label)]))
        return instructions

    def _ir_while_head(self, stmt: While, start_label: str, body_label: str, end_label: str) -> list:
        """Builds (without lowering) the start-label/condition/branch
        IR landing at the given body label -- the caller (gen_while)
        is responsible for the body and the trailing jump/end-label
        around it, since those still go through gen_statement, not
        native IR."""
        cond_ir, cond_value = self.gen_expr_ir(stmt.condition)
        return [IRLabel(start_label)] + cond_ir + [
            IRBranch(cond=cond_value, true_label=body_label, false_label=end_label),
            IRLabel(body_label),
        ]

    def gen_break(self, stmt: Break) -> list[Instruction]:
        # semantic.py already guarantees this only appears inside a
        # loop; this check exists so codegen doesn't trust semantic
        # analysis unconditionally, the same defensive posture
        # _local_offset takes.
        #
        # Real IR now instead -- see _ir_break, an exact one-line swap
        # (IRJump instead of Jmp; IRJump's own lowering already emits
        # exactly this same Jmp, see its own docstring). This old-
        # style path is still reached from gen_statement, the old-
        # style statement dispatch.
        if not self.loop_labels:
            raise CodegenError("'break' outside of a loop")
        _, end_label = self.loop_labels[-1]
        return [Jmp(end_label)]

    def _ir_break(self) -> list:
        """The real-IR counterpart to gen_break: identical loop-label
        lookup and the identical defensive check, just IRJump instead
        of a raw Jmp -- IRJump's own lowering already emits exactly
        that same Jmp (see codegen/ir.py's own docstring), so this is
        a direct, zero-behavioral-difference swap, not a translation
        needing any new reasoning."""
        if not self.loop_labels:
            raise CodegenError("'break' outside of a loop")
        _, end_label = self.loop_labels[-1]
        return [IRJump(end_label)]

    def gen_continue(self, stmt: Continue) -> list[Instruction]:
        # Real IR now instead -- see _ir_continue, the identical one-
        # line swap _ir_break's own docstring describes. This old-
        # style path is still reached from gen_statement.
        if not self.loop_labels:
            raise CodegenError("'continue' outside of a loop")
        start_label, _ = self.loop_labels[-1]
        return [Jmp(start_label)]

    def _ir_continue(self) -> list:
        """The real-IR counterpart to gen_continue -- see _ir_break's
        own docstring for why this is a direct, zero-behavioral-
        difference swap."""
        if not self.loop_labels:
            raise CodegenError("'continue' outside of a loop")
        start_label, _ = self.loop_labels[-1]
        return [IRJump(start_label)]

    def gen_expr_stmt(self, stmt: ExprStmt) -> list[Instruction]:
        # Evaluated the same way as any other expression, into %eax --
        # just with nothing done with the result afterward. Still real
        # instructions that really run (a standalone `1 / 0` genuinely
        # crashes).
        #
        # An ArrayLiteral is the one exception: it can't be computed
        # via gen_expr_into at all (doesn't fit in a single register),
        # and unlike a VarDecl/Assign's use of one, a bare literal
        # statement has no destination to write the resulting array
        # into -- but it doesn't need one, since nothing ever reads the
        # array as a whole. See gen_array_literal_side_effects_only for
        # the resulting approach: evaluate each element for whatever
        # side effects it might have, without materializing a real
        # array in memory.
        if isinstance(stmt.expr, ArrayLiteral):
            return self.gen_array_literal_side_effects_only(stmt.expr)
        # A Slice expression is the analogous exception for slices --
        # a 24-byte descriptor doesn't fit in a register either -- but
        # unlike ArrayLiteral, this doesn't need its own narrower path:
        # gen_slice_into already computes fully correctly into any
        # Memory destination, including a genuine runtime bounds check
        # (an out-of-range bound still aborts here), so this just
        # reuses the same per-function scratch slot gen_indexable_
        # base_into's Slice-base case already uses
        # (_unnamed_slice_temp_offset) and discards the result. Covers
        # both a bare slice LITERAL statement and an ordinary bare
        # slice of an existing array or slice (`arr[:]` alone,
        # pointless but not an error) with the same code path.
        if isinstance(stmt.expr, Slice):
            return self.gen_slice_into(stmt.expr, Memory('rbp', self._unnamed_slice_temp_offset))
        return self.gen_expr_into(stmt.expr, Register('eax'))

    def _gen_zero_value_into(self, t: Type, dst_mem: Memory) -> list[Instruction]:
        """Writes t's implicit zero value into dst_mem -- what a `T x`
        VarDecl with no initializer gets, instead of genuinely
        uninitialized memory. Dispatches by kind:
          - int/bool/int8/uint8: an ordinary 0 -- a plain 4-byte write
            for int/bool, a 1-byte one (MovB) for int8/uint8, matching
            each type's genuine storage width.
          - str: the address of a single shared, static empty-string
            constant (_get_empty_str_label) -- never a null pointer
            (see that method for why a null zero value would be an
            active hazard).
          - slice: none's {ptr: 0, len: 0, cap: 0} descriptor, reusing
            gen_none_into as-is -- a zero-value slice and a none-valued
            one are, by design, the identical representation.
          - array: delegated to _gen_zero_array_into, which further
            dispatches on the array's leaf type.
          - struct: every field, flattened via _flatten_struct_fields
            the same way struct equality flattens them for comparison,
            recursing back into this method for each field's type.

        dst_mem.base is protected via push/pop across EVERY field's
        zero-fill, when it isn't 'rbp': the array case computes a fresh
        address via _gen_address_of_memory_into with dst_mem.base
        itself as the destination register in some call shapes, which
        can overwrite dst_mem.base's physical register in place.
        Without protecting it, a struct with an array-typed field
        followed by any other field would silently compute that later
        field's address from garbage instead of the struct's real base
        -- the same register-collision failure mode
        _gen_struct_fields_equality_at_addresses guards against for the
        identical reason. Applied unconditionally, even for the
        scalar/slice cases that don't strictly need it.

        Real IR now instead, for a no-initializer array/struct VarDecl
        (see gen_statement_ir's own VarDecl case) and an omitted field
        in a named/partial struct literal (see _ir_write_struct_
        literal_into) alike -- see _ir_write_zero_value_into, which
        needs none of this method's own careful fixed-register
        (%r10/%r12/%r13/...) protection across a recursive call, or
        three separately hand-written array-leaf loop variants: every
        Temp its own array-leaf loop uses is independent of whatever
        Temps a recursive call allocates for itself, and one general-
        purpose loop already dispatches uniformly on any leaf type,
        rather than needing a separate flat-zero/str-address/struct-
        recursive loop for each. This old-style path is still reached
        from an ArrayLiteral/struct-literal-Call-valued VarDecl/
        Assign/IndexAssign/FieldAssign (still old-style itself, so its
        own zero-init needs -- an omitted field's own zero value, for
        instance -- stay old-style too) and gen_var_decl's own
        remaining callers.
        """
        if t.kind == TypeKind.STRUCT:
            protect_dst = dst_mem.base != 'rbp'
            instructions = []
            for field_type, offset in self._flatten_struct_fields(t.struct_name):
                field_mem = Memory(dst_mem.base, dst_mem.offset + offset)
                if protect_dst:
                    instructions.append(Push(Register(dst_mem.base)))
                instructions.extend(self._gen_zero_value_into(field_type, field_mem))
                if protect_dst:
                    instructions.append(Pop(Register(dst_mem.base)))
            return instructions
        if t.kind == TypeKind.ARRAY:
            return self._gen_zero_array_into(t, dst_mem)
        if t.kind == TypeKind.SLICE:
            return self.gen_none_into(dst_mem, t)
        if t == Type.STR:
            # Whichever of rax/rcx isn't dst_mem's own base -- a single
            # scratch register is all this needs, computed and consumed
            # in the same two instructions, with nothing relying on it
            # afterward.
            scratch = Register('rax') if dst_mem.base != 'rax' else Register('rcx')
            return [
                LeaQ(label=self._get_empty_str_label(), dst=scratch),
                MovQ(src=scratch, dst=dst_mem),
            ]
        if t == Type.INT8 or t == Type.UINT8:
            return [MovB(src=Imm(0), dst=dst_mem)]
        return [Mov(src=Imm(0), dst=dst_mem)]  # int or bool

