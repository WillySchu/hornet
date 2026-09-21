"""Statement-level codegen: declarations, assignment, control flow
(if/while/break/continue), and return. Establishes that an
uninitialized declaration gets its type's real zero value rather than
leaving memory untouched, and the label-pair shape (start/end, or
else/end) every branching or looping construct here builds on."""

from ir.errors import IRError
from ir.ir import (
    IRBranch,
    IRCall,
    IRConst,
    IRCopy,
    IRFunction,
    IRJump,
    IRLabel,
    IRLoad,
    IRLocalAddress,
    IRMove,
    IRReturn,
    IRStaticDataAddress,
    IRStore,
)
from ir.utils import COMPOSITE_KINDS, type_of, type_byte_width
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
    def gen_statement_ir(self, stmt: Node, ir_fn: IRFunction) -> list:
        """Builds real IR for every statement kind: Return/If/While
        (recursing into itself for If/While bodies), a scalar VarDecl-
        with-initializer, Assign, IndexAssign, or FieldAssign (an
        IRMove into a variable's own Temp, or an IRStore through an
        address), and a bare expression statement (via gen_expr_ir,
        discarding the result).

        An array/struct/slice-typed VarDecl/Assign/IndexAssign/
        FieldAssign is also real IR, across several value shapes: a
        Variable/Field/Index value (IRCopy between two captured
        addresses -- _ir_copy_assign); a Slice production or an
        append(...) call (slice-typed target only -- _ir_slice_into/
        _ir_append_call plus _ir_write_slice_descriptor); an ordinary
        composite-returning function call, via the hidden-pointer
        convention (_ir_composite_call), writing directly through the
        target's own address; and an array or struct literal,
        positional or named/partial (_ir_write_array_literal_into/
        _ir_write_struct_literal_into/_ir_write_composite_value_into).
        A VarDecl with no initializer gets its type's real zero value
        (_ir_write_zero_value_into/_ir_zero_array_loop for array/
        struct, three IRConst(0)s via _ir_write_slice_descriptor for a
        slice's nil value).

        Return's own composite case covers forwarding another
        composite-returning call's result (_ir_composite_call), a
        Variable/Field/Index value (_ir_copy_into_address), and an
        array or struct literal -- all reuse the current function's
        own received hidden pointer (_ir_hidden_return_ptr). A
        literal's own elements/fields need not be scalar: a nested
        composite is handled by _ir_write_composite_value_into, mutual
        recursion with the two literal-writing methods.

        Break/Continue are real IR too (_ir_break/_ir_continue), as is
        an ArrayLiteral- or Slice-valued ExprStmt (evaluated for side
        effects/the bounds check alone, with nothing materialized or
        read back).

        A slice-typed destination whose own value is a bare `none` is
        real IR via _ir_nil_slice (an all-zero {ptr, len, cap} triple)
        -- an ARRAY/STRUCT-typed destination can never BE `none` at
        all, semantic.py already rejects that.

        Everything else out of scope (a Return whose value is an
        ArrayLiteral/struct-literal Call with some field/element out
        of scope for _ir_write_composite_value_into; a Slice-valued
        ExprStmt whose own base is out of scope for _ir_slice_into)
        raises IRError explicitly."""
        if isinstance(stmt, Return):
            is_composite_return = isinstance(stmt.value, NoneLiteral) or (
                stmt.value is not None and type_of(stmt.value).kind in COMPOSITE_KINDS
            )
            if not is_composite_return:
                return self._ir_return(stmt.value)
            # A composite return whose own value is a bare `none` --
            # only reachable for a SLICE-typed return (ARRAY/STRUCT can
            # never be `none` -- semantic.py rejects that outright).
            # _ir_nil_slice's own all-zero triple, written through the
            # hidden pointer as an ordinary slice descriptor.
            if isinstance(stmt.value, NoneLiteral):
                hidden_ptr_ir, hidden_ptr = self._ir_hidden_return_ptr(ir_fn)
                nil_ir, ptr_value, len_value, cap_value = self._ir_nil_slice()
                write_ir = self._ir_write_slice_descriptor_into_address(hidden_ptr, ptr_value, len_value, cap_value)
                return hidden_ptr_ir + nil_ir + write_ir + [IRReturn(value=None)]
            # A sum-typed return -- checked, and handled completely,
            # BEFORE any of the shape-based dispatch below: every case
            # from here down uses type_of(stmt.value) (the VALUE's own
            # type) as what to write through the hidden pointer, which
            # is exactly WRONG here the same way it was in VarDecl's
            # own dispatch (see the identical comment there) -- `return
            # Circle(5)` from a function declared to return Shape needs
            # ir_fn.return_type (Shape), not type_of(stmt.value)
            # (Circle), or the hidden pointer gets Circle's own raw
            # field bytes with no discriminant tag at all.
            if ir_fn.return_type.kind == TypeKind.SUM:
                hidden_ptr_ir, hidden_ptr = self._ir_hidden_return_ptr(ir_fn)
                write_ir = self._ir_write_sum_type_value_into(hidden_ptr, stmt.value, ir_fn.return_type)
                if write_ir is not None:
                    return hidden_ptr_ir + write_ir + [IRReturn(value=None)]
            # A composite return whose own value is an ordinary
            # function call (`return someFn()`) -- forwards the
            # CURRENT function's own received hidden pointer straight
            # through as the destination for the INNER call, with no
            # intermediate copy materialized. IRReturn(value=None)
            # still runs the epilogue -- the result is already fully
            # written through the pointer by the time control reaches
            # it.
            if (
                    isinstance(stmt.value, Call)
                    and stmt.value.name != 'append'
                    and stmt.value.name not in self.ir_program.struct_registry
            ):
                hidden_ptr_ir, hidden_ptr = self._ir_hidden_return_ptr(ir_fn)
                call_ir = self._ir_composite_call(hidden_ptr, stmt.value)
                return hidden_ptr_ir + call_ir + [IRReturn(value=None)]
            # A composite return whose own value is a Variable/Field/
            # Index (an existing value with a real address to copy
            # from) -- via _ir_copy_into_address directly, this
            # function's own hidden pointer standing in for a freshly-
            # computed destination address.
            if isinstance(stmt.value, (Variable, Field, Index)):
                value_type = type_of(stmt.value)
                hidden_ptr_ir, hidden_ptr = self._ir_hidden_return_ptr(ir_fn)
                copy_ir = self._ir_copy_into_address(hidden_ptr, stmt.value, value_type)
                return hidden_ptr_ir + copy_ir + [IRReturn(value=None)]
            # A composite return whose own value is an array literal,
            # a bare bracketed-list literal resolved to SLICE by this
            # function's own declared return type, or a POSITIONAL
            # struct literal, with every element/field scalar -- writes
            # directly through the hidden pointer via _ir_write_array_
            # literal_into/_ir_slice_literal/_ir_write_struct_literal_
            # into. All return None when out of scope: a nested
            # composite element/field, or named/partial (kwargs)
            # struct construction -- both still need their own
            # recursive real-IR treatment, a separate, later step.
            if isinstance(stmt.value, ArrayLiteral):
                # Dispatches on ir_fn.return_type (this function's own
                # DECLARED return type), NOT type_of(stmt.value): the
                # latter is always ARRAY-kind for an ArrayLiteral,
                # regardless of what the surrounding context (here,
                # this function's own signature) resolves the overall
                # expression to.
                if ir_fn.return_type.kind == TypeKind.ARRAY:
                    hidden_ptr_ir, hidden_ptr = self._ir_hidden_return_ptr(ir_fn)
                    write_ir = self._ir_write_array_literal_into(hidden_ptr, stmt.value, ir_fn.return_type)
                    if write_ir is not None:
                        return hidden_ptr_ir + write_ir + [IRReturn(value=None)]
                elif ir_fn.return_type.kind == TypeKind.SLICE:
                    hidden_ptr_ir, hidden_ptr = self._ir_hidden_return_ptr(ir_fn)
                    production = self._ir_slice_literal(stmt.value)
                    if production is not None:
                        slice_ir, ptr_value, len_value, cap_value = production
                        write_ir = self._ir_write_slice_descriptor_into_address(
                            hidden_ptr, ptr_value, len_value, cap_value)
                        return hidden_ptr_ir + slice_ir + write_ir + [IRReturn(value=None)]
            if isinstance(stmt.value, Call) and stmt.value.name in self.ir_program.struct_registry:
                value_type = type_of(stmt.value)
                hidden_ptr_ir, hidden_ptr = self._ir_hidden_return_ptr(ir_fn)
                write_ir = self._ir_write_struct_literal_into(hidden_ptr, stmt.value, value_type)
                if write_ir is not None:
                    return hidden_ptr_ir + write_ir + [IRReturn(value=None)]
            # A composite return whose own value is a bare Slice node
            # (`return arr[low:high]`, a genuine re-slice, or `return
            # []int[1, 2, 3]`, which resolves to Slice(array=
            # ArrayLiteral(...), low=None, high=None) rather than a
            # bare ArrayLiteral -- a typed bracketed literal is always
            # parsed this way) -- writes through the hidden pointer via
            # _ir_slice_into.
            if isinstance(stmt.value, Slice):
                hidden_ptr_ir, hidden_ptr = self._ir_hidden_return_ptr(ir_fn)
                production = self._ir_slice_into(stmt.value)
                if production is not None:
                    slice_ir, ptr_value, len_value, cap_value = production
                    write_ir = self._ir_write_slice_descriptor_into_address(
                        hidden_ptr, ptr_value, len_value, cap_value)
                    return hidden_ptr_ir + slice_ir + write_ir + [IRReturn(value=None)]
            # Every OTHER composite return shape (an array literal or
            # struct literal with a nested composite element/field, a
            # named/partial struct literal, or `return none`) falls
            # through to this method's own final IRError at the
            # bottom -- a real, deliberate scope boundary.
        elif isinstance(stmt, If):
            then_label = self.ir_program.ids.new_label("if_then")
            else_label = self.ir_program.ids.new_label("if_else")
            end_label = self.ir_program.ids.new_label("if_end")
            ir = self._ir_if_head(stmt, then_label, else_label)
            self._push_scope()
            for s in stmt.then_body:
                ir.extend(self.gen_statement_ir(s, ir_fn))
            self._pop_scope()
            ir.append(IRJump(end_label))
            ir.append(IRLabel(else_label))
            if stmt.else_body is not None:
                self._push_scope()
                for s in stmt.else_body:
                    ir.extend(self.gen_statement_ir(s, ir_fn))
                self._pop_scope()
            # Symmetric with the then-branch's own IRJump(end_label)
            # right above, for the identical reason: without this, an
            # else-less If (else_body is None -- two labels back to
            # back, nothing between them) or an else body whose last
            # statement isn't already a jump/branch/return both fall
            # straight through into end_label, rather than reaching it
            # via an explicit terminator -- the same implicit-
            # fallthrough gap _ir_while_head's own docstring explains,
            # just this op's own version of it. Possibly redundant
            # when the else body already ends in a real terminator on
            # every path (an unconditional return, say) -- exactly as
            # harmless as the then-branch's own identical jump already
            # is in that same situation, an unreachable op costing a
            # few bytes, not a correctness concern.
            ir.append(IRJump(end_label))
            ir.append(IRLabel(end_label))
            return ir
        elif isinstance(stmt, While):
            start_label = self.ir_program.ids.new_label("while_start")
            body_label = self.ir_program.ids.new_label("while_body")
            end_label = self.ir_program.ids.new_label("while_end")
            ir = self._ir_while_head(stmt, start_label, body_label, end_label)
            self.loop_labels.append((start_label, end_label))
            self._push_scope()
            for s in stmt.body:
                ir.extend(self.gen_statement_ir(s, ir_fn))
            self._pop_scope()
            self.loop_labels.pop()
            ir.append(IRJump(start_label))
            ir.append(IRLabel(end_label))
            return ir
        elif isinstance(stmt, VarDecl):
            # A scalar VarDecl, WITH or without an initializer: bind
            # the variable (creating its own persistent Temp -- see
            # _bind_local) and write into that Temp directly, either
            # the initializer's own IR result or -- no initializer --
            # this type's own zero value. An array/struct/slice-typed
            # VarDecl (initialized or not) falls through to the more
            # specific cases below WITHOUT binding here -- each of
            # those does its own single _bind_local call once its own
            # shape is confirmed to apply, and binding twice would
            # just orphan a Temp id, harmlessly but pointlessly.
            var_type = type_from_name(stmt.var_type, self.ir_program.struct_registry, self.ir_program.type_alias_registry, sum_types=self.ir_program.sum_type_registry)
            if not isinstance(stmt.init, NoneLiteral) and var_type.kind not in COMPOSITE_KINDS:
                self._bind_local(stmt, ir_fn)
                if stmt.init is not None:
                    ir, value = self.gen_expr_ir(stmt.init)
                    return ir + [IRMove(dst=self._local_temp(stmt.name), src=value)]
                else:
                    # No initializer at all -- write this type's own
                    # zero value straight into the Temp _bind_local
                    # just created. Never routed through _ir_write_
                    # zero_value_into: that method's own contract is
                    # writing THROUGH an address, which doesn't apply
                    # here -- a named scalar variable already has its
                    # own Temp, never an address.
                    #
                    # str is the one scalar kind whose own zero value
                    # isn't a raw IRConst(0, ...): it's the address of
                    # a shared, static empty-string constant, never a
                    # null pointer (see _ir_write_zero_value_into's own
                    # str case).
                    if var_type == Type.STR:
                        return [IRStaticDataAddress(dst=self._local_temp(stmt.name), label=self._get_empty_str_label())]
                    return [IRMove(dst=self._local_temp(stmt.name), src=IRConst(0, var_type))]
            # A sum-typed VarDecl -- always HAS an initializer
            # (semantic.py's analyze_var_decl already rejects the bare
            # `Shape s` form, since a sum type has no natural zero
            # value), so there's no "no initializer" case to handle
            # here the way ARRAY/STRUCT/SLICE below each need one.
            # Checked, and handled completely, BEFORE the Variable/
            # Field/Index case right below -- deliberately: that case
            # ALSO matches (var_type.kind in COMPOSITE_KINDS now
            # includes SUM), but _ir_copy_assign's own flat, same-
            # shape copy is exactly wrong here for the identical
            # reason _ir_write_composite_value_into's own SUM check
            # (see ir/sum_types.py) has to come first there too: a
            # Circle-typed variable's own address has no discriminant
            # tag and isn't Shape-shaped at all. _ir_write_sum_type_
            # value_into handles BOTH a struct-literal Call and an
            # already-struct-typed Variable/Field/Index init
            # uniformly, by recursing back into _ir_write_composite_
            # value_into for the payload once the tag's written.
            if var_type.kind == TypeKind.SUM:
                slot = self._bind_local(stmt, ir_fn)
                ir = []
                if self._is_heap_allocated(id(stmt), var_type):
                    ir.extend(self._ir_malloc_and_store(var_type, slot))
                dst_ir, dst_address = self._ir_struct_address(Variable(name=stmt.name))
                ir.extend(dst_ir)
                write_ir = self._ir_write_sum_type_value_into(dst_address, stmt.init, var_type)
                if write_ir is not None:
                    return ir + write_ir
            # An array/struct/slice-typed initializer that's itself a
            # Variable/Field/Index (an existing value with a real
            # address to copy from) -- see _ir_copy_assign.
            if (
                    var_type.kind in COMPOSITE_KINDS
                    and isinstance(stmt.init, (Variable, Field, Index))
            ):
                slot = self._bind_local(stmt, ir_fn)
                ir = []
                # A slice variable is never heap-allocated -- a
                # slice's own descriptor is always a small, fixed-size,
                # stack-resident value.
                if var_type.kind != TypeKind.SLICE and self._is_heap_allocated(id(stmt), var_type):
                    # Unlike Assign/IndexAssign/FieldAssign, this
                    # destination is BRAND NEW -- its slot holds
                    # nothing yet, heap-allocated or not. A heap-
                    # allocated one needs its own fresh backing
                    # allocation made here, BEFORE _ir_copy_assign
                    # ever tries to read an address out of this slot --
                    # otherwise it would read whatever pointer-sized
                    # garbage was already sitting there.
                    ir.extend(self._ir_malloc_and_store(var_type, slot))
                return ir + self._ir_copy_assign(Variable(name=stmt.name), stmt.init, var_type)
            # A slice-typed initializer that's a bare `none` -- _ir_
            # nil_slice's own all-zero triple. ARRAY/STRUCT can never
            # be `none` -- semantic.py already rejects that outright.
            if var_type.kind == TypeKind.SLICE and isinstance(stmt.init, NoneLiteral):
                nil_ir, ptr_value, len_value, cap_value = self._ir_nil_slice()
                self._bind_local(stmt, ir_fn)
                return nil_ir + self._ir_write_slice_descriptor(
                    Variable(name=stmt.name), ptr_value, len_value, cap_value)
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
                    self._bind_local(stmt, ir_fn)
                    return slice_ir + self._ir_write_slice_descriptor(
                        Variable(name=stmt.name), ptr_value, len_value, cap_value)
            # A slice-typed initializer that's a bare bracketed-list
            # literal, resolved to SLICE by this VarDecl's own
            # declared type (`[]int s = [1, 2, 3]`) -- disambiguated
            # from the already-real-IR ARRAY-typed VarDecl case (same
            # ArrayLiteral AST shape) by var_type.kind. Same "try
            # first, bind only on success" discipline as Slice
            # production/append above.
            if var_type.kind == TypeKind.SLICE and isinstance(stmt.init, ArrayLiteral):
                production = self._ir_slice_literal(stmt.init)
                if production is not None:
                    slice_ir, ptr_value, len_value, cap_value = production
                    self._bind_local(stmt, ir_fn)
                    return slice_ir + self._ir_write_slice_descriptor(
                        Variable(name=stmt.name), ptr_value, len_value, cap_value)
            # A slice-typed initializer that's a call to the append
            # builtin (`append(s, value)`) -- same "try first, bind
            # only on success" discipline. ANY element type is in
            # scope, including array/slice/struct. What still falls
            # back: s's own base out of scope, or value itself out of
            # scope for _ir_write_append_value_at when the element
            # type is composite.
            if var_type.kind == TypeKind.SLICE and isinstance(stmt.init, Call) and stmt.init.name == 'append':
                production = self._ir_append_call(stmt.init)
                if production is not None:
                    append_ir, ptr_value, len_value, cap_value = production
                    self._bind_local(stmt, ir_fn)
                    return append_ir + self._ir_write_slice_descriptor(
                        Variable(name=stmt.name), ptr_value, len_value, cap_value)
            # An array/struct/slice-typed initializer that's an
            # ordinary function call (not append, not a struct-literal
            # Call) -- writes directly through this variable's own
            # address via the hidden-pointer convention, with no
            # intermediate copy. Needs the same heap-allocation-first
            # step the Variable/Field/Index case above needs: this
            # destination is BRAND NEW.
            if (
                    var_type.kind in COMPOSITE_KINDS
                    and (isinstance(stmt.init, Call)
                         and stmt.init.name != 'append'
                         and stmt.init.name not in self.ir_program.struct_registry)):
                slot = self._bind_local(stmt, ir_fn)
                ir = []
                if var_type.kind != TypeKind.SLICE and self._is_heap_allocated(id(stmt), var_type):
                    ir.extend(self._ir_malloc_and_store(var_type, slot))
                address_fn = {
                    TypeKind.ARRAY: self._ir_array_address,
                    TypeKind.STRUCT: self._ir_struct_address,
                    TypeKind.SLICE: self._ir_slice_address,
                }[var_type.kind]
                dst_ir, dst_address = address_fn(Variable(name=stmt.name))
                ir.extend(dst_ir)
                return ir + self._ir_composite_call(dst_address, stmt.init)
            # An array/struct-typed initializer that's an array literal
            # or a struct literal (positional or named/partial) --
            # writes directly through this variable's own address via
            # _ir_write_array_literal_into/_ir_write_struct_literal_
            # into. Needs the same heap-allocation-first step every
            # other fresh-value-producing case above needs.
            #
            # Binds FIRST, unlike Slice production/append above:
            # producing this value needs the destination's own address
            # as an INPUT, so there's no way to know success without
            # it. Safe regardless of outcome: _bind_local's own Temp
            # always points at this VarDecl's own pre-computed,
            # permanent offset, so a second _bind_local call from a
            # later case resolves to the identical storage location.
            if (
                    var_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT)
                    and (isinstance(stmt.init, ArrayLiteral)
                         or (isinstance(stmt.init, Call) and stmt.init.name in self.ir_program.struct_registry))):
                slot = self._bind_local(stmt, ir_fn)
                ir = []
                if self._is_heap_allocated(id(stmt), var_type):
                    ir.extend(self._ir_malloc_and_store(var_type, slot))
                address_fn = self._ir_array_address if var_type.kind == TypeKind.ARRAY else self._ir_struct_address
                dst_ir, dst_address = address_fn(Variable(name=stmt.name))
                ir.extend(dst_ir)
                if isinstance(stmt.init, ArrayLiteral):
                    writer = self._ir_write_array_literal_into
                else:
                    writer = self._ir_write_struct_literal_into
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
                self._bind_local(stmt, ir_fn)
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
                slot = self._bind_local(stmt, ir_fn)
                ir = []
                if self._is_heap_allocated(id(stmt), var_type):
                    ir.extend(self._ir_malloc_and_store(var_type, slot))
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
            if var_type.kind not in COMPOSITE_KINDS:
                ir, value = self.gen_expr_ir(stmt.value)
                return ir + [IRMove(dst=self._local_temp(stmt.name), src=value)]
            # A sum-typed Assign -- checked, and handled completely,
            # BEFORE the Variable/Field/Index case right below, for the
            # identical reason VarDecl's own SUM check (just above)
            # needs to come first there too: _ir_copy_assign's flat,
            # same-shape copy is wrong here regardless, since the
            # existing variable's own slot is ALREADY Shape-shaped (an
            # Assign never changes a variable's own declared type) but
            # stmt.value itself may be a narrower struct that needs
            # widening on the way in.
            if var_type.kind == TypeKind.SUM:
                dst_ir, dst_address = self._ir_struct_address(Variable(name=stmt.name))
                write_ir = self._ir_write_sum_type_value_into(dst_address, stmt.value, var_type)
                if write_ir is not None:
                    return dst_ir + write_ir
            # Same Variable/Field/Index-shaped copy as VarDecl's own
            # case just above -- no heap-allocation branch needed here
            # at all (unlike VarDecl's own): an existing slice
            # variable is never heap-allocated in the first place, and
            # an existing array/struct one already has its own real
            # allocation from declaration time, reused in place.
            if (
                    var_type.kind in COMPOSITE_KINDS
                    and isinstance(stmt.value, (Variable, Field, Index))
            ):
                return self._ir_copy_assign(Variable(name=stmt.name), stmt.value, var_type)
            # Same none-value case as VarDecl's own, just above -- no
            # binding concern here at all, unlike VarDecl's own.
            if var_type.kind == TypeKind.SLICE and isinstance(stmt.value, NoneLiteral):
                nil_ir, ptr_value, len_value, cap_value = self._ir_nil_slice()
                return nil_ir + self._ir_write_slice_descriptor(
                    Variable(name=stmt.name), ptr_value, len_value, cap_value)
            # Same Slice-production case as VarDecl's own, just above
            # -- no binding concern here at all, unlike VarDecl's own
            # (the destination already exists).
            if var_type.kind == TypeKind.SLICE and isinstance(stmt.value, Slice):
                production = self._ir_slice_into(stmt.value)
                if production is not None:
                    slice_ir, ptr_value, len_value, cap_value = production
                    return slice_ir + self._ir_write_slice_descriptor(
                        Variable(name=stmt.name), ptr_value, len_value, cap_value)
            # Same slice-literal case as VarDecl's own, just above --
            # no binding concern here at all, unlike VarDecl's own.
            if var_type.kind == TypeKind.SLICE and isinstance(stmt.value, ArrayLiteral):
                production = self._ir_slice_literal(stmt.value)
                if production is not None:
                    slice_ir, ptr_value, len_value, cap_value = production
                    return slice_ir + self._ir_write_slice_descriptor(
                        Variable(name=stmt.name), ptr_value, len_value, cap_value)
            # Same append-call case as VarDecl's own, just above -- no
            # binding concern here at all, unlike VarDecl's own.
            if var_type.kind == TypeKind.SLICE and isinstance(stmt.value, Call) and stmt.value.name == 'append':
                production = self._ir_append_call(stmt.value)
                if production is not None:
                    append_ir, ptr_value, len_value, cap_value = production
                    return append_ir + self._ir_write_slice_descriptor(
                        Variable(name=stmt.name), ptr_value, len_value, cap_value)
            # Same ordinary-function-call case as VarDecl's own --
            # append already handled, more specifically, just above.
            if (
                    var_type.kind in COMPOSITE_KINDS
                    and (isinstance(stmt.value, Call)
                         and stmt.value.name != 'append'
                         and stmt.value.name not in self.ir_program.struct_registry)
            ):
                dst_ir, dst_address = {
                    TypeKind.ARRAY: self._ir_array_address,
                    TypeKind.STRUCT: self._ir_struct_address,
                    TypeKind.SLICE: self._ir_slice_address,
                }[var_type.kind](Variable(name=stmt.name))
                return dst_ir + self._ir_composite_call(dst_address, stmt.value)
            # Same array-literal/struct-literal case as VarDecl's own,
            # just above -- no binding concern here at all, unlike
            # VarDecl's own (the destination already exists).
            if (
                    var_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT)
                    and (isinstance(stmt.value, ArrayLiteral)
                         or (isinstance(stmt.value, Call) and stmt.value.name in self.ir_program.struct_registry))
            ):
                address_fn = self._ir_array_address if var_type.kind == TypeKind.ARRAY else self._ir_struct_address
                dst_ir, dst_address = address_fn(Variable(name=stmt.name))
                if isinstance(stmt.value, ArrayLiteral):
                    writer = self._ir_write_array_literal_into
                else:
                    writer = self._ir_write_struct_literal_into
                write_ir = writer(dst_address, stmt.value, var_type)
                if write_ir is not None:
                    return dst_ir + write_ir
        elif isinstance(stmt, IndexAssign):
            # ARRAY never occurs here at all -- IndexAssign's own
            # grammar can't produce an array-typed element (unlike
            # FieldAssign, whose own case below explains the
            # contrast).
            element_type = type_of(stmt.array).element_type
            # A sum-typed element -- checked, and handled completely,
            # BEFORE the scalar check right below, for the identical
            # reason VarDecl/Assign/Return's own SUM checks need to
            # come first there too: element_type.kind not in (SLICE,
            # STRUCT) is true for SUM as well (SUM is neither), so
            # without this, a struct value being widened into a Shape-
            # typed ELEMENT would fall into _ir_index_assign's own
            # scalar path instead -- which just calls gen_expr_ir on a
            # struct-literal Call, something that path was never built
            # to handle at all (not even the "wrong width" class of
            # bug the other three fixes were -- this one doesn't
            # produce any real IR whatsoever, straight to IRError).
            if element_type.kind == TypeKind.SUM:
                result = self._ir_index_address(Index(array=stmt.array, index=stmt.index))
                if result is None:
                    raise IRError(
                        f"_ir_index_address returned None for a sum-typed "
                        f"IndexAssign ({stmt!r}) -- expected to always succeed "
                        f"for a reachable base")
                dst_ir, dst_address = result
                write_ir = self._ir_write_sum_type_value_into(dst_address, stmt.value, element_type)
                if write_ir is not None:
                    return dst_ir + write_ir
            if element_type.kind not in (TypeKind.SLICE, TypeKind.STRUCT):
                return self._ir_index_assign(stmt, element_type)
            if (
                    element_type.kind in (TypeKind.STRUCT, TypeKind.SLICE)
                    and isinstance(stmt.value, (Variable, Field, Index))
            ):
                dst_expr = Index(array=stmt.array, index=stmt.index)
                return self._ir_copy_assign(dst_expr, stmt.value, element_type)
            if element_type.kind == TypeKind.SLICE and isinstance(stmt.value, NoneLiteral):
                nil_ir, ptr_value, len_value, cap_value = self._ir_nil_slice()
                dst_expr = Index(array=stmt.array, index=stmt.index)
                return nil_ir + self._ir_write_slice_descriptor(dst_expr, ptr_value, len_value, cap_value)
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
            if (
                    element_type.kind in (TypeKind.STRUCT, TypeKind.SLICE)
                    and (isinstance(stmt.value, Call)
                         and stmt.value.name != 'append'
                         and stmt.value.name not in self.ir_program.struct_registry)
            ):
                dst_expr = Index(array=stmt.array, index=stmt.index)
                address_fn = self._ir_struct_address if element_type.kind == TypeKind.STRUCT else self._ir_slice_address
                dst_ir, dst_address = address_fn(dst_expr)
                return dst_ir + self._ir_composite_call(dst_address, stmt.value)
            # A struct-literal (positional or named/partial) value --
            # no ArrayLiteral case needed here at all, unlike VarDecl/
            # Assign/FieldAssign's own: ARRAY never occurs as an
            # IndexAssign element type in the first place (see this
            # case's own opening comment).
            if (
                    element_type.kind == TypeKind.STRUCT
                    and isinstance(stmt.value, Call)
                    and stmt.value.name in self.ir_program.struct_registry
            ):
                dst_expr = Index(array=stmt.array, index=stmt.index)
                result = self._ir_struct_address(dst_expr)
                if result is None:
                    raise IRError(
                        f"_ir_struct_address returned None for an IndexAssign's own "
                        f"STRUCT-typed destination ({dst_expr!r}) -- expected to "
                        f"always succeed for a reachable base")
                dst_ir, dst_address = result
                write_ir = self._ir_write_struct_literal_into(dst_address, stmt.value, element_type)
                if write_ir is not None:
                    return dst_ir + write_ir
        elif isinstance(stmt, FieldAssign):
            # Same idea one level over -- FieldAssign's grammar can
            # ALSO produce an ARRAY-typed field (unlike IndexAssign).
            field_type = self._check_struct_and_field_type(stmt.base, stmt.name)
            if field_type.kind not in COMPOSITE_KINDS:
                return self._ir_field_assign(stmt, field_type)
            if (
                    field_type.kind in COMPOSITE_KINDS
                    and isinstance(stmt.value, (Variable, Field, Index))
            ):
                dst_expr = Field(base=stmt.base, name=stmt.name)
                return self._ir_copy_assign(dst_expr, stmt.value, field_type)
            # Same none-value case as IndexAssign's own, one level over.
            if field_type.kind == TypeKind.SLICE and isinstance(stmt.value, NoneLiteral):
                nil_ir, ptr_value, len_value, cap_value = self._ir_nil_slice()
                dst_expr = Field(base=stmt.base, name=stmt.name)
                return nil_ir + self._ir_write_slice_descriptor(dst_expr, ptr_value, len_value, cap_value)
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
            if (
                    field_type.kind in COMPOSITE_KINDS
                    and (isinstance(stmt.value, Call)
                         and stmt.value.name != 'append'
                         and stmt.value.name not in self.ir_program.struct_registry)
            ):
                dst_expr = Field(base=stmt.base, name=stmt.name)
                address_fn = {
                    TypeKind.ARRAY: self._ir_array_address,
                    TypeKind.STRUCT: self._ir_struct_address,
                    TypeKind.SLICE: self._ir_slice_address,
                }[field_type.kind]
                dst_ir, dst_address = address_fn(dst_expr)
                return dst_ir + self._ir_composite_call(dst_address, stmt.value)
            # Same array-literal/struct-literal case as VarDecl/
            # Assign's own -- FieldAssign's grammar can ALSO produce
            # an array-typed field (unlike IndexAssign, per this
            # case's own opening comment), so both apply here, just
            # like VarDecl/Assign's own.
            if (
                    field_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT)
                    and (isinstance(stmt.value, ArrayLiteral)
                         or (isinstance(stmt.value, Call)
                         and stmt.value.name in self.ir_program.struct_registry))
            ):
                dst_expr = Field(base=stmt.base, name=stmt.name)
                address_fn = self._ir_array_address if field_type.kind == TypeKind.ARRAY else self._ir_struct_address
                dst_ir, dst_address = address_fn(dst_expr)
                if isinstance(stmt.value, ArrayLiteral):
                    writer = self._ir_write_array_literal_into
                else:
                    writer = self._ir_write_struct_literal_into
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
            # A bare `none` has no side effect of any kind -- a pure
            # literal, nothing to compute or discard -- so the correct
            # treatment is zero instructions, the same as a bare
            # Constant/BoolLiteral ExprStmt.
            return []
        elif isinstance(stmt, ExprStmt) and isinstance(stmt.expr, Call) and stmt.expr.name == 'append':
            # append(s, value) used directly as a bare statement, its
            # own resulting {ptr, len, cap} triple entirely discarded
            # -- only the growth/write side effect matters. Checked
            # before the ordinary composite-Call case below: append is
            # a builtin, never a compiled function, so falling through
            # there would try to call a symbol literally named
            # 'append' that doesn't exist.
            production = self._ir_append_call(stmt.expr)
            if production is not None:
                append_ir, _, _, _ = production
                return append_ir
        elif (
                isinstance(stmt, ExprStmt)
                and type_of(stmt.expr).kind in COMPOSITE_KINDS
                and self._is_ordinary_composite_call(stmt.expr)
        ):
            # An array/struct/slice-returning ordinary Call used
            # directly as a bare statement (`makeArray()` alone), its
            # own result entirely discarded. The call's own side
            # effect still needs to happen, so this materializes it
            # via _ir_materialize_composite_call (never out of scope,
            # unlike _ir_append_call above) and discards the resulting
            # address.
            ir, _ = self._ir_materialize_composite_call(stmt.expr, type_of(stmt.expr))
            return ir
        elif isinstance(stmt, ExprStmt) and isinstance(stmt.expr, Call) and stmt.expr.name in self.ir_program.struct_registry:
            # A struct literal used directly as a bare statement
            # (`Point(1, 2)` alone). _ir_materialize_composite_call
            # doesn't apply here -- it writes through _ir_composite_
            # call, which calls call_expr.name as an ordinary function,
            # not what a struct name's own construction needs -- so
            # this goes through _ir_materialize_struct_literal instead.
            result = self._ir_materialize_struct_literal(stmt.expr)
            if result is None:
                raise IRError(
                    f"_ir_materialize_struct_literal returned None for a struct "
                    f"literal used as a bare statement ({stmt.expr!r}) -- some "
                    f"field is out of scope for real IR"
                )
            ir, _ = result
            return ir
        elif isinstance(stmt, ExprStmt):
            ir, _ = self.gen_expr_ir(stmt.expr)
            return ir
        raise IRError(
            f"No real-IR case for statement of type {type(stmt).__name__}: {stmt!r}"
        )

    def _ir_malloc_and_store(self, var_type, slot: int) -> list:
        """Builds (without lowering) a heap-allocated local's own
        fresh backing allocation as real IR: an ordinary IRCall to
        malloc, then an ordinary IRStore writing the returned pointer
        into this variable's own frame slot, via IRLocalAddress for
        the slot's own address.

        Used wherever a fresh, heap-allocated array/struct local needs
        its own backing allocation made before its address is ever
        computed -- see this method's own call sites in gen_statement_
        ir's own VarDecl case, one per fresh-value-producing shape."""
        size = type_byte_width(var_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        ptr = self.ir_program.ids.new_temp(Type.INT64)
        slot_addr = self.ir_program.ids.new_temp(Type.INT64)
        return [
            IRCall(dst=ptr, name='malloc', args=[IRConst(size, Type.INT64)]),
            IRLocalAddress(dst=slot_addr, slot=slot),
            IRStore(address=slot_addr, value=ptr, value_type=Type.INT64),
        ]

    def _ir_index_assign(self, stmt: IndexAssign, element_type) -> list:
        """Builds (without lowering) the scalar-element case of an
        IndexAssign -- the caller (gen_statement_ir) is responsible for
        already having ruled out SLICE/STRUCT. Captures the address via
        _ir_index_address, builds the value's own IR, then IRStores it
        through the address, at the ELEMENT's own declared width -- not
        necessarily the value's own, per IRStore's own docstring."""
        result = self._ir_index_address(Index(array=stmt.array, index=stmt.index))
        if result is None:
            raise IRError(
                f"_ir_index_address returned None for a scalar-element IndexAssign "
                f"({stmt!r}) -- expected to always succeed for a reachable base")
        addr_ir, addr_value = result
        value_ir, value = self.gen_expr_ir(stmt.value)
        return addr_ir + value_ir + [IRStore(address=addr_value, value=value, value_type=element_type)]

    def _ir_copy_into_address(self, dst_address, src_expr: Node, value_type) -> list:
        """The shared core of _ir_copy_assign: given an already-
        computed destination address (however the caller has it --
        a freshly-computed address for VarDecl/Assign/IndexAssign/
        FieldAssign, or the current function's own received hidden
        pointer for a forwarding `return someVar`), captures src_
        expr's own address, then IRCopies between them.

        Callers are responsible for already having confirmed src_expr
        is a Variable/Field/Index -- see _ir_copy_assign's own
        docstring for why an ArrayLiteral/struct-literal Call/Slice/
        composite-returning Call is a different case."""
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
        so a heap-allocated variable or a nested field/index
        destination is handled identically -- then delegates to _ir_
        copy_into_address for the rest.

        Callers are responsible for already having confirmed src_expr
        is a Variable/Field/Index -- an ArrayLiteral, a struct-literal
        Call, a Slice (slice production), or an ordinary composite-
        returning Call is a genuinely different case (construction, or
        the hidden-output-pointer convention) with no existing address
        to capture at all; see gen_statement_ir's own VarDecl/Assign/
        IndexAssign/FieldAssign cases for where that check happens."""
        address_of = {
            TypeKind.ARRAY: self._ir_array_address,
            TypeKind.STRUCT: self._ir_struct_address,
            TypeKind.SLICE: self._ir_slice_address,
        }[value_type.kind]
        dst_ir, dst_addr = address_of(dst_expr)
        return dst_ir + self._ir_copy_into_address(dst_addr, src_expr, value_type)

    def _ir_field_assign(self, stmt: FieldAssign, field_type) -> list:
        """Builds (without lowering) the scalar-field case of a
        FieldAssign -- the caller (gen_statement_ir) is responsible
        for already having ruled out SLICE/STRUCT/ARRAY. Same shape as
        _ir_index_assign one level over: captures the address via
        _ir_field_address, builds the value's own IR, then IRStores it
        through the address at the FIELD's own declared width."""
        result = self._ir_field_address(Field(base=stmt.base, name=stmt.name))
        if result is None:
            raise IRError(
                f"_ir_field_address returned None for a scalar-typed FieldAssign "
                f"({stmt!r}) -- expected to always succeed for a reachable base")
        addr_ir, addr_value = result
        value_ir, value = self.gen_expr_ir(stmt.value)
        return addr_ir + value_ir + [IRStore(address=addr_value, value=value, value_type=field_type)]

    def _ir_return(self, value_expr) -> list:
        """Builds (without lowering) IRReturn for a bare return
        (value_expr=None) or a scalar return -- the two cases gen_
        statement_ir's own Return case doesn't route through the
        hidden-pointer convention."""
        if value_expr is None:
            return [IRReturn(value=None)]
        ir, value = self.gen_expr_ir(value_expr)
        return ir + [IRReturn(value=value)]

    def _ir_hidden_return_ptr(self, ir_fn: IRFunction) -> tuple:
        """Reads the current function's own received hidden return
        pointer (for a composite-returning function, passed by ITS OWN
        caller, at this function's own fixed, known hidden_return_ptr_
        slot) -- returns (ir, address). IRLocalAddress for the slot's
        own address, then an ordinary IRLoad reading the pointer stored
        there. Shared by every gen_statement_ir Return case that
        forwards into it."""
        slot_addr = self.ir_program.ids.new_temp(Type.INT64)
        hidden_ptr = self.ir_program.ids.new_temp(Type.INT64)
        ir = [
            IRLocalAddress(dst=slot_addr, slot=ir_fn.hidden_return_ptr_slot),
            IRLoad(dst=hidden_ptr, address=slot_addr),
        ]
        return ir, hidden_ptr

    def _ir_if_head(self, stmt: If, then_label: str, else_label: str) -> list:
        """Builds (without lowering) the condition-and-branch IR
        landing at the given then/else labels -- the caller (gen_
        statement_ir's own If case) is responsible for the bodies and
        the trailing jump/labels around them."""
        cond_ir, cond_value = self.gen_expr_ir(stmt.condition)
        return cond_ir + [
            IRBranch(cond=cond_value, true_label=then_label, false_label=else_label),
            IRLabel(then_label),
        ]

    def _ir_while_head(self, stmt: While, start_label: str, body_label: str, end_label: str) -> list:
        """Builds (without lowering) the start-label/condition/branch
        IR landing at the given body label -- the caller (gen_
        statement_ir's own While case) is responsible for the body and
        the trailing jump/end-label around it.

        Opens with an explicit IRJump(start_label), not just IRLabel
        (start_label) directly: whatever precedes a While has no
        reason to already end in a terminator, so without this jump,
        entering the loop's own condition check the first time would
        rely on falling out of the preceding op into this label --
        the implicit fallthrough ir.ir's own module docstring says
        never to rely on. Redundant once lowered (a jump straight to
        the next instruction) -- exactly what TODO.md's own "IRBranch
        peephole optimization" item would clean up."""
        cond_ir, cond_value = self.gen_expr_ir(stmt.condition)
        return [IRJump(start_label), IRLabel(start_label)] + cond_ir + [
            IRBranch(cond=cond_value, true_label=body_label, false_label=end_label),
            IRLabel(body_label),
        ]

    def _ir_break(self) -> list:
        """Builds real IR for a bare `break`: the innermost loop's own
        end label (see loop_labels), raising IRError if none is
        currently open -- an ordinary IRJump to it."""
        if not self.loop_labels:
            raise IRError("'break' outside of a loop")
        _, end_label = self.loop_labels[-1]
        return [IRJump(end_label)]

    def _ir_continue(self) -> list:
        """Builds real IR for a bare `continue`: the innermost loop's
        own start label (see loop_labels), raising IRError if
        none is currently open -- an ordinary IRJump to it."""
        if not self.loop_labels:
            raise IRError("'continue' outside of a loop")
        start_label, _ = self.loop_labels[-1]
        return [IRJump(start_label)]



