"""Pointer support's own IR construction: `&x` (address-of) and `*p`
(dereference), both reached from gen_expr_ir's own dispatch (ir/
dispatch.py) as special cases of Unary, ahead of its generic IRUnOp
case -- neither shape fits that generic case at all (ADDRESS_OF never
even evaluates its operand as a VALUE; DEREFERENCE needs an IRLoad,
not an IRUnOp).

Auto-deref for field access and method-call receivers lives in ir/
structs.py instead (_ir_struct_address's own Variable case), not here
-- that's about STRUCT addressing recognizing a pointer-to-struct base,
not about `&`/`*` themselves."""

from ir.errors import IRError
from ir.ir import IRBinOp, IRConst, IRCopy, IRLoad, IRLocalAddress, IRStore, IRValue
from ir.utils import COMPOSITE_KINDS, SUM_TYPE_TAG_WIDTH, is_composite_addressable, type_of
from parser import BinaryOp, Call, DerefAssign, Field, Index, Unary, Variable
from semantic import Type, TypeKind


class PointersMixin:
    def _ir_address_of(self, expr: Unary) -> tuple[list, IRValue]:
        """`&x` -- computes x's own address, via the identical
        IRLocalAddress instruction _ir_struct_address's own Variable
        case already uses for a COMPOSITE local's address, now also
        applied to a SCALAR's slot (every variable gets one at _bind_
        local time regardless of its own type -- see _local_slot's own
        docstring).

        A HEAP-ALLOCATED x (its own composite size over the threshold,
        or its OWN address escaping past this function via a pointer
        -- see _is_heap_allocated/analyze_array_escapes) needs the
        identical extra indirection _ir_struct_address's own Variable
        case already applies: x's own slot holds a POINTER to x's real
        storage in that case, not x's data directly, so &x is that
        stored pointer (one more IRLoad through the slot's own
        address), not the slot's own address itself. An ORDINARY
        (non-narrowed) scalar x is never heap-allocated at all right
        now -- is_heap_allocated's own size check is false for every
        scalar type, and a scalar whose OWN address escapes is
        rejected outright by semantic.py instead of ever reaching
        real-IR generation (see check_unary's own ADDRESS_OF case).
        But a narrowed-to-scalar/pointer binding (`if m is int as n:
        &n`) is a real exception: n's own slot is m's, a SUM-typed
        one, which very much can be heap-allocated like any other
        composite -- so this branch, unreachable for an ordinary
        scalar, is genuinely reachable and needed for a narrowed one.

        This is exactly what register_allocator.py's own eligible_
        intervals (see its own docstring) keys off of to stay sound
        for the NON-heap-allocated case: any slot an IRLocalAddress
        instruction anywhere in this function ever targets is excluded
        from register allocation entirely, so x's value always lives
        in memory, never purely in a register -- otherwise a read or
        write through this pointer could see a stale value the
        register alone was holding. A heap-allocated x's own slot
        holds just a pointer, read once here and then never written
        again by this function specifically for x's OWN address, so
        it doesn't need this same protection the same way -- though
        it's excluded anyway, harmlessly, since IRLocalAddress still
        targets it.

        expr.operand is guaranteed a bare Variable, a struct literal,
        a Field, or an Index by semantic.py's own check_unary (see
        UnaryOp.ADDRESS_OF's own docstring there -- a Field/Index chain
        rooted in anything other than a named variable, e.g. a function
        call's own result, is excluded there too, not reachable here).

        A struct-literal operand (`&Circle(5)`) is handled first,
        entirely separately from the Variable case below: there's no
        existing slot/decl_id to look anything up by -- see ir/
        builder.py's own _collect_argument_temps_in_expr for where one
        gets reserved (or not, for an escaping literal) ahead of time,
        keyed by id(expr.operand) exactly like a named declaration's
        own slot is keyed by id(its VarDecl/Param node). _ir_
        materialize_struct_literal already knows how to build the
        right IR either way (a reserved stack slot, or a fresh malloc
        when none was reserved) -- its own returned address IS this
        expression's own value, unlike the Variable case below, which
        still needs its own extra heap-allocated indirection check.

        A Field or Index operand (`&s.field`, `&arr[i]`) is handled
        next, delegating straight to _ir_field_address/_ir_index_
        address -- both already compute exactly the address needed
        here for any other purpose (a scalar field/element's own
        write, or a composite one's own copy source), already handle
        arbitrary chain depth and auto-deref through a pointer-typed
        base internally, and already resolve to a real address with
        no construction step involved (unlike the struct-literal case
        above) -- so their own returned address IS this expression's
        own value too, no extra heap-allocated indirection check
        needed here either: whichever named variable this chain is
        ultimately rooted in is what escape analysis's own new ADDRESS_
        OF case (contribution(), codegen/escape_analysis.py) already
        attributes THIS address to, so if it escapes, is_heap_
        allocated already reports that ROOT declaration as heap-
        allocated -- and _ir_field_address/_ir_index_address already
        read a heap-allocated base correctly on their own, the same
        way they already do for every other caller."""
        if isinstance(expr.operand, Call):
            result = self._ir_materialize_struct_literal(expr.operand)
            if result is None:
                raise IRError(
                    f"_ir_materialize_struct_literal returned None for a "
                    f"struct-literal '&' operand ({expr.operand!r}) -- "
                    f"expected to always succeed for a reachable shape"
                )
            return result
        if isinstance(expr.operand, Field):
            result = self._ir_field_address(expr.operand)
            if result is None:
                raise IRError(
                    f"_ir_field_address returned None for a Field '&' "
                    f"operand ({expr.operand!r}) -- expected to always "
                    f"succeed for a reachable shape"
                )
            return result
        if isinstance(expr.operand, Index):
            result = self._ir_index_address(expr.operand)
            if result is None:
                raise IRError(
                    f"_ir_index_address returned None for an Index '&' "
                    f"operand ({expr.operand!r}) -- expected to always "
                    f"succeed for a reachable shape"
                )
            return result
        if not isinstance(expr.operand, Variable):
            raise IRError(
                f"IRError: '&' operand is {type(expr.operand).__name__}, not a "
                f"Variable -- semantic.py's own check_unary should have already "
                f"rejected this before real-IR generation ever runs"
            )
        name = expr.operand.name
        slot = self._local_slot(name)
        slot_type = self._local_type(name)
        slot_addr = self.ir_program.ids.new_temp(type_of(expr))
        ir = [IRLocalAddress(dst=slot_addr, slot=slot)]
        if self._is_heap_allocated(self._local_decl_id(name), slot_type):
            addr_temp = self.ir_program.ids.new_temp(type_of(expr))
            ir.append(IRLoad(dst=addr_temp, address=slot_addr))
            base_addr = addr_temp
        else:
            base_addr = slot_addr
        narrowed_type = expr.operand.resolved_type
        if slot_type.kind == TypeKind.SUM and narrowed_type is not None and narrowed_type != slot_type:
            # &n means n's own value's address, not the whole sum-
            # typed slot's -- see this method's own docstring.
            payload_addr = self.ir_program.ids.new_temp(type_of(expr))
            ir.append(IRBinOp(
                dst=payload_addr, op=BinaryOp.ADD,
                left=base_addr, right=IRConst(SUM_TYPE_TAG_WIDTH, Type.INT64),
            ))
            return ir, payload_addr
        return ir, base_addr

    def _ir_dereference(self, expr: Unary) -> tuple[list, IRValue]:
        """`*p` -- reads the pointee's own value: gen_expr_ir(expr.
        operand) gives p's OWN value (an ordinary scalar-shaped Temp,
        the address itself -- pointers are 8-byte scalars for every
        purpose other than what they point AT), then IRLoad reads
        type_of(expr) (the pointee's own type) worth of data through
        it.

        Dereferencing a pointer to a composite (struct/array/slice/sum)
        type never reaches this method at all: semantic.py's own
        check_unary already rejects it outright, for this first slice
        of pointer support (see its own comment on UnaryOp.DEREFERENCE
        for why -- a scope decision, not a soundness one). So type_of
        (expr) is always a scalar here, and this is always safe to
        IRLoad as one."""
        operand_ir, operand_value = self.gen_expr_ir(expr.operand)
        return self._ir_load(operand_ir, operand_value, type_of(expr))

    def _ir_deref_assign(self, stmt: DerefAssign) -> list:
        """`*pointer = value` -- writes through a pointer, overwriting
        whatever it points at. The TARGET address is always just
        gen_expr_ir(stmt.pointer) -- the pointer's own value, an
        ordinary scalar read -- never _ir_struct_address's own
        machinery: unlike FieldAssign/IndexAssign, there's no "base
        expression" to resolve here, just the pointer itself.

        Scoped to a scalar pointee (an ordinary IRStore), a struct
        pointee, via either an existing Variable/Field/Index source
        (_ir_struct_address to capture its own address, then IRCopy)
        or a struct literal (_ir_write_struct_literal_into) -- the two
        shapes already exercised at the semantic level (analyze_deref_
        assign's own tests) -- or a str pointee, via _ir_str_value
        (uniform over every str-typed shape: a StringLiteral, a
        concatenation, an existing value, or an ordinary str-returning
        Call) writing directly through ptr_value, which already IS the
        destination descriptor's own address. An array- or slice-typed
        pointee is deferred, matching this slice of pointer support's
        other narrow-scope choices (see UnaryOp.ADDRESS_OF's/
        DEREFERENCE's own comments): slice specifically has several of
        its own value-shape cases (nil, a Slice production, append)
        that would need their own dedicated handling, mirroring why
        gen_statement_ir's own FieldAssign case is so much larger than
        IndexAssign's -- not something to fold in as an afterthought
        here.

        stmt.compound_op (`*p += 1`) mirrors ir/statements.py's own
        IndexAssign/FieldAssign handling exactly -- see _ir_compound_
        assign_through_address's own docstring. Only ever reachable
        for a scalar pointee (semantic.py's own check_binary, which
        _check_compound_assign already routes every compound_op
        through, would already have rejected a struct/array/slice
        one), so it's checked here, in the scalar branch, not
        threaded into the struct branch below at all."""
        pointer_type = type_of(stmt.pointer)
        pointee_type = pointer_type.element_type
        ptr_ir, ptr_value = self.gen_expr_ir(stmt.pointer)

        if pointee_type.kind not in COMPOSITE_KINDS:
            if stmt.compound_op is not None:
                return ptr_ir + self._ir_compound_assign_through_address(ptr_value, stmt.compound_op, stmt.value, pointee_type)
            value_ir, value = self.gen_expr_ir(stmt.value)
            return ptr_ir + value_ir + [IRStore(address=ptr_value, value=value, value_type=pointee_type)]

        if pointee_type.kind == TypeKind.STRUCT and is_composite_addressable(stmt.value):
            src_ir, src_address = self._ir_struct_address(stmt.value)
            return ptr_ir + src_ir + [IRCopy(dst_address=ptr_value, src_address=src_address, value_type=pointee_type)]

        if (
                pointee_type.kind == TypeKind.STRUCT
                and isinstance(stmt.value, Call)
                and stmt.value.name in self.ir_program.struct_registry
        ):
            write_ir = self._ir_write_struct_literal_into(ptr_value, stmt.value, pointee_type)
            if write_ir is not None:
                return ptr_ir + write_ir

        if pointee_type.kind == TypeKind.STR:
            # ptr_value already IS the pointee's own {ptr, len}
            # descriptor's address (the pointer's own value, read
            # above like any other) -- no separate _ir_str_address
            # call needed the way the STRUCT case above needs one for
            # its OWN source: there, an existing struct value has its
            # OWN address to capture; here, the DESTINATION already is
            # one, directly. _ir_str_value produces the {ptr, len}
            # pair for whatever shape stmt.value actually is
            # (StringLiteral, concatenation, an existing value, an
            # ordinary str-returning Call -- all uniformly, the same
            # dispatcher every other str-typed destination already
            # goes through).
            value_ir, ptr, length = self._ir_str_value(stmt.value)
            return ptr_ir + value_ir + self._ir_write_str_descriptor_into_address(ptr_value, ptr, length)

        raise IRError(
            f"No real-IR case for DerefAssign with pointee kind "
            f"{pointee_type.kind} and value {stmt.value!r}"
        )
