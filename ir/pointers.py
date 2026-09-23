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
from ir.ir import IRCopy, IRLoad, IRLocalAddress, IRStore, IRValue
from ir.utils import COMPOSITE_KINDS, is_composite_addressable, type_of
from parser import Call, DerefAssign, Field, Index, Unary, Variable
from semantic import TypeKind


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
        address), not the slot's own address itself. A scalar x is
        never heap-allocated at all right now -- is_heap_allocated's
        own size check is false for every scalar type, and a scalar
        whose OWN address escapes is rejected outright by semantic.py
        instead of ever reaching real-IR generation (see check_unary's
        own ADDRESS_OF case) -- so this branch is unreachable for a
        scalar x today, but written generally rather than assuming
        that stays true forever.

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

        expr.operand is guaranteed a bare Variable OR a struct literal
        by semantic.py's own check_unary (see UnaryOp.ADDRESS_OF's own
        docstring there for why this slice of pointer support restricts
        it that way -- struct fields, array/slice elements, and array
        literals are planned, not yet reachable here).

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
        still needs its own extra heap-allocated indirection check."""
        if isinstance(expr.operand, Call):
            result = self._ir_materialize_struct_literal(expr.operand)
            if result is None:
                raise IRError(
                    f"_ir_materialize_struct_literal returned None for a "
                    f"struct-literal '&' operand ({expr.operand!r}) -- "
                    f"expected to always succeed for a reachable shape"
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
        slot_addr = self.ir_program.ids.new_temp(type_of(expr))
        ir = [IRLocalAddress(dst=slot_addr, slot=slot)]
        if self._is_heap_allocated(self._local_decl_id(name), self._local_type(name)):
            addr_temp = self.ir_program.ids.new_temp(type_of(expr))
            ir.append(IRLoad(dst=addr_temp, address=slot_addr))
            return ir, addr_temp
        return ir, slot_addr

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

        Scoped to a scalar pointee (an ordinary IRStore) or a struct
        pointee, via either an existing Variable/Field/Index source
        (_ir_struct_address to capture its own address, then IRCopy)
        or a struct literal (_ir_write_struct_literal_into) -- the two
        shapes already exercised at the semantic level (analyze_deref_
        assign's own tests). An array- or slice-typed pointee is
        deferred, matching this slice of pointer support's other
        narrow-scope choices (see UnaryOp.ADDRESS_OF's/DEREFERENCE's
        own comments): slice specifically has several of its own
        value-shape cases (nil, a Slice production, append) that would
        need their own dedicated handling, mirroring why gen_statement_
        ir's own FieldAssign case is so much larger than IndexAssign's
        -- not something to fold in as an afterthought here."""
        pointer_type = type_of(stmt.pointer)
        pointee_type = pointer_type.element_type
        ptr_ir, ptr_value = self.gen_expr_ir(stmt.pointer)

        if pointee_type.kind not in COMPOSITE_KINDS:
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

        raise IRError(
            f"No real-IR case for DerefAssign with pointee kind "
            f"{pointee_type.kind} and value {stmt.value!r}"
        )
