"""Struct layout and value semantics. Fields sit at sequential byte
offsets in declaration order with no padding (x86-64 doesn't require
aligned access); a struct's nominal typing -- two structs with
identical fields but different names are different types -- falls out
of Type's existing structural-equality machinery for free, needing no
new mechanism here. Struct assignment, parameter passing, and return
all copy the whole value, never alias, exactly like arrays one level
over; equality and zero-initialization work field by field, recursing
into nested structs and delegating array-typed fields back to
arrays_slices.py."""

from ir.errors import IRError
from ir.ir import IRBinOp, IRConst, IRStore, IRLoad, IRLocalAddress, IRCall
from ir.utils import COMPOSITE_KINDS, SUM_TYPE_TAG_WIDTH, type_byte_width, type_of
from parser import Node, Variable, Field, Index, Call, BinaryOp, Unary, UnaryOp
from semantic import TypeKind, Type


class StructsMixin:
    def _field_offset(self, struct_name: str, field_name: str) -> int:
        """Byte offset of `field_name` within struct_name: the sum of
        every preceding field's width, in declaration order. No
        padding or alignment is inserted between fields -- x86-64
        doesn't require aligned access."""
        offset = 0
        for name, field_type in self.ir_program.struct_registry[struct_name].fields.items():
            if name == field_name:
                return offset
            offset += type_byte_width(field_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        raise IRError(f"Struct '{struct_name}' has no field '{field_name}'")

    def _ir_struct_address(self, expr: Node) -> tuple[list, object]:
        """Builds (without lowering) the address of a struct-typed
        expr -- Variable, Field, Index, or an ordinary composite-
        returning Call (materialized via _ir_materialize_composite_
        call) -- as real IR. Field/Index delegate to _ir_field_
        address/_ir_index_address, which call back into this method
        for their own STRUCT-typed base, so a chain of arbitrary depth
        (`a.b.c`, `rows[0].f`) falls out with no special-casing.

        The Variable case is the one genuine leaf: a named struct
        variable's own address is either a fixed %rbp-relative offset
        (IRLocalAddress directly) or, if heap-allocated, the pointer
        stored at that offset (IRLocalAddress for the slot's own
        address, then an IRLoad reading the pointer through it).

        A NARROWED occurrence needs one more step: expr.resolved_type
        (set by semantic.py's own analyze_if, e.g. Circle for a `shape
        is Circle` branch) differing from struct_type (_local_type --
        the SLOT's own, unchanging declared type, e.g. Shape, decided
        once at _bind_local time and never re-shadowed here the way
        semantic.py's OWN scopes are) means this occurrence is
        narrowed: the struct's real bytes start SUM_TYPE_TAG_WIDTH
        into the slot's own address (the discriminant tag sits at the
        front), not at its start. A genuinely sum-typed reference --
        every OTHER caller of this same Variable case, from before
        narrowing existed: widening a struct INTO this slot, print(),
        an ordinary argument or return -- has resolved_type ==
        struct_type (both the same sum type) and takes the unchanged,
        no-offset path, exactly as it always has; deliberately NOT
        decided by struct_type.kind == SUM alone, which would
        incorrectly add the offset for every one of those other,
        already-working cases too.

        Read directly off expr.resolved_type, not through type_of --
        an internal Variable node synthesized purely for IR building
        (e.g. Variable(name=stmt.name), all over VarDecl/Assign's own
        handling in ir/statements.py) never goes through semantic
        analysis and so never has one at all; None is treated the same
        as "not narrowed" here rather than type_of's own "raise, this
        is a bug" stance, because such a node can PROVABLY never
        legitimately be a narrowed occurrence: a VarDecl re-declaring
        an actively-narrowed name is already rejected as a duplicate
        declaration in the same scope, and an Assign to one is already
        rejected outright by analyze_assign -- both well before this
        ever runs. Every node that COULD be narrowed is an actual,
        parsed source reference, which always has a real resolved_type
        from check_expr."""
        if isinstance(expr, (Field, Index)) and expr.resolved_type is not None and expr.resolved_type.kind == TypeKind.POINTER:
            # Auto-deref, one level over from the Variable case just
            # below: `b.next.value`, where `b.next` (a POINTER-typed
            # field access, not a bare name) needs ITS OWN value (the
            # address it holds), not b.next's own ADDRESS (where the
            # pointer's 8 bytes live within b's own layout) -- the
            # identical principle, just reached through a Field or
            # Index base instead of a Variable one.
            #
            # Read directly off expr.resolved_type, not through type_
            # of -- exactly the same reasoning as the Variable case's
            # own comment below, just for Field/Index instead of
            # Variable: a SYNTHESIZED Field/Index node (e.g. dst_expr
            # = Field(base=stmt.base, name=stmt.name), built fresh in
            # gen_statement_ir's own FieldAssign/IndexAssign handling
            # as a COPY DESTINATION, reusing this same address-
            # computation machinery) never goes through semantic
            # analysis and so never has a resolved_type at all. That's
            # fine here: such a node is always constructed specifically
            # because its own field_type/element_type is ALREADY known
            # to be a composite kind (ARRAY/STRUCT/SLICE -- see
            # COMPOSITE_KINDS checks at each such call site), never
            # POINTER, so treating a missing resolved_type as "not
            # pointer, don't auto-deref" is correct here, not just a
            # safe approximation.
            return self.gen_expr_ir(expr)
        if isinstance(expr, Variable):
            var_type = self._local_type(expr.name)
            if var_type.kind == TypeKind.POINTER:
                # Auto-deref: p.field (p: *Circle) needs p's OWN
                # value (the address it holds), not p's own SLOT's
                # address (which would give the address of the
                # pointer VARIABLE itself, not what it points at). An
                # ordinary scalar read, via gen_expr_ir -- exactly how
                # a pointer's own value is read everywhere else, since
                # a pointer is just an 8-byte scalar for every purpose
                # other than what it points at.
                #
                # p.method() (_check_method_call's own, separate auto-
                # deref) reaches this SAME path too, once desugar_
                # methods has already rewritten it into an ordinary
                # call with p as the first argument -- Field's own
                # Variable base here is identical in shape either way.
                return self.gen_expr_ir(expr)
            slot = self._local_slot(expr.name)
            struct_type = var_type
            slot_addr = self.ir_program.ids.new_temp(Type.INT64)
            ir = [IRLocalAddress(dst=slot_addr, slot=slot)]
            if self._is_heap_allocated(self._local_decl_id(expr.name), struct_type):
                addr_temp = self.ir_program.ids.new_temp(Type.INT64)
                ir.append(IRLoad(dst=addr_temp, address=slot_addr))
                base_addr = addr_temp
            else:
                base_addr = slot_addr
            if struct_type.kind == TypeKind.SUM:
                # narrowed_type is expr.resolved_type directly, not
                # type_of(expr) -- None is expected and MEANINGFUL
                # here, not a bug to raise on: an internal Variable
                # node synthesized purely for IR building (e.g.
                # Variable(name=stmt.name), all over VarDecl/Assign's
                # own handling in ir/statements.py, reusing this same
                # address-computation machinery for a declaration's or
                # assignment's own target name) never goes through
                # semantic analysis at all, so it never HAS a resolved_
                # type. That's fine here specifically, because such a
                # node can PROVABLY never be a narrowed occurrence: a
                # VarDecl re-declaring an actively-narrowed name is
                # already rejected as a duplicate declaration in the
                # same scope (the narrowing's own shadow already
                # occupies it), and an Assign to one is already
                # rejected outright by analyze_assign's own check --
                # both at the semantic level, well before this ever
                # runs. Every node that COULD legitimately be narrowed
                # is an actual, parsed source reference, which always
                # has a real resolved_type from check_expr.
                narrowed_type = expr.resolved_type
                if narrowed_type is None or narrowed_type == struct_type:
                    return ir, base_addr  # genuinely sum-typed (or an internal, always-whole-value node), not narrowed
                if narrowed_type.kind != TypeKind.STRUCT:
                    raise IRError(
                        f"Variable '{expr.name}' has sum type {struct_type} but its own "
                        f"resolved_type {narrowed_type} is neither that same sum type nor "
                        f"a struct -- expected only these two shapes for a sum-typed name"
                    )
                payload_addr = self.ir_program.ids.new_temp(Type.INT64)
                ir.append(IRBinOp(
                    dst=payload_addr, op=BinaryOp.ADD,
                    left=base_addr, right=IRConst(SUM_TYPE_TAG_WIDTH, Type.INT64),
                ))
                return ir, payload_addr
            return ir, base_addr
        if isinstance(expr, Field):
            return self._ir_field_address(expr)
        if isinstance(expr, Index):
            return self._ir_index_address(expr)
        if isinstance(expr, Unary) and expr.op == UnaryOp.DEREFERENCE:
            # `*p` (p: *Circle) read as a whole STRUCT value -- the
            # pointer's own value already IS the address of its own
            # pointee's bytes, exactly the same principle the auto-
            # deref case at the very top of this method already uses
            # for a POINTER-typed Field/Index base (`b.next.value`):
            # there, the base's own resolved_type being POINTER means
            # ITS value, not ITS address, is what's needed; here,
            # expr.operand IS that pointer expression directly, with
            # no base/field indirection to unwrap first. No narrowing
            # concern reaches here: semantic.py's own check_unary
            # still rejects a SUM-typed pointee outright (see its own
            # DEREFERENCE case), so struct_type here is never a sum
            # type, and expr itself -- an internal Unary, never
            # produced by _bind_local/_bind_param the way a narrowed
            # Variable occurrence's own resolved_type comparison
            # depends on -- has no slot of its own to compare against
            # in the first place.
            return self.gen_expr_ir(expr.operand)
        if self._is_ordinary_composite_call(expr):
            return self._ir_materialize_composite_call(expr, type_of(expr))
        raise IRError(f"Cannot compute a struct address for: {expr!r}")

    def _ir_field_address(self, expr: Field) -> tuple[list, object]:
        """Builds (without lowering) the address of `expr.base.expr.
        name` as real IR: the base's own address (recursively, via
        _ir_struct_address) plus expr.name's fixed byte offset, via
        IRBinOp(ADD). Skips the add when offset is 0.

        A POINTER-to-struct base auto-dereferences here too -- see
        _check_struct_and_field_type's own docstring for why type_of
        (expr.base) alone isn't enough (it reports p's own LITERAL
        declared type, *Circle, never the auto-dereferenced Circle).
        _ir_struct_address(expr.base) below already handles this
        correctly on its own (its Variable case's own auto-deref), so
        only THIS method's own base_type/struct_name resolution, used
        for the field offset lookup, needs the identical check
        repeated -- the address computation itself doesn't."""
        base_type = type_of(expr.base)
        if base_type.kind == TypeKind.POINTER and base_type.element_type.kind == TypeKind.STRUCT:
            base_type = base_type.element_type
        if base_type.kind != TypeKind.STRUCT:
            raise IRError(
                f"Cannot access field '{expr.name}' on a value of "
                f"non-struct type {base_type}"
            )
        offset = self._field_offset(base_type.struct_name, expr.name)
        result = self._ir_struct_address(expr.base)
        if result is None:
            raise IRError(
                f"_ir_struct_address returned None for a Field's own STRUCT-typed "
                f"base ({expr.base!r}) -- expected to always succeed for a "
                f"reachable base")
        base_ir, base_addr = result
        if offset == 0:
            return base_ir, base_addr
        result = self.ir_program.ids.new_temp(Type.INT64)
        add_op = IRBinOp(dst=result, op=BinaryOp.ADD, left=base_addr, right=IRConst(offset, Type.INT64))
        return base_ir + [add_op], result

    def _ir_write_struct_literal_into(self, dst_address, expr: Call, struct_type: Type):
        """Builds (without lowering) a struct literal's fields as real
        IR, written through dst_address. Returns None only when some
        field's own value is out of scope for _ir_write_composite_
        value_into.

        Named construction (expr.kwargs) is normalized to the same
        (field_name, value_expr, field_type) shape positional
        construction iterates, walking every field in declaration
        order. An omitted field's value_expr comes back None, filled
        via _ir_write_zero_value_into (a TOTAL function).

        Each field's own address is dst_address + its own _field_
        offset, skipped for the first field (offset 0). A scalar
        field's value is evaluated via gen_expr_ir and written via
        IRStore; a composite field delegates to _ir_write_composite_
        value_into. If any field's own value is out of scope, the
        whole literal falls back (returns None)."""
        struct_info = self.ir_program.struct_registry[struct_type.struct_name]
        field_items = list(struct_info.fields.items())
        if expr.kwargs is not None:
            provided = dict(expr.kwargs)
            entries = [(field_name, provided.get(field_name), field_type) for field_name, field_type in field_items]
        else:
            entries = [
                (field_name, arg_expr, field_type) for arg_expr, (field_name, field_type) in zip(expr.args, field_items)
            ]
        ir = []
        for field_name, arg_expr, field_type in entries:
            offset = self._field_offset(struct_type.struct_name, field_name)
            if arg_expr is None:
                # Omitted field in a named, partial literal (positional
                # construction is exhaustive, so arg_expr is never None
                # there) -- gets its own type's zero value via _ir_
                # write_zero_value_into, a TOTAL function that never
                # needs to fall back.
                if offset == 0:
                    field_addr = dst_address
                else:
                    field_addr = self.ir_program.ids.new_temp(Type.INT64)
                    ir.append(
                        IRBinOp(dst=field_addr, op=BinaryOp.ADD, left=dst_address, right=IRConst(offset, Type.INT64)))
                ir.extend(self._ir_write_zero_value_into(field_addr, field_type))
            elif field_type.kind in COMPOSITE_KINDS:
                if offset == 0:
                    field_addr = dst_address
                else:
                    field_addr = self.ir_program.ids.new_temp(Type.INT64)
                    ir.append(
                        IRBinOp(dst=field_addr, op=BinaryOp.ADD, left=dst_address, right=IRConst(offset, Type.INT64)))
                field_ir = self._ir_write_composite_value_into(field_addr, arg_expr, field_type)
                if field_ir is None:
                    return None
                ir.extend(field_ir)
            else:
                arg_ir, arg_value = self.gen_expr_ir(arg_expr)
                ir.extend(arg_ir)
                if offset == 0:
                    field_addr = dst_address
                else:
                    field_addr = self.ir_program.ids.new_temp(Type.INT64)
                    ir.append(
                        IRBinOp(dst=field_addr, op=BinaryOp.ADD, left=dst_address, right=IRConst(offset, Type.INT64)))
                ir.append(IRStore(address=field_addr, value=arg_value, value_type=field_type))
        return ir

    def _ir_materialize_struct_literal(self, expr: Call):
        """Builds (without lowering) a struct-literal Call's own
        materialized address as real IR -- returns (ir, address), or
        None when some field is out of scope for _ir_write_struct_
        literal_into. The struct-literal counterpart to _ir_
        materialize_array_literal in arrays_slices.py, sharing its
        same skeleton (a reserved slot, or malloc when none was
        reserved).

        No value_type ambiguity here, unlike an ArrayLiteral: a
        struct-literal Call's own type is unambiguously its own struct
        name, so type_of(expr) is simply, always correct."""
        struct_type = type_of(expr)
        if id(expr) in self._argument_temp_slots:
            slot = self._argument_temp_slots[id(expr)]
            addr = self.ir_program.ids.new_temp(Type.INT64)
            addr_ir = [IRLocalAddress(dst=addr, slot=slot)]
        else:
            addr = self.ir_program.ids.new_temp(Type.INT64)
            size = type_byte_width(struct_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
            addr_ir = [IRCall(dst=addr, name='malloc', args=[IRConst(size, Type.INT64)])]
        write_ir = self._ir_write_struct_literal_into(addr, expr, struct_type)
        if write_ir is None:
            return None
        return addr_ir + write_ir, addr

    def _check_struct_and_field_type(self, base_expr: Node, field_name: str) -> Type:
        """Returns field_name's declared type within base_expr's
        struct type. Doesn't raise on an invalid access -- already
        validated by semantic.py before codegen runs.

        A POINTER-to-struct base auto-dereferences here too, mirroring
        _ir_struct_address's own Variable-case decision (and semantic.
        py's _check_struct_and_field, the ORIGINAL source of this same
        decision) -- type_of(base_expr) reports a Variable's own
        LITERAL declared type (*Circle for `p`), never the auto-
        dereferenced one, since semantic analysis validates the access
        without rewriting the reference's own resolved_type -- so this
        needs to make the identical call itself, not assume type_of
        already made it."""
        base_type = type_of(base_expr)
        if base_type.kind == TypeKind.POINTER and base_type.element_type.kind == TypeKind.STRUCT:
            base_type = base_type.element_type
        return self.ir_program.struct_registry[base_type.struct_name].fields[field_name]



