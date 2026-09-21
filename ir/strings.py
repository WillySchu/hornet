"""A str value is a plain pointer -- to a literal's static address, or
a concatenation's malloc'd buffer -- never copied for its bytes the
way an array or struct is. Also builds the runtime type-descriptor
tree print() needs (_get_or_build_type_descriptor) and print()'s own
real-IR call itself (_ir_print_call): the descriptor is walked, and
stringification happens, entirely in C, by the runtime's own hornet_
stringify/hornet_print (see runtime/runtime.c) -- this file just
builds the IRCall against hornet_print and the static descriptor data
it reads."""

from ir.errors import IRError
from ir.ir import IRBinOp, IRConst, IRCall, IRStaticDataAddress, IRLocalAddress, IRStore
from ir.utils import type_byte_width, type_of
from parser import Call, Binary, Node, BinaryOp
from semantic import Type, TypeKind


# Kind tags for print's runtime type descriptors (see
# _get_or_build_type_descriptor and AsmProgram.type_descriptors).
# Plain small ints, embedded as a descriptor's first .quad field;
# hornet_stringify switches on this to interpret the rest.
_TYPEDESC_INT = 0
_TYPEDESC_BOOL = 1
_TYPEDESC_STR = 2
_TYPEDESC_ARRAY = 3
_TYPEDESC_SLICE = 4
_TYPEDESC_STRUCT = 5
_TYPEDESC_INT8 = 6
_TYPEDESC_UINT8 = 7
_TYPEDESC_INT64 = 8


class StringsMixin:
    def _get_or_build_type_descriptor(self, t: Type, in_progress: dict[Type, str]) -> str:
        """Returns the label of t's runtime type descriptor, building
        and registering it into self.ir_program.type_descriptors if it
        hasn't already been built WITHIN THIS ONE CALL (in_progress,
        keyed on t -- Type is frozen, so it's a safe dict key).

        `in_progress` is scoped to a single top-level call, not a
        whole-program cache -- two print() calls on the same struct
        type each build their own tree from scratch. Reuse WITHIN one
        call isn't optional, though: it's the only way a self-
        referential struct (`type Node struct: int value; []Node
        children`) can be represented as a finite amount of static
        data at all. Reserving this type's label BEFORE recursing into
        anything it contains is what breaks that cycle.

        Every non-leaf kind (ARRAY, SLICE, STRUCT) carries its own
        type-name string (e.g. "[3]int", "Point") as a second field
        right after the kind tag; hornet_stringify prints this
        immediately before that kind's opening bracket/brace at every
        level it appears. INT/INT8/UINT8/BOOL/STR carry no name field
        at all."""
        if t in in_progress:
            return in_progress[t]
        label = self.ir_program.ids.new_label("typedesc")
        in_progress[t] = label

        if t.kind == TypeKind.INT:
            self.ir_program.type_descriptors.append((label, [_TYPEDESC_INT]))
        elif t == Type.INT8:
            self.ir_program.type_descriptors.append((label, [_TYPEDESC_INT8]))
        elif t == Type.UINT8:
            self.ir_program.type_descriptors.append((label, [_TYPEDESC_UINT8]))
        elif t == Type.INT64:
            self.ir_program.type_descriptors.append((label, [_TYPEDESC_INT64]))
        elif t.kind == TypeKind.BOOL:
            self.ir_program.type_descriptors.append((label, [_TYPEDESC_BOOL]))
        elif t.kind == TypeKind.STR:
            self.ir_program.type_descriptors.append((label, [_TYPEDESC_STR]))
        elif t.kind == TypeKind.ARRAY:
            name_label = self.ir_program.ids.new_label("typedesc_name")
            self.ir_program.string_literals.append((name_label, str(t)))
            elem_label = self._get_or_build_type_descriptor(t.element_type, in_progress)
            elem_width = type_byte_width(t.element_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
            self.ir_program.type_descriptors.append((label, [_TYPEDESC_ARRAY, name_label, elem_label, t.size, elem_width]))
        elif t.kind == TypeKind.SLICE:
            name_label = self.ir_program.ids.new_label("typedesc_name")
            self.ir_program.string_literals.append((name_label, str(t)))
            elem_label = self._get_or_build_type_descriptor(t.element_type, in_progress)
            elem_width = type_byte_width(t.element_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
            self.ir_program.type_descriptors.append((label, [_TYPEDESC_SLICE, name_label, elem_label, elem_width]))
        elif t.kind == TypeKind.STRUCT:
            name_label = self.ir_program.ids.new_label("typedesc_name")
            self.ir_program.string_literals.append((name_label, str(t)))
            struct_info = self.ir_program.struct_registry[t.struct_name]
            field_fields: list = []
            field_count = 0
            for field_name, field_type in struct_info.fields.items():
                field_name_label = self.ir_program.ids.new_label("typedesc_fname")
                self.ir_program.string_literals.append((field_name_label, field_name))
                field_type_label = self._get_or_build_type_descriptor(field_type, in_progress)
                field_offset = self._field_offset(t.struct_name, field_name)
                field_fields.extend([field_name_label, field_type_label, field_offset])
                field_count += 1
            self.ir_program.type_descriptors.append((label, [_TYPEDESC_STRUCT, name_label, field_count] + field_fields))
        else:
            raise IRError(f"No type descriptor rule for: {t}")

        return label

    def _get_empty_str_label(self) -> str:
        """A shared, static, empty ("") string constant -- str's zero
        value. Deliberately NOT a null pointer: every string operation
        this file generates dereferences a str value with no null
        check, so a zero-initialized str has to point at a real, valid
        (if empty) C string -- a null pointer would segfault the
        instant anything touched it."""
        if self.ir_program._empty_str_label is None:
            self.ir_program._empty_str_label = self.ir_program.ids.new_label("empty_str")
            self.ir_program.string_literals.append((self.ir_program._empty_str_label, ""))
        return self.ir_program._empty_str_label

    def _ir_print_call(self, expr: Call):
        """Builds (without lowering) print(x)'s own real IR -- returns
        (ir, None), a void call.

        Unlike an ordinary call, which passes a scalar BY VALUE and a
        composite by address, this always computes an ADDRESS for x,
        regardless of its own type: hornet_print's signature is
        uniformly `void hornet_print(void *value_addr, const unsigned
        char *type_desc)`, and hornet_stringify dereferences that
        address at whatever width its own type descriptor says to.

        ARRAY/STRUCT-typed x: reuses _ir_composite_operand_address
        unchanged. SLICE-typed x: _ir_slice_arg unifies every
        reachable shape into one {ptr, len, cap} triple, written into
        the shared, unconditionally-reserved 24-byte _unnamed_slice_
        temp_slot scratch slot, whose address is then taken. Otherwise
        a scalar: computed via gen_expr_ir, written into the shared
        8-byte _print_scalar_temp_slot scratch slot -- a scalar has no
        other "address of this value" concept in real IR, since it
        only ever needs to live in a Temp; this is a deliberate,
        narrow exception, since hornet_print is a genuine C-ABI
        boundary this compiler's own output has to cross."""
        arg = expr.args[0]
        arg_type = type_of(arg)

        if arg_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT):
            result = self._ir_composite_operand_address(arg, arg_type)
            if result is None:
                raise IRError(
                    f"_ir_composite_operand_address returned None for print()'s own "
                    f"ARRAY/STRUCT-typed argument ({arg!r}) -- expected to always "
                    f"succeed, since semantic.py already restricts print's own "
                    f"argument to exactly the shapes that method covers"
                )
            value_addr_ir, value_addr = result
        elif arg_type.kind == TypeKind.SLICE:
            slice_ir, ptr_value, len_value, cap_value = self._ir_slice_arg(arg)
            slice_addr = self.ir_program.ids.new_temp(Type.INT64)
            value_addr_ir = slice_ir + [IRLocalAddress(dst=slice_addr, slot=self._unnamed_slice_temp_slot)]
            value_addr_ir.extend(
                self._ir_write_slice_descriptor_into_address(slice_addr, ptr_value, len_value, cap_value))
            value_addr = slice_addr
        else:
            expr_ir, value = self.gen_expr_ir(arg)
            scalar_addr = self.ir_program.ids.new_temp(Type.INT64)
            value_addr_ir = expr_ir + [IRLocalAddress(dst=scalar_addr, slot=self._print_scalar_temp_slot)]
            value_addr_ir.append(IRStore(address=scalar_addr, value=value, value_type=arg_type))
            value_addr = scalar_addr

        desc_label = self._get_or_build_type_descriptor(arg_type, {})
        desc_addr = self.ir_program.ids.new_temp(Type.INT64)
        desc_ir = [IRStaticDataAddress(dst=desc_addr, label=desc_label)]

        call_ir = [IRCall(dst=None, name='hornet_print', args=[value_addr, desc_addr])]
        return value_addr_ir + desc_ir + call_ir, None

    def _ir_string_concat(self, expr: Binary):
        """Builds (without lowering) `left + right` (both str) as real
        IR -- returns (ir, value). Reuses ordinary IRCall directly for
        strlen/malloc/strcpy/strcat: IRCall's own lowering is already
        fully generic (just `CallInstr(instr.name)`), with no notion
        of "Hornet function" baked in, so an external C library call
        needs no new IR concept at all.

        left/right are each evaluated via gen_expr_ir, so left_value
        is already sitting safely in its own Temp home (register or
        memory) by the time right is evaluated, regardless of what
        that does internally -- no "protect left across evaluating
        right" dance needed.

        MEMORY: _ir_free_if_fresh_concat replicates gen_string_concat_
        into's own AST-shape check, called once left's bytes are
        copied (strcpy) and once right's are (strcat)."""
        left_ir, left_value = self.gen_expr_ir(expr.left)
        right_ir, right_value = self.gen_expr_ir(expr.right)

        len_left = self.ir_program.ids.new_temp(Type.INT64)
        len_right = self.ir_program.ids.new_temp(Type.INT64)
        total_len = self.ir_program.ids.new_temp(Type.INT64)
        plus_one = self.ir_program.ids.new_temp(Type.INT64)
        new_buf = self.ir_program.ids.new_temp(Type.STR)

        ir = left_ir + right_ir + [
            IRCall(dst=len_left, name='strlen', args=[left_value]),
            IRCall(dst=len_right, name='strlen', args=[right_value]),
            IRBinOp(dst=total_len, op=BinaryOp.ADD, left=len_left, right=len_right),
            IRBinOp(dst=plus_one, op=BinaryOp.ADD, left=total_len, right=IRConst(1, Type.INT64)),
            IRCall(dst=new_buf, name='malloc', args=[plus_one]),
            IRCall(dst=None, name='strcpy', args=[new_buf, left_value]),
        ]
        ir += self._ir_free_if_fresh_concat(expr.left, left_value)
        ir += [IRCall(dst=None, name='strcat', args=[new_buf, right_value])]
        ir += self._ir_free_if_fresh_concat(expr.right, right_value)
        return ir, new_buf

    def _ir_free_if_fresh_concat(self, operand: Node, value) -> list:
        """If `operand` is itself a Binary(ADD, ...) node -- meaning
        `value` is a fresh buffer _ir_string_concat just malloc'd for
        *this* expression alone -- frees it via IRCall(free).
        Everything else is left alone: a StringLiteral points into
        static `.data` (freeing it would corrupt the allocator); a
        Variable or a Call's return value might be aliased elsewhere
        -- telling those apart from a genuinely fresh, exclusively-
        owned buffer is a real escape-analysis problem this narrow
        check deliberately doesn't attempt."""
        if isinstance(operand, Binary) and operand.op == BinaryOp.ADD:
            return [IRCall(dst=None, name='free', args=[value])]
        return []

    def _ir_string_compare(self, expr: Binary):
        """Builds (without lowering) `left == right` / `left != right`
        (both str) as real IR -- returns (ir, value). Same shape as
        _ir_string_concat: IRCall(strcmp) plus an ordinary IRBinOp for
        the 0/1 bool conversion, via IRBinOp's own existing comparison
        lowering."""
        left_ir, left_value = self.gen_expr_ir(expr.left)
        right_ir, right_value = self.gen_expr_ir(expr.right)

        cmp_result = self.ir_program.ids.new_temp(Type.INT)
        t_result = self.ir_program.ids.new_temp(Type.BOOL)

        ir = left_ir + right_ir + [IRCall(dst=cmp_result, name='strcmp', args=[left_value, right_value])]
        ir += self._ir_free_if_fresh_concat(expr.left, left_value)
        ir += self._ir_free_if_fresh_concat(expr.right, right_value)
        ir += [IRBinOp(dst=t_result, op=expr.op, left=cmp_result, right=IRConst(0, Type.INT))]
        return ir, t_result



