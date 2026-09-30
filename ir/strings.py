"""Strings: immutable 16-byte {ptr, len} descriptors (len at offset 8). No NUL terminator.
Copies copy the descriptor only. Also print and its runtime type descriptors.
"""

from ir.errors import IRError
from ir.ir import IRBinOp, IRBoundsCheck, IRBranch, IRConst, IRCall, IRJump, IRLabel, IRSliceBoundsCheck, IRStaticDataAddress, IRLocalAddress, IRLoad, IRMove, IRStore
from ir.utils import type_of
from typesys import SUM_TYPE_TAG_WIDTH, type_byte_width
from parser import (
    Call,
    Binary,
    Field,
    Index,
    Node,
    Slice,
    StringLiteral,
    Unary,
    Variable,
)
from ops import BinaryOp, UnaryOp
from typesys import Type, TypeKind


# Kind tags for print's type descriptors; source of truth for runtime/generate_typedesc_header.py.
_TYPEDESC_INT = 0
_TYPEDESC_BOOL = 1
_TYPEDESC_STR = 2
_TYPEDESC_ARRAY = 3
_TYPEDESC_SLICE = 4
_TYPEDESC_STRUCT = 5
_TYPEDESC_INT8 = 6
_TYPEDESC_UINT8 = 7
_TYPEDESC_SUM = 9
_TYPEDESC_POINTER = 10
_TYPEDESC_DICT = 11
_TYPEDESC_INT32 = 12


class StringsMixin:
    def _get_or_build_type_descriptor(self, t: Type, in_progress: dict[Type, str]) -> str:
        """Label of t's type descriptor, built once per program."""
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
        elif t == Type.INT32:
            self.ir_program.type_descriptors.append((label, [_TYPEDESC_INT32]))
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
        elif t.kind == TypeKind.SUM:
            # [tag, variant_count, variant_desc...]
            sum_type_info = self.ir_program.sum_type_registry[t.sum_type_name]
            variant_desc_labels = [
                self._get_or_build_type_descriptor(variant_type, in_progress)
                for variant_type in sum_type_info.variants
            ]
            self.ir_program.type_descriptors.append((label, [_TYPEDESC_SUM, len(variant_desc_labels)] + variant_desc_labels))
        elif t.kind == TypeKind.POINTER:
            # [tag]; pointers print as addresses.
            self.ir_program.type_descriptors.append((label, [_TYPEDESC_POINTER]))
        elif t.kind == TypeKind.DICT:
            # [tag, name, key_desc, key_width, value_desc, value_width]
            name_label = self.ir_program.ids.new_label("typedesc_name")
            self.ir_program.string_literals.append((name_label, str(t)))
            key_label = self._get_or_build_type_descriptor(t.key_type, in_progress)
            key_width = type_byte_width(t.key_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
            value_label = self._get_or_build_type_descriptor(t.element_type, in_progress)
            value_width = type_byte_width(t.element_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
            self.ir_program.type_descriptors.append(
                (label, [_TYPEDESC_DICT, name_label, key_label, key_width, value_label, value_width]))
        else:
            raise IRError(f"No type descriptor rule for: {t}")

        return label

    def _ir_str_address(self, expr: Node):
        """Address of a str descriptor."""
        if isinstance(expr, Variable):
            slot_type = self._local_type(expr)
            slot = self._local_slot(expr)
            slot_addr = self.ir_program.ids.new_temp(Type.INT64)
            ir = [IRLocalAddress(dst=slot_addr, slot=slot)]
            if self._is_heap_allocated(self._local_decl_id(expr), slot_type):
                addr_temp = self.ir_program.ids.new_temp(Type.INT64)
                ir.append(IRLoad(dst=addr_temp, address=slot_addr))
                base_addr = addr_temp
            else:
                base_addr = slot_addr
            if slot_type.kind == TypeKind.SUM and expr.resolved_type is not None and expr.resolved_type != slot_type:
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
            # `*p` as a whole str.
            return self.gen_expr_ir(expr.operand)
        return None

    def _ir_read_str_descriptor_from_address(self, descriptor_addr) -> tuple:
        """Read {ptr, len} from an address."""
        ptr_temp = self.ir_program.ids.new_temp(Type.INT64)
        len_addr = self.ir_program.ids.new_temp(Type.INT64)
        len_temp = self.ir_program.ids.new_temp(Type.INT)
        ir = [
            IRLoad(dst=ptr_temp, address=descriptor_addr),
            IRBinOp(dst=len_addr, op=BinaryOp.ADD, left=descriptor_addr, right=IRConst(8, Type.INT64)),
            IRLoad(dst=len_temp, address=len_addr),
        ]
        return ir, ptr_temp, len_temp

    def _ir_write_str_descriptor_into_address(self, dst_address, ptr_value, len_value) -> list:
        """Write {ptr, len} to an address."""
        len_addr = self.ir_program.ids.new_temp(Type.INT64)
        return [
            IRStore(address=dst_address, value=ptr_value, value_type=Type.INT64),
            IRBinOp(dst=len_addr, op=BinaryOp.ADD, left=dst_address, right=IRConst(8, Type.INT64)),
            IRStore(address=len_addr, value=len_value, value_type=Type.INT),
        ]

    def _ir_write_str_descriptor(self, dst_expr: Node, ptr_value, len_value) -> list:
        """Write {ptr, len} to dst_expr's slot."""
        result = self._ir_str_address(dst_expr)
        if result is None:
            raise IRError(
                f"_ir_str_address returned None for a str-descriptor write's "
                f"own destination ({dst_expr!r}) -- expected to always succeed, "
                f"since an assignment's own destination is always a Variable/"
                f"Field/Index, never a str production or Call")
        dst_ir, dst_addr = result
        return dst_ir + self._ir_write_str_descriptor_into_address(dst_addr, ptr_value, len_value)

    def _ir_zero_str_value(self) -> tuple:
        """{0, 0}."""
        ptr = self.ir_program.ids.new_temp(Type.INT64)
        length = self.ir_program.ids.new_temp(Type.INT)
        ir = [
            IRMove(dst=ptr, src=IRConst(0, Type.INT64)),
            IRMove(dst=length, src=IRConst(0, Type.INT)),
        ]
        return ir, ptr, length

    def _ir_str_value(self, expr: Node) -> tuple:
        """{ptr, len} of any str expression."""
        if isinstance(expr, StringLiteral):
            ptr = self.ir_program.ids.new_temp(Type.INT64)
            label = self.ir_program.ids.new_label("str")
            self.ir_program.string_literals.append((label, expr.value))
            return [IRStaticDataAddress(dst=ptr, label=label)], ptr, IRConst(len(expr.value), Type.INT)
        if isinstance(expr, Binary) and expr.op == BinaryOp.ADD:
            return self._ir_string_concat(expr)
        if isinstance(expr, Slice):
            return self._ir_str_slice_into(expr)
        if isinstance(expr, Call) and self.ir_program.intrinsic_original_names.get(expr.name) == '_from_raw_parts':
            # from_raw_parts intrinsic: builds {ptr, len} inline.
            ptr_ir, ptr_value = self.gen_expr_ir(expr.args[0])
            len_ir, len_value = self.gen_expr_ir(expr.args[1])
            return ptr_ir + len_ir, ptr_value, len_value
        result = self._ir_str_address(expr)
        if result is not None:
            addr_ir, addr_value = result
            read_ir, ptr_value, len_value = self._ir_read_str_descriptor_from_address(addr_value)
            return addr_ir + read_ir, ptr_value, len_value
        if isinstance(expr, Call) and expr.name in self.ir_program.struct_registry:
            return None
        if isinstance(expr, Call):
            addr_ir, addr_value = self._ir_materialize_composite_call(expr, Type.STR)
            read_ir, ptr_value, len_value = self._ir_read_str_descriptor_from_address(addr_value)
            return addr_ir + read_ir, ptr_value, len_value
        return None

    def _ir_str_slice_into(self, expr: Slice):
        """{ptr, len} of s[low:high]."""
        result = self._ir_str_value(expr.array)
        if result is None:
            return None
        base_ir, base_ptr, length_value = result

        if expr.high is not None:
            high_ir, high_value = self.gen_expr_ir(expr.high)
        else:
            high_ir, high_value = [], length_value
        if expr.low is not None:
            low_ir, low_value = self.gen_expr_ir(expr.low)
        else:
            low_ir, low_value = [], IRConst(0, Type.INT)

        checks = [
            IRSliceBoundsCheck(value=low_value, bound=length_value),
            IRSliceBoundsCheck(value=high_value, bound=length_value),
            IRSliceBoundsCheck(value=low_value, bound=high_value),
        ]

        new_len = self.ir_program.ids.new_temp(Type.INT)
        ptr = self.ir_program.ids.new_temp(Type.INT64)
        arithmetic = [
            IRBinOp(dst=new_len, op=BinaryOp.SUBTRACT, left=high_value, right=low_value),
            IRBinOp(dst=ptr, op=BinaryOp.ADD, left=base_ptr, right=low_value),
        ]

        ir = base_ir + high_ir + low_ir + checks + arithmetic
        return ir, ptr, new_len

    def _ir_str_index_into(self, expr: Index):
        """Byte s[index]."""
        result = self._ir_str_value(expr.array)
        if result is None:
            return None
        base_ir, base_ptr, length_value = result

        index_ir, index_value = self.gen_expr_ir(expr.index)
        check = IRBoundsCheck(index=index_value, length=length_value)

        byte_addr = self.ir_program.ids.new_temp(Type.INT64)
        add = IRBinOp(dst=byte_addr, op=BinaryOp.ADD, left=base_ptr, right=index_value)

        value = self.ir_program.ids.new_temp(Type.UINT8)
        load = IRLoad(dst=value, address=byte_addr)

        return base_ir + index_ir + [check, add, load], value

    def _ir_print_call(self, expr: Call):
        """print(x): pass x's address and type descriptor to hornet_print."""
        arg = expr.args[0]
        arg_type = type_of(arg)

        if isinstance(arg, Call) and arg.name in self.ir_program.struct_registry:
            result = self._ir_materialize_struct_literal(arg)
            if result is None:
                raise IRError(
                    f"_ir_materialize_struct_literal returned None for print()'s own "
                    f"struct-literal argument ({arg!r}) -- expected to always succeed, "
                    f"since semantic.py already validated every one of its fields"
                )
            value_addr_ir, value_addr = result
        elif arg_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT, TypeKind.SUM):
            result = self._ir_composite_operand_address(arg, arg_type)
            if result is None:
                raise IRError(
                    f"_ir_composite_operand_address returned None for print()'s own "
                    f"ARRAY/STRUCT/SUM-typed argument ({arg!r}) -- expected to always "
                    f"succeed, since semantic.py already restricts print's own "
                    f"argument to exactly the shapes that method covers"
                )
            value_addr_ir, value_addr = result
        elif arg_type.kind == TypeKind.STR:
            str_ir, ptr_value, len_value = self._ir_str_value(arg)
            str_addr = self.ir_program.ids.new_temp(Type.INT64)
            value_addr_ir = str_ir + [IRLocalAddress(dst=str_addr, slot=self._unnamed_str_temp_slot)]
            value_addr_ir.extend(self._ir_write_str_descriptor_into_address(str_addr, ptr_value, len_value))
            value_addr = str_addr
        elif arg_type.kind == TypeKind.SLICE:
            slice_ir, ptr_value, len_value, cap_value = self._ir_slice_arg(arg)
            slice_addr = self.ir_program.ids.new_temp(Type.INT64)
            value_addr_ir = slice_ir + [IRLocalAddress(dst=slice_addr, slot=self._unnamed_slice_temp_slot)]
            value_addr_ir.extend(
                self._ir_write_slice_descriptor_into_address(slice_addr, ptr_value, len_value, cap_value))
            value_addr = slice_addr
        elif arg_type.kind == TypeKind.DICT:
            result = self._ir_materialize_composite_call(arg, arg_type) if isinstance(arg, Call) else self._ir_dict_address(arg)
            if result is None:
                raise IRError(
                    f"_ir_dict_address returned None for print()'s own dict-typed "
                    f"argument ({arg!r}) -- only a bare variable (or a field/element "
                    f"access rooted in one) is supported as a dict-typed print() "
                    f"argument at this stage"
                )
            value_addr_ir, value_addr = result
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
        """`left + right`: malloc and copy. Never freed."""
        left_ir, left_ptr, left_len = self._ir_str_value(expr.left)
        right_ir, right_ptr, right_len = self._ir_str_value(expr.right)

        total_len = self.ir_program.ids.new_temp(Type.INT)
        new_buf = self.ir_program.ids.new_temp(Type.INT64)
        right_dst = self.ir_program.ids.new_temp(Type.INT64)

        ir = left_ir + right_ir + [
            IRBinOp(dst=total_len, op=BinaryOp.ADD, left=left_len, right=right_len),
            IRCall(dst=new_buf, name='malloc', args=[total_len]),
            IRCall(dst=None, name='memcpy', args=[new_buf, left_ptr, left_len]),
            IRBinOp(dst=right_dst, op=BinaryOp.ADD, left=new_buf, right=left_len),
            IRCall(dst=None, name='memcpy', args=[right_dst, right_ptr, right_len]),
        ]
        return ir, new_buf, total_len

    def _ir_string_compare(self, expr: Binary):
        """`==`/`!=`: compare lengths, then bytes."""
        left_ir, left_ptr, left_len = self._ir_str_value(expr.left)
        right_ir, right_ptr, right_len = self._ir_str_value(expr.right)

        lengths_equal = self.ir_program.ids.new_temp(Type.BOOL)
        cmp_result = self.ir_program.ids.new_temp(Type.INT32)  # C int
        t_result = self.ir_program.ids.new_temp(Type.BOOL)

        lengths_equal_label = self.ir_program.ids.new_label("str_cmp_lengths_equal")
        lengths_differ_label = self.ir_program.ids.new_label("str_cmp_lengths_differ")
        end_label = self.ir_program.ids.new_label("str_cmp_end")

        mismatch_result = 0 if expr.op == BinaryOp.EQUAL else 1

        ir = left_ir + right_ir + [
            IRBinOp(dst=lengths_equal, op=BinaryOp.EQUAL, left=left_len, right=right_len),
            IRBranch(cond=lengths_equal, true_label=lengths_equal_label, false_label=lengths_differ_label),
            IRLabel(lengths_equal_label),
            IRCall(dst=cmp_result, name='memcmp', args=[left_ptr, right_ptr, left_len]),
            IRBinOp(dst=t_result, op=expr.op, left=cmp_result, right=IRConst(0, Type.INT32)),
            IRJump(end_label),
            IRLabel(lengths_differ_label),
            IRMove(dst=t_result, src=IRConst(mismatch_result, Type.BOOL)),
            IRJump(end_label),
            IRLabel(end_label),
        ]
        return ir, t_result

