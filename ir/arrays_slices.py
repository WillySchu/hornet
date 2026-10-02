"""Arrays (value types, bounds-checked, stack unless promoted) and slices ({ptr, len, cap} views; append grows into cap)."""

from typing import Union

from ir.errors import IRError
from ir.ir import (
    IRBinOp,
    IRBoundsCheck,
    IRBranch,
    IRCall,
    IRConst,
    IRCopy,
    IRJump,
    IRLabel,
    IRLoad,
    IRLocalAddress,
    IRMove,
    IRSliceBoundsCheck,
    IRStaticDataAddress,
    IRStore, Temp,
)
from ir.utils import COMPOSITE_KINDS, is_composite_addressable, root_variable, type_of
from typesys import SUM_TYPE_TAG_WIDTH, type_byte_width
from parser import (
    Node,
    ArrayLiteral,
    Call,
    DictLiteral,
    Field,
    ForIn,
    Index,
    Slice,
    Variable,
    Unary,
)
from ops import BinaryOp, UnaryOp
from typesys import Type, TypeKind


class ArraysSlicesMixin:
    def _ir_array_address(self, expr: Node):
        """Address of an array-typed expression."""
        if isinstance(expr, Variable):
            slot = self._local_slot(expr)
            array_type = self._local_type(expr)
            slot_addr = self.ir_program.ids.new_temp(Type.INT64)
            ir = [IRLocalAddress(dst=slot_addr, slot=slot)]
            if self._is_heap_allocated(self._local_decl_id(expr), array_type):
                addr_temp = self.ir_program.ids.new_temp(Type.INT64)
                ir.append(IRLoad(dst=addr_temp, address=slot_addr))
                base_addr = addr_temp
            else:
                base_addr = slot_addr
            if array_type.kind == TypeKind.SUM and expr.resolved_type is not None and expr.resolved_type != array_type:
                payload_addr = self.ir_program.ids.new_temp(Type.INT64)
                ir.append(IRBinOp(
                    dst=payload_addr, op=BinaryOp.ADD,
                    left=base_addr, right=IRConst(SUM_TYPE_TAG_WIDTH, Type.INT64),
                ))
                return ir, payload_addr
            return ir, base_addr
        if isinstance(expr, Index):
            return self._ir_index_address(expr)
        if isinstance(expr, Field):
            return self._ir_field_address(expr)
        if isinstance(expr, Unary) and expr.op == UnaryOp.DEREFERENCE:
            # `*p` as a whole array: p is the address.
            return self._ir_pointer(expr.operand)
        return None

    def _ir_slice_address(self, expr: Node):
        """Address of a slice descriptor (through the box if heap-allocated)."""
        if isinstance(expr, Variable):
            slot_type = self._local_type(expr)
            if slot_type.kind != TypeKind.SUM:
                slot = self._local_slot(expr)
                addr_temp = self.ir_program.ids.new_temp(Type.INT64)
                ir = [IRLocalAddress(dst=addr_temp, slot=slot)]
                if self._is_heap_allocated(self._local_decl_id(expr), slot_type):
                    boxed = self.ir_program.ids.new_temp(Type.INT64)
                    return ir + [IRLoad(dst=boxed, address=addr_temp)], boxed
                return ir, addr_temp
            slot = self._local_slot(expr)
            slot_addr = self.ir_program.ids.new_temp(Type.INT64)
            ir = [IRLocalAddress(dst=slot_addr, slot=slot)]
            if self._is_heap_allocated(self._local_decl_id(expr), slot_type):
                addr_temp = self.ir_program.ids.new_temp(Type.INT64)
                ir.append(IRLoad(dst=addr_temp, address=slot_addr))
                base_addr = addr_temp
            else:
                base_addr = slot_addr
            if expr.resolved_type is not None and expr.resolved_type != slot_type:
                payload_addr = self.ir_program.ids.new_temp(Type.INT64)
                ir.append(IRBinOp(
                    dst=payload_addr, op=BinaryOp.ADD,
                    left=base_addr, right=IRConst(SUM_TYPE_TAG_WIDTH, Type.INT64),
                ))
                return ir, payload_addr
            return ir, base_addr
        if isinstance(expr, Index):
            return self._ir_index_address(expr)
        if isinstance(expr, Field):
            return self._ir_field_address(expr)
        if isinstance(expr, Unary) and expr.op == UnaryOp.DEREFERENCE:
            # `*p` as a whole slice: p is the descriptor address.
            return self._ir_pointer(expr.operand)
        return None

    def _ir_materialize_composite_call(
            self, call_expr: Call, value_type: Type) -> tuple[Union[IRLocalAddress, IRCall], Temp]:
        """Materialize a composite call's result; returns (ir, address)."""
        if call_expr.nid in self._argument_temp_slots:
            slot = self._argument_temp_slots[call_expr.nid]
            addr = self.ir_program.ids.new_temp(Type.INT64)
            addr_ir = [IRLocalAddress(dst=addr, slot=slot)]
        else:
            addr = self.ir_program.ids.new_temp(Type.INT64)
            size = type_byte_width(value_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
            addr_ir = [IRCall(dst=addr, name='malloc', args=[IRConst(size, Type.INT64)])]
        call_ir = self._ir_composite_call(addr, call_expr)
        return addr_ir + call_ir, addr

    def _ir_materialize_array_literal(self, expr: ArrayLiteral):
        """Materialize an ArrayLiteral; returns (ir, address) or None."""
        array_type = type_of(expr)
        if expr.nid in self._argument_temp_slots:
            slot = self._argument_temp_slots[expr.nid]
            addr = self.ir_program.ids.new_temp(Type.INT64)
            addr_ir = [IRLocalAddress(dst=addr, slot=slot)]
        else:
            addr = self.ir_program.ids.new_temp(Type.INT64)
            size = type_byte_width(array_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
            addr_ir = [IRCall(dst=addr, name='malloc', args=[IRConst(size, Type.INT64)])]
        write_ir = self._ir_write_array_literal_into(addr, expr, array_type)
        if write_ir is None:
            return None
        return addr_ir + write_ir, addr

    def _ir_read_slice_descriptor_from_address(self, descriptor_addr) -> tuple:
        """Read {ptr, len, cap} from a descriptor address."""
        ptr_temp = self.ir_program.ids.new_temp(Type.INT64)
        len_addr = self.ir_program.ids.new_temp(Type.INT64)
        len_temp = self.ir_program.ids.new_temp(Type.INT)
        cap_addr = self.ir_program.ids.new_temp(Type.INT64)
        cap_temp = self.ir_program.ids.new_temp(Type.INT)
        ir = [
            IRLoad(dst=ptr_temp, address=descriptor_addr),
            IRBinOp(dst=len_addr, op=BinaryOp.ADD, left=descriptor_addr, right=IRConst(8, Type.INT64)),
            IRLoad(dst=len_temp, address=len_addr),
            IRBinOp(dst=cap_addr, op=BinaryOp.ADD, left=descriptor_addr, right=IRConst(16, Type.INT64)),
            IRLoad(dst=cap_temp, address=cap_addr),
        ]
        return ir, ptr_temp, len_temp, cap_temp

    def _ir_indexable_base(self, expr: Node):
        """(ir, addr, length, cap) of an indexable base, or None."""
        base_type = type_of(expr)
        if base_type.kind == TypeKind.ARRAY:
            if self._is_ordinary_composite_call(expr):
                addr_ir, addr_value = self._ir_materialize_composite_call(expr, base_type)
                size_const = IRConst(base_type.size, Type.INT)
                return addr_ir, addr_value, size_const, size_const
            if isinstance(expr, ArrayLiteral):
                production = self._ir_materialize_array_literal(expr)
                if production is None:
                    return None
                addr_ir, addr_value = production
                size_const = IRConst(base_type.size, Type.INT)
                return addr_ir, addr_value, size_const, size_const
            if not is_composite_addressable(expr):
                return None
            addr_result = self._ir_array_address(expr)
            if addr_result is None:
                return None
            addr_ir, addr_value = addr_result
            size_const = IRConst(base_type.size, Type.INT)
            return addr_ir, addr_value, size_const, size_const
        if base_type.kind == TypeKind.SLICE:
            if isinstance(expr, Variable):
                addr_result = self._ir_slice_address(expr)
                if addr_result is None:
                    return None
                addr_ir, descriptor_addr = addr_result
                ir, ptr_temp, len_temp, cap_temp = self._ir_read_slice_descriptor_from_address(descriptor_addr)
                return addr_ir + ir, ptr_temp, len_temp, cap_temp
            if isinstance(expr, (Field, Index)):
                addr_result = self._ir_field_address(expr) if isinstance(expr, Field) else self._ir_index_address(expr)
                if addr_result is None:
                    return None
                addr_ir, descriptor_addr = addr_result
                ir, ptr_temp, len_temp, cap_temp = self._ir_read_slice_descriptor_from_address(descriptor_addr)
                return addr_ir + ir, ptr_temp, len_temp, cap_temp
            if isinstance(expr, Unary) and expr.op == UnaryOp.DEREFERENCE:
                # `*p` as a slice base.
                addr_ir, descriptor_addr = self._ir_pointer(expr.operand)
                ir, ptr_temp, len_temp, cap_temp = self._ir_read_slice_descriptor_from_address(descriptor_addr)
                return addr_ir + ir, ptr_temp, len_temp, cap_temp
            if isinstance(expr, Call) and expr.name == 'append':
                return self._ir_append_call(expr)
            if self._is_ordinary_composite_call(expr):
                addr_ir, descriptor_addr = self._ir_materialize_composite_call(expr, base_type)
                ir, ptr_temp, len_temp, cap_temp = self._ir_read_slice_descriptor_from_address(descriptor_addr)
                return addr_ir + ir, ptr_temp, len_temp, cap_temp
            if isinstance(expr, Slice):
                return self._ir_slice_into(expr)
            return None
        return None

    def _ir_slice_arg(self, expr: Node):
        """Slice call argument as {ptr, len, cap}."""
        return self._ir_indexable_base(expr)

    def _ir_index_address(self, expr: Index):
        """Bounds-checked address of expr.array[expr.index]."""
        if type_of(expr.array).kind == TypeKind.DICT:
            return self._ir_dict_lookup(expr.array, expr.index, type_of(expr.array))
        base = self._ir_indexable_base(expr.array)
        if base is None:
            return None
        base_ir, base_addr, length_value, _cap_value = base
        element_type = type_of(expr.array).element_type
        element_stride = type_byte_width(element_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)

        index_ir, index_value = self.gen_expr_ir(expr.index)
        check = IRBoundsCheck(index=index_value, length=length_value)

        offset_temp = self.ir_program.ids.new_temp(Type.INT)
        multiply = IRBinOp(
            dst=offset_temp, op=BinaryOp.MULTIPLY, left=index_value, right=IRConst(element_stride, Type.INT))

        result = self.ir_program.ids.new_temp(Type.INT64)
        add = IRBinOp(dst=result, op=BinaryOp.ADD, left=base_addr, right=offset_temp)

        return base_ir + index_ir + [check, multiply, add], result

    def _ir_slice_into(self, expr: Slice):
        """{ptr, len, cap} of expr.array[low:high]."""
        base = self._ir_indexable_base(expr.array)
        if base is None:
            return None
        base_ir, base_addr, length_value, cap_value = base
        element_stride = type_byte_width(type_of(expr.array).element_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)

        if expr.high is not None:
            high_ir, high_value = self.gen_expr_ir(expr.high)
        else:
            high_ir, high_value = [], length_value
        if expr.low is not None:
            low_ir, low_value = self.gen_expr_ir(expr.low)
        else:
            low_ir, low_value = [], IRConst(0, Type.INT)

        checks = [
            IRSliceBoundsCheck(value=low_value, bound=cap_value),
            IRSliceBoundsCheck(value=high_value, bound=cap_value),
            IRSliceBoundsCheck(value=low_value, bound=high_value),
        ]

        new_cap = self.ir_program.ids.new_temp(Type.INT)
        new_len = self.ir_program.ids.new_temp(Type.INT)
        offset_temp = self.ir_program.ids.new_temp(Type.INT)
        ptr = self.ir_program.ids.new_temp(Type.INT64)
        arithmetic = [
            IRBinOp(dst=new_cap, op=BinaryOp.SUBTRACT, left=cap_value, right=low_value),
            IRBinOp(dst=new_len, op=BinaryOp.SUBTRACT, left=high_value, right=low_value),
            IRBinOp(dst=offset_temp, op=BinaryOp.MULTIPLY, left=low_value, right=IRConst(element_stride, Type.INT)),
            IRBinOp(dst=ptr, op=BinaryOp.ADD, left=base_addr, right=offset_temp),
        ]

        ir = base_ir + high_ir + low_ir + checks + arithmetic
        return ir, ptr, new_len, new_cap

    def _ir_slice_literal(self, expr: ArrayLiteral):
        """Heap-allocated slice from a bracketed literal."""
        array_type = type_of(expr)
        count = len(expr.elements)
        size = max(1, type_byte_width(array_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry))
        ptr = self.ir_program.ids.new_temp(Type.INT64)
        malloc_ir = [IRCall(dst=ptr, name='malloc', args=[IRConst(size, Type.INT64)])]
        write_ir = self._ir_write_array_literal_into(ptr, expr, array_type)
        if write_ir is None:
            return None
        len_value = IRConst(count, Type.INT)
        cap_value = IRConst(count, Type.INT)
        return malloc_ir + write_ir, ptr, len_value, cap_value

    def _ir_write_slice_descriptor_into_address(self, dst_address, ptr_value, len_value, cap_value) -> list:
        """Write {ptr, len, cap} to an address."""
        len_addr = self.ir_program.ids.new_temp(Type.INT64)
        cap_addr = self.ir_program.ids.new_temp(Type.INT64)
        return [
            IRStore(address=dst_address, value=ptr_value, value_type=Type.INT64),
            IRBinOp(dst=len_addr, op=BinaryOp.ADD, left=dst_address, right=IRConst(8, Type.INT64)),
            IRStore(address=len_addr, value=len_value, value_type=Type.INT),
            IRBinOp(dst=cap_addr, op=BinaryOp.ADD, left=dst_address, right=IRConst(16, Type.INT64)),
            IRStore(address=cap_addr, value=cap_value, value_type=Type.INT),
        ]

    def _ir_write_slice_descriptor(self, dst_expr: Node, ptr_value, len_value, cap_value) -> list:
        """Write {ptr, len, cap} to dst_expr's slot."""
        result = self._ir_slice_address(dst_expr)
        if result is None:
            raise IRError(
                f"_ir_slice_address returned None for a slice-descriptor write's "
                f"own destination ({dst_expr!r}) -- expected to always succeed, "
                f"since an assignment's own destination is always a Variable/"
                f"Field/Index, never a Slice production or Call")
        dst_ir, dst_addr = result
        return dst_ir + self._ir_write_slice_descriptor_into_address(dst_addr, ptr_value, len_value, cap_value)

    def _ir_write_zero_value_into(self, dst_address, value_type: Type) -> list:
        """Write value_type's zero value through dst_address."""
        if value_type.kind == TypeKind.SUM:
            # Semantic analysis allows this only for a sum with a `none` variant: that variant is the zero.
            variants = self.ir_program.sum_type_registry[value_type.sum_type_name].variants
            return [IRStore(address=dst_address, value=IRConst(variants.index(Type.NONE), Type.INT32), value_type=Type.INT32)]
        if value_type.kind == TypeKind.SLICE:
            zero_ptr = IRConst(0, Type.INT64)
            zero_int = IRConst(0, Type.INT)
            return self._ir_write_slice_descriptor_into_address(dst_address, zero_ptr, zero_int, zero_int)
        if value_type.kind == TypeKind.STRUCT:
            struct_info = self.ir_program.struct_registry[value_type.struct_name]
            ir = []
            for field_name, field_type in struct_info.fields.items():
                offset = self._field_offset(value_type.struct_name, field_name)
                if offset == 0:
                    field_addr = dst_address
                else:
                    field_addr = self.ir_program.ids.new_temp(Type.INT64)
                    ir.append(
                        IRBinOp(dst=field_addr, op=BinaryOp.ADD, left=dst_address, right=IRConst(offset, Type.INT64)))
                ir.extend(self._ir_write_zero_value_into(field_addr, field_type))
            return ir
        if value_type.kind == TypeKind.ARRAY:
            return self._ir_zero_array_loop(dst_address, value_type.element_type, value_type.size)
        if value_type == Type.STR:
            zero_ir, ptr_value, len_value = self._ir_zero_str_value()
            return zero_ir + self._ir_write_str_descriptor_into_address(dst_address, ptr_value, len_value)
        if value_type.kind == TypeKind.DICT:
            return self._ir_new_empty_dict_into(dst_address)
        return [IRStore(address=dst_address, value=IRConst(0, value_type), value_type=value_type)]

    def _ir_zero_array_loop(self, dst_address, element_type: Type, count: int) -> list:
        """Loop zeroing `count` elements."""
        element_width = type_byte_width(element_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        i = self.ir_program.ids.new_temp(Type.INT)
        start_label = self.ir_program.ids.new_label("zero_array_start")
        body_label = self.ir_program.ids.new_label("zero_array_body")
        end_label = self.ir_program.ids.new_label("zero_array_end")
        cond = self.ir_program.ids.new_temp(Type.BOOL)
        ir = [
            IRMove(dst=i, src=IRConst(0, Type.INT)),
            IRJump(start_label),
            IRLabel(start_label),
            IRBinOp(dst=cond, op=BinaryOp.LESS_THAN, left=i, right=IRConst(count, Type.INT)),
            IRBranch(cond=cond, true_label=body_label, false_label=end_label),
            IRLabel(body_label),
        ]
        offset_temp = self.ir_program.ids.new_temp(Type.INT)
        ir.append(IRBinOp(dst=offset_temp, op=BinaryOp.MULTIPLY, left=i, right=IRConst(element_width, Type.INT)))
        elem_addr = self.ir_program.ids.new_temp(Type.INT64)
        ir.append(IRBinOp(dst=elem_addr, op=BinaryOp.ADD, left=dst_address, right=offset_temp))
        ir.extend(self._ir_write_zero_value_into(elem_addr, element_type))
        next_i = self.ir_program.ids.new_temp(Type.INT)
        ir.append(IRBinOp(dst=next_i, op=BinaryOp.ADD, left=i, right=IRConst(1, Type.INT)))
        ir.append(IRMove(dst=i, src=next_i))
        ir.append(IRJump(start_label))
        ir.append(IRLabel(end_label))
        return ir

    def _ir_composite_equal(self, left_addr, right_addr, value_type: Type, mismatch_label: str) -> list:
        """Jump to mismatch_label on the first differing element/field/byte; else fall through."""
        if value_type.kind == TypeKind.ARRAY:
            element_type = value_type.element_type
            element_width = type_byte_width(element_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
            i = self.ir_program.ids.new_temp(Type.INT)
            start_label = self.ir_program.ids.new_label("eq_array_start")
            body_label = self.ir_program.ids.new_label("eq_array_body")
            end_label = self.ir_program.ids.new_label("eq_array_end")
            cond = self.ir_program.ids.new_temp(Type.BOOL)
            ir = [
                IRMove(dst=i, src=IRConst(0, Type.INT)),
                IRJump(start_label),
                IRLabel(start_label),
                IRBinOp(dst=cond, op=BinaryOp.LESS_THAN, left=i, right=IRConst(value_type.size, Type.INT)),
                IRBranch(cond=cond, true_label=body_label, false_label=end_label),
                IRLabel(body_label),
            ]
            offset_temp = self.ir_program.ids.new_temp(Type.INT)
            ir.append(IRBinOp(dst=offset_temp, op=BinaryOp.MULTIPLY, left=i, right=IRConst(element_width, Type.INT)))
            left_elem_addr = self.ir_program.ids.new_temp(Type.INT64)
            ir.append(IRBinOp(dst=left_elem_addr, op=BinaryOp.ADD, left=left_addr, right=offset_temp))
            right_elem_addr = self.ir_program.ids.new_temp(Type.INT64)
            ir.append(IRBinOp(dst=right_elem_addr, op=BinaryOp.ADD, left=right_addr, right=offset_temp))
            ir.extend(self._ir_composite_equal(left_elem_addr, right_elem_addr, element_type, mismatch_label))
            next_i = self.ir_program.ids.new_temp(Type.INT)
            ir.append(IRBinOp(dst=next_i, op=BinaryOp.ADD, left=i, right=IRConst(1, Type.INT)))
            ir.append(IRMove(dst=i, src=next_i))
            ir.append(IRJump(start_label))
            ir.append(IRLabel(end_label))
            return ir
        if value_type.kind == TypeKind.STRUCT:
            struct_info = self.ir_program.struct_registry[value_type.struct_name]
            ir = []
            for field_name, field_type in struct_info.fields.items():
                offset = self._field_offset(value_type.struct_name, field_name)
                if offset == 0:
                    left_field_addr = left_addr
                    right_field_addr = right_addr
                else:
                    left_field_addr = self.ir_program.ids.new_temp(Type.INT64)
                    ir.append(
                        IRBinOp(
                            dst=left_field_addr, op=BinaryOp.ADD, left=left_addr, right=IRConst(offset, Type.INT64),
                        ),
                    )
                    right_field_addr = self.ir_program.ids.new_temp(Type.INT64)
                    ir.append(
                        IRBinOp(
                            dst=right_field_addr, op=BinaryOp.ADD, left=right_addr, right=IRConst(offset, Type.INT64),
                        ),
                    )
                ir.extend(self._ir_composite_equal(left_field_addr, right_field_addr, field_type, mismatch_label))
            return ir
        continue_label = self.ir_program.ids.new_label("eq_continue")
        if value_type == Type.STR:
            left_read_ir, left_ptr, left_len = self._ir_read_str_descriptor_from_address(left_addr)
            right_read_ir, right_ptr, right_len = self._ir_read_str_descriptor_from_address(right_addr)
            lengths_equal = self.ir_program.ids.new_temp(Type.BOOL)
            lengths_equal_label = self.ir_program.ids.new_label("eq_str_lengths_equal")
            cmp_result = self.ir_program.ids.new_temp(Type.INT32)  # C int
            mismatch_cond = self.ir_program.ids.new_temp(Type.BOOL)
            return left_read_ir + right_read_ir + [
                IRBinOp(dst=lengths_equal, op=BinaryOp.EQUAL, left=left_len, right=right_len),
                IRBranch(cond=lengths_equal, true_label=lengths_equal_label, false_label=mismatch_label),
                IRLabel(lengths_equal_label),
                IRCall(dst=cmp_result, name='memcmp', args=[left_ptr, right_ptr, left_len]),
                IRBinOp(dst=mismatch_cond, op=BinaryOp.NOT_EQUAL, left=cmp_result, right=IRConst(0, Type.INT32)),
                IRBranch(cond=mismatch_cond, true_label=mismatch_label, false_label=continue_label),
                IRLabel(continue_label),
            ]
        # int, bool, int8, uint8
        left_val = self.ir_program.ids.new_temp(value_type)
        right_val = self.ir_program.ids.new_temp(value_type)
        mismatch_cond = self.ir_program.ids.new_temp(Type.BOOL)
        return [
            IRLoad(dst=left_val, address=left_addr),
            IRLoad(dst=right_val, address=right_addr),
            IRBinOp(dst=mismatch_cond, op=BinaryOp.NOT_EQUAL, left=left_val, right=right_val),
            IRBranch(cond=mismatch_cond, true_label=mismatch_label, false_label=continue_label),
            IRLabel(continue_label),
        ]

    def _ir_array_slice_contains(self, needle_expr: Node, collection_expr: Node, element_type: Type) -> tuple[list, object]:
        """`x in array/slice`: linear scan."""
        needle_ir, needle_addr = self._ir_materialize_value_into_scratch(
            needle_expr, element_type, self.ir_fn, "in_needle")
        base = self._ir_indexable_base(collection_expr)
        if base is None:
            raise IRError(
                f"_ir_indexable_base returned None for an ARRAY/SLICE-typed "
                f"'in' right operand ({collection_expr!r}) -- expected to "
                f"always succeed for a reachable indexable base")
        base_ir, base_addr, length_value, _ = base
        element_width = type_byte_width(element_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)

        i = self.ir_program.ids.new_temp(Type.INT)
        start_label = self.ir_program.ids.new_label("in_start")
        body_label = self.ir_program.ids.new_label("in_body")
        next_label = self.ir_program.ids.new_label("in_next")
        found_label = self.ir_program.ids.new_label("in_found")
        not_found_label = self.ir_program.ids.new_label("in_not_found")
        done_label = self.ir_program.ids.new_label("in_done")
        cond = self.ir_program.ids.new_temp(Type.BOOL)
        ir = needle_ir + base_ir + [
            IRMove(dst=i, src=IRConst(0, Type.INT)),
            IRJump(start_label),
            IRLabel(start_label),
            IRBinOp(dst=cond, op=BinaryOp.LESS_THAN, left=i, right=length_value),
            IRBranch(cond=cond, true_label=body_label, false_label=not_found_label),
            IRLabel(body_label),
        ]
        offset_temp = self.ir_program.ids.new_temp(Type.INT)
        ir.append(IRBinOp(dst=offset_temp, op=BinaryOp.MULTIPLY, left=i, right=IRConst(element_width, Type.INT)))
        elem_addr = self.ir_program.ids.new_temp(Type.INT64)
        ir.append(IRBinOp(dst=elem_addr, op=BinaryOp.ADD, left=base_addr, right=offset_temp))
        ir.extend(self._ir_composite_equal(needle_addr, elem_addr, element_type, next_label))
        ir.append(IRJump(found_label))
        ir.append(IRLabel(next_label))
        next_i = self.ir_program.ids.new_temp(Type.INT)
        ir.append(IRBinOp(dst=next_i, op=BinaryOp.ADD, left=i, right=IRConst(1, Type.INT)))
        ir.append(IRMove(dst=i, src=next_i))
        ir.append(IRJump(start_label))
        t = self.ir_program.ids.new_temp(Type.BOOL)
        ir.extend([
            IRLabel(not_found_label),
            IRMove(dst=t, src=IRConst(0, Type.BOOL)),
            IRJump(done_label),
            IRLabel(found_label),
            IRMove(dst=t, src=IRConst(1, Type.BOOL)),
            IRJump(done_label),
            IRLabel(done_label),
        ])
        return ir, t

    def _ir_bind_for_in_value(self, stmt: ForIn, binding_index: int, binding_type: Type, source_addr, ir_fn) -> list:
        """Copy a value at source_addr into a for-in binding."""
        if binding_type.kind in COMPOSITE_KINDS:
            decl_id = stmt.symbols[binding_index].id
            slot = ir_fn.var_slots[decl_id]
            if self._is_heap_allocated(decl_id, binding_type):
                ir = self._ir_malloc_and_store(binding_type, slot)
            else:
                ir = []
            address_of = {
                TypeKind.ARRAY: self._ir_array_address,
                TypeKind.STRUCT: self._ir_struct_address,
                TypeKind.SLICE: self._ir_slice_address,
                TypeKind.DICT: self._ir_dict_address,
                TypeKind.SUM: self._ir_struct_address,
                TypeKind.STR: self._ir_str_address,
            }[binding_type.kind]
            dst_ir, dst_addr = address_of(Variable(name=stmt.binding_names[binding_index], decl_id=decl_id))
            ir.extend(dst_ir)
            ir.append(IRCopy(dst_address=dst_addr, src_address=source_addr, value_type=binding_type))
            return ir
        value = self.ir_program.ids.new_temp(binding_type)
        ir = [IRLoad(dst=value, address=source_addr)]
        ir.extend(self._ir_finish_scalar_var_decl(stmt.binding_names[binding_index], stmt.symbols[binding_index].id, binding_type, value))
        return ir

    def _ir_for_in_array_slice(self, stmt: ForIn, ir_fn) -> list:
        """`for [i,] x in array/slice/str`."""
        iterable_type = type_of(stmt.iterable)
        if iterable_type.kind == TypeKind.STR:
            str_ir, str_ptr, str_len = self._ir_str_value(stmt.iterable)
            base = (str_ir, str_ptr, str_len, None)
            element_type, element_width, is_slice = Type.UINT8, 1, False
        else:
            element_type = iterable_type.element_type
            element_width = type_byte_width(element_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
            is_slice = iterable_type.kind == TypeKind.SLICE
            base = self._ir_indexable_base(stmt.iterable)
        if base is None:
            raise IRError(
                f"_ir_indexable_base returned None for an ARRAY/SLICE-typed "
                f"'for ... in' iterable ({stmt.iterable!r}) -- expected to "
                f"always succeed for a reachable indexable base")
        base_ir, base_addr, length_value, _ = base
        ir = base_ir

        # Slices are rechecked each iteration: panic if the root's ptr changed (reallocated by append).
        # recheck_base is the root's ptr, not base_addr (offset for re-slices).
        descriptor_expr = None
        recheck_base = base_addr
        if is_slice and not isinstance(stmt.iterable, ArrayLiteral):
            if isinstance(stmt.iterable, Slice):
                root = root_variable(stmt.iterable)
                if root is None:
                    raise IRError(
                        f"root_variable returned None for a Slice-"
                        f"shaped 'for ... in' iterable ({stmt.iterable!r}) "
                        f"-- expected to always succeed, semantic.py's own "
                        f"analyze_for_in already having confirmed a real "
                        f"root variable exists before this method ever runs")
                if self._local_type(root).kind == TypeKind.SLICE:
                    descriptor_expr = root
            else:
                descriptor_expr = stmt.iterable

        if descriptor_expr is not None:
            descriptor_result = self._ir_slice_address(descriptor_expr)
            if descriptor_result is None:
                raise IRError(
                    f"_ir_slice_address returned None for a SLICE-typed "
                    f"'for ... in' iterable's own recheck target "
                    f"({descriptor_expr!r}) -- expected to always succeed, "
                    f"having already succeeded once above via "
                    f"_ir_indexable_base or resolved to a plain Variable")
            descriptor_ir, descriptor_addr = descriptor_result
            ir = ir + descriptor_ir
            if descriptor_expr is not stmt.iterable:
                root_ptr = self.ir_program.ids.new_temp(Type.INT64)
                ir.append(IRLoad(dst=root_ptr, address=descriptor_addr))
                recheck_base = root_ptr

        i = self.ir_program.ids.new_temp(Type.INT)
        start_label = self.ir_program.ids.new_label("for_in_start")
        body_label = self.ir_program.ids.new_label("for_in_body")
        continue_label = self.ir_program.ids.new_label("for_in_continue")
        end_label = self.ir_program.ids.new_label("for_in_end")

        binding_types = stmt.binding_types
        slots = [self._bind_for_in_binding(stmt, idx, bt, ir_fn) for idx, bt in enumerate(binding_types)]

        ir = ir + [
            IRMove(dst=i, src=IRConst(0, Type.INT)),
            IRJump(start_label),
            IRLabel(start_label),
        ]
        cond = self.ir_program.ids.new_temp(Type.BOOL)
        ir.append(IRBinOp(dst=cond, op=BinaryOp.LESS_THAN, left=i, right=length_value))
        ir.append(IRBranch(cond=cond, true_label=body_label, false_label=end_label))
        ir.append(IRLabel(body_label))

        if descriptor_expr is not None:
            recheck_ptr = self.ir_program.ids.new_temp(Type.INT64)
            ir.append(IRLoad(dst=recheck_ptr, address=descriptor_addr))
            mutated = self.ir_program.ids.new_temp(Type.BOOL)
            ir.append(IRBinOp(dst=mutated, op=BinaryOp.NOT_EQUAL, left=recheck_ptr, right=recheck_base))
            mutated_label = self.ir_program.ids.new_label("for_in_mutated")
            safe_label = self.ir_program.ids.new_label("for_in_safe")
            ir.append(IRBranch(cond=mutated, true_label=mutated_label, false_label=safe_label))
            ir.append(IRLabel(mutated_label))
            msg_ptr = self.ir_program.ids.new_temp(Type.INT64)
            msg_label = self.ir_program.ids.new_label("for_in_mutated_msg")
            self.ir_program.string_literals.append(
                (msg_label, "for ... in: slice was reallocated (e.g. by append) during iteration"))
            ir.append(IRStaticDataAddress(dst=msg_ptr, label=msg_label))
            ir.append(IRCall(dst=None, name='hornet_panic', args=[msg_ptr]))
            ir.append(IRJump(safe_label))  # unreachable; the verifier requires a terminator
            ir.append(IRLabel(safe_label))

        offset_temp = self.ir_program.ids.new_temp(Type.INT)
        ir.append(IRBinOp(dst=offset_temp, op=BinaryOp.MULTIPLY, left=i, right=IRConst(element_width, Type.INT)))
        elem_addr = self.ir_program.ids.new_temp(Type.INT64)
        ir.append(IRBinOp(dst=elem_addr, op=BinaryOp.ADD, left=base_addr, right=offset_temp))

        if len(slots) == 2:
            ir.extend(self._ir_finish_scalar_var_decl(stmt.binding_names[0], stmt.symbols[0].id, Type.INT, i))
            element_index = 1
        else:
            element_index = 0
        ir.extend(self._ir_bind_for_in_value(stmt, element_index, element_type, elem_addr, ir_fn))

        self.loop_labels.append((continue_label, end_label))
        for s in stmt.body:
            ir.extend(self.gen_statement_ir(s, ir_fn))
        self.loop_labels.pop()

        ir.append(IRJump(continue_label))
        ir.append(IRLabel(continue_label))
        next_i = self.ir_program.ids.new_temp(Type.INT)
        ir.append(IRBinOp(dst=next_i, op=BinaryOp.ADD, left=i, right=IRConst(1, Type.INT)))
        ir.append(IRMove(dst=i, src=next_i))
        ir.append(IRJump(start_label))
        ir.append(IRLabel(end_label))
        return ir

    def _ir_write_composite_value_into(self, dst_address, value_expr: Node, value_type: Type):
        """Write any composite value expression through dst_address."""
        if value_type.kind == TypeKind.SUM and type_of(value_expr).kind != TypeKind.SUM:
            return self._ir_write_sum_type_value_into(dst_address, value_expr, value_type)
        if is_composite_addressable(value_expr):
            return self._ir_copy_into_address(dst_address, value_expr, value_type)
        if value_type.kind == TypeKind.SLICE and isinstance(value_expr, Slice):
            production = self._ir_slice_into(value_expr)
            if production is None:
                return None
            slice_ir, ptr_value, len_value, cap_value = production
            return slice_ir + self._ir_write_slice_descriptor_into_address(dst_address, ptr_value, len_value, cap_value)
        if value_type.kind == TypeKind.SLICE and isinstance(value_expr, Call) and value_expr.name == 'append':
            production = self._ir_append_call(value_expr)
            if production is None:
                return None
            append_ir, ptr_value, len_value, cap_value = production
            return append_ir + self._ir_write_slice_descriptor_into_address(
                dst_address, ptr_value, len_value, cap_value)
        if value_type.kind == TypeKind.ARRAY and isinstance(value_expr, ArrayLiteral):
            return self._ir_write_array_literal_into(dst_address, value_expr, value_type)
        if value_type.kind == TypeKind.SLICE and isinstance(value_expr, ArrayLiteral):
            production = self._ir_slice_literal(value_expr)
            if production is None:
                return None
            slice_ir, ptr_value, len_value, cap_value = production
            return slice_ir + self._ir_write_slice_descriptor_into_address(dst_address, ptr_value, len_value, cap_value)
        if value_type.kind == TypeKind.DICT and isinstance(value_expr, DictLiteral):
            return self._ir_write_dict_literal_into(dst_address, value_expr, value_type, self.ir_fn)
        if value_type.kind == TypeKind.STR:
            # str literal or concatenation.
            value_ir, ptr_value, len_value = self._ir_str_value(value_expr)
            return value_ir + self._ir_write_str_descriptor_into_address(dst_address, ptr_value, len_value)
        if isinstance(value_expr, Call) and value_expr.name in self.ir_program.struct_registry:
            return self._ir_write_struct_literal_into(dst_address, value_expr, value_type)
        if isinstance(value_expr, Call):
            return self._ir_composite_call(dst_address, value_expr)
        return None

    def _ir_write_array_literal_into(self, dst_address, expr: ArrayLiteral, array_type: Type):
        """Write an array literal's elements through dst_address, or None."""
        element_type = array_type.element_type
        element_width = type_byte_width(element_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        ir = []
        for i, elem_expr in enumerate(expr.elements):
            if element_type.kind in COMPOSITE_KINDS:
                if i == 0:
                    elem_addr = dst_address
                else:
                    elem_addr = self.ir_program.ids.new_temp(Type.INT64)
                    ir.append(
                        IRBinOp(
                            dst=elem_addr,
                            op=BinaryOp.ADD,
                            left=dst_address,
                            right=IRConst(i * element_width, Type.INT64),
                        ),
                    )
                elem_ir = self._ir_write_composite_value_into(elem_addr, elem_expr, element_type)
                if elem_ir is None:
                    return None
                ir.extend(elem_ir)
            else:
                elem_ir, elem_value = self.gen_expr_ir(elem_expr)
                ir.extend(elem_ir)
                if i == 0:
                    elem_addr = dst_address
                else:
                    elem_addr = self.ir_program.ids.new_temp(Type.INT64)
                    ir.append(
                        IRBinOp(
                            dst=elem_addr,
                            op=BinaryOp.ADD,
                            left=dst_address,
                            right=IRConst(i * element_width, Type.INT64),
                        ),
                    )
                ir.append(IRStore(address=elem_addr, value=elem_value, value_type=element_type))
        return ir

    def _ir_array_literal_side_effects_only(self, expr: ArrayLiteral) -> list:
        """Evaluate a bare array literal statement for side effects only."""
        ir = []
        for element in expr.elements:
            if isinstance(element, ArrayLiteral):
                ir.extend(self._ir_array_literal_side_effects_only(element))
                continue
            element_type = type_of(element)
            if element_type.kind in COMPOSITE_KINDS:
                raise IRError(
                    f"A bare array-literal statement can't have a "
                    f"{type(element).__name__} element of type "
                    f"{element_type} -- assign the literal to a "
                    f"variable first if you need this element's value "
                    f"or side effect evaluated"
                )
            elem_ir, _ = self.gen_expr_ir(element)
            ir.extend(elem_ir)
        return ir

    def _ir_write_append_value_at(self, target_addr, value_arg: Node, element_type: Type):
        """Write append's value at target_addr."""
        if element_type.kind in COMPOSITE_KINDS:
            return self._ir_write_composite_value_into(target_addr, value_arg, element_type)
        value_ir, value = self.gen_expr_ir(value_arg)
        return value_ir + [IRStore(address=target_addr, value=value, value_type=element_type)]

    def _ir_append_call(self, expr: Call):
        """append(s, v): reuse spare capacity or grow; returns (ir, ptr, len, cap)."""
        slice_arg, value_arg = expr.args
        slice_type = type_of(slice_arg)
        element_type = slice_type.element_type
        element_width = type_byte_width(element_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)

        base = self._ir_indexable_base(slice_arg)
        if base is None:
            return None
        base_ir, ptr, length, cap = base

        result_ptr = self.ir_program.ids.new_temp(Type.INT64)
        result_len = self.ir_program.ids.new_temp(Type.INT)
        result_cap = self.ir_program.ids.new_temp(Type.INT)

        reuse_label = self.ir_program.ids.new_label("ir_append_reuse")
        realloc_label = self.ir_program.ids.new_label("ir_append_realloc")
        end_label = self.ir_program.ids.new_label("ir_append_end")

        needs_realloc = self.ir_program.ids.new_temp(Type.BOOL)
        check = IRBinOp(dst=needs_realloc, op=BinaryOp.GREATER_THAN_OR_EQUAL, left=length, right=cap)
        branch = IRBranch(cond=needs_realloc, true_label=realloc_label, false_label=reuse_label)

        # len < cap: write in place.
        offset_temp = self.ir_program.ids.new_temp(Type.INT)
        reuse_target_addr = self.ir_program.ids.new_temp(Type.INT64)
        new_len = self.ir_program.ids.new_temp(Type.INT)
        reuse_write_ir = self._ir_write_append_value_at(reuse_target_addr, value_arg, element_type)
        if reuse_write_ir is None:
            return None
        reuse_ir = [
            IRLabel(reuse_label),
            IRBinOp(dst=offset_temp, op=BinaryOp.MULTIPLY, left=length, right=IRConst(element_width, Type.INT)),
            IRBinOp(dst=reuse_target_addr, op=BinaryOp.ADD, left=ptr, right=offset_temp),
        ] + reuse_write_ir + [
            IRBinOp(dst=new_len, op=BinaryOp.ADD, left=length, right=IRConst(1, Type.INT)),
            IRMove(dst=result_ptr, src=ptr),
            IRMove(dst=result_len, src=new_len),
            IRMove(dst=result_cap, src=cap),
            IRJump(end_label),
        ]

        # len == cap: new_cap = cap == 0 ? 1 : cap < 256 ? cap * 2 : cap + cap / 4, then grow and write.
        ids = self.ir_program.ids
        grow_ptr = ids.new_temp(Type.INT64)
        grow_cap = ids.new_temp(Type.INT)
        is_zero, is_small, quarter = ids.new_temp(Type.BOOL), ids.new_temp(Type.BOOL), ids.new_temp(Type.INT)
        zero_label, nonzero_label = ids.new_label("append_cap_zero"), ids.new_label("append_cap_nonzero")
        small_label, large_label = ids.new_label("append_cap_small"), ids.new_label("append_cap_large")
        grow_label = ids.new_label("append_grow")
        grow = [
            IRBinOp(dst=is_zero, op=BinaryOp.EQUAL, left=cap, right=IRConst(0, Type.INT)),
            IRBranch(cond=is_zero, true_label=zero_label, false_label=nonzero_label),
            IRLabel(zero_label),
            IRMove(dst=grow_cap, src=IRConst(1, Type.INT)),
            IRJump(grow_label),
            IRLabel(nonzero_label),
            IRBinOp(dst=is_small, op=BinaryOp.LESS_THAN, left=cap, right=IRConst(256, Type.INT)),
            IRBranch(cond=is_small, true_label=small_label, false_label=large_label),
            IRLabel(small_label),
            IRBinOp(dst=grow_cap, op=BinaryOp.ADD, left=cap, right=cap),
            IRJump(grow_label),
            IRLabel(large_label),
            IRBinOp(dst=quarter, op=BinaryOp.SHIFT_RIGHT, left=cap, right=IRConst(2, Type.INT)),
            IRBinOp(dst=grow_cap, op=BinaryOp.ADD, left=cap, right=quarter),
            IRJump(grow_label),
            IRLabel(grow_label),
            IRCall(dst=grow_ptr, name='hornet_slice_grow',
                   args=[ptr, length, grow_cap, IRConst(element_width, Type.INT)]),
        ]
        realloc_offset_temp = self.ir_program.ids.new_temp(Type.INT)
        realloc_target_addr = self.ir_program.ids.new_temp(Type.INT64)
        realloc_new_len = self.ir_program.ids.new_temp(Type.INT)
        realloc_write_ir = self._ir_write_append_value_at(realloc_target_addr, value_arg, element_type)
        if realloc_write_ir is None:
            return None
        realloc_ir = [
            IRLabel(realloc_label),
        ] + grow + [
            IRBinOp(dst=realloc_offset_temp, op=BinaryOp.MULTIPLY, left=length, right=IRConst(element_width, Type.INT)),
            IRBinOp(dst=realloc_target_addr, op=BinaryOp.ADD, left=grow_ptr, right=realloc_offset_temp),
        ] + realloc_write_ir + [
            IRBinOp(dst=realloc_new_len, op=BinaryOp.ADD, left=length, right=IRConst(1, Type.INT)),
            IRMove(dst=result_ptr, src=grow_ptr),
            IRMove(dst=result_len, src=realloc_new_len),
            IRMove(dst=result_cap, src=grow_cap),
            IRJump(end_label),
        ]

        ir = base_ir + [check, branch] + reuse_ir + realloc_ir + [IRLabel(end_label)]
        return ir, result_ptr, result_len, result_cap

    def _ir_len_call(self, expr: Call):
        """len(x)."""
        arg = expr.args[0]
        if type_of(arg).kind == TypeKind.DICT:
            result = self._ir_dict_header(arg)
            if result is None:
                return None
            dict_ir, descriptor_addr = result
            count_addr = self.ir_program.ids.new_temp(Type.INT64)
            length = self.ir_program.ids.new_temp(Type.INT)
            return dict_ir + [
                IRBinOp(dst=count_addr, op=BinaryOp.ADD, left=descriptor_addr, right=IRConst(8, Type.INT64)),
                IRLoad(dst=length, address=count_addr),
            ], length
        if type_of(arg).kind == TypeKind.STR:
            result = self._ir_str_value(arg)
            if result is None:
                return None
            str_ir, ptr, length = result
            return str_ir, length
        base = self._ir_indexable_base(arg)
        if base is None:
            return None
        base_ir, ptr, length, cap = base
        return base_ir, length

