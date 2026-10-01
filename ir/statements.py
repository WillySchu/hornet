"""Statement IR: declarations, assignment, control flow, return. Uninitialized declarations get their type's zero value."""

from ir.errors import IRError
from ir.ir import (
    IRBinOp,
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
    IRStore,
)
from ir.utils import COMPOSITE_KINDS, is_composite_addressable, type_of
from typesys import type_byte_width
from parser import (
    ArrayLiteral,
    Assign,
    Break,
    Call,
    Continue,
    DerefAssign,
    DictLiteral,
    ExprStmt,
    Field,
    FieldAssign,
    For,
    ForIn,
    If,
    Index,
    IndexAssign,
    IsCheck,
    Node,
    NoneLiteral,
    Return,
    Slice,
    VarDecl,
    While,
)
from ops import BinaryOp
from typesys import Type, TypeKind


class StatementsMixin:
    def gen_statement_ir(self, stmt: Node, ir_fn: IRFunction) -> list:
        """Build IR for one statement."""
        if isinstance(stmt, Return):
            is_composite_return = (isinstance(stmt.value, NoneLiteral) and ir_fn.return_type.kind == TypeKind.SLICE) or (
                stmt.value is not None and (
                    type_of(stmt.value).kind in COMPOSITE_KINDS
                    or ir_fn.return_type.kind == TypeKind.SUM
                )
            )
            if not is_composite_return:
                return self._ir_return(stmt.value)
            # `return none` (slice).
            if isinstance(stmt.value, NoneLiteral):
                hidden_ptr_ir, hidden_ptr = self._ir_hidden_return_ptr(ir_fn)
                nil_ir, ptr_value, len_value, cap_value = self._ir_nil_slice()
                write_ir = self._ir_write_slice_descriptor_into_address(hidden_ptr, ptr_value, len_value, cap_value)
                return hidden_ptr_ir + nil_ir + write_ir + [IRReturn(value=None)]
            # Widen a variant into a sum return.
            if ir_fn.return_type.kind == TypeKind.SUM and type_of(stmt.value).kind != TypeKind.SUM:
                hidden_ptr_ir, hidden_ptr = self._ir_hidden_return_ptr(ir_fn)
                write_ir = self._ir_write_sum_type_value_into(hidden_ptr, stmt.value, ir_fn.return_type)
                if write_ir is not None:
                    return hidden_ptr_ir + write_ir + [IRReturn(value=None)]
            # `return f()`: forward our hidden pointer.
            if (
                    isinstance(stmt.value, Call)
                    and stmt.value.name != 'append'
                    and stmt.value.name not in self.ir_program.struct_registry
                    and stmt.value.name not in self.ir_program.intrinsic_original_names
            ):
                hidden_ptr_ir, hidden_ptr = self._ir_hidden_return_ptr(ir_fn)
                call_ir = self._ir_composite_call(hidden_ptr, stmt.value)
                return hidden_ptr_ir + call_ir + [IRReturn(value=None)]
            # Copy from an addressable value.
            if is_composite_addressable(stmt.value):
                value_type = type_of(stmt.value)
                hidden_ptr_ir, hidden_ptr = self._ir_hidden_return_ptr(ir_fn)
                copy_ir = self._ir_copy_into_address(hidden_ptr, stmt.value, value_type)
                return hidden_ptr_ir + copy_ir + [IRReturn(value=None)]
            # str literal or concatenation.
            if ir_fn.return_type.kind == TypeKind.STR:
                hidden_ptr_ir, hidden_ptr = self._ir_hidden_return_ptr(ir_fn)
                value_ir, ptr_value, len_value = self._ir_str_value(stmt.value)
                write_ir = self._ir_write_str_descriptor_into_address(hidden_ptr, ptr_value, len_value)
                return hidden_ptr_ir + value_ir + write_ir + [IRReturn(value=None)]
            # Literal: dispatch on the declared return type, since a literal's own type is always ARRAY.
            if isinstance(stmt.value, ArrayLiteral):
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
            if isinstance(stmt.value, DictLiteral):
                # Dict literal.
                hidden_ptr_ir, hidden_ptr = self._ir_hidden_return_ptr(ir_fn)
                write_ir = self._ir_write_dict_literal_into(hidden_ptr, stmt.value, ir_fn.return_type, ir_fn)
                return hidden_ptr_ir + write_ir + [IRReturn(value=None)]
            if isinstance(stmt.value, Call) and stmt.value.name in self.ir_program.struct_registry:
                value_type = type_of(stmt.value)
                hidden_ptr_ir, hidden_ptr = self._ir_hidden_return_ptr(ir_fn)
                write_ir = self._ir_write_struct_literal_into(hidden_ptr, stmt.value, value_type)
                if write_ir is not None:
                    return hidden_ptr_ir + write_ir + [IRReturn(value=None)]
            # `return a[lo:hi]`.
            if isinstance(stmt.value, Slice):
                hidden_ptr_ir, hidden_ptr = self._ir_hidden_return_ptr(ir_fn)
                production = self._ir_slice_into(stmt.value)
                if production is not None:
                    slice_ir, ptr_value, len_value, cap_value = production
                    write_ir = self._ir_write_slice_descriptor_into_address(
                        hidden_ptr, ptr_value, len_value, cap_value)
                    return hidden_ptr_ir + slice_ir + write_ir + [IRReturn(value=None)]
            # Remaining composite shapes.
        elif isinstance(stmt, If):
            then_label = self.ir_program.ids.new_label("if_then")
            else_label = self.ir_program.ids.new_label("if_else")
            end_label = self.ir_program.ids.new_label("if_end")
            # Bind an IsCheck binding before the condition.
            has_binding = isinstance(stmt.condition, IsCheck) and stmt.condition.binding_decl is not None
            ir = []
            if has_binding:
                ir.extend(self.gen_statement_ir(stmt.condition.binding_decl, ir_fn))
            ir.extend(self._ir_if_head(stmt, then_label, else_label))
            for s in stmt.then_body:
                ir.extend(self.gen_statement_ir(s, ir_fn))
            ir.append(IRJump(end_label))
            ir.append(IRLabel(else_label))
            if stmt.else_body is not None:
                for s in stmt.else_body:
                    ir.extend(self.gen_statement_ir(s, ir_fn))
            # Needed for else-less Ifs too: blocks can't fall through.
            ir.append(IRJump(end_label))
            ir.append(IRLabel(end_label))
            return ir
        elif isinstance(stmt, While):
            start_label = self.ir_program.ids.new_label("while_start")
            body_label = self.ir_program.ids.new_label("while_body")
            end_label = self.ir_program.ids.new_label("while_end")
            ir = self._ir_while_head(stmt, start_label, body_label, end_label)
            self.loop_labels.append((start_label, end_label))
            for s in stmt.body:
                ir.extend(self.gen_statement_ir(s, ir_fn))
            self.loop_labels.pop()
            ir.append(IRJump(start_label))
            ir.append(IRLabel(end_label))
            return ir
        elif isinstance(stmt, For):
            start_label = self.ir_program.ids.new_label("for_start")
            body_label = self.ir_program.ids.new_label("for_body")
            increment_label = self.ir_program.ids.new_label("for_increment")
            end_label = self.ir_program.ids.new_label("for_end")
            ir = self._ir_for_head(stmt, start_label, body_label, end_label, ir_fn)
            self.loop_labels.append((increment_label, end_label))
            for s in stmt.body:
                ir.extend(self.gen_statement_ir(s, ir_fn))
            self.loop_labels.pop()
            ir.append(IRJump(increment_label))
            ir.append(IRLabel(increment_label))
            ir.extend(self._ir_fresh_loop_variable(stmt))
            ir.extend(self.gen_statement_ir(stmt.increment, ir_fn))
            ir.append(IRJump(start_label))
            ir.append(IRLabel(end_label))
            return ir
        elif isinstance(stmt, ForIn):
            # Dispatch on iterable type (array, slice, or dict).
            if type_of(stmt.iterable).kind == TypeKind.DICT:
                return self._ir_for_in_dict(stmt, ir_fn)
            return self._ir_for_in_array_slice(stmt, ir_fn)
        elif isinstance(stmt, VarDecl):
            # Scalar VarDecl.
            var_type = stmt.resolved_type
            if var_type.kind not in COMPOSITE_KINDS:
                # Also covers `none` for pointers.
                self._bind_local(stmt, ir_fn)
                if stmt.init is not None:
                    ir, value = self.gen_expr_ir(stmt.init)
                    return ir + self._ir_finish_scalar_var_decl(stmt.name, stmt.symbol.id, var_type, value)
                else:
                    # No initializer: zero value.
                    t = IRConst(0, var_type)
                    ir = []
                    return ir + self._ir_finish_scalar_var_decl(stmt.name, stmt.symbol.id, var_type, t)
            # Widen a variant into a sum; sums always have initializers.
            if var_type.kind == TypeKind.SUM and type_of(stmt.init).kind != TypeKind.SUM:
                slot = self._bind_local(stmt, ir_fn)
                ir = []
                if self._is_heap_allocated(stmt.symbol.id, var_type):
                    ir.extend(self._ir_malloc_and_store(var_type, slot))
                dst_ir, dst_address = self._ir_struct_address(self._var_ref(stmt))
                ir.extend(dst_ir)
                write_ir = self._ir_write_sum_type_value_into(dst_address, stmt.init, var_type)
                if write_ir is not None:
                    return ir + write_ir
            # Copy from an addressable value.
            if (
                    var_type.kind in COMPOSITE_KINDS
                    and is_composite_addressable(stmt.init)
            ):
                slot = self._bind_local(stmt, ir_fn)
                ir = []
                if self._is_heap_allocated(stmt.symbol.id, var_type):
                    # New destination: allocate before writing.
                    ir.extend(self._ir_malloc_and_store(var_type, slot))
                return ir + self._ir_copy_assign(self._var_ref(stmt), stmt.init, var_type)
            # Non-addressable str initializer.
            if var_type.kind == TypeKind.STR:
                slot = self._bind_local(stmt, ir_fn)
                ir = []
                if self._is_heap_allocated(stmt.symbol.id, var_type):
                    ir.extend(self._ir_malloc_and_store(var_type, slot))
                if stmt.init is not None:
                    value_ir, ptr_value, len_value = self._ir_str_value(stmt.init)
                else:
                    value_ir, ptr_value, len_value = self._ir_zero_str_value()
                return ir + value_ir + self._ir_write_str_descriptor(self._var_ref(stmt), ptr_value, len_value)
            # `none` slice.
            if var_type.kind == TypeKind.SLICE and isinstance(stmt.init, NoneLiteral):
                nil_ir, ptr_value, len_value, cap_value = self._ir_nil_slice()
                nil_ir += self._ir_box_local(stmt, var_type, self._bind_local(stmt, ir_fn))
                return nil_ir + self._ir_write_slice_descriptor(
                    self._var_ref(stmt), ptr_value, len_value, cap_value)
            # Slice production.
            if var_type.kind == TypeKind.SLICE and isinstance(stmt.init, Slice):
                production = self._ir_slice_into(stmt.init)
                if production is not None:
                    slice_ir, ptr_value, len_value, cap_value = production
                    slice_ir += self._ir_box_local(stmt, var_type, self._bind_local(stmt, ir_fn))
                    return slice_ir + self._ir_write_slice_descriptor(
                        self._var_ref(stmt), ptr_value, len_value, cap_value)
            # Slice literal.
            if var_type.kind == TypeKind.SLICE and isinstance(stmt.init, ArrayLiteral):
                production = self._ir_slice_literal(stmt.init)
                if production is not None:
                    slice_ir, ptr_value, len_value, cap_value = production
                    slice_ir += self._ir_box_local(stmt, var_type, self._bind_local(stmt, ir_fn))
                    return slice_ir + self._ir_write_slice_descriptor(
                        self._var_ref(stmt), ptr_value, len_value, cap_value)
            # append.
            if var_type.kind == TypeKind.SLICE and isinstance(stmt.init, Call) and stmt.init.name == 'append':
                production = self._ir_append_call(stmt.init)
                if production is not None:
                    append_ir, ptr_value, len_value, cap_value = production
                    append_ir += self._ir_box_local(stmt, var_type, self._bind_local(stmt, ir_fn))
                    return append_ir + self._ir_write_slice_descriptor(
                        self._var_ref(stmt), ptr_value, len_value, cap_value)
            # Composite-returning call: write through the variable's address.
            if (
                    var_type.kind in COMPOSITE_KINDS
                    and (isinstance(stmt.init, Call)
                         and stmt.init.name != 'append'
                         and stmt.init.name not in self.ir_program.struct_registry)):
                slot = self._bind_local(stmt, ir_fn)
                ir = []
                if self._is_heap_allocated(stmt.symbol.id, var_type):
                    ir.extend(self._ir_malloc_and_store(var_type, slot))
                address_fn = {
                    TypeKind.ARRAY: self._ir_array_address,
                    TypeKind.STRUCT: self._ir_struct_address,
                    TypeKind.SLICE: self._ir_slice_address,
                    TypeKind.DICT: self._ir_dict_address,
                    TypeKind.SUM: self._ir_struct_address,
                }[var_type.kind]
                dst_ir, dst_address = address_fn(self._var_ref(stmt))
                ir.extend(dst_ir)
                return ir + self._ir_composite_call(dst_address, stmt.init)
            # Array or struct literal.
            if (
                    var_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT)
                    and (isinstance(stmt.init, ArrayLiteral)
                         or (isinstance(stmt.init, Call) and stmt.init.name in self.ir_program.struct_registry))):
                slot = self._bind_local(stmt, ir_fn)
                ir = []
                if self._is_heap_allocated(stmt.symbol.id, var_type):
                    ir.extend(self._ir_malloc_and_store(var_type, slot))
                address_fn = self._ir_array_address if var_type.kind == TypeKind.ARRAY else self._ir_struct_address
                dst_ir, dst_address = address_fn(self._var_ref(stmt))
                ir.extend(dst_ir)
                if isinstance(stmt.init, ArrayLiteral):
                    writer = self._ir_write_array_literal_into
                else:
                    writer = self._ir_write_struct_literal_into
                write_ir = writer(dst_address, stmt.init, var_type)
                if write_ir is not None:
                    return ir + write_ir
            # Dict literal.
            if var_type.kind == TypeKind.DICT and isinstance(stmt.init, DictLiteral):
                slot = self._bind_local(stmt, ir_fn)
                ir = []
                if self._is_heap_allocated(stmt.symbol.id, var_type):
                    ir.extend(self._ir_malloc_and_store(var_type, slot))
                dst_ir, dst_address = self._ir_dict_address(self._var_ref(stmt))
                ir.extend(dst_ir)
                return ir + self._ir_write_dict_literal_into(dst_address, stmt.init, var_type, ir_fn)
            # Nil slice.
            if var_type.kind == TypeKind.SLICE and stmt.init is None:
                box_ir = self._ir_box_local(stmt, var_type, self._bind_local(stmt, ir_fn))
                zero_ptr = IRConst(0, Type.INT64)
                zero_int = IRConst(0, Type.INT)
                return box_ir + self._ir_write_slice_descriptor(self._var_ref(stmt), zero_ptr, zero_int, zero_int)
            # Nil dict.
            if var_type.kind == TypeKind.DICT and stmt.init is None:
                slot = self._bind_local(stmt, ir_fn)
                ir = []
                if self._is_heap_allocated(stmt.symbol.id, var_type):
                    ir.extend(self._ir_malloc_and_store(var_type, slot))
                dst_ir, dst_address = self._ir_dict_address(self._var_ref(stmt))
                ir.extend(dst_ir)
                zero64 = IRConst(0, Type.INT64)
                zero_int = IRConst(0, Type.INT)
                count_addr = self.ir_program.ids.new_temp(Type.INT64)
                tombstones_addr = self.ir_program.ids.new_temp(Type.INT64)
                capacity_addr = self.ir_program.ids.new_temp(Type.INT64)
                ir.extend([
                    IRStore(address=dst_address, value=zero64, value_type=Type.INT64),
                    IRBinOp(dst=count_addr, op=BinaryOp.ADD, left=dst_address, right=IRConst(8, Type.INT64)),
                    IRStore(address=count_addr, value=zero_int, value_type=Type.INT),
                    IRBinOp(dst=tombstones_addr, op=BinaryOp.ADD, left=dst_address, right=IRConst(16, Type.INT64)),
                    IRStore(address=tombstones_addr, value=zero_int, value_type=Type.INT),
                    IRBinOp(dst=capacity_addr, op=BinaryOp.ADD, left=dst_address, right=IRConst(24, Type.INT64)),
                    IRStore(address=capacity_addr, value=zero64, value_type=Type.INT64),
                ])
                return ir
            # Zero array or struct.
            if var_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT) and stmt.init is None:
                slot = self._bind_local(stmt, ir_fn)
                ir = []
                if self._is_heap_allocated(stmt.symbol.id, var_type):
                    ir.extend(self._ir_malloc_and_store(var_type, slot))
                address_fn = self._ir_array_address if var_type.kind == TypeKind.ARRAY else self._ir_struct_address
                dst_ir, dst_address = address_fn(self._var_ref(stmt))
                ir.extend(dst_ir)
                return ir + self._ir_write_zero_value_into(dst_address, var_type)
        elif isinstance(stmt, Assign):
            # Scalar Assign.
            var_type = self._local_type(stmt)
            if var_type.kind not in COMPOSITE_KINDS:
                ir, value = self.gen_expr_ir(stmt.value)
                if self._is_heap_allocated(self._local_decl_id(stmt), var_type):
                    # Heap-promoted: store through the existing box.
                    return ir + [IRStore(address=self._local_temp(stmt), value=value, value_type=var_type)]
                return ir + [IRMove(dst=self._local_temp(stmt), src=value)]
            # Widen a variant into a sum.
            if var_type.kind == TypeKind.SUM and type_of(stmt.value).kind != TypeKind.SUM:
                dst_ir, dst_address = self._ir_struct_address(self._var_ref(stmt))
                write_ir = self._ir_write_sum_type_value_into(dst_address, stmt.value, var_type)
                if write_ir is not None:
                    return dst_ir + write_ir
            # Copy from an addressable value.
            if (
                    var_type.kind in COMPOSITE_KINDS
                    and is_composite_addressable(stmt.value)
            ):
                return self._ir_copy_assign(self._var_ref(stmt), stmt.value, var_type)
            # Non-addressable str value.
            if var_type.kind == TypeKind.STR:
                value_ir, ptr_value, len_value = self._ir_str_value(stmt.value)
                return value_ir + self._ir_write_str_descriptor(self._var_ref(stmt), ptr_value, len_value)
            # `none` slice.
            if var_type.kind == TypeKind.SLICE and isinstance(stmt.value, NoneLiteral):
                nil_ir, ptr_value, len_value, cap_value = self._ir_nil_slice()
                return nil_ir + self._ir_write_slice_descriptor(
                    self._var_ref(stmt), ptr_value, len_value, cap_value)
            # Slice production.
            if var_type.kind == TypeKind.SLICE and isinstance(stmt.value, Slice):
                production = self._ir_slice_into(stmt.value)
                if production is not None:
                    slice_ir, ptr_value, len_value, cap_value = production
                    return slice_ir + self._ir_write_slice_descriptor(
                        self._var_ref(stmt), ptr_value, len_value, cap_value)
            # Slice literal.
            if var_type.kind == TypeKind.SLICE and isinstance(stmt.value, ArrayLiteral):
                production = self._ir_slice_literal(stmt.value)
                if production is not None:
                    slice_ir, ptr_value, len_value, cap_value = production
                    return slice_ir + self._ir_write_slice_descriptor(
                        self._var_ref(stmt), ptr_value, len_value, cap_value)
            # append.
            if var_type.kind == TypeKind.SLICE and isinstance(stmt.value, Call) and stmt.value.name == 'append':
                production = self._ir_append_call(stmt.value)
                if production is not None:
                    append_ir, ptr_value, len_value, cap_value = production
                    return append_ir + self._ir_write_slice_descriptor(
                        self._var_ref(stmt), ptr_value, len_value, cap_value)
            # Composite-returning call.
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
                    TypeKind.DICT: self._ir_dict_address,
                    TypeKind.SUM: self._ir_struct_address,
                }[var_type.kind](self._var_ref(stmt))
                return dst_ir + self._ir_composite_call(dst_address, stmt.value)
            # Array or struct literal.
            if (
                    var_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT)
                    and (isinstance(stmt.value, ArrayLiteral)
                         or (isinstance(stmt.value, Call) and stmt.value.name in self.ir_program.struct_registry))
            ):
                address_fn = self._ir_array_address if var_type.kind == TypeKind.ARRAY else self._ir_struct_address
                dst_ir, dst_address = address_fn(self._var_ref(stmt))
                if isinstance(stmt.value, ArrayLiteral):
                    writer = self._ir_write_array_literal_into
                else:
                    writer = self._ir_write_struct_literal_into
                write_ir = writer(dst_address, stmt.value, var_type)
                if write_ir is not None:
                    return dst_ir + write_ir
        elif isinstance(stmt, IndexAssign):
            # Array elements are handled by composite copy paths.
            element_type = type_of(stmt.array).element_type
            # Dict element store.
            if type_of(stmt.array).kind == TypeKind.DICT:
                if stmt.compound_op is not None:
                    return self._ir_dict_compound_assign(
                        stmt.array, stmt.index, stmt.compound_op, stmt.value, type_of(stmt.array), ir_fn)
                return self._ir_dict_set(stmt.array, stmt.index, stmt.value, type_of(stmt.array), ir_fn)
            # Widen a variant into a sum element.
            if element_type.kind == TypeKind.SUM and type_of(stmt.value).kind != TypeKind.SUM:
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
            if element_type.kind not in COMPOSITE_KINDS:
                return self._ir_index_assign(stmt, element_type)
            if (
                    element_type.kind in (TypeKind.STRUCT, TypeKind.SLICE, TypeKind.SUM, TypeKind.STR)
                    and is_composite_addressable(stmt.value)
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
                    element_type.kind in (TypeKind.STRUCT, TypeKind.SLICE, TypeKind.SUM)
                    and (isinstance(stmt.value, Call)
                         and stmt.value.name != 'append'
                         and stmt.value.name not in self.ir_program.struct_registry)
            ):
                dst_expr = Index(array=stmt.array, index=stmt.index)
                address_fn = self._ir_slice_address if element_type.kind == TypeKind.SLICE else self._ir_struct_address
                dst_ir, dst_address = address_fn(dst_expr)
                return dst_ir + self._ir_composite_call(dst_address, stmt.value)
            # Non-addressable str value.
            if element_type.kind == TypeKind.STR:
                dst_expr = Index(array=stmt.array, index=stmt.index)
                value_ir, ptr_value, len_value = self._ir_str_value(stmt.value)
                return value_ir + self._ir_write_str_descriptor(dst_expr, ptr_value, len_value)
            # Struct literal.
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
            # Fields may be arrays, unlike index targets.
            field_type = self._check_struct_and_field_type(stmt.base, stmt.name)
            if field_type.kind not in COMPOSITE_KINDS:
                return self._ir_field_assign(stmt, field_type)
            if (
                    field_type.kind in COMPOSITE_KINDS
                    and is_composite_addressable(stmt.value)
            ):
                dst_expr = Field(base=stmt.base, name=stmt.name)
                return self._ir_copy_assign(dst_expr, stmt.value, field_type)
            # `none` slice.
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
                    TypeKind.DICT: self._ir_dict_address,
                    TypeKind.STR: self._ir_str_address,
                }[field_type.kind]
                dst_ir, dst_address = address_fn(dst_expr)
                return dst_ir + self._ir_composite_call(dst_address, stmt.value)
            # str literal or concatenation.
            if field_type.kind == TypeKind.STR:
                dst_expr = Field(base=stmt.base, name=stmt.name)
                value_ir, ptr_value, len_value = self._ir_str_value(stmt.value)
                return value_ir + self._ir_write_str_descriptor(dst_expr, ptr_value, len_value)
            # Array or struct literal.
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
        elif isinstance(stmt, DerefAssign):
            return self._ir_deref_assign(stmt)
        elif isinstance(stmt, Break):
            return self._ir_break()
        elif isinstance(stmt, Continue):
            return self._ir_continue()
        elif isinstance(stmt, ExprStmt) and isinstance(stmt.expr, ArrayLiteral):
            return self._ir_array_literal_side_effects_only(stmt.expr)
        elif isinstance(stmt, ExprStmt) and isinstance(stmt.expr, Slice) and type_of(stmt.expr).kind == TypeKind.SLICE:
            production = self._ir_slice_into(stmt.expr)
            if production is not None:
                slice_ir, _, _, _ = production
                return slice_ir
        elif isinstance(stmt, ExprStmt) and isinstance(stmt.expr, NoneLiteral):
            # Bare `none`: nothing to emit.
            return []
        elif isinstance(stmt, ExprStmt) and isinstance(stmt.expr, Call) and stmt.expr.name == 'append':
            # Bare append: keep side effects, discard the result.
            production = self._ir_append_call(stmt.expr)
            if production is not None:
                append_ir, _, _, _ = production
                return append_ir
        elif (
                isinstance(stmt, ExprStmt)
                and type_of(stmt.expr).kind in COMPOSITE_KINDS
                and self._is_ordinary_composite_call(stmt.expr)
        ):
            # Bare composite call: result discarded.
            ir, _ = self._ir_materialize_composite_call(stmt.expr, type_of(stmt.expr))
            return ir
        elif isinstance(stmt, ExprStmt) and isinstance(stmt.expr, Call) and stmt.expr.name in self.ir_program.struct_registry:
            # Bare struct literal.
            result = self._ir_materialize_struct_literal(stmt.expr)
            if result is None:
                raise IRError(
                    f"_ir_materialize_struct_literal returned None for a struct "
                    f"literal used as a bare statement ({stmt.expr!r}) -- some "
                    f"field is out of scope for real IR"
                )
            ir, _ = result
            return ir
        elif isinstance(stmt, ExprStmt) and type_of(stmt.expr).kind in COMPOSITE_KINDS:
            return self._ir_discarded_composite(stmt.expr)
        elif isinstance(stmt, ExprStmt):
            ir, _ = self.gen_expr_ir(stmt.expr)
            return ir
        fallback = self._ir_assign_composite_fallback(stmt)
        if fallback is not None:
            return fallback
        raise IRError(
            f"No real-IR case for statement of type {type(stmt).__name__}: {stmt!r}"
        )

    def _ir_fresh_loop_variable(self, stmt: For) -> list:
        """Each iteration has its own loop variable: when its address outlives an iteration it lives
        on the heap, and the next iteration gets a new copy before the increment runs."""
        decl = stmt.init
        if not isinstance(decl, VarDecl):
            return []
        var_type = self._local_type(decl)
        if not self._is_heap_allocated(decl.symbol.id, var_type):
            return []
        size = type_byte_width(var_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        fresh = self.ir_program.ids.new_temp(Type.INT64)
        ir = [IRCall(dst=fresh, name='malloc', args=[IRConst(size, Type.INT64)])]
        if var_type.kind in COMPOSITE_KINDS:
            slot_addr, old = self.ir_program.ids.new_temp(Type.INT64), self.ir_program.ids.new_temp(Type.INT64)
            return ir + [
                IRLocalAddress(dst=slot_addr, slot=self._local_slot(decl)),
                IRLoad(dst=old, address=slot_addr),
                IRCopy(dst_address=fresh, src_address=old, value_type=var_type),
                IRStore(address=slot_addr, value=fresh, value_type=Type.INT64),
            ]
        box = self._local_temp(decl.symbol.id)
        value = self.ir_program.ids.new_temp(var_type)
        return ir + [
            IRLoad(dst=value, address=box),
            IRStore(address=fresh, value=value, value_type=var_type),
            IRMove(dst=box, src=fresh),
        ]

    def _ir_discarded_composite(self, expr):
        """A composite value used as a statement: evaluate it (calls, bounds checks, lookups) into a
        scratch slot and discard it."""
        value_type = type_of(expr)
        width = type_byte_width(value_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        slot = self.ir_program.ids.new_slot(width, "discarded_value", self.ir_fn)
        address = self.ir_program.ids.new_temp(Type.INT64)
        write_ir = self._ir_write_composite_value_into(address, expr, value_type)
        if write_ir is None:
            raise IRError(f"No real-IR case for a {value_type} value used as a statement: {expr!r}")
        return [IRLocalAddress(dst=address, slot=slot)] + write_ir

    def _ir_assign_composite_fallback(self, stmt):
        """Composite Assign/IndexAssign/FieldAssign shapes without a dedicated case: take the
        destination's address and write with the general composite writer. None if unsupported."""
        if isinstance(stmt, Assign):
            dst_expr, value_type = self._var_ref(stmt), self._local_type(stmt)
        elif isinstance(stmt, IndexAssign) and stmt.compound_op is None:
            dst_expr, value_type = Index(array=stmt.array, index=stmt.index), type_of(stmt.array).element_type
        elif isinstance(stmt, FieldAssign) and stmt.compound_op is None:
            dst_expr = Field(base=stmt.base, name=stmt.name)
            value_type = self._check_struct_and_field_type(stmt.base, stmt.name)
        else:
            return None
        if value_type.kind not in COMPOSITE_KINDS:
            return None
        address_fn = {
            TypeKind.ARRAY: self._ir_array_address, TypeKind.STRUCT: self._ir_struct_address,
            TypeKind.SUM: self._ir_struct_address, TypeKind.SLICE: self._ir_slice_address,
            TypeKind.DICT: self._ir_dict_address, TypeKind.STR: self._ir_str_address,
        }[value_type.kind]
        result = address_fn(dst_expr)
        if result is None:
            return None
        dst_ir, dst_address = result
        write_ir = self._ir_write_composite_value_into(dst_address, stmt.value, value_type)
        return None if write_ir is None else dst_ir + write_ir

    def _ir_box_local(self, stmt, var_type, slot) -> list:
        """malloc storage for a heap-allocated local before its first write; [] otherwise."""
        return self._ir_malloc_and_store(var_type, slot) if self._is_heap_allocated(stmt.symbol.id, var_type) else []

    def _ir_malloc_and_store(self, var_type, slot: int) -> list:
        """malloc a heap local's storage and store the pointer in its slot."""
        size = type_byte_width(var_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        ptr = self.ir_program.ids.new_temp(Type.INT64)
        slot_addr = self.ir_program.ids.new_temp(Type.INT64)
        return [
            IRCall(dst=ptr, name='malloc', args=[IRConst(size, Type.INT64)]),
            IRLocalAddress(dst=slot_addr, slot=slot),
            IRStore(address=slot_addr, value=ptr, value_type=Type.INT64),
        ]

    def _ir_finish_scalar_var_decl(self, name: str, decl_id: int, var_type, value) -> list:
        """First write of a scalar VarDecl, boxing it if heap-promoted."""
        if not self._is_heap_allocated(decl_id, var_type):
            return [IRMove(dst=self._local_temp(decl_id), src=value)]
        size = type_byte_width(var_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        ptr = self.ir_program.ids.new_temp(Type.INT64)
        return [
            IRCall(dst=ptr, name='malloc', args=[IRConst(size, Type.INT64)]),
            IRStore(address=ptr, value=value, value_type=var_type),
            IRMove(dst=self._local_temp(decl_id), src=ptr),
        ]

    def _ir_compound_assign_through_address(self, addr_value, compound_op, value_expr, scalar_type) -> list:
        """Read-modify-write through an address."""
        current = self.ir_program.ids.new_temp(scalar_type)
        load_ir = [IRLoad(dst=current, address=addr_value)]
        value_ir, value = self.gen_expr_ir(value_expr)
        result = self.ir_program.ids.new_temp(scalar_type)
        binop_ir = [IRBinOp(dst=result, op=compound_op, left=current, right=value)]
        store_ir = [IRStore(address=addr_value, value=result, value_type=scalar_type)]
        return load_ir + value_ir + binop_ir + store_ir

    def _ir_index_assign(self, stmt: IndexAssign, element_type) -> list:
        """Scalar-element IndexAssign."""
        result = self._ir_index_address(Index(array=stmt.array, index=stmt.index))
        if result is None:
            raise IRError(
                f"_ir_index_address returned None for a scalar-element IndexAssign "
                f"({stmt!r}) -- expected to always succeed for a reachable base")
        addr_ir, addr_value = result
        if stmt.compound_op is not None:
            return addr_ir + self._ir_compound_assign_through_address(addr_value, stmt.compound_op, stmt.value, element_type)
        value_ir, value = self.gen_expr_ir(stmt.value)
        return addr_ir + value_ir + [IRStore(address=addr_value, value=value, value_type=element_type)]

    def _ir_copy_into_address(self, dst_address, src_expr: Node, value_type) -> list:
        """Copy a composite value into a destination address."""
        address_of = {
            TypeKind.ARRAY: self._ir_array_address,
            TypeKind.STRUCT: self._ir_struct_address,
            TypeKind.SLICE: self._ir_slice_address,
            TypeKind.DICT: self._ir_dict_address,
            TypeKind.SUM: self._ir_struct_address,
            TypeKind.STR: self._ir_str_address,
        }[value_type.kind]
        src_ir, src_addr = address_of(src_expr)
        return src_ir + [IRCopy(dst_address=dst_address, src_address=src_addr, value_type=value_type)]

    def _ir_copy_assign(self, dst_expr: Node, src_expr: Node, value_type) -> list:
        """Whole-value copy assignment."""
        address_of = {
            TypeKind.ARRAY: self._ir_array_address,
            TypeKind.STRUCT: self._ir_struct_address,
            TypeKind.SLICE: self._ir_slice_address,
            TypeKind.DICT: self._ir_dict_address,
            TypeKind.SUM: self._ir_struct_address,
            TypeKind.STR: self._ir_str_address,
        }[value_type.kind]
        dst_ir, dst_addr = address_of(dst_expr)
        return dst_ir + self._ir_copy_into_address(dst_addr, src_expr, value_type)

    def _ir_field_assign(self, stmt: FieldAssign, field_type) -> list:
        """Scalar-field FieldAssign."""
        result = self._ir_field_address(Field(base=stmt.base, name=stmt.name))
        if result is None:
            raise IRError(
                f"_ir_field_address returned None for a scalar-typed FieldAssign "
                f"({stmt!r}) -- expected to always succeed for a reachable base")
        addr_ir, addr_value = result
        if stmt.compound_op is not None:
            return addr_ir + self._ir_compound_assign_through_address(addr_value, stmt.compound_op, stmt.value, field_type)
        value_ir, value = self.gen_expr_ir(stmt.value)
        return addr_ir + value_ir + [IRStore(address=addr_value, value=value, value_type=field_type)]

    def _ir_return(self, value_expr) -> list:
        """Bare or scalar return."""
        if value_expr is None:
            return [IRReturn(value=None)]
        ir, value = self.gen_expr_ir(value_expr)
        return ir + [IRReturn(value=value)]

    def _ir_hidden_return_ptr(self, ir_fn: IRFunction) -> tuple:
        """Load this function's hidden return pointer."""
        slot_addr = self.ir_program.ids.new_temp(Type.INT64)
        hidden_ptr = self.ir_program.ids.new_temp(Type.INT64)
        ir = [
            IRLocalAddress(dst=slot_addr, slot=ir_fn.hidden_return_ptr_slot),
            IRLoad(dst=hidden_ptr, address=slot_addr),
        ]
        return ir, hidden_ptr

    def _ir_if_head(self, stmt: If, then_label: str, else_label: str) -> list:
        """Condition and branch to then/else labels."""
        cond_ir, cond_value = self.gen_expr_ir(stmt.condition)
        return cond_ir + [
            IRBranch(cond=cond_value, true_label=then_label, false_label=else_label),
            IRLabel(then_label),
        ]

    def _ir_while_head(self, stmt: While, start_label: str, body_label: str, end_label: str) -> list:
        """Loop start, condition, branch."""
        cond_ir, cond_value = self.gen_expr_ir(stmt.condition)
        return [IRJump(start_label), IRLabel(start_label)] + cond_ir + [
            IRBranch(cond=cond_value, true_label=body_label, false_label=end_label),
            IRLabel(body_label),
        ]

    def _ir_for_head(self, stmt: For, start_label: str, body_label: str, end_label: str, ir_fn) -> list:
        """Init, then loop start, condition, branch."""
        init_ir = self.gen_statement_ir(stmt.init, ir_fn)
        cond_ir, cond_value = self.gen_expr_ir(stmt.condition)
        return init_ir + [IRJump(start_label), IRLabel(start_label)] + cond_ir + [
            IRBranch(cond=cond_value, true_label=body_label, false_label=end_label),
            IRLabel(body_label),
        ]

    def _ir_break(self) -> list:
        """Jump to the innermost loop's end."""
        if not self.loop_labels:
            raise IRError("'break' outside of a loop")
        _, end_label = self.loop_labels[-1]
        return [IRJump(end_label)]

    def _ir_continue(self) -> list:
        """Jump to the innermost loop's continue target."""
        if not self.loop_labels:
            raise IRError("'continue' outside of a loop")
        continue_label, _ = self.loop_labels[-1]
        return [IRJump(continue_label)]

