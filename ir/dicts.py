"""Dicts: 24-byte {buckets_ptr, count, capacity} descriptor over a calloc'd open-addressed table.
Bucket = state byte (0 empty, 1 occupied, 2 tombstone) + key + value, unpadded. Capacity is a power of two.
Hashing, probing, and growth live in runtime.c.
"""

from ir.errors import IRError
from ir.ir import IRBinOp, IRBranch, IRCall, IRConst, IRJump, IRLabel, IRLoad, IRLocalAddress, IRMove, IRStaticDataAddress, IRStore
from ir.utils import COMPOSITE_KINDS, type_byte_width, type_of, for_in_binding_types
from parser import BinaryOp, Call, DictLiteral, Field, ForIn, Index, Node, Unary, UnaryOp, Variable
from semantic import Type, TypeKind


def _next_pow2_at_least(n: int) -> int:
    """Smallest power of two >= n."""
    p = 1
    while p < n:
        p *= 2
    return p


class DictsMixin:
    def _ir_dict_address(self, expr: Node):
        """Address of a dict descriptor."""
        if isinstance(expr, Variable):
            slot_type = self._local_type(expr.name)
            slot = self._local_slot(expr.name)
            addr_temp = self.ir_program.ids.new_temp(Type.INT64)
            ir = [IRLocalAddress(dst=addr_temp, slot=slot)]
            if self._is_heap_allocated(self._local_decl_id(expr.name), slot_type):
                loaded = self.ir_program.ids.new_temp(Type.INT64)
                ir.append(IRLoad(dst=loaded, address=addr_temp))
                return ir, loaded
            return ir, addr_temp
        if isinstance(expr, DictLiteral):
            # Dict literal as for-in iterable.
            return self._ir_materialize_dict_literal(expr)
        if isinstance(expr, Index):
            return self._ir_index_address(expr)
        if isinstance(expr, Field):
            return self._ir_field_address(expr)
        if isinstance(expr, Unary) and expr.op == UnaryOp.DEREFERENCE:
            # `*p` as a whole dict.
            return self.gen_expr_ir(expr.operand)
        return None

    def _ir_materialize_value_into_scratch(self, expr: Node, value_type: Type, ir_fn, label: str):
        """Store expr's value in a scratch slot; returns (ir, address)."""
        width = type_byte_width(value_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        slot = self.ir_program.ids.new_slot(width, label, ir_fn)
        addr = self.ir_program.ids.new_temp(Type.INT64)
        ir = [IRLocalAddress(dst=addr, slot=slot)]
        if value_type.kind == TypeKind.STR:
            str_ir, ptr_value, len_value = self._ir_str_value(expr)
            ir.extend(str_ir)
            ir.extend(self._ir_write_str_descriptor_into_address(addr, ptr_value, len_value))
            return ir, addr
        if value_type.kind in COMPOSITE_KINDS:
            write_ir = self._ir_write_composite_value_into(addr, expr, value_type)
            if write_ir is None:
                raise IRError(
                    f"_ir_write_composite_value_into returned None for a dict "
                    f"literal's own entry ({expr!r}) -- a named/partial struct "
                    f"literal as a dict entry's key or value isn't supported yet"
                )
            ir.extend(write_ir)
            return ir, addr
        value_ir, value = self.gen_expr_ir(expr)
        ir.extend(value_ir)
        ir.append(IRStore(address=addr, value=value, value_type=value_type))
        return ir, addr

    def _ir_materialize_dict_literal(self, expr: DictLiteral):
        """Materialize a dict literal; returns (ir, address)."""
        dict_type = type_of(expr)
        if id(expr) in self._argument_temp_slots:
            slot = self._argument_temp_slots[id(expr)]
            addr = self.ir_program.ids.new_temp(Type.INT64)
            addr_ir = [IRLocalAddress(dst=addr, slot=slot)]
        else:
            addr = self.ir_program.ids.new_temp(Type.INT64)
            size = type_byte_width(dict_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
            addr_ir = [IRCall(dst=addr, name='malloc', args=[IRConst(size, Type.INT64)])]
        write_ir = self._ir_write_dict_literal_into(addr, expr, dict_type, self.ir_fn)
        return addr_ir + write_ir, addr

    def _ir_write_dict_literal_into(self, dst_address, expr: DictLiteral, dict_type: Type, ir_fn) -> list:
        """Write a dict literal's descriptor through dst_address."""
        key_type = dict_type.key_type
        value_type = dict_type.element_type
        key_width = type_byte_width(key_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        value_width = type_byte_width(value_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        bucket_stride = 1 + key_width + value_width
        capacity = _next_pow2_at_least(max(8, len(expr.entries) * 2))

        buckets_addr = self.ir_program.ids.new_temp(Type.INT64)
        ir = [IRCall(
            dst=buckets_addr, name='calloc',
            args=[IRConst(capacity, Type.INT64), IRConst(bucket_stride, Type.INT64)],
        )]

        count_value = IRConst(0, Type.INT)
        for i, (key_expr, value_expr) in enumerate(expr.entries):
            value_ir, value_addr = self._ir_materialize_value_into_scratch(
                value_expr, value_type, ir_fn, f"dict_literal_value_{i}")
            ir.extend(value_ir)
            insert_result = self.ir_program.ids.new_temp(Type.INT)
            if key_type.kind == TypeKind.STR:
                key_ir, key_ptr, key_len = self._ir_str_value(key_expr)
                ir.extend(key_ir)
                ir.append(IRCall(
                    dst=insert_result, name='hornet_dict_insert_str_key',
                    args=[buckets_addr, IRConst(capacity, Type.INT64), IRConst(bucket_stride, Type.INT64),
                          key_ptr, key_len, value_addr, IRConst(value_width, Type.INT64)],
                ))
            else:
                key_ir, key_addr = self._ir_materialize_value_into_scratch(
                    key_expr, key_type, ir_fn, f"dict_literal_key_{i}")
                ir.extend(key_ir)
                ir.append(IRCall(
                    dst=insert_result, name='hornet_dict_insert_scalar_key',
                    args=[buckets_addr, IRConst(capacity, Type.INT64), IRConst(bucket_stride, Type.INT64),
                          key_addr, IRConst(key_width, Type.INT64), value_addr, IRConst(value_width, Type.INT64)],
                ))
            new_count = self.ir_program.ids.new_temp(Type.INT)
            # insert_result is 1 for a new key, 0 for an overwrite.
            ir.append(IRBinOp(dst=new_count, op=BinaryOp.ADD, left=count_value, right=insert_result))
            count_value = new_count

        count_addr = self.ir_program.ids.new_temp(Type.INT64)
        tombstones_addr = self.ir_program.ids.new_temp(Type.INT64)
        capacity_addr = self.ir_program.ids.new_temp(Type.INT64)
        ir.extend([
            IRStore(address=dst_address, value=buckets_addr, value_type=Type.INT64),
            IRBinOp(dst=count_addr, op=BinaryOp.ADD, left=dst_address, right=IRConst(8, Type.INT64)),
            IRStore(address=count_addr, value=count_value, value_type=Type.INT),
            # tombstone count at offset 12
            IRBinOp(dst=tombstones_addr, op=BinaryOp.ADD, left=dst_address, right=IRConst(12, Type.INT64)),
            IRStore(address=tombstones_addr, value=IRConst(0, Type.INT), value_type=Type.INT),
            IRBinOp(dst=capacity_addr, op=BinaryOp.ADD, left=dst_address, right=IRConst(16, Type.INT64)),
            IRStore(address=capacity_addr, value=IRConst(capacity, Type.INT64), value_type=Type.INT64),
        ])
        return ir

    def _ir_dict_lookup(self, dict_expr: Node, key_expr: Node, dict_type: Type):
        """`d[k]` read: address of the value (panics if missing)."""
        key_type = dict_type.key_type
        value_type = dict_type.element_type
        value_width = type_byte_width(value_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        result = self._ir_dict_address(dict_expr)
        if result is None:
            raise IRError(
                f"_ir_dict_address returned None for a dict lookup's own base "
                f"({dict_expr!r}) -- expected to always succeed for a reachable base"
            )
        dict_ir, descriptor_addr = result
        result_addr = self.ir_program.ids.new_temp(Type.INT64)
        if key_type.kind == TypeKind.STR:
            key_ir, key_ptr, key_len = self._ir_str_value(key_expr)
            call_ir = [IRCall(
                dst=result_addr, name='hornet_dict_lookup_str_key',
                args=[descriptor_addr, IRConst(value_width, Type.INT64), key_ptr, key_len],
            )]
            return dict_ir + key_ir + call_ir, result_addr
        key_width = type_byte_width(key_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        key_value_ir, key_value = self.gen_expr_ir(key_expr)
        scratch_addr = self.ir_program.ids.new_temp(Type.INT64)
        key_ir = key_value_ir + [
            IRLocalAddress(dst=scratch_addr, slot=self._dict_key_scratch_slot),
            IRStore(address=scratch_addr, value=key_value, value_type=key_type),
        ]
        call_ir = [IRCall(
            dst=result_addr, name='hornet_dict_lookup_scalar_key',
            args=[descriptor_addr, IRConst(key_width, Type.INT64), IRConst(value_width, Type.INT64), scratch_addr],
        )]
        return dict_ir + key_ir + call_ir, result_addr

    def _ir_dict_contains(self, key_expr: Node, dict_expr: Node, dict_type: Type):
        """`k in d`."""
        key_type = dict_type.key_type
        value_width = type_byte_width(
            dict_type.element_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        result = self._ir_dict_address(dict_expr)
        if result is None:
            raise IRError(
                f"_ir_dict_address returned None for 'in's own dict operand "
                f"({dict_expr!r}) -- expected to always succeed for a reachable base"
            )
        dict_ir, descriptor_addr = result
        raw_result = self.ir_program.ids.new_temp(Type.INT)
        if key_type.kind == TypeKind.STR:
            key_ir, key_ptr, key_len = self._ir_str_value(key_expr)
            call_ir = [IRCall(
                dst=raw_result, name='hornet_dict_contains_str_key',
                args=[descriptor_addr, IRConst(value_width, Type.INT64), key_ptr, key_len],
            )]
        else:
            key_width = type_byte_width(key_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
            key_value_ir, key_value = self.gen_expr_ir(key_expr)
            scratch_addr = self.ir_program.ids.new_temp(Type.INT64)
            key_ir = key_value_ir + [
                IRLocalAddress(dst=scratch_addr, slot=self._dict_key_scratch_slot),
                IRStore(address=scratch_addr, value=key_value, value_type=key_type),
            ]
            call_ir = [IRCall(
                dst=raw_result, name='hornet_dict_contains_scalar_key',
                args=[descriptor_addr, IRConst(key_width, Type.INT64), IRConst(value_width, Type.INT64), scratch_addr],
            )]
        bool_result = self.ir_program.ids.new_temp(Type.BOOL)
        convert_ir = [IRBinOp(dst=bool_result, op=BinaryOp.NOT_EQUAL, left=raw_result, right=IRConst(0, Type.INT))]
        return dict_ir + key_ir + call_ir + convert_ir, bool_result

    def _ir_dict_set(self, dict_expr: Node, key_expr: Node, value_expr: Node, dict_type: Type, ir_fn) -> list:
        """`d[k] = v`."""
        key_type = dict_type.key_type
        value_type = dict_type.element_type
        value_width = type_byte_width(value_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        result = self._ir_dict_address(dict_expr)
        if result is None:
            raise IRError(
                f"_ir_dict_address returned None for a dict assignment's own base "
                f"({dict_expr!r}) -- expected to always succeed for a reachable base"
            )
        dict_ir, descriptor_addr = result
        value_ir, value_addr = self._ir_materialize_value_into_scratch(value_expr, value_type, ir_fn, "dict_set_value")
        if key_type.kind == TypeKind.STR:
            key_ir, key_ptr, key_len = self._ir_str_value(key_expr)
            call_ir = [IRCall(
                dst=None, name='hornet_dict_set_str_key',
                args=[descriptor_addr, IRConst(value_width, Type.INT64), key_ptr, key_len, value_addr],
            )]
            return dict_ir + value_ir + key_ir + call_ir
        key_width = type_byte_width(key_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        key_ir, key_addr = self._ir_materialize_value_into_scratch(key_expr, key_type, ir_fn, "dict_set_key")
        call_ir = [IRCall(
            dst=None, name='hornet_dict_set_scalar_key',
            args=[descriptor_addr, IRConst(key_width, Type.INT64), IRConst(value_width, Type.INT64),
                  key_addr, value_addr],
        )]
        return dict_ir + value_ir + key_ir + call_ir

    def _ir_dict_compound_assign(self, dict_expr: Node, key_expr: Node, compound_op, value_expr: Node,
                                  dict_type: Type, ir_fn) -> list:
        """`d[k] OP= v`."""
        key_type = dict_type.key_type
        value_type = dict_type.element_type
        value_width = type_byte_width(value_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)

        lookup_ir, value_addr = self._ir_dict_lookup(dict_expr, key_expr, dict_type)
        current = self.ir_program.ids.new_temp(value_type)
        load_ir = [IRLoad(dst=current, address=value_addr)]
        rhs_ir, rhs_value = self.gen_expr_ir(value_expr)
        combined = self.ir_program.ids.new_temp(value_type)
        binop_ir = [IRBinOp(dst=combined, op=compound_op, left=current, right=rhs_value)]

        # materialize combined so _ir_dict_set can read it
        result_slot = self.ir_program.ids.new_slot(value_width, "dict_compound_result", ir_fn)
        result_addr = self.ir_program.ids.new_temp(Type.INT64)
        store_result_ir = [
            IRLocalAddress(dst=result_addr, slot=result_slot),
            IRStore(address=result_addr, value=combined, value_type=value_type),
        ]

        addr_result = self._ir_dict_address(dict_expr)
        if addr_result is None:
            raise IRError(
                f"_ir_dict_address returned None for a dict compound assignment's "
                f"own base ({dict_expr!r}) -- expected to always succeed for a "
                f"reachable base"
            )
        dict_ir, descriptor_addr = addr_result
        if key_type.kind == TypeKind.STR:
            key_ir, key_ptr, key_len = self._ir_str_value(key_expr)
            set_ir = [IRCall(
                dst=None, name='hornet_dict_set_str_key',
                args=[descriptor_addr, IRConst(value_width, Type.INT64), key_ptr, key_len, result_addr],
            )]
        else:
            key_width = type_byte_width(key_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
            key_ir, key_addr = self._ir_materialize_value_into_scratch(
                key_expr, key_type, ir_fn, "dict_compound_key")
            set_ir = [IRCall(
                dst=None, name='hornet_dict_set_scalar_key',
                args=[descriptor_addr, IRConst(key_width, Type.INT64), IRConst(value_width, Type.INT64),
                      key_addr, result_addr],
            )]
        return lookup_ir + load_ir + rhs_ir + binop_ir + store_result_ir + dict_ir + key_ir + set_ir

    def _ir_del_call(self, expr: Call) -> tuple:
        """`del(d, k)`."""
        dict_expr, key_expr = expr.args
        dict_type = type_of(dict_expr)
        key_type = dict_type.key_type
        value_width = type_byte_width(
            dict_type.element_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        result = self._ir_dict_address(dict_expr)
        if result is None:
            raise IRError(
                f"_ir_dict_address returned None for del()'s own dict argument "
                f"({dict_expr!r}) -- expected to always succeed for a reachable base"
            )
        dict_ir, descriptor_addr = result
        if key_type.kind == TypeKind.STR:
            key_ir, key_ptr, key_len = self._ir_str_value(key_expr)
            call_ir = [IRCall(
                dst=None, name='hornet_dict_delete_str_key',
                args=[descriptor_addr, IRConst(value_width, Type.INT64), key_ptr, key_len],
            )]
            return dict_ir + key_ir + call_ir, None
        key_width = type_byte_width(key_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        key_value_ir, key_value = self.gen_expr_ir(key_expr)
        scratch_addr = self.ir_program.ids.new_temp(Type.INT64)
        key_ir = key_value_ir + [
            IRLocalAddress(dst=scratch_addr, slot=self._dict_key_scratch_slot),
            IRStore(address=scratch_addr, value=key_value, value_type=key_type),
        ]
        call_ir = [IRCall(
            dst=None, name='hornet_dict_delete_scalar_key',
            args=[descriptor_addr, IRConst(key_width, Type.INT64), IRConst(value_width, Type.INT64), scratch_addr],
        )]
        return dict_ir + key_ir + call_ir, None

    def _ir_dict_none_comparison(self, expr):
        """`d == none` / `!= none`."""
        dict_expr = expr.left if type_of(expr.left).kind == TypeKind.DICT else expr.right
        result = self._ir_dict_address(dict_expr)
        if result is None:
            return None
        addr_ir, addr = result
        ptr_value = self.ir_program.ids.new_temp(Type.INT64)
        load_ir = [IRLoad(dst=ptr_value, address=addr)]
        t_result = self.ir_program.ids.new_temp(Type.BOOL)
        check = IRBinOp(dst=t_result, op=expr.op, left=ptr_value, right=IRConst(0, Type.INT64))
        return addr_ir + load_ir + [check], t_result

    def _ir_for_in_dict(self, stmt: ForIn, ir_fn) -> list:
        """`for k[, v] in d`: scan every bucket."""
        dict_type = type_of(stmt.iterable)
        key_type = dict_type.key_type
        value_type = dict_type.element_type
        key_width = type_byte_width(key_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        value_width = type_byte_width(value_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        bucket_stride = 1 + key_width + value_width
        needs_recheck = not isinstance(stmt.iterable, DictLiteral)

        result = self._ir_dict_address(stmt.iterable)
        if result is None:
            raise IRError(
                f"_ir_dict_address returned None for a DICT-typed 'for ... "
                f"in' iterable ({stmt.iterable!r}) -- expected to always "
                f"succeed for a reachable indexable base")
        dict_ir, descriptor_addr = result

        buckets_ptr = self.ir_program.ids.new_temp(Type.INT64)
        capacity = self.ir_program.ids.new_temp(Type.INT64)
        capacity_addr = self.ir_program.ids.new_temp(Type.INT64)
        setup_ir = dict_ir + [
            IRLoad(dst=buckets_ptr, address=descriptor_addr),
            IRBinOp(dst=capacity_addr, op=BinaryOp.ADD, left=descriptor_addr, right=IRConst(16, Type.INT64)),
            IRLoad(dst=capacity, address=capacity_addr),
        ]

        i = self.ir_program.ids.new_temp(Type.INT64)
        start_label = self.ir_program.ids.new_label("for_in_start")
        body_label = self.ir_program.ids.new_label("for_in_body")
        entry_label = self.ir_program.ids.new_label("for_in_entry")
        continue_label = self.ir_program.ids.new_label("for_in_continue")
        end_label = self.ir_program.ids.new_label("for_in_end")

        self._push_scope()
        binding_types = for_in_binding_types(stmt, dict_type)
        for idx, binding_type in enumerate(binding_types):
            self._bind_for_in_binding(stmt, idx, binding_type, ir_fn)

        ir = setup_ir + [
            IRMove(dst=i, src=IRConst(0, Type.INT64)),
            IRJump(start_label),
            IRLabel(start_label),
        ]
        cond = self.ir_program.ids.new_temp(Type.BOOL)
        ir.append(IRBinOp(dst=cond, op=BinaryOp.LESS_THAN, left=i, right=capacity))
        ir.append(IRBranch(cond=cond, true_label=body_label, false_label=end_label))
        ir.append(IRLabel(body_label))

        # Recheck buckets_ptr before use: panic if the dict was rehashed.
        if needs_recheck:
            recheck_ptr = self.ir_program.ids.new_temp(Type.INT64)
            ir.append(IRLoad(dst=recheck_ptr, address=descriptor_addr))
            mutated = self.ir_program.ids.new_temp(Type.BOOL)
            ir.append(IRBinOp(dst=mutated, op=BinaryOp.NOT_EQUAL, left=recheck_ptr, right=buckets_ptr))
            mutated_label = self.ir_program.ids.new_label("for_in_mutated")
            safe_label = self.ir_program.ids.new_label("for_in_safe")
            ir.append(IRBranch(cond=mutated, true_label=mutated_label, false_label=safe_label))
            ir.append(IRLabel(mutated_label))
            msg_ptr = self.ir_program.ids.new_temp(Type.INT64)
            msg_label = self.ir_program.ids.new_label("for_in_mutated_msg")
            self.ir_program.string_literals.append(
                (msg_label, "for ... in: dict's own buckets were reallocated (e.g. by an "
                            "insert that triggered growth) during iteration"))
            ir.append(IRStaticDataAddress(dst=msg_ptr, label=msg_label))
            ir.append(IRCall(dst=None, name='hornet_panic', args=[msg_ptr]))
            ir.append(IRJump(safe_label))  # unreachable; the verifier requires a terminator
            ir.append(IRLabel(safe_label))

        offset_temp = self.ir_program.ids.new_temp(Type.INT64)
        ir.append(IRBinOp(dst=offset_temp, op=BinaryOp.MULTIPLY, left=i, right=IRConst(bucket_stride, Type.INT64)))
        bucket_addr = self.ir_program.ids.new_temp(Type.INT64)
        ir.append(IRBinOp(dst=bucket_addr, op=BinaryOp.ADD, left=buckets_ptr, right=offset_temp))

        state = self.ir_program.ids.new_temp(Type.UINT8)
        ir.append(IRLoad(dst=state, address=bucket_addr))
        is_occupied = self.ir_program.ids.new_temp(Type.BOOL)
        ir.append(IRBinOp(dst=is_occupied, op=BinaryOp.EQUAL, left=state, right=IRConst(1, Type.UINT8)))  # HORNET_DICT_BUCKET_OCCUPIED
        ir.append(IRBranch(cond=is_occupied, true_label=entry_label, false_label=continue_label))
        ir.append(IRLabel(entry_label))

        key_addr = self.ir_program.ids.new_temp(Type.INT64)
        ir.append(IRBinOp(dst=key_addr, op=BinaryOp.ADD, left=bucket_addr, right=IRConst(1, Type.INT64)))
        ir.extend(self._ir_bind_for_in_value(stmt, 0, key_type, key_addr, ir_fn))
        if len(binding_types) == 2:
            value_addr = self.ir_program.ids.new_temp(Type.INT64)
            ir.append(IRBinOp(dst=value_addr, op=BinaryOp.ADD, left=key_addr, right=IRConst(key_width, Type.INT64)))
            ir.extend(self._ir_bind_for_in_value(stmt, 1, value_type, value_addr, ir_fn))

        self.loop_labels.append((continue_label, end_label))
        for s in stmt.body:
            ir.extend(self.gen_statement_ir(s, ir_fn))
        self.loop_labels.pop()

        ir.append(IRJump(continue_label))
        ir.append(IRLabel(continue_label))
        next_i = self.ir_program.ids.new_temp(Type.INT64)
        ir.append(IRBinOp(dst=next_i, op=BinaryOp.ADD, left=i, right=IRConst(1, Type.INT64)))
        ir.append(IRMove(dst=i, src=next_i))
        ir.append(IRJump(start_label))
        ir.append(IRLabel(end_label))
        self._pop_scope()
        return ir
