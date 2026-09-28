"""A dict value is a 24-byte {buckets_ptr, count, capacity} descriptor
-- structurally identical in shape to slice's own {ptr, len, cap}, and
treated the same way throughout codegen: a small, FIXED-size value
(never individually heap-promoted by size -- see is_heap_allocated's
own tuple, which dict is deliberately not in, same as slice), whose
own address can still be heap-promoted if it escapes past its
function, pointing at a SEPARATE, always-malloc'd (calloc'd) buckets
array it merely owns a pointer to.

The buckets array is a flat, open-addressed (linear probing) hash
table: capacity buckets, each bucket_stride = 1 (state: 0 empty, 1
occupied) + type_byte_width(key_type) + type_byte_width(value_type)
bytes, contiguous, no padding between buckets. capacity is always a
power of two, so probing can use a bitwise AND instead of a modulo --
both this file's own construction code and runtime.c's own insert
helpers agree on this invariant.

Stage 1 (this file, for now) only ever constructs a dict from a
literal, sized once, upfront, to the literal's own entry count with
generous headroom (capacity = next power of two >= max(8, entry count
* 2), keeping the load factor at construction time at or under 50%) --
there is no insert-after-construction path yet (that's indexing,
stage 2), so no growth/rehash logic exists here or in runtime.c yet
either. The actual hash/probe/insert mechanics for each entry are
delegated entirely to runtime.c's own hornet_dict_insert_scalar_key/
hornet_dict_insert_str_key -- this file's own job is just marshaling
each entry's key and value into addressable bytes and calling one of
those two, mirroring how slice growth keeps its own meaty logic in
runtime.c's hornet_slice_grow rather than emitting a hand-rolled loop
as real IR.

A scalar key or value needs a REAL address to hash/memcpy from, but
often starts out as a plain expression with no address of its own (a
bare int literal, say) -- _ir_materialize_value_into_scratch is the
shared piece that gives any expression, scalar or composite, a fresh,
addressable home via a compiler-internal scratch stack slot (ir_
program.ids.new_slot -- the identical mechanism _reserve_argument_temp
already uses for its own, differently-shaped need), independent of any
named source-level variable.
"""

from ir.errors import IRError
from ir.ir import IRBinOp, IRCall, IRConst, IRLoad, IRLocalAddress, IRStore
from ir.utils import COMPOSITE_KINDS, type_byte_width
from parser import BinaryOp, DictLiteral, Field, Index, Node, Unary, UnaryOp, Variable
from semantic import Type, TypeKind


def _next_pow2_at_least(n: int) -> int:
    """The smallest power of two >= n -- capacity's own sizing rule,
    shared by the one call site that needs it (for now)."""
    p = 1
    while p < n:
        p *= 2
    return p


class DictsMixin:
    def _ir_dict_address(self, expr: Node):
        """Mirrors _ir_slice_address's own shape -- a dict is the
        identical kind of fixed-size, address-based descriptor a
        slice is, never individually heap-promoted by SIZE (see this
        file's own module docstring). Unlike slice, though, a dict's
        own address CAN escape via `&d`, and this method IS still
        reached afterward (e.g. via a VarDecl's own destination-
        address computation) -- so, unlike _ir_slice_address's own
        fast path, this always checks _is_heap_allocated. (Bug fix: an
        earlier version copied _ir_slice_address's own fast path
        verbatim, including its skipped heap-check -- safe for slice
        only because &s never actually reaches _ir_slice_address at
        all, handled entirely within _ir_address_of instead. An
        escaping dict's own descriptor was left stack-allocated,
        printing as empty once its owning function returned.)

        No sum-type narrowing branch: dict isn't wired into the sum-
        type-variant grammar yet (a separate follow-up, once _ir_
        write_composite_value_into also learns to widen a DictLiteral,
        which it doesn't yet either). Index/Field recurse into _ir_
        index_address/_ir_field_address exactly like every other
        method in this codebase does."""
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
        if isinstance(expr, Index):
            return self._ir_index_address(expr)
        if isinstance(expr, Field):
            return self._ir_field_address(expr)
        if isinstance(expr, Unary) and expr.op == UnaryOp.DEREFERENCE:
            # `*p` (p: *dict[K]V) read as a whole DICT value -- p's
            # own value already IS the address of its own pointee's
            # bytes, the identical principle _ir_array_address/_ir_
            # struct_address's own matching case uses.
            return self.gen_expr_ir(expr.operand)
        return None

    def _ir_materialize_value_into_scratch(self, expr: Node, value_type: Type, ir_fn, label: str):
        """Gives expr's own VALUE a fresh, addressable home -- a
        compiler-internal scratch stack slot, sized to type_byte_width
        (value_type), never a named source-level variable -- and
        returns (ir, address). Used for a dict literal's own scalar
        key (which otherwise has no address to hash from at all) and
        for every entry's own value (which needs an address to memcpy
        FROM, regardless of whether it's already composite-addressable
        or not -- always materializing fresh here is simpler than
        special-casing an already-addressable value separately, and
        costs nothing extra: nothing here is large or hot enough for
        the extra copy to matter).

        Dispatches on value_type.kind: str goes through _ir_str_value
        + _ir_write_str_descriptor_into_address (the {ptr, len} pair,
        NOT the string's own content bytes -- see runtime.c's own
        hornet_dict_insert_str_key for why hashing/comparing a str KEY
        needs the content bytes specifically, a distinction that
        applies to a str used as a key, never as a value, which is
        just copied as an ordinary 16-byte descriptor like any other
        composite here); any other composite kind goes through _ir_
        write_composite_value_into; anything else (a scalar) goes
        through gen_expr_ir + an ordinary IRStore."""
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

    def _ir_write_dict_literal_into(self, dst_address, expr: DictLiteral, dict_type: Type, ir_fn) -> list:
        """Writes a dict literal's own {buckets_ptr, count, capacity}
        descriptor through dst_address: a fresh, calloc'd bucket array
        sized once upfront (see this file's own module docstring for
        the sizing rule and why no growth path is needed here), each
        entry inserted via runtime.c's own hornet_dict_insert_scalar_
        key/hornet_dict_insert_str_key in turn, with the running count
        built up as a chain of IRBinOp ADDs over each insert's own
        1-or-0 return value (see runtime.c's own docstring for why
        that return value means what it does) -- NOT simply len(expr.
        entries), since a runtime-computed key could still collide in
        VALUE with an earlier entry even when semantic.py's own
        compile-time duplicate check found nothing (that check only
        ever catches two LITERAL keys spelled identically -- see
        check_dict_literal's own docstring)."""
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
            ir.append(IRBinOp(dst=new_count, op=BinaryOp.ADD, left=count_value, right=insert_result))
            count_value = new_count

        count_addr = self.ir_program.ids.new_temp(Type.INT64)
        capacity_addr = self.ir_program.ids.new_temp(Type.INT64)
        ir.extend([
            IRStore(address=dst_address, value=buckets_addr, value_type=Type.INT64),
            IRBinOp(dst=count_addr, op=BinaryOp.ADD, left=dst_address, right=IRConst(8, Type.INT64)),
            IRStore(address=count_addr, value=count_value, value_type=Type.INT),
            IRBinOp(dst=capacity_addr, op=BinaryOp.ADD, left=dst_address, right=IRConst(16, Type.INT64)),
            IRStore(address=capacity_addr, value=IRConst(capacity, Type.INT64), value_type=Type.INT64),
        ])
        return ir
