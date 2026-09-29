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
from ir.ir import IRBinOp, IRBranch, IRCall, IRConst, IRJump, IRLabel, IRLoad, IRLocalAddress, IRMove, IRStaticDataAddress, IRStore
from ir.utils import COMPOSITE_KINDS, type_byte_width, type_of, for_in_binding_types
from parser import BinaryOp, Call, DictLiteral, Field, ForIn, Index, Node, Unary, UnaryOp, Variable
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

    def _ir_materialize_dict_literal(self, expr: DictLiteral):
        """Builds (without lowering) a dict-literal expression's own
        materialized address as real IR -- returns (ir, address).
        The dict-literal counterpart to _ir_materialize_struct_literal
        (ir/structs.py), sharing its same skeleton (a reserved slot,
        or malloc when none was reserved) -- unlike that one, this
        never returns None: _ir_write_dict_literal_into itself never
        does either (see its own docstring), so there's no out-of-
        scope case here to propagate.

        No value_type ambiguity here, unlike an ArrayLiteral: a
        DictLiteral always carries its own key_type/value_type
        explicitly (see DictLiteral's own docstring in parser.py), so
        type_of(expr) is simply, always correct. ir_fn (needed by
        _ir_write_dict_literal_into to reserve each entry's own key/
        value scratch slots) comes from self.ir_fn -- see gen_
        function_ir's own docstring for why that's how every caller
        of _ir_write_dict_literal_into gets it now, not a parameter
        threaded through this method's own signature."""
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
            # insert_result is 0 (overwrote an existing key -- a
            # runtime-computed key CAN still collide in value even
            # when semantic.py's own compile-time duplicate check
            # found nothing, see check_dict_literal's own docstring),
            # 1 (a fresh entry into an EMPTY slot), or -- runtime.c's
            # own hornet_dict_insert_*_key, tombstone-aware since del
            # exists now -- 2 (a fresh entry reusing a TOMBSTONE's own
            # slot instead). Adding it directly to count_value like
            # this is only correct because this bucket array is
            # FRESHLY calloc'd, never having held a tombstone at all
            # -- insert_result can therefore never actually BE 2 here,
            # only 0 or 1, both of which already mean exactly "how
            # much to add to count" on their own.
            ir.append(IRBinOp(dst=new_count, op=BinaryOp.ADD, left=count_value, right=insert_result))
            count_value = new_count

        count_addr = self.ir_program.ids.new_temp(Type.INT64)
        tombstones_addr = self.ir_program.ids.new_temp(Type.INT64)
        capacity_addr = self.ir_program.ids.new_temp(Type.INT64)
        ir.extend([
            IRStore(address=dst_address, value=buckets_addr, value_type=Type.INT64),
            IRBinOp(dst=count_addr, op=BinaryOp.ADD, left=dst_address, right=IRConst(8, Type.INT64)),
            IRStore(address=count_addr, value=count_value, value_type=Type.INT),
            # tombstones (offset 12, the 4 bytes of padding between
            # count and capacity -- see runtime.c's own dict_
            # tombstones docstring) is always 0 for a freshly-
            # constructed literal (nothing has ever been deleted from
            # it yet), but still needs writing explicitly: this
            # address is otherwise whatever garbage was already on
            # the stack, not zeroed for free the way a fresh calloc's
            # own memory is.
            IRBinOp(dst=tombstones_addr, op=BinaryOp.ADD, left=dst_address, right=IRConst(12, Type.INT64)),
            IRStore(address=tombstones_addr, value=IRConst(0, Type.INT), value_type=Type.INT),
            IRBinOp(dst=capacity_addr, op=BinaryOp.ADD, left=dst_address, right=IRConst(16, Type.INT64)),
            IRStore(address=capacity_addr, value=IRConst(capacity, Type.INT64), value_type=Type.INT64),
        ])
        return ir

    def _ir_dict_lookup(self, dict_expr: Node, key_expr: Node, dict_type: Type):
        """`dict_expr[key_expr]`, as a READ -- builds (without
        lowering) the address of the matching entry's own VALUE, via
        runtime.c's own hornet_dict_lookup_scalar_key/hornet_dict_
        lookup_str_key, which panic outright (never returning) if the
        key isn't present, per this feature's own confirmed design.
        Returns (ir, address) -- _ir_index_address, the sole caller,
        either loads a scalar from this address (an ordinary Index
        read) or hands the address on unchanged for a COMPOSITE-
        valued entry (a dict of structs, say), exactly the same split
        every other composite-address method's own Index delegation
        already has no special awareness of at all.

        A scalar key needs its own value materialized into a small,
        SHARED scratch slot first (_dict_key_scratch_slot, reserved
        once per function -- see ir/builder.py's own reservation
        comment for why a shared slot is safe here specifically,
        unlike a dict WRITE's own per-call one) so it has an address
        to hash/compare from at all; a str key instead uses its own
        {ptr, len} CONTENT directly, no scratch slot needed (see
        runtime.c's own hornet_hash_bytes docstring for why)."""
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
        """`key_expr in dict_expr` -- builds (without lowering) a
        BOOL-typed Temp: whether key_expr's own value is a LIVE (not
        tombstoned) entry in dict_expr. Mirrors _ir_dict_lookup's own
        key handling exactly, but calls runtime.c's own hornet_dict_
        contains_scalar_key/hornet_dict_contains_str_key instead of a
        lookup one -- these never panic on a miss (absence is the
        ordinary, expected FALSE result of a membership test, not an
        error) and return an int (0 or 1) rather than an address.

        That raw C int is converted to a real Hornet BOOL via an
        ordinary IRBinOp NOT_EQUAL against 0, the same two-step shape
        ir/strings.py's own string-equality lowering already uses for
        a C call's own int-valued result -- keeping this consistent
        with every other C-call-backed boolean result here, rather
        than being the one place that skips the conversion step."""
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
        """`dict_expr[key_expr] = value_expr` -- via runtime.c's own
        hornet_dict_set_scalar_key/hornet_dict_set_str_key, which grow
        (double capacity, rehash every existing entry) the dict's own
        backing bucket array first if inserting one more would exceed
        a 75% load factor, then insert-or-overwrite via the identical
        probe logic this file's own literal-construction path already
        uses. Unlike _ir_dict_lookup's own read side, ir_fn IS
        available here (gen_statement_ir's own IndexAssign case, the
        sole caller, already has it) -- so both key and value are
        materialized via _ir_materialize_value_into_scratch's own
        fresh-slot-per-call, rather than needing a shared one the way
        the read side's own scalar key does."""
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
        """`dict_expr[key_expr] OP= value_expr` -- unlike _ir_compound_
        assign_through_address's own shared array/field/deref version,
        a dict's own read and write sides aren't the same address
        reused twice: writing back goes through _ir_dict_set's own
        full grow-then-probe machinery again, not a bare IRStore. So
        key_expr is evaluated ONCE, up front for the lookup, and its
        VALUE (not the expression itself) reused for the write --
        re-evaluating it a second time would be a real bug for a key
        expression with a side effect, not just wasted work.

        Reading first via _ir_dict_lookup also means a compound
        assignment on a missing key panics, like a bare read already
        does. The write back can only ever be overwriting that SAME,
        already-confirmed-present key, so hornet_dict_set_*_key's own
        growth check can never actually find a reason to grow here."""
        key_type = dict_type.key_type
        value_type = dict_type.element_type
        value_width = type_byte_width(value_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)

        lookup_ir, value_addr = self._ir_dict_lookup(dict_expr, key_expr, dict_type)
        current = self.ir_program.ids.new_temp(value_type)
        load_ir = [IRLoad(dst=current, address=value_addr)]
        rhs_ir, rhs_value = self.gen_expr_ir(value_expr)
        combined = self.ir_program.ids.new_temp(value_type)
        binop_ir = [IRBinOp(dst=combined, op=compound_op, left=current, right=rhs_value)]

        # combined is already a computed IRValue, not an expression
        # with its own AST node _ir_dict_set could re-evaluate -- so
        # it's materialized into its own scratch slot directly here,
        # then hornet_dict_set_scalar_key/hornet_dict_set_str_key
        # called directly (bypassing _ir_dict_set itself, which would
        # otherwise re-evaluate key_expr a second time -- see this
        # method's own docstring for why that's wrong, not just
        # wasteful).
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
        """`del(d, key)` -- mirrors _ir_dict_lookup's own address/key
        handling exactly (a scalar key materialized into the same
        shared scratch slot; a str key's own {ptr, len} content used
        directly), but calls runtime.c's own hornet_dict_delete_
        scalar_key/hornet_dict_delete_str_key instead of a lookup one
        -- void, mutating d's own bucket array in place (marking the
        matching bucket a tombstone) rather than returning an address,
        and panicking on a missing key for the identical reason a
        lookup already does (see check_del_call's own docstring in
        semantic.py for the confirmed design this matches).

        Returns (ir, None) -- del is Type.VOID, gen_expr_ir's own sole
        caller (its 'del' dispatch, mirroring 'print's) never reads
        the second half of this tuple, but every gen_expr_ir case
        still returns a pair, so this does too rather than being a
        special exception."""
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
        """Builds (without lowering) `dict_expr == none` or `dict_expr
        != none` (in either operand order) as real IR -- returns (ir,
        value), or None when dict_expr's own base is out of scope.
        Mirrors _ir_slice_none_comparison's own shape (ir/arrays_
        slices.py) exactly, just reading buckets_ptr (offset 0 of the
        24-byte descriptor -- see runtime.c's own dict_tombstones
        docstring for the full layout) through _ir_dict_address rather
        than a slice's own {ptr, len, cap} triple _ir_indexable_base
        already unpacks for free: a nil dict (ir/statements.py's own
        VarDecl-with-no-initializer case) and only a nil dict has
        buckets_ptr == 0, the identical "check ptr specifically"
        reasoning slice's own comparison already uses."""
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
        """Builds (without lowering) `for k in d:` / `for k, v in d:`
        as real IR -- a bounded loop over EVERY bucket slot (0..
        capacity, not 0..count: an occupied one can sit anywhere among
        them), skipping empty/tombstone ones entirely (no binding, no
        body run for those) -- mirrors runtime.c's own hornet_
        stringify dict case structurally (the one other place that
        already walks every bucket this same way, to print a dict's
        own contents), rebuilt here in IR since the loop body has to
        run as compiled Hornet statements, not a C callback.

        bucket_stride/key/value offsets are exactly runtime.c's own
        (see this file's own module docstring): a bucket is 1 (state)
        + key_width + value_width bytes, key at buckets + i *
        bucket_stride + 1, value right after it at that address +
        key_width. A str-typed key needs no special handling AT ALL
        here despite living inline in its own bucket, rather than
        behind a separately-heap-allocated descriptor the way an
        ordinary str variable's OWN storage usually is: hornet_dict_
        insert_str_key already writes {ptr, len} there in exactly str's
        own ordinary descriptor layout (key_region_width == 16 ==
        type_byte_width(str) exactly), so _ir_bind_for_in_value's own
        STR case (an ordinary composite IRCopy) already handles it
        correctly, unchanged, the address it's given already being a
        real, valid str descriptor's own address, byte for byte.

        continue_label serves BOTH an empty/tombstone bucket's own
        "skip this one, no binding, no body" jump AND an explicit
        `continue` from inside the body -- both need the identical
        "advance i, re-check the loop condition" behavior, so sharing
        one target is correct, not incidental.

        Mutation safety: buckets_ptr (offset 0 of the 24-byte
        descriptor) is cached at loop start and re-read every
        iteration, panicking via hornet_panic (the same runtime
        function bounds-check failures, and _ir_for_in_array_slice's
        own identical SLICE check, already use) if it's changed --
        exactly the one operation that's actually memory-unsafe here:
        an insert that crosses the growth threshold reallocates the
        WHOLE buckets array, orphaning this iterator's own cached
        base address and bucket_stride math. Ordinary insert-without-
        growth, in-place value overwrite, and delete (tombstoning) are
        all still safe and unchecked -- none of them touch buckets_
        ptr at all."""
        dict_type = type_of(stmt.iterable)
        key_type = dict_type.key_type
        value_type = dict_type.element_type
        key_width = type_byte_width(key_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        value_width = type_byte_width(value_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
        bucket_stride = 1 + key_width + value_width

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

        # Mutation-safety recheck FIRST, before buckets_ptr is used for
        # anything -- computing bucket_addr from a stale buckets_ptr is
        # already the unsafe operation, regardless of what this
        # particular iteration's own bucket state byte turns out to be.
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
        ir.append(IRJump(safe_label))  # unreachable -- hornet_panic never
        # returns -- but the verifier requires every block to end in an
        # explicit terminator, the same reasoning _ir_for_in_array_
        # slice's own identical SLICE-case jump already has.
        ir.append(IRLabel(safe_label))

        offset_temp = self.ir_program.ids.new_temp(Type.INT64)
        ir.append(IRBinOp(dst=offset_temp, op=BinaryOp.MULTIPLY, left=i, right=IRConst(bucket_stride, Type.INT64)))
        bucket_addr = self.ir_program.ids.new_temp(Type.INT64)
        ir.append(IRBinOp(dst=bucket_addr, op=BinaryOp.ADD, left=buckets_ptr, right=offset_temp))

        state = self.ir_program.ids.new_temp(Type.UINT8)
        ir.append(IRLoad(dst=state, address=bucket_addr))
        is_occupied = self.ir_program.ids.new_temp(Type.BOOL)
        ir.append(IRBinOp(dst=is_occupied, op=BinaryOp.EQUAL, left=state, right=IRConst(1, Type.UINT8)))  # HORNET_DICT_BUCKET_OCCUPIED == 1
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
