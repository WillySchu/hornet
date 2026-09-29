"""Arrays are fixed-size, stack-allocated (unless promoted) value
types: `b = a` copies every element into independent storage, and
every index is bounds-checked at runtime. Slices are 24-byte {ptr,
len, cap} descriptors that alias an array's backing storage rather
than copying it -- cap exists to let `append` grow into existing
spare room instead of always reallocating. Heap promotion combines a
pure size threshold with real escape analysis (see
escape_analysis.py): a small array whose slice might outlive the
function it's declared in still needs heap-backed storage regardless
of size."""

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
    IRSliceGrow,
    IRStaticDataAddress,
    IRStore, Temp,
)
from ir.utils import COMPOSITE_KINDS, SUM_TYPE_TAG_WIDTH, is_composite_addressable, type_of, type_byte_width, for_in_binding_types
from parser import Node, ArrayLiteral, Call, DictLiteral, Field, ForIn, Index, Slice, Variable, NoneLiteral, Binary, BinaryOp, Unary, UnaryOp
from semantic import TypeKind, Type


class ArraysSlicesMixin:
    def _ir_array_address(self, expr: Node):
        """The address of an array-typed expr -- Variable is the one
        genuine leaf (a fixed %rbp-relative offset via IRLocalAddress
        directly, or, if heap-allocated, the pointer STORED at that
        offset via IRLocalAddress plus an IRLoad through it), Index/
        Field recurse into _ir_index_address/_ir_field_address.

        The Variable case also needs _ir_struct_address's own sum-type
        narrowing branch now that an array can itself be a sum type's
        own variant: without it, a narrowed array read starts
        SUM_TYPE_TAG_WIDTH bytes too early, into the discriminant tag
        itself (see _ir_struct_address's own docstring for the fuller
        reasoning).

        Returns None for an ArrayLiteral (construction, not an
        existing address -- see _ir_materialize_array_literal)."""
        if isinstance(expr, Variable):
            slot = self._local_slot(expr.name)
            array_type = self._local_type(expr.name)
            slot_addr = self.ir_program.ids.new_temp(Type.INT64)
            ir = [IRLocalAddress(dst=slot_addr, slot=slot)]
            if self._is_heap_allocated(self._local_decl_id(expr.name), array_type):
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
            # `*p` (p: *[N]int) read as a whole ARRAY value -- p's own
            # value already IS the address of its own pointee's bytes,
            # the identical principle _ir_struct_address's own matching
            # case uses (see its own docstring there for the fuller
            # explanation, including why no narrowing concern reaches
            # here either).
            return self.gen_expr_ir(expr.operand)
        return None

    def _ir_slice_address(self, expr: Node):
        """Mirrors _ir_array_address's own shape for a slice-typed
        expr, but simpler: a GENUINELY slice-typed variable is never
        heap-allocated -- a slice's own 24-byte descriptor is always a
        small, fixed-size, stack-resident value -- so that Variable
        leaf is just a single LeaQFrame, no heap-vs-stack branch
        needed. Index/Field recurse into _ir_index_address/_ir_field_
        address exactly like _ir_array_address's own do.

        A slice-NARROWED occurrence of a sum-typed variable is
        different on both counts, needing _ir_array_address's own
        fuller shape instead: the SLOT's own declared type is SUM, not
        SLICE, so it CAN be heap-allocated like any other composite;
        and its real bytes start SUM_TYPE_TAG_WIDTH into the slot's
        own address, past the discriminant tag (see _ir_struct_
        address's own docstring for the fuller narrowing reasoning).

        Returns None for a Slice (`arr[a:b]`, slice production) or a
        Call (a slice-returning function call) -- both out of scope
        for this method."""
        if isinstance(expr, Variable):
            slot_type = self._local_type(expr.name)
            if slot_type.kind != TypeKind.SUM:
                slot = self._local_slot(expr.name)
                addr_temp = self.ir_program.ids.new_temp(Type.INT64)
                return [IRLocalAddress(dst=addr_temp, slot=slot)], addr_temp
            slot = self._local_slot(expr.name)
            slot_addr = self.ir_program.ids.new_temp(Type.INT64)
            ir = [IRLocalAddress(dst=slot_addr, slot=slot)]
            if self._is_heap_allocated(self._local_decl_id(expr.name), slot_type):
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
            # `*p` (p: *[]int) read as a whole SLICE value -- p's own
            # value already IS the address of its own pointee's 24-
            # byte descriptor, the identical principle _ir_struct_
            # address's own matching case uses (see its own docstring
            # there for the fuller explanation).
            return self.gen_expr_ir(expr.operand)
        return None

    def _ir_materialize_composite_call(
            self, call_expr: Call, value_type: Type) -> tuple[Union[IRLocalAddress, IRCall], Temp]:
        """Builds (without lowering) an ordinary composite-returning
        Call's own materialized address as real IR -- returns (ir,
        address). Used wherever such a call sits directly at an
        addressable-base position (Index.array, Field.base, Slice.
        array) with no address of its own to compute.

        Whether the result lands on the stack or the heap is decided
        entirely by whether _collect_argument_temps_in_expr's own
        pre-pass reserved a slot for id(call_expr) -- see its own
        docstring for which of the three base positions get one
        (Index/Field, when small enough) and which never do (Slice,
        which always escapes regardless of size). No reservation found
        means malloc.

        Either way, the destination address is handed to _ir_
        composite_call exactly as any other composite-returning call's
        own destination would be."""
        if id(call_expr) in self._argument_temp_slots:
            slot = self._argument_temp_slots[id(call_expr)]
            addr = self.ir_program.ids.new_temp(Type.INT64)
            addr_ir = [IRLocalAddress(dst=addr, slot=slot)]
        else:
            addr = self.ir_program.ids.new_temp(Type.INT64)
            size = type_byte_width(value_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)
            addr_ir = [IRCall(dst=addr, name='malloc', args=[IRConst(size, Type.INT64)])]
        call_ir = self._ir_composite_call(addr, call_expr)
        return addr_ir + call_ir, addr

    def _ir_materialize_array_literal(self, expr: ArrayLiteral):
        """Builds (without lowering) an ArrayLiteral's own materialized
        address as real IR -- returns (ir, address), or None when some
        element is out of scope for _ir_write_array_literal_into. Used
        wherever such a literal sits directly at an addressable-base
        position (`[1, 2, 3][i]`, `[1, 2, 3][a:b]`) -- the ArrayLiteral
        counterpart to _ir_materialize_composite_call, sharing its
        same skeleton (a reserved slot, or malloc when none was
        reserved).

        A struct-literal Call at one of these same base positions
        needs no equivalent: semantic.py already rejects that outright
        wherever it would appear.

        Only ever reached from _ir_indexable_base's own ARRAY branch,
        never its SLICE branch: type_of on a bare ArrayLiteral used
        directly as a base is always ARRAY-kind, so a Slice production
        from this kind of base (`[1, 2, 3][a:b]`) still reaches this
        same ARRAY-branch materialization for its own base, with _ir_
        slice_into itself producing the actual slice descriptor one
        level up.

        No value_type parameter: type_of(expr) is already exactly the
        ARRAY-kind Type this needs, with no destination-type ambiguity
        possible at this position."""
        array_type = type_of(expr)
        if id(expr) in self._argument_temp_slots:
            slot = self._argument_temp_slots[id(expr)]
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
        """Builds (without lowering) the {ptr, len, cap} triple read out
        of an ALREADY-computed slice-descriptor address as real IR --
        returns (ir, ptr_value, len_value, cap_value). ptr is read
        directly off the descriptor's own address; len/cap live at
        fixed +8/+16 offsets from it, each computed via an ordinary
        IRBinOp before its own IRLoad, the same address-as-a-Temp
        pattern used everywhere else in this file. len/cap are captured
        as INT (32-bit) Temps -- an array/slice's own length or
        capacity always fits in 32 bits.

        The shared core of _ir_indexable_base's own SLICE-typed
        dispatch (see its own docstring) -- every one of its leaves
        (Variable, Field/Index, an ordinary composite-returning Call,
        and now a pointer dereference too) differs only in HOW it
        gets descriptor_addr in the first place, never in what happens
        once it has it. Factored out here rather than left duplicated
        three (now four) times over: exactly the kind of duplication
        that let one of those four leaves quietly miss a fix (a
        pointer dereference's own Unary case) the other three already
        had, until this method existed to make missing one impossible
        by construction."""
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
        """Builds (without lowering) the address, length, and capacity
        of an indexable base as real IR -- returns (ir, addr_value,
        length_value, cap_value), or None when expr's own shape is out
        of scope (a struct-literal Call directly at this position --
        moot in practice, semantic.py already rejects that outright).
        cap_value is always computed, even by a caller (_ir_index_
        address) that never reads it back.

        ARRAY-typed: delegates to _ir_array_address for a Variable/
        Field/Index base, _ir_materialize_composite_call for an
        ordinary composite-returning Call, or _ir_materialize_array_
        literal for a bare bracketed-list literal; length and cap are
        both the same compile-time IRConst either way -- an array has
        no separate capacity.

        SLICE-typed, Variable: ptr read straight off the descriptor's
        fixed frame address via IRLoad; len/cap via their own +8/+16
        addresses computed through ordinary IRBinOp. len/cap are
        captured as INT (32-bit) Temps -- an array/slice's own length
        or capacity always fits in 32 bits.

        SLICE-typed, Field/Index (`p.values`, `rows[i]`): same read
        shape as Variable, but the descriptor's own address is itself
        a runtime value, computed via _ir_field_address/_ir_index_
        address (already generic over the result's own type). This is
        what makes a slice reached through a chain (`outer.inner.
        values[0]`, `rows[i][j]`) fall out for free: those two methods
        already recurse through arbitrary Variable/Field/Index chains.

        SLICE-typed, ordinary composite-returning Call
        (`makeSlice()[i]`): same read shape, with the descriptor's own
        address coming from _ir_materialize_composite_call instead.
        Also reached, via _ir_slice_into's own delegation to this
        method, by a slice PRODUCED from this call's result
        (`makeArray()[a:b]`).

        SLICE-typed, append (`append(s, x)[i]`, `append(s, x) == none`,
        ...): _ir_append_call already returns exactly (ir, ptr, len,
        cap), so this delegates straight to it. Checked before the
        ordinary-composite-Call case above, since append is a builtin,
        never a compiled function.

        SLICE-typed, Slice (`arr[:][0]`, `s[a:b][c:d]`): delegates
        straight to _ir_slice_into, whose own return shape already
        matches this method's exactly. This is what makes arbitrarily
        deep chains (`arr[a:b][c:d][e]`) fall out for free: each nested
        _ir_slice_into call gets its own fresh Temps."""
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
                # `*p` (p: *[]int) read as a whole SLICE value, for a
                # function-call argument or a further index/slice base
                # -- p's own value already IS the address of its own
                # pointee's 24-byte descriptor, the identical principle
                # _ir_slice_address's own matching case already uses
                # (see its own docstring there) for every OTHER slice-
                # typed position -- this was the one leaf that hadn't
                # caught up to it yet.
                addr_ir, descriptor_addr = self.gen_expr_ir(expr.operand)
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

    def _ir_nil_slice(self):
        """Builds (without lowering) a nil slice's own {ptr, len, cap}
        triple as real IR -- all-zero, matching this compiler's own
        nil-slice representation (ptr=0, never dereferenced since
        len=0 always gets checked first). Returns (ir, ptr_value,
        len_value, cap_value), the same shape _ir_indexable_base/_ir_
        append_call/_ir_slice_into return.

        Shared by every VarDecl/Assign/IndexAssign/FieldAssign/Return
        case initializing a slice-typed destination to a bare `none`,
        and by _ir_slice_arg's own NoneLiteral case."""
        ptr = self.ir_program.ids.new_temp(Type.INT64)
        length = self.ir_program.ids.new_temp(Type.INT)
        cap = self.ir_program.ids.new_temp(Type.INT)
        ir = [
            IRMove(dst=ptr, src=IRConst(0, Type.INT64)),
            IRMove(dst=length, src=IRConst(0, Type.INT)),
            IRMove(dst=cap, src=IRConst(0, Type.INT)),
        ]
        return ir, ptr, length, cap

    def _ir_slice_arg(self, expr: Node):
        """Builds (without lowering) a slice-typed function-call argument's own
        {ptr, len, cap} triple as real IR, returns (ir, ptr_value, len_value,
        cap_value).

        NoneLiteral (`none` passed directly as an argument) is its own leaf
        here, _ir_nil_slice's own triple."""
        if isinstance(expr, NoneLiteral):
            return self._ir_nil_slice()

        return self._ir_indexable_base(expr)

    def _ir_index_address(self, expr: Index):
        """Builds (without lowering) the address of expr.array[expr.
        index] as real IR -- the shared foundation for a scalar
        element read/write and a sub-array (this method's own
        recursive base case, via _ir_array_address, when expr.array is
        itself an Index, so `matrix[i][j]` falls out with no special-
        casing for depth).

        Returns None when expr.array's own base is out of scope for
        real IR -- see _ir_indexable_base.

        A dict-typed base (`ages['bob']`) is checked FIRST, before any
        of this -- delegated straight to _ir_dict_lookup, whose own
        hash-and-probe mechanics have nothing in common with an array/
        slice's own fixed-stride arithmetic below. Checking here,
        rather than only in gen_expr_ir's own dispatch, is what lets a
        COMPOSITE-valued dict entry (a dict of structs, say) work too:
        every composite-address method already delegates an Index base
        to this one method, so this is the single choke point a dict
        lookup's own result needs to reach through.

        The bounds check aside (IRBoundsCheck), the actual arithmetic
        is ordinary IRBinOp: multiply the index by the element's own
        stride, add to the base. The bounds check guarantees a small,
        non-negative index, so a 32-bit multiply is safe, and its own
        write zero-extends into the full 64-bit register the
        following ADD reads."""
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
        """Builds (without lowering) expr.array[expr.low:expr.high]'s
        resulting {ptr, len, cap} triple as real IR -- returns (ir,
        ptr_value, len_value, cap_value), or None when expr.array's
        own base is out of scope.

        Resolves low/high (each defaulting to 0 / the base's own
        length -- high defaults to LENGTH, not CAP: `arr[3:]` means
        "to the current end", not "to the full capacity"; only an
        explicitly-given high can reach cap), three bounds checks
        (IRSliceBoundsCheck -- low <= cap, high <= cap, low <= high),
        then the three derived values (new_cap = cap - low, new_len =
        high - low, ptr = addr + low*stride) via ordinary IRBinOp.

        low_value/high_value/cap_value are immutable once computed --
        every IRBinOp below just reads them -- so new_cap/new_len/ptr
        can be computed in any order, and no push/pop protection is
        needed anywhere: every intermediate value already has its own
        Temp home, safely written before whatever evaluates next."""
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
        """Builds (without lowering) a bare bracketed-list literal
        resolved to SLICE by context (`[]int s = [1, 2, 3]`; the
        general TYPED form, `[]int[1, 2, 3]`, is a different AST shape
        -- a Slice node wrapping an ArrayLiteral) as real IR -- returns
        (ir, ptr_value, len_value, cap_value), or None when some
        element is out of scope for _ir_write_composite_value_into.
        The same (ir, ptr, len, cap) shape _ir_slice_into produces, so
        this drops into every place that already consumes it.

        Always mallocs a fresh backing array, sized to fit -- at LEAST
        1 byte, even for an empty literal (`[]int[]`), guaranteeing a
        genuine, non-null, unique pointer regardless of malloc(0)'s
        implementation-defined behavior: this is what makes `s ==
        none` correctly false for an intentionally empty slice literal
        (a real, live, zero-length slice, not a nil one, same as
        `arr[5:5]`). Always heap-allocated regardless of size, unlike
        an ordinary array variable's own stack threshold: a slice
        literal's backing has to outlive the statement that creates
        it.

        The actual element-writing is _ir_write_array_literal_into,
        reused unchanged: that method only ever reads value_type's own
        .element_type, never .kind or .size, so handing it a SLICE-
        kind Type instead of an ARRAY-kind one works without any
        change on its own end. Nested composite elements (a slice-of-
        structs, a slice-of-slices) fall out for free the same way:
        _ir_write_array_literal_into already recurses into _ir_write_
        composite_value_into for those.

        len and cap are both the literal's own element count -- a
        fresh literal's own backing has no spare room to grow into
        yet.

        Takes no value_type parameter, deliberately: type_of(expr) is
        ALWAYS an ARRAY-kind Type, correctly sized to the literal's own
        element count, regardless of what the surrounding context
        resolves the overall expression to (SLICE, here) -- semantic.py
        annotates an ArrayLiteral node by its own literal shape, never
        by how its caller happens to use it. Computing this method's
        own malloc size from the CALLER's own SLICE-kind Type instead
        would under-allocate for a slice-of-slices literal: type_byte_
        width of ANY slice is always 24 (the descriptor's own fixed
        size), regardless of what it describes."""
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
        """The shared core of _ir_write_slice_descriptor: given an
        ALREADY-computed destination address (an ordinary IRValue --
        however the caller has it), writes an already-produced {ptr,
        len, cap} triple there at offsets 0/8/16, via three ordinary
        IRStores. The +8/+16 offsets are themselves computed via
        ordinary IRBinOp, the same address-as-a-Temp pattern used
        everywhere else in this file."""
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
        """Writes an already-produced {ptr, len, cap} triple through
        dst_expr's own address -- always via _ir_slice_address, since
        the destination is always slice-typed here -- then delegates
        to _ir_write_slice_descriptor_into_address for the rest.
        Shared by every gen_statement_ir case that produces a fresh
        slice value via _ir_slice_into -- see its own docstring, and
        gen_statement_ir's own VarDecl/Assign/IndexAssign/FieldAssign
        cases for where the two are glued together."""
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
        """Builds (without lowering) value_type's implicit zero value
        as real IR, written through dst_address. Unlike _ir_write_
        composite_value_into, this is a TOTAL function: it never
        returns None, since a type's own zero value depends only on
        the type itself, always fully, recursively computable at
        compile time.

        Dispatches on value_type.kind:
          - SLICE: none's own {0, 0, 0} descriptor, via _ir_write_
            slice_descriptor_into_address -- a zero-value slice and a
            none-valued one are, by design, the identical
            representation.
          - STRUCT: every field, each field's own address computed via
            ordinary IRBinOp (skipped for the first field, offset 0),
            recursing back into this method for each field's own type.
          - ARRAY: delegates to _ir_zero_array_loop -- a genuine
            runtime loop, not per-element unrolling, since an array's
            own element count can be large.
          - str: {ptr=0, len=0} (_ir_zero_str_value), written through
            dst_address via _ir_write_str_descriptor_into_address --
            see ir/strings.py's own module docstring for why this is
            safe now, unlike the OLD C-string scheme's shared, static
            empty-string constant.
          - int/bool/int8/uint8: an ordinary IRConst(0, value_type),
            written via IRStore at value_type's own declared width."""
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
        return [IRStore(address=dst_address, value=IRConst(0, value_type), value_type=value_type)]

    def _ir_zero_array_loop(self, dst_address, element_type: Type, count: int) -> list:
        """Builds (without lowering) a genuine, bounded-count real-IR
        loop zeroing `count` consecutive elements of element_type,
        starting at dst_address -- the array-leaf counterpart to _ir_
        write_zero_value_into's own scalar/slice/struct cases, all of
        which can be straight-line code since their own size is always
        small and fixed.

        Each iteration recomputes its own element address from a loop
        counter i (an ordinary, persistent Temp, reassigned each
        iteration via IRMove into a fresh next-value Temp), via the
        same 32-bit-multiply-then-64-bit-add shape _ir_index_address
        uses for ordinary indexing. Then delegates to _ir_write_zero_
        value_into recursively for that one element, which could
        itself be another array (a multi-dimensional zero-init), a
        struct, or a scalar/slice leaf.

        Every Temp this loop uses (i, the byte offset, the per-
        iteration address) is independent of whatever Temps that
        recursive call allocates for itself."""
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
        """Builds (without lowering) real IR that jumps to
        mismatch_label the moment any element/field/byte differs
        between the two given addresses -- falls through only once
        everything has matched. Recurses for ARRAY (a bounded loop,
        mirroring _ir_zero_array_loop's own shape, just comparing
        instead of zeroing) and STRUCT (per field), reaching str (a
        length-first check, then memcmp -- see _ir_string_compare's
        own docstring in ir/strings.py for why length-first, not just
        a straight memcmp, matters here too: comparing memcmp(left,
        right, left_len) before knowing right_len matches it would
        risk reading past the end of a genuinely shorter right buffer)
        and int/bool/int8/uint8 (an ordinary IRLoad-into-value_type's-
        own-width plus IRBinOp comparison) as its base cases.

        The scalar-field/element case's own width-aware load matters:
        comparing at a fixed 4-byte width regardless of the field's
        own declared width would silently read past an int8/uint8
        field's own 1-byte storage into adjacent memory. Going through
        IRLoad-at-value_type's-own-width makes the width correct by
        construction rather than needing its own special case."""
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
            cmp_result = self.ir_program.ids.new_temp(Type.INT)
            mismatch_cond = self.ir_program.ids.new_temp(Type.BOOL)
            return left_read_ir + right_read_ir + [
                IRBinOp(dst=lengths_equal, op=BinaryOp.EQUAL, left=left_len, right=right_len),
                IRBranch(cond=lengths_equal, true_label=lengths_equal_label, false_label=mismatch_label),
                IRLabel(lengths_equal_label),
                IRCall(dst=cmp_result, name='memcmp', args=[left_ptr, right_ptr, left_len]),
                IRBinOp(dst=mismatch_cond, op=BinaryOp.NOT_EQUAL, left=cmp_result, right=IRConst(0, Type.INT)),
                IRBranch(cond=mismatch_cond, true_label=mismatch_label, false_label=continue_label),
                IRLabel(continue_label),
            ]
        # int, bool, int8, uint8 -- an ordinary, width-aware
        # load-and-compare (see this method's own docstring).
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
        """Builds (without lowering) `needle_expr in collection_expr`
        for an ARRAY- or SLICE-typed collection -- returns (ir,
        value), a plain bool Temp. A linear scan, unlike dict's own
        O(1) hash lookup (_ir_dict_contains): "is this exact value an
        element" has no better mechanism for either ARRAY or SLICE, so
        both share this one implementation rather than each getting
        their own -- _ir_indexable_base already gives a uniform
        (address, length) pair for either (an array's own length a
        compile-time IRConst, a slice's own read off its descriptor at
        runtime), needing no branching here on which of the two this
        actually is.

        needle_expr's own address comes from _ir_materialize_value_
        into_scratch (built for dict-literal construction, but already
        generic over any value, scalar or composite) -- always fresh,
        never reusing an existing address even when needle_expr
        already has one, the same "simpler than special-casing, costs
        nothing extra" reasoning that method's own docstring gives.

        Each element's own comparison reuses _ir_composite_equal
        exactly as array equality's own loop does (ir/arrays_slices.
        py's own module-level element-loop shape), but INVERTED: a
        mismatch there means "keep scanning" here (jumps to next_
        label, not an outright failure), and falling through (no
        mismatch -- a match) jumps straight to a TRUE result instead
        of continuing. The loop exhausting with no match falls through
        to a FALSE result."""
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
        """Binds a value already sitting at source_addr into stmt's
        own binding_names[binding_index] -- shared by every for-in
        binding that comes from an ALREADY-ADDRESSABLE source (an
        array/slice element, a dict's own key or value region within
        a bucket), as opposed to the index binding, which is already
        a plain value in a Temp with nothing to load at all (see _ir_
        for_in_array_slice's own separate, direct _ir_finish_scalar_
        var_decl call for that one).

        Mirrors gen_statement_ir's own scalar-vs-composite VarDecl
        split exactly, just fed source_addr instead of an initializer
        expression to evaluate: an ordinary scalar IRLoads its own
        value out of source_addr first, then _ir_finish_scalar_var_
        decl (the exact helper an ordinary scalar VarDecl's own first
        write already goes through -- no real VarDecl node needed, see
        its own signature). A composite (including STR, so a str-
        typed dict key -- stored inline in its own bucket as a {ptr,
        len} pair, byte-for-byte identical to str's own ordinary
        descriptor layout, see runtime.c's own hornet_dict_insert_
        str_key -- needs no special handling beyond this, either)
        mallocs a fresh box first if heap-allocated (never for SLICE,
        matching every other composite VarDecl case's identical
        exclusion), then IRCopies source_addr's own bytes into the
        destination's own address."""
        if binding_type.kind in COMPOSITE_KINDS:
            decl_id = (id(stmt), binding_index)
            slot = ir_fn.var_slots[decl_id]
            if binding_type.kind != TypeKind.SLICE and self._is_heap_allocated(decl_id, binding_type):
                ir = self._ir_malloc_and_store(binding_type, slot)
            else:
                ir = []
            address_of = {
                TypeKind.ARRAY: self._ir_array_address,
                TypeKind.STRUCT: self._ir_struct_address,
                TypeKind.SLICE: self._ir_slice_address,
                TypeKind.DICT: self._ir_dict_address,
                TypeKind.SUM: self._ir_struct_address,  # generic address computation -- see its own docstring
                TypeKind.STR: self._ir_str_address,
            }[binding_type.kind]
            dst_ir, dst_addr = address_of(Variable(name=stmt.binding_names[binding_index]))
            ir.extend(dst_ir)
            ir.append(IRCopy(dst_address=dst_addr, src_address=source_addr, value_type=binding_type))
            return ir
        value = self.ir_program.ids.new_temp(binding_type)
        ir = [IRLoad(dst=value, address=source_addr)]
        ir.extend(self._ir_finish_scalar_var_decl(stmt.binding_names[binding_index], (id(stmt), binding_index), binding_type, value))
        return ir

    def _ir_for_in_array_slice(self, stmt: ForIn, ir_fn) -> list:
        """Builds (without lowering) `for x in arr:` / `for i, x in
        arr:` (equally for a SLICE) as real IR -- a bounded loop over
        _ir_indexable_base's own uniform (address, length), binding
        each element (and, for the two-binding form, the index too)
        into stmt's own binding_names on every pass, rather than
        comparing against one the way _ir_array_slice_contains does.

        continue_label is DISTINCT from start_label, existing
        specifically so `continue` still advances the loop before
        re-checking the condition -- the identical reason For's own
        increment_label exists (see gen_statement_ir's own For case);
        here the "increment" is just i += 1, synthesized directly
        rather than a separate AST node the way For's own increment
        clause is.

        Binding itself delegates to _ir_bind_for_in_value (shared with
        _ir_for_in_dict's own key/value bindings) for the element,
        fed the element's own computed address (base_addr + i *
        element_width) to copy or load from -- works identically
        whether element_type is scalar or composite, that method's
        own concern, not this one's. The two-binding form's own index
        binding is the one exception handled directly here: i is
        already a plain value in a Temp, not an address to read FROM,
        so it goes through _ir_finish_scalar_var_decl on its own.

        Mutation safety: ARRAY needs no check at all (nothing about
        iterating it can change its own fixed address). SLICE caches
        its own DESCRIPTOR's address once at loop start (_ir_slice_
        address, called separately from _ir_indexable_base -- see the
        comment at its own call site below for why this can't just
        reuse _ir_indexable_base's own internal computation) and re-
        LOADS its ptr field from that same, fixed address every
        iteration, comparing against the value cached at loop start --
        an append inside the body that reallocates would change it,
        meaning this iterator's own cached elem_addr math is no
        longer valid -- panicking via hornet_panic (the same runtime
        function bounds-check failures already use) rather than
        silently walking freed or orphaned memory. Ordinary insert-
        without-growth, in-place overwrite, and (for dict, a
        different method entirely, see Stage 3) delete are all still
        safe and unchecked -- this only ever catches a base-address
        change, the one operation that's actually memory-unsafe.

        Re-LOADING from a cached descriptor address, rather than re-
        CALLING _ir_indexable_base(stmt.iterable) a second time, is
        the whole point, not an optimization: stmt.iterable can be an
        Index/Field chain (`for x in rows[i]:`) whose own descriptor
        address depends on evaluating some OTHER expression (i here)
        -- re-evaluating that expression on every recheck would re-
        resolve it against i's CURRENT value, not the one loop start
        actually captured, so a body that reassigns i (`rows[i]`'s own
        i, unrelated to this loop's own counter) would silently
        compare two DIFFERENT slices' own ptr values and panic with a
        false "reallocated" report -- confirmed as a real, constructed
        bug, not a hypothetical one, before this fix existed."""
        iterable_type = type_of(stmt.iterable)
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

        if is_slice:
            # Computed ONCE here, separately from _ir_indexable_base's
            # own (already complete, already correct) ptr/len/cap
            # unpacking just above -- that method has no reason to
            # expose the descriptor's own address alongside the
            # values it already reads out of it, and widening its own
            # return shape for this one, narrow caller would touch
            # every other already-working call site (indexing, 'in',
            # ...) for no benefit to them. A second address
            # computation is harmless -- both this and the one inside
            # _ir_indexable_base run here, at loop start, before the
            # body could ever mutate anything an Index/Field chain's
            # own address depends on.
            descriptor_result = self._ir_slice_address(stmt.iterable)
            if descriptor_result is None:
                raise IRError(
                    f"_ir_slice_address returned None for a SLICE-typed "
                    f"'for ... in' iterable ({stmt.iterable!r}) -- expected "
                    f"to always succeed, having already succeeded once "
                    f"above via _ir_indexable_base")
            descriptor_ir, descriptor_addr = descriptor_result
            ir = ir + descriptor_ir

        i = self.ir_program.ids.new_temp(Type.INT)
        start_label = self.ir_program.ids.new_label("for_in_start")
        body_label = self.ir_program.ids.new_label("for_in_body")
        continue_label = self.ir_program.ids.new_label("for_in_continue")
        end_label = self.ir_program.ids.new_label("for_in_end")

        self._push_scope()
        binding_types = for_in_binding_types(stmt, iterable_type)
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

        if is_slice:
            recheck_ptr = self.ir_program.ids.new_temp(Type.INT64)
            ir.append(IRLoad(dst=recheck_ptr, address=descriptor_addr))
            mutated = self.ir_program.ids.new_temp(Type.BOOL)
            ir.append(IRBinOp(dst=mutated, op=BinaryOp.NOT_EQUAL, left=recheck_ptr, right=base_addr))
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
            ir.append(IRJump(safe_label))  # unreachable -- hornet_panic
            # never returns -- but the verifier requires every block to
            # end in an explicit terminator, the same "harmless, costs
            # a few bytes" reasoning gen_statement_ir's own If case
            # already applies to its own possibly-redundant jumps.
            ir.append(IRLabel(safe_label))

        offset_temp = self.ir_program.ids.new_temp(Type.INT)
        ir.append(IRBinOp(dst=offset_temp, op=BinaryOp.MULTIPLY, left=i, right=IRConst(element_width, Type.INT)))
        elem_addr = self.ir_program.ids.new_temp(Type.INT64)
        ir.append(IRBinOp(dst=elem_addr, op=BinaryOp.ADD, left=base_addr, right=offset_temp))

        if len(slots) == 2:
            # The index binding is always INT -- always scalar --
            # so _ir_finish_scalar_var_decl (the exact helper an
            # ordinary scalar VarDecl's own first write already goes
            # through) applies directly: name/decl_id/value are all
            # it needs, no real VarDecl node required (see its own
            # signature -- name is a plain str, decl_id a plain int,
            # neither tied to any particular node shape).
            ir.extend(self._ir_finish_scalar_var_decl(stmt.binding_names[0], (id(stmt), 0), Type.INT, i))
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
        self._pop_scope()
        return ir

    def _ir_write_composite_value_into(self, dst_address, value_expr: Node, value_type: Type):
        """The general-purpose dispatcher underlying nested literal
        construction: writes value_expr's own value through dst_
        address, by dispatching on value_expr's own shape. Returns
        None when out of scope (a named/partial struct literal,
        chiefly).

        CRITICAL: the ArrayLiteral case below checks value_type.kind
        == ARRAY explicitly, not just isinstance(value_expr,
        ArrayLiteral) -- the identical bracketed-list AST shape is
        ALSO how a slice literal parses (`[1, 2]` used where a slice
        is expected, e.g. the inner literals of the array-of-slices
        `[2][]int[[1, 2], [3, 4]]`). Without this check, value_type.
        kind == SLICE would reach _ir_write_array_literal_into anyway,
        silently treating a slice's own 24-byte descriptor width as if
        it were the outer array's own element width -- corrupting
        every subsequent element's computed address. A slice-typed
        ArrayLiteral-shaped value_expr goes through _ir_slice_literal
        instead.

        Unifies, into one place, every composite-producing shape: an
        existing value's own address (_ir_copy_into_address), a Slice
        production or append call for a slice-typed value (_ir_write_
        slice_descriptor_into_address), an ordinary composite-
        returning Call (_ir_composite_call), or nested literal
        construction itself -- recursing back into _ir_write_array_
        literal_into/_ir_write_struct_literal_into/_ir_write_dict_
        literal_into, which call back into THIS dispatcher for any of
        their own composite elements/fields/values in turn.

        `append` is checked, and handled, BEFORE the generic ordinary-
        Call case below -- append is a Call whose own name is never in
        struct_registry, so without this earlier check it would
        silently reach _ir_composite_call instead, trying to call a
        function literally named 'append' that was never compiled.

        The SUM check comes FIRST, before even the Variable/Field/
        Index case -- deliberately: value_type.kind == SUM must take
        priority over value_expr's own shape regardless of what that
        shape is, PROVIDED value_expr itself is genuinely narrower (a
        struct, a scalar, or str) -- true widening. A Circle-typed
        variable widening into a Shape-typed address is NOT an
        ordinary same-shape copy (_ir_copy_into_address would flatly
        copy type_byte_width(Shape) bytes starting at the Circle's own
        address, reading past its real bounds into whatever memory
        happens to follow, and would never write a discriminant tag at
        all) -- it needs _ir_write_sum_type_value_into's own tag-then-
        payload treatment regardless of whether value_expr is a bare
        struct literal, an already-struct-typed value, or a scalar/str
        one. But value_expr may ALREADY be sum-typed itself (`[2]Shape
        shapes = [a, b]`, a and b already Shape-typed variables) -- NOT
        widening at all, and _ir_write_sum_type_value_into would crash
        outright trying to look up an already-sum-typed source's own
        index in the variant list (never itself a listed variant).
        That case is deliberately excluded here and falls through to
        the ordinary Variable/Field/Index/composite-call cases right
        below, same type on both sides, needing no widening logic."""
        if value_type.kind == TypeKind.SUM and type_of(value_expr).kind != TypeKind.SUM:
            return self._ir_write_sum_type_value_into(dst_address, value_expr, value_type)
        if is_composite_addressable(value_expr):
            return self._ir_copy_into_address(dst_address, value_expr, value_type)
        if isinstance(value_expr, NoneLiteral):
            zero_ptr = IRConst(0, Type.INT64)
            zero_int = IRConst(0, Type.INT)
            return self._ir_write_slice_descriptor_into_address(dst_address, zero_ptr, zero_int, zero_int)
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
            # _ir_write_dict_literal_into (ir/dicts.py) already exists
            # and is already fully general -- dst_address is a plain
            # parameter, not tied to any particular variable or slot --
            # so this is wiring, not new construction logic. Its own
            # ir_fn parameter comes from self.ir_fn (set once per
            # function in gen_function_ir) rather than threaded
            # through this method's own signature and its three other
            # callers, none of which otherwise need it.
            return self._ir_write_dict_literal_into(dst_address, value_expr, value_type, self.ir_fn)
        if value_type.kind == TypeKind.STR:
            # The one str-typed shape none of the cases above already
            # cover: a StringLiteral or a Binary(ADD) concatenation --
            # an existing addressable value and a str-returning
            # ordinary Call both already fell through to their own,
            # generic cases above (is_composite_addressable/the
            # trailing isinstance(value_expr, Call) catch-all, both
            # already generic over value_type). _ir_str_value produces
            # the {ptr, len} pair; _ir_write_str_descriptor_into_
            # address writes it through dst_address.
            value_ir, ptr_value, len_value = self._ir_str_value(value_expr)
            return value_ir + self._ir_write_str_descriptor_into_address(dst_address, ptr_value, len_value)
        if isinstance(value_expr, Call) and value_expr.name in self.ir_program.struct_registry:
            return self._ir_write_struct_literal_into(dst_address, value_expr, value_type)
        if isinstance(value_expr, Call):
            return self._ir_composite_call(dst_address, value_expr)
        return None

    def _ir_write_array_literal_into(self, dst_address, expr: ArrayLiteral, array_type: Type):
        """Builds (without lowering) an array literal's elements as
        real IR, written through dst_address. Returns None when some
        element's own value is out of scope for _ir_write_composite_
        value_into (a named/partial struct literal, chiefly) -- an
        element that's itself composite is handled via mutual
        recursion through that method.

        Each element's own address is dst_address + i*element_width
        via ordinary IRBinOp, skipped for element 0. A scalar
        element's value is evaluated via gen_expr_ir and written via
        IRStore; a composite element delegates to _ir_write_composite_
        value_into. If any element is out of scope, the whole literal
        falls back (returns None) -- any IR already built for earlier
        elements is simply discarded by the caller, with no risk of a
        partial write since none of it is ever emitted."""
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
        """A bare array-literal statement (`[3]int[1, 2, 3]` alone,
        with no assignment) never needs its VALUE materialized
        anywhere, so this just evaluates each of the literal's
        directly-written elements for whatever side effects it might
        have, discarding every result.

        Recurses for a nested ArrayLiteral element. An element that's
        itself some other, non-literal array-, slice-, or struct-typed
        expression (a Variable, an indexed sub-array, an array/struct-
        returning Call, ...) is a deliberately out-of-scope gap:
        reading a bare array-typed Variable has no side effect worth
        preserving, but an array-returning Call might, and correctly
        distinguishing the two isn't implemented here. Raises a clear
        error rather than silently skipping (which could drop a real
        side effect)."""
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

    def _ir_slice_none_comparison(self, expr: Binary):
        """Builds (without lowering) `slice_expr == none` or
        `slice_expr != none` (in either operand order) as real IR --
        returns (ir, value), or None when slice_expr's own base is
        out of scope -- moot in practice: every reachable slice-typed
        base shape already succeeds through _ir_indexable_base. The
        caller (gen_expr_ir) raises IRError explicitly on a None here
        rather than silently propagating it further up the call
        chain.

        Reuses _ir_indexable_base for the slice's own address,
        discarding length/cap. An ordinary IRBinOp (expr.op, ptr,
        IRConst(0, INT64)) produces the right comparison -- checking
        specifically whether the slice's ptr field is null (Go's
        nil-vs-empty-slice distinction), exactly the semantics a
        slice-vs-none comparison needs."""
        slice_expr = expr.left if type_of(expr.left).kind == TypeKind.SLICE else expr.right
        base = self._ir_indexable_base(slice_expr)
        if base is None:
            return None
        base_ir, ptr, length, cap = base
        t_result = self.ir_program.ids.new_temp(Type.BOOL)
        check = IRBinOp(dst=t_result, op=expr.op, left=ptr, right=IRConst(0, Type.INT64))
        return base_ir + [check], t_result

    def _ir_write_append_value_at(self, target_addr, value_arg: Node, element_type: Type):
        """Writes append(s, value)'s own `value` argument at an
        already-computed target_addr -- called independently by both
        of _ir_append_call's own REUSE and REALLOCATE branches, each
        reaching a different target_addr through a different path.
        Returns None when out of scope (propagating _ir_write_
        composite_value_into's own None).

        Scalar element types (int/bool/str/int64) go through an
        ordinary IRStore. Composite element types (array/slice/struct)
        reuse _ir_write_composite_value_into unchanged -- the same
        dispatcher every other "write a composite value into a known
        address" site uses.

        Called once per branch, not once upfront with the result
        reused in both: a composite value has no single Temp of its
        own to compute once and reuse, since WRITING it is the whole
        operation. This duplicates the value argument's own codegen
        across both branches, but never double-EXECUTES it at runtime
        -- exactly one of the two branches ever runs for a given
        call."""
        if element_type.kind in COMPOSITE_KINDS:
            return self._ir_write_composite_value_into(target_addr, value_arg, element_type)
        value_ir, value = self.gen_expr_ir(value_arg)
        return value_ir + [IRStore(address=target_addr, value=value, value_type=element_type)]

    def _ir_append_call(self, expr: Call):
        """Builds (without lowering) append(s, value)'s resulting
        {ptr, len, cap} triple as real IR -- returns (ir, ptr_value,
        len_value, cap_value), or None when out of scope: s's own base
        out of scope, or `value` itself out of scope for _ir_write_
        append_value_at when the element type is composite.

        ANY element type is in scope, including array/slice/struct --
        see _ir_write_append_value_at's own docstring for how writing
        the value works for each.

        The reuse-vs-reallocate decision is real IR: an ordinary
        IRBinOp comparison (length >= cap; a plain SIGNED comparison
        is correct here, unlike IRBoundsCheck's own unsigned trick --
        len/cap are compiler-maintained invariants, never a user-
        supplied value that could be negative) plus an ordinary
        IRBranch. The REUSE path (length < cap): compute the target
        address, write the value, increment length. The REALLOCATE
        path (length >= cap): grows first via IRSliceGrow, then writes
        the value into the newly-available slot at the same offset
        (length * element_width) within the fresh backing.

        Both paths write into the SAME three result Temps (result_ptr/
        result_len/result_cap), then jump to a shared end label."""
        slice_arg, value_arg = expr.args
        slice_type = type_of(slice_arg)
        element_type = slice_type.element_type
        element_width = type_byte_width(element_type, self.ir_program.struct_registry, self.ir_program.sum_type_registry)

        if isinstance(slice_arg, NoneLiteral):
            ptr = self.ir_program.ids.new_temp(Type.INT64)
            length = self.ir_program.ids.new_temp(Type.INT)
            cap = self.ir_program.ids.new_temp(Type.INT)
            base_ir = [
                IRMove(dst=ptr, src=IRConst(0, Type.INT64)),
                IRMove(dst=length, src=IRConst(0, Type.INT)),
                IRMove(dst=cap, src=IRConst(0, Type.INT)),
            ]
        else:
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

        # REUSE path (length < cap): write directly into the existing
        # backing at its own next-free slot.
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

        # REALLOCATE path (length >= cap): grow first (IRSliceGrow,
        # touching neither length nor the new value at all), THEN
        # write the new value into the newly-available slot, at the
        # same offset (length * element_width) the REUSE path's own
        # slot would have been at, just within the fresh backing.
        grow_ptr = self.ir_program.ids.new_temp(Type.INT64)
        grow_cap = self.ir_program.ids.new_temp(Type.INT)
        grow = IRSliceGrow(
            dst_ptr=grow_ptr, dst_cap=grow_cap,
            ptr=ptr, length=length, cap=cap,
            element_width=element_width,
        )
        realloc_offset_temp = self.ir_program.ids.new_temp(Type.INT)
        realloc_target_addr = self.ir_program.ids.new_temp(Type.INT64)
        realloc_new_len = self.ir_program.ids.new_temp(Type.INT)
        realloc_write_ir = self._ir_write_append_value_at(realloc_target_addr, value_arg, element_type)
        if realloc_write_ir is None:
            return None
        realloc_ir = [
            IRLabel(realloc_label),
            grow,
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
        """Builds (without lowering) len(x)'s own result as real IR --
        returns (ir, value), or None when x's own base is out of
        scope (see _ir_indexable_base's own docstring) -- an
        ArrayLiteral, or a Call, when x is slice-typed.

        str-typed x is handled first, entirely separately: _ir_
        indexable_base's own "address plus length" abstraction is
        specifically for an INDEXABLE base (array/slice), which str
        isn't (str indexing/slicing is its own, later, deliberately
        deferred follow-up) -- _ir_str_value is the identical
        "evaluate x once, however its own shape produces a {ptr, len}
        pair" abstraction, just under a different name for a type
        outside _ir_indexable_base's own scope. ptr is computed and
        then discarded, but NOT skipped, for the identical reason x's
        own address/cap are below: x is still fully evaluated
        regardless, so any side effect buried in it genuinely runs
        (`len(s + t)` still concatenates, even though only the
        resulting length is ever read).

        Reuses _ir_indexable_base directly for ARRAY/SLICE, the same
        "address plus length, however each is represented" abstraction
        indexing and slicing already share. x's own address (and cap)
        are computed and then discarded, but NOT skipped: x is still
        fully evaluated regardless, so any bounds check or side effect
        buried in it genuinely runs -- deliberately: a length-only
        read must not silently skip a bounds check or side effect the
        full expression would otherwise trigger (`len(arr[i])` still
        aborts if i is out of range).

        For an ARRAY base, the returned value is a compile-time
        IRConst (x's declared size, never read out of x at runtime);
        for a SLICE base, an ordinary INT Temp holding a runtime
        value read from x's own descriptor -- either way, exactly
        _ir_indexable_base's own second return value, unchanged.

        A DICT base is handled first, entirely separately, for the
        identical reason str is: dict isn't indexable/sliceable
        either. Reads count straight out of x's own descriptor
        (offset 8 -- see runtime.c's own dict_tombstones docstring for
        why count specifically already means LIVE entries, no
        tombstone adjustment needed) via _ir_dict_address, so any side
        effect or bounds check buried in x still genuinely runs."""
        arg = expr.args[0]
        if type_of(arg).kind == TypeKind.DICT:
            result = self._ir_dict_address(arg)
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

