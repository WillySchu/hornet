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

from codegen.errors import CodegenError
from ir.ir import (
    IRBinOp,
    IRBoundsCheck,
    IRBranch,
    IRCall,
    IRConst,
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
from codegen.utils import type_of, type_byte_width
from parser import Node, ArrayLiteral, Call, Field, Index, Slice, Variable, NoneLiteral, Binary, BinaryOp
from semantic import TypeKind, Type


class ArraysSlicesMixin:
    def _ir_array_address(self, expr: Node):
        """The address of an array-typed expr -- Variable is the one
        genuine leaf (a fixed, compile-time
        %rbp-relative offset -- IRLocalAddress directly -- or, if
        heap-allocated, the pointer STORED at that offset --
        IRLocalAddress for the slot's own address, then an ordinary
        IRLoad reading the pointer through it, the same composition
        _ir_struct_address's own Variable case uses, for the identical
        reason: IRLocalAddress always means "the address of this
        slot," never "the value stored there"), Index/Field recurse
        into _ir_index_address/_ir_field_address.

        Returns None for an ArrayLiteral (construction, not an
        existing address -- out of scope for this method entirely; a
        bare ArrayLiteral has its own, separate real-IR handling
        elsewhere, e.g. _ir_materialize_array_literal)."""
        if isinstance(expr, Variable):
            slot = self._local_slot(expr.name)
            array_type = self._local_type(expr.name)
            slot_addr = self.ir_program.ids.new_temp(Type.INT64)
            ir = [IRLocalAddress(dst=slot_addr, slot=slot)]
            if self._is_heap_allocated(self._local_decl_id(expr.name), array_type):
                addr_temp = self.ir_program.ids.new_temp(Type.INT64)
                ir.append(IRLoad(dst=addr_temp, address=slot_addr))
                return ir, addr_temp
            return ir, slot_addr
        if isinstance(expr, Index):
            return self._ir_index_address(expr)
        if isinstance(expr, Field):
            return self._ir_field_address(expr)
        return None

    def _ir_slice_address(self, expr: Node):
        """Mirrors _ir_array_address's own shape for a slice-typed
        expr, but simpler: a slice variable is never heap-allocated
        (every one of _is_heap_allocated's own call sites scopes it to
        ARRAY/STRUCT only -- a slice's own 24-byte descriptor is always a
        small, fixed-size, stack-resident value, regardless of
        whether its backing array is heap-allocated), so the Variable
        leaf is just a single LeaQFrame, no heap-vs-stack branch
        needed at all. Index/Field recurse into _ir_index_address/
        _ir_field_address exactly like _ir_array_address's own do --
        neither cares what the result's own type is, only its address
        and byte width (already correct for SLICE via type_byte_
        width/_field_offset, with no changes needed there).

        Returns None for a Slice (`arr[a:b]`, slice production -- a
        fresh descriptor, not an existing address) or a Call (a
        slice-returning function call) -- both genuinely out of scope
        for now, real, separate follow-up work."""
        if isinstance(expr, Variable):
            slot = self._local_slot(expr.name)
            addr_temp = self.ir_program.ids.new_temp(Type.INT64)
            return [IRLocalAddress(dst=addr_temp, slot=slot)], addr_temp
        if isinstance(expr, Index):
            return self._ir_index_address(expr)
        if isinstance(expr, Field):
            return self._ir_field_address(expr)
        return None

    def _ir_materialize_composite_call(
            self, call_expr: Call, value_type: Type) -> tuple[Union[IRLocalAddress, IRCall], Temp]:
        """Builds (without lowering) an ordinary composite-returning
        Call's own materialized address as real IR -- returns (ir,
        address). Used wherever such a call sits directly at an
        addressable-base position (Index.array, Field.base, Slice.
        array) with no address of its own to compute, unlike a
        Variable/Field/Index base.

        Whether the result lands on the stack or the heap is decided
        entirely by whether _collect_argument_temps_in_expr's own
        pre-pass reserved a slot for id(call_expr) -- see its own
        docstring for exactly which of the three base positions get
        one (Index/Field, when small enough) and which never do
        (Slice, which always escapes, regardless of size, since the
        slice it produces can outlive this statement). No reservation
        found here means malloc, matching the identical "no slot
        reserved -> malloc" contract _gen_materialize_argument_temp_
        into's own docstring already establishes for a composite-
        returning Call used as an ordinary function argument -- this
        is the same mechanism, reused for three more AST positions,
        not a new one.

        Either way, the destination address is handed to _ir_
        composite_call exactly as any other composite-returning
        call's own destination would be -- the result is written
        through it via the ordinary hidden-pointer convention, with
        no new IR concept needed at all."""
        if id(call_expr) in self._argument_temp_slots:
            slot = self._argument_temp_slots[id(call_expr)]
            addr = self.ir_program.ids.new_temp(Type.INT64)
            addr_ir = [IRLocalAddress(dst=addr, slot=slot)]
        else:
            addr = self.ir_program.ids.new_temp(Type.INT64)
            size = type_byte_width(value_type, self.ir_program.struct_registry)
            addr_ir = [IRCall(dst=addr, name='malloc', args=[IRConst(size, Type.INT64)])]
        call_ir = self._ir_composite_call(addr, call_expr, value_type)
        return addr_ir + call_ir, addr

    def _ir_materialize_array_literal(self, expr: ArrayLiteral):
        """Builds (without lowering) an ArrayLiteral's own materialized
        address as real IR -- returns (ir, address), or None when some
        element is itself out of scope for _ir_write_array_literal_
        into. Used wherever such a literal sits directly at an
        addressable-base position (`[1, 2, 3][i]`, `[1, 2, 3][a:b]`)
        with no address of its own to compute -- the ArrayLiteral
        counterpart to _ir_materialize_composite_call, sharing its
        exact same skeleton (a reserved slot, or malloc when none was
        reserved), just populating the backing via _ir_write_array_
        literal_into instead of _ir_composite_call.

        A struct-literal Call sitting at one of these same base
        positions needs no equivalent of its own: semantic.py already
        rejects that outright, wherever it would appear (Point(1,2).x
        fails to type-check at all, naming the specific, narrow set of
        positions a struct literal IS allowed in, none of which
        include being a base) -- there is no gap here for codegen to
        close.

        Only ever reached from _ir_indexable_base's own ARRAY branch,
        never its SLICE branch: type_of on a bare ArrayLiteral used
        directly as a base (with no declared destination type to
        resolve it against, unlike a VarDecl/Assign/FieldAssign/
        IndexAssign/Return's own declared type) is always ARRAY-kind,
        the same fact _ir_slice_literal's own docstring already
        establishes -- so a Slice production from this kind of base
        (`[1, 2, 3][a:b]`) still reaches this same ARRAY-branch
        materialization for its own base, exactly like an ordinary
        index read does, with _ir_slice_into itself producing the
        actual slice descriptor one level up.

        No value_type parameter, deliberately, for the identical
        reason _ir_slice_literal's own docstring gives for dropping
        its own: type_of(expr) is already exactly the ARRAY-kind Type
        this needs, with no destination-type ambiguity possible here
        at all (there IS no destination at this position)."""
        array_type = type_of(expr)
        if id(expr) in self._argument_temp_slots:
            slot = self._argument_temp_slots[id(expr)]
            addr = self.ir_program.ids.new_temp(Type.INT64)
            addr_ir = [IRLocalAddress(dst=addr, slot=slot)]
        else:
            addr = self.ir_program.ids.new_temp(Type.INT64)
            size = type_byte_width(array_type, self.ir_program.struct_registry)
            addr_ir = [IRCall(dst=addr, name='malloc', args=[IRConst(size, Type.INT64)])]
        write_ir = self._ir_write_array_literal_into(addr, expr, array_type)
        if write_ir is None:
            return None
        return addr_ir + write_ir, addr

    def _ir_indexable_base(self, expr: Node):
        """Builds (without lowering) the address, length, and capacity
        of an indexable base as real IR -- returns (ir, addr_value,
        length_value, cap_value), or None when expr's own shape is
        still out of scope (a struct-literal Call used directly at
        this position -- moot in practice, since semantic.py already
        rejects that outright wherever it would appear, not just here;
        see _ir_materialize_array_literal's own docstring). cap_value
        is always computed, even by a caller (_ir_index_address) that
        never reads it back -- cheap enough that one uniform, three-
        value contract beats making it optional.

        ARRAY-typed: delegates to _ir_array_address for a Variable/
        Field/Index base, to _ir_materialize_composite_call for an
        ordinary composite-returning Call (`makeArray()[i]`), or to
        _ir_materialize_array_literal for a bare bracketed-list
        literal (`[1, 2, 3][i]`) -- see each one's own docstring;
        length and cap are both always the same
        compile-time IRConst either way -- an array has no separate
        capacity.

        SLICE-typed, Variable: now the identical shape as the Field/
        Index case just below, just starting from IRLocalAddress (a
        fixed, compile-time frame offset) instead of _ir_field_
        address/_ir_index_address (a runtime-computed one) -- ptr read
        straight off that address via IRLoad, len/cap via their own
        +8/+16 addresses computed first through ordinary IRBinOp. len/
        cap are captured as INT (32-bit) Temps directly, not INT64 --
        an array/slice's own length or capacity always fits in 32
        bits, the same assumption the old-style bounds check's own
        len_reg_32/cap_operand already make.

        SLICE-typed, Field/Index (`p.values`, `rows[i]`): unlike the
        Variable case, there's no fixed, compile-time offset to read
        three fields from directly -- the descriptor's own address is
        itself a runtime value, computed via _ir_field_address/_ir_
        index_address (already generic over the result's own type, so
        neither needed any change to support this). ptr is read
        straight off that address via IRLoad; len/cap need their own
        +8/+16 addresses computed first, via ordinary IRBinOp, since
        IRLoad always reads at its address's own location, with no
        offset field of its own. This is also what makes a slice
        reached through a chain (`outer.inner.values[0]`, or a slice-
        of-slices `rows[i][j]`, where rows[i] is itself SLICE-typed
        and reached via this exact branch recursively) fall out for
        free: _ir_field_address/_ir_index_address already recurse
        through arbitrary Variable/Field/Index chains on their own.

        SLICE-typed, ordinary composite-returning Call
        (`makeSlice()[i]`): same read shape as Field/Index just above,
        just with the descriptor's own address coming from _ir_
        materialize_composite_call instead. Also reached, via _ir_
        slice_into's own delegation to this method for its OWN base,
        by a slice PRODUCED from this call's result (`makeArray()
        [a:b]`) -- see _ir_materialize_composite_call's own docstring
        for why that specific shape always heap-allocates, regardless
        of size, unlike an ordinary index/field read of the same call.

        SLICE-typed, append (`append(s, x)[i]`, `append(s, x) ==
        none`, `sumSlice(append(s, x))`, ...): _ir_append_call already
        returns exactly (ir, ptr, len, cap) -- the identical four-
        tuple this method itself returns -- so this delegates straight
        to it, the same "no materialization needed, the values already
        are what this method wants back" shape the Slice case just
        below uses for _ir_slice_into. append is checked here, before
        the ordinary-composite-Call case just above, for the identical
        reason it's always checked before an ordinary Call everywhere
        else in this arc: it's a builtin, never a compiled function,
        so _is_ordinary_composite_call already excludes it by name --
        without a dedicated case here, `append(...)` used directly as
        a base fell through to `if not isinstance(expr, (Variable,
        Field, Index)): return None`, forcing every caller composed
        through this method (index/field addressing, VarDecl/Assign/
        IndexAssign/FieldAssign/Return's own copy logic, equality,
        argument-passing, none-comparison) down old-style fallback
        paths that were never actually exercised by this arc's own
        test suite before -- three of which turned out to be
        genuinely broken (a link-time "undefined reference to
        `append`", a KeyError from a pre-pass assumption that never
        held for this shape, and an implicit None return crashing
        with a TypeError several frames away) rather than just slower.
        This one, two-line addition is the fix for all of them at
        once, not four separate patches: every one of those call
        sites already recurses through this method for its own base
        resolution, so they compose correctly automatically once this
        method itself does.

        SLICE-typed, Slice (`arr[:][0]`, `s[a:b][c:d]`): delegates
        straight to _ir_slice_into, whose own return shape (ir, ptr,
        len, cap) already matches this method's own exactly -- there
        is no scratch-slot materialization to do here at all, unlike
        the old-style gen_indexable_base_into's own Slice case. That
        mechanism existed only to work around hand-written assembly's
        fixed registers; _ir_slice_into already produces its own
        triple as ordinary, independent Temps, so using them directly
        is not a shortcut around some missing step, it's simply what
        the values already are. This is also what makes arbitrarily
        deep chains (`arr[a:b][c:d][e]`) fall out for free, via the
        same mutual recursion _ir_index_address/_ir_array_address/
        this method already use for multi-dimensional arrays -- each
        nested _ir_slice_into call gets its own fresh Temps, so unlike
        the old-style shared slot, there is no nested-lifetime safety
        argument to make here at all."""
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
            if not isinstance(expr, (Variable, Field, Index)):
                return None
            addr_result = self._ir_array_address(expr)
            if addr_result is None:
                return None
            addr_ir, addr_value = addr_result
            size_const = IRConst(base_type.size, Type.INT)
            return addr_ir, addr_value, size_const, size_const
        if base_type.kind == TypeKind.SLICE:
            if isinstance(expr, Variable):
                slot = self._local_slot(expr.name)
                descriptor_addr = self.ir_program.ids.new_temp(Type.INT64)
                ptr_temp = self.ir_program.ids.new_temp(Type.INT64)
                len_addr = self.ir_program.ids.new_temp(Type.INT64)
                len_temp = self.ir_program.ids.new_temp(Type.INT)
                cap_addr = self.ir_program.ids.new_temp(Type.INT64)
                cap_temp = self.ir_program.ids.new_temp(Type.INT)
                ir = [
                    IRLocalAddress(dst=descriptor_addr, slot=slot),
                    IRLoad(dst=ptr_temp, address=descriptor_addr),
                    IRBinOp(dst=len_addr, op=BinaryOp.ADD, left=descriptor_addr, right=IRConst(8, Type.INT64)),
                    IRLoad(dst=len_temp, address=len_addr),
                    IRBinOp(dst=cap_addr, op=BinaryOp.ADD, left=descriptor_addr, right=IRConst(16, Type.INT64)),
                    IRLoad(dst=cap_temp, address=cap_addr),
                ]
                return ir, ptr_temp, len_temp, cap_temp
            if isinstance(expr, (Field, Index)):
                addr_result = self._ir_field_address(expr) if isinstance(expr, Field) else self._ir_index_address(expr)
                if addr_result is None:
                    return None
                addr_ir, descriptor_addr = addr_result
                ptr_temp = self.ir_program.ids.new_temp(Type.INT64)
                len_addr = self.ir_program.ids.new_temp(Type.INT64)
                len_temp = self.ir_program.ids.new_temp(Type.INT)
                cap_addr = self.ir_program.ids.new_temp(Type.INT64)
                cap_temp = self.ir_program.ids.new_temp(Type.INT)
                ir = addr_ir + [
                    IRLoad(dst=ptr_temp, address=descriptor_addr),
                    IRBinOp(dst=len_addr, op=BinaryOp.ADD, left=descriptor_addr, right=IRConst(8, Type.INT64)),
                    IRLoad(dst=len_temp, address=len_addr),
                    IRBinOp(dst=cap_addr, op=BinaryOp.ADD, left=descriptor_addr, right=IRConst(16, Type.INT64)),
                    IRLoad(dst=cap_temp, address=cap_addr),
                ]
                return ir, ptr_temp, len_temp, cap_temp
            if isinstance(expr, Call) and expr.name == 'append':
                return self._ir_append_call(expr)
            if self._is_ordinary_composite_call(expr):
                addr_ir, descriptor_addr = self._ir_materialize_composite_call(expr, base_type)
                ptr_temp = self.ir_program.ids.new_temp(Type.INT64)
                len_addr = self.ir_program.ids.new_temp(Type.INT64)
                len_temp = self.ir_program.ids.new_temp(Type.INT)
                cap_addr = self.ir_program.ids.new_temp(Type.INT64)
                cap_temp = self.ir_program.ids.new_temp(Type.INT)
                ir = addr_ir + [
                    IRLoad(dst=ptr_temp, address=descriptor_addr),
                    IRBinOp(dst=len_addr, op=BinaryOp.ADD, left=descriptor_addr, right=IRConst(8, Type.INT64)),
                    IRLoad(dst=len_temp, address=len_addr),
                    IRBinOp(dst=cap_addr, op=BinaryOp.ADD, left=descriptor_addr, right=IRConst(16, Type.INT64)),
                    IRLoad(dst=cap_temp, address=cap_addr),
                ]
                return ir, ptr_temp, len_temp, cap_temp
            if isinstance(expr, Slice):
                return self._ir_slice_into(expr)
            return None
        return None

    def _ir_nil_slice(self):
        """Builds (without lowering) a nil slice's own {ptr, len, cap}
        triple as real IR -- all-zero, matching this compiler's own
        nil-slice representation throughout (ptr=0 -- never
        dereferenced, since len=0 always gets checked first before any
        element access; len=cap=0). Returns (ir, ptr_value, len_value,
        cap_value), the identical shape _ir_indexable_base/_ir_
        append_call/_ir_slice_into already return.

        Shared by every VarDecl/Assign/IndexAssign/FieldAssign/Return
        case initializing a slice-typed destination to a bare `none`,
        and by _ir_slice_arg's own NoneLiteral case just below --
        factored out here specifically so all of them build the
        IDENTICAL triple from one place, rather than each duplicating
        this same three-IRMove sequence independently."""
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
        element read/write (dispatch.py's _ir_load / this module's
        own _ir_index_assign) and a sub-array (this method's own
        recursive base case, via _ir_array_address, when expr.array
        is itself an Index -- this method and _ir_array_address call
        back into each other for exactly this reason, so `matrix[i][j]`
        falls out with no special-casing for depth).

        Returns None when expr.array's own base is out of scope for
        real IR right now -- see _ir_indexable_base.

        The bounds check aside (IRBoundsCheck -- see its own
        docstring for why no ordinary BinaryOp can express it), the
        actual arithmetic is ordinary IRBinOp: multiply the index by
        the element's own stride, add to the base. Reuses exactly the
        same 32-bit-multiply-then-64-bit-add shape gen_index_address_
        into's own comment already justifies -- the bounds check
        guarantees a small, non-negative index, so a 32-bit multiply
        is safe, and its own write already zero-extends into the full
        64-bit register the following ADD reads (IRBinOp's own
        lowering derives each operation's width from its LEFT operand
        -- INT for the multiply, INT64 for the add -- with no explicit
        width juggling needed here at all).

        No push/pop protection is needed around evaluating expr.index,
        unlike the old-style version: base_addr/length_value are
        already safely stored in their own Temp homes (register or
        memory, via the allocator) the moment _ir_indexable_base
        finishes, before expr.index (which could itself be arbitrarily
        complex, even a call) ever runs -- the same "a Temp's home is
        independent of what computed it" property that already made
        IRCopy's own two-address capture protection-free."""
        base = self._ir_indexable_base(expr.array)
        if base is None:
            return None
        base_ir, base_addr, length_value, _cap_value = base
        element_type = type_of(expr.array).element_type
        element_stride = type_byte_width(element_type, self.ir_program.struct_registry)

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
        own base is out of scope (see _ir_indexable_base) -- an
        ArrayLiteral, or a Call, when expr.array is slice-typed.

        Mirrors the old-style version's own logic exactly, just via
        real IR: resolve low/high (each defaulting to 0 / the base's own
        length -- high defaults to LENGTH, not CAP: `arr[3:]` means
        "to the current end", not "to the full capacity"; only an
        explicitly-given high is allowed to reach cap), three bounds
        checks (IRSliceBoundsCheck -- low <= cap, high <= cap, low <=
        high, matching the old-style version's own order and its own
        comment for why cap, not len, is the bound), then the three
        derived values (new_cap = cap - low, new_len = high - low, ptr = addr
        + low*stride) via ordinary IRBinOp.

        Unlike the old-style version's own careful "compute new_cap/new_len
        BEFORE low is scaled" ordering (needed there because low_32
        is mutated in place by the following IMul), no such ordering
        matters here at all: low_value/high_value/cap_value are
        immutable once computed -- every IRBinOp below just reads
        them, never mutates them -- so new_cap/new_len/ptr can be
        computed in any order.

        No push/pop protection needed anywhere in here either, unlike
        the old-style version's own extensive stack discipline: every
        intermediate value already has its own, independent Temp home,
        safely written before whatever evaluates next (even expr.low/
        expr.high, which could be arbitrarily complex) ever runs --
        the same property that already made _ir_index_address
        protection-free."""
        base = self._ir_indexable_base(expr.array)
        if base is None:
            return None
        base_ir, base_addr, length_value, cap_value = base
        element_stride = type_byte_width(type_of(expr.array).element_type, self.ir_program.struct_registry)

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
        entirely -- a Slice node wrapping an ArrayLiteral, reached
        instead through _ir_indexable_base's own, still out-of-scope
        ArrayLiteral case) as real IR -- returns (ir, ptr_value,
        len_value, cap_value), or None when some element is itself out
        of scope for _ir_write_composite_value_into. The identical
        (ir, ptr, len, cap) shape _ir_slice_into already produces, so
        this drops into every place that already consumes it (_ir_
        write_slice_descriptor, and _ir_write_composite_value_into's
        own Slice/append cases) with no new consuming code needed.

        Always mallocs a fresh backing array, sized to fit -- at LEAST
        1 byte, even for an empty literal (`[]int[]`), guaranteeing a
        genuine, non-null, unique pointer regardless of malloc(0)'s
        implementation-defined behavior, the same reason gen_array_
        literal_heap_alloc_into's own docstring gives: this is what
        makes `s == none` correctly false for an intentionally empty
        slice literal (a real, live, zero-length slice, not a nil
        one, same as `arr[5:5]`) -- rather than a genuinely new rule,
        this replicates that old-style method's own exact behavior.
        Always heap-allocated regardless of size, unlike an ordinary
        array variable's own 16KB stack threshold: a slice literal's
        backing has to outlive the statement that creates it, the
        identical reasoning _ir_materialize_composite_call's own
        docstring gives for why a slice PRODUCED from a materialized
        call always escapes too.

        The actual element-writing is _ir_write_array_literal_into,
        reused completely UNCHANGED, not a new, parallel builder
        mirroring it: that method only ever reads value_type's own
        .element_type, never .kind or .size, so handing it a SLICE-
        kind Type instead of an ARRAY-kind one works without any
        change on its own end -- exactly what the old-style version
        already did, passing type_of(expr) (a SLICE-kind Type here) as
        its own array_type argument: this isn't a new capability being
        assumed, it's the same one the old code already relied on.
        Nested composite elements (a slice-of-structs, a slice-of-
        slices) fall out for free the same way: _ir_write_array_
        literal_into already recurses into _ir_write_composite_value_
        into for those.

        len and cap are both the literal's own element count -- a
        fresh literal's own backing has no spare room to grow into
        yet, matching the old-style version's own ArrayLiteral case
        exactly ('cap is set equal to len').

        Takes no value_type parameter, deliberately: type_of(expr) is
        ALWAYS an ARRAY-kind Type, correctly sized to the literal's
        own element count, regardless of what the surrounding context
        resolves the overall expression to (SLICE, here) -- semantic.py
        annotates an ArrayLiteral node by its own literal shape ("N
        elements of type X"), never by how its caller happens to use
        it. A REAL BUG, found and fixed before this ever shipped: an
        earlier version of this method computed its own malloc size
        from the CALLER's own SLICE-kind Type instead (type_byte_width
        of ANY slice is always 24, the descriptor's own fixed size,
        regardless of what it describes) -- for a slice-of-slices
        literal with more than one outer element, this under-allocated
        by exactly half (24 bytes reserved for what needed count * 24),
        corrupting adjacent heap memory the moment the second element
        was written. Caught by test_untyped_nested_slice_literal,
        which returned 11 instead of 10 -- traced directly rather than
        assumed, confirming type_of(expr) was already 48 bytes (2 * 24,
        correct) while the old computation gave 24."""
        array_type = type_of(expr)
        count = len(expr.elements)
        size = max(1, type_byte_width(array_type, self.ir_program.struct_registry))
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
            raise CodegenError(
                f"_ir_slice_address returned None for a slice-descriptor write's "
                f"own destination ({dst_expr!r}) -- expected to always succeed, "
                f"since an assignment's own destination is always a Variable/"
                f"Field/Index, never a Slice production or Call")
        dst_ir, dst_addr = result
        return dst_ir + self._ir_write_slice_descriptor_into_address(dst_addr, ptr_value, len_value, cap_value)

    def _ir_write_zero_value_into(self, dst_address, value_type: Type) -> list:
        """Builds (without lowering) value_type's implicit zero value
        as real IR, written through dst_address (an ordinary INT64-
        typed IRValue, however the caller already has it -- see _ir_
        composite_call's own docstring for the same "doesn't care how"
        contract). Unlike _ir_write_composite_value_into, this is a
        TOTAL function: it never returns None, since a type's own zero
        value depends only on the type itself, never on some
        unpredictable runtime expression's own shape -- always fully,
        recursively computable at compile time.

        Dispatches on value_type.kind, mirroring the old-style _gen_
        zero_value_into's own dispatch, structured instead the same
        way this arc's other composite writers already are (per-
        field/per-element, recursing for a nested composite) rather
        than that method's own flattening (_flatten_struct_fields) --
        purely a style choice, both are equally correct, but this
        keeps the shape consistent with _ir_write_struct_literal_into
        right below:
          - SLICE: none's own {0, 0, 0} descriptor, via the exact same
            _ir_write_slice_descriptor_into_address every other slice-
            producing case in this arc already uses -- a zero-value
            slice and a none-valued one are, by design, the identical
            representation.
          - STRUCT: every field, each field's own address computed via
            ordinary IRBinOp (skipped for the first field, offset 0
            needing no addition -- the same shortcut used throughout
            this arc), recursing back into this method for each
            field's own type.
          - ARRAY: delegates to _ir_zero_array_loop -- a genuine
            runtime loop, not per-element unrolling: an array's own
            element count can be large, and the old-style _gen_zero_
            array_into's own choice to always loop, never unroll,
            regardless of size, is worth preserving exactly, not
            silently regressing into `count` separate IRStores.
          - str: the address of a single shared, static empty-string
            constant (_get_empty_str_label) -- never a null pointer,
            for the exact reason _get_empty_str_label's own docstring
            gives (a null zero value would be an active hazard). An
            ordinary IRStaticDataAddress captured into a Temp, the
            same real-IR leaf gen_expr_ir's own StringLiteral case
            already uses.
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
            addr_temp = self.ir_program.ids.new_temp(Type.STR)
            ir = [
                IRStaticDataAddress(dst=addr_temp, label=self._get_empty_str_label()),
                IRStore(address=dst_address, value=addr_temp, value_type=Type.STR)
            ]
            return ir
        return [IRStore(address=dst_address, value=IRConst(0, value_type), value_type=value_type)]

    def _ir_zero_array_loop(self, dst_address, element_type: Type, count: int) -> list:
        """Builds (without lowering) a genuine, bounded-count real-IR
        loop zeroing `count` consecutive elements of element_type,
        starting at dst_address -- the array-leaf counterpart to _ir_
        write_zero_value_into's own scalar/slice/struct cases, all of
        which can be straight-line code since their own size is
        always small and fixed. Mirrors _ir_while_head's own IRLabel/
        IRBranch/IRJump shape exactly, the first
        genuine bounded-iteration loop built as real IR in this arc --
        everything before this has been straight-line or branch-and-
        merge, never actual iteration.

        Each iteration recomputes its own element address from a loop
        counter i (an ordinary, persistent Temp -- reassigned each
        iteration via IRMove into a fresh next-value Temp, the exact
        same shape an ordinary named `i = i + 1` statement already
        compiles to, not a raw in-place IRBinOp), via the identical
        32-bit-multiply-then-64-bit-add shape _ir_index_address's own
        comment already justifies for ordinary indexing (a bounds-
        checked runtime index there; a loop-bounded counter, never
        exceeding `count`, here) -- IRBinOp's own lowering derives
        each operation's width from its LEFT operand with no explicit
        width juggling needed. Then delegates to _ir_write_zero_value_
        into recursively for that one element, which could itself be
        another array (a multi-dimensional zero-init), a struct, or a
        scalar/slice leaf.

        Needs NONE of the old-style _gen_array_struct_zero_loop's own
        careful fixed-register (%r12/%r13) protection across that
        recursive call -- and, unlike the old code's own three,
        separately hand-written sibling loops (flat-zero/str-address/
        struct-recursive, one per leaf shape, each needing its own
        register-collision reasoning), needs only this ONE, because
        _ir_write_zero_value_into already dispatches uniformly on the
        leaf's own type. Every Temp this loop uses (i, the byte
        offset, the per-iteration address) is independent of whatever
        Temps that recursive call allocates for itself, the same "a
        Temp's home is independent of what computed it" property this
        whole arc has relied on repeatedly -- a genuine simplification
        over the old code's own approach, not just a translation of
        it, made possible by Temps replacing fixed registers
        entirely."""
        element_width = type_byte_width(element_type, self.ir_program.struct_registry)
        i = self.ir_program.ids.new_temp(Type.INT)
        start_label = self.ir_program.ids.new_label("zero_array_start")
        body_label = self.ir_program.ids.new_label("zero_array_body")
        end_label = self.ir_program.ids.new_label("zero_array_end")
        cond = self.ir_program.ids.new_temp(Type.BOOL)
        ir = [
            IRMove(dst=i, src=IRConst(0, Type.INT)),
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
        mirroring _ir_zero_array_loop's own shape exactly, just
        comparing instead of zeroing) and STRUCT (per field, NOT
        flattened via _flatten_struct_fields -- matching _ir_write_
        struct_literal_into's own style, not the old-style _gen_
        struct_fields_equality_at_addresses' -- both are equally
        correct, this just keeps the shape consistent with this arc's
        own newer code), reaching str (IRCall(strcmp), reused exactly
        as _ir_string_compare already uses it) and int/bool/int8/
        uint8 (an ordinary IRLoad-into-value_type's-own-width +
        IRBinOp comparison) as its base cases.

        A REAL BUG, found and fixed here rather than carried forward:
        the old-style _gen_struct_fields_equality_at_addresses' own
        scalar-field branch always did a 4-byte compare regardless of
        the field's own declared width, silently reading past an
        int8/uint8 field's own 1-byte storage into whatever garbage
        happened to sit adjacent on the stack -- the exact same class
        of bug the old-style array-equality loop was already found and
        fixed for, for the ARRAY-of-int8 case, just never applied to
        the STRUCT-field case too. Here,
        both go through the identical IRLoad-at-value_type's-own-
        width call, so the width is correct by construction rather
        than needing its own special case: IRLoad already reads at
        dst.type's own declared width universally (see its own
        docstring), int8/uint8 included, the same choke point
        _gen_read_scalar_into's own docstring describes.

        Two Temps' own live ranges spanning arbitrary further real IR
        (nested loops, calls) is exactly the property this whole arc
        has relied on repeatedly to avoid the old-style code's own
        careful, fixed-register (%rbx/%r12/%r13/%r14/%r15) protection
        across recursive comparisons, at any nesting depth -- every
        Temp here gets its own, independent home from the allocator,
        so there is no shared-register hazard to protect against at
        all, unlike the old-style _gen_array_struct_equality_loop,
        whose own docstring used to spend several paragraphs on
        exactly that hazard."""
        if value_type.kind == TypeKind.ARRAY:
            element_type = value_type.element_type
            element_width = type_byte_width(element_type, self.ir_program.struct_registry)
            i = self.ir_program.ids.new_temp(Type.INT)
            start_label = self.ir_program.ids.new_label("eq_array_start")
            body_label = self.ir_program.ids.new_label("eq_array_body")
            end_label = self.ir_program.ids.new_label("eq_array_end")
            cond = self.ir_program.ids.new_temp(Type.BOOL)
            ir = [
                IRMove(dst=i, src=IRConst(0, Type.INT)),
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
            left_val = self.ir_program.ids.new_temp(Type.STR)
            right_val = self.ir_program.ids.new_temp(Type.STR)
            cmp_result = self.ir_program.ids.new_temp(Type.INT)
            mismatch_cond = self.ir_program.ids.new_temp(Type.BOOL)
            return [
                IRLoad(dst=left_val, address=left_addr),
                IRLoad(dst=right_val, address=right_addr),
                IRCall(dst=cmp_result, name='strcmp', args=[left_val, right_val]),
                IRBinOp(dst=mismatch_cond, op=BinaryOp.NOT_EQUAL, left=cmp_result, right=IRConst(0, Type.INT)),
                IRBranch(cond=mismatch_cond, true_label=mismatch_label, false_label=continue_label),
                IRLabel(continue_label),
            ]
        # int, bool, int8, uint8 -- an ordinary, width-aware
        # load-and-compare (see this method's own docstring for why
        # this is the actual bug fix, not just a translation).
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

    def _ir_write_composite_value_into(self, dst_address, value_expr: Node, value_type: Type):
        """The general-purpose dispatcher underlying nested literal
        construction: writes value_expr's own value through
        dst_address (an already-computed, ordinary IRValue), by
        dispatching on value_expr's own shape. Returns None when out
        of scope (a named/partial struct literal, chiefly -- see _ir_
        write_struct_literal_into's own docstring for why that stays
        deferred).

        CRITICAL: the ArrayLiteral case below checks value_type.kind
        == ARRAY explicitly, not just isinstance(value_expr,
        ArrayLiteral) -- the identical bracketed-list AST shape is
        ALSO how a slice literal parses (`[1, 2]` used where a slice
        is expected, e.g. the inner literals of the array-of-slices
        `[2][]int[[1, 2], [3, 4]]`).
        Without this check, value_type.kind == SLICE reaches _ir_
        write_array_literal_into anyway, which silently treats a
        slice's own 24-byte descriptor width as if it were the outer
        array's own element width -- corrupting every subsequent
        element's computed address. A real bug, caught only once
        VarDecl/Assign/IndexAssign/FieldAssign's own wiring (mirroring
        Return's) finally exercised an array-of-slices literal through
        this dispatcher for the first time; slice-typed elements never
        reached here via any of this arc's own earlier test coverage
        before that point.

        A slice-typed ArrayLiteral-shaped value_expr is now real IR
        too, via _ir_slice_literal -- real slice-LITERAL construction
        (as opposed to slice PRODUCTION via `arr[a:b]`, already real IR
        via _ir_slice_into), a genuinely separate piece of work from
        everything else in this method, built and wired in as its own,
        later step. See _ir_slice_literal's own docstring for why it
        takes no value_type parameter at all, unlike this method's own
        ARRAY case just above -- and for two more real bugs (a malloc-
        size one, and a Return-dispatch one) the identical ARRAY-vs-
        SLICE ambiguity this docstring already documents above caused
        during THAT migration too, in two different, more subtle
        forms.

        Unifies, into one place, every composite-producing shape this
        arc has already built SEPARATELY, each for its own original
        call site: an existing value's own address (_ir_copy_into_
        address, originally built for Return/VarDecl/Assign/
        IndexAssign/FieldAssign's own Variable/Field/Index case), a
        Slice production or append call for a slice-typed value (both
        via _ir_write_slice_descriptor_into_address, originally built
        for VarDecl/Assign/IndexAssign/FieldAssign's own slice-
        producing cases), an ordinary composite-returning Call (_ir_
        composite_call, originally built for Return/VarDecl/Assign/
        IndexAssign/FieldAssign's own forwarding case), or nested
        literal construction itself -- recursing back into _ir_write_
        array_literal_into/_ir_write_struct_literal_into, which now
        call back into THIS dispatcher for any of their own composite
        elements/fields in turn. Mutual recursion, the same shape
        address computation's own Field/Index handling already relies
        on elsewhere in this arc (_ir_array_address calling into
        itself, indirectly, through _ir_field_address/_ir_index_
        address, for a chain of arbitrary depth).

        `append` is checked, and handled, BEFORE the generic ordinary-
        Call case below -- append is a Call whose own name is never in
        struct_registry, so without this explicit, earlier check it
        would silently reach _ir_composite_call instead, trying to
        call a function literally named 'append' that was never
        compiled at all. This is the exact same ordering bug already
        found and fixed in gen_statement_ir's own Return/VarDecl/
        Assign/IndexAssign/FieldAssign cases -- worth being explicit
        about here too, rather than risk reintroducing it in a new
        location."""
        if isinstance(value_expr, (Variable, Field, Index)):
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
        if isinstance(value_expr, Call) and value_expr.name in self.ir_program.struct_registry:
            return self._ir_write_struct_literal_into(dst_address, value_expr, value_type)
        if isinstance(value_expr, Call):
            return self._ir_composite_call(dst_address, value_expr, value_type)
        return None

    def _ir_write_array_literal_into(self, dst_address, expr: ArrayLiteral, array_type: Type):
        """Builds (without lowering) an array literal's elements as
        real IR, written through dst_address -- an ordinary INT64-
        typed IRValue, however the caller already has it (see _ir_
        composite_call's own docstring for the same "doesn't care how"
        contract). Returns None when out of scope: only when some
        element's own value is itself out of scope for _ir_write_
        composite_value_into (a named/partial struct literal, chiefly
        -- see its own docstring) -- an element that's ITSELF composite
        (an array of arrays/slices/structs) is no longer automatically
        out of scope, unlike this method's own earlier version: mutual
        recursion through _ir_write_composite_value_into handles it,
        the same shape address computation's own Field/Index handling
        already relies on elsewhere in this arc.

        Each element's own address is dst_address + i*element_width
        via ordinary IRBinOp -- skipped entirely for element 0 (offset
        0 needs no addition, matching _ir_write_slice_descriptor's own
        identical shortcut for its own ptr field). A scalar element's
        value is evaluated via gen_expr_ir (so a migrated sub-
        expression stays real IR) and written via IRStore; a composite
        element delegates entirely to _ir_write_composite_value_into.
        If ANY element turns out to be out of scope, the whole literal
        falls back (this method returns None) -- any IR already built
        for earlier elements (including a few now-orphaned Temp ids
        from _new_temp) is simply discarded by the caller in favor of
        old-style construction for the ENTIRE literal, the same
        harmless-but-pointless waste already accepted elsewhere in
        this arc (a VarDecl's own hidden-pointer leaf, built and then
        discarded, when its own composite-call case turns out not to
        apply) -- there's no risk of a partial, inconsistent write,
        since none of this IR is ever actually emitted in that case."""
        element_type = array_type.element_type
        element_width = type_byte_width(element_type, self.ir_program.struct_registry)
        ir = []
        for i, elem_expr in enumerate(expr.elements):
            if element_type.kind in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT):
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
        anywhere -- nothing ever reads it as a coherent array -- so
        rather than reserving a scratch slot sized to fit it (an array
        literal has no natural upper bound the way a slice's fixed
        24-byte descriptor does), this just evaluates each of the
        literal's directly-written elements for whatever side effects
        it might have (e.g. a function call), discarding every result
        -- like any other bare expression statement.

        Recurses for a nested ArrayLiteral element (a multi-dimensional
        literal used bare), via gen_expr_ir instead of the old-style
        gen_expr_into for a scalar element, so a migrated sub-
        expression (e.g. a call) stays real IR too. An element that's
        itself some other, non-literal array-, slice-, or struct-typed
        expression (a Variable, an indexed sub-array, an array/struct-
        returning Call, ...) is a real, deliberately out-of-scope gap:
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
            if element_type.kind in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT):
                raise CodegenError(
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
        out of scope (see _ir_indexable_base's own docstring) -- moot
        in practice, confirmed exhaustively: every reachable slice-
        typed base shape (Variable/Field/Index/ArrayLiteral/ordinary-
        Call/append/Slice) already succeeds through _ir_indexable_
        base. The caller (gen_expr_ir) raises CodegenError explicitly
        on a None here rather than silently propagating it further up
        the call chain, where it would eventually surface as an
        unrelated TypeError somewhere else entirely.

        Reuses _ir_indexable_base for the slice's own address,
        discarding length/cap -- the same "keep one of three, discard
        the rest" shape _ir_len_call already uses. An ordinary
        IRBinOp (expr.op, ptr, IRConst(0, INT64)) already produces
        exactly the right comparison, with no new IR op needed at
        all: gen_binary_op's own existing dispatch already picks CmpQ
        (64-bit) whenever operand_type is INT64, for ANY comparison
        operator, not just arithmetic ones -- see its own docstring.
        This checks specifically whether the slice's ptr field is
        null (Go's nil-vs-empty-slice distinction), exactly the
        semantics a slice-vs-none comparison needs."""
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
        composite_value_into's own None, for a composite element whose
        own nested content is itself out of scope).

        Scalar element types (int/bool/str/int64) go through an
        ordinary IRStore, exactly as append's own real-IR work always
        has. Composite element types (array/slice/struct) reuse _ir_
        write_composite_value_into completely UNCHANGED -- the
        identical dispatcher every other "write a composite value into
        a known address" site in this arc already uses (VarDecl/
        Assign/IndexAssign/FieldAssign/Return/a struct field/an array
        element), so a slice-typed element correctly writes a fresh
        {ptr, len, cap} descriptor, an array-typed element correctly
        copies element-by-element, and a struct-typed element
        correctly writes field-by-field -- append needed no new
        writing logic of its own at all, only this one, small piece of
        wiring calling into what already existed.

        Called once per branch, not once upfront with the result
        reused in both (the way a purely scalar value's own old
        design could): a composite value has no single Temp of its
        own to compute once and reuse, since WRITING it is the whole
        operation -- there's no separate "the value" to hold onto
        independently of where it gets written. This duplicates the
        value argument's own codegen across both branches (ordinary
        branch-and-merge code growth, the same any if/else already
        accepts when both arms independently compute something), but
        never double-EXECUTES it at runtime: exactly one of the two
        branches ever runs for a given call, decided by the length-
        vs-cap check before either branch's own code is ever reached."""
        if element_type.kind in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT):
            return self._ir_write_composite_value_into(target_addr, value_arg, element_type)
        value_ir, value = self.gen_expr_ir(value_arg)
        return value_ir + [IRStore(address=target_addr, value=value, value_type=element_type)]

    def _ir_append_call(self, expr: Call):
        """Builds (without lowering) append(s, value)'s resulting
        {ptr, len, cap} triple as real IR -- returns (ir, ptr_value,
        len_value, cap_value), or None when out of scope: s's own base
        out of scope (see _ir_indexable_base) -- anything but a
        Variable/Field/Index/Slice/ArrayLiteral/an ordinary composite-
        returning Call, or NoneLiteral -- or `value` itself out of
        scope for _ir_write_append_value_at, when the element type is
        composite (a named/partial struct literal nested inside it,
        chiefly).

        ANY element type is now in scope, including array/slice/
        struct -- see _ir_write_append_value_at's own docstring for
        how writing the value itself works for each. Previously
        excluded entirely (falling back to the old-style, fully
        general gen_append_call_into) purely because the growth path
        (see below) used to fuse growth together with writing the new
        value into one op (the old IRAppendGrow), scoped to a scalar
        element only as a direct consequence.

        The reuse-vs-reallocate decision itself is real IR: an
        ordinary IRBinOp comparison (length >= cap; a plain SIGNED
        comparison is correct here, unlike IRBoundsCheck's own
        unsigned trick -- len/cap are compiler-maintained invariants,
        never a user-supplied value that could be negative) plus an
        ordinary IRBranch, exactly the same shape every other
        conditional decision in gen_expr_ir/gen_statement_ir already
        uses. The REUSE path (length < cap) is entirely real IR:
        compute the target address, write the value, increment length.
        The REALLOCATE path (length >= cap) grows first via IRSliceGrow
        -- see its own docstring for why growth and writing the value
        are two separate steps here, not fused into one op the way the
        old, scalar-only IRAppendGrow used to -- THEN writes the value
        into the newly-available slot at the same, already-known
        offset (length * element_width) within the fresh backing.

        Both paths write into the SAME three result Temps (result_ptr/
        result_len/result_cap), then jump to a shared end label -- the
        same "both branches assign the same Temp" pattern an if/else
        already uses for a named-local variable, just for an anonymous
        one here."""
        slice_arg, value_arg = expr.args
        slice_type = type_of(slice_arg)
        element_type = slice_type.element_type
        element_width = type_byte_width(element_type, self.ir_program.struct_registry)

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

        Reuses _ir_indexable_base directly, the same "address plus
        length, however each is represented" abstraction indexing and
        slicing already share. x's own address (and
        cap) are computed and then discarded, but NOT skipped: x is
        still fully evaluated regardless, so any bounds check or
        side effect buried in it genuinely runs -- deliberately: a
        length-only read must not silently skip a bounds check or
        side effect the full expression would otherwise trigger
        (`len(arr[i])` still aborts if i is out of range).

        For an ARRAY base, the returned value is a compile-time
        IRConst (x's declared size, never read out of x at runtime);
        for a SLICE base, an ordinary INT Temp holding a runtime
        value read from x's own descriptor -- either way, exactly
        _ir_indexable_base's own second return value, unchanged."""
        arg = expr.args[0]
        base = self._ir_indexable_base(arg)
        if base is None:
            return None
        base_ir, ptr, length, cap = base
        return base_ir, length

