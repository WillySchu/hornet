"""A str value is a 16-byte {ptr, len} descriptor -- ptr at offset 0,
len (an ordinary 32-bit INT, in its own 8-byte slot -- same convention
slice's own len/cap already use, see _ir_read_slice_descriptor_from_
address) at offset 8. Immutable: ptr always points at bytes nobody
ever writes back into, whether a literal's static address or a fresh
malloc'd buffer built once, by concatenation, and never touched again.
No null terminator anywhere in this scheme, and none relied upon: an
embedded '\\0' byte is ordinary string content, ends nothing, exactly
as long-lived as every other byte around it.

Structurally str is much closer to slice than to struct/array: a
small, FIXED-size descriptor (16 bytes, always) whose own two fields
point at and measure a separately-sized payload elsewhere, rather than
a type whose own size varies with, and directly holds, its full
content. So str reuses slice's own conventions throughout -- a
COMPOSITE kind (see ir/utils.py's own COMPOSITE_KINDS), addressed via
_ir_str_address exactly like _ir_slice_address, read/written through
that address via _ir_read_str_descriptor_from_address/_ir_write_str_
descriptor_into_address exactly like slice's own triple, returned
through the hidden-return-pointer convention exactly like slice -- NOT
struct/array's own deep, whole-value copy: a str "copy" is always just
copying its own 16-byte descriptor, never the bytes it points at,
identical in spirit to how copying a slice never copies the array it
aliases.

No ownership tracking of any kind: a concatenation's own malloc'd
buffer, and every buffer a string literal's own bytes already live in
statically, are never freed. Deliberate -- see this feature's own
design discussion for why (the previous concat-only free heuristic
predated real escape analysis, would already be unsound once strings
could alias into a shared buffer via slicing, and is fundamentally
incompatible with a future tracing GC regardless); strings leak in
this scheme, by design, until real memory management exists.

Also builds the runtime type-descriptor tree print() needs
(_get_or_build_type_descriptor) and print()'s own real-IR call itself
(_ir_print_call): the descriptor is walked, and stringification
happens, entirely in C, by the runtime's own hornet_stringify/hornet_
print (see runtime/runtime.c) -- this file just builds the IRCall
against hornet_print and the static descriptor data it reads."""

from ir.errors import IRError
from ir.ir import IRBinOp, IRBoundsCheck, IRBranch, IRConst, IRCall, IRJump, IRLabel, IRSliceBoundsCheck, IRStaticDataAddress, IRLocalAddress, IRLoad, IRMove, IRStore
from ir.utils import type_byte_width, type_of
from parser import Call, Binary, Field, Index, Node, Slice, StringLiteral, Unary, UnaryOp, Variable, BinaryOp
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
_TYPEDESC_SUM = 9
_TYPEDESC_POINTER = 10


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
        elif t.kind == TypeKind.SUM:
            # Descriptor shape: [tag, variant_count, variant_desc_ptr,
            # variant_desc_ptr, ...] -- deliberately no name field, and
            # no per-variant name either (unlike a struct field, which
            # names itself): a sum-typed value prints EXACTLY as
            # whatever its active variant would on its own (`Circle
            # (radius: 5)`, not `Shape(Circle(radius: 5))`) -- see
            # hornet_stringify's own SUM case for why this makes the
            # variant list a plain array of pointers, indexed directly
            # by the runtime discriminant, rather than the (name,
            # type, offset) triples a struct's own fields need: a
            # variant has no name or offset of its own to record here
            # at all, just a type to recurse into once the tag picks
            # which one. Variant order matters -- it's what the
            # discriminant already indexes by (see SumTypeDef's own
            # docstring in parser.py), so sum_type_registry's own
            # preserved order is used unchanged, not re-sorted or
            # deduplicated here.
            sum_type_info = self.ir_program.sum_type_registry[t.sum_type_name]
            variant_desc_labels = [
                self._get_or_build_type_descriptor(Type(TypeKind.STRUCT, struct_name=variant_name), in_progress)
                for variant_name in sum_type_info.variants
            ]
            self.ir_program.type_descriptors.append((label, [_TYPEDESC_SUM, len(variant_desc_labels)] + variant_desc_labels))
        elif t.kind == TypeKind.POINTER:
            # Descriptor shape: just [tag] -- no name field, and no
            # pointee-type field either, unlike ARRAY/SLICE's own
            # elem_label: print(p) prints p's own raw address (see
            # UnaryOp.DEREFERENCE's/hornet_stringify's own POINTER
            # case), never recurses into what it points at, so there's
            # nothing here to build a descriptor FOR beyond the tag
            # itself -- the same "no extra fields" shape INT/BOOL/STR
            # already have, for the identical reason: all four print
            # as one raw value, not a structured container.
            self.ir_program.type_descriptors.append((label, [_TYPEDESC_POINTER]))
        else:
            raise IRError(f"No type descriptor rule for: {t}")

        return label

    def _ir_str_address(self, expr: Node):
        """Builds (without lowering) the address of a str-typed expr's
        own 16-byte {ptr, len} descriptor -- Variable, Field, Index,
        or a pointer dereference -- as real IR. Returns (ir, address),
        or None for a shape with no existing address at all (a
        StringLiteral, a Binary(ADD) concatenation, an ordinary str-
        returning Call): see _ir_str_value for those instead, which
        this method's own Field/Index callers (_ir_field_address/_ir_
        index_address, already generic over the result's own type)
        fall back to when their own base isn't itself addressable.

        The Variable case DOES still need a heap-allocation check,
        unlike this file's earlier assumption that str, like slice,
        is simply never individually heap-promoted: that reasoning
        holds for slice only because a slice's own address can never
        actually be taken today (no `&mySlice` exists), so the
        question never arises in practice -- but `&s`, for a str-
        typed s, was ALREADY legal before this arc (str was scalar
        then, and check_unary's own ADDRESS_OF restriction allows any
        bare Variable regardless of type), so a str local's address
        genuinely can still escape past this function, exactly the
        way an int/bool/pointer local's already can (see codegen/
        escape_analysis.py). Mirrors _ir_struct_address's own Variable
        case for this reason -- minus its sum-type narrowing branch,
        since str can never itself be sum-typed, so there is nothing
        to narrow."""
        if isinstance(expr, Variable):
            slot = self._local_slot(expr.name)
            slot_addr = self.ir_program.ids.new_temp(Type.INT64)
            ir = [IRLocalAddress(dst=slot_addr, slot=slot)]
            if self._is_heap_allocated(self._local_decl_id(expr.name), self._local_type(expr.name)):
                addr_temp = self.ir_program.ids.new_temp(Type.INT64)
                ir.append(IRLoad(dst=addr_temp, address=slot_addr))
                return ir, addr_temp
            return ir, slot_addr
        if isinstance(expr, Field):
            return self._ir_field_address(expr)
        if isinstance(expr, Index):
            return self._ir_index_address(expr)
        if isinstance(expr, Unary) and expr.op == UnaryOp.DEREFERENCE:
            # `*p` (p: *str) read as a whole str value -- the
            # pointer's own value already IS the address of its own
            # pointee's descriptor, the same principle _ir_struct_
            # address's own identical case already uses: an ordinary
            # scalar read, via gen_expr_ir, since a pointer is just an
            # 8-byte scalar for every purpose other than what it
            # points at.
            return self.gen_expr_ir(expr.operand)
        return None

    def _ir_read_str_descriptor_from_address(self, descriptor_addr) -> tuple:
        """Builds (without lowering) the {ptr, len} pair read out of
        an ALREADY-computed str-descriptor address as real IR --
        returns (ir, ptr_value, len_value). Exactly _ir_read_slice_
        descriptor_from_address's own shape, minus the cap field: ptr
        read directly off the descriptor's own address, len at a
        fixed +8 offset from it, via an ordinary IRBinOp before its
        own IRLoad. len is captured as an INT (32-bit) Temp -- same
        convention slice's own len/cap already use, so a length never
        needs more than 32 bits of range here either."""
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
        """The shared core of _ir_write_str_descriptor: given an
        ALREADY-computed destination address, writes an already-
        produced {ptr, len} pair there at offsets 0/8, via two
        ordinary IRStores -- exactly _ir_write_slice_descriptor_into_
        address's own shape, minus the cap field."""
        len_addr = self.ir_program.ids.new_temp(Type.INT64)
        return [
            IRStore(address=dst_address, value=ptr_value, value_type=Type.INT64),
            IRBinOp(dst=len_addr, op=BinaryOp.ADD, left=dst_address, right=IRConst(8, Type.INT64)),
            IRStore(address=len_addr, value=len_value, value_type=Type.INT),
        ]

    def _ir_write_str_descriptor(self, dst_expr: Node, ptr_value, len_value) -> list:
        """Writes an already-produced {ptr, len} pair through dst_
        expr's own address -- always via _ir_str_address, since the
        destination is always str-typed here -- then delegates to
        _ir_write_str_descriptor_into_address for the rest. Exactly
        _ir_write_slice_descriptor's own shape."""
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
        """Builds (without lowering) str's own zero value, {ptr=0,
        len=0}, as real IR -- returns (ir, ptr_value, len_value).
        Unlike the OLD, pre-this-arc C-string representation (a
        shared static empty-string constant, since every operation
        dereferenced ptr unconditionally), ptr=0 is safe here: len=0
        means no operation this file generates -- concatenation,
        comparison, print -- ever reads even one byte through ptr, so
        there is nothing left for a null ptr to be dangerous FOR. This
        also means the zero value needs no static data of its own at
        all, unlike before."""
        ptr = self.ir_program.ids.new_temp(Type.INT64)
        length = self.ir_program.ids.new_temp(Type.INT)
        ir = [
            IRMove(dst=ptr, src=IRConst(0, Type.INT64)),
            IRMove(dst=length, src=IRConst(0, Type.INT)),
        ]
        return ir, ptr, length

    def _ir_str_value(self, expr: Node) -> tuple:
        """Builds (without lowering) any str-typed expr's own {ptr,
        len} pair as real IR -- returns (ir, ptr_value, len_value).
        The str counterpart to _ir_slice_arg/_ir_indexable_base:
        every caller that needs a str's own actual content (concat-
        enation, comparison, print, an argument or return value, a
        struct-literal field, an array-literal element) goes through
        here rather than gen_expr_ir, exactly as every slice-aware
        caller already goes through _ir_slice_arg rather than gen_
        expr_ir -- see this file's own module docstring, and ir/
        dispatch.py's own gen_expr_ir docstring, for why str, like
        slice, never reaches that method at all.

        A StringLiteral needs no runtime work whatsoever: ptr is a
        compile-time-known static address (IRStaticDataAddress) and
        len is a compile-time-known constant (len(expr.value) --
        ASCII-width bytes, matching escape_for_asciz's own assumption
        elsewhere in this compiler), exactly as cheap as a bare
        Constant already is in gen_expr_ir.

        A Variable/Field/Index/pointer-dereference (an EXISTING value
        with a real address) reads its pair straight off that address
        via _ir_str_address + _ir_read_str_descriptor_from_address.

        A Binary(ADD) (concatenation) delegates to _ir_string_concat.
        A Slice (`s[low:high]`) delegates to _ir_str_slice_into.
        An ordinary str-returning Call materializes through the
        hidden-return-pointer convention (_ir_materialize_composite_
        call, the SAME helper every other composite-returning Call
        already uses -- generic over value_type, so str needs no
        Call-specific machinery of its own here at all) and reads the
        result back the same way an existing value's address would be.

        Returns None when expr's own shape is out of scope (a bare
        `none`, which str can never be -- semantic.py already rejects
        that outright)."""
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
            # from_raw_parts -- see IntrinsicDecl's own docstring in
            # parser.py for the whole mechanism. Not an ordinary call
            # at all: its own two arguments (a *byte, an int --
            # already confirmed by semantic.py's own signature
            # checking) already ARE a str's own {ptr, len} pair, by
            # definition -- there's nothing to materialize or read
            # back through a hidden return pointer the way an
            # ordinary str-returning Call needs (the branch just
            # below this one), just the two argument values evaluated
            # and handed back directly as ptr_value/len_value.
            ptr_ir, ptr_value = self.gen_expr_ir(expr.args[0])
            len_ir, len_value = self.gen_expr_ir(expr.args[1])
            return ptr_ir + len_ir, ptr_value, len_value
        result = self._ir_str_address(expr)
        if result is not None:
            addr_ir, addr_value = result
            read_ir, ptr_value, len_value = self._ir_read_str_descriptor_from_address(addr_value)
            return addr_ir + read_ir, ptr_value, len_value
        if isinstance(expr, Call) and expr.name in self.ir_program.struct_registry:
            return None  # a str-literal Call can never exist -- str has no literal-construction syntax of its own
        if isinstance(expr, Call):
            addr_ir, addr_value = self._ir_materialize_composite_call(expr, Type.STR)
            read_ir, ptr_value, len_value = self._ir_read_str_descriptor_from_address(addr_value)
            return addr_ir + read_ir, ptr_value, len_value
        return None

    def _ir_str_slice_into(self, expr: Slice):
        """Builds (without lowering) expr.array[expr.low:expr.high]'s
        resulting {ptr, len} pair (expr.array itself str-typed) as
        real IR -- returns (ir, ptr_value, len_value), or None when
        expr.array's own base is out of scope.

        Closely mirrors _ir_slice_into's own shape (ir/arrays_slices.
        py), with the two differences str's own representation forces:
        no cap at all -- str is immutable and never grows, so there's
        no separate capacity to expose or bounds-check against; high
        defaults to, and bounds-checks against, len itself, not a
        wider cap the way slice's own high does -- and element_stride
        is always 1, str's own "elements" being individual bytes with
        no wider declared width to compute.

        No stack-safety concern here the way _ir_slice_into's own
        aliasing raises for an array/slice base (see Slice's own
        docstring in parser.py, and semantic.py's check_slice, for the
        fuller explanation): the resulting str's own ptr always still
        points at whatever expr.array's own ptr already did -- static
        data or a heap buffer, never the stack -- so this never needs
        escape analysis's own involvement at all.

        low_value/high_value/length_value are immutable once computed
        -- the one IRBinOp below just reads them -- so no push/pop
        protection is needed anywhere: every intermediate value
        already has its own Temp home, safely written before whatever
        evaluates next."""
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
        """Builds (without lowering) expr.array[expr.index]'s
        resulting byte (expr.array itself str-typed) as real IR --
        returns (ir, value), or None when expr.array's own base is
        out of scope.

        Closely mirrors _ir_index_address's own shape (ir/
        arrays_slices.py), but produces the byte's own VALUE directly
        (an IRLoad at UINT8 width, dst.type's own width deciding that
        -- see IRLoad's own docstring) rather than just its address:
        str indexing's whole result IS that one byte, with no wider
        caller here that still needs an address to do something else
        with, the way an ordinary array-element read or an Index-
        assignment target both do. element_stride is always 1, the
        same reasoning _ir_str_slice_into's own docstring already
        gives for high/low: str's own "elements" are individual
        bytes, with no wider declared width to multiply by.

        Reuses IRBoundsCheck (index >= length) directly, NOT
        IRSliceBoundsCheck: this is ordinary INDEXING, not slicing, so
        the ordinary indexing check -- and its own "array index out
        of bounds" message -- applies here exactly like it already
        does for an array/slice element read, not slicing's own
        different comparison and message (see IRSliceBoundsCheck's
        own docstring for why the two are deliberately separate ops
        rather than one, mode-flagged version)."""
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
        """Builds (without lowering) print(x)'s own real IR -- returns
        (ir, None), a void call.

        Unlike an ordinary call, which passes a scalar BY VALUE and a
        composite by address, this always computes an ADDRESS for x,
        regardless of its own type: hornet_print's signature is
        uniformly `void hornet_print(void *value_addr, const unsigned
        char *type_desc)`, and hornet_stringify dereferences that
        address at whatever width its own type descriptor says to.

        A struct-literal x (`print(Circle(5))`) is checked FIRST,
        before the general ARRAY/STRUCT/SUM branch below: _ir_
        composite_operand_address's own dispatch never covers a bare
        struct literal at all (see its own docstring -- it's built for
        Binary equality too, where a struct literal is deliberately
        NOT a valid operand, so it can't just be added there without
        wrongly opening that door as well) and would return None for
        one, same as it would for print's own array/sum branches below
        if either could ever receive an analogous literal shape (an
        ArrayLiteral is handled inside _ir_composite_operand_address
        itself; a bare, unwidened sum-typed literal doesn't exist as
        its own AST shape at all). _ir_materialize_struct_literal is
        the SAME helper &Circle(5) (ir/pointers.py) and a struct-
        literal ExprStmt (ir/statements.py) already use for exactly
        this shape -- print's own argument needs nothing new, just
        this one extra dispatch case recognizing it applies here too.

        ARRAY/STRUCT/SUM-typed x otherwise: reuses _ir_composite_
        operand_address unchanged -- a SUM-typed value's own address
        is exactly what that function already computes generically
        for a Variable/Field/Index or an ordinary composite-returning
        Call (see its own docstring), same as a struct's. STR-typed x:
        _ir_str_value (ir/strings.py) unifies every reachable shape
        into one {ptr, len} pair, written into the shared,
        unconditionally-reserved 16-byte _unnamed_str_temp_slot
        scratch slot, whose address is then taken -- exactly SLICE's
        own treatment just below, minus the cap field. SLICE-typed x:
        _ir_slice_arg unifies every reachable shape into one {ptr,
        len, cap} triple, written into the shared, unconditionally-
        reserved 24-byte _unnamed_slice_temp_slot scratch slot, whose
        address is then taken. Otherwise a scalar: computed via gen_
        expr_ir, written into the shared 8-byte _print_scalar_temp_
        slot scratch slot -- a scalar has no other "address of this
        value" concept in real IR, since it only ever needs to live in
        a Temp; this is a deliberate, narrow exception, since hornet_
        print is a genuine C-ABI boundary this compiler's own output
        has to cross."""
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
        IR -- returns (ir, ptr_value, len_value), the same {ptr, len}
        shape every other _ir_str_value branch returns (see its own
        docstring -- this is one of them, dispatched to directly
        rather than duplicated).

        Unlike the OLD, strlen-based version, both lengths are already
        KNOWN values here, never something to scan for: left/right
        are each read via _ir_str_value, so left_len/right_len are
        already sitting in their own Temps by the time this needs
        them, with no strlen call at all. total_len = left's own
        length + right's; malloc that many bytes (no "+1" for a
        trailing null -- nothing downstream of this scheme ever
        expects or reads one); memcpy left's own bytes in first, then
        right's own bytes starting right after them.

        left/right are each evaluated via _ir_str_value BEFORE either
        memcpy runs, so left_ptr/left_len are already sitting safely
        in their own Temp homes by the time right is evaluated,
        regardless of what that does internally -- no "protect left
        across evaluating right" dance needed, the same guarantee
        gen_expr_ir's own ordering already gave the OLD version.

        No freeing of any kind, for either operand or the fresh buffer
        this itself produces -- see this file's own module docstring
        for why: strings leak in this scheme, by design, until real
        memory management exists."""
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
        """Builds (without lowering) `left == right` / `left !=
        right` (both str) as real IR -- returns (ir, value). Length-
        first, not byte-first: two strings of different lengths can
        never be equal, and -- unlike the OLD strcmp-based version,
        which relied on a null terminator to know where to stop
        regardless of either buffer's own true size -- memcmp(left,
        right, left_len) once lengths are already known equal is safe
        exactly because they're equal; calling it BEFORE checking that
        would risk reading past the end of a genuinely shorter right
        buffer, an out-of-bounds read this scheme cannot risk.
        lengths_differ_label lets a mismatch skip memcmp entirely,
        both for safety and as a cheap, common-case optimization.

        t_result is written on BOTH paths (the lengths-differ short
        circuit, and the memcmp path taken once they're already known
        equal), each followed by a jump to the same end_label -- the
        identical register-merge technique _ir_short_circuit already
        uses for AND/OR, adapted here to a two-outcome comparison
        rather than a boolean short-circuit.

        A length mismatch's own outcome (mismatch_result) is decided
        here, in Python, from expr.op itself -- a compile-time-known
        constant, not a runtime comparison of it: two unequal-length
        strings are always NOT_EQUAL and never EQUAL, whichever
        operator this expression actually uses. The equal-lengths
        path's own result reuses expr.op directly against memcmp's
        own 0/nonzero result, via an ordinary IRBinOp -- exactly the
        OLD version's own final line, unchanged, since EQUAL/NOT_EQUAL
        both already fall out of that one comparison correctly."""
        left_ir, left_ptr, left_len = self._ir_str_value(expr.left)
        right_ir, right_ptr, right_len = self._ir_str_value(expr.right)

        lengths_equal = self.ir_program.ids.new_temp(Type.BOOL)
        cmp_result = self.ir_program.ids.new_temp(Type.INT)
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
            IRBinOp(dst=t_result, op=expr.op, left=cmp_result, right=IRConst(0, Type.INT)),
            IRJump(end_label),
            IRLabel(lengths_differ_label),
            IRMove(dst=t_result, src=IRConst(mismatch_result, Type.BOOL)),
            IRJump(end_label),
            IRLabel(end_label),
        ]
        return ir, t_result

