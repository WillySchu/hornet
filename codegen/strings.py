"""A str value is a plain pointer -- to a literal's static address, or
a concatenation's malloc'd buffer -- never copied for its bytes the
way an array or struct is. Also builds the runtime type-descriptor
tree print() needs (_get_or_build_type_descriptor) and print()'s own
real-IR call itself (_ir_print_call): the descriptor is walked, and
the actual stringification happens, entirely in C now, by the
runtime's own hornet_stringify/hornet_print (see runtime/runtime.c) --
this file just builds the IRCall against hornet_print and the static
descriptor data it reads. Used to also hand-build hornet_stringify
itself, and the whole growable-buffer-append machinery underneath it,
as literal x86-64 assembly directly in this file (build_
stringify_function and its own helpers) -- removed entirely once
print() migrated to the real runtime (see this file's own git history
if that's ever useful, and ir.py's own top docstring for the broader
old-style dead-code cleanup this was folded into)."""

from codegen.errors import CodegenError
from codegen.utils import type_byte_width, type_of
from codegen.ir import IRBinOp, IRConst, IRCall, IRStaticDataAddress, IRLocalAddress, IRStore
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
        and registering it into self.type_descriptors if it hasn't
        already been built WITHIN THIS ONE CALL (in_progress, keyed on
        t -- Type is frozen, so it's already a safe, correct dict key).

        `in_progress` is scoped to a single top-level call (one fresh
        dict per print() call site that needs a descriptor tree), not
        a whole-program cache -- two print() calls on the same struct
        type each build their own tree from scratch, matching gen_
        expr_ir's own StringLiteral case's identical "no cross-
        occurrence dedup, keep it simple" choice.

        Reuse WITHIN one call isn't optional, though: it's the only
        way a self-referential struct (`struct Node: int value; []Node
        children`) can be represented as a finite amount of static
        data at all. Reserving this type's label BEFORE recursing into
        anything it contains is what breaks that cycle -- a nested
        reference back to the same type finds its label already in
        in_progress and reuses it.

        Every non-leaf kind (ARRAY, SLICE, STRUCT) carries its own
        type-name string (e.g. "[3]int", "Point") as a second field
        right after the kind tag; hornet_stringify prints this
        immediately before that kind's opening bracket/brace at EVERY
        level it appears, not just the outermost value a print() call
        names directly. INT/INT8/UINT8/BOOL/STR carry no name field at
        all -- a genuine per-kind layout difference, not a uniform
        field every descriptor has.
        """
        if t in in_progress:
            return in_progress[t]
        label = self.new_label("typedesc")
        in_progress[t] = label

        if t.kind == TypeKind.INT:
            self.type_descriptors.append((label, [_TYPEDESC_INT]))
        elif t == Type.INT8:
            self.type_descriptors.append((label, [_TYPEDESC_INT8]))
        elif t == Type.UINT8:
            self.type_descriptors.append((label, [_TYPEDESC_UINT8]))
        elif t == Type.INT64:
            self.type_descriptors.append((label, [_TYPEDESC_INT64]))
        elif t.kind == TypeKind.BOOL:
            self.type_descriptors.append((label, [_TYPEDESC_BOOL]))
        elif t.kind == TypeKind.STR:
            self.type_descriptors.append((label, [_TYPEDESC_STR]))
        elif t.kind == TypeKind.ARRAY:
            name_label = self.new_label("typedesc_name")
            self.string_literals.append((name_label, str(t)))
            elem_label = self._get_or_build_type_descriptor(t.element_type, in_progress)
            elem_width = type_byte_width(t.element_type, self.struct_registry)
            self.type_descriptors.append((label, [_TYPEDESC_ARRAY, name_label, elem_label, t.size, elem_width]))
        elif t.kind == TypeKind.SLICE:
            name_label = self.new_label("typedesc_name")
            self.string_literals.append((name_label, str(t)))
            elem_label = self._get_or_build_type_descriptor(t.element_type, in_progress)
            elem_width = type_byte_width(t.element_type, self.struct_registry)
            self.type_descriptors.append((label, [_TYPEDESC_SLICE, name_label, elem_label, elem_width]))
        elif t.kind == TypeKind.STRUCT:
            name_label = self.new_label("typedesc_name")
            self.string_literals.append((name_label, str(t)))
            struct_info = self.struct_registry[t.struct_name]
            field_fields: list = []
            field_count = 0
            for field_name, field_type in struct_info.fields.items():
                field_name_label = self.new_label("typedesc_fname")
                self.string_literals.append((field_name_label, field_name))
                field_type_label = self._get_or_build_type_descriptor(field_type, in_progress)
                field_offset = self._field_offset(t.struct_name, field_name)
                field_fields.extend([field_name_label, field_type_label, field_offset])
                field_count += 1
            self.type_descriptors.append((label, [_TYPEDESC_STRUCT, name_label, field_count] + field_fields))
        else:
            raise CodegenError(f"No type descriptor rule for: {t}")

        return label





    def _get_empty_str_label(self) -> str:
        """A shared, static, empty ("") string constant -- str's zero
        value. Deliberately NOT a null pointer: every string operation
        this file generates dereferences a str value with no null
        check, so a zero-initialized str has to point at a real, valid
        (if empty) C string -- a null pointer would segfault the
        instant anything touched it."""
        if self._empty_str_label is None:
            self._empty_str_label = self.new_label("empty_str")
            self.string_literals.append((self._empty_str_label, ""))
        return self._empty_str_label






    def _ir_print_call(self, expr: Call):
        """Builds (without lowering) print(x)'s own real IR -- returns
        (ir, None), a void call, matching _ir_call's own contract for
        one.

        Unlike an ordinary function call (_ir_call/_ir_call_arguments),
        which passes a scalar BY VALUE and a composite by address,
        this always computes an ADDRESS for x, regardless of its own
        type: hornet_print's own signature is uniformly `void
        hornet_print(void *value_addr, const unsigned char
        *type_desc)`, and hornet_stringify (called internally, not by
        this compiler at all anymore) dereferences that address at
        whatever width its own type descriptor says to. This is why
        print needs its own dedicated entry point rather than
        routing through _ir_call the way an ordinary call does --
        the same reason _ir_len_call has always needed one, and
        exactly why gen_expr_ir's own dispatch has always excluded
        'print' from the ordinary-Call case (`expr.name not in
        ('print', 'len')`), even before this method existed to fill
        that gap with real IR.

        ARRAY/STRUCT-typed x: reuses _ir_composite_operand_address
        completely unchanged -- confirmed directly that the shapes it
        already covers (Variable/Field/Index, a bare ArrayLiteral,
        an ordinary composite-returning Call) are EXACTLY what
        semantic.py restricts print's own argument to as well: a
        struct-literal Call (`print(Point(1, 2))`), the one shape
        that method doesn't cover, is rejected as print's own
        argument the same way it's rejected as an equality operand;
        an ordinary composite-returning Call (`print(makePoint())`)
        is allowed in both places. Raises CodegenError, via that
        method's own contract, on the None it would otherwise return
        for a shape genuinely out of scope -- moot in practice,
        for the identical reason it's already moot at every other
        caller of that method.

        SLICE-typed x: _ir_slice_arg already unifies every reachable
        slice-typed shape -- an existing Variable/Field/Index, or a
        freshly-produced value (a Slice production, append, an
        ordinary Call) -- into one {ptr, len, cap} triple. Rather than
        separately handling "already has an address" (Variable/Field/
        Index, via _ir_slice_address) from "needs one materialized"
        (everything else), this always writes that triple into the
        SAME shared, unconditionally-reserved 24-byte
        _unnamed_slice_temp_offset scratch slot (reserved
        unconditionally for every function, in gen_function_ir -- see
        its own comment) and takes THAT slot's own address -- one path
        for every shape, not two.

        Otherwise, a scalar (int/bool/str/int8/uint8/int64): the one
        shape with no existing "address of this value" concept in
        real IR at all, since a scalar has only ever needed to live in
        a Temp before now, never at a durable address. Computes the
        value via gen_expr_ir, writes it into the existing,
        unconditionally-reserved 8-byte _print_scalar_temp_offset
        scratch slot (reserved unconditionally for every function, in
        gen_function_ir -- see its own comment), and takes that slot's
        own address. A deliberate, narrowly-scoped exception to
        keeping IR conceptual rather than physical: there's no way
        around a real address here, since hornet_print is a genuine
        C-ABI boundary this compiler's own output has to cross.

        The type descriptor lookup/build itself (_get_or_build_type_
        descriptor) is unchanged -- already a pure, compile-time
        operation, feeding this new call site exactly as it always fed
        the old one."""
        arg = expr.args[0]
        arg_type = type_of(arg)

        if arg_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT):
            result = self._ir_composite_operand_address(arg, arg_type)
            if result is None:
                raise CodegenError(
                    f"_ir_composite_operand_address returned None for print()'s own "
                    f"ARRAY/STRUCT-typed argument ({arg!r}) -- expected to always "
                    f"succeed, since semantic.py already restricts print's own "
                    f"argument to exactly the shapes that method covers"
                )
            value_addr_ir, value_addr = result
        elif arg_type.kind == TypeKind.SLICE:
            slice_ir, ptr_value, len_value, cap_value = self._ir_slice_arg(arg)
            slice_addr = self._new_temp(Type.INT64)
            value_addr_ir = slice_ir + [IRLocalAddress(dst=slice_addr, offset=self._unnamed_slice_temp_offset)]
            value_addr_ir.extend(
                self._ir_write_slice_descriptor_into_address(slice_addr, ptr_value, len_value, cap_value))
            value_addr = slice_addr
        else:
            expr_ir, value = self.gen_expr_ir(arg)
            scalar_addr = self._new_temp(Type.INT64)
            value_addr_ir = expr_ir + [IRLocalAddress(dst=scalar_addr, offset=self._print_scalar_temp_offset)]
            value_addr_ir.append(IRStore(address=scalar_addr, value=value, value_type=arg_type))
            value_addr = scalar_addr

        desc_label = self._get_or_build_type_descriptor(arg_type, {})
        desc_addr = self._new_temp(Type.INT64)
        desc_ir = [IRStaticDataAddress(dst=desc_addr, label=desc_label)]

        call_ir = [IRCall(dst=None, name='hornet_print', args=[value_addr, desc_addr])]
        return value_addr_ir + desc_ir + call_ir, None



    def _ir_string_concat(self, expr: Binary):
        """Builds (without lowering) `left + right` (both str) as real
        IR -- returns (ir, value). Reuses ordinary IRCall directly for
        strlen/malloc/strcpy/strcat: IRCall's own lowering is already
        fully generic (just `CallInstr(instr.name)`), with no notion
        of "Hornet function" baked in, so an external C library call
        needs no new IR concept at all, unlike append's own
        IRSliceGrow -- this is real IR entirely from EXISTING pieces
        (IRCall, IRBinOp), composed.

        left/right are each evaluated via gen_expr_ir now (a migrated
        sub-expression -- another concatenation, a scalar Call --
        stays real IR instead of being immediately, separately
        lowered), rather than the old-style version's own gen_expr_
        into. This is also what makes the old-style version's own
        careful "protect left across evaluating right" stack dance
        unnecessary here: left_value is already sitting safely in its
        own Temp home (register or memory, via the allocator) the
        moment gen_expr_ir(expr.left) returns, regardless of what
        evaluating right does internally -- the same "a Temp's home is
        independent of what computed it" property this whole arc has
        relied on repeatedly (IRCopy's own two-address capture,
        _ir_index_address's own protection-free design, ...).

        MEMORY: _ir_free_if_fresh_concat replicates gen_string_concat_
        into's own AST-shape check exactly (see its own docstring for
        why this narrow check is safe with no broader escape
        analysis) -- called once left's bytes are copied (strcpy) and
        once right's are (strcat), matching the old-style ordering.
        No stash-before-free dance is needed here either, for the
        identical Temp-independence reason: the result buffer's own
        Temp isn't touched by an intervening free() call the way the
        old code's own fixed %eax would be."""
        left_ir, left_value = self.gen_expr_ir(expr.left)
        right_ir, right_value = self.gen_expr_ir(expr.right)

        len_left = self._new_temp(Type.INT64)
        len_right = self._new_temp(Type.INT64)
        total_len = self._new_temp(Type.INT64)
        plus_one = self._new_temp(Type.INT64)
        new_buf = self._new_temp(Type.STR)

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
        *this* expression alone, which could never have been stored
        into a variable, returned, or passed as an argument -- frees
        it, via an ordinary IRCall(free). Everything else is left
        alone: a StringLiteral points into static `.data` and was
        never heap-allocated (freeing it would corrupt the allocator);
        a Variable or a Call's return value might be aliased by code
        we have no visibility into here -- telling those apart from a
        genuinely fresh, exclusively-owned buffer is a real escape-
        analysis problem this narrow check deliberately doesn't
        attempt to solve."""
        if isinstance(operand, Binary) and operand.op == BinaryOp.ADD:
            return [IRCall(dst=None, name='free', args=[value])]
        return []

    def _ir_string_compare(self, expr: Binary):
        """Builds (without lowering) `left == right` / `left != right`
        (both str) as real IR -- returns (ir, value). Same shape as
        _ir_string_concat: ordinary IRCall(strcmp) plus an ordinary
        IRBinOp for the 0/1 bool conversion, reusing
        _COMPARISON_CONDITION_CODES[op] indirectly via IRBinOp's own
        existing comparison lowering rather than reimplementing the
        cmp/SetCC/MovZX sequence here.

        No stash-before-free dance needed here either, for the same
        Temp-independence reason _ir_string_concat's own docstring
        gives: strcmp's own result is already safely captured into
        cmp_result before either free() call ever runs, so freeing a
        fresh operand can't clobber it the way it could the old code's
        own fixed %eax."""
        left_ir, left_value = self.gen_expr_ir(expr.left)
        right_ir, right_value = self.gen_expr_ir(expr.right)

        cmp_result = self._new_temp(Type.INT)
        t_result = self._new_temp(Type.BOOL)

        ir = left_ir + right_ir + [IRCall(dst=cmp_result, name='strcmp', args=[left_value, right_value])]
        ir += self._ir_free_if_fresh_concat(expr.left, left_value)
        ir += self._ir_free_if_fresh_concat(expr.right, right_value)
        ir += [IRBinOp(dst=t_result, op=expr.op, left=cmp_result, right=IRConst(0, Type.INT))]
        return ir, t_result



