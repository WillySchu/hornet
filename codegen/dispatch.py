"""The routing layer: gen_expr_into and gen_binary_into are the only
two methods in this codebase whose actual job is to inspect an AST
node's type or kind and hand off to whichever feature file
(arrays_slices, structs, strings, scalars) owns that case -- every
other dispatch-shaped method elsewhere is really just implementing one
branch of one of these two."""

from codegen.assembly_ast import Operand, Instruction, MovQ, Imm, Mov, Memory, Register
from codegen.errors import CodegenError
from codegen.ir import (
    IRBinOp, IRValue, IRConst, IRLoad, IRMove, IRJump, IRLabel, IRStaticDataAddress, IRUnOp, IRCast
)
from codegen.utils import as_qword_register, type_of
from typing import Optional
from parser import (
    ArrayLiteral,
    Binary,
    BinaryOp,
    BoolLiteral,
    Call,
    Cast,
    Constant,
    Field,
    Index,
    Node,
    NoneLiteral,
    StringLiteral,
    Slice,
    Unary,
    Variable,
)
from semantic import Type, TypeKind


class DispatchMixin:
    def gen_expr_into(self, expr: Node, dst: Operand) -> list[Instruction]:
        """Emits the instructions needed to compute `expr` and leave its
        result sitting in `dst`.

        This (rather than "return an Operand") is the right shape for
        expression codegen once operators are involved: a Constant can
        be represented as a bare Imm operand, but "the result of negating
        something" can't -- it has to actually be computed by an
        instruction acting on a register. So every expression, constants
        included, is generated the same way: as instructions that leave
        their answer in `dst`.
        """
        if isinstance(expr, Constant):
            if expr.resolved_type == Type.INT64:
                # A full 64-bit immediate move (`movq $9000000000,
                # %rax`) -- GNU as accepts an immediate this wide
                # specifically for movq (silently using the `movabs`
                # encoding under the hood), the one exception to
                # ordinary x86-64 instructions being limited to a
                # 32-bit immediate operand. An ordinary 32-bit Mov
                # here would either truncate the value or simply fail
                # to assemble for anything outside int32's own range.
                return [MovQ(src=Imm(expr.value), dst=as_qword_register(dst))]
            return [Mov(src=Imm(expr.value), dst=dst)]
        if isinstance(expr, BoolLiteral):
            # bool has the same 4-byte runtime representation as int
            # (0/1 in a register or stack slot) -- semantic.py is what
            # keeps the two from being mixed up; codegen just needs an
            # immediate.
            return [Mov(src=Imm(1 if expr.value else 0), dst=dst)]
        if isinstance(expr, StringLiteral):
            return self.gen_string_literal_into(expr, dst)
        if isinstance(expr, ArrayLiteral):
            # Never reachable in correct codegen -- an array literal's
            # value can't fit in a single register, so every producer
            # of one routes through gen_array_value_into/
            # gen_array_literal_into instead of calling gen_expr_into
            # on it directly. A clear error here catches a codegen bug
            # immediately rather than silently truncating the array.
            raise CodegenError(
                "Cannot compute an array literal via gen_expr_into -- "
                "arrays don't fit in a single register; use "
                "gen_array_value_into instead"
            )
        if isinstance(expr, Slice):
            # Same reasoning as ArrayLiteral above: a slice's value is
            # a 24-byte descriptor, which doesn't fit in a register
            # either.
            raise CodegenError(
                "Cannot compute a slice expression via gen_expr_into -- "
                "slices don't fit in a single register; use "
                "gen_slice_value_into instead"
            )
        if isinstance(expr, NoneLiteral):
            # Rejected for a different reason than ArrayLiteral/Slice
            # above: not a size problem (none's {0,0,0} descriptor is
            # exactly as wide as any other slice) but that none has no
            # one fixed target type to compute INTO -- its callers
            # (gen_var_decl/gen_assign's NoneLiteral short-circuit)
            # already know and pass the target type explicitly, which
            # this method's signature has no way to supply. `s == none`
            # is handled entirely separately, via
            # gen_slice_none_comparison_into, dispatched from
            # gen_binary_into before it would ever reach here.
            raise CodegenError(
                "Cannot compute 'none' via gen_expr_into -- it's only "
                "supported as a slice's zero value (see gen_none_into) "
                "or as one side of a slice comparison (see "
                "gen_slice_none_comparison_into), never as a general-"
                "purpose expression value"
            )
        if isinstance(expr, Variable):
            offset = self._local_offset(expr.name)
            var_type = self._local_type(expr.name)
            if var_type.kind == TypeKind.ARRAY:
                raise CodegenError(
                    f"Cannot read array-typed variable '{expr.name}' via "
                    f"gen_expr_into -- arrays don't fit in a single "
                    f"register; use gen_array_value_into or "
                    f"gen_array_address_into instead"
                )
            if var_type.kind == TypeKind.SLICE:
                raise CodegenError(
                    f"Cannot read slice-typed variable '{expr.name}' via "
                    f"gen_expr_into -- slices don't fit in a single "
                    f"register; use gen_slice_value_into instead"
                )
            if var_type.kind == TypeKind.STRUCT:
                raise CodegenError(
                    f"Cannot read struct-typed variable '{expr.name}' via "
                    f"gen_expr_into -- a struct doesn't fit in a single "
                    f"register; use gen_struct_value_into or "
                    f"gen_struct_address_into instead"
                )
            if var_type == Type.STR:
                return [MovQ(src=Memory('rbp', offset), dst=as_qword_register(dst))]
            return self._gen_read_scalar_into(Memory('rbp', offset), var_type, dst)
        if isinstance(expr, Index):
            element_type = type_of(expr)
            if element_type.kind == TypeKind.ARRAY:
                # Reading a sub-array (`matrix[i]` alone) has the same
                # register-width problem as ArrayLiteral above --
                # `[3]int row = matrix[i]` goes through
                # gen_array_value_into instead.
                raise CodegenError(
                    "Cannot read a sub-array via gen_expr_into -- arrays "
                    "don't fit in a single register; use "
                    "gen_array_value_into or gen_array_address_into instead"
                )
            if element_type.kind == TypeKind.STRUCT:
                # Same reasoning, for a struct-typed array element.
                raise CodegenError(
                    "Cannot read a struct-typed array element via "
                    "gen_expr_into -- a struct doesn't fit in a single "
                    "register; use gen_struct_value_into or "
                    "gen_struct_address_into instead"
                )
            addr_reg = as_qword_register(dst)
            instructions = self.gen_index_address_into(expr, addr_reg)
            if element_type == Type.STR:
                instructions.append(MovQ(src=Memory(addr_reg.name, 0), dst=addr_reg))
            else:
                instructions.extend(self._gen_read_scalar_into(Memory(addr_reg.name, 0), element_type, dst))
            return instructions
        if isinstance(expr, Field):
            field_type = type_of(expr)
            if field_type.kind == TypeKind.ARRAY:
                raise CodegenError(
                    "Cannot read an array-typed field via gen_expr_into "
                    "-- arrays don't fit in a single register; use "
                    "gen_array_value_into or gen_array_address_into instead"
                )
            if field_type.kind == TypeKind.SLICE:
                raise CodegenError(
                    "Cannot read a slice-typed field via gen_expr_into -- "
                    "slices don't fit in a single register; use "
                    "gen_slice_value_into instead"
                )
            if field_type.kind == TypeKind.STRUCT:
                raise CodegenError(
                    "Cannot read a struct-typed field via gen_expr_into -- "
                    "a struct doesn't fit in a single register; use "
                    "gen_struct_value_into or gen_struct_address_into "
                    "instead"
                )
            addr_reg = as_qword_register(dst)
            instructions = self.gen_field_address_into(expr, addr_reg)
            if field_type == Type.STR:
                instructions.append(MovQ(src=Memory(addr_reg.name, 0), dst=addr_reg))
            else:
                instructions.extend(self._gen_read_scalar_into(Memory(addr_reg.name, 0), field_type, dst))
            return instructions
        if isinstance(expr, Call):
            if type_of(expr).kind == TypeKind.ARRAY:
                # Never reachable in correct codegen -- same reasoning
                # as ArrayLiteral above. An array-returning call's
                # result is only ever consumed via gen_array_value_
                # into's own Call case (hidden-pointer convention),
                # never landed in a single register here.
                raise CodegenError(
                    f"Cannot call '{expr.name}' (which returns an array) "
                    f"via gen_expr_into -- arrays don't fit in a single "
                    f"register; use gen_array_value_into instead"
                )
            if type_of(expr).kind == TypeKind.SLICE:
                raise CodegenError(
                    f"Cannot call '{expr.name}' (which returns a slice) "
                    f"via gen_expr_into -- a slice descriptor doesn't "
                    f"fit in a single register; use gen_slice_call_into "
                    f"instead"
                )
            if type_of(expr).kind == TypeKind.STRUCT:
                raise CodegenError(
                    f"Cannot call '{expr.name}' (which returns a struct) "
                    f"via gen_expr_into -- a struct doesn't fit in a "
                    f"single register; use gen_struct_call_into instead"
                )
            if expr.name == 'print':
                return self.gen_print_call_into(expr, dst)
            if expr.name == 'len':
                return self.gen_len_call_into(expr, dst)
            return self.gen_call_into(expr, dst)
        if isinstance(expr, Cast):
            # Compute the source into dst first (already correctly
            # widened if it was int8/uint8-typed), then re-narrow dst's
            # LOW BYTE if the target is int8/uint8 -- see
            # gen_cast_narrowing_into. A target of int needs nothing
            # further: the source's already-widened value already IS a
            # valid int.
            instructions = self.gen_expr_into(expr.expr, dst)
            instructions.extend(self.gen_cast_narrowing_into(expr.resolved_type, dst))
            return instructions
        if isinstance(expr, Unary):
            # Compute the operand into dst first, then apply this
            # node's operator to whatever's now there -- what makes
            # chained operators (`~-2`) work.
            #
            # operand_type reads type_of(expr) -- this OUTER node's own
            # resolved_type -- not type_of(expr.operand). The two are
            # ordinarily identical, EXCEPT for a widened literal:
            # `int64 x = -5` sets resolved_type to int64 on the OUTER
            # Unary node, but the INNER Constant(5)'s own resolved_type
            # is still Type.INT (widening only ever touches the
            # outermost node of a literal expression). Reading the
            # inner one here was a real, found bug: it silently fed the
            # wrong operand_type into gen_unary_op, using 32-bit Neg
            # instead of NegQ for `-5` widened to int64.
            instructions = self.gen_expr_into(expr.operand, dst)
            instructions.extend(self.gen_unary_op(expr.op, dst, operand_type=type_of(expr)))
            return instructions
        if isinstance(expr, Binary):
            # ADD and the two equality operators are overloaded for
            # str (concatenation and strcmp-backed comparison) --
            # everything else goes through gen_binary_into unchanged.
            if expr.op == BinaryOp.ADD and type_of(expr.left) == Type.STR:
                return self.gen_string_concat_into(expr, dst)
            if expr.op in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL) and type_of(expr.left) == Type.STR:
                return self.gen_string_compare_into(expr, dst)
            return self.gen_binary_into(expr, dst)
        raise CodegenError(f"No codegen rule for expression: {expr!r}")

    def gen_expr_ir(self, expr: Node) -> tuple[list, Optional[IRValue]]:
        """The IR-native counterpart to gen_expr_into: builds real IR
        for the node kinds that have it -- a bare Constant/BoolLiteral
        (see below), a scalar Variable (see below), a scalar-typed
        Index/Field read (see _ir_load -- the underlying address
        computation stays old-style; only the load itself is real
        IR), Binary (see _ir_expr_binary), a len(x) call (see _ir_
        len_call, its own dedicated case, not routed through _ir_call
        at all -- len isn't an ordinary function call), and an
        ordinary scalar-or-void-returning Call (see _ir_call) -- and
        raises CodegenError explicitly for everything else, with no
        fallback left to defer to (see ir.py's own module docstring
        for IRRaw's own removal, once this was the last remaining
        site).

        Every other node kind reaching here is a genuine bug, not a
        deliberate scope boundary anymore -- see the note below on
        exactly why ArrayLiteral/Slice/NoneLiteral/a composite-
        returning Call never actually reach this point in practice.

        This method does NOT "cover" an ArrayLiteral, Slice,
        NoneLiteral, or a composite-returning Call the way an earlier
        version of this docstring claimed -- when the old-style
        fallback still existed here, gen_expr_into itself defensively
        REJECTED every one of those (they don't fit in a single
        register), so reaching that fallback with one of them crashed,
        it didn't handle it. A REAL BUG, found this way: a bare `none`
        or a bare, discarded composite-returning Call used directly as
        a statement (`none` or `makeArray()` alone on a line) used to
        crash outright. The actual fix lives one level up, in
        gen_statement_ir's own ExprStmt dispatch -- explicit
        NoneLiteral/append/ordinary-composite-Call cases there route
        around this method entirely for exactly those shapes, the
        same "handle it before it ever reaches the generic case" shape
        that method's own ArrayLiteral/Slice cases already used. An
        ArrayLiteral or Slice reaching this method directly (not via a
        bare ExprStmt) still isn't reachable in practice: every other
        AST position either has its own, earlier real-IR case (a
        VarDecl's own initializer, a function-call argument, ...) or
        is one semantic.py itself doesn't allow these two shapes to
        appear in at all.

        Returns (ir, value) -- value is None only for a void call,
        which can only legally appear via a bare ExprStmt (see
        gen_statement_ir), never as another expression's operand.

        A Constant/BoolLiteral or a Variable reference each cost ZERO
        instructions here -- the former is just IRConst, a compile-
        time value IR already had a case for since it was first
        designed (see ir.py), simply never actually produced until
        now: every existing caller of gen_expr_into's own Mov-emitting
        Constant/BoolLiteral case still works exactly as before, this
        only changes what a NEW caller building real IR gets instead.
        The latter's `expr.name` already has its own persistent Temp
        (see _bind_local), so reading it is just handing back that
        same Temp -- materializing either kind of value into a real
        register only happens later, lazily, wherever something
        actually needs it. Never reached for a composite-typed
        Variable (array/slice/struct): gen_expr_into's own Variable
        case already rejects those before this method could ever be
        called on one, the same guarantee that already makes its
        fallback below safe."""
        if isinstance(expr, Constant):
            return [], IRConst(expr.value, type_of(expr))
        if isinstance(expr, BoolLiteral):
            return [], IRConst(1 if expr.value else 0, Type.BOOL)
        if isinstance(expr, StringLiteral):
            # A static .data label's own address -- IRStaticDataAddress
            # directly, not gen_string_literal_into's own IRRaw-wrapped
            # LeaQ: this IS exactly the leaf IRStaticDataAddress exists
            # for (see its own docstring), just never converted until
            # now. Registers this literal's own content for later
            # emission the identical way gen_string_literal_into
            # already does -- a fresh label per occurrence, even for
            # identical content, no deduplication -- just inlined here
            # rather than delegating to that method for what's now a
            # two-line operation with no old-style instruction sequence
            # left to wrap at all.
            t = self._new_temp(Type.STR)
            label = self.new_label("str")
            self.string_literals.append((label, expr.value))
            return [IRStaticDataAddress(dst=t, label=label)], t
        if isinstance(expr, Variable):
            return [], self._local_temp(expr.name)
        if isinstance(expr, Index) and type_of(expr).kind not in (TypeKind.ARRAY, TypeKind.STRUCT):
            result = self._ir_index_address(expr)
            if result is None:
                raise CodegenError(
                    f"_ir_index_address returned None for a scalar-typed Index read "
                    f"({expr!r}) -- expected to always succeed for a reachable base")
            addr_ir, addr_value = result
            return self._ir_load(addr_ir, addr_value, type_of(expr))
        if isinstance(expr, Field) and type_of(expr).kind not in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT):
            result = self._ir_field_address(expr)
            if result is None:
                raise CodegenError(
                    f"_ir_field_address returned None for a scalar-typed Field read "
                    f"({expr!r}) -- expected to always succeed for a reachable base")
            addr_ir, addr_value = result
            return self._ir_load(addr_ir, addr_value, type_of(expr))
        if isinstance(expr, Binary):
            return self._ir_expr_binary(expr)
        if isinstance(expr, Call) and expr.name == 'print':
            # A dedicated case, not routed through _ir_call at all --
            # print isn't an ordinary function call (it always needs
            # an address for its argument, unlike an ordinary call's
            # own by-value/by-address split, and its second
            # argument -- a type descriptor -- has no corresponding
            # Hornet expression to run gen_expr_ir on at all), so it
            # needs its own entry point the same way len's own
            # exclusion from _ir_call already does. Never falls
            # through to the catch-all below -- unlike len,
            # _ir_print_call's own contract is total, matching
            # semantic.py's own guarantee that this is always exactly
            # one, well-formed argument by the time codegen ever sees
            # it.
            return self._ir_print_call(expr)
        if isinstance(expr, Call) and expr.name == 'len':
            # A dedicated case, not routed through _ir_call at all --
            # len isn't an ordinary function call (no calling
            # convention, no argument-register placement), so it
            # needs its own entry point the same way print's own
            # exclusion from _ir_call already does. Falls through to
            # the ordinary catch-all below when out of scope (see
            # _ir_len_call's own docstring) -- _ir_call's own name
            # exclusion ('len', alongside 'print') already keeps that
            # dispatch from ever re-attempting this as an ordinary
            # call.
            result = self._ir_len_call(expr)
            if result is not None:
                return result
        if isinstance(expr, Call) and expr.name not in ('print', 'len') and type_of(expr).kind not in (
                TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT):
            return self._ir_call(expr)
        if isinstance(expr, Unary):
            # IRUnOp already existed, fully lowered (via gen_unary_op,
            # the same already-proven old-style function IRBinOp's own
            # lowering already reuses for binary operators) and
            # already register-allocator-supported -- it was just
            # never constructed anywhere. Recursing into expr.operand
            # FIRST is what makes a chained operator (`~-2`) compose
            # correctly: the inner Unary node builds its own IRUnOp
            # through this identical case, so the outer one's own
            # operand is already an ordinary Temp by the time it runs,
            # no different from any other nested expression in this
            # arc.
            operand_ir, operand_value = self.gen_expr_ir(expr.operand)
            t = self._new_temp(type_of(expr))
            return operand_ir + [IRUnOp(dst=t, op=expr.op, operand=operand_value)], t
        if isinstance(expr, Cast):
            # IRCast is new -- see its own docstring for why an
            # ordinary IRMove into a differently-typed Temp doesn't
            # suffice here, unlike every other "produce a value of a
            # specific type" case in this arc. Lowering reuses gen_
            # cast_narrowing_into completely unchanged, the identical
            # "wrap the one proven old-style helper, don't reimplement
            # it" shape IRUnOp's own lowering already uses for gen_
            # unary_op.
            src_ir, src_value = self.gen_expr_ir(expr.expr)
            t = self._new_temp(type_of(expr))
            return src_ir + [IRCast(dst=t, src=src_value)], t
        raise CodegenError(
            f"No real-IR case for expression of type {type(expr).__name__}: {expr!r} -- "
            f"every language construct this arc's own tests exercise (including every "
            f"shape print() can take, the last remaining user of what used to be this "
            f"method's own IRRaw-wrapped fallback) is confirmed to reach real IR without "
            f"ever falling back here. A genuine bug if this fires, not a deliberate scope "
            f"boundary -- there is no old-style fallback left to silently defer to anymore."
        )

    def _ir_load(self, addr_ir: list, addr_value, value_type) -> tuple[list, IRValue]:
        """Shared by gen_expr_ir's Index/Field cases: given an
        address already built as real IR (see _ir_index_address/_ir_
        field_address, this method's own two callers), IRLoads
        value_type's own width through it. This method itself has no
        fallback of any kind -- it's a pure two-instruction leaf over
        whatever address its caller already produced."""
        t = self._new_temp(value_type)
        return addr_ir + [IRLoad(dst=t, address=addr_value)], t

    def _ir_composite_operand_address(self, expr: Node, value_type: Type):
        """Builds (without lowering) an ARRAY- or STRUCT-typed
        equality operand's own address as real IR -- returns (ir,
        address), or None when out of scope. Both of _ir_expr_binary's
        own equality operands need this identical dispatch, so it's
        factored out here rather than duplicated inline the way _ir_
        indexable_base/_ir_call_arguments each dispatch their own,
        different single operand -- this one genuinely needs the same
        logic run twice within one call.

        In dispatch order: a Variable/Field/Index (an existing
        address, via _ir_array_address/_ir_struct_address); a bare
        bracketed-list literal (ARRAY only -- _ir_materialize_array_
        literal; there is no SLICE/STRUCT equivalent, since a struct
        literal can never be compared this way at all -- see below);
        an ordinary composite-returning Call (_ir_materialize_
        composite_call, shared by both ARRAY and STRUCT).

        A struct-literal Call needs no case here, unlike the ARRAY
        side's own ArrayLiteral one: semantic.py already rejects a
        struct literal as a Binary operand outright (`Point(1,2) ==
        Point(1,2)` fails to type-check, naming the same narrow
        whitelist of allowed positions this arc has already run into
        at a Field/Index/Slice base) -- there is no gap here for
        codegen to close, the identical situation _ir_materialize_
        array_literal's own docstring already documents for those
        other base positions.

        A REAL BUG, found and fixed here rather than carried forward:
        `[1,2,3] == [1,2,4]` and an ordinary array/struct-returning
        call used directly as an equality operand both used to raise
        a hard CodegenError, tracing back to the OLD-style gen_array_
        address_into itself -- which never handled an ArrayLiteral or
        Call operand at all. This was never supported, even old-style,
        the identical situation a composite-returning call used
        directly as an addressable base was in before this arc's own
        earlier work there."""
        if isinstance(expr, (Variable, Field, Index)):
            address_fn = self._ir_array_address if value_type.kind == TypeKind.ARRAY else self._ir_struct_address
            return address_fn(expr)
        if value_type.kind == TypeKind.ARRAY and isinstance(expr, ArrayLiteral):
            return self._ir_materialize_array_literal(expr)
        if self._is_ordinary_composite_call(expr):
            return self._ir_materialize_composite_call(expr, value_type)
        return None

    def _ir_expr_binary(self, expr: Binary) -> tuple[list, IRValue]:
        """The IR-native counterpart to gen_binary_into's own three-way
        dispatch: short-circuit AND/OR (already real IR, via
        _ir_short_circuit), slice-vs-none comparison (real IR too, for
        every reachable slice-typed base shape, confirmed
        exhaustively -- see _ir_slice_none_comparison's own docstring;
        a None here, genuinely never observed, raises CodegenError
        rather than silently propagating further up the call chain),
        string concat/compare (also real IR now, via
        _ir_string_concat/_ir_string_compare -- composed entirely from
        ordinary IRCall/IRBinOp, since IRCall's own lowering is
        already generic over any callee name, external C library
        functions included, with no new IR concept needed), array/
        struct equality (now real IR too, for a Variable/Field/Index,
        a bare bracketed-list literal (ARRAY only), or an ordinary
        composite-returning Call, in any combination on either side --
        see _ir_composite_operand_address/_ir_composite_equal; a
        struct-literal Call is the one shape genuinely still out of
        scope here, moot in practice since semantic.py already rejects
        it as a Binary operand outright, not just here -- so once this
        branch is entered at all, i.e. type_of(expr.left).kind is
        ARRAY/STRUCT, both operands' own addresses are expected to
        always resolve; a CodegenError, not a silent fall-through to
        the ordinary scalar case just below, is what a None here now
        gets), or the
        ordinary arithmetic/comparison case (already real IR, via
        _ir_binary)."""
        if expr.op == BinaryOp.AND:
            return self._ir_short_circuit(expr, short_circuit_value=0, label_prefix="and")
        if expr.op == BinaryOp.OR:
            return self._ir_short_circuit(expr, short_circuit_value=1, label_prefix="or")
        if type_of(expr.left) == Type.STR:
            if expr.op == BinaryOp.ADD:
                return self._ir_string_concat(expr)
            if expr.op in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL):
                return self._ir_string_compare(expr)
        if expr.op in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL):
            if type_of(expr.left).kind == TypeKind.SLICE or type_of(expr.right).kind == TypeKind.SLICE:
                result = self._ir_slice_none_comparison(expr)
                if result is None:
                    raise CodegenError(
                        f"_ir_slice_none_comparison returned None for a slice-vs-"
                        f"none comparison ({expr.left!r} {expr.op} {expr.right!r}) "
                        f"-- expected to always succeed for a reachable slice-typed "
                        f"base, with no old-style fallback remaining to catch it")
                return result
            if type_of(expr.left).kind in (TypeKind.ARRAY, TypeKind.STRUCT):
                value_type = type_of(expr.left)
                left_result = self._ir_composite_operand_address(expr.left, value_type)
                right_result = self._ir_composite_operand_address(expr.right, value_type)
                if left_result is None or right_result is None:
                    raise CodegenError(
                        f"_ir_composite_operand_address returned None for an "
                        f"ARRAY/STRUCT equality operand ({expr.left!r} or "
                        f"{expr.right!r}) -- expected to always succeed once "
                        f"type_of(expr.left).kind is ARRAY/STRUCT at all, since a "
                        f"struct-literal Call (the one shape it doesn't cover) is "
                        f"already rejected as a Binary operand by semantic.py before "
                        f"codegen ever runs")
                left_ir, left_addr = left_result
                right_ir, right_addr = right_result
                mismatch_label = self.new_label("eq_mismatch")
                done_label = self.new_label("eq_done")
                cmp_ir = self._ir_composite_equal(left_addr, right_addr, value_type, mismatch_label)
                t = self._new_temp(Type.BOOL)
                ir = left_ir + right_ir + cmp_ir + [
                    IRMove(dst=t, src=IRConst(1 if expr.op == BinaryOp.EQUAL else 0, Type.BOOL)),
                    IRJump(done_label),
                    IRLabel(mismatch_label),
                    IRMove(dst=t, src=IRConst(0 if expr.op == BinaryOp.EQUAL else 1, Type.BOOL)),
                    IRLabel(done_label),
                ]
                return ir, t
        return self._ir_binary(expr)

    def gen_binary_into(self, expr: Binary, dst: Operand) -> list[Instruction]:
        """Computes `expr.left OP expr.right` into `dst`.

        AND/OR are handled separately (see gen_short_circuit) since
        they must not unconditionally evaluate both sides. Every other
        binary operator -- arithmetic and comparisons alike -- always
        evaluates both operands. Requires `dst` to be a register.
        """
        if expr.op == BinaryOp.AND:
            return self.gen_short_circuit(
                expr, dst,
                short_circuit_value=0,   # left (or then right) false -> whole thing false
                label_prefix="and",
            )
        if expr.op == BinaryOp.OR:
            return self.gen_short_circuit(
                expr, dst,
                short_circuit_value=1,   # left (or then right) true -> whole thing true
                label_prefix="or",
            )

        if not isinstance(dst, Register):
            raise CodegenError(f"Binary codegen requires a register destination, got: {dst!r}")

        # A slice compared to `none` (either order) needs its own
        # dedicated codegen: a slice's "value" is a 24-byte descriptor,
        # which doesn't fit in a single Temp. semantic.py's check_binary
        # already guarantees exactly one side is slice-typed and the
        # other none-typed by the time this is reached.
        #
        # ARRAY and STRUCT equality are dispatched the same way, for
        # the same reason: neither value fits in a single Temp.
        # check_binary guarantees both sides are the same, comparable
        # array/struct type (no slice nested anywhere inside) -- see
        # gen_array_equality_into/gen_struct_equality_into for how
        # each dispatches internally.
        if expr.op in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL):
            if type_of(expr.left).kind == TypeKind.SLICE or type_of(expr.right).kind == TypeKind.SLICE:
                return self.gen_slice_none_comparison_into(expr, dst)
            if type_of(expr.left).kind == TypeKind.ARRAY:
                return self.gen_array_equality_into(expr, dst)
            if type_of(expr.left).kind == TypeKind.STRUCT:
                return self.gen_struct_equality_into(expr, dst)

        # The ordinary case (arithmetic, or a comparison between two
        # scalars): build its IR (see _ir_binary), lower it, read the
        # result into dst.
        ir, t_result = self._ir_binary(expr)
        instructions = self._instruction_selector.lower_ir(ir)
        instructions.extend(
            self._gen_read_scalar_into(self._instruction_selector._temp_mem(t_result), t_result.type, dst))
        return instructions

    def _ir_binary(self, expr: Binary) -> tuple[list, object]:
        """Builds (without lowering) the ordinary arithmetic/comparison
        case's IR: evaluate each operand (via gen_expr_ir, so a nested
        migrated sub-expression -- another Binary, a scalar Call --
        stays real IR instead of being immediately, separately
        lowered), combine via IRBinOp. lower_ir reuses gen_binary_op
        unchanged as this op's own instruction-selection rule, so the
        arithmetic itself isn't reimplemented here. Returns
        (ir, t_result)."""
        left_ir, left_value = self.gen_expr_ir(expr.left)
        right_ir, right_value = self.gen_expr_ir(expr.right)
        t_result = self._new_temp(type_of(expr))
        ir = left_ir + right_ir + [
            IRBinOp(dst=t_result, op=expr.op, left=left_value, right=right_value),
        ]
        return ir, t_result
