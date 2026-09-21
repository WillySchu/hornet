"""Scalar value production and storage -- int, int8, uint8, int64, and bool. _gen_read_scalar_into/_gen_write_scalar_from are the one choke
point for every scalar memory access in this compiler, so int8/uint8
being genuinely 1 byte and int64 genuinely 8 only ever needed teaching
to these two methods, not rediscovered at each read/write site: every
caller passes a value's ordinary 32-bit-named register, and these
(along with gen_binary_op/gen_unary_op/gen_cast_narrowing_into) decide
internally which actual width to operate on."""

from codegen.calling_convention import total_arg_slots
from ir.errors import IRError
from ir.ir import IRBranch, IRJump, IRLabel, IRMove, IRConst, IRCall
from ir.utils import type_of
from parser import Call, Binary, Variable, Field, Index, NoneLiteral, ArrayLiteral
from semantic import Type, TypeKind


class ScalarsMixin:
    def _ir_call_arguments(self, args: list, callee_name: str) -> tuple:
        """The shared per-argument marshaling loop between _ir_call
        and _ir_composite_call, returning (arg_ir, arg_values) rather
        than building the final IRCall itself -- the two callers
        differ only in that: an ordinary dst Temp vs. a hidden-pointer
        argument with dst=None.

        ARRAY-typed argument: a Variable/Field/Index (an existing
        address, via _ir_array_address), a bare bracketed-list literal
        (_ir_materialize_array_literal), or an ordinary composite-
        returning Call (_ir_materialize_composite_call). Every ARRAY-
        typed expression is one of these three shapes -- an unmatched
        shape, or either still-possible None return, raises IRError
        rather than leaving `ir`/`addr_value` unbound or stale.

        STRUCT-typed argument: identical shape, just Variable/Field/
        Index via _ir_struct_address, a struct-literal Call via _ir_
        materialize_struct_literal, then an ordinary composite-
        returning Call again -- UNLESS `callee_name`'s own declared
        parameter type at this position is a SUM type that lists this
        struct as a variant, in which case it's WIDENING, not an
        ordinary struct argument at all: _ir_materialize_sum_type_
        value (ir/sum_types.py), sized and tagged for the wider sum
        type, not the narrower struct this argument's own expression
        actually is.

        SUM-typed argument (already sum-typed, no widening needed --
        `takesShape(s)`, s already Shape): the identical Variable/
        Field/Index / ordinary-composite-call shape STRUCT has, minus
        a literal-Call case -- there's no sum-type literal syntax to
        parse into one (see SumTypeDef's own docstring in parser.py).

        `callee_name` is None-able for the one case that means there's
        no declared signature to consult at all: a builtin (print/len/
        append never appear in function_registry, and none has a sum-
        typed parameter to widen into regardless) -- when None, or
        when the callee just isn't found (shouldn't happen for a real
        call, but this stays a plain lookup miss rather than a raise),
        every argument is treated exactly as it was before sum types
        existed.

        `ir`/`addr_value` are plain local variables, reused across
        every argument -- leaving them unbound on a second argument
        (after an earlier one already assigned them) would silently
        reuse that earlier argument's own address for this one instead
        of crashing; the explicit IRError above closes that."""
        param_types = self.ir_program.function_registry[callee_name][0] if callee_name in self.ir_program.function_registry else None
        arg_ir = []
        arg_values = []
        for i, arg in enumerate(args):
            arg_type = type_of(arg)
            is_widening = (
                param_types is not None
                and arg_type.kind == TypeKind.STRUCT
                and param_types[i].kind == TypeKind.SUM
            )
            if is_widening:
                result = self._ir_materialize_sum_type_value(arg, param_types[i])
                if result is None:
                    raise IRError(
                        f"_ir_materialize_sum_type_value returned None for a "
                        f"struct-literal argument widening into a sum-typed "
                        f"parameter ({arg!r}) -- some field is out of scope for "
                        f"real IR, with no old-style fallback remaining to catch it")
                ir, addr_value = result
                arg_ir.extend(ir)
                arg_values.append(addr_value)
            elif arg_type.kind == TypeKind.ARRAY:
                if isinstance(arg, (Variable, Field, Index)):
                    result = self._ir_array_address(arg)
                    if result is None:
                        raise IRError(
                            f"_ir_array_address returned None for an ARRAY-typed "
                            f"Variable/Field/Index argument ({arg!r}) -- expected to "
                            f"always succeed for this shape")
                    ir, addr_value = result
                elif isinstance(arg, ArrayLiteral):
                    result = self._ir_materialize_array_literal(arg)
                    if result is None:
                        raise IRError(
                            f"_ir_materialize_array_literal returned None for an "
                            f"ARRAY-typed literal argument ({arg!r}) -- some element "
                            f"is out of scope for real IR, with no old-style fallback "
                            f"remaining to catch it")
                    ir, addr_value = result
                elif self._is_ordinary_composite_call(arg):
                    ir, addr_value = self._ir_materialize_composite_call(arg, arg_type)
                else:
                    raise IRError(
                        f"No codegen rule for an ARRAY-typed call argument of shape "
                        f"{type(arg).__name__}: {arg!r}")
                arg_ir.extend(ir)
                arg_values.append(addr_value)
            elif arg_type.kind == TypeKind.STRUCT:
                if isinstance(arg, (Variable, Field, Index)):
                    result = self._ir_struct_address(arg)
                    if result is None:
                        raise IRError(
                            f"_ir_struct_address returned None for a STRUCT-typed "
                            f"Variable/Field/Index argument ({arg!r}) -- expected to "
                            f"always succeed for this shape")
                    ir, addr_value = result
                elif isinstance(arg, Call) and arg.name in self.ir_program.struct_registry:
                    result = self._ir_materialize_struct_literal(arg)
                    if result is None:
                        raise IRError(
                            f"_ir_materialize_struct_literal returned None for a "
                            f"STRUCT-typed literal argument ({arg!r}) -- some field "
                            f"is out of scope for real IR, with no old-style fallback "
                            f"remaining to catch it")
                    ir, addr_value = result
                elif self._is_ordinary_composite_call(arg):
                    ir, addr_value = self._ir_materialize_composite_call(arg, arg_type)
                else:
                    raise IRError(
                        f"No codegen rule for a STRUCT-typed call argument of shape "
                        f"{type(arg).__name__}: {arg!r}")
                arg_ir.extend(ir)
                arg_values.append(addr_value)
            elif arg_type.kind == TypeKind.SUM:
                if isinstance(arg, (Variable, Field, Index)):
                    result = self._ir_struct_address(arg)  # generic address computation -- see its own docstring
                    if result is None:
                        raise IRError(
                            f"_ir_struct_address returned None for a SUM-typed "
                            f"Variable/Field/Index argument ({arg!r}) -- expected to "
                            f"always succeed for this shape")
                    ir, addr_value = result
                elif self._is_ordinary_composite_call(arg):
                    ir, addr_value = self._ir_materialize_composite_call(arg, arg_type)
                else:
                    raise IRError(
                        f"No codegen rule for a SUM-typed call argument of shape "
                        f"{type(arg).__name__}: {arg!r}")
                arg_ir.extend(ir)
                arg_values.append(addr_value)
            elif arg_type.kind == TypeKind.SLICE or isinstance(arg, NoneLiteral):
                result = self._ir_slice_arg(arg)
                if result is None:
                    raise IRError(
                        f"_ir_slice_arg returned None for a SLICE-typed call "
                        f"argument ({arg!r}) -- expected to always succeed for any "
                        f"reachable shape")
                ir, ptr_value, len_value, cap_value = result
                arg_ir.extend(ir)
                arg_values.extend([ptr_value, len_value, cap_value])
            else:
                ir, value = self.gen_expr_ir(arg)
                arg_ir.extend(ir)
                arg_values.append(value)
        return arg_ir, arg_values

    def _ir_call(self, expr: Call) -> tuple[list, object]:
        """Builds (without lowering) an ordinary function call's IR.
        Each argument's own shape (scalar, array/struct address, or a
        slice's own {ptr, len, cap} triple) is computed independently
        into its own Temp(s) first; IRCall's own lowering places them
        all into argument registers together, immediately before the
        call.

        See _ir_call_arguments for the per-argument dispatch (shared
        with _ir_composite_call). Returns (ir, t_result), t_result
        being None for a void call."""
        total_slots = total_arg_slots(expr.args)
        if total_slots > 6:
            raise IRError(
                f"Call to '{expr.name}' needs {total_slots} argument "
                f"register(s) (a slice-typed argument needs 3); this "
                f"compiler only supports up to 6 (passed via registers "
                f"per the SysV ABI -- stack-passed arguments aren't "
                f"implemented)"
            )
        result_type = type_of(expr)
        t_result = None if result_type == Type.VOID else self.ir_program.ids.new_temp(result_type)
        arg_ir, arg_values = self._ir_call_arguments(expr.args, expr.name)
        ir = arg_ir + [IRCall(dst=t_result, name=expr.name, args=arg_values)]
        return ir, t_result

    def _ir_composite_call(self, dst_address, call_expr: Call) -> list:
        """Builds (without lowering) a composite-returning function
        call's IR, writing its result through dst_address (however the
        caller already has it -- a freshly-computed address, or the
        current function's own received hidden pointer for a
        forwarding `return someFn()`).

        The hidden-pointer convention is just "dst_address as args[0],
        every genuine argument shifted one register position later,
        dst=None" -- which IRCall's own, already-generic lowering
        already produces, with no changes needed to IRCall itself.
        dst=None here means "nothing to capture from %eax," not "this
        call is void" -- the call is not void at the Hornet-language
        level, it just has no scalar result for %eax to hold.

        Reuses _ir_call_arguments for the genuine arguments, then
        prepends dst_address as the actual first argument value."""
        total_slots = 1 + total_arg_slots(call_expr.args)
        if total_slots > 6:
            raise IRError(
                f"Call to '{call_expr.name}' needs {total_slots} argument "
                f"register(s) (the hidden return pointer needs its own "
                f"slot, plus 3 for a slice-typed argument); this "
                f"compiler only supports up to 6 (passed via registers "
                f"per the SysV ABI -- stack-passed arguments aren't "
                f"implemented)"
            )
        arg_ir, arg_values = self._ir_call_arguments(call_expr.args, call_expr.name)
        return arg_ir + [IRCall(dst=None, name=call_expr.name, args=[dst_address] + arg_values)]

    def _ir_short_circuit(self, expr: Binary, *, short_circuit_value: int, label_prefix: str) -> tuple[list, object]:
        """Builds (without lowering) the shared IR for AND/OR -- mirror
        images: each evaluates its left side and branches on it,
        jumping past the right side entirely if that alone already
        decides the answer -- AND with short_circuit_value=0, OR with
        short_circuit_value=1. This is what makes `0 and (1 / 0)`
        return 0 instead of crashing: the division is real IR, built
        and lowered like any other, but control flow jumps clean over
        it. Returns (ir, t_result)."""
        fallthrough_value = 1 - short_circuit_value
        rhs_label = self.ir_program.ids.new_label(f"{label_prefix}_rhs")
        short_label = self.ir_program.ids.new_label(f"{label_prefix}_short")
        fallthrough_label = self.ir_program.ids.new_label(f"{label_prefix}_fallthrough")
        end_label = self.ir_program.ids.new_label(f"{label_prefix}_end")

        def targets(continue_label: str) -> tuple[str, str]:
            # (true_target, false_target): whichever outcome matches
            # short_circuit_value goes to `short_label`; the other
            # goes to `continue_label`.
            if short_circuit_value == 1:
                return short_label, continue_label
            return continue_label, short_label

        t_result = self.ir_program.ids.new_temp(Type.BOOL)
        left_true, left_false = targets(rhs_label)
        right_true, right_false = targets(fallthrough_label)

        left_ir, left_value = self.gen_expr_ir(expr.left)
        right_ir, right_value = self.gen_expr_ir(expr.right)
        ir = [
            *left_ir,
            IRBranch(cond=left_value, true_label=left_true, false_label=left_false),
            IRLabel(rhs_label),
            *right_ir,
            IRBranch(cond=right_value, true_label=right_true, false_label=right_false),
            IRLabel(fallthrough_label),
            IRMove(dst=t_result, src=IRConst(fallthrough_value, Type.BOOL)),
            IRJump(end_label),
            IRLabel(short_label),
            IRMove(dst=t_result, src=IRConst(short_circuit_value, Type.BOOL)),
            IRJump(end_label),
            IRLabel(end_label),
        ]
        return ir, t_result

