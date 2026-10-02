"""Scalar values: int, int8, uint8, int64, bool."""

from ir.errors import IRError
from ir.ir import IRBinOp, IRBranch, IRJump, IRLabel, IRLoad, IRMove, IRConst, IRCall
from ops import BinaryOp
from ir.utils import is_composite_addressable, type_of
from parser import Call, Binary, Variable, Field, Index, ArrayLiteral, DictLiteral
from typesys import Type, TypeKind


class ScalarsMixin:
    def _ir_call_arguments(self, args: list, callee_name: str) -> tuple:
        """Marshal call arguments; returns (arg_ir, arg_values)."""
        param_types = self.ir_program.function_registry[callee_name][0] if callee_name in self.ir_program.function_registry else None
        arg_ir = []
        arg_values = []
        for i, arg in enumerate(args):
            arg_type = type_of(arg)
            if (param_types is not None and param_types[i].kind == TypeKind.SLICE
                    and isinstance(arg, ArrayLiteral) and arg.type_expr is None):
                # An untyped literal for a slice parameter (`f([1, 2])`, `f([])`): build that slice and
                # pass its three words, like any slice.
                ir, descriptor = self._ir_materialize_value_into_scratch(arg, param_types[i], self.ir_fn, "slice_literal_arg")
                arg_ir.extend(ir)
                for offset in (0, 8, 16):
                    field_addr, word = self.ir_program.ids.new_temp(Type.INT64), self.ir_program.ids.new_temp(
                        Type.INT64 if offset == 0 else Type.INT)
                    arg_ir += [IRBinOp(dst=field_addr, op=BinaryOp.ADD, left=descriptor, right=IRConst(offset, Type.INT64)),
                               IRLoad(dst=word, address=field_addr)]
                    arg_values.append(word)
                continue
            is_widening = (
                param_types is not None
                and param_types[i].kind == TypeKind.SUM
                and arg_type.kind != TypeKind.SUM
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
                if is_composite_addressable(arg):
                    result = self._ir_array_address(arg)
                    if result is None:
                        raise IRError(
                            f"_ir_array_address returned None for an ARRAY-typed "
                            f"Variable/Field/Index/dereference argument ({arg!r}) -- "
                            f"expected to always succeed for this shape")
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
                if is_composite_addressable(arg):
                    result = self._ir_struct_address(arg)
                    if result is None:
                        raise IRError(
                            f"_ir_struct_address returned None for a STRUCT-typed "
                            f"Variable/Field/Index/dereference argument ({arg!r}) -- "
                            f"expected to always succeed for this shape")
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
            elif arg_type.kind == TypeKind.DICT:
                if is_composite_addressable(arg):
                    result = self._ir_dict_address(arg)
                    if result is None:
                        raise IRError(
                            f"_ir_dict_address returned None for a DICT-typed "
                            f"Variable/Field/Index/dereference argument ({arg!r}) -- "
                            f"expected to always succeed for this shape")
                    ir, addr_value = result
                elif isinstance(arg, DictLiteral):
                    ir, addr_value = self._ir_materialize_dict_literal(arg)
                elif self._is_ordinary_composite_call(arg):
                    ir, addr_value = self._ir_materialize_composite_call(arg, arg_type)
                else:
                    raise IRError(
                        f"No codegen rule for a DICT-typed call argument of shape "
                        f"{type(arg).__name__}: {arg!r}")
                arg_ir.extend(ir)
                arg_values.append(addr_value)
            elif arg_type.kind == TypeKind.SUM:
                if isinstance(arg, (Variable, Field, Index)):
                    result = self._ir_struct_address(arg)
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
            elif arg_type.kind == TypeKind.STR:
                result = self._ir_str_value(arg)
                if result is None:
                    raise IRError(
                        f"_ir_str_value returned None for a str-typed call "
                        f"argument ({arg!r}) -- expected to always succeed for any "
                        f"reachable shape")
                ir, ptr_value, len_value = result
                arg_ir.extend(ir)
                arg_values.extend([ptr_value, len_value])
            elif arg_type.kind == TypeKind.SLICE:
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
        """Ordinary call IR."""
        result_type = type_of(expr)
        t_result = None if result_type == Type.VOID else self.ir_program.ids.new_temp(result_type)
        arg_ir, arg_values = self._ir_call_arguments(expr.args, expr.name)
        ir = arg_ir + [IRCall(dst=t_result, name=expr.name, args=arg_values)]
        return ir, t_result

    def _ir_composite_call(self, dst_address, call_expr: Call) -> list:
        """Composite-returning call; result written through dst_address."""
        arg_ir, arg_values = self._ir_call_arguments(call_expr.args, call_expr.name)
        return arg_ir + [IRCall(dst=None, name=call_expr.name, args=[dst_address] + arg_values)]

    def _ir_short_circuit(self, expr: Binary, *, short_circuit_value: int, label_prefix: str) -> tuple[list, object]:
        """AND/OR with short-circuit branches."""
        fallthrough_value = 1 - short_circuit_value
        rhs_label = self.ir_program.ids.new_label(f"{label_prefix}_rhs")
        short_label = self.ir_program.ids.new_label(f"{label_prefix}_short")
        fallthrough_label = self.ir_program.ids.new_label(f"{label_prefix}_fallthrough")
        end_label = self.ir_program.ids.new_label(f"{label_prefix}_end")

        def targets(continue_label: str) -> tuple[str, str]:
            # whichever outcome short-circuits jumps to short_label
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

