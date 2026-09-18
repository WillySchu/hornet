"""Scalar value production and storage -- int, int8, uint8, int64, and
bool. _gen_read_scalar_into/_gen_write_scalar_from are the one choke
point for every scalar memory access in this compiler, so int8/uint8
being genuinely 1 byte and int64 genuinely 8 only ever needed teaching
to these two methods, not rediscovered at each read/write site: every
caller passes a value's ordinary 32-bit-named register, and these
(along with gen_binary_op/gen_unary_op/gen_cast_narrowing_into) decide
internally which actual width to operate on."""

from codegen.calling_convention import total_arg_slots
from codegen.errors import CodegenError
from ir.ir import IRBranch, IRJump, IRLabel, IRMove, IRConst, IRCall
from codegen.utils import type_of
from parser import Call, Binary, Variable, Field, Index, NoneLiteral, ArrayLiteral
from semantic import Type, TypeKind


class ScalarsMixin:
    def _ir_call_arguments(self, args: list) -> tuple:
        """The shared per-argument marshaling loop between _ir_call
        and _ir_composite_call -- see _ir_call's own docstring for the
        general per-argument real-IR-or-fallback architecture; this is
        just the extracted loop, returning (arg_ir, arg_values) rather
        than building the final IRCall itself, since the two callers
        differ in exactly that (an ordinary dst Temp vs. a hidden
        hidden-pointer argument with dst=None).

        ARRAY-typed argument, in dispatch order: a Variable/Field/
        Index (an existing address, via _ir_array_address -- itself
        can still return None for an Index/Field whose own BASE is out
        of scope, so this checks for that explicitly rather than
        assuming success); a bare bracketed-list literal (_ir_
        materialize_array_literal, which can also still return None
        for its own, deeper reason -- some element out of scope); an
        ordinary composite-returning Call (_ir_materialize_composite_
        call, which never returns None). Every ARRAY-typed expression
        is one of these three shapes -- an unmatched fourth shape, or
        either of the two genuinely-still-possible None returns above,
        raises CodegenError explicitly rather than either silently
        leaving ir/addr_value unbound (an old-style, IRRaw-wrapped
        fallback used to catch exactly this here, since removed once
        every shape this arc's own tests exercise was confirmed to
        never need it) or -- worse -- proceeding with a stale value
        from a previous loop iteration.

        STRUCT-typed argument: identical shape (including the same
        explicit CodegenError on an unmatched shape or either
        genuinely-still-possible None), just Variable/Field/Index via
        _ir_struct_address, a struct-literal Call (name found in
        self.host.struct_registry) via _ir_materialize_struct_literal, then
        an ordinary composite-returning Call via _ir_materialize_
        composite_call again -- the same three exhaustive shapes, one
        level over.

        This function-scoped, not loop-scoped, matters: ir/addr_value
        are plain local variables, reused across every argument in
        args, not fresh per iteration. Silently leaving them unbound
        on a first ARRAY/STRUCT argument crashes (Python's own
        UnboundLocalError) -- but leaving them unbound on a SECOND one,
        after an earlier argument already assigned them successfully,
        would silently reuse that EARLIER argument's own address for
        this one instead, duplicating it into arg_values rather than
        crashing at all. The explicit CodegenError above closes that
        silent-duplication risk too, not just the unclear-crash one.

        Both of _ir_materialize_array_literal/_ir_materialize_struct_
        literal are new here -- reusing a reservation _collect_
        argument_temps has made for exactly this AST position since
        before this arc's own addressable-base work existed at all;
        this is that reservation finally getting a real-IR consumer of
        its own, not a new mechanism. A struct-literal Call sitting
        anywhere as an ARRAY-typed argument needs no case of its own:
        no struct literal is ever ARRAY-typed in the first place."""
        arg_ir = []
        arg_values = []
        for arg in args:
            arg_type = type_of(arg)
            if arg_type.kind == TypeKind.ARRAY:
                if isinstance(arg, (Variable, Field, Index)):
                    result = self._ir_array_address(arg)
                    if result is None:
                        raise CodegenError(
                            f"_ir_array_address returned None for an ARRAY-typed "
                            f"Variable/Field/Index argument ({arg!r}) -- expected to "
                            f"always succeed for this shape")
                    ir, addr_value = result
                elif isinstance(arg, ArrayLiteral):
                    result = self._ir_materialize_array_literal(arg)
                    if result is None:
                        raise CodegenError(
                            f"_ir_materialize_array_literal returned None for an "
                            f"ARRAY-typed literal argument ({arg!r}) -- some element "
                            f"is out of scope for real IR, with no old-style fallback "
                            f"remaining to catch it")
                    ir, addr_value = result
                elif self._is_ordinary_composite_call(arg):
                    ir, addr_value = self._ir_materialize_composite_call(arg, arg_type)
                else:
                    raise CodegenError(
                        f"No codegen rule for an ARRAY-typed call argument of shape "
                        f"{type(arg).__name__}: {arg!r}")
                arg_ir.extend(ir)
                arg_values.append(addr_value)
            elif arg_type.kind == TypeKind.STRUCT:
                if isinstance(arg, (Variable, Field, Index)):
                    result = self._ir_struct_address(arg)
                    if result is None:
                        raise CodegenError(
                            f"_ir_struct_address returned None for a STRUCT-typed "
                            f"Variable/Field/Index argument ({arg!r}) -- expected to "
                            f"always succeed for this shape")
                    ir, addr_value = result
                elif isinstance(arg, Call) and arg.name in self.host.struct_registry:
                    result = self._ir_materialize_struct_literal(arg)
                    if result is None:
                        raise CodegenError(
                            f"_ir_materialize_struct_literal returned None for a "
                            f"STRUCT-typed literal argument ({arg!r}) -- some field "
                            f"is out of scope for real IR, with no old-style fallback "
                            f"remaining to catch it")
                    ir, addr_value = result
                elif self._is_ordinary_composite_call(arg):
                    ir, addr_value = self._ir_materialize_composite_call(arg, arg_type)
                else:
                    raise CodegenError(
                        f"No codegen rule for a STRUCT-typed call argument of shape "
                        f"{type(arg).__name__}: {arg!r}")
                arg_ir.extend(ir)
                arg_values.append(addr_value)
            elif arg_type.kind == TypeKind.SLICE or isinstance(arg, NoneLiteral):
                result = self._ir_slice_arg(arg)
                if result is None:
                    raise CodegenError(
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

        Argument marshaling is per-argument, not per-call: each
        argument's own shape (scalar, array/struct address, or a
        slice's own {ptr, len, cap} triple) is computed independently
        into its own Temp(s) first, and IRCall's own lowering is what
        places all of them into argument registers, together,
        immediately before the call -- the same "a Temp's home is
        independent of what computed it" property this whole arc has
        relied on repeatedly. This is also what makes IRCall's own
        lowering never need a push/pop dance between arguments (see
        its own comment): nothing a Temp could be assigned to overlaps
        an argument register until IRCall's own lowering runs.

        See _ir_call_arguments for the actual per-argument dispatch
        (shared with _ir_composite_call, for a composite-returning
        call, which needs the identical marshaling for its own
        ordinary arguments, just with a hidden pointer prepended and
        dst=None instead of an ordinary result Temp).

        Returns (ir, t_result), t_result being None for a void call.
        """
        total_slots = total_arg_slots(expr.args)
        if total_slots > 6:
            raise CodegenError(
                f"Call to '{expr.name}' needs {total_slots} argument "
                f"register(s) (a slice-typed argument needs 3); this "
                f"compiler only supports up to 6 (passed via registers "
                f"per the SysV ABI -- stack-passed arguments aren't "
                f"implemented)"
            )
        result_type = type_of(expr)
        t_result = None if result_type == Type.VOID else self.host.ids.new_temp(result_type)
        arg_ir, arg_values = self._ir_call_arguments(expr.args)
        ir = arg_ir + [IRCall(dst=t_result, name=expr.name, args=arg_values)]
        return ir, t_result

    def _ir_composite_call(self, dst_address, call_expr: Call, value_type: Type) -> list:
        """Builds (without lowering) a composite-returning function
        call's IR, writing its result through dst_address -- an
        ordinary INT64-typed IRValue, however the caller already has
        it: a freshly-computed _ir_array_address/_ir_struct_address/
        _ir_slice_address for a VarDecl/Assign/IndexAssign/FieldAssign
        target, or the current function's own received hidden pointer
        for a forwarding `return someFn()` (see gen_statement_ir's own
        Return case). This method doesn't care which -- both are just
        an address to it.

        The key realization making this possible with no new IR
        concept at all: the hidden-pointer convention IS just
        "dst_address as args[0], every genuine argument shifted one
        register position later, dst=None since the callee never
        leaves a scalar result in %eax at all" -- which IRCall's own,
        already-generic lowering (`for i, arg_value in enumerate(
        instr.args): load into ARG_REGISTERS_32[i]`) already produces
        exactly, with no changes needed to IRCall itself. dst=None
        here means "nothing to capture from %eax," not "this call is
        void" -- the two happen to coincide for every OTHER IRCall
        this compiler builds, but not this one: the call is
        definitely not void at the Hornet-language level, it just has
        no scalar result for %eax to hold.

        Reuses _ir_call_arguments directly for the genuine arguments
        (the identical per-argument real-IR-or-fallback dispatch
        _ir_call itself uses), then prepends dst_address as the
        actual first argument value -- nothing about that loop needs
        to know a hidden pointer is involved at all."""
        total_slots = 1 + total_arg_slots(call_expr.args)
        if total_slots > 6:
            raise CodegenError(
                f"Call to '{call_expr.name}' needs {total_slots} argument "
                f"register(s) (the hidden return pointer needs its own "
                f"slot, plus 3 for a slice-typed argument); this "
                f"compiler only supports up to 6 (passed via registers "
                f"per the SysV ABI -- stack-passed arguments aren't "
                f"implemented)"
            )
        arg_ir, arg_values = self._ir_call_arguments(call_expr.args)
        return arg_ir + [IRCall(dst=None, name=call_expr.name, args=[dst_address] + arg_values)]

    def _ir_short_circuit(self, expr: Binary, *, short_circuit_value: int, label_prefix: str) -> tuple[list, object]:
        """Builds (without lowering) the shared IR for AND/OR --
        mirror images of each other: each evaluates its left side and
        branches on it, jumping past the right side entirely (never
        emitting the IR that would compute it) if that alone already
        decides the answer -- AND with short_circuit_value=0 (left
        false makes the whole thing false without evaluating further),
        OR with short_circuit_value=1 (left true makes the whole thing
        true). This is what makes `0 and (1 / 0)` return 0 instead of
        crashing: the division is real IR, built and lowered like any
        other, but control flow jumps clean over it. Both left and
        right feed an IRBranch on their own truthiness, sharing one
        `short` label whichever one triggers it (`targets` below picks
        between "go evaluate right" for left, or "use the canonical
        fallthrough value" for right -- the only difference between
        the two). Returns (ir, t_result)."""
        fallthrough_value = 1 - short_circuit_value
        rhs_label = self.host.ids.new_label(f"{label_prefix}_rhs")
        short_label = self.host.ids.new_label(f"{label_prefix}_short")
        fallthrough_label = self.host.ids.new_label(f"{label_prefix}_fallthrough")
        end_label = self.host.ids.new_label(f"{label_prefix}_end")

        def targets(continue_label: str) -> tuple[str, str]:
            # (true_target, false_target): whichever outcome matches
            # short_circuit_value goes to `short_label`; the other
            # goes to `continue_label`.
            if short_circuit_value == 1:
                return short_label, continue_label
            return continue_label, short_label

        t_result = self.host.ids.new_temp(Type.BOOL)
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
            IRLabel(end_label),
        ]
        return ir, t_result

