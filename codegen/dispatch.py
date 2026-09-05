"""The routing layer: gen_expr_into and gen_binary_into are the only
two methods in this codebase whose actual job is to inspect an AST
node's type or kind and hand off to whichever feature file
(arrays_slices, structs, strings, scalars) owns that case -- every
other dispatch-shaped method elsewhere is really just implementing one
branch of one of these two."""

from codegen.assembly_ast import Operand, Instruction, MovQ, Imm, Mov, Memory, Register
from codegen.errors import CodegenError
from codegen.ir import IRRaw, IRBinOp
from codegen.utils import as_qword_register, type_of
from parser import (
    Node,
    Constant,
    BoolLiteral,
    StringLiteral,
    ArrayLiteral,
    Slice,
    NoneLiteral,
    Variable,
    Index,
    Field,
    Call,
    Cast,
    Unary,
    Binary,
    BinaryOp,
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

    def gen_binary_into(self, expr: Binary, dst: Operand) -> list[Instruction]:
        """Computes `expr.left OP expr.right` into `dst`.

        AND/OR are handled separately (see gen_short_circuit) since
        they must not unconditionally evaluate both sides. Every other
        binary operator -- arithmetic and comparisons alike -- goes
        through the stack-spill scheme below, which always evaluates
        both sides. Requires `dst` to be a register (a real 32-bit
        register and its 64-bit alias get pushed/popped along the way,
        which an Imm can't do).
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
        # which can't flow through the single-register stack-spill
        # scheme below. semantic.py's check_binary already guarantees
        # exactly one side is slice-typed and the other none-typed by
        # the time this is reached.
        #
        # ARRAY and STRUCT equality are dispatched the same way, for
        # the same reason: neither value fits in a single register.
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
        # scalars) is now built as a tiny, self-contained IR fragment
        # -- evaluate each operand into its own Temp, combine via
        # IRBinOp, read the result back into `dst` -- rather than the
        # hand-woven push/pop dance this used to do directly. lower_ir
        # reuses gen_binary_op (unchanged) as this op's own instruction-
        # selection rule, so the actual arithmetic -- including the
        # int64-vs-32-bit view decision -- isn't reimplemented here.
        operand_type = type_of(expr.left)  # left and right are guaranteed the same type by semantic.py's own check_binary
        t_left = self._new_temp(operand_type)
        t_right = self._new_temp(operand_type)
        t_result = self._new_temp(type_of(expr))
        instructions = self.lower_ir([
            IRRaw(self.gen_expr_into(expr.left, Register('eax')), dst=t_left),
            IRRaw(self.gen_expr_into(expr.right, Register('eax')), dst=t_right),
            IRBinOp(dst=t_result, op=expr.op, left=t_left, right=t_right),
        ])
        instructions.extend(self._gen_read_scalar_into(self._temp_mem(t_result), t_result.type, dst))
        return instructions
