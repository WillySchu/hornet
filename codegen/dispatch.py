"""TODO"""

from codegen.assembly_ast import Operand, Instruction, MovQ, Imm, Mov, Memory, Je, Register, Jne, Push, Pop
from codegen.errors import CodegenError
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
            # of one (VarDecl init, Assign, a nested literal element)
            # routes through gen_array_value_into/gen_array_literal_into
            # instead of ever calling gen_expr_into on it directly. A
            # clear error here catches a codegen bug immediately rather
            # than silently truncating an array down to whatever
            # happens to fit in %eax.
            raise CodegenError(
                "Cannot compute an array literal via gen_expr_into -- "
                "arrays don't fit in a single register; use "
                "gen_array_value_into instead"
            )
        if isinstance(expr, Slice):
            # Never reachable in correct codegen -- a slice's value is
            # a 24-byte {ptr, len, cap} descriptor, which can't fit in a
            # single register either. Every producer of one (VarDecl
            # init, Assign) routes through gen_slice_value_into/
            # gen_slice_into instead of ever calling gen_expr_into on
            # it directly -- see ArrayLiteral's own case just above
            # for the identical reasoning.
            raise CodegenError(
                "Cannot compute a slice expression via gen_expr_into -- "
                "slices don't fit in a single register; use "
                "gen_slice_value_into instead"
            )
        if isinstance(expr, NoneLiteral):
            # Never reachable in correct codegen either, for a
            # different reason than ArrayLiteral/Slice above: it's not
            # a SIZE problem here at all (rejected for the same size
            # reason Slice is, now that none's own {0, 0, 0} descriptor
            # is exactly as wide as any other slice's) -- it's that
            # none has no ONE fixed target type of its own to compute
            # INTO -- see gen_none_into's own docstring for why its
            # callers (gen_var_decl/gen_assign's own NoneLiteral
            # short-circuit) have to already know and pass the target
            # type explicitly, something gen_expr_into's own signature
            # has no way to supply. A slice-vs-none comparison (`s ==
            # none`) is handled entirely separately too, via
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
                # Reading a sub-array (e.g. `matrix[i]` alone, not yet
                # fully indexed down to a scalar) has the same "doesn't
                # fit in a register" problem as an array literal --
                # `[3]int row = matrix[i]` is handled via
                # gen_array_value_into instead, which calls
                # gen_array_address_into directly rather than ever
                # reaching this method for the sub-array's VALUE.
                raise CodegenError(
                    "Cannot read a sub-array via gen_expr_into -- arrays "
                    "don't fit in a single register; use "
                    "gen_array_value_into or gen_array_address_into instead"
                )
            if element_type.kind == TypeKind.STRUCT:
                # Same reasoning, for a struct-typed array element
                # (`rows[i]` where rows is an array of structs) --
                # `Point p = rows[i]` is handled via gen_struct_value_
                # into instead.
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
                # Never reachable in correct codegen -- see ArrayLiteral
                # and the Index sub-array case just above for the same
                # reasoning. An array-returning call's result is only
                # ever consumed via gen_array_value_into's own Call case
                # (which writes it, through the hidden-pointer
                # convention, straight into a given destination), never
                # by trying to land it in a single register here.
                raise CodegenError(
                    f"Cannot call '{expr.name}' (which returns an array) "
                    f"via gen_expr_into -- arrays don't fit in a single "
                    f"register; use gen_array_value_into instead"
                )
            if type_of(expr).kind == TypeKind.SLICE:
                # Same reasoning, for the same underlying cause: a
                # slice-returning call now writes its result through
                # the hidden-pointer convention too (see gen_slice_
                # call_into), just like an array-returning one, never
                # by trying to land it in a single register here. Only
                # ever reached via gen_slice_value_into's own Call case
                # (VarDecl/Assign) or gen_return's own forwarding case.
                raise CodegenError(
                    f"Cannot call '{expr.name}' (which returns a slice) "
                    f"via gen_expr_into -- a slice descriptor doesn't "
                    f"fit in a single register; use gen_slice_call_into "
                    f"instead"
                )
            if type_of(expr).kind == TypeKind.STRUCT:
                # Same reasoning again, for a struct-returning call --
                # only ever reached via gen_struct_value_into's own
                # Call case or gen_return's own forwarding case.
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
            # Compute the source expression into dst first (already
            # correctly widened if it was itself int8/uint8-typed --
            # see _gen_read_scalar_into), then re-narrow dst's own
            # LOW BYTE if the target is int8/uint8 -- see gen_cast_
            # narrowing_into's own docstring for why this single,
            # register-to-register instruction is enough regardless
            # of what the SOURCE type actually was. A target of int
            # needs nothing further at all: the source's own already-
            # widened value already IS a valid int.
            instructions = self.gen_expr_into(expr.expr, dst)
            instructions.extend(self.gen_cast_narrowing_into(expr.resolved_type, dst))
            return instructions
        if isinstance(expr, Unary):
            # Compute the operand into dst first, then apply this node's
            # operator to whatever's now there. This is what makes chained
            # operators (`~-2`) work: the inner Unary's instructions run
            # first, then the outer operator's instructions run on top.
            #
            # operand_type reads type_of(expr) -- this OUTER node's own
            # resolved_type -- not type_of(expr.operand). The two are
            # ordinarily identical (check_unary's own rule is "stays the
            # operand's own type"), EXCEPT for a widened literal: `int64
            # x = -5` sets resolved_type to int64 on the OUTER Unary node
            # (see _check_value_flowing_into's own case 3), but the INNER
            # Constant(5) node's own resolved_type is still whatever
            # check_expr's earlier, ordinary recursive pass already set
            # it to (Type.INT) -- never updated, since the widening logic
            # only ever touches the outermost node of a literal
            # expression. Reading the inner one here was a real, found
            # bug: it silently fed the wrong operand_type into gen_
            # unary_op, using 32-bit Neg instead of NegQ for `-5` widened
            # to int64 -- invisible for int8/uint8 only because THEIR
            # own unary dispatch never branched on operand_type at all
            # before int64 existed, so any operand_type value produced
            # the identical, correct 32-bit instruction either way.
            instructions = self.gen_expr_into(expr.operand, dst)
            instructions.extend(self.gen_unary_op(expr.op, dst, operand_type=type_of(expr)))
            return instructions
        if isinstance(expr, Binary):
            # ADD and the two equality operators are overloaded for str
            # (concatenation and strcmp-backed comparison respectively;
            # see the module docstring's STRINGS section) -- everything
            # else, and ADD/==/!= between two ints or bools, goes
            # through the original gen_binary_into completely unchanged.
            if expr.op == BinaryOp.ADD and type_of(expr.left) == Type.STR:
                return self.gen_string_concat_into(expr, dst)
            if expr.op in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL) and type_of(expr.left) == Type.STR:
                return self.gen_string_compare_into(expr, dst)
            return self.gen_binary_into(expr, dst)
        raise CodegenError(f"No codegen rule for expression: {expr!r}")

    def gen_binary_into(self, expr: Binary, dst: Operand) -> list[Instruction]:
        """Computes `expr.left OP expr.right` into `dst`.

        AND/OR are handled entirely separately (see gen_short_circuit)
        since they must not unconditionally evaluate both sides. Every
        other binary operator -- arithmetic and comparisons alike -- goes
        through the stack-spill scheme described in the module
        docstring, which always evaluates both sides. Requires `dst` to
        be a register (there's a real 32-bit register and its 64-bit
        alias pushed/popped along the way, which an Imm can't do).
        """
        if expr.op == BinaryOp.AND:
            return self.gen_short_circuit(
                expr, dst,
                short_circuit_jump=Je,   # jump early when the left side is already false
                short_circuit_value=0,   # ...and the overall result is false
                fallthrough_value=1,     # both sides were truthy -> true
                label_prefix="and",
            )
        if expr.op == BinaryOp.OR:
            return self.gen_short_circuit(
                expr, dst,
                short_circuit_jump=Jne,  # jump early when the left side is already true
                short_circuit_value=1,   # ...and the overall result is true
                fallthrough_value=0,     # both sides were falsy -> false
                label_prefix="or",
            )

        if not isinstance(dst, Register):
            raise CodegenError(f"Binary codegen requires a register destination, got: {dst!r}")

        # A slice compared to `none` (in either order -- `s == none`
        # and `none == s` both reach here) needs its own dedicated
        # codegen too, for a related but distinct reason from AND/OR
        # above: a slice's "value" is a 24-byte descriptor, which
        # can't flow through the ordinary single-register stack-spill
        # scheme below the way an int/bool/str value can. semantic.py's
        # check_binary already guarantees, by the time this is reached,
        # that exactly one side is slice-typed and the other is none-
        # typed -- a real slice compared to another real slice
        # (`s1 == s2`), or none compared to none, is rejected earlier
        # -- so this doesn't need to re-derive or defensively check
        # which side is which beyond that.
        #
        # ARRAY and STRUCT equality are dispatched the same way, for
        # the same root reason: neither one's own value fits through a
        # single register the way an int/bool/str value does.
        # semantic.py's own check_binary already guarantees, by the
        # time this is reached, that both sides are the exact same
        # array or struct type, and that the type is actually
        # comparable (see _is_comparable_type) -- no slice anywhere
        # inside it, directly or nested through a struct field or an
        # array element -- see gen_array_equality_into/gen_struct_
        # equality_into's own docstrings for how each dispatches
        # internally.
        if expr.op in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL):
            if type_of(expr.left).kind == TypeKind.SLICE or type_of(expr.right).kind == TypeKind.SLICE:
                return self.gen_slice_none_comparison_into(expr, dst)
            if type_of(expr.left).kind == TypeKind.ARRAY:
                return self.gen_array_equality_into(expr, dst)
            if type_of(expr.left).kind == TypeKind.STRUCT:
                return self.gen_struct_equality_into(expr, dst)

        scratch = Register('ecx')  # holds the right-hand value while combining
        operand_type = type_of(expr.left)  # left and right are guaranteed the same type by semantic.py's own check_binary
        instructions = self.gen_expr_into(expr.left, dst)   # dst = left
        instructions.append(Push(as_qword_register(dst)))   # save left on the stack
        instructions.extend(self.gen_expr_into(expr.right, dst))  # dst = right (left is safe)
        if operand_type == Type.INT64:
            # An ordinary 32-bit Mov here would silently drop the
            # right-hand value's own high 32 bits -- scratch has to
            # receive the SAME 64-bit view dst's own value was just
            # computed into (see gen_expr_into's own Constant/Variable/
            # Binary/Unary/Cast cases, all of which compute an int64
            # result into dst's 64-bit view specifically for this
            # reason), not just its low half.
            instructions.append(MovQ(src=as_qword_register(dst), dst=as_qword_register(scratch)))
        else:
            instructions.append(Mov(src=dst, dst=scratch))       # scratch = right
        instructions.append(Pop(as_qword_register(dst)))     # dst = left (restored)
        instructions.extend(self.gen_binary_op(expr.op, src=scratch, dst=dst, operand_type=operand_type))
        return instructions
