"""TODO"""
from codegen.assembly_ast import Operand, Instruction, MovQ, Imm, Mov, Memory, Je, Jne, Register, Push, Pop, AddQ, SubQ, \
    IMulQ, Cqto, IDivQ, AndQ, OrQ, XorQ, ShiftLeftQ, ShiftRightArithmeticQ, Add, Sub, IMul, Cdq, IDiv, And, Or, Xor, \
    ShiftLeft, ShiftRightArithmetic, CmpQ, Cmp, SetCC, MovZX, NegQ, Neg, NotQ, Not, MovSX, MovSXD, MovB, CallInstr, LeaQ
from codegen.errors import CodegenError
from codegen.utils import as_qword_register, type_of, COMPARISON_CONDITION_CODES, as_byte_register
from parser import Node, Constant, BoolLiteral, StringLiteral, ArrayLiteral, Slice, NoneLiteral, Variable, Index, Field, \
    Call, Cast, Unary, Binary, BinaryOp, UnaryOp
from semantic import Type, TypeKind


class ScalarsMixin:
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

    def gen_binary_op(
            self, op: BinaryOp, src: Operand, dst: Operand, operand_type: Type = Type.INT) -> list[Instruction]:
        """Emits the actual operator instruction(s) for `op`, given
        that `src`/`dst` already hold the right-hand/left-hand values
        (see gen_binary_into's own orchestration for how they get
        there). `src`/`dst` are always passed as their ordinary
        32-bit-named register (e.g. 'eax'/'ecx'), the SAME convention
        every other caller in this file follows -- this method itself
        decides internally, via `operand_type`, whether to actually
        operate on that register's own 64-bit VIEW (as_qword_register)
        for int64, exactly the same "caller always passes the 32-bit
        name, the callee decides which view to use" pattern _gen_read_
        scalar_into/_gen_write_scalar_from already established for
        int8/uint8/int64's own storage access -- rather than pushing
        that decision out onto gen_binary_into or every other call
        site.

        A COMPARISON's own RESULT, though, is always an ordinary
        32-bit bool regardless of operand_type: SetCC/MovZX always
        target dst's own 32-bit view even when the comparison itself
        (Cmp vs CmpQ) operated on its 64-bit one, since a bool value
        is never itself wider than 4 bytes no matter how wide the two
        values being compared were -- this is why the comparison
        branch converts to a 64-bit view locally, just for the Cmp/
        CmpQ instruction itself, rather than reusing a single dst64
        variable the way every arithmetic branch above it does."""
        is_64bit = operand_type == Type.INT64
        if is_64bit and op not in COMPARISON_CONDITION_CODES:
            src64 = as_qword_register(src)
            dst64 = as_qword_register(dst)
            if op == BinaryOp.ADD:
                return [AddQ(src=src64, dst=dst64)]
            if op == BinaryOp.SUBTRACT:
                return [SubQ(src=src64, dst=dst64)]
            if op == BinaryOp.MULTIPLY:
                return [IMulQ(src=src64, dst=dst64)]
            if op == BinaryOp.DIVIDE:
                # idivq divides %rdx:%rax by its operand, so by the time
                # this runs, the dividend (dst64, i.e. left) must be in
                # %rax and the divisor (src64, i.e. right) must be in a
                # register -- both guaranteed by how gen_binary_into
                # calls this, exactly like the 32-bit case below.
                if dst64 != Register('rax'):
                    raise CodegenError("Division currently requires its destination to be %rax")
                return [Cqto(), IDivQ(src64)]
            if op == BinaryOp.MODULO:
                # Exactly the same Cqto+IDivQ sequence as DIVIDE --
                # idivq always computes both the quotient (%rax) and the
                # remainder (%rdx) in one instruction -- just followed
                # by moving the remainder into dst64 instead of leaving
                # the quotient there.
                if dst64 != Register('rax'):
                    raise CodegenError("Modulo currently requires its destination to be %rax")
                return [Cqto(), IDivQ(src64), MovQ(src=Register('rdx'), dst=Register('rax'))]
            if op == BinaryOp.BITWISE_AND:
                return [AndQ(src=src64, dst=dst64)]
            if op == BinaryOp.BITWISE_OR:
                return [OrQ(src=src64, dst=dst64)]
            if op == BinaryOp.BITWISE_XOR:
                return [XorQ(src=src64, dst=dst64)]
            if op == BinaryOp.SHIFT_LEFT:
                # `src64` is never referenced here -- ShiftLeftQ
                # hardcodes %cl as its count operand, the identical
                # reason ShiftLeft's own docstring already explains one
                # register-width down; the count itself is never wider
                # than a byte regardless of the value being shifted.
                return [ShiftLeftQ(dst=dst64)]
            if op == BinaryOp.SHIFT_RIGHT:
                return [ShiftRightArithmeticQ(dst=dst64)]
            raise CodegenError(f"No codegen rule for binary operator: {op}")

        if op == BinaryOp.ADD:
            return [Add(src=src, dst=dst)]
        if op == BinaryOp.SUBTRACT:
            return [Sub(src=src, dst=dst)]
        if op == BinaryOp.MULTIPLY:
            return [IMul(src=src, dst=dst)]
        if op == BinaryOp.DIVIDE:
            # idivl divides %edx:%eax by its operand, so by the time this
            # runs, the dividend (`dst`, i.e. left) must be in %eax and
            # the divisor (`src`, i.e. right) must be in a register --
            # both guaranteed by how gen_binary_into calls this.
            if dst != Register('eax'):
                raise CodegenError("Division currently requires its destination to be %eax")
            return [Cdq(), IDiv(src)]
        if op == BinaryOp.MODULO:
            # Exactly the same Cdq+IDiv sequence as DIVIDE -- idivl
            # always computes both the quotient (%eax) and the remainder
            # (%edx) in one instruction -- just followed by moving the
            # remainder into dst instead of leaving the quotient there.
            if dst != Register('eax'):
                raise CodegenError("Modulo currently requires its destination to be %eax")
            return [Cdq(), IDiv(src), Mov(src=Register('edx'), dst=Register('eax'))]
        if op == BinaryOp.BITWISE_AND:
            return [And(src=src, dst=dst)]
        if op == BinaryOp.BITWISE_OR:
            return [Or(src=src, dst=dst)]
        if op == BinaryOp.BITWISE_XOR:
            return [Xor(src=src, dst=dst)]
        if op == BinaryOp.SHIFT_LEFT:
            # `src` (== %ecx, per gen_binary_into) is never referenced
            # here -- ShiftLeft hardcodes %cl as its count operand,
            # since that's the only register x86 allows there, and %ecx
            # is already where the right-hand operand ends up by the
            # time gen_binary_op is called for any binary operator.
            return [ShiftLeft(dst=dst)]
        if op == BinaryOp.SHIFT_RIGHT:
            return [ShiftRightArithmetic(dst=dst)]
        if op in COMPARISON_CONDITION_CODES:
            # Cmp(src=right, dst=left) computes (left - right) and sets
            # flags from that; SetCC turns the relevant flag combination
            # into a 0/1 byte; MovZX zero-extends that byte back out to
            # fill the full destination register (same pattern used for
            # NOT -- see gen_unary_op -- just against a computed `right`
            # instead of the literal 0). For a 64-bit operand_type, the
            # comparison ITSELF (CmpQ, against the 64-bit views) needs
            # the full value to compare correctly -- comparing only the
            # low 32 bits could, e.g., call two large int64 values equal
            # when they aren't -- but the RESULT byte/register (byte_dst,
            # dst) stays exactly as it already was: a bool is always
            # 32-bit-or-narrower regardless of what was being compared.
            byte_dst = as_byte_register(dst)
            cmp_instr = CmpQ(
                src=as_qword_register(src), dst=as_qword_register(dst)) if is_64bit else Cmp(src=src, dst=dst)
            return [
                cmp_instr,
                SetCC(cc=COMPARISON_CONDITION_CODES[op], operand=byte_dst),
                MovZX(src=byte_dst, dst=dst),
            ]
        raise CodegenError(f"No codegen rule for binary operator: {op}")

    def gen_unary_op(self, op: UnaryOp, dst: Operand, operand_type: Type = Type.INT) -> list[Instruction]:
        """`operand_type` follows the identical convention gen_binary_
        op's own new parameter does -- `dst` is always passed as its
        ordinary 32-bit-named register, and this method decides
        internally whether to operate on its 64-bit view for int64.
        UnaryOp.NOT never reaches the int64 branch at all: `not`
        requires a bool operand (see check_unary), which int64 can
        never be, so its own Cmp-against-0/SetCC/MovZX sequence stays
        exactly as it always has, unconditionally 32-bit."""
        if op == UnaryOp.NEGATE:
            if operand_type == Type.INT64:
                return [NegQ(as_qword_register(dst))]
            return [Neg(dst)]
        if op == UnaryOp.COMPLEMENT:
            if operand_type == Type.INT64:
                return [NotQ(as_qword_register(dst))]
            return [Not(dst)]
        if op == UnaryOp.NOT:
            # `not x` is "1 if x == 0, else 0" -- the same cmp/setCC/movzx
            # pattern used for comparisons, just always against 0 and
            # always with cc='e'.
            byte_dst = as_byte_register(dst)
            return [
                Cmp(src=Imm(0), dst=dst),
                SetCC(cc='e', operand=byte_dst),
                MovZX(src=byte_dst, dst=dst),
            ]
        raise CodegenError(f"No codegen rule for unary operator: {op}")

    def gen_cast_narrowing_into(self, target_type: Type, dst: Register) -> list[Instruction]:
        """The actual work behind an explicit `TYPE(expr)` cast (see
        gen_expr_into's own Cast case): re-narrows `dst`'s own value
        to genuinely, correctly represent target_type, given that
        gen_expr_into has already computed the SOURCE expression into
        it (already correctly widened, if the source happened to be
        int8/uint8-typed itself -- see _gen_read_scalar_into).

        A target of int needs NOTHING further: the source's own
        already-widened 32-bit value already IS a valid int, whatever
        the source type actually was (an int8/uint8 source is already
        sign/zero-extended; an int source needs no widening at all).

        A target of int8 or uint8 needs exactly ONE more instruction:
        MovSX (int8) or MovZX (uint8) applied to dst's OWN low-byte
        alias, writing the result back into dst itself -- a single,
        purely register-to-register re-widening (MovSX/MovZX's own
        src operand doesn't have to be memory; see MovZX's own
        docstring), no memory round-trip needed at all. This is
        DELIBERATE, not just an optimization: a cast's own RESULT has
        to be a genuinely, correctly narrowed value immediately, not
        merely "correct once eventually written to int8/uint8-typed
        storage" the way _gen_write_scalar_from's own truncation is --
        `int8(300) + int8(5)` needs 300 already wrapped to 44 BEFORE
        the addition happens, or the arithmetic itself would silently
        be wrong, since every later int8/uint8 operation assumes its
        own operands already correctly represent a narrow value, never
        re-validating that itself.

        Correct regardless of what the source type actually was, not
        just for a narrowing int-to-int8/uint8 cast: re-extending
        whatever's already sitting in the low byte is exactly as
        correct for a same-width REINTERPRETATION (int8-to-uint8 or
        back) as it is for genuine narrowing, since both are really
        the same operation -- "take the low byte, reinterpret it under
        a new sign convention" -- differing only in whether the high
        bytes being discarded happened to already be a trivial (int8/
        uint8 source) or a real (int source) sign/zero-extension of
        that byte. Verified against concrete cases during design, not
        just asserted: int(300) as int8 gives 44 (300's own low byte,
        0x2C, has its high bit clear, so sign-extension leaves it
        positive); int(200) as int8 gives -56 (200's own low byte,
        0xC8, has its high bit set, so sign-extension correctly
        produces the negative two's-complement reinterpretation) --
        both match a real 8-bit truncate-then-reinterpret exactly.

        A target of int64 needs exactly one instruction too, in the
        OPPOSITE direction: MovSXD, sign-extending dst's own 32-bit
        view up into its 64-bit one -- correct regardless of whether
        the source was int, int8, or uint8, since all three are
        already read into a genuinely correct, ordinary 32-bit value
        by the time this runs (see _gen_read_scalar_into), and a non-
        negative 32-bit value's own sign bit is already clear, so
        sign-extending it produces the identical result zero-extending
        it would have. NARROWING out of int64 (int64(x) targeting int,
        int8, or uint8) needs NO new instruction here at all: dst's own
        32-bit view is always simply the low half of whatever's in its
        64-bit view, so falling through to the existing int/int8/uint8
        branches above -- which already operate on dst's own 32-bit
        view or its own low-byte alias -- is already exactly correct,
        the same way it already was before int64 existed."""
        if target_type == Type.INT8:
            return [MovSX(src=as_byte_register(dst), dst=dst)]
        if target_type == Type.UINT8:
            return [MovZX(src=as_byte_register(dst), dst=dst)]
        if target_type == Type.INT64:
            return [MovSXD(src=dst, dst=as_qword_register(dst))]
        return []

    def _gen_read_scalar_into(self, mem: Memory, t: Type, dst: Register) -> list[Instruction]:
        """Reads a scalar value of type `t` (int, int8, uint8, int64,
        or bool) from `mem` into `dst` -- the one choke point every
        scalar READ site in this file goes through, so int8/uint8's
        own genuinely narrow (1-byte) storage AND int64's own genuinely
        wide (8-byte) storage (see type_byte_width) only ever needed
        teaching to ONE place, not rediscovering at every Variable/
        Field/Index read site individually.

        int8 needs a SIGN-extending read (MovSX) and uint8 a ZERO-
        extending one (MovZX, already built for SetE's own unrelated
        need) rather than an ordinary 4-byte Mov, which would read
        three bytes of adjacent memory that were never part of this
        value at all -- and for int8 specifically, would also silently
        misinterpret a negative value as a large positive one (int8(-1)
        == 0xFF read as a raw 4-byte int would become 0x000000FF ==
        255, not -1) even if the adjacent bytes happened to be zero.
        Every later arithmetic/comparison instruction in this file
        already assumes it's operating on a genuinely correct 32-bit
        value, so getting the WIDENING right here, once, is what lets
        everything downstream stay completely unaware int8/uint8 are
        narrower than int at all.

        int64 needs a full 8-byte read (MovQ) into `dst`'s own 64-bit
        VIEW (as_qword_register(dst), e.g. %eax -> %rax) -- `dst`
        itself is always passed as a 32-bit-named register by every
        caller in this file (the same convention str's own handling
        already established elsewhere in gen_expr_into), with THIS
        method responsible for deciding which actual view to read
        into, exactly the way int8/uint8's own narrowing decision is
        made here rather than pushed onto every caller. An ordinary
        4-byte Mov here would silently drop int64's own high 32 bits
        entirely, not just read a stale/incorrect value -- reading is
        the one direction narrower-than-needed storage access can
        outright discard real, distinct bits of a value rather than
        merely mis-INTERPRET already-present ones the way int8/uint8's
        own narrow case can.

        int and bool are untouched -- an ordinary 4-byte Mov, exactly
        as before this method existed."""
        if t == Type.INT8:
            return [MovSX(src=mem, dst=dst)]
        if t == Type.UINT8:
            return [MovZX(src=mem, dst=dst)]
        if t == Type.INT64:
            return [MovQ(src=mem, dst=as_qword_register(dst))]
        return [Mov(src=mem, dst=dst)]

    def _gen_write_scalar_from(self, src: Register, t: Type, dst_mem: Memory) -> list[Instruction]:
        """Writes a scalar value of type `t`, already computed into
        `src`, into `dst_mem` -- the WRITE-side counterpart to _gen_
        read_scalar_into, and the other half of the same one-choke-
        point principle: every scalar WRITE site in this file goes
        through this, rather than each one separately remembering that
        int8/uint8 need a narrower store or int64 a wider one.

        int8/uint8 need a 1-byte, TRUNCATING store (MovB, of src's own
        low-byte alias -- see as_byte_register) rather than an ordinary
        4-byte Mov, which would clobber whatever adjacent memory
        happens to immediately follow this value (an adjacent struct
        field, the next array element, ...) -- exactly the kind of
        silent, hard-to-diagnose corruption a narrow type's own
        storage existing at all is supposed to make possible to write
        correctly, not introduce a new way to get wrong.

        int64 needs a full 8-byte store (MovQ, of src's own 64-bit VIEW
        -- as_qword_register(src)) -- CALLERS are responsible for
        having already computed the value into that same 64-bit view
        before reaching this method (every gen_expr_into case that can
        produce an int64 result does exactly this -- see its own
        Constant/Variable/Binary/Unary/Cast cases), not just src's own
        low 32 bits: an ordinary 4-byte Mov here would write only the
        low half of a value whose own high half might be meaningful,
        and reading src's own 64-bit view when only the low 32 bits
        were ever actually computed would write whatever stale garbage
        happened to occupy that register's own high bits, silently
        corrupting the stored value in a way that could be very hard
        to trace back to its actual cause.

        int and bool are untouched -- an ordinary 4-byte Mov, exactly
        as before this method existed."""
        if t == Type.INT8 or t == Type.UINT8:
            return [MovB(src=as_byte_register(src), dst=dst_mem)]
        if t == Type.INT64:
            return [MovQ(src=as_qword_register(src), dst=dst_mem)]
        return [Mov(src=src, dst=dst_mem)]

    def gen_call_into(self, expr: Call, dst: Operand) -> list[Instruction]:
        """`name(arg1, arg2, ...)`: evaluates and passes every
        argument via the shared _gen_call_arguments_into (see its own
        docstring for the full push-then-pop-in-reverse discipline,
        and how a slice-typed argument's own three register slots are
        placed correctly among any ordinary scalar/array ones), then
        calls the function.

        The result already ends up exactly where gen_expr_into's
        contract expects it (%rax/%eax, matching `dst`, which is always
        Register('eax') throughout this file), so there's nothing left
        to move once the call returns. This method is never reached at
        all for a callee that returns an array or a slice -- see
        gen_array_call_into and gen_slice_call_into, which now share
        the exact same hidden-pointer return-value convention (see the
        module docstring's SLICE PARAMETERS AND RETURNS section), and
        that convention doesn't fit a single generic `dst` the way an
        ordinary scalar return does.
        """
        total_slots = self._total_arg_slots(expr.args)
        if total_slots > 6:
            raise CodegenError(
                f"Call to '{expr.name}' needs {total_slots} argument "
                f"register(s) (a slice-typed argument needs 3); this "
                f"compiler only supports up to 6 (passed via registers "
                f"per the SysV ABI -- stack-passed arguments aren't "
                f"implemented)"
            )
        if dst != Register('eax'):
            raise CodegenError(f"Call codegen requires dst == %eax, got: {dst!r}")

        instructions = self._gen_call_arguments_into(expr.args)
        instructions.append(CallInstr(expr.name))
        return instructions

    def _gen_zero_value_into(self, t: Type, dst_mem: Memory) -> list[Instruction]:
        """Writes t's own implicit zero value into dst_mem -- what a
        `T x` VarDecl with no initializer now gets, instead of the
        genuinely uninitialized memory it used to leave behind (see
        gen_var_decl's own note on the earlier, now-superseded
        behavior). Dispatches by kind:
          - int/bool/int8/uint8: an ordinary 0 -- a plain 4-byte write
            for int/bool, a 1-byte one (MovB) for int8/uint8, matching
            each type's own genuine storage width (see type_byte_
            width).
          - str: the address of a single shared, static empty-string
            constant (_get_empty_str_label) -- NEVER a null pointer;
            see that method's own docstring for why a null zero value
            would be an active hazard, not just an unusual choice.
          - slice: none's own {ptr: 0, len: 0, cap: 0} descriptor,
            reusing gen_none_into exactly as-is -- this needed no new
            code at all, since a zero-value slice and a none-valued
            one are, by design, the identical representation.
          - array: delegated to _gen_zero_array_into, which further
            dispatches on the array's own LEAF type -- see its own
            docstring for why int/bool/slice, str, and struct leaves
            each need a genuinely different strategy.
          - struct: every one of the struct's own fields, flattened
            via _flatten_struct_fields exactly the way struct equality
            already flattens them for comparison -- recursing back
            into THIS method for each field's own (never struct-kind,
            since flattening already unwrapped any nested struct away)
            type.

        dst_mem.base is protected via push/pop across EVERY field's own
        zero-fill, when it isn't 'rbp' (rbp itself is never at risk --
        nothing in this file ever treats it as scratch, so pushing and
        popping it would be actively wrong, not merely unnecessary; see
        gen_struct_literal_into's own identical `!= 'rbp'` guard for
        the same reasoning applied to a different write). Needed
        because the array case below computes a fresh address by
        calling _gen_address_of_memory_into with dst_mem.base itself as
        the destination register in some call shapes -- which, unlike
        every scalar/slice write here, can OVERWRITE dst_mem.base's own
        physical register in place. Without protecting it, a struct
        with an array-typed field followed by ANY other field would
        silently compute that later field's own address from garbage
        (whatever the array zero-loop happened to leave behind) instead
        of the struct's real base -- the exact register-collision
        failure mode _gen_struct_fields_equality_at_addresses's own
        docstring already documents at length for the identical reason,
        one construct over. Applied unconditionally, even for the
        scalar/slice cases that don't strictly need it, matching that
        same method's own "protect everything, don't try to be clever
        about which fields actually need it" posture."""
        if t.kind == TypeKind.STRUCT:
            protect_dst = dst_mem.base != 'rbp'
            instructions = []
            for field_type, offset in self._flatten_struct_fields(t.struct_name):
                field_mem = Memory(dst_mem.base, dst_mem.offset + offset)
                if protect_dst:
                    instructions.append(Push(Register(dst_mem.base)))
                instructions.extend(self._gen_zero_value_into(field_type, field_mem))
                if protect_dst:
                    instructions.append(Pop(Register(dst_mem.base)))
            return instructions
        if t.kind == TypeKind.ARRAY:
            return self._gen_zero_array_into(t, dst_mem)
        if t.kind == TypeKind.SLICE:
            return self.gen_none_into(dst_mem, t)
        if t == Type.STR:
            # Whichever of rax/rcx isn't dst_mem's own base -- a single
            # scratch register is all this needs, computed and consumed
            # in the same two instructions, with nothing relying on it
            # afterward.
            scratch = Register('rax') if dst_mem.base != 'rax' else Register('rcx')
            return [
                LeaQ(label=self._get_empty_str_label(), dst=scratch),
                MovQ(src=scratch, dst=dst_mem),
            ]
        if t == Type.INT8 or t == Type.UINT8:
            return [MovB(src=Imm(0), dst=dst_mem)]
        return [Mov(src=Imm(0), dst=dst_mem)]  # int or bool
