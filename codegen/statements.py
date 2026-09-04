"""TODO"""

from codegen.assembly_ast import Instruction, MovQ, Register, Memory, Imm, Push, Pop, Mov, Cmp, Je, Jmp, Label, Operand
from codegen.errors import CodegenError
from codegen.utils import type_of
from parser import Node, VarDecl, Assign, IndexAssign, FieldAssign, Return, If, While, Break, Continue, ExprStmt, \
    NoneLiteral, ArrayLiteral, Index, Slice, Binary
from semantic import TypeKind, Type


class StatementsMixin:
    def gen_statement(self, stmt: Node) -> list[Instruction]:
        if isinstance(stmt, VarDecl):
            return self.gen_var_decl(stmt)
        if isinstance(stmt, Assign):
            return self.gen_assign(stmt)
        if isinstance(stmt, IndexAssign):
            return self.gen_index_assign(stmt)
        if isinstance(stmt, FieldAssign):
            return self.gen_field_assign(stmt)
        if isinstance(stmt, Return):
            return self.gen_return(stmt)
        if isinstance(stmt, If):
            return self.gen_if(stmt)
        if isinstance(stmt, While):
            return self.gen_while(stmt)
        if isinstance(stmt, Break):
            return self.gen_break(stmt)
        if isinstance(stmt, Continue):
            return self.gen_continue(stmt)
        if isinstance(stmt, ExprStmt):
            return self.gen_expr_stmt(stmt)
        raise CodegenError(f"No codegen rule for statement: {stmt!r}")

    def gen_var_decl(self, stmt: VarDecl) -> list[Instruction]:
        # _collect_locals already reserved this VarDecl's slot (that's
        # what sizes the frame); _bind_local just needs to make its name
        # resolvable in the current scope, and return where to store the
        # initializer, if there is one. `int a` with no initializer now
        # gets its type's own implicit zero value (see _gen_zero_value_
        # into) -- 0 for int/bool, the shared empty-string constant for
        # str, none's own {0,0,0} for slice, and recursively for array/
        # struct -- rather than the genuinely uninitialized memory this
        # used to leave behind. The same holds for a heap-allocated
        # array or struct's own malloc'd memory below: allocated, and
        # now always written through (either the real initializer, if
        # there is one, or its type's own zero value if there isn't),
        # never left as raw malloc garbage.
        offset = self._bind_local(stmt)
        var_type = self._local_type(stmt.name)
        if self._is_heap_allocated(id(stmt), var_type):
            # A fresh backing allocation, made exactly once here at
            # declaration time -- see gen_assign's own array case for
            # why a later assignment reuses this same allocation
            # rather than mallocing again. %rax still holds the
            # pointer right after storing it into the slot (that store
            # only READS %rax, it doesn't touch it), so it's safe to
            # use directly as the destination for the initializer, if
            # there is one -- the same "destination is a register-held
            # address" shape gen_return already established for the
            # hidden output pointer, and gen_array_value_into/
            # gen_array_literal_into already handle generically.
            instructions = self._gen_malloc_array(var_type)
            instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', offset)))
            if stmt.init is not None:
                if var_type.kind == TypeKind.STRUCT:
                    instructions.extend(self.gen_struct_value_into(stmt.init, Memory('rax', 0), var_type))
                else:
                    instructions.extend(self.gen_array_value_into(stmt.init, Memory('rax', 0), var_type))
            else:
                instructions.extend(self._gen_zero_value_into(var_type, Memory('rax', 0)))
            return instructions
        if stmt.init is None:
            return self._gen_zero_value_into(var_type, Memory('rbp', offset))
        if isinstance(stmt.init, NoneLiteral):
            # none's own resolved type (Type.NONE) never equals
            # var_type -- semantic.py's _types_compatible is what lets
            # this declaration through despite that (see its own
            # docstring) -- so this needs var_type, the TARGET type,
            # passed explicitly, rather than going through _gen_store's
            # ordinary dispatch, which only ever needs the value
            # expression itself since every OTHER kind of value's own
            # resolved type already matches what needs to be stored.
            return self.gen_none_into(Memory('rbp', offset), var_type)
        if isinstance(stmt.init, ArrayLiteral) and var_type.kind == TypeKind.SLICE:
            # `[]int s = [1, 2, 3]` -- an UNTYPED array literal used
            # directly as a slice's own initializer, treated exactly
            # like the general, explicitly-typed form (`[]int s =
            # []int[1, 2, 3]`, a Slice wrapping an ArrayLiteral --
            # see gen_indexable_base_into's own ArrayLiteral case):
            # construct a new, heap-allocated backing array and
            # produce a descriptor for the whole thing. Needed here,
            # separately, specifically because stmt.init's own
            # resolved type (Type(ARRAY, ...) -- see semantic.py's
            # _check_value_flowing_into) never equals var_type
            # (Type(SLICE, ...)), so _gen_store's ordinary dispatch,
            # which trusts the value's own resolved type completely,
            # would never route this to slice-producing codegen at all
            # on its own.
            instructions = self.gen_array_literal_heap_alloc_into(stmt.init)
            instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', offset)))
            instructions.append(MovQ(src=Imm(len(stmt.init.elements)), dst=Memory('rbp', offset + 8)))
            return instructions
        return self._gen_store(offset, stmt.init)

    def gen_assign(self, stmt: Assign) -> list[Instruction]:
        offset = self._local_offset(stmt.name)
        if isinstance(stmt.value, NoneLiteral):
            # See gen_var_decl's own identical case just above for why
            # this needs the TARGET type (the variable's own declared
            # type), not stmt.value's own resolved type (Type.NONE).
            var_type = self._local_type(stmt.name)
            return self.gen_none_into(Memory('rbp', offset), var_type)
        var_type = self._local_type(stmt.name)
        if isinstance(stmt.value, ArrayLiteral) and var_type.kind == TypeKind.SLICE:
            # See gen_var_decl's own identical case just above for the
            # full reasoning -- unlike an array's own Assign (just
            # below), this always mallocs a FRESH allocation rather
            # than reusing an existing one: an assigned-to slice
            # variable might currently be pointing at a DIFFERENT
            # array (or none at all) of a completely different size,
            # so there's no existing allocation here that could
            # possibly be safe to reuse in place.
            instructions = self.gen_array_literal_heap_alloc_into(stmt.value)
            instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', offset)))
            instructions.append(MovQ(src=Imm(len(stmt.value.elements)), dst=Memory('rbp', offset + 8)))
            return instructions
        value_type = type_of(stmt.value)
        if value_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT) and self._is_heap_allocated(self._local_decl_id(stmt.name), value_type):
            # Reuses the EXISTING allocation from this variable's own
            # declaration -- a fixed-size array's (or struct's own)
            # footprint never changes across its lifetime, so there's
            # nothing to reallocate here, only to load the existing
            # pointer and write the new value through it, exactly like
            # gen_return does for the hidden pointer it receives.
            instructions = [MovQ(src=Memory('rbp', offset), dst=Register('rax'))]
            if value_type.kind == TypeKind.STRUCT:
                instructions.extend(self.gen_struct_value_into(stmt.value, Memory('rax', 0), value_type))
            else:
                instructions.extend(self.gen_array_value_into(stmt.value, Memory('rax', 0), value_type))
            return instructions
        return self._gen_store(offset, stmt.value)

    def gen_index_assign(self, stmt: IndexAssign) -> list[Instruction]:
        """`array[index] = value` -- computes the target element's
        address (via gen_index_address_into, which includes the
        runtime bounds check), protects it on the stack while the
        value expression is evaluated (the same push-before-recursing
        pattern used throughout this file), then writes through it.
        The element's own DECLARED type -- derived from stmt.array's
        own type, not stmt.value's -- decides the store width exactly
        like _gen_store does for an ordinary variable -- str needs
        `movq`, everything else `movl`, and a SLICE element (`rows[i]
        = someSlice`, one element of an array OF slices) needs its own
        24-byte descriptor write, via gen_slice_value_into -- which
        already protects an arbitrary dst_mem.base internally (see its
        own docstring), so this can just hand it Memory('rax', 0)
        directly rather than needing its own, separate push/pop
        dance the way the scalar path below still does.

        Deliberately NOT stmt.value's own resolved type (type_of(stmt.value)),
        the way this used to be computed: an UNTYPED array literal flowing into
        a SLICE-typed element (`rows[0] = [9, 9, 9]`) has its own resolved type
        set to the ARRAY it actually builds (see semantic.py's
        _check_value_flowing_into), not the slice it's being treated as -- so
        dispatching on the VALUE's own type would miss this case entirely and
        fall through to the scalar path below, the same bug-class already
        fixed in gen_var_decl/gen_assign (see their own docstrings)
        and analyze_index_assign, just at a third call site.

        An ARRAY-typed element (writing a whole sub-array via
        `matrix[i] = other_row`) isn't reachable here at all:
        IndexAssign's own grammar only ever produces a single
        leaf-level element write; a whole-row assignment would need
        `matrix[i]` to appear as an ordinary Assign target, which
        parser.py doesn't produce (see IndexAssign's own docstring).
        """
        base_type = type_of(stmt.array)
        element_type = base_type.element_type
        addr_reg = Register('rax')
        instructions = self.gen_index_address_into(Index(array=stmt.array, index=stmt.index), addr_reg)
        if element_type.kind == TypeKind.SLICE:
            instructions.extend(self.gen_slice_value_into(stmt.value, Memory('rax', 0)))
            return instructions
        if element_type.kind == TypeKind.STRUCT:
            return instructions + self.gen_struct_value_into(stmt.value, Memory('rax', 0), element_type)
        instructions.append(Push(addr_reg))
        if element_type == Type.STR:
            instructions.extend(self.gen_expr_into(stmt.value, Register('eax')))
            instructions.append(MovQ(src=Register('rax'), dst=Register('r8')))  # value survives the pop below
            instructions.append(Pop(addr_reg))
            instructions.append(MovQ(src=Register('r8'), dst=Memory('rax', 0)))
        else:
            instructions.extend(self.gen_expr_into(stmt.value, Register('eax')))
            if element_type == Type.INT64:
                instructions.append(MovQ(src=Register('rax'), dst=Register('r8')))
            else:
                instructions.append(Mov(src=Register('eax'), dst=Register('r8d')))
            instructions.append(Pop(addr_reg))
            instructions.extend(self._gen_write_scalar_from(Register('r8d'), element_type, Memory('rax', 0)))
        return instructions

    def gen_field_assign(self, stmt: FieldAssign) -> list[Instruction]:
        """`base.name = value` -- mirrors gen_index_assign exactly, one
        level over: computes the target field's address (via
        gen_field_address_into), protects it on the stack while the
        value expression is evaluated, then writes through it. The
        field's own DECLARED type -- derived from stmt.base's own
        struct type, not stmt.value's -- decides the store width
        exactly like gen_index_assign already does for an array
        element, for the identical reason (an untyped literal flowing
        into an already-typed slot has its OWN resolved type set to
        whatever it actually built, not the slot's type -- see
        gen_index_assign's own docstring).

        A STRUCT-typed field (`s.inner = otherInner`) is handled the
        same way an array-typed field would be if IndexAssign could
        ever produce one (which, per its own docstring, it can't) --
        via gen_struct_value_into's own flat copy, since a field write
        of a whole struct value is exactly as much "copy N bytes" as
        any other struct value production is. An array-typed field
        (`s.arr = otherArr`) works the same way, via gen_array_value_
        into -- unlike IndexAssign, FieldAssign's own grammar CAN
        produce this shape (a struct field can itself be a whole
        array, and `.` doesn't consume it element by element the way
        `[...]` does), so this needs a real case for it, not just a
        comment explaining why it's unreachable.

        A NoneLiteral value flowing into a slice-typed field (`s.values
        = none`) needs the identical short-circuit gen_var_decl/gen_
        assign already have, checked BEFORE the SLICE dispatch below
        rather than falling into gen_slice_value_into's own ordinary
        dispatch -- for the identical reason those two already need
        it: none's own resolved type (Type.NONE) never equals the
        field's own declared type, so gen_slice_value_into's dispatch
        (which only ever needs the expression itself, since every
        OTHER kind of value's own resolved type already matches what
        needs to be stored) has no case for it at all. This was a
        real, separately-rooted gap found by testing, not caught by
        gen_field_assign's own original design: FieldAssign simply
        didn't exist as a reachable path for a slice-typed value until
        slice-typed fields were supported at all, so this short-
        circuit was never needed until now."""
        field_type = self._check_struct_and_field_type(stmt.base, stmt.name)
        addr_reg = Register('rax')
        instructions = self.gen_field_address_into(stmt, addr_reg)
        if field_type.kind == TypeKind.SLICE:
            if isinstance(stmt.value, NoneLiteral):
                return instructions + self.gen_none_into(Memory('rax', 0), field_type)
            instructions.extend(self.gen_slice_value_into(stmt.value, Memory('rax', 0)))
            return instructions
        if field_type.kind == TypeKind.STRUCT:
            return instructions + self.gen_struct_value_into(stmt.value, Memory('rax', 0), field_type)
        if field_type.kind == TypeKind.ARRAY:
            return instructions + self.gen_array_value_into(stmt.value, Memory('rax', 0), field_type)
        instructions.append(Push(addr_reg))
        if field_type == Type.STR:
            instructions.extend(self.gen_expr_into(stmt.value, Register('eax')))
            instructions.append(MovQ(src=Register('rax'), dst=Register('r8')))  # value survives the pop below
            instructions.append(Pop(addr_reg))
            instructions.append(MovQ(src=Register('r8'), dst=Memory('rax', 0)))
        else:
            instructions.extend(self.gen_expr_into(stmt.value, Register('eax')))
            if field_type == Type.INT64:
                instructions.append(MovQ(src=Register('rax'), dst=Register('r8')))
            else:
                instructions.append(Mov(src=Register('eax'), dst=Register('r8d')))
            instructions.append(Pop(addr_reg))
            instructions.extend(self._gen_write_scalar_from(Register('r8d'), field_type, Memory('rax', 0)))
        return instructions

    def _gen_store(self, offset: int, value_expr: Node) -> list[Instruction]:
        """Shared by VarDecl-with-initializer and Assign: both are just
        "compute this expression, then write the result into that
        variable's slot". Which store instruction depends on the
        value's type: an array or a struct can't fit into a single
        register at all, so each is dispatched to gen_array_value_into
        or gen_struct_value_into entirely separately (see their own
        docstrings); a slice is a fixed-size 24-byte descriptor,
        dispatched to gen_slice_value_into (see its own docstring) the
        same way; a str is an 8-byte pointer sitting in %rax and needs
        `movq`; int/bool/int8/uint8 all compute the same way (via
        gen_expr_into, unmodified and oblivious to which of the four it
        actually is) and then write out via _gen_write_scalar_from,
        which is the one place that actually distinguishes a narrow,
        1-byte store (int8/uint8) from an ordinary 4-byte one (int/
        bool) -- everything about gen_expr_into/gen_binary_into/
        gen_unary_op's own internals stays exactly as it always has,
        oblivious to str (or arrays, slices, or structs) entirely;
        only this one call site needs to ask "which width, or which
        entirely different mechanism, am I storing"."""
        value_type = type_of(value_expr)
        if value_type.kind == TypeKind.ARRAY:
            return self.gen_array_value_into(value_expr, Memory('rbp', offset), value_type)
        if value_type.kind == TypeKind.SLICE:
            return self.gen_slice_value_into(value_expr, Memory('rbp', offset))
        if value_type.kind == TypeKind.STRUCT:
            return self.gen_struct_value_into(value_expr, Memory('rbp', offset), value_type)
        instructions = self.gen_expr_into(value_expr, Register('eax'))
        if value_type == Type.STR:
            instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', offset)))
        else:
            instructions.extend(self._gen_write_scalar_from(Register('eax'), value_type, Memory('rbp', offset)))
        return instructions

    def gen_return(self, stmt: Return) -> list[Instruction]:
        # A bare `return` (no value at all -- valid exactly when this
        # function has no declared return type, see Return's own
        # docstring in parser.py and analyze_return's own check) needs
        # nothing computed at all, just the ordinary epilogue.
        if stmt.value is None:
            return self._gen_epilogue()

        if isinstance(stmt.value, NoneLiteral):
            # none's own resolved type (Type.NONE) never equals SLICE
            # -- semantic.py's _types_compatible is what lets `return
            # none` through despite that (see its own docstring), and
            # already guarantees it's only ever valid when THIS
            # function's own declared return type IS a slice, since
            # slices are the only nilable type that exists. Written
            # directly (not via gen_none_into, which needs a real
            # target_type to defensively check against -- not readily
            # available here, and not worth threading through just for
            # this) through the hidden return pointer, exactly like
            # every other slice-typed return value now (see below) --
            # this used to write straight into %rax/%rdx, back when a
            # slice's own descriptor still fit two registers.
            ptr_reg = Register('rax')
            instructions = [MovQ(src=Memory('rbp', self._hidden_return_ptr_offset), dst=ptr_reg)]
            instructions.append(MovQ(src=Imm(0), dst=Memory('rax', 0)))
            instructions.append(MovQ(src=Imm(0), dst=Memory('rax', 8)))
            instructions.append(MovQ(src=Imm(0), dst=Memory('rax', 16)))
            instructions.extend(self._gen_epilogue())
            return instructions

        # An array- OR slice-typed return writes directly through the
        # hidden pointer this function received (see gen_function's
        # own prologue handling and the module docstring's ARRAYS
        # section) instead of ever putting anything in %eax/%rax --
        # nothing reads a return value that way for an array- or
        # slice-returning call (see gen_array_value_into/gen_slice_
        # value_into's own Call cases, the only way such a call's
        # result is ever consumed). Loading the pointer back out of
        # its slot and handing it to gen_array_value_into/gen_slice_
        # value_into as an ordinary Memory destination is also what
        # makes `return bar()` (forwarding another array- or slice-
        # returning call's result straight out) free: the Call case
        # just passes that SAME address one level deeper via gen_
        # array_call_into/gen_slice_call_into, with no intermediate
        # copy ever materialized.
        value_type = type_of(stmt.value)
        if value_type.kind == TypeKind.ARRAY:
            ptr_reg = Register('rax')
            instructions = [MovQ(src=Memory('rbp', self._hidden_return_ptr_offset), dst=ptr_reg)]
            instructions.extend(self.gen_array_value_into(stmt.value, Memory('rax', 0), value_type))
        elif value_type.kind == TypeKind.SLICE:
            ptr_reg = Register('rax')
            instructions = [MovQ(src=Memory('rbp', self._hidden_return_ptr_offset), dst=ptr_reg)]
            instructions.extend(self.gen_slice_value_into(stmt.value, Memory('rax', 0)))
        elif value_type.kind == TypeKind.STRUCT:
            # Same hidden-pointer mechanism, one more time -- see this
            # method's own comment just above for the full reasoning,
            # unchanged in every respect for struct: gen_struct_value_
            # into already knows how to write into an arbitrary Memory
            # destination (a fixed local slot OR a received pointer
            # alike), so `return bar()` (forwarding another struct-
            # returning call's result) is exactly as free here as it
            # already is for arrays and slices, via gen_struct_value_
            # into's own Call case.
            ptr_reg = Register('rax')
            instructions = [MovQ(src=Memory('rbp', self._hidden_return_ptr_offset), dst=ptr_reg)]
            instructions.extend(self.gen_struct_value_into(stmt.value, Memory('rax', 0), value_type))
        else:
            dst = Register('eax')
            instructions = self.gen_expr_into(stmt.value, dst)
        # None of the epilogue touches %eax/%rax/%rdx, so a scalar
        # return value computed above is unaffected regardless of what
        # these registers held during the body (e.g. if the return
        # expression itself did string work that reused them as
        # scratch in between).
        instructions.extend(self._gen_epilogue())
        return instructions

    def gen_if(self, stmt: If) -> list[Instruction]:
        """Computes the condition into %eax and compares it to 0, exactly
        like the short-circuit AND/OR codegen already does -- then jumps
        past the `then` body when it's false:

            <condition>          ; -> %eax
            cmpl $0, %eax
            je   .Lif_else_N     ; false -> skip straight to else (or end)
            <then_body>
            jmp  .Lif_end_N      ; true -> skip over else after then runs
        .Lif_else_N:
            <else_body>          ; only emitted if else_body is present
        .Lif_end_N:

        then_body and else_body each get their own pushed/popped scope
        (see _push_scope), matching semantic.py's independent-branch
        scoping -- and since an elif is just a nested If sitting inside
        else_body (see parser.py's If docstring), gen_statement's
        ordinary recursion handles a whole elif/else chain of any
        length with no extra logic here at all.
        """
        dst = Register('eax')
        else_label = self.new_label("if_else")
        end_label = self.new_label("if_end")

        instructions = self.gen_expr_into(stmt.condition, dst)
        instructions.append(Cmp(src=Imm(0), dst=dst))
        instructions.append(Je(else_label))

        self._push_scope()
        for s in stmt.then_body:
            instructions.extend(self.gen_statement(s))
        self._pop_scope()
        instructions.append(Jmp(end_label))

        instructions.append(Label(else_label))
        if stmt.else_body is not None:
            self._push_scope()
            for s in stmt.else_body:
                instructions.extend(self.gen_statement(s))
            self._pop_scope()
        instructions.append(Label(end_label))

        return instructions

    def gen_while(self, stmt: While) -> list[Instruction]:
        """Computes the condition, re-checked before every iteration
        (including the first), with the body sitting between two labels
        that break/continue jump to:

            .Lwhile_start_N:
                <condition>          ; -> %eax
                cmpl $0, %eax
                je   .Lwhile_end_N   ; false -> exit the loop entirely
                <body>
                jmp  .Lwhile_start_N ; loop back to re-check the condition
            .Lwhile_end_N:

        Both labels get pushed onto self.loop_labels for the duration
        of generating the body, so any Break/Continue statement inside
        it -- including ones nested inside an If -- can find its way
        back here via gen_break/gen_continue without this method needing
        to know anything about where inside the body they are. Popped
        again once the body's done, so a Break/Continue *after* this
        while (or in a sibling loop) can't accidentally resolve to this
        loop's labels -- see the module docstring's LOOPS section for
        why that matters once loops nest.

        The body gets its own pushed/popped scope, same as an If's
        then/else bodies, even though it's the same physical stack slots
        being reused on every iteration (see _collect_locals) -- this is
        purely about name resolution during code generation, not
        anything that happens at runtime.
        """
        dst = Register('eax')
        start_label = self.new_label("while_start")
        end_label = self.new_label("while_end")

        instructions = [Label(start_label)]
        instructions.extend(self.gen_expr_into(stmt.condition, dst))
        instructions.append(Cmp(src=Imm(0), dst=dst))
        instructions.append(Je(end_label))

        self.loop_labels.append((start_label, end_label))
        self._push_scope()
        for s in stmt.body:
            instructions.extend(self.gen_statement(s))
        self._pop_scope()
        self.loop_labels.pop()

        instructions.append(Jmp(start_label))
        instructions.append(Label(end_label))
        return instructions

    def gen_break(self, stmt: Break) -> list[Instruction]:
        # semantic.py already guarantees this only appears inside a
        # loop; the IndexError-avoiding check here is the same defensive
        # posture as _local_offset's -- see the module docstring on
        # generate_asm/compile_to_asm for why codegen still checks for
        # itself rather than trusting semantic analysis unconditionally.
        if not self.loop_labels:
            raise CodegenError("'break' outside of a loop")
        _, end_label = self.loop_labels[-1]
        return [Jmp(end_label)]

    def gen_continue(self, stmt: Continue) -> list[Instruction]:
        if not self.loop_labels:
            raise CodegenError("'continue' outside of a loop")
        start_label, _ = self.loop_labels[-1]
        return [Jmp(start_label)]

    def gen_expr_stmt(self, stmt: ExprStmt) -> list[Instruction]:
        # Evaluated the same way as any other expression, into %eax --
        # just with nothing done with the result afterward. Still real
        # instructions that really run; see the module docstring for
        # how that's verified (a standalone `1 / 0` genuinely crashes).
        #
        # An ArrayLiteral is the one exception: it can't be computed
        # via gen_expr_into at all (an array doesn't fit in a single
        # register), and unlike a VarDecl/Assign's own use of one, a
        # bare literal statement has no destination to write the
        # resulting array into -- but it doesn't need one, since
        # nothing ever reads the array as a whole. See gen_array_
        # literal_side_effects_only's own docstring for the resulting,
        # narrower approach: evaluate each element for whatever side
        # effects it might have, without ever materializing a real
        # array in memory at all.
        if isinstance(stmt.expr, ArrayLiteral):
            return self.gen_array_literal_side_effects_only(stmt.expr)
        # A Slice expression is the analogous exception for slices --
        # a 24-byte descriptor doesn't fit in a single register either
        # -- but unlike ArrayLiteral, this doesn't need its own
        # narrower, side-effects-only path: gen_slice_into already
        # computes fully correctly into any Memory destination,
        # including a genuine runtime bounds check on low/high (an
        # out-of-range bound still aborts here, matching how any other
        # bare expression statement's real instructions genuinely run
        # -- see this method's own opening comment), so this just
        # reuses the same per-function scratch slot gen_indexable_
        # base_into's own Slice-base case already uses (_unnamed_
        # slice_temp_offset) and discards the result -- nothing ever
        # reads it. Covers both a bare slice LITERAL statement
        # (`[]int[se(), 2, 3]`, parsed as a Slice wrapping an
        # ArrayLiteral -- see parser.py's own _parse_bracketed_
        # literal) and an ordinary bare slice of an EXISTING array or
        # slice (`arr[:]` alone, pointless but not an error) with the
        # exact same code path.
        if isinstance(stmt.expr, Slice):
            return self.gen_slice_into(stmt.expr, Memory('rbp', self._unnamed_slice_temp_offset))
        return self.gen_expr_into(stmt.expr, Register('eax'))

    def gen_short_circuit(
            self, expr: Binary,
            dst: Operand, *,
            short_circuit_jump: type,
            short_circuit_value: int,
            fallthrough_value: int,
            label_prefix: str) -> list[Instruction]:
        """Shared codegen for AND and OR -- they're mirror images of each
        other: each evaluates its left side, tests it against 0, and
        jumps straight past the right side entirely (never emitting the
        instructions that would compute it as *executed* code) if that
        test already decides the answer. Only if it doesn't -- left was
        truthy for AND, falsy for OR -- does the right side actually get
        evaluated, and *that* result decides the answer instead.

          AND: jump early (to `short_circuit_value=0`) when left == 0.
          OR:  jump early (to `short_circuit_value=1`) when left != 0.

        This is what makes `0 and (1 / 0)` return 0 instead of crashing:
        the division is real code sitting in the binary, but control
        flow jumps clean over it.
        """
        if not isinstance(dst, Register):
            raise CodegenError(f"Binary codegen requires a register destination, got: {dst!r}")

        short_label = self.new_label(f"{label_prefix}_short")
        end_label = self.new_label(f"{label_prefix}_end")

        instructions = self.gen_expr_into(expr.left, dst)
        instructions.append(Cmp(src=Imm(0), dst=dst))
        instructions.append(short_circuit_jump(short_label))

        instructions.extend(self.gen_expr_into(expr.right, dst))
        instructions.append(Cmp(src=Imm(0), dst=dst))
        instructions.append(short_circuit_jump(short_label))

        instructions.append(Mov(src=Imm(fallthrough_value), dst=dst))
        instructions.append(Jmp(end_label))
        instructions.append(Label(short_label))
        instructions.append(Mov(src=Imm(short_circuit_value), dst=dst))
        instructions.append(Label(end_label))
        return instructions
