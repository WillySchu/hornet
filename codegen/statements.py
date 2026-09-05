"""Statement-level codegen: declarations, assignment, control flow
(if/while/break/continue), and return. Establishes that an
uninitialized declaration gets its type's real zero value rather than
leaving memory untouched, and the label-pair shape (start/end, or
else/end) every branching or looping construct here builds on."""

from codegen.assembly_ast import Instruction, MovQ, Register, Memory, Imm, Push, Pop, Mov, Cmp, Je, Jmp, Label, \
    LeaQ, MovB
from codegen.errors import CodegenError
from codegen.ir import IRRaw, IRReturn
from codegen.utils import type_of
from parser import Node, VarDecl, Assign, IndexAssign, FieldAssign, Return, If, While, Break, Continue, ExprStmt, \
    NoneLiteral, ArrayLiteral, Index, Slice
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
        # _collect_locals already reserved this VarDecl's slot;
        # _bind_local just makes its name resolvable in the current
        # scope and returns where to store the initializer, if there
        # is one. `int a` with no initializer gets its type's implicit
        # zero value (see _gen_zero_value_into) rather than genuinely
        # uninitialized memory -- the same holds for a heap-allocated
        # array/struct's malloc'd memory below: always written through,
        # never left as raw malloc garbage.
        offset = self._bind_local(stmt)
        var_type = self._local_type(stmt.name)
        if self._is_heap_allocated(id(stmt), var_type):
            # A fresh backing allocation, made exactly once here at
            # declaration time (see gen_assign's array case for why a
            # later assignment reuses this allocation instead of
            # mallocing again). %rax still holds the pointer right
            # after storing it into the slot, so it's safe to use
            # directly as the initializer's destination.
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
            # none's resolved type (Type.NONE) never equals var_type --
            # semantic.py's _types_compatible is what lets this
            # declaration through despite that -- so this needs
            # var_type, the TARGET type, passed explicitly, rather than
            # going through _gen_store's ordinary dispatch, which only
            # needs the value expression since every other kind of
            # value's resolved type already matches what's being stored.
            return self.gen_none_into(Memory('rbp', offset), var_type)
        if isinstance(stmt.init, ArrayLiteral) and var_type.kind == TypeKind.SLICE:
            # `[]int s = [1, 2, 3]` -- an untyped array literal used
            # directly as a slice's initializer, treated like the
            # general, explicitly-typed form (`[]int s = []int[1, 2,
            # 3]`): construct a new, heap-allocated backing array and
            # produce a descriptor for the whole thing. Needed here
            # separately because stmt.init's resolved type
            # (Type(ARRAY,...)) never equals var_type (Type(SLICE,...)),
            # so _gen_store's ordinary dispatch, which trusts the
            # value's own resolved type, would never route this to
            # slice-producing codegen on its own.
            instructions = self.gen_array_literal_heap_alloc_into(stmt.init)
            instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', offset)))
            instructions.append(MovQ(src=Imm(len(stmt.init.elements)), dst=Memory('rbp', offset + 8)))
            return instructions
        return self._gen_store(offset, stmt.init)

    def gen_assign(self, stmt: Assign) -> list[Instruction]:
        offset = self._local_offset(stmt.name)
        if isinstance(stmt.value, NoneLiteral):
            # See gen_var_decl's identical case above: needs the
            # TARGET type (the variable's declared type), not
            # stmt.value's resolved type (Type.NONE).
            var_type = self._local_type(stmt.name)
            return self.gen_none_into(Memory('rbp', offset), var_type)
        var_type = self._local_type(stmt.name)
        if isinstance(stmt.value, ArrayLiteral) and var_type.kind == TypeKind.SLICE:
            # See gen_var_decl's identical case for the full reasoning
            # -- unlike an array's own Assign below, this always
            # mallocs a FRESH allocation rather than reusing an
            # existing one: an assigned-to slice variable might
            # currently point at a different array (or none at all) of
            # a different size, so there's no existing allocation here
            # that could be safe to reuse in place.
            instructions = self.gen_array_literal_heap_alloc_into(stmt.value)
            instructions.append(MovQ(src=Register('rax'), dst=Memory('rbp', offset)))
            instructions.append(MovQ(src=Imm(len(stmt.value.elements)), dst=Memory('rbp', offset + 8)))
            return instructions
        value_type = type_of(stmt.value)
        if value_type.kind in (TypeKind.ARRAY, TypeKind.STRUCT) and self._is_heap_allocated(self._local_decl_id(stmt.name), value_type):
            # Reuses the EXISTING allocation from this variable's
            # declaration -- a fixed-size array's (or struct's) own
            # footprint never changes across its lifetime, so there's
            # nothing to reallocate, only to load the existing pointer
            # and write the new value through it.
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
        value expression is evaluated, then writes through it. The
        element's DECLARED type -- derived from stmt.array's type, not
        stmt.value's -- decides the store width, exactly like
        _gen_store does for an ordinary variable: str needs `movq`,
        everything else `movl`, and a SLICE element (`rows[i] =
        someSlice`) needs its own 24-byte descriptor write via
        gen_slice_value_into, which already protects an arbitrary
        dst_mem.base internally, so this can hand it Memory('rax', 0)
        directly without its own push/pop dance.

        Deliberately NOT stmt.value's resolved type: an untyped array
        literal flowing into a SLICE-typed element (`rows[0] = [9, 9,
        9]`) has its resolved type set to the ARRAY it actually builds,
        not the slice it's being treated as -- dispatching on the
        VALUE's type would miss this case and fall through to the
        scalar path below (the same bug-class already fixed in
        gen_var_decl/gen_assign, just at a third call site).

        An ARRAY-typed element (`matrix[i] = other_row`) isn't
        reachable here: IndexAssign's grammar only ever produces a
        single leaf-level element write.
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
        """`base.name = value` -- mirrors gen_index_assign one level
        over: computes the target field's address (via
        gen_field_address_into), protects it on the stack while the
        value expression is evaluated, then writes through it. The
        field's DECLARED type -- derived from stmt.base's struct type,
        not stmt.value's -- decides the store width, for the same
        reason gen_index_assign uses the element's declared type.

        A STRUCT-typed field (`s.inner = otherInner`) is handled via
        gen_struct_value_into's flat copy, since a field write of a
        whole struct value is exactly as much "copy N bytes" as any
        other struct value production. An array-typed field (`s.arr =
        otherArr`) works the same way via gen_array_value_into --
        unlike IndexAssign, FieldAssign's grammar CAN produce this
        shape (a struct field can itself be a whole array), so this
        needs a real case for it.

        A NoneLiteral value flowing into a slice-typed field
        (`s.values = none`) needs the identical short-circuit gen_var_
        decl/gen_assign already have, checked BEFORE the SLICE
        dispatch: none's resolved type (Type.NONE) never equals the
        field's declared type, so gen_slice_value_into's ordinary
        dispatch has no case for it. This was a real gap found by
        testing: FieldAssign wasn't a reachable path for a slice-typed
        value until slice-typed fields existed at all."""
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
        value's type: an array or struct can't fit in a single
        register, so each is dispatched to gen_array_value_into or
        gen_struct_value_into separately; a slice is a fixed-size
        24-byte descriptor, dispatched to gen_slice_value_into the same
        way; a str is an 8-byte pointer sitting in %rax and needs
        `movq`; int/bool/int8/uint8 all compute the same way (via
        gen_expr_into, oblivious to which of the four it actually is)
        and then write out via _gen_write_scalar_from, the one place
        that distinguishes a narrow 1-byte store (int8/uint8) from an
        ordinary 4-byte one -- only this call site needs to ask "which
        width, or which entirely different mechanism, am I storing"."""
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
        # A bare `return` (no value -- valid exactly when this function
        # has no declared return type) needs nothing computed, just
        # the ordinary epilogue -- represented as an IRReturn with no
        # value at all, exactly like a scalar return below, just with
        # nothing to load into %eax first.
        if stmt.value is None:
            return self.lower_ir([IRReturn(value=None)])

        if isinstance(stmt.value, NoneLiteral):
            # none's resolved type (Type.NONE) never equals SLICE --
            # semantic.py's _types_compatible is what lets `return
            # none` through despite that, and already guarantees it's
            # only valid when this function's declared return type IS
            # a slice. Written directly (not via gen_none_into, which
            # needs a real target_type to check against, not readily
            # available here) through the hidden return pointer, like
            # every other slice-typed return value.
            ptr_reg = Register('rax')
            instructions = [MovQ(src=Memory('rbp', self._hidden_return_ptr_offset), dst=ptr_reg)]
            instructions.append(MovQ(src=Imm(0), dst=Memory('rax', 0)))
            instructions.append(MovQ(src=Imm(0), dst=Memory('rax', 8)))
            instructions.append(MovQ(src=Imm(0), dst=Memory('rax', 16)))
            instructions.extend(self._gen_epilogue())
            return instructions

        # An array- OR slice-typed return writes directly through the
        # hidden pointer this function received instead of ever
        # putting anything in %eax/%rax -- nothing reads a return
        # value that way for an array- or slice-returning call.
        # Loading the pointer back out of its slot and handing it to
        # gen_array_value_into/gen_slice_value_into as an ordinary
        # Memory destination is also what makes `return bar()`
        # (forwarding another array/slice-returning call's result)
        # free: the Call case just passes that same address one level
        # deeper, with no intermediate copy ever materialized.
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
            # Same hidden-pointer mechanism -- unchanged for struct:
            # gen_struct_value_into already knows how to write into an
            # arbitrary Memory destination, so `return bar()` is
            # exactly as free here as it is for arrays and slices.
            ptr_reg = Register('rax')
            instructions = [MovQ(src=Memory('rbp', self._hidden_return_ptr_offset), dst=ptr_reg)]
            instructions.extend(self.gen_struct_value_into(stmt.value, Memory('rax', 0), value_type))
        else:
            # A scalar return, built as a small IR fragment: compute
            # the value into a Temp, then IRReturn -- which lower_ir
            # turns into "load it into %eax (or %rax, wherever its
            # type needs), then the ordinary epilogue" -- exactly the
            # same lowering the bare-return case above uses, just with
            # a real value to load first. None of the epilogue touches
            # %eax/%rax/%rdx, so this is unaffected by whatever those
            # registers held during the body.
            t_result = self._new_temp(value_type)
            return self.lower_ir([
                IRRaw(self.gen_expr_into(stmt.value, Register('eax')), dst=t_result),
                IRReturn(value=t_result),
            ])
        instructions.extend(self._gen_epilogue())
        return instructions

    def gen_if(self, stmt: If) -> list[Instruction]:
        """Computes the condition into %eax and compares it to 0, like
        the short-circuit AND/OR codegen -- then jumps past the `then`
        body when it's false:

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
        scoping -- and since an elif is just a nested If inside
        else_body, gen_statement's ordinary recursion handles a whole
        elif/else chain of any length with no extra logic here.
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
        (including the first), with the body sitting between two
        labels that break/continue jump to:

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
        back here via gen_break/gen_continue with no need to know
        anything about where inside the body they are. Popped again
        once the body's done, so a Break/Continue after this while (or
        in a sibling loop) can't accidentally resolve to this loop's
        labels.

        The body gets its own pushed/popped scope, same as an If's
        then/else bodies, even though it's the same physical stack
        slots being reused on every iteration -- this is purely about
        name resolution during code generation, not anything that
        happens at runtime.
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
        # loop; this check exists so codegen doesn't trust semantic
        # analysis unconditionally, the same defensive posture
        # _local_offset takes.
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
        # instructions that really run (a standalone `1 / 0` genuinely
        # crashes).
        #
        # An ArrayLiteral is the one exception: it can't be computed
        # via gen_expr_into at all (doesn't fit in a single register),
        # and unlike a VarDecl/Assign's use of one, a bare literal
        # statement has no destination to write the resulting array
        # into -- but it doesn't need one, since nothing ever reads the
        # array as a whole. See gen_array_literal_side_effects_only for
        # the resulting approach: evaluate each element for whatever
        # side effects it might have, without materializing a real
        # array in memory.
        if isinstance(stmt.expr, ArrayLiteral):
            return self.gen_array_literal_side_effects_only(stmt.expr)
        # A Slice expression is the analogous exception for slices --
        # a 24-byte descriptor doesn't fit in a register either -- but
        # unlike ArrayLiteral, this doesn't need its own narrower path:
        # gen_slice_into already computes fully correctly into any
        # Memory destination, including a genuine runtime bounds check
        # (an out-of-range bound still aborts here), so this just
        # reuses the same per-function scratch slot gen_indexable_
        # base_into's Slice-base case already uses
        # (_unnamed_slice_temp_offset) and discards the result. Covers
        # both a bare slice LITERAL statement and an ordinary bare
        # slice of an existing array or slice (`arr[:]` alone,
        # pointless but not an error) with the same code path.
        if isinstance(stmt.expr, Slice):
            return self.gen_slice_into(stmt.expr, Memory('rbp', self._unnamed_slice_temp_offset))
        return self.gen_expr_into(stmt.expr, Register('eax'))

    def _gen_zero_value_into(self, t: Type, dst_mem: Memory) -> list[Instruction]:
        """Writes t's implicit zero value into dst_mem -- what a `T x`
        VarDecl with no initializer gets, instead of genuinely
        uninitialized memory. Dispatches by kind:
          - int/bool/int8/uint8: an ordinary 0 -- a plain 4-byte write
            for int/bool, a 1-byte one (MovB) for int8/uint8, matching
            each type's genuine storage width.
          - str: the address of a single shared, static empty-string
            constant (_get_empty_str_label) -- never a null pointer
            (see that method for why a null zero value would be an
            active hazard).
          - slice: none's {ptr: 0, len: 0, cap: 0} descriptor, reusing
            gen_none_into as-is -- a zero-value slice and a none-valued
            one are, by design, the identical representation.
          - array: delegated to _gen_zero_array_into, which further
            dispatches on the array's leaf type.
          - struct: every field, flattened via _flatten_struct_fields
            the same way struct equality flattens them for comparison,
            recursing back into this method for each field's type.

        dst_mem.base is protected via push/pop across EVERY field's
        zero-fill, when it isn't 'rbp': the array case computes a fresh
        address via _gen_address_of_memory_into with dst_mem.base
        itself as the destination register in some call shapes, which
        can overwrite dst_mem.base's physical register in place.
        Without protecting it, a struct with an array-typed field
        followed by any other field would silently compute that later
        field's address from garbage instead of the struct's real base
        -- the same register-collision failure mode
        _gen_struct_fields_equality_at_addresses guards against for the
        identical reason. Applied unconditionally, even for the
        scalar/slice cases that don't strictly need it."""
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

