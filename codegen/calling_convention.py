"""TODO"""

from codegen.assembly_ast import Instruction, Register, Push, Pop, Memory, MovQ, LeaQFrame
from codegen.escape_analysis import is_heap_allocated
from codegen.utils import type_of, ARG_REGISTERS_64
from parser import Node, NoneLiteral, Variable, Field, Index
from semantic import TypeKind, Type

# Registers gen_string_concat_into/gen_string_compare_into use as
# scratch. Every function's prologue/epilogue saves and restores
# these unconditionally, regardless of whether that function itself
# uses them, since the callee-saved contract must hold for any call.
CALLEE_SAVED_SCRATCH_REGISTERS = ['rbx', 'r12', 'r13', 'r14']


class CallingConventionMixin:
    def _gen_call_arguments_into(self, args: list[Node], reg_shift: int = 0) -> list[Instruction]:
        """Shared by gen_call_into, gen_array_call_into, and
        gen_slice_call_into: evaluates each argument in order
        (gen_expr_into for a scalar, gen_array_arg_address_into for an
        array, gen_slice_arg_into for a slice/none, which needs 3
        register slots), pushing each result onto the stack
        immediately, then pops everything back off in reverse into the
        real SysV argument registers.

        Compute-then-stack-then-pop, rather than moving each result
        straight into its register, avoids clobbering: if argument 2
        is itself a call, evaluating it could overwrite a scratch
        register argument 1's value was still sitting in.

        `reg_shift` is 1 when the callee returns an array/slice (its
        hidden output pointer already occupies register slot 0,
        handled separately by the caller) and 0 otherwise. Slice
        arguments consume 3 consecutive slots, so slot indices are
        tracked with a running count rather than 1:1 with argument
        position.
        """
        instructions: list[Instruction] = []
        arg_slot_counts = []
        for arg in args:
            arg_type = type_of(arg)
            if arg_type.kind == TypeKind.SLICE or isinstance(arg, NoneLiteral):
                instructions.extend(self.gen_slice_arg_into(arg, Register('rax'), Register('rdx'), Register('rcx')))
                instructions.append(Push(Register('rax')))
                instructions.append(Push(Register('rdx')))
                instructions.append(Push(Register('rcx')))
                arg_slot_counts.append(3)
            elif arg_type.kind == TypeKind.ARRAY:
                instructions.extend(self.gen_array_arg_address_into(arg, Register('rax')))
                instructions.append(Push(Register('rax')))
                arg_slot_counts.append(1)
            elif arg_type.kind == TypeKind.STRUCT:
                # Same convention as an array argument just above --
                # pass the address, let the callee copy from it on
                # entry (see gen_function's parameter loop). A
                # Variable/Field/Index already has a real address (via
                # gen_struct_address_into directly, rather than gen_
                # array_arg_address_into, since a struct argument can
                # also be a Field -- `foo(s.inner)` -- which that
                # method doesn't handle at all); anything else -- a
                # struct literal, or an ordinary struct-returning Call
                # -- has none, so it's materialized first (see _gen_
                # materialize_argument_temp_into), the exact same
                # mechanism the ARRAY branch above now uses for the
                # identical shape of problem.
                if isinstance(arg, (Variable, Field, Index)):
                    instructions.extend(self.gen_struct_address_into(arg, Register('rax')))
                else:
                    instructions.extend(self._gen_materialize_argument_temp_into(arg, arg_type, Register('rax')))
                instructions.append(Push(Register('rax')))
                arg_slot_counts.append(1)
            else:
                instructions.extend(self.gen_expr_into(arg, Register('eax')))
                instructions.append(Push(Register('rax')))
                arg_slot_counts.append(1)

        slot = sum(arg_slot_counts) - 1 + reg_shift
        for count in reversed(arg_slot_counts):
            if count == 3:
                instructions.append(Pop(Register(ARG_REGISTERS_64[slot])))
                instructions.append(Pop(Register(ARG_REGISTERS_64[slot - 1])))
                instructions.append(Pop(Register(ARG_REGISTERS_64[slot - 2])))
            else:
                instructions.append(Pop(Register(ARG_REGISTERS_64[slot])))
            slot -= count
        return instructions

    def _total_arg_slots(self, args: list[Node]) -> int:
        """Total argument-register slots `args` will need: 3 per
        slice-typed (or `none`) argument, 1 for everything else.
        `none` is checked via isinstance rather than its resolved
        type, since Type.NONE != SLICE; semantic.py guarantees `none`
        only ever appears where a slice is expected, so this is safe.
        Used by every call-codegen entry point's "too many arguments"
        check."""
        return sum(
            3 if type_of(a).kind == TypeKind.SLICE or isinstance(a, NoneLiteral) else 1
            for a in args
        )

    def _gen_materialize_argument_temp_into(self, expr: Node, t: Type, dst: Register) -> list[Instruction]:
        """Materializes an array- or struct-typed expression with no
        address of its own (an ArrayLiteral, a struct literal, or an
        array/struct-returning Call) into addressable storage, leaving
        that address in `dst`. Also used by gen_print_call_into for an
        array-typed print argument, not just call arguments.

        Producing the value itself reuses gen_array_value_into/
        gen_struct_value_into (the same methods gen_var_decl/
        gen_assign/gen_return already call) into whichever destination
        is chosen below:

          - SMALL (within is_heap_allocated's size threshold):
            _collect_argument_temps already reserved this expr a stack
            slot (keyed by id(expr)) -- write into it, then take its
            address.
          - LARGE: no slot was reserved -- malloc a fresh block sized
            to fit (_gen_malloc_array), write into it, and use the
            pointer directly. Never freed (matches this compiler's
            current memory model); never needs to survive past the
            call, since the callee's own entry-time copy is the only
            thing that reads through it.
        """
        gen_value_into = self.gen_struct_value_into if t.kind == TypeKind.STRUCT else self.gen_array_value_into
        if is_heap_allocated(t, self.struct_registry):
            instructions = self._gen_malloc_array(t)  # leaves the fresh pointer in %rax
            instructions.extend(gen_value_into(expr, Memory('rax', 0), t))
            if dst.name != 'rax':
                instructions.append(MovQ(src=Register('rax'), dst=dst))
            return instructions
        offset = self._argument_temp_offsets[id(expr)]
        instructions = gen_value_into(expr, Memory('rbp', offset), t)
        instructions.append(LeaQFrame(offset=offset, dst=dst))
        return instructions
