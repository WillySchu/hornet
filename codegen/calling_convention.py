"""TODO"""

from codegen.assembly_ast import Instruction, Register, Push, Pop, Memory, MovQ, LeaQFrame
from codegen.escape_analysis import is_heap_allocated
from codegen.utils import type_of, ARG_REGISTERS_64
from parser import Node, NoneLiteral, Variable, Field, Index
from semantic import TypeKind, Type

# Registers gen_string_concat_into/gen_string_compare_into use as
# scratch (see STRINGS). Now that functions can call each other,
# *every* function's prologue/epilogue saves and restores these
# unconditionally -- see gen_function and gen_return -- regardless of
# whether that particular function happens to use them, because the
# callee-saved contract has to hold for any call, not just ones this
# compiler happens to know use string operations. See the module
# docstring's FUNCTIONS section for why this became necessary.
CALLEE_SAVED_SCRATCH_REGISTERS = ['rbx', 'r12', 'r13', 'r14']


class CallingConventionMixin:
    def _gen_call_arguments_into(self, args: list[Node], reg_shift: int = 0) -> list[Instruction]:
        """Shared by gen_call_into, gen_array_call_into, and
        gen_slice_call_into: evaluates every argument -- in order,
        each via the ordinary gen_expr_into (a scalar), gen_array_arg_
        address_into (an array), or gen_slice_arg_into (a slice or
        none, which needs THREE consecutive register slots, not one)
        -- immediately pushing each one's resulting value(s) onto the
        stack before moving on to the next. Only *after* every
        argument has been safely computed and stacked does this start
        popping everything back off, in reverse, into the actual SysV
        argument registers (see _ARG_REGISTERS_64).

        This "compute and stack everything, then pop into place" order
        is what avoids the same register-clobbering hazard that
        motivated saving %rbx/%r12/%r13/%r14 across calls in the first
        place (see the module docstring's FUNCTIONS section): if
        argument 2 happens to be a nested call and argument 1's value
        were sitting in a scratch register instead of safely on the
        stack while argument 2 gets computed, argument 2's own use of
        that same scratch register would corrupt argument 1.

        `reg_shift` shifts every argument's own register slot later by
        that many positions -- 1 for a call to a function that returns
        an array or slice (whose hidden output pointer already
        occupies the first slot, pushed/popped separately by gen_
        array_call_into/gen_slice_call_into themselves, outside this
        method entirely), 0 otherwise.

        Because a slice argument needs three consecutive slots, the
        mapping from argument index to register index isn't the
        simple 1:1 one it used to be -- this tracks a running slot
        count instead, matching exactly how a real C compiler would
        place `struct{void*,long,long}` arguments among ordinary
        scalar ones: pushed in the same left-to-right order as written
        (a slice contributing its ptr, then its len, then its cap),
        and popped in exact reverse, so each slot lands in its own
        correct register regardless of whether it came from a whole
        scalar argument or a third of a slice one.
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
        """How many argument registers `args` will collectively need
        -- 3 for each slice-typed (or `none`) argument, 1 for
        everything else. `none`'s own resolved type (Type.NONE) never
        equals SLICE, so it's checked for separately here -- the same
        "check isinstance(expr, NoneLiteral) directly rather than
        relying on its own resolved type" pattern gen_var_decl/
        gen_assign/gen_return already use, safe for the same reason:
        semantic.py's _types_compatible already guarantees `none` is
        only ever valid where a slice is expected, since slices are
        the only nilable type that exists. Shared by every call-
        codegen entry point's own "too many arguments" check."""
        return sum(
            3 if type_of(a).kind == TypeKind.SLICE or isinstance(a, NoneLiteral) else 1
            for a in args
        )

    def _gen_materialize_argument_temp_into(self, expr: Node, t: Type, dst: Register) -> list[Instruction]:
        """Materializes an array- or struct-typed expression that has
        no address of its own -- an ArrayLiteral, a struct literal, or
        an ordinary array/struct-returning Call -- into real,
        addressable storage, and leaves that address in `dst` (a
        64-bit register). Despite its name, not exclusively for
        function-call arguments: gen_print_call_into's own array-typed
        argument reuses this exact method too (see its own docstring),
        since _collect_argument_temps already reserves a slot for
        print's own argument the same, uniform way it does for an
        ordinary call's (see that method's own note on why it doesn't
        distinguish between the two at all). This is the piece a
        Variable/Index/Field argument never needs at all (gen_array_
        address_into/gen_struct_address_into already have a real
        address for any of those); see _collect_argument_temps's own
        docstring for why this can't reuse a single shared scratch
        slot the way an unnamed slice does, and why the stack-vs-heap
        decision below mirrors is_heap_allocated's own size threshold.

        Producing the actual VALUE needs no new mechanism at all: gen_
        array_value_into and gen_struct_value_into already handle
        EVERY one of these shapes (an ArrayLiteral or a struct literal,
        via their own Call-name-is-a-struct dispatch; an ordinary
        returning Call, via gen_array_call_into/gen_struct_call_into)
        written into an arbitrary Memory destination -- the exact same
        methods gen_var_decl/gen_assign/gen_return already call. All
        that's genuinely new here is deciding WHERE to write the
        result and then taking ITS address, which is what the two
        branches below do:

          - SMALL (t is within the same is_heap_allocated size
            threshold every named local/parameter already uses):
            _collect_argument_temps already reserved this exact expr
            (keyed by id(expr)) its own permanent stack slot -- write
            directly into it, then LeaQFrame its address.
          - LARGE: no slot was reserved for it at all (see _reserve_
            argument_temp) -- malloc a fresh block sized to fit
            instead (_gen_malloc_array, despite its name already used
            generically for structs too -- see gen_var_decl's own heap
            case), write into it the exact same way, and use the
            resulting pointer directly. Never freed, matching this
            compiler's existing memory model, and never stashed
            anywhere durable either: the callee's own entry-time copy
            (see gen_function's parameter loop) is the only thing that
            ever reads through this pointer, so unlike a named heap-
            allocated local, nothing here needs to survive past the
            call itself.
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
