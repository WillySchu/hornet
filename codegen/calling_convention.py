"""Shared, small pieces of calling-convention accounting used by both
real-IR call construction (see _ir_call/_ir_composite_call in
scalars.py, and _ir_call_arguments) and gen_function_ir's own
prologue/epilogue: _total_arg_slots (this module's own one remaining
method) and CALLEE_SAVED_SCRATCH_REGISTERS.

Used to also hold the old-style argument-marshaling layer itself
(_gen_call_arguments_into and _gen_materialize_argument_temp_into) --
removed entirely as part of this arc's old-style dead-code cleanup
(see ir.py's own top docstring): every call this compiler builds now
marshals its own arguments through _ir_call_arguments instead (see
scalars.py), which these two methods had no remaining callers left to
serve."""

from codegen.utils import type_of
from parser import Node, NoneLiteral
from semantic import TypeKind

# Registers gen_string_concat_into/gen_string_compare_into used to use
# as scratch, back when they were old-style methods (both since
# removed -- see this module's own top docstring): every function's
# prologue/epilogue still saves and restores these unconditionally
# regardless of whether that specific function uses them itself,
# since the callee-saved contract must hold for any call.
CALLEE_SAVED_SCRATCH_REGISTERS = ['rbx', 'r12', 'r13', 'r14']


class CallingConventionMixin:

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

