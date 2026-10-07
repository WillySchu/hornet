"""Build and verify an IRProgram from an analyzed Program."""

from dataclasses import fields, replace

from escape_analysis import compute_escape_summaries
from ir.errors import IRError
from ir.id_allocator import IdAllocator
from ir.ir import IRProgram
from ir.division_checks import insert_division_checks
from ir.null_checks import expand_null_checks
from ir.typed_builder import NotYetPorted, TypedFunctionBuilder
from ir.verify import verify_program
import typed_ast as typed


def _called(node, names: set) -> None:
    """Add the name of every Hornet function called under `node` (a typed node, or a tuple of them)."""
    if isinstance(node, tuple):
        for item in node:
            _called(item, names)
    elif isinstance(node, typed._Node):
        if isinstance(node, typed.Call) and node.kind == 'function':
            names.add(node.name)
        for f in fields(node):
            _called(getattr(node, f.name), names)


def reachable_functions(program: typed.Program) -> tuple:
    """The functions that `main` can reach through calls, in the program's order. Every call is
    resolved by now, and nothing else can call a function (there are no function values, and C
    can't call in), so the rest can never run. A program without `main` keeps them all."""
    by_name = {fn.name: fn for fn in program.functions}
    if 'main' not in by_name:
        return program.functions
    reached, pending = set(), ['main']
    while pending:
        name = pending.pop()
        if name in reached or name not in by_name:  # (an extern has no function here)
            continue
        reached.add(name)
        called: set = set()
        _called(by_name[name].body, called)
        pending.extend(called)
    return tuple(fn for fn in program.functions if fn.name in reached)


def build_ir_program(program: typed.Program, keep_unreachable: bool = False) -> IRProgram:
    """IR for a typed program (what semantic.analyze() returns); nothing from the parser's tree.
    Functions `main` can't reach are left out, checked though they were, unless `keep_unreachable`:
    so are the strings and descriptors only they use, which are made as a function is built."""
    if not isinstance(program, typed.Program):
        raise IRError(f"build_ir_program takes semantic.analyze()'s typed program, not a {type(program).__name__}")
    if not keep_unreachable:
        program = replace(program, functions=reachable_functions(program))
    ir_program = IRProgram(struct_registry=program.structs, sum_type_registry=program.sum_types,
                           enum_registry=program.enums, ids=IdAllocator())
    ir_program.escape_summaries = compute_escape_summaries(program.functions, program.structs)
    ir_program.functions = []
    for typed_fn in program.functions:
        try:
            ir_program.functions.append(TypedFunctionBuilder(ir_program).build(typed_fn))
        except NotYetPorted as e:
            raise IRError(f"No IR for {typed_fn.name}: {e}") from e
    for ir_fn in ir_program.functions:
        expand_null_checks(ir_fn, ir_program)
        insert_division_checks(ir_fn, ir_program)
    verify_program(ir_program)
    return ir_program
