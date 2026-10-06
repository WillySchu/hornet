"""What each stage of the compiler produces, as text: `compile.py --dump STAGE`.

Every dump is deterministic, with one token, node, or instruction per line. A dump runs the
compiler only as far as its stage, so `tokens` works on a file that doesn't parse and `tree` on one
that doesn't type-check. `tokens` and `tree` are of the one file named; `typed`, `ir`, and
`optimized-ir` are of the whole program, its imported modules included.
"""

from dataclasses import fields
from enum import Enum

import parser as syntax
import typed_ast
from ir.ir import (
    IRBinOp, IRBoundsCheck, IRBranch, IRCall, IRCast, IRConst, IRCopy, IRFunction, IRJump, IRLabel, IRLoad,
    IRLocalAddress, IRMove, IRNullCheck, IRProgram, IRReturn, IRSliceBoundsCheck, IRStaticDataAddress, IRStore,
    IRUnOp,
)
from ir.program_builder import build_ir_program
from lexer import TokenType, lex
from modules import discover_modules
from ops import BinaryOp
from optimize.optimizer import optimize
from semantic import analyze
from typed_ast import quoted, scalar_text
from typesys import Type

STAGES = ('tokens', 'tree', 'typed', 'ir', 'optimized-ir')


def dump(source: str, stage: str) -> str:
    """The text of what `stage` produces for the program in the file `source`."""
    if stage == 'tokens':
        return dump_tokens(lex(source))
    if stage == 'tree':
        return dump_tree(syntax.Parser(lex(source)).parse_program())
    entry_program, discovered_modules = discover_modules(source)
    program = analyze(entry_program, discovered_modules)
    if stage == 'typed':
        return typed_ast.dump(program)
    ir_program = build_ir_program(program)
    return dump_ir(ir_program if stage == 'ir' else optimize(ir_program))


# ---- tokens

# Tokens whose text is theirs alone (the rest are keywords and punctuation, named by their kind).
_WITH_TEXT = (TokenType.IDENTIFIER, TokenType.NUMBER, TokenType.STRING, TokenType.BYTE)


def dump_tokens(tokens: list) -> str:
    """`line:col KIND`, then the text of a name or a literal, as it is written in the source."""
    lines = []
    for token in tokens:
        text = f" {token.val}" if token.type in _WITH_TEXT else ''
        lines.append(f"{token.line}:{token.col} {token.type.name}{text}")
    return '\n'.join(lines) + '\n'


# ---- the parser's tree

_TYPE_EXPRS = (syntax.ArrayTypeExpr, syntax.SliceTypeExpr, syntax.PointerTypeExpr, syntax.DictTypeExpr,
               syntax.QualifiedTypeExpr)
_HIDDEN = {'line', 'col', 'file', 'nid'}


def _type_text(expr) -> str:
    """A type as it is written."""
    if isinstance(expr, syntax.ArrayTypeExpr):
        size = expr.size if isinstance(expr.size, int) else _expression_text(expr.size)
        return f"[{size}]{_type_text(expr.element_type)}"
    if isinstance(expr, syntax.SliceTypeExpr):
        return f"[]{_type_text(expr.element_type)}"
    if isinstance(expr, syntax.PointerTypeExpr):
        return f"*{_type_text(expr.pointee_type)}"
    if isinstance(expr, syntax.DictTypeExpr):
        return f"dict[{_type_text(expr.key_type)}]{_type_text(expr.value_type)}"
    if isinstance(expr, syntax.QualifiedTypeExpr):
        return f"{expr.module}.{expr.name}"
    return str(expr)


def _expression_text(node) -> str:
    """A constant expression in a type (an array's size) on one line."""
    if isinstance(node, syntax.Constant):
        return str(node.value)
    if isinstance(node, syntax.Variable):
        return node.name
    if isinstance(node, syntax.Field):
        return f"{_expression_text(node.base)}.{node.name}"
    if isinstance(node, syntax.Binary):
        return f"({_expression_text(node.left)} {_operator(node.op)} {_expression_text(node.right)})"
    return ' '.join(node.pretty().split())


def _operator(op) -> str:
    return op.name if isinstance(op, Enum) else str(op)


def _is_scalar(value) -> bool:
    return value is None or isinstance(value, (bool, int, float, str, Enum)) or isinstance(value, _TYPE_EXPRS)


def _tree_scalar(name: str, value) -> str:
    if isinstance(value, _TYPE_EXPRS):
        return _type_text(value)
    return _operator(value) if isinstance(value, Enum) else scalar_text(name, value)


def _tree_value(value, depth: int, lines: list) -> None:
    """A node, a list of values, or a tuple (an entry: a name and its value, a check and its body)."""
    indent = "  " * depth
    if isinstance(value, syntax.Node) and not isinstance(value, _TYPE_EXPRS):
        shown = [(f.name, getattr(value, f.name)) for f in fields(value) if f.name not in _HIDDEN]
        attrs = ''.join(f" {name}={_tree_scalar(name, v)}" for name, v in shown if _is_scalar(v) and v is not None)
        lines.append(indent + type(value).__name__ + attrs)
        for name, v in shown:
            if not _is_scalar(v) and (v or isinstance(v, syntax.Node)):
                lines.append(f"{indent}  {name}:")
                _tree_value(v, depth + 2, lines)
    elif isinstance(value, list):
        for item in value:
            _tree_value(item, depth, lines)
    elif isinstance(value, tuple):
        lines.append(indent + "entry")
        for item in value:
            _tree_value(item, depth + 1, lines)
    else:
        lines.append(indent + _tree_scalar('', value))


def dump_tree(program: syntax.Program) -> str:
    """One node per line as `Kind field=value ...`, its child nodes indented under `field:`."""
    lines = []
    _tree_value(program, 0, lines)
    return '\n'.join(lines) + '\n'


# ---- IR

def _operand(value) -> str:
    """`t3` for a Temp; a constant's value, with its type unless that is int."""
    if isinstance(value, IRConst):
        return str(value.value) if value.type == Type.INT else f"{value.value}:{value.type}"
    return f"t{value.id}"


def _place(address, offset: int) -> str:
    return _operand(address) + (f"+{offset}" if offset else '')


def _defined(temp) -> str:
    return f"t{temp.id}: {temp.type}"


def _instruction(instr) -> str:
    # Where a panic it can raise is reported: a check's, or a division's.
    panics = not isinstance(instr, IRBinOp) or instr.op in (BinaryOp.DIVIDE, BinaryOp.MODULO)
    at = f" at {instr.where}" if panics and getattr(instr, 'where', None) else ''
    if isinstance(instr, IRLabel):
        return f"{instr.name}:"
    if isinstance(instr, IRMove):
        text = f"{_defined(instr.dst)} = {_operand(instr.src)}"
    elif isinstance(instr, IRBinOp):
        text = f"{_defined(instr.dst)} = {instr.op.name.lower()} {_operand(instr.left)}, {_operand(instr.right)}"
    elif isinstance(instr, IRUnOp):
        text = f"{_defined(instr.dst)} = {instr.op.name.lower()} {_operand(instr.operand)}"
    elif isinstance(instr, IRCast):
        text = f"{_defined(instr.dst)} = cast {_operand(instr.src)}"
    elif isinstance(instr, IRCall):
        call = f"call {instr.name}({', '.join(_operand(a) for a in instr.args)})"
        text = call if instr.dst is None else f"{_defined(instr.dst)} = {call}"
    elif isinstance(instr, IRReturn):
        text = "return" if instr.value is None else f"return {_operand(instr.value)}"
    elif isinstance(instr, IRLoad):
        text = f"{_defined(instr.dst)} = load {_place(instr.address, instr.offset)}"
    elif isinstance(instr, IRStore):
        text = f"store {_place(instr.address, instr.offset)} = {_operand(instr.value)} as {instr.value_type}"
    elif isinstance(instr, IRLocalAddress):
        text = f"{_defined(instr.dst)} = address of slot {instr.slot}"
    elif isinstance(instr, IRStaticDataAddress):
        text = f"{_defined(instr.dst)} = address of {instr.label}"
    elif isinstance(instr, IRCopy):
        text = (f"copy {_place(instr.dst_address, instr.dst_offset)} = {_place(instr.src_address, instr.src_offset)} "
                f"as {instr.value_type}")
    elif isinstance(instr, IRNullCheck):
        text = f"check {_operand(instr.pointer)} is not none"
    elif isinstance(instr, IRBoundsCheck):
        text = f"check {_operand(instr.index)} < {_operand(instr.length)}"
    elif isinstance(instr, IRSliceBoundsCheck):
        text = f"check {_operand(instr.value)} <= {_operand(instr.bound)}"
    elif isinstance(instr, IRJump):
        text = f"jump {instr.label}"
    elif isinstance(instr, IRBranch):
        text = f"branch {_operand(instr.cond)} ? {instr.true_label} : {instr.false_label}"
    else:
        raise TypeError(f"no dump for the instruction {type(instr).__name__}")
    return "  " + text + at


def _function(fn: IRFunction, lines: list) -> None:
    result = f" -> {fn.return_type}" if fn.return_type is not None and str(fn.return_type) != 'void' else ''
    lines.append(f"function {fn.name}({', '.join(_defined(p) for p in fn.params)}){result}")
    for slot in sorted(fn.slot_widths):
        label = fn.slot_labels.get(slot)
        returned = "; the result's address" if slot == fn.hidden_return_ptr_slot else ''
        lines.append(f"  slot {slot}: {fn.slot_widths[slot]} bytes{f' ({label})' if label else ''}{returned}")
    for temp_id in sorted(fn.temp_homes):
        lines.append(f"  t{temp_id} lives in slot {fn.temp_homes[temp_id]}")
    lines.extend(_instruction(instr) for instr in fn.body)
    lines.append("")


def dump_ir(program: IRProgram) -> str:
    """Each function (its parameters, frame slots, and instructions), then the program's static data:
    its strings, and its type descriptors as their words."""
    lines = []
    for fn in program.functions:
        _function(fn, lines)
    for label, text in program.string_literals:
        lines.append(f"string {label} = {quoted(text)}")
    for label, words in program.type_descriptors:
        lines.append(f"words {label} = {', '.join(str(word) for word in words)}")
    return '\n'.join(lines) + '\n'
