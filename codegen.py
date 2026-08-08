"""Codegen

Transforms the AST produced by parser.py into x64 assembly (AT&T syntax).

This follows the classic two-stage codegen structure used by many
introductory compilers:

    AST (parser.py)  -->  Assembly AST  -->  Assembly text

Keeping the "Assembly AST" as its own intermediate representation --
rather than emitting strings directly from the source AST -- makes this
much easier to extend later. Once expressions get more complex you'll
need to allocate registers/stack slots as part of building that assembly
AST; once you add more statements you'll add more instruction types.
None of that should require touching the string-formatting code at all,
and none of the string-formatting quirks (spacing, symbol prefixes, etc.)
should leak into the code-generation logic.

Supported so far (matches what parser.py can currently produce):

    Program  -> one or more Functions
    Function -> a name and a body of statements
    Return   -> evaluate an expression into %eax, then `ret`
    Constant -> an immediate value
    Unary    -> NEGATE ('-'), COMPLEMENT ('~'), and NOT ('!'), each
                applied in place to whatever's already in the
                destination register

Unlike Constant, a Unary expression can't be represented as a bare
Operand (there's no such thing as "the immediate value of `-2`" as an
assembly-level concept -- the negation has to actually happen on a
register). So expression codegen works by emitting instructions that
compute the expression's value directly into a destination (see
`gen_expr_into`), rather than by returning an Operand. Nested unary
operators fall out of this for free: `~-2` recurses inward first
(computing -2 into %eax), then applies the outer operator to whatever's
now sitting in that register (`notl %eax`).
"""

import argparse
from dataclasses import dataclass, field
from typing import List

from lexer import lex
from parser import (
    Constant,
    Function,
    Node,
    Parser,
    Program,
    Return,
    Unary,
    UnaryOp,
)


# ---------------------------------------------------------------------------
# Assembly AST
# ---------------------------------------------------------------------------

class Operand:
    def emit(self) -> str:
        raise NotImplementedError


@dataclass
class Imm(Operand):
    value: int

    def emit(self) -> str:
        return f"${self.value}"


@dataclass
class Register(Operand):
    name: str  # e.g. 'eax'

    def emit(self) -> str:
        return f"%{self.name}"


class Instruction:
    """Base class for assembly instructions.

    Subclasses set `mnemonic` and implement `operands()`; `emit()` is
    generic and handles column alignment (via `mnemonic.ljust(8)`) so
    every instruction lines up the same way regardless of how long its
    mnemonic is -- compare `movl` (4 chars) and `movzbl` (6 chars) in the
    examples below, both of which align their first operand to column 8.
    """

    mnemonic: str = ""

    def operands(self) -> List[str]:
        return []

    def emit(self) -> str:
        ops = self.operands()
        if not ops:
            return self.mnemonic
        return f"{self.mnemonic:<8}{', '.join(ops)}"


@dataclass
class Mov(Instruction):
    src: Operand
    dst: Operand
    mnemonic = "movl"

    def operands(self) -> List[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class Neg(Instruction):
    """Two's-complement arithmetic negation, in place: dst = -dst."""
    operand: Operand
    mnemonic = "negl"

    def operands(self) -> List[str]:
        return [self.operand.emit()]


@dataclass
class Not(Instruction):
    """Bitwise complement, in place: dst = ~dst."""
    operand: Operand
    mnemonic = "notl"

    def operands(self) -> List[str]:
        return [self.operand.emit()]


@dataclass
class Cmp(Instruction):
    """Compares src and dst by computing dst - src and setting flags
    (notably ZF) accordingly -- doesn't modify either operand."""
    src: Operand
    dst: Operand
    mnemonic = "cmpl"

    def operands(self) -> List[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class SetE(Instruction):
    """Sets an 8-bit operand to 1 if the last Cmp found its operands
    equal (ZF set), else 0. Used to implement logical NOT: `!x` is
    exactly "was x equal to 0?"."""
    operand: Operand
    mnemonic = "sete"

    def operands(self) -> List[str]:
        return [self.operand.emit()]


@dataclass
class MovZX(Instruction):
    """Zero-extends an 8-bit src into a 32-bit dst. Needed after SetE,
    since `sete` only ever writes the low byte (e.g. %al) and leaves the
    rest of the containing 32-bit register (e.g. %eax) untouched -- so
    without this, %eax could still hold garbage in its upper 24 bits."""
    src: Operand
    dst: Operand
    mnemonic = "movzbl"

    def operands(self) -> List[str]:
        return [self.src.emit(), self.dst.emit()]


@dataclass
class Ret(Instruction):
    mnemonic = "ret"


@dataclass
class AsmFunction:
    name: str
    instructions: List[Instruction] = field(default_factory=list)


@dataclass
class AsmProgram:
    functions: List[AsmFunction] = field(default_factory=list)


# 32-bit register name -> its 8-bit low-byte alias (e.g. %eax -> %al).
# `sete` (and friends) can only target an 8-bit operand, so codegen needs
# to be able to get from "the register I'm working in" to "its byte
# alias". Only registers actually in use are listed; extend this table
# alongside whatever new registers the code generator starts using.
_BYTE_REGISTER_ALIASES = {
    'eax': 'al',
}


def as_byte_register(reg: Operand) -> Register:
    if not isinstance(reg, Register) or reg.name not in _BYTE_REGISTER_ALIASES:
        raise CodegenError(f"No 8-bit alias known for register operand: {reg!r}")
    return Register(_BYTE_REGISTER_ALIASES[reg.name])


# ---------------------------------------------------------------------------
# AST -> Assembly AST
# ---------------------------------------------------------------------------

class CodegenError(Exception):
    """Raised when the code generator encounters an AST node it doesn't
    know how to translate yet."""


class CodeGenerator:
    """Walks the source AST (Program/Function/Return/Constant/...) and
    produces an equivalent AsmProgram."""

    def generate(self, program: Program) -> AsmProgram:
        return AsmProgram(functions=[self.gen_function(fn) for fn in program.functions])

    def gen_function(self, fn: Function) -> AsmFunction:
        instructions: List[Instruction] = []
        for stmt in fn.body:
            instructions.extend(self.gen_statement(stmt))
        return AsmFunction(name=fn.name, instructions=instructions)

    def gen_statement(self, stmt: Node) -> List[Instruction]:
        if isinstance(stmt, Return):
            return self.gen_return(stmt)
        raise CodegenError(f"No codegen rule for statement: {stmt!r}")

    def gen_return(self, stmt: Return) -> List[Instruction]:
        dst = Register('eax')
        instructions = self.gen_expr_into(stmt.value, dst)
        instructions.append(Ret())
        return instructions

    def gen_expr_into(self, expr: Node, dst: Operand) -> List[Instruction]:
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
            return [Mov(src=Imm(expr.value), dst=dst)]
        if isinstance(expr, Unary):
            # Compute the operand into dst first, then apply this node's
            # operator to whatever's now there. This is what makes chained
            # operators (`~-2`) work: the inner Unary's instructions run
            # first, then the outer operator's instructions run on top.
            instructions = self.gen_expr_into(expr.operand, dst)
            instructions.extend(self.gen_unary_op(expr.op, dst))
            return instructions
        raise CodegenError(f"No codegen rule for expression: {expr!r}")

    def gen_unary_op(self, op: UnaryOp, dst: Operand) -> List[Instruction]:
        if op == UnaryOp.NEGATE:
            return [Neg(dst)]
        if op == UnaryOp.COMPLEMENT:
            return [Not(dst)]
        if op == UnaryOp.NOT:
            # `!x` is "1 if x == 0, else 0". cmpl sets flags from
            # (dst - 0); sete writes 1/0 into the low byte based on
            # those flags; movzbl zero-extends that byte back out to
            # fill the full 32-bit destination register.
            byte_dst = as_byte_register(dst)
            return [
                Cmp(src=Imm(0), dst=dst),
                SetE(byte_dst),
                MovZX(src=byte_dst, dst=dst),
            ]
        raise CodegenError(f"No codegen rule for unary operator: {op}")


# ---------------------------------------------------------------------------
# Assembly AST -> text
# ---------------------------------------------------------------------------

class Emitter:
    """Renders an AsmProgram as textual x64 AT&T-syntax assembly.

    `platform` controls the two portability wrinkles that matter at this
    stage of the compiler:
      - macOS (Mach-O) requires a leading underscore on external symbols
        (e.g. `_main`); Linux (ELF) does not.
      - Linux toolchains generally expect a `.note.GNU-stack` section so
        the linker doesn't warn about an executable stack; macOS doesn't
        use this.
    """

    def __init__(self, platform: str = 'macos'):
        if platform not in ('macos', 'linux'):
            raise ValueError("platform must be 'macos' or 'linux'")
        self.platform = platform

    def symbol(self, name: str) -> str:
        return f"_{name}" if self.platform == 'macos' else name

    def emit(self, program: AsmProgram) -> str:
        lines: List[str] = []
        for fn in program.functions:
            lines.extend(self.emit_function(fn))
            lines.append("")  # blank line between functions
        if self.platform == 'linux':
            lines.append('.section .note.GNU-stack,"",@progbits')
        return "\n".join(lines).rstrip() + "\n"

    def emit_function(self, fn: AsmFunction) -> List[str]:
        sym = self.symbol(fn.name)
        lines = [f"    .globl {sym}", f"{sym}:"]
        for instr in fn.instructions:
            lines.append(f"    {instr.emit()}")
        return lines


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------

def generate_asm(program: Program, platform: str = 'macos') -> str:
    asm_program = CodeGenerator().generate(program)
    return Emitter(platform=platform).emit(asm_program)


def compile_to_asm(filename: str, platform: str = 'macos') -> str:
    tokens = lex(filename)
    ast = Parser(tokens).parse_program()
    return generate_asm(ast, platform=platform)


def main():
    arg_parser = argparse.ArgumentParser(description='Assembly generator')
    arg_parser.add_argument('file', type=str, help='Source file to compile.')
    arg_parser.add_argument(
        '--platform', choices=['macos', 'linux'], default='macos',
        help="Target platform; affects symbol naming. Default: macos",
    )
    arg_parser.add_argument(
        '-o', '--output', type=str, default=None,
        help='Write assembly to this file instead of stdout.',
    )
    args = arg_parser.parse_args()

    asm = compile_to_asm(args.file, platform=args.platform)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(asm)
    else:
        print(asm, end='')


if __name__ == '__main__':
    main()
