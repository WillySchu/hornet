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
    def emit(self) -> str:
        raise NotImplementedError


@dataclass
class Mov(Instruction):
    src: Operand
    dst: Operand

    def emit(self) -> str:
        return f"movl    {self.src.emit()}, {self.dst.emit()}"


@dataclass
class Ret(Instruction):
    def emit(self) -> str:
        return "ret"


@dataclass
class AsmFunction:
    name: str
    instructions: List[Instruction] = field(default_factory=list)


@dataclass
class AsmProgram:
    functions: List[AsmFunction] = field(default_factory=list)


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
        value = self.gen_expression(stmt.value)
        return [
            Mov(src=value, dst=Register('eax')),
            Ret(),
        ]

    def gen_expression(self, expr: Node) -> Operand:
        if isinstance(expr, Constant):
            return Imm(expr.value)
        raise CodegenError(f"No codegen rule for expression: {expr!r}")


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
