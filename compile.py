"""Compiler CLI: .ht -> assembly."""

import argparse
import sys

from backend import lower_to_asm
from diagnostics import CompileError, run_cli, write_output
from dump import STAGES, dump as dump_stage
from ir.program_builder import build_ir_program
from modules import discover_modules
from optimize.optimizer import optimize
from semantic import analyze
from target import TARGET_NAMES, as_target, default_target
from typed_ast import dump


def add_target_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--target', choices=TARGET_NAMES, default=str(default_target()),
                        help=f'Target (arch-os). Default: {default_target()}')


def main():
    parser = argparse.ArgumentParser(description='Hornet compiler.')
    parser.add_argument('file', type=str, help='Source file to compile.')
    add_target_argument(parser)
    parser.add_argument('-o', '--output', type=str, default=None, help='Write assembly to this file instead of stdout')
    parser.add_argument('--traceback', action='store_true', help='Show Python tracebacks for all errors')
    parser.add_argument('--dump', choices=STAGES, default=None,
                        help="Print what a stage of the compiler produces, instead of assembly (see dump.py)")

    args = parser.parse_args()
    def produce():
        if args.dump:
            text = dump_stage(args.file, args.dump)
        else:
            text = compile_to_asm(args.file, args.target)
        # Latin-1: str literals are raw bytes 0-255; UTF-8 would re-encode >= 128.
        if args.output:
            write_output(args.output, text)
        else:
            sys.stdout.buffer.write(text.encode('latin-1'))

    run_cli(produce, args.traceback)


def generate_asm(program, target=None) -> str:
    """Build IR from a typed program (semantic.analyze()'s result), optimize, and lower to assembly
    for `target` (a Target, an `arch-os` string, or None for the default)."""
    return lower_to_asm(optimize(build_ir_program(program)), as_target(target))


def typed_tree(source: str) -> str:
    """The typed tree of `source` (typed_ast.dump)."""
    entry_program, discovered_modules = discover_modules(source)
    return dump(analyze(entry_program, discovered_modules))


def compile_to_asm(source: str, target=None, require_main: bool = False) -> str:
    """Discover modules, analyze, and lower `source` to assembly. An executable needs
    `require_main`; without it the assembly may be code for another program to link."""
    entry_program, discovered_modules = discover_modules(source)
    program = analyze(entry_program, discovered_modules)
    if require_main and not any(fn.name == 'main' for fn in program.functions):
        raise CompileError("no 'main' function: a program starts at 'def int main()'", source)
    return generate_asm(program, target)


if __name__ == '__main__':
    main()
