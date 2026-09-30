"""Compiler CLI: .ht -> assembly."""

import argparse
import sys

from desugar import desugar_methods
from backend import lower_to_asm
from diagnostics import run_cli
from ir.program_builder import build_ir_program
from merge import merge_programs
from modules import discover_modules
from optimize.optimizer import optimize
from semantic import analyze
from target import TARGET_NAMES, as_target, default_target


def add_target_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--target', choices=TARGET_NAMES, default=str(default_target()),
                        help=f'Target (arch-os). Default: {default_target()}')


def main():
    parser = argparse.ArgumentParser(description='Hornet compiler.')
    parser.add_argument('file', type=str, help='Source file to compile.')
    add_target_argument(parser)
    parser.add_argument('-o', '--output', type=str, default=None, help='Write assembly to this file instead of stdout')
    parser.add_argument('--traceback', action='store_true', help='Show Python tracebacks for all errors')

    args = parser.parse_args()

    asm = run_cli(lambda: compile_to_asm(args.file, args.target), args.traceback)
    # Latin-1: str literals are raw bytes 0-255; UTF-8 would re-encode >= 128.
    if args.output:
        with open(args.output, 'w', encoding='latin-1') as f:
            f.write(asm)
    else:
        sys.stdout.buffer.write(asm.encode('latin-1'))


def generate_asm(program, target=None) -> str:
    """Build IR from an analyzed Program, optimize, and lower to assembly for `target`
    (a Target, an `arch-os` string, or None for the default)."""
    return lower_to_asm(optimize(build_ir_program(program)), as_target(target))


def compile_to_asm(source: str, target=None) -> str:
    """Discover, merge, desugar, analyze, and lower `source` to assembly."""
    entry_program, discovered_modules = discover_modules(source)
    ast = merge_programs(entry_program, discovered_modules)
    desugar_methods(ast)  # Must precede analyze().
    analyze(ast)
    return generate_asm(ast, target)


if __name__ == '__main__':
    main()
