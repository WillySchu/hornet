"""Entry point into the compiler."""

import argparse

from codegen.codegen import generate_asm
from desugar import desugar_methods
from lexer import lex
from parser import Parser
from semantic import analyze


def main():
    parser = argparse.ArgumentParser(description='Hornet compiler.')
    parser.add_argument('file', type=str, help='Source file to compile.')
    parser.add_argument('--platform', choices=['macos', 'linux'], default='linux', help='Target platform. Default: linux')
    parser.add_argument('-o', '--output', type=str, default=None, help='Write assembly to this file instead of stdout')

    args = parser.parse_args()

    asm = compile_to_asm(args.file, args.platform)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(asm)
    else:
        print(asm, end='')


def compile_to_asm(source: str, platform: str = 'macos') -> str:
    tokens = lex(source)
    ast = Parser(tokens).parse_program()
    desugar_methods(ast)  # must run before analyze() -- see its own module docstring for why
    analyze(ast)  # raises SemanticError before any code is generated
    return generate_asm(ast, platform=platform)


if __name__ == '__main__':
    main()
