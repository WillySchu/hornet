"""Compiler CLI: .ht -> assembly."""

import argparse
import sys

from codegen.codegen import generate_asm
from desugar import desugar_methods
from merge import merge_programs
from modules import discover_modules
from semantic import analyze

# gcc on Apple Silicon defaults to arm64; match build.py's host detection.
HOST_IS_MACOS = sys.platform == "darwin"
DEFAULT_PLATFORM = "macos" if HOST_IS_MACOS else "linux"


def main():
    parser = argparse.ArgumentParser(description='Hornet compiler.')
    parser.add_argument('file', type=str, help='Source file to compile.')
    parser.add_argument('--platform', choices=['macos', 'linux'], default=DEFAULT_PLATFORM, help=f'Target platform. Default: {DEFAULT_PLATFORM} (this host)')
    parser.add_argument('-o', '--output', type=str, default=None, help='Write assembly to this file instead of stdout')

    args = parser.parse_args()

    asm = compile_to_asm(args.file, args.platform)
    # Latin-1: str literals are raw bytes 0-255; UTF-8 would re-encode >= 128.
    if args.output:
        with open(args.output, 'w', encoding='latin-1') as f:
            f.write(asm)
    else:
        sys.stdout.buffer.write(asm.encode('latin-1'))


def compile_to_asm(source: str, platform: str = 'macos') -> str:
    """Discover, merge, desugar, analyze, and lower `source` to assembly."""
    entry_program, discovered_modules = discover_modules(source)
    ast = merge_programs(entry_program, discovered_modules)
    desugar_methods(ast)  # Must precede analyze().
    analyze(ast)
    return generate_asm(ast, platform=platform)


if __name__ == '__main__':
    main()
