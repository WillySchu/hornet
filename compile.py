"""Entry point into the compiler."""

import argparse
import sys

from codegen.codegen import generate_asm
from desugar import desugar_methods
from lexer import lex
from parser import Parser
from semantic import analyze

# Same convention build.py's own HOST_IS_MACOS/DEFAULT_PLATFORM and
# tests/test_compiler.py's own HOST_IS_MACOS/ASM_PLATFORM already use --
# see build.py's own comment for why this matters (gcc's default
# target on Apple Silicon is arm64, which can't assemble this
# compiler's x86-64 AT&T-syntax output at all). This CLI's own
# --platform default previously just hardcoded 'linux' regardless of
# host, unlike either of those -- a macOS user invoking this file
# directly, with no --platform given, would silently get Linux-shaped
# assembly (no leading underscore on symbols) their own host's gcc/as
# can't consume correctly.
HOST_IS_MACOS = sys.platform == "darwin"
DEFAULT_PLATFORM = "macos" if HOST_IS_MACOS else "linux"


def main():
    parser = argparse.ArgumentParser(description='Hornet compiler.')
    parser.add_argument('file', type=str, help='Source file to compile.')
    parser.add_argument('--platform', choices=['macos', 'linux'], default=DEFAULT_PLATFORM, help=f'Target platform. Default: {DEFAULT_PLATFORM} (this host)')
    parser.add_argument('-o', '--output', type=str, default=None, help='Write assembly to this file instead of stdout')

    args = parser.parse_args()

    asm = compile_to_asm(args.file, args.platform)
    # Latin-1, not the default UTF-8: an emitted str literal's own raw
    # bytes (see ir/strings.py's own module docstring -- str is a byte
    # sequence, not Unicode text) can legitimately include any 0-255
    # value now that string/byte literals support \xNN escapes (see
    # parser.py's own _unescape_quoted_literal) -- Latin-1 is the one
    # encoding where every code point 0-255 maps to exactly that one
    # byte, so this is the only choice that keeps the emitted .data
    # byte count matching len()'s own, compile-time-computed value
    # exactly, for every byte, not just the ASCII ones. UTF-8 would
    # silently re-encode anything >= 128 into two or more bytes,
    # corrupting that byte-for-byte correspondence.
    if args.output:
        with open(args.output, 'w', encoding='latin-1') as f:
            f.write(asm)
    else:
        sys.stdout.buffer.write(asm.encode('latin-1'))


def compile_to_asm(source: str, platform: str = 'macos') -> str:
    tokens = lex(source)
    ast = Parser(tokens).parse_program()
    desugar_methods(ast)  # must run before analyze() -- see its own module docstring for why
    analyze(ast)  # raises SemanticError before any code is generated
    return generate_asm(ast, platform=platform)


if __name__ == '__main__':
    main()
