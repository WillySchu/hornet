"""Entry point into the compiler."""

import argparse

from codegen.codegen import compile_to_asm


def main():
    parser = argparse.ArgumentParser(description='Hornet compiler.')
    parser.add_argument('file', type=str, help='Source file to compile.')
    parser.add_argument('--platform', choices=['macos', 'linux'], default='linux', help='Target platform. Default: linux')
    parser.add_argument('-o', '--output', type=str, default=None, help='Write assembly to this file instead of stdout')

    args = parser.parse_args()

    asm = compile_to_asm(args.file, platform=args.platform)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(asm)
    else:
        print(asm, end='')


if __name__ == '__main__':
    main()
