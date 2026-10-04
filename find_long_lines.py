#!/usr/bin/env python3
"""Recursively scan a directory and report lines longer than a given length.

Usage:
    python find_long_lines.py [directory] [--max-length N] [--exclude NAME ...]

Output format (one hit per line):
    path:line_number:length: line contents
"""

import argparse
import os
import sys


def is_binary(path):
    """Treat a file as binary if its first chunk contains a NUL byte."""
    with open(path, "rb") as f:
        return b"\0" in f.read(8192)


def find_long_lines(root, max_length, exclude):
    """Yield (path, line_number, line) for every line longer than max_length."""
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories in place so os.walk doesn't descend into them.
        dirnames[:] = sorted(d for d in dirnames if d not in exclude)

        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            if not os.path.isfile(path):  # skip sockets, FIFOs, broken symlinks
                continue
            if name in exclude:
                continue
            try:
                if is_binary(path):
                    continue
                with open(path, encoding="utf-8", errors="replace") as f:
                    for line_number, line in enumerate(f, start=1):
                        line = line.rstrip("\r\n")
                        if len(line) > max_length:
                            yield path, line_number, line
            except OSError as err:
                print(f"warning: could not read {path}: {err}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Report lines longer than a given length in all files under a directory."
    )
    parser.add_argument("directory", nargs="?", default=".", help="directory to scan (default: current)")
    parser.add_argument("-n", "--max-length", type=int, default=120, help="length limit (default: 120)")
    parser.add_argument(
        "-x", "--exclude", nargs="*", default=[], metavar="NAME",
        help="directory names to skip, e.g. .git node_modules",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        parser.error(f"not a directory: {args.directory}")

    count = 0
    for path, line_number, line in find_long_lines(args.directory, args.max_length, set(args.exclude)):
        print(f"{path}:{line_number}:{len(line)}: {line}")
        count += 1

    print(f"{count} line(s) over {args.max_length} characters", file=sys.stderr)


if __name__ == "__main__":
    main()
