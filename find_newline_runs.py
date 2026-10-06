#!/usr/bin/env python3
"""Recursively scan a directory and report runs of consecutive newlines.

By default a run is reported when it has more than two newlines in a row,
which is the same as two or more blank lines together.

Usage:
    python find_newline_runs.py [directory] [--max-newlines N] [--whitespace] [--exclude NAME ...]

Output format (one hit per line):
    path:line_number: N newlines in a row

The line number is the first blank line of the run.
"""

import argparse
import os
import sys


def is_binary(path):
    """Treat a file as binary if its first chunk contains a NUL byte."""
    with open(path, "rb") as f:
        return b"\0" in f.read(8192)


def find_runs_in_file(path, max_newlines, whitespace_is_blank):
    """Yield (first_blank_line, newline_count) for each run longer than max_newlines."""
    run = 0          # newline characters in the current run
    first_blank = 0  # line number of the first blank line in the run

    # Text mode translates \r\n (and lone \r) to \n, so Windows files count the same.
    with open(path, encoding="utf-8", errors="replace") as f:
        for line_number, line in enumerate(f, start=1):
            ends_with_newline = line.endswith("\n")
            body = line[:-1] if ends_with_newline else line
            if whitespace_is_blank:
                body = body.strip()

            if body:
                # A non-blank line ends the run; its own newline starts the next one.
                if run > max_newlines:
                    yield first_blank, run
                run = int(ends_with_newline)
                first_blank = line_number + 1
            else:
                if run == 0:
                    first_blank = line_number
                run += int(ends_with_newline)

    if run > max_newlines:
        yield first_blank, run


def find_runs(root, max_newlines, whitespace_is_blank, exclude):
    """Yield (path, first_blank_line, newline_count) for every run under root."""
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune excluded directories in place so os.walk doesn't descend into them.
        dirnames[:] = sorted(d for d in dirnames if d not in exclude)

        for name in sorted(filenames):
            path = os.path.join(dirpath, name)
            if not os.path.isfile(path):  # skip sockets, FIFOs, broken symlinks
                continue
            try:
                if is_binary(path):
                    continue
                for first_blank, count in find_runs_in_file(path, max_newlines, whitespace_is_blank):
                    yield path, first_blank, count
            except OSError as err:
                print(f"warning: could not read {path}: {err}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Report runs of consecutive newlines in all files under a directory."
    )
    parser.add_argument("directory", nargs="?", default=".", help="directory to scan (default: current)")
    parser.add_argument(
        "-n", "--max-newlines", type=int, default=2,
        help="longest run of newlines to allow (default: 2)",
    )
    parser.add_argument(
        "-w", "--whitespace", action="store_true",
        help="treat lines containing only spaces or tabs as blank",
    )
    parser.add_argument(
        "-x", "--exclude", nargs="*", default=[], metavar="NAME",
        help="directory names to skip, e.g. .git node_modules",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        parser.error(f"not a directory: {args.directory}")

    count = 0
    for path, line_number, newlines in find_runs(
        args.directory, args.max_newlines, args.whitespace, set(args.exclude)
    ):
        print(f"{path}:{line_number}: {newlines} newlines in a row")
        count += 1

    print(f"{count} run(s) of more than {args.max_newlines} newlines", file=sys.stderr)


if __name__ == "__main__":
    main()
