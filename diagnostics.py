"""Compiler diagnostics: error hierarchy and user-facing formatting."""

import os
from typing import List, Optional


class CompileError(Exception):
    """A problem in the user's program. str() keeps the legacy "msg at line L, column C" form."""

    def __init__(self, message: str, file: Optional[str] = None, line: int = 0, col: int = 0,
                 legacy: Optional[str] = None):
        self.message = message
        self.file = file
        self.line = line
        self.col = col
        if legacy is None:
            legacy = f"{message} at line {line}, column {col}" if (line or col) else message
        super().__init__(legacy)

    @property
    def errors(self) -> List['CompileError']:
        return [self]


class InternalCompilerError(Exception):
    """A compiler bug, never the user's fault."""


def _display_path(path: str) -> str:
    try:
        rel = os.path.relpath(path)
    except ValueError:
        return path
    return path if rel.startswith('..') else rel


def _source_line(path: Optional[str], line: int) -> Optional[str]:
    if not path or line <= 0:
        return None
    try:
        with open(path, encoding='latin-1') as f:
            for i, text in enumerate(f, 1):
                if i == line:
                    return text.rstrip('\n')
    except OSError:
        return None
    return None


def format_error(err: CompileError) -> str:
    """`file:line:col: error: message`, then the source line and a caret."""
    if not err.file and not err.line:
        return f"error: {err.message}"
    loc = _display_path(err.file) if err.file else '<input>'
    if err.line:
        loc += f":{err.line}"
        if err.col:
            loc += f":{err.col}"
    out = [f"{loc}: error: {err.message}"]
    text = _source_line(err.file, err.line)
    if text is not None:
        out.append(f"    {text.expandtabs(1)}")
        if err.col:
            out.append("    " + " " * (err.col - 1) + "^")
    return '\n'.join(out)


def format_errors(err: CompileError) -> str:
    """All errors carried by `err`, blank-line separated."""
    return '\n\n'.join(format_error(e) for e in err.errors)


def run_cli(action, show_traceback: bool = False):
    """Run `action`, reporting errors to stderr. Exit 1 for user errors, 2 for compiler bugs."""
    import sys
    import traceback
    try:
        return action()
    except CompileError as e:
        if show_traceback:
            raise
        print(format_errors(e), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        if show_traceback:
            raise
        print(f"internal compiler error: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(2)
