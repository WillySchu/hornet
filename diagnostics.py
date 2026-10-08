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


def path_text(path) -> str:
    """A file's name as diagnostics and generated code hold text: a byte to a character, as source is read."""
    return os.fsencode(str(path)).decode('latin-1')


def quoted_text(text: str) -> str:
    """`text` in quotes, for a message. Source text and file names are held a byte to a character
    and written back as those bytes, so the bytes of a character that isn't ASCII are left as they
    are (repr() would escape some of them, and so break the character up); only control characters,
    which can't be shown, are escaped, and the backslash, so that it isn't read as an escape."""
    return "'" + ''.join(c if c >= ' ' and c not in '\\\x7f' else repr(c)[1:-1] for c in text) + "'"


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
    loc = path_text(_display_path(err.file)) if err.file else '<input>'
    if err.line:
        loc += f":{err.line}"
        if err.col:
            loc += f":{err.col}"
    out = [f"{loc}: error: {err.message}"]
    text = _source_line(err.file, err.line)
    if text is not None:
        out.append(f"    {text.expandtabs(1)}")
        if err.col:
            out.append("    " + " " * _columns_shown(text[:err.col - 1]) + "^")
    return '\n'.join(out)


def _columns_shown(text: str) -> int:
    """How many columns `text` takes up when shown. A source line is read a byte to a character and
    written back as those bytes (write_diagnostic), and in UTF-8 a character's bytes after its first
    (0x80 to 0xBF) take no column of their own."""
    return sum(1 for c in text if not '\x80' <= c <= '\xbf')


def write_diagnostic(text: str, stream) -> None:
    """Write a diagnostic and a newline to `stream` as the bytes it stands for: source text in it
    was read a byte to a character, so latin-1 gives the terminal back what the file held."""
    data = (text + '\n').encode('latin-1', errors='replace')
    if hasattr(stream, 'buffer'):
        stream.flush()
        stream.buffer.write(data)
        stream.buffer.flush()
    else:
        stream.write(data.decode('latin-1'))


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
        write_diagnostic(format_errors(e), sys.stderr)
        sys.exit(1)
    except Exception as e:
        if show_traceback:
            raise
        print(f"internal compiler error: {type(e).__name__}: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(2)
