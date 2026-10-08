"""What the parser reports."""

from diagnostics import CompileError


class ParseError(CompileError):
    """Malformed input."""
