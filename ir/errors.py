"""IR errors."""

from diagnostics import InternalCompilerError


class IRError(InternalCompilerError):
    """AST shape the IR builder can't translate."""
