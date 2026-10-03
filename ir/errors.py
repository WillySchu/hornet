"""IR errors."""

from diagnostics import InternalCompilerError


class IRError(InternalCompilerError):
    """Something the IR builder can't translate: a compiler bug."""
