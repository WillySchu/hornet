"""Codegen errors."""

from diagnostics import InternalCompilerError


class CodegenError(InternalCompilerError):
    """IR the backend can't lower."""
