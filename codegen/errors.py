"""Errors for the codegen package."""


class CodegenError(Exception):
    """Raised when the code generator encounters an IR object it doesn't know
    how to translate."""
