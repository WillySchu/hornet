"""Errors for the IR package."""


class IRError(Exception):
    """Raised when the IR generator encounters an AST node it doesn't know how
    to translate."""
