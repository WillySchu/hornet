"""The errors semantic analysis reports."""

from typing import List, Optional

from diagnostics import CompileError
from parser import Node


class SemanticError(CompileError):
    """Semantic error; `node` gives the position."""
    def __init__(self, message: str, node: Optional[Node] = None):
        if node is None:
            super().__init__(message)
        else:
            super().__init__(message, node.file, node.line, node.col)


class SemanticErrors(SemanticError):
    """Several functions failed. Behaves like the first error; `errors` holds all."""
    def __init__(self, errors: List[SemanticError]):
        first = errors[0]
        Exception.__init__(self, str(first))
        self.message, self.file, self.line, self.col = first.message, first.file, first.line, first.col
        self._errors = errors

    @property
    def errors(self) -> List[CompileError]:
        return self._errors
