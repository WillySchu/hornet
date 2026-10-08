"""The analyzer: runs the phases of semantic analysis in order, and holds what they share.

Declarations first, all of them (and with them the constants); then each function's body, against
those; then the typed tree, from what checking recorded. The checking itself is the other modules'
(see the package's docstring for which does what)."""

from typing import List, Optional

from parser import Program
from scopes import build_module_set
from semantic.context import Context
from semantic.declarations import DeclarationResolver, Declarations
from semantic.errors import SemanticError, SemanticErrors
from semantic.expressions import ExpressionChecker
from semantic.facts import Facts
from semantic.flow import Scopes
from semantic.statements import StatementChecker
from semantic.typed_tree_builder import TypedTreeBuilder
from semantic.type_resolution import TypeResolver
from symbols import SymbolTable
import typed_ast as typed

# Function bodies checked before giving up.
MAX_ERRORS = 20


class SemanticAnalyzer:
    """Type- and scope-checks a Program: runs the phases in order, and holds what they share."""

    def __init__(self):
        self.context = Context()  # where checking is (semantic/context.py), shared with what checks expressions
        self.decls = Declarations()  # what the program declares (semantic/declarations.py), read as `self.decls.X`

    def analyze(self, entry: Program, modules: Optional[dict] = None) -> "typed.Program":
        """Check the entry file and the modules it imports (discover_modules's result) and return the
        typed tree; no parser tree is changed."""
        self.symbols = SymbolTable()
        self.facts = Facts()
        # Each declaration under its key, with its file's scope (see scopes.py).
        program = build_module_set(entry, modules or {})
        self.module_set = program
        self.context.scope = program.files[0][1]
        # Declarations first, all of them: what each body is then checked against.
        self.decls = Declarations()
        self.context.scopes = Scopes(self.symbols, self.decls, self.facts)  # (no locals, until a function is checked)
        self.types = TypeResolver(self.decls, program, self.facts.array_sizes)
        self.expressions = ExpressionChecker(self.context, self.decls, self.facts, self.types, program, self.symbols)
        self.statements = StatementChecker(
            self.context, self.decls, self.facts, self.symbols, self.types, self.expressions, program)
        constants = self.expressions.constants
        DeclarationResolver(
            program, self.decls, self.types, self.facts, constants, self.expressions.check_array_size).resolve()

        # Then each body, collecting at most one error per function.
        errors: List[SemanticError] = []
        for fn in self.decls.all_functions:
            try:
                self.statements.analyze_function(fn)
            except SemanticError as e:
                if e.file is None:
                    e.file = fn.file
                errors.append(e)
                if len(errors) >= MAX_ERRORS:
                    break
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise SemanticErrors(errors)

        # The typed tree: what later stages consume.
        return TypedTreeBuilder(self.facts, self.decls, constants.values, self.symbols).program(
            self.decls.all_functions)
