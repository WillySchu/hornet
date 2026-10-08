"""Semantic analysis: name resolution, type checking, and control-flow checks; its result is the
typed tree (typed_ast.py), which every later stage consumes.

Strict typing: no implicit conversions; integer operands must match exactly (an integer literal
takes the other operand's type). Blocks scope lexically and may shadow. Non-void functions must
return on all paths.

The parser's tree is never changed: checking records what it learns in Facts, keyed by node number,
and TypedTreeBuilder builds each function's typed tree from those facts at the end of analyze().

SemanticAnalyzer runs the phases in order and holds what they share. The work is in the modules
beside this one:

    declarations.py   what the program declares, resolved before any body is checked
    types.py          a written type to a Type
    constants.py      the values of constants and constant expressions
    statements.py     a function's body, statement by statement
    expressions.py    the type of each expression (calls.py: calls and struct literals)
    flow.py           the names in scope and what is known of them at each point
    context.py        where checking is, shared by what checks
    facts.py          what checking learned
    typed_tree_builder.py   the typed tree, from the facts
"""

import argparse
from typing import List, Optional

from lexer import lex
from parser import Parser, Program
from scopes import build_module_set
from semantic.context import Context
from semantic.declarations import DeclarationResolver, Declarations
from semantic.errors import SemanticError, SemanticErrors
from semantic.expressions import ExpressionChecker
from semantic.facts import Facts
from semantic.flow import Scopes
from semantic.statements import StatementChecker
from semantic.typed_tree_builder import TypedTreeBuilder
from semantic.types import TypeResolver, type_from_name  # noqa: F401 (type_from_name: for those who import it here)
from symbols import SymbolTable
import typed_ast as typed
from typesys import StructInfo, SumTypeInfo, Type, TypeKind  # noqa: F401 (for those who import them here)

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


def analyze(program: Program, modules: Optional[dict] = None) -> "typed.Program":
    """Check `program` (an entry file) and the modules it imports (discover_modules's result), and
    return the typed tree: everything later stages need."""
    return SemanticAnalyzer().analyze(program, modules)


def analyze_source(filename: str) -> Program:
    """Lex, parse, and analyze a file."""
    tokens = lex(filename)
    program = Parser(tokens).parse_program()
    analyze(program)
    return program


def main():
    arg_parser = argparse.ArgumentParser(description='Semantic analyzer')
    arg_parser.add_argument('file', type=str, help='File to check.')
    args = arg_parser.parse_args()
    analyze_source(args.file)
    print("OK: no semantic errors found")


if __name__ == '__main__':
    main()
