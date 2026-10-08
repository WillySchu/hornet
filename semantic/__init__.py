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
import contextlib
import dataclasses
from typing import List, Optional

from lexer import lex
from parser import Call, ConstDecl, Field, Node, Parser, Program, Variable
from scopes import build_module_set, display_name as shown
from semantic.constants import ConstEvaluator
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
from typesys import INTEGER_TYPES, StructInfo, SumTypeInfo, Type, TypeKind  # noqa: F401 (some for importers)

# Function bodies checked before giving up.
MAX_ERRORS = 20

_INTEGER_TYPES = INTEGER_TYPES

_MAIN_PARAMS = ([], [Type.INT, Type(TypeKind.POINTER, element_type=Type.UINT8)])

# Types the C runtime can read as main's int exit status.
_MAIN_RETURN_TYPES = (Type.INT, Type.INT32, Type.INT8, Type.UINT8, Type.BOOL)


def _check_main_signature(fn, param_types: list, return_type: Type) -> None:
    """`main` is called by the C runtime: it returns the exit status (normally `int`) and takes no
    parameters or `(int argc, *byte argv)`."""
    if return_type not in _MAIN_RETURN_TYPES:
        raise SemanticError(
            f"'main' must return int (the program's exit status), not "
            f"{'nothing' if return_type == Type.VOID else return_type} -- declare it 'def int main()'", fn)
    if param_types not in _MAIN_PARAMS:
        raise SemanticError(
            "'main' takes no parameters, or exactly '(int argc, *byte argv)'", fn.params[0] if fn.params else fn)


class SemanticAnalyzer:
    """Type- and scope-checks a Program."""

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
        self.constants = ConstEvaluator(self.facts, self.decls.enums, self._check_const_declaration)
        self.expressions = ExpressionChecker(
            self.context, self.decls, self.facts, self.constants, self.types, program, self.symbols)
        self.statements = StatementChecker(
            self.context, self.decls, self.facts, self.symbols, self.types, self.expressions, program)
        DeclarationResolver(
            program, self.decls, self.types, self.facts, self.constants, self._array_size_value).resolve()

        # Then each body, collecting at most one error per function.
        errors: List[SemanticError] = []
        for fn in self.decls.all_functions:
            try:
                self.statements.analyze_function(fn)
                if fn.name == 'main':
                    _check_main_signature(fn, *self.decls.functions['main'])
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
        return TypedTreeBuilder(self.facts, self.decls, self.constants.values, self.symbols).program(
            self.decls.all_functions)

    # -- constant expressions, for the declaration phase

    def _enter(self, decl: Node) -> None:
        """Check `decl` (a top-level declaration, under its key) in its own file's scope."""
        self.context.scope = self.module_set.scope_of[decl.nid]

    def _type(self, type_expr, node: Node, sums: bool = True) -> Type:
        return self.types.resolve(type_expr, node, self.context.scope, sums)

    def _array_size_value(self, expr: Node, scope) -> int:
        """The value of an array-size expression written in the file whose scope is `scope`: a positive
        integer computed from literals and constants only. (For DeclarationResolver.)"""
        saved_scope, self.context.scope = self.context.scope, scope
        try:
            return self._array_size_value_here(expr)
        finally:
            self.context.scope = saved_scope

    def _array_size_value_here(self, expr: Node) -> int:
        stack = [expr]
        while stack:
            node = stack.pop()
            if node.nid in self.module_set.qualified and isinstance(node, Field):
                continue  # `alias.NAME`: checked as a constant below
            if isinstance(node, Call) and node.name == 'len' and len(node.args) == 1 and node.receiver is None \
                    and self.expressions.enum_named_by(node.args[0]) is not None:
                continue  # `len(Enum)`: a constant
            if isinstance(node, Call):
                raise SemanticError("Array size must be a constant expression, not a call", node)
            if isinstance(node, Variable) and self.expressions.const_key(node.name) is None:
                raise SemanticError(
                    f"Array size must be a constant expression, but '{node.name}' isn't a constant", node)
            if dataclasses.is_dataclass(node):
                stack.extend(v for f in dataclasses.fields(node) if isinstance(v := getattr(node, f.name), Node))
        with self._no_locals():
            size_type = self.expressions.check_expr(expr)
            if size_type not in _INTEGER_TYPES:
                raise SemanticError(f"Array size must be an integer, got {size_type}", expr)
            value = self.constants.evaluate(expr)
        if value <= 0:
            raise SemanticError(f"Array size must be positive, got {value}", expr)
        return value

    def _check_const_declaration(self, cd: ConstDecl) -> Type:
        """Check a constant's declaration, in the scope of the file that declares it and with no
        local in sight; its type. (For ConstEvaluator, which then works out the value.)"""
        saved_scope = self.context.scope
        self._enter(cd)
        try:
            with self._no_locals():
                const_type = self._type(cd.const_type, cd)
                if const_type not in _INTEGER_TYPES and const_type not in (Type.BOOL, Type.STR) \
                        and const_type.kind != TypeKind.ENUM:
                    raise SemanticError(
                        f"Constant '{shown(cd.name)}' has type {const_type} -- constants must be an integer type, "
                        f"bool, str, or an enum", cd)
                value_type = self.expressions.check_value_flowing_into(cd.value, const_type)
                if not self.expressions.types_compatible(value_type, const_type):
                    raise SemanticError(
                        f"Constant '{shown(cd.name)}' is declared {const_type} but its value has type {value_type}",
                        cd)
        finally:
            self.context.scope = saved_scope
        return const_type

    @contextlib.contextmanager
    def _no_locals(self):
        """While a constant expression is checked: no local is in scope, wherever checking was."""
        saved, self.context.scopes = self.context.scopes, Scopes(self.symbols, self.decls, self.facts)
        try:
            yield
        finally:
            self.context.scopes = saved


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
