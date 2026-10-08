"""Semantic analysis: name resolution, type checking, and control-flow checks; its result is the
typed tree (typed_ast.py), which every later stage consumes.

Strict typing: no implicit conversions; integer operands must match exactly (an integer literal
takes the other operand's type). Blocks scope lexically and may shadow. Non-void functions must
return on all paths.

The parser's tree is never changed: checking records what it learns in Facts, keyed by node number,
and TypedTreeBuilder builds each function's typed tree from those facts at the end of analyze().

The entry points are here (analyze). The work is in the modules beside this one:

    analyzer.py         runs the phases in order, and holds what they share
    declarations.py     what the program declares, resolved before any body is checked
    type_resolution.py  a written type to a Type
    constants.py        the values of constants and constant expressions
    statements.py       a function's body, statement by statement
    expressions.py      the type of each expression (calls.py: calls and struct literals)
    flow.py             the names in scope and what is known of them at each point
    context.py          where checking is, shared by what checks
    facts.py            what checking learned
    typed_tree_builder.py   the typed tree, from the facts
"""

import argparse
from typing import Optional

from lexer import lex
from parser import Parser, Program
from semantic.analyzer import SemanticAnalyzer
from semantic.errors import SemanticError, SemanticErrors
from semantic.type_resolution import type_from_name
import typed_ast as typed
from typesys import StructInfo, SumTypeInfo, Type, TypeKind

# What is used from here: the entry points below, and names that were this module's own when semantic
# analysis was one file, which other code still imports from it.
__all__ = [
    'analyze',
    'analyze_source',
    'SemanticAnalyzer',
    'SemanticError',
    'SemanticErrors',
    'StructInfo',
    'SumTypeInfo',
    'Type',
    'TypeKind',
    'type_from_name',
]


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
    """The command line: `python -m semantic FILE` (see __main__.py)."""
    arg_parser = argparse.ArgumentParser(description='Semantic analyzer')
    arg_parser.add_argument('file', type=str, help='File to check.')
    args = arg_parser.parse_args()
    analyze_source(args.file)
    print("OK: no semantic errors found")
