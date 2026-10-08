"""Constants: the program's `const` declarations, and the values of constant expressions.

A constant's value is worked out the first time it is needed, which may be while another's is being
worked out (or while a function that names it is being checked), so the order they are declared in
doesn't matter; one defined in terms of itself is an error.

The evaluator reads a checked expression from `Facts` alone: its type, and what each name in it
refers to, were recorded when it was checked. So it needs nothing of the analyzer's scopes. The one
thing it asks of the analyzer is to check a declaration, which is the analyzer's work and happens in
the scope of the file that declares it."""

from typing import Callable, Dict, Iterable, Tuple

from folding import fold_binary_op, fold_cast, fold_unary_op
from ops import ORDERING_OPS, BinaryOp, UnaryOp
from parser import (
    Binary, BoolLiteral, ByteLiteral, Call, Cast, ConstDecl, Constant, Field, Node, StringLiteral, Unary, Variable,
)
from scopes import display_name as shown
from semantic.errors import SemanticError
from semantic.facts import Facts
from typesys import INTEGER_TYPES, EnumInfo, Type, TypeKind


class ConstEvaluator:
    """The constants of a program, by key, and the evaluation of constant expressions."""

    def __init__(self, facts: Facts, enums: Dict[str, EnumInfo], check_declaration: Callable[[ConstDecl], Type]):
        """`check_declaration(decl)` type-checks a constant's declaration and returns its type; `facts`
        then describes its value's expression."""
        self.facts = facts
        self.enums = enums
        self._check_declaration = check_declaration
        self.decls: Dict[str, ConstDecl] = {}
        self.values: Dict[str, Tuple[Type, object]] = {}  # key -> (type, value), once worked out
        self._in_progress: set = set()

    def declare(self, decls: Iterable[ConstDecl]) -> None:
        for decl in decls:
            if decl.name in self.decls:
                raise SemanticError(f"Constant '{shown(decl.name)}' is already declared", decl)
            self.decls[decl.name] = decl

    def constant(self, key: str) -> Tuple[Type, object]:
        """(type, value) of the constant `key`, working it out (and what it depends on) the first time."""
        if key in self.values:
            return self.values[key]
        decl = self.decls[key]
        if key in self._in_progress:
            raise SemanticError(f"Constant '{shown(key)}' is defined in terms of itself", decl)
        self._in_progress.add(key)
        const_type = self._check_declaration(decl)
        value = self.evaluate(decl.value)
        self._in_progress.discard(key)
        self.values[key] = (const_type, value)
        return self.values[key]

    def type_of(self, key: str) -> Type:
        return self.constant(key)[0]

    def evaluate_all(self) -> None:
        for key in self.decls:
            self.constant(key)

    def evaluate(self, expr: Node):
        """Value of an already type-checked constant expression; an enum's is its member's index."""
        facts = self.facts
        t = facts.types.get(expr.nid)
        if expr.nid in facts.enum_members:  # `Enum.Member`
            return facts.enum_members[expr.nid][1]
        if isinstance(expr, Call) and facts.calls.get(expr.nid, (expr.name,))[0] in self.enums:  # `Enum(n)`
            value, members = self.evaluate(expr.args[0]), self.enums[t.enum_name].members
            if not 0 <= value < len(members):
                raise SemanticError(
                    f"{value} is not a member of {t} (its members' values are 0 to {len(members) - 1})", expr)
            return value
        if isinstance(expr, (Constant, ByteLiteral)) and isinstance(expr.value, int):
            return expr.value
        if isinstance(expr, (BoolLiteral, StringLiteral)):
            return expr.value
        if isinstance(expr, Call) and expr.nid in facts.enum_lens:
            return facts.enum_lens[expr.nid]
        if isinstance(expr, (Variable, Field)) and facts.const_refs.get(expr.nid) is not None:  # `NAME`, `alias.NAME`
            return self.constant(facts.const_refs[expr.nid])[1]
        if isinstance(expr, Unary) and expr.op in (UnaryOp.NEGATE, UnaryOp.COMPLEMENT):
            if isinstance(expr.operand, Constant) and expr.operand.value == 2 ** 63:
                return -2 ** 63
            return fold_unary_op(expr.op, self.evaluate(expr.operand), t)
        if isinstance(expr, Unary) and expr.op == UnaryOp.NOT:
            return not self.evaluate(expr.operand)
        if isinstance(expr, Binary):
            return self._binary(expr, t)
        if isinstance(expr, Cast) and t in INTEGER_TYPES:
            return fold_cast(t, self.evaluate(expr.expr), facts.types.get(expr.expr.nid))
        source_type = facts.types.get(expr.expr.nid) if isinstance(expr, Cast) else None
        if t == Type.STR and source_type is not None and source_type.kind == TypeKind.ENUM:  # `str(Enum.Member)`
            return self.enums[source_type.enum_name].members[self.evaluate(expr.expr)]
        raise SemanticError(
            "A constant's value must be built from literals, other constants, operators, and integer casts", expr)

    def _binary(self, expr: Binary, t: Type):
        left_type = self.facts.types.get(expr.left.nid)
        if expr.op in (BinaryOp.AND, BinaryOp.OR):
            left = self.evaluate(expr.left)
            right = self.evaluate(expr.right)
            return (left and right) if expr.op == BinaryOp.AND else (left or right)
        left, right = self.evaluate(expr.left), self.evaluate(expr.right)
        if left_type == Type.STR:
            if expr.op == BinaryOp.ADD:
                return left + right
            if expr.op in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL):
                return (left == right) == (expr.op == BinaryOp.EQUAL)
            if expr.op in ORDERING_OPS:  # each character is a byte, so this is their order
                return {BinaryOp.LESS_THAN: left < right, BinaryOp.GREATER_THAN: left > right,
                        BinaryOp.LESS_THAN_OR_EQUAL: left <= right}.get(expr.op, left >= right)
        elif (left_type == Type.BOOL or left_type.kind == TypeKind.ENUM) \
                and expr.op in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL):
            return (left == right) == (expr.op == BinaryOp.EQUAL)
        else:
            if (expr.op in (BinaryOp.DIVIDE, BinaryOp.MODULO) and right == -1
                    and left == {Type.INT: -(2 ** 63), Type.INT32: -(2 ** 31)}.get(left_type)):
                raise SemanticError("Integer overflow in division in a constant expression", expr)
            value = fold_binary_op(expr.op, left, right, t if t != Type.BOOL else left_type)
            if value is None:
                raise SemanticError("Division by zero in a constant expression", expr)
            return bool(value) if t == Type.BOOL else value
        raise SemanticError(
            "A constant's value must be built from literals, other constants, operators, and integer casts", expr)
