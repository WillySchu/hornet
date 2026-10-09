"""Expressions: the type of each, and whether it is one the language allows.

ExpressionChecker checks an expression where checking is (Context), against what the program
declares, and records what it learns in Facts, from which the typed tree is built. Calls and struct
literals are CallChecker's (semantic/calls.py), which is a part of it, and the values of constants
are ConstEvaluator's (semantic/constants.py), which it also builds: the two take turns, since a
constant's value is an expression and an expression may name a constant. Statements are checked
elsewhere and only call in: nothing here calls back out."""

import contextlib
import dataclasses
from typing import Callable, Optional, Tuple

from diagnostics import quoted_text
from ops import EQUALITY_OPS, LOGICAL_OPS, ORDERING_OPS, BinaryOp, UnaryOp
from parser import (
    READ_AS_A_TYPED_LITERAL, ArrayLiteral, Binary, BoolLiteral, ByteLiteral, Call, Cast, ConstDecl, Constant,
    DictLiteral, Field, Index, IsCheck, Node, NoneLiteral, QualifiedTypeExpr, Slice, SliceLiteral, StringLiteral,
    Unary, VarDecl, Variable,
)
from scopes import display_name as shown
from semantic.calls import CallChecker
from semantic.constants import ConstEvaluator
from semantic.context import Context
from semantic.errors import SemanticError
from semantic.flow import Scopes
from typesys import BYTE_SLICE, INTEGER_TYPES, Type, TypeKind

# ADD is excluded: it also concatenates strings.
_INT_ONLY_BINARY_OPS = {
    BinaryOp.SUBTRACT, BinaryOp.MULTIPLY, BinaryOp.DIVIDE, BinaryOp.MODULO,
    BinaryOp.BITWISE_AND, BinaryOp.BITWISE_OR, BinaryOp.BITWISE_XOR,
    BinaryOp.SHIFT_LEFT, BinaryOp.SHIFT_RIGHT,
}


# Literal ranges for narrow integer types.
_NARROW_INT_RANGES = {
    Type.INT8: (-128, 127),
    Type.UINT8: (0, 255),
    Type.INT32: (-2**31, 2**31 - 1),
}


def _shown_key(key: tuple) -> str:
    """A constant dict key (_constant_key_value's) as it is written."""
    kind, value = key
    if kind in ('str', 'enum'):
        return quoted_text(value)
    return {True: 'true', False: 'false'}[value] if kind == 'bool' else str(value)


def _constant_key_value(expr: Node):
    """(kind, value) of a constant dict key, so 5 and true differ."""
    if isinstance(expr, Constant):
        return ('int', expr.value)
    if isinstance(expr, StringLiteral):
        return ('str', expr.value)
    if isinstance(expr, BoolLiteral):
        return ('bool', expr.value)
    if isinstance(expr, ByteLiteral):
        return ('byte', expr.value)
    return None


def _written(expr: Node) -> Optional[str]:
    """How `expr` is written, if it is a variable or a chain of fields, constant or variable indexes,
    and dereferences from one; None for anything else."""
    if isinstance(expr, Variable):
        return expr.name
    if isinstance(expr, Unary) and expr.op == UnaryOp.DEREFERENCE:
        inner = _written(expr.operand)
        return None if inner is None else f"*{inner}"
    if isinstance(expr, (Field, Index)):
        base = _written(expr.base if isinstance(expr, Field) else expr.array)
        if base is None:
            return None
        if base.startswith('*'):  # `(*p).x`: the dereference is of p alone
            base = f"({base})"
        if isinstance(expr, Field):
            return f"{base}.{expr.name}"
        index = str(expr.index.value) if isinstance(expr.index, Constant) else _written(expr.index)
        return None if index is None else f"{base}[{index}]"
    return None


class ExpressionChecker:
    """Checks expressions. (It is the Checker that CallChecker asks about a call's arguments.)"""

    def __init__(self, context: Context, decls, facts, types, module_set, symbols):
        self.context = context
        self.decls = decls
        self.facts = facts
        self.types = types
        self.module_set = module_set
        self.symbols = symbols
        self.constants = ConstEvaluator(facts, decls.enums, self.check_const_declaration)
        self.calls = CallChecker(self, decls, facts, self.constants, module_set)

    @property
    def scope(self):
        """The scope of the file being checked."""
        return self.context.scope

    def _type(self, type_expr, node: Node, sums: bool = True) -> Type:
        return self.types.resolve(type_expr, node, self.context.scope, sums)

    # -- constant expressions, for the declaration phase: checked in the scope of the file they are
    # written in, wherever checking was, and with no local in sight

    @contextlib.contextmanager
    def _no_locals(self):
        saved, self.context.scopes = self.context.scopes, Scopes(self.symbols, self.decls, self.facts)
        try:
            yield
        finally:
            self.context.scopes = saved

    def check_array_size(self, expr: Node, scope) -> int:
        """The value of an array-size expression written in the file whose scope is `scope`: a positive
        integer computed from literals and constants only. (For DeclarationResolver.)"""
        saved_scope, self.context.scope = self.context.scope, scope
        try:
            return self._array_size_here(expr)
        finally:
            self.context.scope = saved_scope

    def _array_size_here(self, expr: Node) -> int:
        stack = [expr]
        while stack:
            node = stack.pop()
            if node.nid in self.module_set.qualified and isinstance(node, Field):
                continue  # `alias.NAME`: checked as a constant below
            if isinstance(node, Call) and node.name == 'len' and len(node.args) == 1 and node.receiver is None \
                    and self.enum_named_by(node.args[0]) is not None:
                continue  # `len(Enum)`: a constant
            if isinstance(node, Call):
                raise SemanticError("Array size must be a constant expression, not a call", node)
            if isinstance(node, Variable) and self.const_key(node.name) is None:
                raise SemanticError(
                    f"Array size must be a constant expression, but '{node.name}' isn't a constant", node)
            if dataclasses.is_dataclass(node):
                stack.extend(v for f in dataclasses.fields(node) if isinstance(v := getattr(node, f.name), Node))
        with self._no_locals():
            size_type = self.check_expr(expr)
            if size_type not in INTEGER_TYPES:
                raise SemanticError(f"Array size must be an integer, got {size_type}", expr)
            value = self.constants.evaluate(expr)
        if value <= 0:
            raise SemanticError(f"Array size must be positive, got {value}", expr)
        return value

    def check_const_declaration(self, cd: ConstDecl) -> Type:
        """Check a constant's declaration, in the scope of the file that declares it; its type. (For
        ConstEvaluator, which then works out the value.)"""
        saved_scope, self.context.scope = self.context.scope, self.module_set.scope_of[cd.nid]
        try:
            with self._no_locals():
                const_type = self._type(cd.const_type, cd)
                if const_type not in INTEGER_TYPES and const_type not in (Type.BOOL, Type.STR) \
                        and const_type.kind != TypeKind.ENUM:
                    raise SemanticError(
                        f"Constant '{shown(cd.name)}' has type {const_type} -- constants must be an integer type, "
                        f"bool, str, or an enum", cd)
                value_type = self.check_value_flowing_into(cd.value, const_type)
                if not self.types_compatible(value_type, const_type):
                    raise SemanticError(
                        f"Constant '{shown(cd.name)}' is declared {const_type} but its value has type {value_type}",
                        cd)
        finally:
            self.context.scope = saved_scope
        return const_type

    def check_expr(self, expr: Node) -> Type:
        """Type-check `expr`, recording its type."""
        if isinstance(expr, Constant):
            result = self.check_constant(expr)
        elif isinstance(expr, BoolLiteral):
            result = Type.BOOL
        elif isinstance(expr, NoneLiteral):
            result = Type.NONE
        elif isinstance(expr, StringLiteral):
            result = Type.STR
        elif isinstance(expr, ByteLiteral):
            # Range already validated by the parser.
            result = Type.UINT8
        elif isinstance(expr, Variable):
            result = self.check_variable(expr)
        elif isinstance(expr, ArrayLiteral):
            result = self.check_array_literal(expr)
        elif isinstance(expr, SliceLiteral):
            result = self.check_slice_literal(expr)
        elif isinstance(expr, DictLiteral):
            result = self.check_dict_literal(expr)
        elif isinstance(expr, Index):
            result = self.check_index(expr)
        elif isinstance(expr, Field):
            result = self.check_field(expr)
        elif isinstance(expr, Slice):
            result = self.check_slice(expr)
        elif isinstance(expr, Call):
            result = self.calls.check_call(expr)
        elif isinstance(expr, Unary):
            result = self.check_unary(expr)
        elif isinstance(expr, Cast):
            result = self.check_cast(expr)
        elif isinstance(expr, Binary):
            result = self.check_binary(expr)
        elif isinstance(expr, IsCheck):
            result = self.check_is_check(expr)
        else:
            raise SemanticError(f"No semantic rule for expression: {expr!r}", expr)
        self.facts.types[expr.nid] = result
        return result

    def check_constant(self, expr: Constant) -> Type:
        if expr.value > 2**63 - 1:
            raise SemanticError(f"Integer literal {int(expr.value)} is out of range for int (64-bit)", expr)
        return Type.INT

    def check_variable(self, expr: Variable) -> Type:
        t, self.facts.decls[expr.nid] = self.resolve(expr.name, expr)
        if self.facts.decls[expr.nid] is None:
            self.facts.const_refs[expr.nid] = self.const_key(expr.name)
        return t

    def check_is_check(self, expr: IsCheck) -> Type:
        """`NAME is T`, `EXPR is T`, or `EXPR is T as NAME` (NAME already declared, by the statement):
        T is a variant of the subject's sum, or a member of its enum. Only a NAME can be narrowed."""
        if expr.binds and expr.nid not in self.context.bound:
            if expr.nid not in self.context.bindable:
                raise SemanticError(
                    "'as NAME' can bind only in an `if` or `elif` condition, where the check is the whole "
                    "condition or one of the checks joined by 'and'", expr)
            self.declare_is_binding(expr)
        if expr.variable_name is None:  # `EXPR is T`: a test of the expression's value
            variable_type, narrowed = self.check_expr(expr.subject), False
        else:
            variable_type, self.facts.decls[expr.nid] = self.resolve(expr.variable_name, expr)
            narrowed = self.context.scopes.is_narrowed(self.facts.decls[expr.nid])
        if variable_type.kind == TypeKind.ENUM:  # `NAME is Member`: an equality test; nothing is narrowed
            members = self.decls.enums[variable_type.enum_name].members
            member = expr.type_name
            if isinstance(member, QualifiedTypeExpr) \
                    and self.context.scope.resolve(member.module) == variable_type.enum_name:
                member = member.name  # `NAME is Enum.Member`
            if isinstance(member, str) and member in members:
                self.facts.enum_checks[expr.nid] = (members.index(member), variable_type)
                return Type.BOOL
            if not narrowed:  # (a sum's variable narrowed to an enum may be tested as the sum again, below)
                written = f"{member.module}.{member.name}" if isinstance(member, QualifiedTypeExpr) else member
                raise SemanticError(
                    f"'{written}' is not a member of {variable_type} (its members: {', '.join(members)})"
                    if isinstance(written, str) else
                    f"'is' on an enum takes one of its members ({', '.join(members)})", expr)
        if narrowed:  # tested as the sum it is declared as
            variable_type = self.symbols[self.facts.decls[expr.nid]].type
        if variable_type.kind != TypeKind.SUM:
            if expr.variable_name is None:
                raise SemanticError(
                    f"'is' tests a sum type's variant or an enum's member, but this value is {variable_type}",
                    expr.subject)
            if expr.subject is not None:
                raise SemanticError(
                    f"The expression bound to '{expr.variable_name}' "
                    f"(type {variable_type}) is not a sum type -- 'is' "
                    f"only narrows a sum-typed value to one of its own "
                    f"declared variants",
                    expr.subject,
                )
            raise SemanticError(
                f"'{expr.variable_name}' (declared {variable_type}) is "
                f"not a sum type -- 'is' only narrows a sum-typed "
                f"variable to one of its own declared variants",
                expr,
            )
        narrowed_type = self._type(expr.type_name, expr)  # (a variant may be a pointer to a sum, or a slice of one)
        sum_type_info = self.decls.sum_types[variable_type.sum_type_name]
        if narrowed_type.kind == TypeKind.SUM:
            # `x is S`, S a sum: whether x holds any of S's variants, all of which must be x's own.
            foreign = [v for v in self.decls.sum_types[narrowed_type.sum_type_name].variants
                       if v not in sum_type_info.variants]
            if foreign:
                raise SemanticError(
                    f"'{expr.type_name}' can hold {foreign[0]}, which {variable_type} has no variant for -- 'is' "
                    f"tests for one of {variable_type}'s variants "
                    f"({', '.join(str(v) for v in sum_type_info.variants)}), or for a sum type made only of them",
                    expr)
        elif narrowed_type not in sum_type_info.variants:
            raise SemanticError(
                f"'{expr.type_name}' is not one of {variable_type}'s own "
                f"declared variants ({', '.join(str(v) for v in sum_type_info.variants)})",
                expr,
            )
        self.facts.narrowed[expr.nid] = narrowed_type
        return Type.BOOL

    def check_unary(self, expr: Unary) -> Type:
        if (expr.op == UnaryOp.NEGATE and isinstance(expr.operand, Constant)
                and expr.operand.value == 2**63):
            self.facts.types[expr.operand.nid] = Type.INT  # -2**63 is int's minimum
            return Type.INT
        operand_type = self.check_expr_allowing_struct_literal(expr.operand)
        if expr.op in (UnaryOp.NEGATE, UnaryOp.COMPLEMENT):
            if operand_type not in INTEGER_TYPES:
                raise SemanticError(
                    f"'{expr.op.symbol()}' requires an int, int8, "
                    f"uint8, or int32 operand, got {operand_type}",
                    expr,
                )
            return operand_type
        if expr.op == UnaryOp.NOT:
            if operand_type != Type.BOOL:
                raise SemanticError(
                    f"'not' requires a bool operand, got {operand_type} "
                    f"(no implicit int-to-bool conversion -- try "
                    f"`x == 0` instead of `not x`)",
                    expr,
                )
            return Type.BOOL
        if expr.op == UnaryOp.ADDRESS_OF:
            if (isinstance(expr.operand, Variable) and self.facts.decls.get(expr.operand.nid) is None
                    and self.const_key(expr.operand.name) is not None):
                raise SemanticError(f"Cannot take the address of constant '{expr.operand.name}'", expr)
            if expr.operand.nid in self.facts.const_refs:
                raise SemanticError(
                    f"Cannot take the address of constant '{shown(self.facts.const_refs[expr.operand.nid])}'", expr)
            if expr.operand.nid in self.facts.enum_members:  # `&Color.Red`: a value, with nowhere it lives
                raise SemanticError(
                    f"Cannot take the address of the enum member '{operand_type}.{expr.operand.name}'", expr)
            # Only variables, struct literals, and chains rooted in a variable.
            is_struct_literal = self.calls.struct_literal(expr.operand) is not None
            root_variable = self.root_variable_of(expr.operand) if isinstance(expr.operand, (Field, Index)) else None
            is_rooted_field_or_index = isinstance(expr.operand, (Field, Index)) and root_variable is not None
            if not (isinstance(expr.operand, Variable) or is_struct_literal or is_rooted_field_or_index):
                raise SemanticError(
                    f"'&' can only take the address of a bare variable, "
                    f"a struct literal, or a field/element access rooted "
                    f"in a named variable for now, not "
                    f"{type(expr.operand).__name__} -- a function call's "
                    f"own field/element isn't yet supported",
                    expr,
                )
            return Type(TypeKind.POINTER, element_type=operand_type)
        if expr.op == UnaryOp.DEREFERENCE:
            if operand_type.kind != TypeKind.POINTER:
                raise SemanticError(
                    f"'*' requires a pointer operand, got {operand_type}",
                    expr,
                )
            return operand_type.element_type
        raise SemanticError(f"No semantic rule for unary operator: {expr.op}", expr)

    def check_cast(self, expr: Cast) -> Type:
        """`T(expr)` between integer types (literals range-checked against T) and from an enum; `str(...)`
        of a byte, a []byte, or an enum."""
        target_type = self._type(expr.target_type, expr)
        if target_type == Type.STR:
            source_type = self.check_expr(expr.expr)
            if source_type not in (Type.UINT8, BYTE_SLICE) and source_type.kind != TypeKind.ENUM:
                raise SemanticError(f"str(...) takes a byte, a []byte, or an enum (its member's name), "
                                    f"got {source_type}", expr)
            return Type.STR
        if target_type == Type.INT64 and self.as_folded_int_literal(expr.expr) is not None:
            self._record_literal_type(expr.expr, Type.INT64)
            source_type = Type.INT64
        else:
            source_type = self.check_expr(expr.expr)
        if target_type not in INTEGER_TYPES or (source_type not in INTEGER_TYPES
                                                 and source_type.kind != TypeKind.ENUM):
            raise SemanticError(
                f"Cannot cast {source_type} to {target_type} -- casting "
                f"is only supported between int, int8, uint8, and int32, "
                f"and from an enum to one of them, right now",
                expr,
            )
        return target_type

    def check_binary(self, expr: Binary) -> Type:
        if expr.op == BinaryOp.IN and self.enum_named_by(expr.right) is not None:
            # `n in Enum`: whether the integer `n` is a member's value (so `Enum(n)` wouldn't panic).
            enum = self.enum_named_by(expr.right)
            left_type = self.check_expr(expr.left)
            if left_type not in INTEGER_TYPES:
                raise SemanticError(
                    f"'in' with the enum {shown(enum)} on its right tests an integer, got {left_type}", expr.left)
            self.facts.enum_ins[expr.nid] = enum
            return Type.BOOL
        with_literal = self._operand_types_with_an_untyped_array(expr)
        if with_literal is not None:
            left_type, right_type = with_literal
        else:
            left_type = self.check_expr_allowing_struct_literal(expr.left)
            mark = self.context.scopes.mark()
            if expr.op in LOGICAL_OPS:  # the right side runs only when the left was true (`and`) or false (`or`)
                self.context.scopes.apply(self.context.scopes.when(expr.left)[0 if expr.op == BinaryOp.AND else 1])
            right_type = self.check_expr_allowing_struct_literal(expr.right)
            self.context.scopes.end_region(mark)
        op = expr.op
        if op not in LOGICAL_OPS and op != BinaryOp.IN:
            left_type, right_type = self._literal_operand_types(expr.left, left_type, expr.right, right_type)

        if op == BinaryOp.ADD:
            # Same-type integers add; str + str concatenates.
            if left_type in INTEGER_TYPES and left_type == right_type:
                return left_type
            if left_type == Type.STR and right_type == Type.STR:
                return Type.STR
            raise SemanticError(
                f"'+' requires two operands of the same integer type "
                f"(int, int8, uint8, or int32) or two str operands, "
                f"got {left_type} and {right_type}",
                expr,
            )

        if op in _INT_ONLY_BINARY_OPS:
            return self._require_same_integer_type(left_type, right_type, op, expr)

        if op in ORDERING_OPS:
            # Same-type integers; or two strs, ordered byte by byte (as strings.ht's compare orders them).
            if (left_type in INTEGER_TYPES or left_type == Type.STR) and left_type == right_type:
                return Type.BOOL
            raise SemanticError(
                f"'{op.symbol()}' requires two operands of the same integer type "
                f"(int, int8, uint8, or int32) or two str operands, "
                f"got {left_type} and {right_type}",
                expr,
            )

        if op in EQUALITY_OPS:
            # Pointers compare to `none`.
            none_vs_nilable = (
                    (left_type == Type.NONE and right_type.kind == TypeKind.POINTER) or
                    (right_type == Type.NONE and left_type.kind == TypeKind.POINTER)
            )
            if none_vs_nilable:
                return Type.BOOL
            # A sum with a `none` variant: `x == none` is shorthand for `x is none`.
            for side, other in ((left_type, right_type), (right_type, left_type)):
                if other == Type.NONE and side.kind == TypeKind.SUM:
                    if Type.NONE not in self.decls.sum_types[side.sum_type_name].variants:
                        raise SemanticError(
                            f"{side} has no `none` variant, so it is never none", expr)
                    return Type.BOOL
            if Type.NONE in (left_type, right_type) and TypeKind.DICT in (left_type.kind, right_type.kind):
                raise SemanticError("A dict is never none -- check 'len(d) == 0' for an empty one", expr)
            if Type.NONE in (left_type, right_type) and TypeKind.SLICE in (left_type.kind, right_type.kind):
                raise SemanticError("A slice is never none -- check 'len(s) == 0' for an empty one", expr)

            if left_type.kind == TypeKind.ARRAY and right_type.kind == TypeKind.ARRAY:
                if left_type != right_type:
                    raise SemanticError(
                        f"Cannot compare {left_type} to {right_type} with "
                        f"'{op.symbol()}' -- arrays must have the same "
                        f"length and element type",
                        expr,
                    )
                if not self._is_comparable_type(left_type):
                    raise SemanticError(
                        f"'{op.symbol()}' does not support {left_type} "
                        f"operands -- array equality isn't defined yet "
                        f"when the elements are (or contain) a slice "
                        f"or a dict, neither of which has '==' defined yet",
                        expr,
                    )
                return Type.BOOL

            if left_type.kind == TypeKind.STRUCT and right_type.kind == TypeKind.STRUCT:
                if left_type != right_type:
                    raise SemanticError(
                        f"Cannot compare {left_type} to {right_type} with "
                        f"'{op.symbol()}' -- structs must be the exact "
                        f"same type",
                        expr,
                    )
                if not self._is_comparable_type(left_type):
                    raise SemanticError(
                        f"'{op.symbol()}' does not support {left_type} "
                        f"operands -- struct equality isn't defined yet "
                        f"when a field (directly, or nested inside "
                        f"another struct, an array field, or a sum's variant) is a slice "
                        f"or a dict, neither of which has '==' defined yet",
                        expr,
                    )
                return Type.BOOL

            if TypeKind.SUM in (left_type.kind, right_type.kind):
                return self._check_sum_equality(expr, left_type, right_type)

            # Slice and dict equality is undefined.
            if (
                    left_type.kind in (TypeKind.SLICE, TypeKind.VOID, TypeKind.NONE, TypeKind.DICT)
                    or right_type.kind in (TypeKind.SLICE, TypeKind.VOID, TypeKind.NONE, TypeKind.DICT)
            ):
                raise SemanticError(
                    f"'{op.symbol()}' does not support slice, void, "
                    f"dict, or none operands, except comparing a "
                    f"pointer, or a sum type with a `none` variant, to none",
                    expr,
                )
            if left_type != right_type:
                raise SemanticError(
                    f"Cannot compare {left_type} to {right_type} with "
                    f"'{op.symbol()}' -- both sides must be the same type",
                    expr,
                )
            return Type.BOOL

        if op == BinaryOp.IN:
            # `key in dict`, or `value in array/slice`. A literal takes its type from the other side:
            # `1 in xs` from the keys or elements, the elements of `a in [1, 2]` from `a`.
            if right_type.kind in (TypeKind.DICT, TypeKind.ARRAY, TypeKind.SLICE):
                wanted = right_type.key_type if right_type.kind == TypeKind.DICT else right_type.element_type
                if self.as_folded_int_literal(expr.left) is not None and not self._untyped_array_literal(expr.right):
                    left_type = self.check_value_flowing_into(expr.left, wanted)
            if right_type.kind == TypeKind.DICT:
                if not self.types_compatible(left_type, right_type.key_type):
                    raise SemanticError(
                        f"Dict declares key type {right_type.key_type}, but 'in's "
                        f"own left operand is {left_type}",
                        expr.left,
                    )
                return Type.BOOL
            if right_type.kind in (TypeKind.ARRAY, TypeKind.SLICE):
                element_type = right_type.element_type
                if not self._is_comparable_type(element_type):
                    raise SemanticError(
                        f"'in' does not support an element type of "
                        f"{element_type} -- membership isn't defined yet "
                        f"when the elements are (or contain) a slice "
                        f"or a dict",
                        expr.right,
                    )
                if not self.types_compatible(left_type, element_type):
                    raise SemanticError(
                        f"{right_type} declares element type {element_type}, "
                        f"but 'in's own left operand is {left_type}",
                        expr.left,
                    )
                return Type.BOOL
            raise SemanticError(
                f"'in' requires a dict, array, or slice as its right operand, "
                f"got {right_type} -- str membership isn't supported yet",
                expr.right,
            )

        if op in LOGICAL_OPS:
            self._require_type(left_type, Type.BOOL, op, expr)
            self._require_type(right_type, Type.BOOL, op, expr)
            return Type.BOOL

        raise SemanticError(f"No semantic rule for binary operator: {op}", expr)

    def check_array_literal(self, expr: ArrayLiteral, expected_element_type: Optional[Type] = None) -> Type:
        """`[e, ...]` or `[N]T[...]`; homogeneous. An untyped literal's elements take
        `expected_element_type`, which where it is used must give: the declared type it flows into, or
        the other operand of `in`, `==`, or `!=`. Elsewhere its type has to be written."""
        if expr.type_expr is not None:
            try:
                declared_type = self._type(expr.type_expr, expr)
            except SemanticError as problem:
                if not expr.could_be_multiplication:
                    raise
                raise SemanticError(problem.message + READ_AS_A_TYPED_LITERAL, expr) from None
            if len(expr.elements) != declared_type.size:
                raise SemanticError(
                    f"Array literal declares type {declared_type} (size "
                    f"{declared_type.size}), but has {len(expr.elements)} "
                    f"element(s)",
                    expr,
                )
            for i, element in enumerate(expr.elements, start=1):
                element_type = self.check_value_flowing_into_allowing_struct_literal(
                    element, declared_type.element_type)
                if not self.types_compatible(element_type, declared_type.element_type):
                    raise SemanticError(
                        f"Array literal declares element type "
                        f"{declared_type.element_type}, but element {i} "
                        f"is {element_type}",
                        element,
                    )
            return declared_type

        if expected_element_type is not None:
            for i, element in enumerate(expr.elements, start=1):
                element_type = self.check_value_flowing_into_allowing_struct_literal(element, expected_element_type)
                if not self.types_compatible(element_type, expected_element_type):
                    raise SemanticError(
                        f"Array literal's elements must all be "
                        f"{expected_element_type} (to match the "
                        f"declared element type), but element {i} "
                        f"is {element_type}",
                        element,
                    )
            return Type(TypeKind.ARRAY, element_type=expected_element_type, size=len(expr.elements))

        if len(expr.elements) == 0:
            raise SemanticError("Array literals must have at least one element", expr)
        try:  # what its type would be written as, if the first element says
            written = f"[{len(expr.elements)}]{self.check_expr_allowing_struct_literal(expr.elements[0])}"
        except SemanticError:
            written = f"[{len(expr.elements)}]T"
        raise SemanticError(
            f"This array literal has nothing to take its type from -- write the type before it, as in "
            f"`{written}[...]`", expr)

    def check_slice_literal(self, expr: 'SliceLiteral') -> Type:
        """`[]T[e, ...]`."""
        element_type = self._type(expr.element_type, expr)
        for i, element in enumerate(expr.elements, start=1):
            actual = self.check_value_flowing_into_allowing_struct_literal(element, element_type)
            if not self.types_compatible(actual, element_type):
                raise SemanticError(
                    f"Slice literal declares element type {element_type}, but element {i} is {actual}", element)
        return Type(TypeKind.SLICE, element_type=element_type)

    def check_dict_literal(self, expr: DictLiteral) -> Type:
        """`dict[K]V{...}`; keys must be distinct constants."""
        key_type = self._type(expr.key_type, expr)
        value_type = self._type(expr.value_type, expr)
        seen_constant_keys = set()
        for key_expr, value_expr in expr.entries:
            actual_key_type = self.check_value_flowing_into(key_expr, key_type)
            if not self.types_compatible(actual_key_type, key_type):
                raise SemanticError(
                    f"Dict literal declares key type {key_type}, but a "
                    f"key is {actual_key_type}",
                    key_expr,
                )
            actual_value_type = self.check_value_flowing_into_allowing_struct_literal(value_expr, value_type)
            if not self.types_compatible(actual_value_type, value_type):
                raise SemanticError(
                    f"Dict literal declares value type {value_type}, but a "
                    f"value is {actual_value_type}",
                    value_expr,
                )
            constant_key = _constant_key_value(key_expr)
            if key_expr.nid in self.facts.enum_members:
                enum, index = self.facts.enum_members[key_expr.nid]
                constant_key = ('enum', f"{shown(enum)}.{self.decls.enums[enum].members[index]}")
            if constant_key is not None:
                if constant_key in seen_constant_keys:
                    raise SemanticError(
                        f"Dict literal lists the key {_shown_key(constant_key)} more than once",
                        key_expr,
                    )
                seen_constant_keys.add(constant_key)
        return Type(TypeKind.DICT, key_type=key_type, element_type=value_type)

    def check_index(self, expr: Index) -> Type:
        """`base[index]`; str indexing yields uint8."""
        base_type = self.check_expr(expr.array)
        if base_type.kind == TypeKind.STR:
            index_type = self.check_expr(expr.index)
            if index_type != Type.INT:
                raise SemanticError(f"Index must be int, got {index_type}", expr.index)
            return Type.UINT8
        return self.check_indexable_and_index(expr.array, expr.index, base_type)

    def check_field(self, expr: Field) -> Type:
        key = self.module_set.qualified.get(expr.nid)
        if key is not None:  # `alias.NAME`: a constant of another module
            if key not in self.constants.decls:
                raise SemanticError(f"Reference to undeclared variable '{shown(key)}'", expr)
            self.facts.const_refs[expr.nid] = key
            return self.constants.type_of(key)
        enum = self.enum_named_by(expr.base)
        if enum is not None:  # `Enum.Member`
            members = self.decls.enums[enum].members
            if expr.name not in members:
                raise SemanticError(
                    f"Enum '{shown(enum)}' has no member '{expr.name}' (its members: {', '.join(members)})", expr)
            self.facts.enum_members[expr.nid] = (enum, members.index(expr.name))
            return Type(TypeKind.ENUM, enum_name=enum)
        return self.check_struct_and_field(expr.base, expr.name)

    def check_slice(self, expr: Slice) -> Type:
        """`array[low:high]` over an array, slice, or str."""
        base_type = self.check_expr(expr.array)
        if base_type.kind not in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STR):
            raise SemanticError(
                f"Cannot slice a value of type {base_type} -- only "
                f"arrays, slices, and str support slicing",
                expr,
            )
        if expr.low is not None:
            low_type = self.check_expr(expr.low)
            if low_type != Type.INT:
                raise SemanticError(f"Slice low bound must be int, got {low_type}", expr.low)
        if expr.high is not None:
            high_type = self.check_expr(expr.high)
            if high_type != Type.INT:
                raise SemanticError(f"Slice high bound must be int, got {high_type}", expr.high)
        if base_type.kind == TypeKind.STR:
            return Type.STR
        return Type(TypeKind.SLICE, element_type=base_type.element_type)

    def types_compatible(self, value_type: Type, target_type: Type) -> bool:
        """Equality, or NONE into a pointer (slices and dicts are never none), or a variant into its sum
        type, or a sum into a wider one: a sum with every variant it has."""
        if value_type == target_type:
            return True
        if value_type == Type.NONE and target_type.kind == TypeKind.POINTER:
            return True
        if target_type.kind == TypeKind.SUM:
            wider = self.decls.sum_types[target_type.sum_type_name].variants
            if value_type.kind == TypeKind.SUM:
                return all(variant in wider for variant in self.decls.sum_types[value_type.sum_type_name].variants)
            return value_type in wider
        return False

    def _may_hold(self, expr: Optional[Node], value_type: Type) -> list:
        """The variants the sum-typed value of `expr` may hold where checking is: all its type's, or
        fewer if it is a variable that `is` checks have narrowed."""
        variants = self.decls.sum_types[value_type.sum_type_name].variants
        decl_id = self.facts.decls.get(expr.nid) if isinstance(expr, Variable) else None
        if decl_id is not None and self.context.scopes.is_narrowed(decl_id):
            possible = self.context.scopes.possible_variants(decl_id)
            return [v for v in variants if v in possible]
        return variants

    def sum_gap(self, value_type: Type, target_type: Type, expr: Optional[Node] = None) -> str:
        """What to add to "a `value_type` doesn't fit a `target_type`" when both are sums: the first
        variant the one can hold (here, if `expr` is the value) that the other has no place for. ''
        when that isn't the reason."""
        if value_type.kind != TypeKind.SUM or target_type.kind != TypeKind.SUM:
            return ""
        wider = self.decls.sum_types[target_type.sum_type_name].variants
        missing = [v for v in self._may_hold(expr, value_type) if v not in wider]
        if not missing:
            return ""
        article = "an" if str(value_type)[:1].lower() in "aeiou" else "a"
        here = " here" if len(self._may_hold(expr, value_type)) < len(self._may_hold(None, value_type)) else ""
        return f" -- {article} {value_type} can hold {missing[0]}{here}, which {target_type} has no variant for"

    def check_value_flowing_into(self, expr: Node, target_type: Type) -> Type:
        """check_expr for a value flowing into a typed slot; handles untyped array literals and literal range checks."""
        if (
                isinstance(expr, ArrayLiteral)
                and expr.type_expr is None
                and target_type.kind in (TypeKind.SLICE, TypeKind.ARRAY)
        ):
            array_type = self.check_array_literal(expr, expected_element_type=target_type.element_type)
            self.facts.types[expr.nid] = array_type
            return target_type if target_type.kind == TypeKind.SLICE else array_type
        if (target_type.kind == TypeKind.POINTER and target_type.element_type.kind == TypeKind.SUM
                and isinstance(expr, Unary) and expr.op == UnaryOp.ADDRESS_OF
                and self.calls.struct_literal(expr.operand) is not None):
            # `&Variant(...)` where a pointer to the sum is expected: a new sum value holding that variant.
            variant_type = self.calls.check_struct_literal(expr.operand)
            if variant_type in self.decls.sum_types[target_type.element_type.sum_type_name].variants:
                self.facts.boxed[expr.nid] = target_type.element_type
                self.facts.types[expr.nid] = target_type
                return target_type
        value_type = self.check_expr(expr)
        if (target_type.kind == TypeKind.SUM and value_type.kind == TypeKind.SUM and isinstance(expr, Variable)
                and not self.types_compatible(value_type, target_type)):
            # A variable of a wider sum, where `is` checks have left it only variants the target has:
            # it flows in as the narrower sum.
            wanted = self.decls.sum_types[target_type.sum_type_name].variants
            if all(variant in wanted for variant in self._may_hold(expr, value_type)):
                self.facts.narrowed_sums[expr.nid] = target_type
                return target_type
        if value_type == Type.NONE and target_type.kind == TypeKind.SLICE:
            raise SemanticError(f"A slice is never none -- write `[]` (or leave the {target_type} uninitialized) "
                                f"for an empty one", expr)
        if value_type == Type.NONE and target_type.kind == TypeKind.DICT:
            raise SemanticError(f"A dict is never none -- write `{target_type}{{}}` (or leave it uninitialized) "
                                f"for an empty one", expr)
        if value_type == Type.INT and target_type.kind == TypeKind.SUM and self.as_folded_int_literal(expr) is not None:
            return self._literal_in_sum(expr, target_type) or value_type
        if value_type == Type.INT and target_type in _NARROW_INT_RANGES:
            literal_value = self.as_folded_int_literal(expr)
            if literal_value is not None:
                lo, hi = _NARROW_INT_RANGES[target_type]
                if not (lo <= literal_value <= hi):
                    raise SemanticError(
                        f"{literal_value} is out of range for "
                        f"{target_type} ({lo} to {hi})",
                        expr,
                    )
                self._record_literal_type(expr, target_type)
                return target_type
        if value_type == Type.INT and target_type == Type.INT64:
            if self.as_folded_int_literal(expr) is not None:
                self._record_literal_type(expr, target_type)
                return target_type
        return value_type

    def _literal_operand_types(self, left: Node, left_type: Type, right: Node, right_type: Type) -> tuple:
        """Both operand types, once an integer literal (or its negation) beside an operand of a
        narrower integer type has taken that type: `fd < 0`, `1 + a`. It must be in range."""
        if right_type in _NARROW_INT_RANGES and self.as_folded_int_literal(left) is not None:
            left_type = self.check_value_flowing_into(left, right_type)
        elif left_type in _NARROW_INT_RANGES and self.as_folded_int_literal(right) is not None:
            right_type = self.check_value_flowing_into(right, left_type)
        return left_type, right_type

    def _require_type(self, actual: Type, expected: Type, op, node: Optional[Node] = None) -> None:
        if actual != expected:
            raise SemanticError(
                f"'{op.symbol()}' requires {expected} operands, got {actual}",
                node,
            )

    def _require_same_integer_type(self, left_type: Type, right_type: Type, op, node: Optional[Node] = None) -> Type:
        """Require identical integer types."""
        if left_type not in INTEGER_TYPES or left_type != right_type:
            raise SemanticError(
                f"'{op.symbol()}' requires two operands of the same "
                f"integer type (int, int8, uint8, or int32), got "
                f"{left_type} and {right_type}",
                node,
            )
        return left_type

    def _is_comparable_type(self, t: Type) -> bool:
        """Whether `==` is defined for `t`."""
        return self._incomparable_part(t) is None

    def _incomparable_part(self, t: Type) -> Optional[Type]:
        """The slice or dict type that keeps `t` from having `==`: `t` itself, or the first one found
        among its elements, fields, or variants. None if `t` has it."""
        if t.kind == TypeKind.ARRAY:
            return self._incomparable_part(t.element_type)
        if t.kind == TypeKind.STRUCT:
            parts = self.decls.structs[t.struct_name].fields.values()
        elif t.kind == TypeKind.SUM:  # equal when they hold the same variant, and it is equal
            parts = self.decls.sum_types[t.sum_type_name].variants
        elif t.kind in (TypeKind.SLICE, TypeKind.DICT):
            return t
        else:
            return None  # integers, bool, str, enums, pointers (by address), none
        return next((part for part in map(self._incomparable_part, parts) if part is not None), None)

    def _literal_in_sum(self, literal: Node, sum_type: Type) -> Optional[Type]:
        """The variant of `sum_type` that an integer literal is, where one is wanted: the sum's int;
        or, there being no int, its one integer variant, which the literal must be in range of (and is
        recorded as). None if the sum has no integer variant. Several, and no int among them, is an
        error: nothing says which."""
        integers = [v for v in self.decls.sum_types[sum_type.sum_type_name].variants if v in INTEGER_TYPES]
        if Type.INT in integers:
            return Type.INT
        if len(integers) > 1:
            raise SemanticError(
                f"{sum_type} has more than one integer variant ({', '.join(map(str, integers))}) and none is int "
                f"-- say which this is, as in `{integers[0]}({self.as_folded_int_literal(literal)})`", literal)
        return self.check_value_flowing_into(literal, integers[0]) if integers else None

    def _check_sum_equality(self, expr: Binary, left_type: Type, right_type: Type) -> Type:
        """`==` or `!=` with a sum on one side or both. They are compared as one sum type, the wider:
        the other side is a narrower sum, or a value of one of its variants. Equal is holding the
        same variant, with equal payloads."""
        op = expr.op.symbol()
        if left_type.kind == TypeKind.SUM and right_type.kind == TypeKind.SUM:
            if self.types_compatible(left_type, right_type):
                compared = right_type
            elif self.types_compatible(right_type, left_type):
                compared = left_type
            else:
                raise SemanticError(
                    f"Cannot compare {left_type} to {right_type} with '{op}' -- neither has all the other's "
                    f"variants{self.sum_gap(left_type, right_type)}", expr)
        else:
            compared, other, value = (left_type, right_type, expr.right) if left_type.kind == TypeKind.SUM \
                else (right_type, left_type, expr.left)
            variants = self.decls.sum_types[compared.sum_type_name].variants
            if self.as_folded_int_literal(value) is not None:
                other = self._literal_in_sum(value, compared) or other
            if other not in variants:
                raise SemanticError(
                    f"Cannot compare {left_type} to {right_type} with '{op}' -- {other} is not one of {compared}'s "
                    f"variants ({', '.join(map(str, variants))})", expr)
        part = self._incomparable_part(compared)
        if part is not None:
            raise SemanticError(
                f"'{op}' does not support {compared} operands -- equality isn't defined yet for a variant that "
                f"is (or contains) {'a slice' if part.kind == TypeKind.SLICE else 'a dict'} ({part})", expr)
        self.facts.sum_equalities[expr.nid] = compared
        return Type.BOOL

    def as_folded_int_literal(self, expr: Node) -> Optional[int]:
        """Folded value of an int literal or its negation, else None."""
        if isinstance(expr, Constant):
            return expr.value
        if isinstance(expr, Unary) and expr.op == UnaryOp.NEGATE and isinstance(expr.operand, Constant):
            return -expr.operand.value
        return None

    def _record_literal_type(self, expr: Node, target_type: Type) -> None:
        """Record a literal (and a negated literal's operand) as having target_type."""
        self.facts.types[expr.nid] = target_type
        if isinstance(expr, Unary) and expr.op == UnaryOp.NEGATE and isinstance(expr.operand, Constant):
            self.facts.types[expr.operand.nid] = target_type

    def check_expr_allowing_struct_literal(self, expr: Node) -> Type:
        """check_expr, but accepts struct literals."""
        if self.calls.struct_literal(expr) is not None:
            return self.calls.check_struct_literal(expr)
        return self.check_expr(expr)

    def check_value_flowing_into_allowing_struct_literal(self, expr: Node, target_type: Type) -> Type:
        """check_value_flowing_into, but accepts struct literals."""
        if self.calls.struct_literal(expr) is not None:
            return self.calls.check_struct_literal(expr)
        return self.check_value_flowing_into(expr, target_type)

    def check_indexable_and_index(self, base_expr: Node, index_expr: Node, base_type: Optional[Type] = None) -> Type:
        """Check an array/slice/dict base and its index; return the element type. `base_type` is the
        base's type where the caller has checked it: checking it again here would double the work at
        each level of `a[i][j][k]...`."""
        if base_type is None:
            base_type = self.check_expr(base_expr)
        if base_type.kind == TypeKind.STR:
            raise SemanticError(
                "Cannot assign into a str via indexing -- str supports "
                "reading a byte by index (`b = s[i]`), but is immutable, "
                "so `s[i] = ...` isn't allowed",
                base_expr,
            )
        if base_type.kind == TypeKind.DICT:
            index_type = self.check_value_flowing_into(index_expr, base_type.key_type)
            if not self.types_compatible(index_type, base_type.key_type):
                raise SemanticError(
                    f"Dict declares key type {base_type.key_type}, but the "
                    f"index is {index_type}",
                    index_expr,
                )
            return base_type.element_type
        if base_type.kind not in (TypeKind.ARRAY, TypeKind.SLICE):
            raise SemanticError(
                f"Cannot index into a value of type {base_type} -- "
                f"only arrays, slices, str, and dict support indexing",
                base_expr,
            )
        index_type = self.check_expr(index_expr)
        if index_type != Type.INT:
            raise SemanticError(f"Index must be int, got {index_type}", index_expr)
        return base_type.element_type

    def _untyped_array_literal(self, expr: Node) -> bool:
        return isinstance(expr, ArrayLiteral) and expr.type_expr is None

    def _array_literal_of(self, literal: ArrayLiteral, element_type: Type) -> Type:
        """Check an untyped array literal whose elements are to be `element_type`; its type."""
        array_type = self.check_array_literal(literal, expected_element_type=element_type)
        self.facts.types[literal.nid] = array_type
        return array_type

    def _operand_types_with_an_untyped_array(self, expr: Binary) -> Optional[tuple]:
        """The operand types of `x in [a, b]`, whose literal's elements take x's type, and of `xs == [a, b]`
        (or `!=`, either way round), whose literal takes the elements of xs. None for any other expression."""
        left_untyped, right_untyped = (self._untyped_array_literal(e) for e in (expr.left, expr.right))
        if expr.op == BinaryOp.IN and right_untyped and not left_untyped:
            if self.as_folded_int_literal(expr.left) is not None:
                # `1 in [a, b]`: an integer literal has no type of its own either, so the first element
                # that isn't one says what they all are (int, if none does).
                typed_elements = [e for e in expr.right.elements if self.as_folded_int_literal(e) is None]
                element_type = self.check_expr_allowing_struct_literal(typed_elements[0]) if typed_elements \
                    else Type.INT
                right_type = self._array_literal_of(expr.right, element_type)
                return self.check_value_flowing_into(expr.left, element_type), right_type
            left_type = self.check_expr_allowing_struct_literal(expr.left)
            return left_type, self._array_literal_of(expr.right, left_type)
        if expr.op == BinaryOp.IN and left_untyped and not right_untyped:
            # `[1, 2] in rows`: the literal is one of the right side's elements.
            right_type = self.check_expr_allowing_struct_literal(expr.right)
            if right_type.kind in (TypeKind.ARRAY, TypeKind.SLICE) and right_type.element_type.kind == TypeKind.ARRAY:
                return self._array_literal_of(expr.left, right_type.element_type.element_type), right_type
            return None
        if expr.op in EQUALITY_OPS and left_untyped != right_untyped:
            literal, other = (expr.left, expr.right) if left_untyped else (expr.right, expr.left)
            other_type = self.check_expr_allowing_struct_literal(other)
            if other_type.kind not in (TypeKind.ARRAY, TypeKind.SLICE):
                return None  # nothing to take a type from: the literal says so when it is checked
            literal_type = self._array_literal_of(literal, other_type.element_type)
            return (literal_type, other_type) if left_untyped else (other_type, literal_type)
        return None

    def enum_named_by(self, expr: Node) -> Optional[str]:
        """The key of the enum that `expr` names (`Enum`, or `alias.Enum`), unless a variable has the name."""
        if isinstance(expr, Variable) and not self.context.scopes.is_local(expr.name):
            key = self.context.scope.resolve(expr.name)
        elif isinstance(expr, Field):
            key = self.module_set.qualified.get(expr.nid)
        else:
            return None
        return key if key in self.decls.enums else None

    def check_struct_and_field(self, base_expr: Node, field_name: str) -> Type:
        """Check base is a struct (auto-deref pointers) with field `field_name`."""
        base_type = self.check_expr_allowing_struct_literal(base_expr)
        if base_type.kind == TypeKind.POINTER and base_type.element_type.kind == TypeKind.STRUCT:
            base_type = base_type.element_type
        if base_type.kind != TypeKind.STRUCT:
            def has_field(variant: Type) -> bool:
                return (variant.kind == TypeKind.STRUCT and field_name in self.decls.structs[variant.struct_name].fields
                        and not self.hidden(variant.struct_name, field_name))
            raise SemanticError(
                f"Cannot access field '{field_name}' on non-struct type {base_type}"
                + self.reaching_into_a_sum(base_expr, base_type, f"a field '{field_name}'", has_field),
                base_expr,
            )
        struct_info = self.decls.structs[base_type.struct_name]
        if field_name not in struct_info.fields:
            raise SemanticError(
                f"Struct '{shown(base_type.struct_name)}' has no field '{field_name}'",
                base_expr,
            )
        if self.hidden(base_type.struct_name, field_name):
            raise SemanticError(
                f"Field '{field_name}' of '{shown(base_type.struct_name)}' is not visible outside the module that "
                f"defines the struct -- names starting with '_' are private to their own module", base_expr)
        return struct_info.fields[field_name]

    def reaching_into_a_sum(self, expr: Node, type_: Type, wanted: str, has: Callable[[Type], bool]) -> str:
        """What to add to an error about using `expr`, a sum (or a pointer to one), as one of its
        variants: how to get at the variant. `has` says whether a variant has what was `wanted` (a
        struct through a pointer counts). '' if `type_` is no sum."""
        through_pointer = type_.kind == TypeKind.POINTER and type_.element_type.kind == TypeKind.SUM
        sum_type = type_.element_type if through_pointer else type_
        if sum_type.kind != TypeKind.SUM:
            return ""

        def fits(variant: Type) -> bool:
            return has(variant.element_type if variant.kind == TypeKind.POINTER else variant)
        holdable = self.decls.sum_types[sum_type.sum_type_name].variants if through_pointer \
            else self._may_hold(expr, sum_type)
        fitting = [variant for variant in holdable if fits(variant)]
        if not fitting:
            listed = ', '.join(map(str, holdable))
            return f" -- {sum_type} is a sum type, and none of its variants ({listed}) has {wanted}"
        written = _written(expr)
        if isinstance(expr, Variable) and not through_pointer:
            return (f" -- {sum_type} is a sum type: test which variant '{expr.name}' holds first, as in "
                    f"`if {expr.name} is {fitting[0]}:`")
        subject = "..." if written is None else (f"*{written}" if through_pointer else written)
        return (f" -- {sum_type} is a sum type, and 'is' narrows only a variable: bind this value to reach its "
                f"variant, as in `if {subject} is {fitting[0]} as NAME:`")

    def hidden(self, struct: str, name: str) -> bool:
        """Whether `name`, a field or method of `struct`, is private to another module: it starts with
        `_`, and the struct is declared in a different file from the one being checked."""
        declared_in = struct.rsplit('$', 1)[0] if '$' in struct else None
        return name.startswith('_') and declared_in != self.context.scope.module

    def without_zero_value(self, t: Type, seen: Optional[set] = None) -> Optional[Type]:
        """The sum type that keeps `t` from having a zero value, or None if it has one. A sum's zero
        value is its `none` variant; a struct or array has one if all its parts do."""
        seen = set() if seen is None else seen
        while t.kind == TypeKind.ARRAY:
            t = t.element_type
        if t.kind == TypeKind.SUM:
            return None if Type.NONE in self.decls.sum_types[t.sum_type_name].variants else t
        if t.kind == TypeKind.STRUCT and t.struct_name not in seen:
            seen.add(t.struct_name)
            for field_type in self.decls.structs[t.struct_name].fields.values():
                missing = self.without_zero_value(field_type, seen)
                if missing is not None:
                    return missing
        return None

    def root_variable_of(self, expr: Node) -> Optional[Variable]:
        """Variable under a Field/Index/Slice chain, if any."""
        while isinstance(expr, (Field, Index, Slice)):
            expr = expr.base if isinstance(expr, Field) else expr.array
        return expr if isinstance(expr, Variable) else None

    def resolve(self, name: str, node: Optional[Node] = None) -> Tuple[Type, object]:
        """(type, decl id) of `name`, innermost-first; constants (decl id None) after locals."""
        local = self.context.scopes.lookup(name)
        if local is not None:
            return local
        if self.const_key(name) is not None:
            return self.constants.type_of(self.const_key(name)), None
        enum = self.decls.enums.get(self.context.scope.resolve(name))
        if enum is not None:
            raise SemanticError(f"'{name}' is an enum, not a value -- write one of its members, such as "
                                f"{name}.{enum.members[0]}", node)
        raise SemanticError(f"Reference to undeclared variable '{shown(name)}'", node)

    def lookup(self, name: str, node: Optional[Node] = None) -> Type:
        return self.resolve(name, node)[0]

    def const_key(self, name: str) -> Optional[str]:
        """The key of the constant a bare name refers to in the current file, if it names one."""
        key = self.context.scope.consts.get(name)
        return key if key in self.constants.decls else None

    def declare_is_binding(self, check: IsCheck) -> None:
        """`EXPR is T as NAME`: check EXPR and declare NAME, a copy of it."""
        subject_type = self.check_expr(check.subject)
        if isinstance(check.subject, Variable) and self.context.scopes.is_narrowed(self.facts.decls[check.subject.nid]):
            subject_type = self.symbols[self.facts.decls[check.subject.nid]].type  # a narrowed variable: its sum
        self.context.bound.add(check.nid)
        sym = self.symbols.new(check.variable_name, 'narrowing', subject_type, check)
        if subject_type.kind in (TypeKind.SUM, TypeKind.ENUM):  # otherwise rejected when the check is checked
            binding = VarDecl(name=check.variable_name, var_type=subject_type.sum_type_name or subject_type.enum_name,
                              init=check.subject,
                              line=check.line, col=check.col, file=check.file)
            self.facts.bindings[check.nid] = binding
            self.facts.types[binding.nid] = subject_type
            self.facts.symbols[binding.nid] = sym
        self.context.scopes.declare(check.variable_name, subject_type, check, sym.id)
