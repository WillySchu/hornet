"""Semantic analysis: name resolution, type checking, and control-flow checks; its result is the
typed tree (typed_ast.py), which every later stage consumes.

Strict typing: no implicit conversions; integer operands must match exactly (an integer literal
takes the other operand's type). Blocks scope lexically and may shadow. Non-void functions must
return on all paths.

The parser's tree is never changed: checking records what it learns in Facts, keyed by node number,
and _TypedTreeBuilder builds each function's typed tree from those facts at the end of analyze().
"""

import argparse
import contextlib
import dataclasses
from typing import Dict, List, Optional, Tuple

from diagnostics import quoted_text
from lexer import lex
from semantic.constants import ConstEvaluator
from semantic.declarations import DeclarationResolver, Declarations
from semantic.errors import SemanticError, SemanticErrors
from semantic.facts import Facts
from semantic.flow import (
    Scopes, always_ends, always_leaves, always_returns, conditions_after, conjuncts, contains_reachable_break,
)
from semantic.types import TypeResolver, type_from_name  # noqa: F401 (type_from_name: for those who import it here)
from semantic.typed_tree_builder import TypedTreeBuilder
from typesys import INTEGER_TYPES, StructInfo, SumTypeInfo, Type, TypeKind  # noqa: F401 (some for importers)
from ops import EQUALITY_OPS, LOGICAL_OPS, ORDERING_OPS
import typed_ast as typed
from scopes import build_module_set, display_name as shown
from symbols import SymbolTable
from parser import (
    ConstDecl,
    ArrayLiteral,
    Assign,
    Binary,
    BinaryOp,
    BoolLiteral,
    Break,
    ByteLiteral,
    Call,
    Cast,
    Constant,
    Continue,
    DictLiteral,
    ExprStmt,
    Field,
    QualifiedTypeExpr,
    READ_AS_A_TYPED_LITERAL,
    For,
    ForIn,
    Function,
    If,
    Index,
    IsCheck,
    Match,
    Node,
    NoneLiteral,
    Parser,
    Program,
    Return,
    Slice,
    SliceLiteral,
    StringLiteral,
    Unary,
    UnaryOp,
    VarDecl,
    Variable,
    While,
)






























_BYTE_SLICE = Type(TypeKind.SLICE, element_type=Type.UINT8)


# Errors


# Function bodies checked before giving up.
MAX_ERRORS = 20


# Analyzer

# ADD is excluded: it also concatenates strings.
_INT_ONLY_BINARY_OPS = {
    BinaryOp.SUBTRACT, BinaryOp.MULTIPLY, BinaryOp.DIVIDE, BinaryOp.MODULO,
    BinaryOp.BITWISE_AND, BinaryOp.BITWISE_OR, BinaryOp.BITWISE_XOR,
    BinaryOp.SHIFT_LEFT, BinaryOp.SHIFT_RIGHT,
}
_ORDERING_OPS = ORDERING_OPS
_EQUALITY_OPS = EQUALITY_OPS
_LOGICAL_OPS = LOGICAL_OPS

_INTEGER_TYPES = INTEGER_TYPES

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
        self.scopes = None  # the names in scope where checking is (semantic/flow.py): a Scopes per function
        self.loop_depth = 0  # enclosing loop count
        self.decls = Declarations()  # what the program declares (semantic/declarations.py), read as `self.decls.X`

    def analyze(self, entry: Program, modules: Optional[dict] = None) -> "typed.Program":
        """Check the entry file and the modules it imports (discover_modules's result) and return the
        typed tree; no parser tree is changed."""
        self.symbols = SymbolTable()
        self.facts = Facts()
        # Each declaration under its key, with its file's scope (see scopes.py).
        program = build_module_set(entry, modules or {})
        self.module_set = program
        self.scope = program.files[0][1]
        # Declarations first, all of them: what each body is then checked against.
        self.decls = Declarations()
        self.scopes = Scopes(self.symbols, self.decls, self.facts)  # (no locals, until a function is checked)
        self.types = TypeResolver(self.decls, program, self.facts.array_sizes)
        self.constants = ConstEvaluator(self.facts, self.decls.enums, self._check_const_declaration)
        DeclarationResolver(
            program, self.decls, self.types, self.facts, self.constants, self._array_size_value).resolve()

        # 5. Check bodies, collecting at most one error per function.
        errors: List[SemanticError] = []
        for fn in self.decls.all_functions:
            try:
                self.analyze_function(fn)
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

        # 6. The typed tree: what later stages consume.
        return TypedTreeBuilder(self.facts, self.decls, self.constants.values, self.symbols).program(
            self.decls.all_functions)

    # -- names across modules

    def _enter(self, decl: Node) -> None:
        """Check `decl` (a top-level declaration, under its key) in its own file's scope."""
        self.scope = self.module_set.scope_of[decl.nid]

    def _resolve_type_name(self, name):
        """A type name (or `alias.Name`) as written in the current file -> its declaration's key."""
        return self.types.key(name, self.scope)

    def _type(self, type_expr, node: Node, sums: bool = True) -> Type:
        return self.types.resolve(type_expr, node, self.scope, sums)

    def _const_key(self, name: str) -> Optional[str]:
        """The key of the constant a bare name refers to in the current file, if it names one."""
        key = self.scope.consts.get(name)
        constants = getattr(self, 'constants', None)
        return key if constants is not None and key in constants.decls else None

    def _callee(self, expr: Call) -> Optional[str]:
        """The key a call's name refers to (a function, struct, extern, or intrinsic), resolving
        `alias.name(...)`; the name as written if it names nothing at top level (a builtin, or
        undeclared); None for a method call."""
        if expr.nid in self.module_set.qualified:
            return self.module_set.qualified[expr.nid]
        if expr.receiver is not None:
            return None
        return self.scope.resolve(expr.name) or expr.name

    def _struct_literal(self, expr: Node) -> Optional[str]:
        """The struct's key if `expr` is a struct literal."""
        if isinstance(expr, Call):
            name = self._callee(expr)
            if name in self.decls.structs:
                return name
        return None

    # -- constants



    def _array_size_value(self, expr: Node, scope) -> int:
        """The value of an array-size expression written in the file whose scope is `scope`: a positive
        integer computed from literals and constants only. (For DeclarationResolver.)"""
        saved_scope, self.scope = self.scope, scope
        try:
            return self._array_size_value_here(expr)
        finally:
            self.scope = saved_scope

    def _array_size_value_here(self, expr: Node) -> int:
        stack = [expr]
        while stack:
            node = stack.pop()
            if node.nid in self.module_set.qualified and isinstance(node, Field):
                continue  # `alias.NAME`: checked as a constant below
            if isinstance(node, Call) and node.name == 'len' and len(node.args) == 1 and node.receiver is None \
                    and self._enum_named_by(node.args[0]) is not None:
                continue  # `len(Enum)`: a constant
            if isinstance(node, Call):
                raise SemanticError("Array size must be a constant expression, not a call", node)
            if isinstance(node, Variable) and self._const_key(node.name) is None:
                raise SemanticError(
                    f"Array size must be a constant expression, but '{node.name}' isn't a constant", node)
            if dataclasses.is_dataclass(node):
                stack.extend(v for f in dataclasses.fields(node) if isinstance(v := getattr(node, f.name), Node))
        with self._no_locals():
            size_type = self.check_expr(expr)
            if size_type not in _INTEGER_TYPES:
                raise SemanticError(f"Array size must be an integer, got {size_type}", expr)
            value = self.constants.evaluate(expr)
        if value <= 0:
            raise SemanticError(f"Array size must be positive, got {value}", expr)
        return value




    def _check_const_declaration(self, cd: ConstDecl) -> Type:
        """Check a constant's declaration, in the scope of the file that declares it and with no
        local in sight; its type. (For ConstEvaluator, which then works out the value.)"""
        saved_scope = self.scope
        self._enter(cd)
        try:
            with self._no_locals():
                const_type = self._type(cd.const_type, cd)
                if const_type not in _INTEGER_TYPES and const_type not in (Type.BOOL, Type.STR) \
                        and const_type.kind != TypeKind.ENUM:
                    raise SemanticError(
                        f"Constant '{shown(cd.name)}' has type {const_type} -- constants must be an integer type, "
                        f"bool, str, or an enum", cd)
                value_type = self._check_value_flowing_into(cd.value, const_type)
                if not self._types_compatible(value_type, const_type):
                    raise SemanticError(
                        f"Constant '{shown(cd.name)}' is declared {const_type} but its value has type {value_type}",
                        cd)
        finally:
            self.scope = saved_scope
        return const_type

    @contextlib.contextmanager
    def _no_locals(self):
        """While a constant expression is checked: no local is in scope, wherever checking was."""
        saved, self.scopes = self.scopes, Scopes(self.symbols, self.decls, self.facts)
        try:
            yield
        finally:
            self.scopes = saved









    def analyze_function(self, fn: Function) -> None:
        self._enter(fn)
        self.scopes = Scopes(self.symbols, self.decls, self.facts)
        self._bindable, self._bound = set(), set()  # `as NAME` checks that may bind here; those that have
        self._assignments = []  # decl ids, as variables are assigned on paths that reach what follows
        self.loop_depth = 0
        # Params are locals; _declare also catches duplicates.
        for p in fn.params:
            param_type = self._type(p.type, p)
            self.facts.types[p.nid] = param_type
            self.facts.symbols[p.nid] = self.symbols.new(p.name, 'param', param_type, p)
            self.scopes.declare(p.name, param_type, p, self.facts.symbols[p.nid].id)
        return_type = self._return_type(fn)
        self.facts.returns[fn.nid] = return_type
        self._analyze_block(fn.body, return_type)
        if return_type == Type.NEVER and not always_ends(fn.body, self.facts.types, ()):
            raise SemanticError(
                f"Function '{shown(fn.name)}' is declared never, but can finish -- every path must end in a call "
                f"to a never function (such as panic) or in a `while true` it doesn't break out of", fn)
        # Void functions may fall off the end.
        if return_type not in (Type.VOID, Type.NEVER) and not always_returns(fn.body, self.facts.types):
            raise SemanticError(
                f"Function '{shown(fn.name)}' (declared to return {return_type}) "
                f"does not return a value on all code paths",
                fn,
            )

    def _return_type(self, decl) -> Type:
        return self.types.return_type(decl, self.scope)




    # -- narrowing: what `is` checks have established about a sum variable, within a region








    def _analyze_block(self, statements: List[Node], return_type: Type) -> None:
        """Check a block's statements in the current scope. What an `if` or a `while` establishes for
        the code after it narrows the rest of the block; what a statement assigns is forgotten."""
        mark = self.scopes.mark()
        for stmt in statements:
            first = len(self._assignments)
            self.analyze_statement(stmt, return_type)
            assigned = set(self._assignments[first:])  # here, or in nested blocks that reach what follows
            if isinstance(stmt, If):
                known_false, known_true = conditions_after(stmt, self.facts.types)
                for condition in known_false:
                    self.scopes.apply(self.scopes.when(condition)[1])
                if known_true is not None:
                    self.scopes.apply(self.scopes.when(known_true)[0])
            for decl_id in assigned:
                self.scopes.forget(decl_id)
            if isinstance(stmt, While) and not contains_reachable_break(stmt.body):
                self.scopes.apply(self.scopes.when(stmt.condition)[1])  # the loop ended because its condition was false
        self.scopes.end_region(mark)

    def _analyze_body(self, statements: List[Node], return_type: Type, known: Optional[dict] = None) -> None:
        """A block in a scope of its own, narrowed by `known` (one of _when's results)."""
        mark, first = self.scopes.mark(), len(self._assignments)
        self.scopes.push()
        self.scopes.apply(known)
        self._analyze_block(statements, return_type)
        self.scopes.pop()
        self.scopes.end_region(mark)
        if always_leaves(statements, self.facts.types):
            del self._assignments[first:]  # what follows the enclosing statement isn't reached from here

    def _resolve(self, name: str, node: Optional[Node] = None) -> Tuple[Type, object]:
        """(type, decl id) of `name`, innermost-first; constants (decl id None) after locals."""
        local = self.scopes.lookup(name)
        if local is not None:
            return local
        if self._const_key(name) is not None:
            return self.constants.type_of(self._const_key(name)), None
        enum = self.decls.enums.get(self.scope.resolve(name))
        if enum is not None:
            raise SemanticError(f"'{name}' is an enum, not a value -- write one of its members, such as "
                                f"{name}.{enum.members[0]}", node)
        raise SemanticError(f"Reference to undeclared variable '{shown(name)}'", node)

    def _lookup(self, name: str, node: Optional[Node] = None) -> Type:
        return self._resolve(name, node)[0]

    def analyze_statement(self, stmt: Node, return_type: Type) -> None:
        if isinstance(stmt, VarDecl):
            self.analyze_var_decl(stmt)
        elif isinstance(stmt, Assign):
            self.analyze_assign(stmt)
        elif isinstance(stmt, Return):
            self.analyze_return(stmt, return_type)
        elif isinstance(stmt, If):
            self.analyze_if(stmt, return_type)
        elif isinstance(stmt, Match):
            self.analyze_match(stmt, return_type)
        elif isinstance(stmt, While):
            self.analyze_while(stmt, return_type)
        elif isinstance(stmt, For):
            self.analyze_for(stmt, return_type)
        elif isinstance(stmt, ForIn):
            self.analyze_for_in(stmt, return_type)
        elif isinstance(stmt, Break):
            self.analyze_break(stmt)
        elif isinstance(stmt, Continue):
            self.analyze_continue(stmt)
        elif isinstance(stmt, ExprStmt):
            self._check_expr_allowing_struct_literal(stmt.expr)
        else:
            raise SemanticError(f"No semantic rule for statement: {stmt!r}", stmt)

    def _types_compatible(self, value_type: Type, target_type: Type) -> bool:
        """Equality, or NONE into a pointer (slices and dicts are never none), or a variant into its sum type."""
        if value_type == target_type:
            return True
        if value_type == Type.NONE and target_type.kind == TypeKind.POINTER:
            return True
        if target_type.kind == TypeKind.SUM and value_type.kind != TypeKind.SUM:
            return value_type in self.decls.sum_types[target_type.sum_type_name].variants
        return False

    def _as_folded_int_literal(self, expr: Node) -> Optional[int]:
        """Folded value of an int literal or its negation, else None."""
        if isinstance(expr, Constant):
            return expr.value
        if isinstance(expr, Unary) and expr.op == UnaryOp.NEGATE and isinstance(expr.operand, Constant):
            return -expr.operand.value
        return None

    def _check_value_flowing_into(self, expr: Node, target_type: Type) -> Type:
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
                and self._struct_literal(expr.operand) is not None):
            # `&Variant(...)` where a pointer to the sum is expected: a new sum value holding that variant.
            variant_type = self.check_struct_literal(expr.operand)
            if variant_type in self.decls.sum_types[target_type.element_type.sum_type_name].variants:
                self.facts.boxed[expr.nid] = target_type.element_type
                self.facts.types[expr.nid] = target_type
                return target_type
        value_type = self.check_expr(expr)
        if value_type == Type.NONE and target_type.kind == TypeKind.SLICE:
            raise SemanticError(f"A slice is never none -- write `[]` (or leave the {target_type} uninitialized) "
                                f"for an empty one", expr)
        if value_type == Type.NONE and target_type.kind == TypeKind.DICT:
            raise SemanticError(f"A dict is never none -- write `{target_type}{{}}` (or leave it uninitialized) "
                                f"for an empty one", expr)
        if value_type == Type.INT and target_type in _NARROW_INT_RANGES:
            literal_value = self._as_folded_int_literal(expr)
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
            if self._as_folded_int_literal(expr) is not None:
                self._record_literal_type(expr, target_type)
                return target_type
        return value_type

    def _record_literal_type(self, expr: Node, target_type: Type) -> None:
        """Record a literal (and a negated literal's operand) as having target_type."""
        self.facts.types[expr.nid] = target_type
        if isinstance(expr, Unary) and expr.op == UnaryOp.NEGATE and isinstance(expr.operand, Constant):
            self.facts.types[expr.operand.nid] = target_type

    def _check_expr_allowing_struct_literal(self, expr: Node) -> Type:
        """check_expr, but accepts struct literals."""
        if self._struct_literal(expr) is not None:
            return self.check_struct_literal(expr)
        return self.check_expr(expr)

    def _check_value_flowing_into_allowing_struct_literal(self, expr: Node, target_type: Type) -> Type:
        """_check_value_flowing_into, but accepts struct literals."""
        if self._struct_literal(expr) is not None:
            return self.check_struct_literal(expr)
        return self._check_value_flowing_into(expr, target_type)

    def analyze_var_decl(self, stmt: VarDecl) -> None:
        declared_type = self._type(stmt.var_type, stmt)
        missing = self._without_zero_value(declared_type) if stmt.init is None else None
        if missing is not None:
            # Only a sum with a `none` variant has a zero value (that variant).
            what = f"{missing} has" if missing == declared_type else f"{declared_type} contains {missing}, which has"
            raise SemanticError(
                f"'{stmt.name}' (declared {declared_type}) has no initializer -- {what} no zero value, "
                f"so one is required here (a sum type's zero value is its `none` variant, if it has one)",
                stmt,
            )
        if stmt.init is not None:
            # Checked before declaring, so `int a = a` fails.
            init_type = self._check_value_flowing_into_allowing_struct_literal(stmt.init, declared_type)
            if not self._types_compatible(init_type, declared_type):
                raise SemanticError(
                    f"Cannot initialize '{stmt.name}' (declared {declared_type}) "
                    f"with a value of type {init_type}",
                    stmt,
                )
        self.facts.types[stmt.nid] = declared_type
        self.facts.symbols[stmt.nid] = self.symbols.new(stmt.name, 'local', declared_type, stmt)
        self.scopes.declare(stmt.name, declared_type, stmt, self.facts.symbols[stmt.nid].id)

    def analyze_assign(self, stmt: Assign) -> None:
        """`target = value` or `target op= value`, to a name, field, element, or pointee."""
        target = stmt.target
        if isinstance(target, Variable):
            if self._const_key(target.name) is not None and not self.scopes.is_local(target.name):
                raise SemanticError(f"Cannot assign to constant '{target.name}'", stmt)
            _, decl_id = self._resolve(target.name, stmt)
            self.facts.decls[target.nid] = decl_id
            target_type = self.symbols[decl_id].type if decl_id is not None else self._lookup(target.name, stmt)
            what = f"to '{target.name}' (declared {target_type})"
        elif isinstance(target, Index):
            target_type = self._check_indexable_and_index(target.array, target.index)
            what = f"to an array element of type {target_type}"
        elif isinstance(target, Field):
            if target.nid in self.module_set.qualified:
                raise SemanticError(f"Cannot assign to constant '{target.name}'", stmt)
            target_type = self._check_struct_and_field(target.base, target.name)
            what = f"to field '{target.name}' of type {target_type}"
        else:
            pointer_type = self.check_expr(target.operand)
            if pointer_type.kind != TypeKind.POINTER:
                raise SemanticError(
                    f"Cannot dereference a value of type {pointer_type} for assignment -- '*' requires a pointer "
                    f"operand",
                    target.operand,
                )
            target_type = pointer_type.element_type
            what = f"through a pointer to {target_type}"
        self.facts.types[target.nid] = target_type
        if stmt.op is not None:
            # Checked as the operation it performs: `target op value`.
            self.check_binary(Binary(op=stmt.op, left=target, right=stmt.value, line=stmt.line, col=stmt.col,
                                     file=stmt.file))
            return
        value_type = self._check_value_flowing_into_allowing_struct_literal(stmt.value, target_type)
        if not self._types_compatible(value_type, target_type):
            raise SemanticError(f"Cannot assign a value of type {value_type} {what}", stmt)
        if isinstance(target, Variable) and self.facts.decls[target.nid] is not None:
            self._assignments.append(self.facts.decls[target.nid])  # the value was read as narrowed; no longer
            self.scopes.forget(self.facts.decls[target.nid])

    def _check_indexable_and_index(self, base_expr: Node, index_expr: Node, base_type: Optional[Type] = None) -> Type:
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
            index_type = self._check_value_flowing_into(index_expr, base_type.key_type)
            if not self._types_compatible(index_type, base_type.key_type):
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

    def analyze_return(self, stmt: Return, return_type: Type) -> None:
        """`return [expr]`; bare return only in void functions."""
        if return_type == Type.NEVER:
            raise SemanticError("Function is declared never, so it can't return", stmt)
        if stmt.value is None:
            if return_type != Type.VOID:
                raise SemanticError(
                    f"Function is declared to return {return_type}, but "
                    f"this bare 'return' returns nothing",
                    stmt,
                )
            return
        value_type = self._check_value_flowing_into_allowing_struct_literal(stmt.value, return_type)
        if return_type == Type.VOID:
            raise SemanticError(
                f"Function has no declared return type and cannot "
                f"return a value (got {value_type}) -- use a bare "
                f"'return' instead",
                stmt,
            )
        if not self._types_compatible(value_type, return_type):
            raise SemanticError(
                f"Function is declared to return {return_type}, but this "
                f"'return' statement returns {value_type}",
                stmt,
            )

    def analyze_if(self, stmt: If, return_type: Type) -> None:
        # `EXPR is T as NAME` binds NAME when it is the condition, or one of the checks its `and`s join
        # (check_is_check declares it, in evaluation order). A scope of their own holds the names.
        binders = [c for c in conjuncts(stmt.condition) if isinstance(c, IsCheck) and c.binds]
        whole = isinstance(stmt.condition, IsCheck) and stmt.condition.binds
        if binders:
            self.scopes.push()
            self._bindable.update(c.nid for c in binders)

        condition_type = self.check_expr(stmt.condition)
        if condition_type != Type.BOOL:
            raise SemanticError(
                f"'if' condition must be bool, got {condition_type} "
                f"(no implicit int-to-bool conversion -- try `x != 0` "
                f"instead of `x`)",
                stmt.condition,
            )

        # What the condition says when true narrows then_body, and what it says when false narrows
        # else_body (and what follows the `if`, when its other branches always leave: _analyze_block).
        when_true, when_false = self.scopes.when(stmt.condition)
        self._analyze_body(stmt.then_body, return_type, when_true)
        if binders and not whole:
            self.scopes.pop()  # in the `else`, a check after an `and` may never have run: its name is unset
        if stmt.else_body is not None:
            self._analyze_body(stmt.else_body, return_type, when_false)
        if whole:
            self.scopes.pop()

    def analyze_match(self, stmt: 'Match', return_type: Type) -> None:
        """Each arm narrows the subject within its own body, like `if NAME is T:` (the order of checks,
        and so of errors, is that of the equivalent `if`/`else` chain)."""
        first_check = stmt.arms[0][0]
        has_binding = stmt.subject is not None
        if has_binding:
            self.scopes.push()
            self._declare_is_binding(first_check)
        self.check_expr(first_check)
        if first_check.nid in self.facts.enum_checks:
            self._analyze_enum_match(stmt, return_type)
            if has_binding:
                self.scopes.pop()
            return
        tested = set()
        for i, (check, body) in enumerate(stmt.arms):
            self.check_expr(check)
            if i == 0:
                self._check_match_exhaustiveness(stmt)
            variant = self.facts.narrowed[check.nid]
            tested.add(variant)
            self._analyze_body(body, return_type, self.scopes.when(check)[0])
        if stmt.else_body is not None:  # the subject is none of the arms' variants
            decl_id = self.facts.decls[first_check.nid]
            self._analyze_body(stmt.else_body, return_type,
                               {decl_id: (stmt.variable_name, self.scopes.possible_variants(decl_id) - tested)})
        if has_binding:
            self.scopes.pop()

    def _analyze_enum_match(self, stmt: 'Match', return_type: Type) -> None:
        """`match` on an enum: each arm names a member, and the arms cover every member unless there is
        an `else`. Nothing is narrowed."""
        subject_type = self.facts.enum_checks[stmt.arms[0][0].nid][1]
        members = self.decls.enums[subject_type.enum_name].members
        seen = set()
        for check, _ in stmt.arms:
            self.check_expr(check)
            if check.nid not in self.facts.enum_checks:
                raise SemanticError(f"'{check.type_name}' is not a member of {subject_type}", check)
            index = self.facts.enum_checks[check.nid][0]
            if index in seen:
                raise SemanticError(
                    f"'{members[index]}' is tested more than once in this match on '{stmt.variable_name}'", check)
            seen.add(index)
        missing = [member for index, member in enumerate(members) if index not in seen]
        if missing and stmt.else_body is None:
            raise SemanticError(
                f"This match on '{stmt.variable_name}' (declared {subject_type}) doesn't cover every member -- "
                f"missing: {', '.join(missing)} (add an arm for each, or an 'else:' to cover the rest)", stmt)
        for _, body in stmt.arms:
            self._analyze_body(body, return_type)
        if stmt.else_body is not None:
            self._analyze_body(stmt.else_body, return_type)

    def _declare_is_binding(self, check: IsCheck) -> None:
        """`EXPR is T as NAME`: check EXPR and declare NAME, a copy of it."""
        subject_type = self.check_expr(check.subject)
        if isinstance(check.subject, Variable) and self.scopes.is_narrowed(self.facts.decls[check.subject.nid]):
            subject_type = self.symbols[self.facts.decls[check.subject.nid]].type  # a narrowed variable: its sum
        self._bound.add(check.nid)
        sym = self.symbols.new(check.variable_name, 'narrowing', subject_type, check)
        if subject_type.kind in (TypeKind.SUM, TypeKind.ENUM):  # otherwise rejected when the check is checked
            binding = VarDecl(name=check.variable_name, var_type=subject_type.sum_type_name or subject_type.enum_name,
                              init=check.subject,
                              line=check.line, col=check.col, file=check.file)
            self.facts.bindings[check.nid] = binding
            self.facts.types[binding.nid] = subject_type
            self.facts.symbols[binding.nid] = sym
        self.scopes.declare(check.variable_name, subject_type, check, sym.id)

    def _check_match_exhaustiveness(self, stmt: 'Match') -> None:
        """Reject duplicate arms; require exhaustiveness without an else."""
        subject_name = stmt.variable_name
        subject_type = self.symbols[self.facts.decls[stmt.arms[0][0].nid]].type  # as declared, not as narrowed
        sum_type_info = self.decls.sum_types[subject_type.sum_type_name]

        seen: Dict[Type, IsCheck] = {}
        for arm_condition, _ in stmt.arms:
            arm_type = self._type(arm_condition.type_name, arm_condition)
            if arm_type in seen:
                raise SemanticError(
                    f"'{arm_condition.type_name}' is tested more than once in "
                    f"this match on '{subject_name}'",
                    arm_condition,
                )
            seen[arm_type] = arm_condition

        if stmt.else_body is not None:
            return

        missing = [v for v in sum_type_info.variants if v not in seen]
        if missing:
            raise SemanticError(
                f"This match on '{subject_name}' (declared {subject_type}) "
                f"doesn't cover every variant -- missing: "
                f"{', '.join(str(v) for v in missing)} "
                f"(add an arm for each, or an 'else:' to cover the rest)",
                stmt,
            )

    def analyze_while(self, stmt: While, return_type: Type) -> None:
        """The condition narrows the body, like an `if`'s (and, when it is false, what follows a loop
        with no `break`: _analyze_block)."""
        self.scopes.forget_assigned_in(stmt.body)
        condition_type = self.check_expr(stmt.condition)
        if condition_type != Type.BOOL:
            raise SemanticError(
                f"'while' condition must be bool, got {condition_type} "
                f"(no implicit int-to-bool conversion -- try `x != 0` "
                f"instead of `x`)",
                stmt.condition,
            )

        self.loop_depth += 1
        self._analyze_body(stmt.body, return_type, self.scopes.when(stmt.condition)[0])
        self.loop_depth -= 1

    def analyze_for(self, stmt: For, return_type: Type) -> None:
        """`for init; cond; increment:`; one scope spans all clauses."""
        self.scopes.push()
        self.analyze_statement(stmt.init, return_type)
        self.scopes.forget_assigned_in([stmt.body, stmt.increment])
        condition_type = self.check_expr(stmt.condition)
        if condition_type != Type.BOOL:
            raise SemanticError(
                f"'for' condition must be bool, got {condition_type} "
                f"(no implicit int-to-bool conversion -- try `x != 0` "
                f"instead of `x`)",
                stmt.condition,
            )
        self.loop_depth += 1
        mark = self.scopes.mark()
        self.scopes.apply(self.scopes.when(stmt.condition)[0])  # the condition narrows the body, as a `while`'s does
        self._analyze_block(stmt.body, return_type)
        self.scopes.end_region(mark)
        self.loop_depth -= 1
        self.analyze_statement(stmt.increment, return_type)
        self.scopes.pop()

    def analyze_for_in(self, stmt: ForIn, return_type: Type) -> None:
        """`for a[, b] in iterable:` over arrays, slices, dicts, and strings (bytes)."""
        if not isinstance(
                stmt.iterable,
                (Variable, Field, Index, Slice, ArrayLiteral, SliceLiteral, DictLiteral, StringLiteral, Call)):
            raise SemanticError(
                f"'for ... in' requires a variable, field, index, "
                f"slice, or array/dict/str literal as its own iterable, "
                f"not a {type(stmt.iterable).__name__}",
                stmt.iterable,
            )
        if isinstance(stmt.iterable, Call) or (
                isinstance(stmt.iterable, (Field, Index, Slice))
                and self._root_variable_of(stmt.iterable) is None):
            raise SemanticError(
                f"'for ... in' does not support a function call result "
                f"as its own iterable, or anywhere in its own "
                f"iterable's chain -- assign it to a variable first",
                stmt.iterable,
            )
        iterable_type = self.check_expr(stmt.iterable)
        if iterable_type.kind not in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.DICT, TypeKind.STR):
            raise SemanticError(
                f"'for ... in' requires an array, slice, dict, or str as its own "
                f"iterable, got {iterable_type}",
                stmt.iterable,
            )
        num_bindings = len(stmt.binding_names)
        if iterable_type.kind == TypeKind.STR:
            binding_types = [Type.INT, Type.UINT8] if num_bindings == 2 else [Type.UINT8]
        elif iterable_type.kind == TypeKind.DICT:
            binding_types = [iterable_type.key_type, iterable_type.element_type][:num_bindings]
        else:
            binding_types = [
                Type.INT, iterable_type.element_type] if num_bindings == 2 else [iterable_type.element_type]
        self.scopes.push()
        symbols = [
            self.symbols.new(name, 'binding', t, stmt) for name, t in zip(stmt.binding_names, binding_types)
        ]
        self.facts.for_symbols[stmt.nid] = symbols
        for name, binding_type, sym in zip(stmt.binding_names, binding_types, symbols):
            self.scopes.declare(name, binding_type, stmt, sym.id)
        self.scopes.forget_assigned_in(stmt.body)
        self.loop_depth += 1
        self._analyze_block(stmt.body, return_type)
        self.loop_depth -= 1
        self.scopes.pop()

    def analyze_break(self, stmt: Break) -> None:
        if self.loop_depth == 0:
            raise SemanticError("'break' outside of a loop", stmt)

    def analyze_continue(self, stmt: Continue) -> None:
        if self.loop_depth == 0:
            raise SemanticError("'continue' outside of a loop", stmt)

    # Expressions

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
            result = self.check_call(expr)
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

    def check_dict_literal(self, expr: DictLiteral) -> Type:
        """`dict[K]V{...}`; keys must be distinct constants."""
        key_type = self._type(expr.key_type, expr)
        value_type = self._type(expr.value_type, expr)
        seen_constant_keys = set()
        for key_expr, value_expr in expr.entries:
            actual_key_type = self._check_value_flowing_into(key_expr, key_type)
            if not self._types_compatible(actual_key_type, key_type):
                raise SemanticError(
                    f"Dict literal declares key type {key_type}, but a "
                    f"key is {actual_key_type}",
                    key_expr,
                )
            actual_value_type = self._check_value_flowing_into_allowing_struct_literal(value_expr, value_type)
            if not self._types_compatible(actual_value_type, value_type):
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

    def check_slice_literal(self, expr: 'SliceLiteral') -> Type:
        """`[]T[e, ...]`."""
        element_type = self._type(expr.element_type, expr)
        for i, element in enumerate(expr.elements, start=1):
            actual = self._check_value_flowing_into_allowing_struct_literal(element, element_type)
            if not self._types_compatible(actual, element_type):
                raise SemanticError(
                    f"Slice literal declares element type {element_type}, but element {i} is {actual}", element)
        return Type(TypeKind.SLICE, element_type=element_type)

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
                element_type = self._check_value_flowing_into_allowing_struct_literal(
                    element, declared_type.element_type)
                if not self._types_compatible(element_type, declared_type.element_type):
                    raise SemanticError(
                        f"Array literal declares element type "
                        f"{declared_type.element_type}, but element {i} "
                        f"is {element_type}",
                        element,
                    )
            return declared_type

        if expected_element_type is not None:
            for i, element in enumerate(expr.elements, start=1):
                element_type = self._check_value_flowing_into_allowing_struct_literal(element, expected_element_type)
                if not self._types_compatible(element_type, expected_element_type):
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
            written = f"[{len(expr.elements)}]{self._check_expr_allowing_struct_literal(expr.elements[0])}"
        except SemanticError:
            written = f"[{len(expr.elements)}]T"
        raise SemanticError(
            f"This array literal has nothing to take its type from -- write the type before it, as in "
            f"`{written}[...]`", expr)

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
            if self._as_folded_int_literal(expr.left) is not None:
                # `1 in [a, b]`: an integer literal has no type of its own either, so the first element
                # that isn't one says what they all are (int, if none does).
                typed_elements = [e for e in expr.right.elements if self._as_folded_int_literal(e) is None]
                element_type = self._check_expr_allowing_struct_literal(typed_elements[0]) if typed_elements \
                    else Type.INT
                right_type = self._array_literal_of(expr.right, element_type)
                return self._check_value_flowing_into(expr.left, element_type), right_type
            left_type = self._check_expr_allowing_struct_literal(expr.left)
            return left_type, self._array_literal_of(expr.right, left_type)
        if expr.op == BinaryOp.IN and left_untyped and not right_untyped:
            # `[1, 2] in rows`: the literal is one of the right side's elements.
            right_type = self._check_expr_allowing_struct_literal(expr.right)
            if right_type.kind in (TypeKind.ARRAY, TypeKind.SLICE) and right_type.element_type.kind == TypeKind.ARRAY:
                return self._array_literal_of(expr.left, right_type.element_type.element_type), right_type
            return None
        if expr.op in _EQUALITY_OPS and left_untyped != right_untyped:
            literal, other = (expr.left, expr.right) if left_untyped else (expr.right, expr.left)
            other_type = self._check_expr_allowing_struct_literal(other)
            if other_type.kind not in (TypeKind.ARRAY, TypeKind.SLICE):
                return None  # nothing to take a type from: the literal says so when it is checked
            literal_type = self._array_literal_of(literal, other_type.element_type)
            return (literal_type, other_type) if left_untyped else (other_type, literal_type)
        return None

    def check_index(self, expr: Index) -> Type:
        """`base[index]`; str indexing yields uint8."""
        base_type = self.check_expr(expr.array)
        if base_type.kind == TypeKind.STR:
            index_type = self.check_expr(expr.index)
            if index_type != Type.INT:
                raise SemanticError(f"Index must be int, got {index_type}", expr.index)
            return Type.UINT8
        return self._check_indexable_and_index(expr.array, expr.index, base_type)

    def check_field(self, expr: Field) -> Type:
        key = self.module_set.qualified.get(expr.nid)
        if key is not None:  # `alias.NAME`: a constant of another module
            if key not in self.constants.decls:
                raise SemanticError(f"Reference to undeclared variable '{shown(key)}'", expr)
            self.facts.const_refs[expr.nid] = key
            return self.constants.type_of(key)
        enum = self._enum_named_by(expr.base)
        if enum is not None:  # `Enum.Member`
            members = self.decls.enums[enum].members
            if expr.name not in members:
                raise SemanticError(
                    f"Enum '{shown(enum)}' has no member '{expr.name}' (its members: {', '.join(members)})", expr)
            self.facts.enum_members[expr.nid] = (enum, members.index(expr.name))
            return Type(TypeKind.ENUM, enum_name=enum)
        return self._check_struct_and_field(expr.base, expr.name)

    def _enum_named_by(self, expr: Node) -> Optional[str]:
        """The key of the enum that `expr` names (`Enum`, or `alias.Enum`), unless a variable has the name."""
        if isinstance(expr, Variable) and not self.scopes.is_local(expr.name):
            key = self.scope.resolve(expr.name)
        elif isinstance(expr, Field):
            key = self.module_set.qualified.get(expr.nid)
        else:
            return None
        return key if key in self.decls.enums else None

    def _check_struct_and_field(self, base_expr: Node, field_name: str) -> Type:
        """Check base is a struct (auto-deref pointers) with field `field_name`."""
        base_type = self._check_expr_allowing_struct_literal(base_expr)
        if base_type.kind == TypeKind.POINTER and base_type.element_type.kind == TypeKind.STRUCT:
            base_type = base_type.element_type
        if base_type.kind != TypeKind.STRUCT:
            raise SemanticError(
                f"Cannot access field '{field_name}' on non-struct type {base_type}",
                base_expr,
            )
        struct_info = self.decls.structs[base_type.struct_name]
        if field_name not in struct_info.fields:
            raise SemanticError(
                f"Struct '{shown(base_type.struct_name)}' has no field '{field_name}'",
                base_expr,
            )
        if self._hidden(base_type.struct_name, field_name):
            raise SemanticError(
                f"Field '{field_name}' of '{shown(base_type.struct_name)}' is not visible outside the module that "
                f"defines the struct -- names starting with '_' are private to their own module", base_expr)
        return struct_info.fields[field_name]

    def _hidden(self, struct: str, name: str) -> bool:
        """Whether `name`, a field or method of `struct`, is private to another module: it starts with
        `_`, and the struct is declared in a different file from the one being checked."""
        declared_in = struct.rsplit('$', 1)[0] if '$' in struct else None
        return name.startswith('_') and declared_in != self.scope.module

    def check_struct_literal(self, expr: Call) -> Type:
        """`Name(args)`: positional struct literal; must be exhaustive."""
        name = self._struct_literal(expr)
        self._record_call(expr, name)
        struct_info = self.decls.structs[name]
        field_items = list(struct_info.fields.items())
        if expr.kwargs is not None:
            return self._check_named_struct_literal(expr, struct_info, field_items)
        private = [field for field, _ in field_items if self._hidden(name, field)]
        if private:  # it would have to give them values
            raise SemanticError(
                f"'{shown(name)}(...)' gives every field by position, but {', '.join(private)} "
                f"{'is' if len(private) == 1 else 'are'} private to the module that defines '{shown(name)}' -- "
                f"name the public fields instead (`{shown(name)}(field=value)`); private ones start as zero", expr)
        if len(expr.args) != len(field_items):
            field_names = ', '.join(name for name, _ in field_items)
            raise SemanticError(
                f"Struct literal for '{shown(name)}' expects "
                f"{len(field_items)} argument(s) (one per field, in "
                f"declaration order: {field_names}), got {len(expr.args)}",
                expr,
            )
        for i, (arg, (field_name, field_type)) in enumerate(zip(expr.args, field_items), start=1):
            arg_type = self._check_value_flowing_into_allowing_struct_literal(arg, field_type)
            if not self._types_compatible(arg_type, field_type):
                raise SemanticError(
                    f"Argument {i} to struct literal '{shown(name)}' "
                    f"(field '{field_name}') should be {field_type}, "
                    f"got {arg_type}",
                    arg,
                )
        result = Type(TypeKind.STRUCT, struct_name=name)
        self.facts.types[expr.nid] = result
        return result

    def _check_named_struct_literal(self, expr: Call, struct_info: StructInfo, field_items: list) -> Type:
        """`Name(f=v, ...)`: named struct literal; omitted fields are zero."""
        name = struct_info.name
        field_types = struct_info.fields
        valid_names = ', '.join(name for name, _ in field_items)
        seen = set()
        for field_name, value in expr.kwargs:
            if field_name not in field_types:
                raise SemanticError(
                    f"Struct literal for '{shown(name)}' has no field "
                    f"'{field_name}' -- valid fields are: {valid_names}",
                    expr,
                )
            if self._hidden(name, field_name):
                raise SemanticError(
                    f"Field '{field_name}' of '{shown(name)}' is not visible outside the module that defines the "
                    f"struct -- names starting with '_' are private to their own module", expr)
            if field_name in seen:
                raise SemanticError(
                    f"Field '{field_name}' specified more than once in "
                    f"struct literal for '{shown(name)}'",
                    expr,
                )
            seen.add(field_name)
            value_type = self._check_value_flowing_into_allowing_struct_literal(value, field_types[field_name])
            expected_type = field_types[field_name]
            if not self._types_compatible(value_type, expected_type):
                raise SemanticError(
                    f"Field '{field_name}' of struct literal '{shown(name)}' "
                    f"should be {expected_type}, got {value_type}",
                    value,
                )
        for field_name, field_type in field_types.items():
            if field_name not in seen:
                missing = self._without_zero_value(field_type)
                if missing is not None:
                    raise SemanticError(
                        f"Struct literal for '{shown(name)}' omits field '{field_name}', but {missing} has no "
                        f"zero value (only a sum type with a `none` variant does) -- give the field a value",
                        expr)
        result = Type(TypeKind.STRUCT, struct_name=name)
        self.facts.types[expr.nid] = result
        return result

    def _without_zero_value(self, t: Type, seen: Optional[set] = None) -> Optional[Type]:
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
                missing = self._without_zero_value(field_type, seen)
                if missing is not None:
                    return missing
        return None

    def _check_method_call(self, expr: Call) -> Type:
        """Resolve `receiver.name(args)` to its method: a call of the mangled function with the receiver
        (or its address, for a pointer receiver) first."""
        if expr.kwargs is not None:
            raise SemanticError(
                f"'{expr.name}(...)' uses named arguments, which are not "
                f"supported for method calls",
                expr,
            )
        receiver_type = self._check_expr_allowing_struct_literal(expr.receiver)
        has_methods = (TypeKind.STRUCT, TypeKind.ENUM)
        receiver_is_pointer = (receiver_type.kind == TypeKind.POINTER
                               and receiver_type.element_type.kind in has_methods)
        if receiver_is_pointer:
            # auto-deref
            receiver_type = receiver_type.element_type
        if receiver_type.kind not in has_methods:
            raise SemanticError(
                f"Cannot call method '{expr.name}' on a value of type "
                f"{receiver_type} -- methods are only defined on structs and enums",
                expr.receiver,
            )
        is_enum = receiver_type.kind == TypeKind.ENUM
        owner = receiver_type.enum_name if is_enum else receiver_type.struct_name
        key = (owner, expr.name)
        if key not in self.decls.methods:
            raise SemanticError(
                f"{'Enum' if is_enum else 'Struct'} '{shown(owner)}' has no method "
                f"'{expr.name}'",
                expr,
            )
        if self._hidden(owner, expr.name):
            raise SemanticError(
                f"Method '{expr.name}' of '{shown(owner)}' is not visible outside the module "
                f"that defines the {'enum' if is_enum else 'struct'} -- names starting with '_' are private to "
                f"their own module", expr)
        param_types, return_type, mangled_name = self.decls.methods[key]
        receiver = expr.receiver
        if key in self.decls.pointer_receivers and not receiver_is_pointer:
            # Pointer receiver: pass the receiver's address.
            is_place = isinstance(expr.receiver, (Variable, Field, Index)) or (
                    isinstance(expr.receiver, Unary) and expr.receiver.op == UnaryOp.DEREFERENCE)
            if not is_place or expr.receiver.nid in self.facts.enum_members:  # (`Color.Red` is a value)
                raise SemanticError(
                    f"Method '{expr.name}' on '{shown(owner)}' has a pointer receiver, so it "
                    f"needs an addressable receiver (a variable, field, index, or dereference), not a temporary",
                    expr.receiver,
                )
            if isinstance(receiver, Unary):
                receiver = receiver.operand  # &(*p) is p
            else:
                receiver = Unary(op=UnaryOp.ADDRESS_OF, operand=receiver,
                                 line=receiver.line, col=receiver.col, file=receiver.file)
                self.check_expr(receiver)
        if len(expr.args) != len(param_types):
            raise SemanticError(
                f"Method '{expr.name}' on '{shown(owner)}' "
                f"expects {len(param_types)} argument(s), got "
                f"{len(expr.args)}",
                expr,
            )
        for i, (arg, expected_type) in enumerate(zip(expr.args, param_types), start=1):
            actual_type = self._check_value_flowing_into_allowing_struct_literal(arg, expected_type)
            if not self._types_compatible(actual_type, expected_type):
                raise SemanticError(
                    f"Argument {i} to method '{expr.name}' on "
                    f"'{shown(owner)}' should be "
                    f"{expected_type}, got {actual_type}",
                    arg,
                )
        self.facts.calls[expr.nid] = (mangled_name, [receiver] + list(expr.args))
        self.facts.types[expr.nid] = return_type
        return return_type

    def _record_call(self, expr: Call, name: str) -> None:
        """Note the key a call refers to, unless it's the name as written."""
        if name != expr.name or expr.receiver is not None:
            self.facts.calls[expr.nid] = (name, list(expr.args))

    def check_call(self, expr: Call) -> Type:
        name = self._callee(expr)
        if name is None:
            # Receiver present (and not a module) means method call, checked first.
            return self._check_method_call(expr)
        if name in self.decls.structs:
            raise SemanticError(
                f"'{expr.name}(...)' is a struct literal, which is only "
                f"allowed as a variable's initializer, a plain "
                f"assignment's value, a direct function-call or "
                f"method-call argument, a method-call receiver, a "
                f"direct return value, an array literal's own "
                f"element, an assigned element or field, an assigned "
                f"field's base, a field-access "
                f"base, a binary operand, or a bare statement -- not "
                f"most other kinds of expressions (an Index/Slice "
                f"base, a Cast's own expression, ...); assign it to a "
                f"variable first if you need it in one of those "
                f"positions",
                expr,
            )
        if expr.kwargs is not None:
            raise SemanticError(
                f"'{expr.name}(...)' uses named arguments, which are "
                f"only supported for struct literals, not function calls",
                expr,
            )
        if name == 'print':
            return self.check_print_call(expr)
        if name == 'len':
            return self.check_len_call(expr)
        if name == 'append':
            return self.check_append_call(expr)
        if name == 'del':
            return self.check_del_call(expr)
        if name == 'bytes':
            return self.check_bytes_call(expr)
        if name == 'panic':
            return self.check_panic_call(expr)
        if name == 'format':
            return self.check_format_call(expr)
        if name in self.decls.enums:
            return self.check_enum_conversion(expr, name)
        visible = expr.nid in self.module_set.qualified or self.scope.resolve(expr.name) is not None
        if name not in self.decls.functions or not visible:  # another module's extern needs an import too
            raise SemanticError(f"Call to undeclared function '{shown(name)}'", expr)
        param_types, return_type = self.decls.functions[name]
        self._record_call(expr, name)

        if len(expr.args) != len(param_types):
            raise SemanticError(
                f"Function '{shown(name)}' expects {len(param_types)} "
                f"argument(s), got {len(expr.args)}",
                expr,
            )
        for i, (arg, expected_type) in enumerate(zip(expr.args, param_types), start=1):
            actual_type = self._check_value_flowing_into_allowing_struct_literal(arg, expected_type)
            if not self._types_compatible(actual_type, expected_type):
                raise SemanticError(
                    f"Argument {i} to '{shown(name)}' should be "
                    f"{expected_type}, got {actual_type}",
                    arg,
                )
        return return_type

    def check_print_call(self, expr: Call) -> Type:
        """`print(x)` for any non-void type."""
        if len(expr.args) != 1:
            raise SemanticError(
                f"'print' expects exactly 1 argument, got {len(expr.args)}",
                expr,
            )
        arg_type = self._check_expr_allowing_struct_literal(expr.args[0])
        if arg_type in (Type.VOID, Type.NEVER):
            raise SemanticError(
                "'print' cannot be called with the result of a function "
                "that has no declared return type -- there's no value there to print",
                expr.args[0],
            )
        return Type.VOID

    def check_panic_call(self, expr: Call) -> Type:
        """`panic(message)`: report a str with the call's position and abort."""
        if len(expr.args) != 1:
            raise SemanticError(f"'panic' expects exactly 1 argument, got {len(expr.args)}", expr)
        arg_type = self.check_expr(expr.args[0])
        if arg_type != Type.STR:
            raise SemanticError(f"'panic' expects a str, got {arg_type}", expr.args[0])
        return Type.NEVER

    def check_format_call(self, expr: Call) -> Type:
        """`format(template, args...)`: a str, the template with each `{}` replaced by the next
        argument as print shows it. The template is known here, so it is checked against them."""
        if not expr.args:
            raise SemanticError("'format' expects a template, then a value for each '{}' in it", expr)
        template = expr.args[0]
        if self.check_expr(template) != Type.STR:
            raise SemanticError(
                f"'format' expects a str template first, got {self.facts.types[template.nid]}", template)
        try:
            text = self.constants.evaluate(template)
        except SemanticError:
            raise SemanticError(
                "'format' needs its template as a string literal or a constant, so that it can be checked "
                "against the values", template) from None
        try:
            holes = len(typed.format_pieces(text)) - 1
        except ValueError as problem:
            raise SemanticError(f"In this 'format' template: {problem}", template) from None
        values = expr.args[1:]
        if holes != len(values):
            raise SemanticError(
                f"This 'format' template has {holes} '{{}}' placeholder{'' if holes == 1 else 's'}, "
                f"but {len(values)} value{' was' if len(values) == 1 else 's were'} given", expr)
        for value in values:
            if self._check_expr_allowing_struct_literal(value) in (Type.VOID, Type.NEVER):
                raise SemanticError(
                    "'format' cannot show the result of a function that has no declared return type -- there's "
                    "no value there", value)
        self.facts.formats[expr.nid] = text
        return Type.STR

    def check_enum_conversion(self, expr: Call, enum: str) -> Type:
        """`Enum(n)`: the member whose value is the integer `n`. Checked when it runs (a panic if there
        is none), or here when `n` is a literal."""
        members = self.decls.enums[enum].members
        if len(expr.args) != 1:
            raise SemanticError(
                f"'{expr.name}(...)' converts one integer to the enum {shown(enum)}, got {len(expr.args)} arguments",
                expr)
        arg_type = self.check_expr(expr.args[0])
        if arg_type not in _INTEGER_TYPES:
            raise SemanticError(
                f"'{expr.name}(...)' converts an integer to the enum {shown(enum)}, got {arg_type}", expr.args[0])
        literal = self._as_folded_int_literal(expr.args[0])
        if literal is not None and not 0 <= literal < len(members):
            raise SemanticError(
                f"{literal} is not a member of {shown(enum)} (its members' values are 0 to {len(members) - 1})",
                expr.args[0]
            )
        self._record_call(expr, enum)
        return Type(TypeKind.ENUM, enum_name=enum)

    def check_len_call(self, expr: Call) -> Type:
        """`len(x)` for arrays, slices, str, and dicts; `len(Enum)` is an enum's number of members."""
        if len(expr.args) != 1:
            raise SemanticError(
                f"'len' expects exactly 1 argument, got {len(expr.args)}",
                expr,
            )
        enum = self._enum_named_by(expr.args[0])
        if enum is not None:
            self.facts.enum_lens[expr.nid] = len(self.decls.enums[enum].members)
            return Type.INT
        arg_type = self.check_expr(expr.args[0])
        if arg_type.kind not in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STR, TypeKind.DICT):
            raise SemanticError(
                f"'len' requires an array, slice, str, or dict argument, got {arg_type}",
                expr.args[0],
            )
        return Type.INT

    def check_append_call(self, expr: Call) -> Type:
        """`append(s, v)` returns a new slice."""
        if len(expr.args) != 2:
            raise SemanticError(
                f"'append' expects exactly 2 arguments, got {len(expr.args)}",
                expr,
            )
        slice_arg, value_arg = expr.args
        slice_type = self.check_expr(slice_arg)
        if slice_type.kind != TypeKind.SLICE:
            raise SemanticError(
                f"'append' requires a slice as its first argument, "
                f"got {slice_type}",
                slice_arg,
            )
        value_type = self._check_value_flowing_into_allowing_struct_literal(value_arg, slice_type.element_type)
        if not self._types_compatible(value_type, slice_type.element_type):
            raise SemanticError(
                f"'append' cannot append a value of type {value_type} "
                f"to a {slice_type} (element type "
                f"{slice_type.element_type})",
                value_arg,
            )
        return slice_type

    def check_bytes_call(self, expr: Call) -> Type:
        """`bytes(s)`: a new []byte copy of str s."""
        if len(expr.args) != 1 or expr.kwargs:
            raise SemanticError(f"bytes() takes exactly one argument, got {len(expr.args)}", expr)
        arg_type = self.check_expr(expr.args[0])
        if arg_type != Type.STR:
            raise SemanticError(f"bytes() takes a str, got {arg_type}", expr)
        return _BYTE_SLICE

    def check_del_call(self, expr: Call) -> Type:
        """`del(d, key)` mutates d in place."""
        if len(expr.args) != 2:
            raise SemanticError(
                f"'del' expects exactly 2 arguments, got {len(expr.args)}",
                expr,
            )
        dict_arg, key_arg = expr.args
        dict_type = self.check_expr(dict_arg)
        if dict_type.kind != TypeKind.DICT:
            raise SemanticError(
                f"'del' requires a dict as its first argument, got {dict_type}",
                dict_arg,
            )
        key_type = self._check_value_flowing_into(key_arg, dict_type.key_type)
        if not self._types_compatible(key_type, dict_type.key_type):
            raise SemanticError(
                f"'del' cannot look up a key of type {key_type} in a "
                f"{dict_type} (key type {dict_type.key_type})",
                key_arg,
            )
        return Type.VOID

    def check_constant(self, expr: Constant) -> Type:
        if expr.value > 2**63 - 1:
            raise SemanticError(f"Integer literal {int(expr.value)} is out of range for int (64-bit)", expr)
        return Type.INT

    def check_variable(self, expr: Variable) -> Type:
        t, self.facts.decls[expr.nid] = self._resolve(expr.name, expr)
        if self.facts.decls[expr.nid] is None:
            self.facts.const_refs[expr.nid] = self._const_key(expr.name)
        return t

    def check_is_check(self, expr: IsCheck) -> Type:
        """`NAME is T`, `EXPR is T`, or `EXPR is T as NAME` (NAME already declared, by the statement):
        T is a variant of the subject's sum, or a member of its enum. Only a NAME can be narrowed."""
        if expr.binds and expr.nid not in self._bound:
            if expr.nid not in self._bindable:
                raise SemanticError(
                    "'as NAME' can bind only in an `if` or `elif` condition, where the check is the whole "
                    "condition or one of the checks joined by 'and'", expr)
            self._declare_is_binding(expr)
        if expr.variable_name is None:  # `EXPR is T`: a test of the expression's value
            variable_type, narrowed = self.check_expr(expr.subject), False
        else:
            variable_type, self.facts.decls[expr.nid] = self._resolve(expr.variable_name, expr)
            narrowed = self.scopes.is_narrowed(self.facts.decls[expr.nid])
        if variable_type.kind == TypeKind.ENUM:  # `NAME is Member`: an equality test; nothing is narrowed
            members = self.decls.enums[variable_type.enum_name].members
            member = expr.type_name
            if isinstance(member, QualifiedTypeExpr) and self.scope.resolve(member.module) == variable_type.enum_name:
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
        if narrowed_type not in sum_type_info.variants:
            raise SemanticError(
                f"'{expr.type_name}' is not one of {variable_type}'s own "
                f"declared variants ({', '.join(str(v) for v in sum_type_info.variants)})",
                expr,
            )
        self.facts.narrowed[expr.nid] = narrowed_type
        return Type.BOOL

    def _root_variable_of(self, expr: Node) -> Optional[Variable]:
        """Variable under a Field/Index/Slice chain, if any."""
        while isinstance(expr, (Field, Index, Slice)):
            expr = expr.base if isinstance(expr, Field) else expr.array
        return expr if isinstance(expr, Variable) else None

    def check_unary(self, expr: Unary) -> Type:
        if (expr.op == UnaryOp.NEGATE and isinstance(expr.operand, Constant)
                and expr.operand.value == 2**63):
            self.facts.types[expr.operand.nid] = Type.INT  # -2**63 is int's minimum
            return Type.INT
        operand_type = self._check_expr_allowing_struct_literal(expr.operand)
        if expr.op in (UnaryOp.NEGATE, UnaryOp.COMPLEMENT):
            if operand_type not in _INTEGER_TYPES:
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
                    and self._const_key(expr.operand.name) is not None):
                raise SemanticError(f"Cannot take the address of constant '{expr.operand.name}'", expr)
            if expr.operand.nid in self.facts.const_refs:
                raise SemanticError(
                    f"Cannot take the address of constant '{shown(self.facts.const_refs[expr.operand.nid])}'", expr)
            if expr.operand.nid in self.facts.enum_members:  # `&Color.Red`: a value, with nowhere it lives
                raise SemanticError(
                    f"Cannot take the address of the enum member '{operand_type}.{expr.operand.name}'", expr)
            # Only variables, struct literals, and chains rooted in a variable.
            is_struct_literal = self._struct_literal(expr.operand) is not None
            root_variable = self._root_variable_of(expr.operand) if isinstance(expr.operand, (Field, Index)) else None
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
            if source_type not in (Type.UINT8, _BYTE_SLICE) and source_type.kind != TypeKind.ENUM:
                raise SemanticError(f"str(...) takes a byte, a []byte, or an enum (its member's name), "
                                    f"got {source_type}", expr)
            return Type.STR
        if target_type == Type.INT64 and self._as_folded_int_literal(expr.expr) is not None:
            self._record_literal_type(expr.expr, Type.INT64)
            source_type = Type.INT64
        else:
            source_type = self.check_expr(expr.expr)
        if target_type not in _INTEGER_TYPES or (source_type not in _INTEGER_TYPES
                                                 and source_type.kind != TypeKind.ENUM):
            raise SemanticError(
                f"Cannot cast {source_type} to {target_type} -- casting "
                f"is only supported between int, int8, uint8, and int32, "
                f"and from an enum to one of them, right now",
                expr,
            )
        return target_type

    def _is_comparable_type(self, t: Type) -> bool:
        """Whether `==` is defined for `t`."""
        if t.kind == TypeKind.ARRAY:
            return self._is_comparable_type(t.element_type)
        if t.kind == TypeKind.STRUCT:
            struct_info = self.decls.structs[t.struct_name]
            return all(self._is_comparable_type(field_type) for field_type in struct_info.fields.values())
        if t.kind in (TypeKind.SLICE, TypeKind.SUM, TypeKind.DICT):
            return False
        return True  # integers, bool, str, pointers

    def check_binary(self, expr: Binary) -> Type:
        if expr.op == BinaryOp.IN and self._enum_named_by(expr.right) is not None:
            # `n in Enum`: whether the integer `n` is a member's value (so `Enum(n)` wouldn't panic).
            enum = self._enum_named_by(expr.right)
            left_type = self.check_expr(expr.left)
            if left_type not in _INTEGER_TYPES:
                raise SemanticError(
                    f"'in' with the enum {shown(enum)} on its right tests an integer, got {left_type}", expr.left)
            self.facts.enum_ins[expr.nid] = enum
            return Type.BOOL
        with_literal = self._operand_types_with_an_untyped_array(expr)
        if with_literal is not None:
            left_type, right_type = with_literal
        else:
            left_type = self._check_expr_allowing_struct_literal(expr.left)
            mark = self.scopes.mark()
            if expr.op in _LOGICAL_OPS:  # the right side runs only when the left was true (`and`) or false (`or`)
                self.scopes.apply(self.scopes.when(expr.left)[0 if expr.op == BinaryOp.AND else 1])
            right_type = self._check_expr_allowing_struct_literal(expr.right)
            self.scopes.end_region(mark)
        op = expr.op
        if op not in _LOGICAL_OPS and op != BinaryOp.IN:
            left_type, right_type = self._literal_operand_types(expr.left, left_type, expr.right, right_type)

        if op == BinaryOp.ADD:
            # Same-type integers add; str + str concatenates.
            if left_type in _INTEGER_TYPES and left_type == right_type:
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

        if op in _ORDERING_OPS:
            # Same-type integers; or two strs, ordered byte by byte (as strings.ht's compare orders them).
            if (left_type in _INTEGER_TYPES or left_type == Type.STR) and left_type == right_type:
                return Type.BOOL
            raise SemanticError(
                f"'{op.symbol()}' requires two operands of the same integer type "
                f"(int, int8, uint8, or int32) or two str operands, "
                f"got {left_type} and {right_type}",
                expr,
            )

        if op in _EQUALITY_OPS:
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
                        f"when the elements are (or contain) a slice, "
                        f"sum type, or dict, none of which has '==' defined yet",
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
                        f"another struct or an array field) is a slice, "
                        f"sum type, or dict, none of which has '==' defined yet",
                        expr,
                    )
                return Type.BOOL

            # Slice, sum, and dict equality is undefined.
            if (
                    left_type.kind in (TypeKind.SLICE, TypeKind.VOID, TypeKind.NONE, TypeKind.SUM, TypeKind.DICT)
                    or right_type.kind in (TypeKind.SLICE, TypeKind.VOID, TypeKind.NONE, TypeKind.SUM, TypeKind.DICT)
            ):
                raise SemanticError(
                    f"'{op.symbol()}' does not support slice, void, sum "
                    f"type, dict, or none operands, except comparing a "
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
                if self._as_folded_int_literal(expr.left) is not None and not self._untyped_array_literal(expr.right):
                    left_type = self._check_value_flowing_into(expr.left, wanted)
            if right_type.kind == TypeKind.DICT:
                if not self._types_compatible(left_type, right_type.key_type):
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
                        f"when the elements are (or contain) a slice, "
                        f"sum type, or dict",
                        expr.right,
                    )
                if not self._types_compatible(left_type, element_type):
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

        if op in _LOGICAL_OPS:
            self._require_type(left_type, Type.BOOL, op, expr)
            self._require_type(right_type, Type.BOOL, op, expr)
            return Type.BOOL

        raise SemanticError(f"No semantic rule for binary operator: {op}", expr)

    def _require_type(self, actual: Type, expected: Type, op, node: Optional[Node] = None) -> None:
        if actual != expected:
            raise SemanticError(
                f"'{op.symbol()}' requires {expected} operands, got {actual}",
                node,
            )

    def _literal_operand_types(self, left: Node, left_type: Type, right: Node, right_type: Type) -> tuple:
        """Both operand types, once an integer literal (or its negation) beside an operand of a
        narrower integer type has taken that type: `fd < 0`, `1 + a`. It must be in range."""
        if right_type in _NARROW_INT_RANGES and self._as_folded_int_literal(left) is not None:
            left_type = self._check_value_flowing_into(left, right_type)
        elif left_type in _NARROW_INT_RANGES and self._as_folded_int_literal(right) is not None:
            right_type = self._check_value_flowing_into(right, left_type)
        return left_type, right_type

    def _require_same_integer_type(self, left_type: Type, right_type: Type, op, node: Optional[Node] = None) -> Type:
        """Require identical integer types."""
        if left_type not in _INTEGER_TYPES or left_type != right_type:
            raise SemanticError(
                f"'{op.symbol()}' requires two operands of the same "
                f"integer type (int, int8, uint8, or int32), got "
                f"{left_type} and {right_type}",
                node,
            )
        return left_type


# Entry points

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
