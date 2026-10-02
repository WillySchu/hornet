"""Semantic analysis: name resolution, type checking, and control-flow checks; its result is the
typed tree (typed_ast.py), which every later stage consumes.

Strict typing: no implicit conversions; integer operands must match exactly. Blocks
scope lexically and may shadow. Non-void functions must return on all paths.

The parser's tree is never changed: checking records what it learns in Facts, keyed by node number,
and _TypedTreeBuilder builds each function's typed tree from those facts at the end of analyze().
"""

import argparse
import dataclasses
from typing import Dict, List, Optional, Set, Tuple

from diagnostics import CompileError
from lexer import lex
from typesys import StructInfo, SumTypeInfo, Type, TypeKind
from folding import fold_binary_op, fold_cast, fold_unary_op
import parser as syntax
import typed_ast as typed
from scopes import build_module_set, display_name as shown
from symbols import SymbolTable
from parser import (
    ConstDecl,
    ArrayLiteral,
    ArrayTypeExpr,
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
    DerefAssign,
    DictLiteral,
    DictTypeExpr,
    ExprStmt,
    ExternFunctionDecl,
    Field,
    QualifiedTypeExpr,
    FieldAssign,
    For,
    ForIn,
    Function,
    MethodDef,
    Param,
    If,
    Index,
    IndexAssign,
    IntrinsicDecl,
    IsCheck,
    Match,
    Node,
    NoneLiteral,
    Parser,
    PointerTypeExpr,
    Program,
    Return,
    Slice,
    SliceTypeExpr,
    StringLiteral,
    StructDef,
    SumTypeDef,
    TypeAlias,
    Unary,
    UnaryOp,
    VarDecl,
    Variable,
    While,
)


_TYPE_NAMES = {
    'int': Type.INT,
    'int8': Type.INT8,
    'uint8': Type.UINT8,
    # alias, not a distinct kind
    'byte': Type.UINT8,
    'int64': Type.INT64,
    'int32': Type.INT32,
    'bool': Type.BOOL,
    'str': Type.STR,
}


# The value of each `[EXPR]T` size that isn't a plain number, by the ArrayTypeExpr's node number;
# recorded by SemanticAnalyzer._resolve_array_sizes before any type is resolved.
_array_sizes: Dict[int, int] = {}


def type_from_name(
    type_expr,
    structs: Dict[str, StructInfo],
    aliases: Dict[str, Type],
    node: Optional[Node] = None,
    sum_types: Dict[str, SumTypeInfo] = None,
    resolve=None,
) -> Type:
    """Resolve a parsed type expression to a Type. `resolve` maps a name as written in its file (or an
    `alias.Name`) to its declaration's key (see scopes.py)."""
    if isinstance(type_expr, ArrayTypeExpr):
        element = type_from_name(type_expr.element_type, structs, aliases, node, sum_types, resolve)
        size = type_expr.size if isinstance(type_expr.size, int) else _array_sizes[type_expr.nid]
        return Type(TypeKind.ARRAY, element_type=element, size=size)
    if isinstance(type_expr, SliceTypeExpr):
        element = type_from_name(type_expr.element_type, structs, aliases, node, sum_types, resolve)
        return Type(TypeKind.SLICE, element_type=element)
    if isinstance(type_expr, PointerTypeExpr):
        # Pointer-to-pointer is rejected for now.
        pointee = type_from_name(type_expr.pointee_type, structs, aliases, node, sum_types, resolve)
        if pointee.kind == TypeKind.POINTER:
            raise SemanticError(
                "Pointer-to-pointer types aren't supported yet -- "
                "planned for later, once single-level pointers are proven out",
                node,
            )
        return Type(TypeKind.POINTER, element_type=pointee)
    if isinstance(type_expr, DictTypeExpr):
        key_type = type_from_name(type_expr.key_type, structs, aliases, node, sum_types, resolve)
        value_type = type_from_name(type_expr.value_type, structs, aliases, node, sum_types, resolve)
        if key_type not in _VALID_DICT_KEY_TYPES:
            raise SemanticError(
                f"'{key_type}' can't be a dict's own key type -- only "
                f"int, int8, uint8, int32, bool, and str are supported "
                f"as dict keys right now",
                node,
            )
        return Type(TypeKind.DICT, key_type=key_type, element_type=value_type)
    if isinstance(type_expr, QualifiedTypeExpr) and resolve is not None:
        type_expr = resolve(type_expr)
    if type_expr in _TYPE_NAMES:
        return _TYPE_NAMES[type_expr]
    if type_expr == 'none':  # only a sum type's variant, or the type an `is` check narrows to
        return Type.NONE
    if resolve is not None:
        type_expr = resolve(type_expr)
    if type_expr in aliases:
        return aliases[type_expr]
    if type_expr in structs:
        return Type(TypeKind.STRUCT, struct_name=type_expr)
    if sum_types is not None and type_expr in sum_types:
        return Type(TypeKind.SUM, sum_type_name=type_expr)
    raise SemanticError(f"Unknown type '{type_expr}'", node)


def always_returns(statements: List[Node]) -> bool:
    """Whether every path through `statements` returns."""
    for stmt in statements:
        if isinstance(stmt, Return):
            return True
        if isinstance(stmt, Match) and _match_always_returns(stmt):  # exhaustiveness already verified
            return True
        if isinstance(stmt, If):
            if stmt.else_body is not None and always_returns(stmt.then_body) and always_returns(stmt.else_body):
                return True
        if isinstance(stmt, While):
            # `while true` without a reachable break never falls through.
            is_infinite = isinstance(stmt.condition, BoolLiteral) and stmt.condition.value is True
            if is_infinite and not contains_reachable_break(stmt.body):
                return True
    return False


def _match_always_returns(stmt: 'Match') -> bool:
    """Whether every arm of a match returns."""
    if not all(always_returns(body) for _, body in stmt.arms):
        return False
    return stmt.else_body is None or always_returns(stmt.else_body)


def contains_reachable_break(statements: List[Node]) -> bool:
    """Whether `statements` contain a break for this loop (not nested loops)."""
    for stmt in statements:
        if isinstance(stmt, Break):
            return True
        if isinstance(stmt, If):
            if contains_reachable_break(stmt.then_body):
                return True
            if stmt.else_body is not None and contains_reachable_break(stmt.else_body):
                return True
        if isinstance(stmt, Match):
            if any(contains_reachable_break(body) for _, body in stmt.arms):
                return True
            if stmt.else_body is not None and contains_reachable_break(stmt.else_body):
                return True
    return False


def mangle_method_name(struct_name: str, method_name: str) -> str:
    """`Struct.method`; '.' can't appear in identifiers, so no collisions."""
    return f"{struct_name}.{method_name}"


def _method_function(sd: StructDef, md: MethodDef) -> Function:
    """A method as the function it compiles to: its mangled name, and the receiver as the first
    parameter (`*S` for a pointer receiver). New nodes; the program's own are left as they are."""
    receiver_type = sd.name
    if md.receiver_is_pointer:
        receiver_type = PointerTypeExpr(pointee_type=sd.name, line=md.line, col=md.col, file=md.file)
    receiver = Param(name=md.receiver_name, type=receiver_type, line=md.line, col=md.col, file=md.file)
    return Function(name=mangle_method_name(sd.name, md.name), return_type=md.return_type,
                    params=[receiver] + md.params, body=md.body, line=md.line, col=md.col, file=md.file)


# Builtins; see check_call.
_BUILTIN_FUNCTION_NAMES = {'print', 'len', 'append', 'del', 'bytes'}
_BYTE_SLICE = Type(TypeKind.SLICE, element_type=Type.UINT8)


# Errors

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


# Function bodies checked before giving up.
MAX_ERRORS = 20


# Analyzer

# ADD is excluded: it also concatenates strings.
_INT_ONLY_BINARY_OPS = {
    BinaryOp.SUBTRACT, BinaryOp.MULTIPLY, BinaryOp.DIVIDE, BinaryOp.MODULO,
    BinaryOp.BITWISE_AND, BinaryOp.BITWISE_OR, BinaryOp.BITWISE_XOR,
    BinaryOp.SHIFT_LEFT, BinaryOp.SHIFT_RIGHT,
}
_ORDERING_OPS = {BinaryOp.LESS_THAN, BinaryOp.GREATER_THAN,
                  BinaryOp.LESS_THAN_OR_EQUAL, BinaryOp.GREATER_THAN_OR_EQUAL}
_EQUALITY_OPS = {BinaryOp.EQUAL, BinaryOp.NOT_EQUAL}
_LOGICAL_OPS = {BinaryOp.AND, BinaryOp.OR}

_INTEGER_TYPES = {Type.INT, Type.INT8, Type.UINT8, Type.INT64, Type.INT32}
_VALID_DICT_KEY_TYPES = _INTEGER_TYPES | {Type.BOOL, Type.STR}

# Literal ranges for narrow integer types.
_NARROW_INT_RANGES = {
    Type.INT8: (-128, 127),
    Type.UINT8: (0, 255),
    Type.INT32: (-2**31, 2**31 - 1),
}


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


@dataclasses.dataclass
class Facts:
    """What checking learned, keyed by parser-node number (Node.nid); the parser's nodes are never
    changed. _TypedTreeBuilder builds the typed tree from these."""
    types: dict = dataclasses.field(default_factory=dict)  # expression, VarDecl, Param -> Type
    decls: dict = dataclasses.field(default_factory=dict)  # Variable, Assign, IsCheck -> Symbol.id (None: a constant)
    symbols: dict = dataclasses.field(default_factory=dict)  # VarDecl, Param -> Symbol
    for_symbols: dict = dataclasses.field(default_factory=dict)  # ForIn -> [Symbol] per binding
    bindings: dict = dataclasses.field(default_factory=dict)  # IsCheck with `as NAME` -> synthetic VarDecl for NAME
    narrowed: dict = dataclasses.field(default_factory=dict)  # IsCheck -> the variant it tests for
    boxed: dict = dataclasses.field(default_factory=dict)  # Unary `&Variant(...)` -> the sum it boxes into
    returns: dict = dataclasses.field(default_factory=dict)  # Function -> return Type
    calls: dict = dataclasses.field(default_factory=dict)  # Call -> (callee's key, args) unless name(args) as written
    const_refs: dict = dataclasses.field(default_factory=dict)  # Variable, or `alias.NAME` Field -> constant's key


class SemanticAnalyzer:
    """Type- and scope-checks a Program."""

    def __init__(self):
        self.scopes: List[Dict[str, Tuple[Type, object]]] = []  # name -> (type, decl id)
        self.loop_depth = 0  # enclosing loop count
        self.functions: Dict[str, tuple] = {}  # name -> (param types, return type)
        self.structs: Dict[str, StructInfo] = {}
        self.methods: Dict[Tuple[str, str], Tuple[List[Type], Type, str]] = {}  # (struct, method) -> (param types, return type, mangled name)
        self.pointer_receivers: set = set()  # (struct, method) with a `*receiver`
        self.type_aliases: Dict[str, Type] = {}
        self.sum_types: Dict[str, SumTypeInfo] = {}
        self._narrowed_names: set = set()  # currently narrowed variable names

    def analyze(self, entry: Program, modules: Optional[dict] = None) -> "typed.Program":
        """Check the entry file and the modules it imports (discover_modules's result) and return the
        typed tree; no parser tree is changed."""
        self.symbols = SymbolTable()
        self.facts = Facts()
        # Each declaration under its key, with its file's scope (see scopes.py).
        program = build_module_set(entry, modules or {})
        self.module_set = program
        self.scope = program.files[0][1]
        # Each method is checked and compiled as a function of its own (see _method_function).
        methods = [(_method_function(sd, md), sd) for sd in program.structs for md in sd.methods]
        for fn, sd in methods:
            program.scope_of[fn.nid] = program.scope_of[sd.nid]
        self.all_functions = list(program.functions) + [fn for fn, _ in methods]
        # Order matters: 0. constant array sizes become literals before any type is resolved.
        self._collect_consts(program)
        self._resolve_array_sizes(program)

        # 1. Reserve struct names so aliases can target them.
        struct_registry = self._reserve_struct_names(program.structs)

        # 2. Resolve aliases before struct fields, which may use them.
        self.type_aliases = self._collect_type_aliases(program.type_aliases, struct_registry)

        # 3. Resolve struct fields; sum-type names are reserved first so fields can name them.
        reserved_sums = {std.name: None for std in program.sum_types}
        self.structs = self._resolve_struct_fields(program.structs, struct_registry, reserved_sums)

        # 3.5. Sum types need resolved structs. Then no type may contain itself by value.
        self.sum_types = self._resolve_sum_types(program.sum_types, self.structs)
        self._check_value_containment(program)

        # 3.6. Methods, after struct resolution.
        self.methods = self._collect_methods(program)

        # 4. All signatures before any body, so order doesn't matter.
        self.functions = {}
        self.intrinsic_original_names = {}  # mangled name -> original_name
        for fn in self.all_functions:
            self._enter(fn)
            if fn.name in _BUILTIN_FUNCTION_NAMES:
                raise SemanticError(
                    f"'{shown(fn.name)}' is a builtin and can't be redefined as "
                    f"a function",
                    fn,
                )
            if fn.name in self.structs:
                raise SemanticError(
                    f"Function '{shown(fn.name)}' collides with a struct of the "
                    f"same name -- struct and function names share one "
                    f"namespace and can never be the same, since "
                    f"'{shown(fn.name)}(...)' would otherwise be ambiguous "
                    f"between a call and a struct literal",
                    fn,
                )
            if fn.name in self.type_aliases:
                raise SemanticError(
                    f"Function '{shown(fn.name)}' collides with a type alias "
                    f"of the same name -- function and type-alias "
                    f"names share one namespace and can never be the "
                    f"same",
                    fn,
                )
            if fn.name in self.sum_types:
                raise SemanticError(
                    f"Function '{shown(fn.name)}' collides with a sum type "
                    f"of the same name -- function and sum-type names "
                    f"share one namespace and can never be the same",
                    fn,
                )
            if fn.name in self.functions:
                raise SemanticError(f"Function '{shown(fn.name)}' is already declared", fn)
            param_types = [self._type(p.type, p) for p in fn.params]
            return_type = Type.VOID if fn.return_type is None else self._type(fn.return_type, fn)
            self.functions[fn.name] = (param_types, return_type)

        # 4.5. Externs share the function registry.
        self.extern_names = {ext.name for ext in program.extern_functions}
        for ext in program.extern_functions:
            self._enter(ext)
            self.check_extern_function_decl(ext)

        # 4.6. Intrinsics share the function registry.
        for ic in program.intrinsics:
            self._enter(ic)
            self.check_intrinsic_decl(ic)

        # 4.7. Constant values, in dependency order.
        for name in self.const_decls:
            self._const_value(name)
        self._check_const_name_collisions(program)

        # 5. Check bodies, collecting at most one error per function.
        errors: List[SemanticError] = []
        for fn in self.all_functions:
            try:
                self.analyze_function(fn)
                if fn.name == 'main':
                    _check_main_signature(fn, *self.functions['main'])
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

        # 7. The typed tree: what later stages consume.
        return _TypedTreeBuilder(self).program(self.all_functions)

    # -- names across modules

    def _enter(self, decl: Node) -> None:
        """Check `decl` (a top-level declaration, under its key) in its own file's scope."""
        self.scope = self.module_set.scope_of[decl.nid]

    def _resolve_type_name(self, name):
        """A type name (or `alias.Name`) as written in the current file -> its declaration's key."""
        if isinstance(name, QualifiedTypeExpr):
            return self.module_set.qualified[name.nid]
        return (self.scope.resolve(name) or name) if isinstance(name, str) else name

    def _type(self, type_expr, node: Node, sums: bool = True) -> Type:
        return type_from_name(type_expr, self.structs, self.type_aliases, node, self.sum_types if sums else None,
                              resolve=self._resolve_type_name)

    def _const_key(self, name: str) -> Optional[str]:
        """The key of the constant a bare name refers to in the current file, if it names one."""
        key = self.scope.consts.get(name)
        return key if key in getattr(self, 'const_decls', {}) else None

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
            if name in self.structs:
                return name
        return None

    # -- constants

    def _collect_consts(self, program: Program) -> None:
        self.const_decls: Dict[str, ConstDecl] = {}
        self.consts: Dict[str, Tuple[Type, object]] = {}  # name -> (type, value)
        self._const_in_progress: set = set()
        for cd in program.consts:
            if cd.name in self.const_decls:
                raise SemanticError(f"Constant '{shown(cd.name)}' is already declared", cd)
            self.const_decls[cd.name] = cd

    def _resolve_array_sizes(self, program: Program) -> None:
        """Record each `[EXPR]T` size's value (in _array_sizes): a positive integer computed from
        literals and constants only (whose types must therefore be builtin)."""
        for file_program, scope in program.files:
            self.scope = scope
            self._resolve_array_sizes_in(file_program)
        self.scope = program.files[0][1]

    def _resolve_array_sizes_in(self, program: Program) -> None:
        seen = set()
        stack = [program]
        while stack:
            node = stack.pop()
            if node.nid in seen:
                continue
            seen.add(node.nid)
            if isinstance(node, ArrayTypeExpr) and not isinstance(node.size, int):
                _array_sizes[node.nid] = self._array_size_value(node.size)
            if dataclasses.is_dataclass(node) and not isinstance(node, type):
                for f in dataclasses.fields(node):
                    value = getattr(node, f.name)
                    for v in value if isinstance(value, (list, tuple)) else [value]:
                        if isinstance(v, (Node, Program)):
                            stack.append(v)
                        elif isinstance(v, tuple):
                            stack.extend(x for x in v if isinstance(x, Node))

    def _array_size_value(self, expr: Node) -> int:
        stack = [expr]
        while stack:
            node = stack.pop()
            if node.nid in self.module_set.qualified and isinstance(node, Field):
                continue  # `alias.NAME`: checked as a constant below
            if isinstance(node, Call):
                raise SemanticError("Array size must be a constant expression, not a call", node)
            if isinstance(node, Variable) and self._const_key(node.name) is None:
                raise SemanticError(
                    f"Array size must be a constant expression, but '{node.name}' isn't a constant", node)
            if dataclasses.is_dataclass(node):
                stack.extend(v for f in dataclasses.fields(node) if isinstance(v := getattr(node, f.name), Node))
        saved_scopes, self.scopes = self.scopes, [{}]
        try:
            size_type = self.check_expr(expr)
            if size_type not in _INTEGER_TYPES:
                raise SemanticError(f"Array size must be an integer, got {size_type}", expr)
            value = self._const_eval(expr)
        finally:
            self.scopes = saved_scopes
        if value <= 0:
            raise SemanticError(f"Array size must be positive, got {value}", expr)
        return value

    def _check_const_name_collisions(self, program: Program) -> None:
        for name, cd in self.const_decls.items():
            for kind, table in (('function', self.functions), ('struct', self.structs),
                                ('type alias', self.type_aliases), ('sum type', self.sum_types)):
                if name in table:
                    raise SemanticError(f"Constant '{shown(name)}' collides with a {kind} of the same name", cd)
            if name in _BUILTIN_FUNCTION_NAMES:
                raise SemanticError(f"'{shown(name)}' is a builtin and can't be used as a constant name", cd)

    def _const_value(self, name: str) -> Tuple[Type, object]:
        """(type, value) of constant `name`, evaluating it (and what it depends on) the first time."""
        if name in self.consts:
            return self.consts[name]
        cd = self.const_decls[name]
        if name in self._const_in_progress:
            raise SemanticError(f"Constant '{shown(name)}' is defined in terms of itself", cd)
        self._const_in_progress.add(name)
        saved_scope = self.scope
        self._enter(cd)
        const_type = self._type(cd.const_type, cd)
        if const_type not in _INTEGER_TYPES and const_type not in (Type.BOOL, Type.STR):
            raise SemanticError(
                f"Constant '{shown(name)}' has type {const_type} -- constants must be an integer type, bool, or str", cd)
        saved_scopes, self.scopes = self.scopes, [{}]
        try:
            value_type = self._check_value_flowing_into(cd.value, const_type)
            if not self._types_compatible(value_type, const_type):
                raise SemanticError(
                    f"Constant '{shown(name)}' is declared {const_type} but its value has type {value_type}", cd)
            value = self._const_eval(cd.value)
        finally:
            self.scopes = saved_scopes
            self.scope = saved_scope
        self._const_in_progress.discard(name)
        self.consts[name] = (const_type, value)
        return self.consts[name]

    def _const_eval(self, expr: Node):
        """Value of an already type-checked constant expression."""
        t = self.facts.types.get(expr.nid)
        if isinstance(expr, (Constant, ByteLiteral)) and isinstance(expr.value, int):
            return expr.value
        if isinstance(expr, (BoolLiteral, StringLiteral)):
            return expr.value
        if isinstance(expr, Variable) and self._const_key(expr.name) is not None:
            return self._const_value(self._const_key(expr.name))[1]
        if isinstance(expr, Field) and self.module_set.qualified.get(expr.nid) in self.const_decls:
            return self._const_value(self.module_set.qualified[expr.nid])[1]
        if isinstance(expr, Unary) and expr.op in (UnaryOp.NEGATE, UnaryOp.COMPLEMENT):
            if isinstance(expr.operand, Constant) and expr.operand.value == 2 ** 63:
                return -2 ** 63
            return fold_unary_op(expr.op, self._const_eval(expr.operand), t)
        if isinstance(expr, Unary) and expr.op == UnaryOp.NOT:
            return not self._const_eval(expr.operand)
        if isinstance(expr, Binary):
            left_type = self.facts.types.get(expr.left.nid)
            if expr.op in (BinaryOp.AND, BinaryOp.OR):
                left = self._const_eval(expr.left)
                right = self._const_eval(expr.right)
                return (left and right) if expr.op == BinaryOp.AND else (left or right)
            left, right = self._const_eval(expr.left), self._const_eval(expr.right)
            if left_type == Type.STR:
                if expr.op == BinaryOp.ADD:
                    return left + right
                if expr.op in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL):
                    return (left == right) == (expr.op == BinaryOp.EQUAL)
            elif left_type == Type.BOOL and expr.op in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL):
                return (left == right) == (expr.op == BinaryOp.EQUAL)
            else:
                if (expr.op in (BinaryOp.DIVIDE, BinaryOp.MODULO) and right == -1
                        and left == {Type.INT: -(2 ** 63), Type.INT32: -(2 ** 31)}.get(left_type)):
                    raise SemanticError("Integer overflow in division in a constant expression", expr)
                value = fold_binary_op(expr.op, left, right, t if t != Type.BOOL else left_type)
                if value is None:
                    raise SemanticError("Division by zero in a constant expression", expr)
                return bool(value) if t == Type.BOOL else value
        if isinstance(expr, Cast) and t in _INTEGER_TYPES:
            return fold_cast(t, self._const_eval(expr.expr), self.facts.types.get(expr.expr.nid))
        raise SemanticError(
            "A constant's value must be built from literals, other constants, operators, and integer casts", expr)

    def _resolve_sum_types(self, sum_type_defs: List[SumTypeDef], structs: Dict[str, StructInfo]) -> Dict[str, SumTypeInfo]:
        """Resolve sum type variants and check name collisions."""
        registry: Dict[str, SumTypeInfo] = {}
        for std in sum_type_defs:
            if std.name in _BUILTIN_FUNCTION_NAMES:
                raise SemanticError(
                    f"'{shown(std.name)}' is a builtin and can't be used as a "
                    f"sum type name",
                    std,
                )
            if std.name in structs:
                raise SemanticError(
                    f"Sum type '{shown(std.name)}' collides with a struct of "
                    f"the same name -- struct and sum-type names share "
                    f"one namespace and can never be the same",
                    std,
                )
            if std.name in self.type_aliases:
                raise SemanticError(
                    f"Sum type '{shown(std.name)}' collides with a type alias "
                    f"of the same name -- type-alias and sum-type names "
                    f"share one namespace and can never be the same",
                    std,
                )
            if std.name in registry:
                raise SemanticError(f"Sum type '{shown(std.name)}' is already declared", std)

            self._enter(std)
            resolved_variants: List[Type] = []
            for variant_name in std.variants:
                if any(sd.name == self._resolve_type_name(variant_name) for sd in sum_type_defs):
                    raise SemanticError(
                        f"Sum type '{shown(std.name)}' names '{variant_name}' as "
                        f"a variant, but '{variant_name}' is itself a sum "
                        f"type -- a sum type's variants can't include "
                        f"another sum type yet",
                        std,
                    )
                try:
                    variant_type = type_from_name(
                        variant_name, structs, self.type_aliases, std, resolve=self._resolve_type_name)
                except SemanticError:
                    # Name what's allowed for a simple typo.
                    if not isinstance(variant_name, str):
                        raise
                    raise SemanticError(
                        f"Sum type '{shown(std.name)}' names '{variant_name}' as "
                        f"a variant, but '{variant_name}' isn't a declared "
                        f"struct, `none`, or a valid scalar/str/array/slice/pointer/dict "
                        f"type",
                        std,
                    )
                if variant_type in resolved_variants:
                    raise SemanticError(
                        f"Sum type '{shown(std.name)}' lists '{variant_type}' "
                        f"as a variant more than once",
                        std,
                    )
                resolved_variants.append(variant_type)

            registry[std.name] = SumTypeInfo(name=std.name, variants=resolved_variants)
        return registry

    def _collect_methods(self, program: Program) -> Dict[Tuple[str, str], Tuple[List[Type], Type, str]]:
        """Reject duplicate method names per struct; return the method registry."""
        methods: Dict[Tuple[str, str], Tuple[List[Type], Type, str]] = {}
        for sd in program.structs:
            self._enter(sd)
            seen_names: Set[str] = set()
            for md in sd.methods:
                if md.name in seen_names:
                    raise SemanticError(
                        f"Method '{shown(md.name)}' is already declared on "
                        f"struct '{shown(sd.name)}'",
                        md,
                    )
                seen_names.add(md.name)
                param_types = [
                    self._type(p.type, p) for p in md.params
                ]
                return_type = Type.VOID if md.return_type is None else self._type(md.return_type, md)
                methods[(sd.name, md.name)] = (param_types, return_type, mangle_method_name(sd.name, md.name))
                if md.receiver_is_pointer:
                    self.pointer_receivers.add((sd.name, md.name))
        return methods

    def _collect_type_aliases(self, alias_defs: List[TypeAlias], structs: Dict[str, StructInfo]) -> Dict[str, Type]:
        """Resolve every alias fully, detecting cycles."""
        # Pass 1: reserve names, rejecting duplicates and collisions.
        seen: Dict[str, TypeAlias] = {}
        for ad in alias_defs:
            if ad.name in _BUILTIN_FUNCTION_NAMES:
                raise SemanticError(
                    f"'{shown(ad.name)}' is a builtin and can't be used as a "
                    f"type alias name",
                    ad,
                )
            if ad.name in structs:
                raise SemanticError(
                    f"Type alias '{shown(ad.name)}' collides with a struct of "
                    f"the same name -- struct and type-alias names "
                    f"share one namespace and can never be the same",
                    ad,
                )
            if ad.name in seen:
                raise SemanticError(f"Type alias '{shown(ad.name)}' is already declared", ad)
            seen[ad.name] = ad

        resolved: Dict[str, Type] = {}
        resolving: Set[str] = set()

        def resolve(name: str) -> Type:
            if name in resolved:
                return resolved[name]
            if name in resolving:
                raise SemanticError(
                    f"Type alias '{shown(name)}' is defined in terms of "
                    f"itself (a cycle)",
                    seen[name],
                )
            resolving.add(name)
            result = resolve_target(seen[name].target_type, seen[name])
            resolving.discard(name)
            resolved[name] = result
            return result

        def resolve_target(target, alias_node: TypeAlias) -> Type:
            """`alias_node` is for error positions."""
            if isinstance(target, ArrayTypeExpr):
                size = target.size if isinstance(target.size, int) else _array_sizes[target.nid]
                return Type(TypeKind.ARRAY, element_type=resolve_target(target.element_type, alias_node), size=size)
            if isinstance(target, SliceTypeExpr):
                return Type(TypeKind.SLICE, element_type=resolve_target(target.element_type, alias_node))
            if isinstance(target, str) and target in _TYPE_NAMES:
                return _TYPE_NAMES[target]
            saved_scope = self.scope
            self._enter(alias_node)
            target = self._resolve_type_name(target)
            self.scope = saved_scope
            if target in seen:
                return resolve(target)
            if target in structs:
                return Type(TypeKind.STRUCT, struct_name=target)
            raise SemanticError(
                f"Unknown type '{target}' in a type alias's own target "
                f"-- expected int, bool, str, a struct name, or "
                f"another type alias",
                alias_node,
            )

        for ad in alias_defs:
            resolve(ad.name)
        return resolved

    def _reserve_struct_names(self, struct_defs: List[StructDef]) -> Dict[str, StructInfo]:
        """Reserve struct names (None placeholders) so fields can forward-reference."""
        registry: Dict[str, StructInfo] = {}
        for sd in struct_defs:
            if sd.name in _BUILTIN_FUNCTION_NAMES:
                raise SemanticError(
                    f"'{shown(sd.name)}' is a builtin and can't be used as a "
                    f"struct name",
                    sd,
                )
            if sd.name in registry:
                raise SemanticError(f"Struct '{shown(sd.name)}' is already declared", sd)
            registry[sd.name] = None
        return registry

    def _resolve_struct_fields(self, struct_defs: List[StructDef], registry: Dict[str, StructInfo],
                               sum_names: Dict[str, None]) -> Dict[str, StructInfo]:
        """Resolve field types (which may name structs and sum types declared anywhere)."""
        for sd in struct_defs:
            self._enter(sd)
            fields: Dict[str, Type] = {}
            for f in sd.fields:
                if f.name in fields:
                    raise SemanticError(
                        f"Field '{f.name}' is already declared in struct '{shown(sd.name)}'",
                        f,
                    )
                fields[f.name] = type_from_name(
                    f.field_type, registry, self.type_aliases, f, sum_names, resolve=self._resolve_type_name)
            registry[sd.name] = StructInfo(name=sd.name, fields=fields)
        return registry

    def _check_value_containment(self, program: Program) -> None:
        """No struct or sum type may contain itself by value (directly, through arrays, or through
        another struct or sum): it would have no finite size. Pointers, slices and dicts are
        indirections, so recursion through them is fine."""
        decl = {sd.name: sd for sd in program.structs}
        decl.update({std.name: std for std in program.sum_types})

        def embedded(t: Type) -> Optional[str]:
            while t.kind == TypeKind.ARRAY:
                t = t.element_type
            if t.kind == TypeKind.STRUCT:
                return t.struct_name
            if t.kind == TypeKind.SUM:
                return t.sum_type_name
            return None

        def contents(name: str) -> List[Tuple[str, str]]:
            """(member description, embedded type name) for each by-value member."""
            if name in self.structs:
                pairs = [(f"{shown(name)}.{field}", embedded(t)) for field, t in self.structs[name].fields.items()]
            else:
                pairs = [(f"{shown(name)}'s {t} variant", embedded(t)) for t in self.sum_types[name].variants]
            return [(desc, inner) for desc, inner in pairs if inner is not None]

        def visit(name: str, path: List[Tuple[str, str]]) -> None:
            for desc, inner in contents(name):
                if any(n == inner for n, _ in path) or inner == path[0][0]:
                    chain = [d for _, d in path[1:]] + [desc]
                    raise SemanticError(
                        f"'{inner}' contains itself by value ({' -> '.join(chain)}), so it would have no "
                        f"finite size -- hold it through a pointer (*{inner}), slice, or dict instead",
                        decl[inner])
                visit(inner, path + [(inner, desc)])

        for name in decl:
            visit(name, [(name, name)])

    def check_extern_function_decl(self, ext: ExternFunctionDecl) -> None:
        """Validate an extern signature (scalars and pointers only) and register it."""
        if ext.name in _BUILTIN_FUNCTION_NAMES:
            raise SemanticError(
                f"'{shown(ext.name)}' is a builtin and can't be redefined as "
                f"an extern function",
                ext,
            )
        if ext.name in self.structs:
            raise SemanticError(
                f"Extern function '{shown(ext.name)}' collides with a struct "
                f"of the same name -- struct and function names share "
                f"one namespace and can never be the same, since "
                f"'{shown(ext.name)}(...)' would otherwise be ambiguous "
                f"between a call and a struct literal",
                ext,
            )
        if ext.name in self.type_aliases:
            raise SemanticError(
                f"Extern function '{shown(ext.name)}' collides with a type "
                f"alias of the same name -- function and type-alias "
                f"names share one namespace and can never be the same",
                ext,
            )
        if ext.name in self.sum_types:
            raise SemanticError(
                f"Extern function '{shown(ext.name)}' collides with a sum "
                f"type of the same name -- function and sum-type "
                f"names share one namespace and can never be the same",
                ext,
            )
        if ext.name in self.functions:
            raise SemanticError(f"Function '{shown(ext.name)}' is already declared", ext)

        param_types = [self._type(p.type, p) for p in ext.params]
        return_type = Type.VOID if ext.return_type is None else self._type(ext.return_type, ext)

        for p, p_type in zip(ext.params, param_types):
            if p_type.kind in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT, TypeKind.SUM, TypeKind.STR):
                raise SemanticError(
                    f"Extern function '{shown(ext.name)}''s parameter '{p.name}' has "
                    f"type {p_type} -- only scalar and pointer types are "
                    f"supported in an extern function's signature for now "
                    f"(array/slice/struct/sum/str-typed parameters aren't yet)",
                    p,
                )
        if return_type.kind in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT, TypeKind.SUM, TypeKind.STR):
            raise SemanticError(
                f"Extern function '{shown(ext.name)}' returns {return_type} -- "
                f"only scalar and pointer types are supported as an "
                f"extern function's own return type for now "
                f"(array/slice/struct/sum/str aren't yet)",
                ext,
            )

        self.functions[ext.name] = (param_types, return_type)

    def check_intrinsic_decl(self, ic: IntrinsicDecl) -> None:
        """Validate an intrinsic signature and register it."""
        if ic.name in _BUILTIN_FUNCTION_NAMES:
            raise SemanticError(
                f"'{shown(ic.name)}' is a builtin and can't be redefined as "
                f"an intrinsic",
                ic,
            )
        if ic.name in self.structs:
            raise SemanticError(
                f"Intrinsic '{shown(ic.name)}' collides with a struct "
                f"of the same name -- struct and function names share "
                f"one namespace and can never be the same, since "
                f"'{shown(ic.name)}(...)' would otherwise be ambiguous "
                f"between a call and a struct literal",
                ic,
            )
        if ic.name in self.type_aliases:
            raise SemanticError(
                f"Intrinsic '{shown(ic.name)}' collides with a type "
                f"alias of the same name -- function and type-alias "
                f"names share one namespace and can never be the same",
                ic,
            )
        if ic.name in self.sum_types:
            raise SemanticError(
                f"Intrinsic '{shown(ic.name)}' collides with a sum "
                f"type of the same name -- function and sum-type "
                f"names share one namespace and can never be the same",
                ic,
            )
        if ic.name in self.functions:
            raise SemanticError(f"Function '{shown(ic.name)}' is already declared", ic)

        param_types = [self._type(p.type, p) for p in ic.params]
        return_type = Type.VOID if ic.return_type is None else self._type(ic.return_type, ic)

        self.functions[ic.name] = (param_types, return_type)
        self.intrinsic_original_names[ic.name] = ic.original_name

    def analyze_function(self, fn: Function) -> None:
        self._enter(fn)
        self.scopes = [{}]
        self.loop_depth = 0
        # Params are locals; _declare also catches duplicates.
        for p in fn.params:
            param_type = self._type(p.type, p)
            self.facts.types[p.nid] = param_type
            self.facts.symbols[p.nid] = self.symbols.new(p.name, 'param', param_type, p)
            self._declare(p.name, param_type, p, self.facts.symbols[p.nid].id)
        return_type = Type.VOID if fn.return_type is None else self._type(fn.return_type, fn)
        self.facts.returns[fn.nid] = return_type
        for stmt in fn.body:
            self.analyze_statement(stmt, return_type)
        # Void functions may fall off the end.
        if return_type != Type.VOID and not always_returns(fn.body):
            raise SemanticError(
                f"Function '{shown(fn.name)}' (declared to return {return_type}) "
                f"does not return a value on all code paths",
                fn,
            )


    def _push_scope(self) -> None:
        self.scopes.append({})

    def _pop_scope(self) -> None:
        self.scopes.pop()

    def _declare(self, name: str, type_: Type, node: Optional[Node], decl_id) -> None:
        """Declare in the innermost scope; shadowing outer scopes is allowed.
        decl_id is the declaration's Symbol.id."""
        if name in self.scopes[-1]:
            raise SemanticError(f"Variable '{shown(name)}' is already declared in this scope", node)
        self.scopes[-1][name] = (type_, decl_id)

    def _reject_typed_literal_read_as_multiplication(self, expr: Binary) -> None:
        """`[2][1]*P[...]` parses as `[2][1] * P[...]` (an indexed literal times an index): explain when
        the right side's name is a type, which is surely what was meant."""
        def root(node):
            while isinstance(node, Index):
                node = node.array
            return node
        name = root(expr.right)
        if (isinstance(name, Variable) and isinstance(root(expr.left), ArrayLiteral) and isinstance(expr.left, Index)
                and self._resolve_type_name(name.name) in {**self.structs, **self.sum_types, **self.type_aliases}
                and not any(name.name in scope for scope in self.scopes)):
            raise SemanticError(
                f"'{name.name}' is a type, but this reads as a multiplication: a typed literal of pointers with "
                f"only fixed sizes, like `[2][1]*{name.name}[...]`, can't be written inline -- give the variable "
                f"(or parameter) the type and use an untyped literal, e.g. `[2][1]*{name.name} g = [...]`", expr)

    def _resolve(self, name: str, node: Optional[Node] = None) -> Tuple[Type, object]:
        """(type, decl id) of `name`, innermost-first; constants (decl id None) after locals."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        if self._const_key(name) is not None:
            return self._const_value(self._const_key(name))[0], None
        raise SemanticError(f"Reference to undeclared variable '{shown(name)}'", node)

    def _lookup(self, name: str, node: Optional[Node] = None) -> Type:
        return self._resolve(name, node)[0]


    def analyze_statement(self, stmt: Node, return_type: Type) -> None:
        if isinstance(stmt, VarDecl):
            self.analyze_var_decl(stmt)
        elif isinstance(stmt, Assign):
            self.analyze_assign(stmt)
        elif isinstance(stmt, IndexAssign):
            self.analyze_index_assign(stmt)
        elif isinstance(stmt, FieldAssign):
            self.analyze_field_assign(stmt)
        elif isinstance(stmt, DerefAssign):
            self.analyze_deref_assign(stmt)
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
            return value_type in self.sum_types[target_type.sum_type_name].variants
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
            if variant_type in self.sum_types[target_type.element_type.sum_type_name].variants:
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
        self._declare(stmt.name, declared_type, stmt, self.facts.symbols[stmt.nid].id)

    def analyze_assign(self, stmt: Assign) -> None:
        if stmt.name in self._narrowed_names:
            raise SemanticError(
                f"Cannot reassign '{stmt.name}' while it's narrowed by "
                f"an enclosing 'is' check -- assign to a different "
                f"variable instead",
                stmt,
            )
        if self._const_key(stmt.name) is not None and not any(stmt.name in scope for scope in self.scopes):
            raise SemanticError(f"Cannot assign to constant '{stmt.name}'", stmt)
        declared_type, self.facts.decls[stmt.nid] = self._resolve(stmt.name, stmt)
        value_type = self._check_value_flowing_into_allowing_struct_literal(stmt.value, declared_type)
        if not self._types_compatible(value_type, declared_type):
            raise SemanticError(
                f"Cannot assign a value of type {value_type} to '{stmt.name}' "
                f"(declared {declared_type})",
                stmt,
            )

    def _check_compound_assign(
            self,
            compound_op: BinaryOp,
            target_type: Type,
            target_expr_for_check: Node,
            value_expr: Node,
            stmt: Node) -> None:
        """Check a compound assignment via a synthetic Binary."""
        if target_type == Type.STR and compound_op == BinaryOp.ADD:
            raise SemanticError(
                f"Compound assignment ('+=') to a str-typed target isn't "
                f"supported yet -- string concatenation has a different "
                f"codegen shape than arithmetic compound assignment; "
                f"write it as a plain '=' instead",
                stmt,
            )
        synthetic = Binary(
            op=compound_op, left=target_expr_for_check, right=value_expr, line=stmt.line, col=stmt.col, file=stmt.file)
        self.check_binary(synthetic)

    def analyze_index_assign(self, stmt: IndexAssign) -> None:
        """`array[index] = value`."""
        element_type = self._check_indexable_and_index(stmt.array, stmt.index)
        if stmt.compound_op is not None:
            target_expr = Index(array=stmt.array, index=stmt.index, line=stmt.line, col=stmt.col, file=stmt.file)
            value_expr = stmt.value
            self._check_compound_assign(stmt.compound_op, element_type, target_expr, value_expr, stmt)
            return
        value_type = self._check_value_flowing_into_allowing_struct_literal(stmt.value, element_type)
        if not self._types_compatible(value_type, element_type):
            raise SemanticError(
                f"Cannot assign a value of type {value_type} to an array "
                f"element of type {element_type}",
                stmt,
            )

    def analyze_field_assign(self, stmt: FieldAssign) -> None:
        """`base.name = value`."""
        field_type = self._check_struct_and_field(stmt.base, stmt.name)
        if stmt.compound_op is not None:
            target_expr = Field(base=stmt.base, name=stmt.name, line=stmt.line, col=stmt.col, file=stmt.file)
            self._check_compound_assign(stmt.compound_op, field_type, target_expr, stmt.value, stmt)
            return
        value_type = self._check_value_flowing_into_allowing_struct_literal(stmt.value, field_type)
        if not self._types_compatible(value_type, field_type):
            raise SemanticError(
                f"Cannot assign a value of type {value_type} to field "
                f"'{stmt.name}' of type {field_type}",
                stmt,
            )

    def analyze_deref_assign(self, stmt: DerefAssign) -> None:
        """`*pointer = value`."""
        pointer_type = self.check_expr(stmt.pointer)
        if pointer_type.kind != TypeKind.POINTER:
            raise SemanticError(
                f"Cannot dereference a value of type {pointer_type} for "
                f"assignment -- '*' requires a pointer operand",
                stmt.pointer,
            )
        pointee_type = pointer_type.element_type
        if stmt.compound_op is not None:
            target_expr = Unary(
                op=UnaryOp.DEREFERENCE, operand=stmt.pointer, line=stmt.line, col=stmt.col, file=stmt.file)
            self._check_compound_assign(stmt.compound_op, pointee_type, target_expr, stmt.value, stmt)
            return
        value_type = self._check_value_flowing_into_allowing_struct_literal(stmt.value, pointee_type)
        if not self._types_compatible(value_type, pointee_type):
            raise SemanticError(
                f"Cannot assign a value of type {value_type} through a "
                f"pointer to {pointee_type}",
                stmt,
            )

    def _check_indexable_and_index(self, base_expr: Node, index_expr: Node) -> Type:
        """Check an array/slice/dict base and its index; return the element type."""
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
        # Declare an `EXPR is T as NAME` binding before checking the condition.
        has_binding = isinstance(stmt.condition, IsCheck) and stmt.condition.subject is not None
        if has_binding:
            self._push_scope()
            self._declare_is_binding(stmt.condition)

        condition_type = self.check_expr(stmt.condition)
        if condition_type != Type.BOOL:
            raise SemanticError(
                f"'if' condition must be bool, got {condition_type} "
                f"(no implicit int-to-bool conversion -- try `x != 0` "
                f"instead of `x`)",
                stmt.condition,
            )

        self._push_scope()
        # Narrow the IsCheck variable within then_body only.
        narrowed_name = None
        if isinstance(stmt.condition, IsCheck):
            narrowed_name = stmt.condition.variable_name
            narrowed_type = self._type(stmt.condition.type_name, stmt.condition, sums=False)
            self.facts.narrowed[stmt.condition.nid] = narrowed_type
            self._declare(narrowed_name, narrowed_type, stmt.condition, self.facts.decls[stmt.condition.nid])
            self._narrowed_names.add(narrowed_name)
        for s in stmt.then_body:
            self.analyze_statement(s, return_type)
        if narrowed_name is not None:
            self._narrowed_names.discard(narrowed_name)
        self._pop_scope()

        if stmt.else_body is not None:
            self._push_scope()
            for s in stmt.else_body:
                self.analyze_statement(s, return_type)
            self._pop_scope()

        if has_binding:
            self._pop_scope()

    def analyze_match(self, stmt: 'Match', return_type: Type) -> None:
        """Each arm narrows the subject within its own body, like `if NAME is T:` (the order of checks,
        and so of errors, is that of the equivalent `if`/`else` chain)."""
        first_check = stmt.arms[0][0]
        has_binding = stmt.subject is not None
        if has_binding:
            self._push_scope()
            self._declare_is_binding(first_check)
        for i, (check, body) in enumerate(stmt.arms):
            self.check_expr(check)
            if i == 0:
                self._check_match_exhaustiveness(stmt)
            self._push_scope()
            narrowed_type = self._type(check.type_name, check, sums=False)
            self.facts.narrowed[check.nid] = narrowed_type
            self._declare(check.variable_name, narrowed_type, check, self.facts.decls[check.nid])
            self._narrowed_names.add(check.variable_name)
            for s in body:
                self.analyze_statement(s, return_type)
            self._narrowed_names.discard(check.variable_name)
            self._pop_scope()
        if stmt.else_body is not None:
            self._push_scope()
            for s in stmt.else_body:
                self.analyze_statement(s, return_type)
            self._pop_scope()
        if has_binding:
            self._pop_scope()

    def _declare_is_binding(self, check: IsCheck) -> None:
        """`EXPR is T as NAME`: check EXPR and declare NAME, a copy of it."""
        subject_type = self.check_expr(check.subject)
        sym = self.symbols.new(check.variable_name, 'narrowing', subject_type, check)
        if subject_type.kind == TypeKind.SUM:  # otherwise rejected when the check itself is checked
            binding = VarDecl(name=check.variable_name, var_type=subject_type.sum_type_name, init=check.subject,
                              line=check.line, col=check.col, file=check.file)
            self.facts.bindings[check.nid] = binding
            self.facts.types[binding.nid] = subject_type
            self.facts.symbols[binding.nid] = sym
        self._declare(check.variable_name, subject_type, check, sym.id)

    def _check_match_exhaustiveness(self, stmt: 'Match') -> None:
        """Reject duplicate arms; require exhaustiveness without an else."""
        subject_name = stmt.variable_name
        subject_type = self._lookup(subject_name, stmt.arms[0][0])
        sum_type_info = self.sum_types[subject_type.sum_type_name]

        seen: Dict[Type, IsCheck] = {}
        for arm_condition, _ in stmt.arms:
            arm_type = self._type(arm_condition.type_name, arm_condition, sums=False)
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
        condition_type = self.check_expr(stmt.condition)
        if condition_type != Type.BOOL:
            raise SemanticError(
                f"'while' condition must be bool, got {condition_type} "
                f"(no implicit int-to-bool conversion -- try `x != 0` "
                f"instead of `x`)",
                stmt.condition,
            )

        self.loop_depth += 1
        self._push_scope()
        for s in stmt.body:
            self.analyze_statement(s, return_type)
        self._pop_scope()
        self.loop_depth -= 1

    def analyze_for(self, stmt: For, return_type: Type) -> None:
        """`for init; cond; increment:`; one scope spans all clauses."""
        self._push_scope()
        self.analyze_statement(stmt.init, return_type)
        condition_type = self.check_expr(stmt.condition)
        if condition_type != Type.BOOL:
            raise SemanticError(
                f"'for' condition must be bool, got {condition_type} "
                f"(no implicit int-to-bool conversion -- try `x != 0` "
                f"instead of `x`)",
                stmt.condition,
            )
        self.loop_depth += 1
        for s in stmt.body:
            self.analyze_statement(s, return_type)
        self.loop_depth -= 1
        self.analyze_statement(stmt.increment, return_type)
        self._pop_scope()

    def analyze_for_in(self, stmt: ForIn, return_type: Type) -> None:
        """`for a[, b] in iterable:` over arrays, slices, dicts, and strings (bytes)."""
        if not isinstance(
                stmt.iterable, (Variable, Field, Index, Slice, ArrayLiteral, DictLiteral, StringLiteral, Call)):
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
        self._push_scope()
        symbols = [
            self.symbols.new(name, 'binding', t, stmt) for name, t in zip(stmt.binding_names, binding_types)
        ]
        self.facts.for_symbols[stmt.nid] = symbols
        for name, binding_type, sym in zip(stmt.binding_names, binding_types, symbols):
            self._declare(name, binding_type, stmt, sym.id)
        self.loop_depth += 1
        for s in stmt.body:
            self.analyze_statement(s, return_type)
        self.loop_depth -= 1
        self._pop_scope()

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
            if constant_key is not None:
                if constant_key in seen_constant_keys:
                    raise SemanticError(
                        f"Dict literal lists the key {constant_key[1]!r} more than once",
                        key_expr,
                    )
                seen_constant_keys.add(constant_key)
        return Type(TypeKind.DICT, key_type=key_type, element_type=value_type)

    def check_array_literal(self, expr: ArrayLiteral, expected_element_type: Optional[Type] = None) -> Type:
        """`[e, ...]` or `[N]T[...]`; homogeneous. Untyped literals need an expected element type."""
        if expr.type_expr is not None:
            declared_type = self._type(expr.type_expr, expr)
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
        element_types = [self._check_expr_allowing_struct_literal(e) for e in expr.elements]
        first = element_types[0]
        for i, t in enumerate(element_types[1:], start=2):
            if t != first:
                raise SemanticError(
                    f"Array literal elements must all be the same type -- "
                    f"element 1 is {first}, element {i} is {t}",
                    expr.elements[i - 1],
                )
        return Type(TypeKind.ARRAY, element_type=first, size=len(expr.elements))

    def check_index(self, expr: Index) -> Type:
        """`base[index]`; str indexing yields uint8."""
        base_type = self.check_expr(expr.array)
        if base_type.kind == TypeKind.STR:
            index_type = self.check_expr(expr.index)
            if index_type != Type.INT:
                raise SemanticError(f"Index must be int, got {index_type}", expr.index)
            return Type.UINT8
        return self._check_indexable_and_index(expr.array, expr.index)

    def check_field(self, expr: Field) -> Type:
        key = self.module_set.qualified.get(expr.nid)
        if key is not None:  # `alias.NAME`: a constant of another module
            if key not in self.const_decls:
                raise SemanticError(f"Reference to undeclared variable '{shown(key)}'", expr)
            self.facts.const_refs[expr.nid] = key
            return self._const_value(key)[0]
        return self._check_struct_and_field(expr.base, expr.name)

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
        struct_info = self.structs[base_type.struct_name]
        if field_name not in struct_info.fields:
            raise SemanticError(
                f"Struct '{shown(base_type.struct_name)}' has no field '{field_name}'",
                base_expr,
            )
        return struct_info.fields[field_name]

    def check_struct_literal(self, expr: Call) -> Type:
        """`Name(args)`: positional struct literal; must be exhaustive."""
        name = self._struct_literal(expr)
        self._record_call(expr, name)
        struct_info = self.structs[name]
        field_items = list(struct_info.fields.items())
        if expr.kwargs is not None:
            return self._check_named_struct_literal(expr, struct_info, field_items)
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
        name = struct_info.name
        """`Name(f=v, ...)`: named struct literal; omitted fields are zero."""
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
            return None if Type.NONE in self.sum_types[t.sum_type_name].variants else t
        if t.kind == TypeKind.STRUCT and t.struct_name not in seen:
            seen.add(t.struct_name)
            for field_type in self.structs[t.struct_name].fields.values():
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
        receiver_is_pointer = (receiver_type.kind == TypeKind.POINTER
                               and receiver_type.element_type.kind == TypeKind.STRUCT)
        if receiver_is_pointer:
            # auto-deref
            receiver_type = receiver_type.element_type
        if receiver_type.kind != TypeKind.STRUCT:
            raise SemanticError(
                f"Cannot call method '{expr.name}' on a value of type "
                f"{receiver_type} -- methods are only defined on structs",
                expr.receiver,
            )
        key = (receiver_type.struct_name, expr.name)
        if key not in self.methods:
            raise SemanticError(
                f"Struct '{shown(receiver_type.struct_name)}' has no method "
                f"'{expr.name}'",
                expr,
            )
        param_types, return_type, mangled_name = self.methods[key]
        receiver = expr.receiver
        if key in self.pointer_receivers and not receiver_is_pointer:
            # Pointer receiver: pass the receiver's address.
            if not isinstance(expr.receiver, (Variable, Field, Index)) and not (
                    isinstance(expr.receiver, Unary) and expr.receiver.op == UnaryOp.DEREFERENCE):
                raise SemanticError(
                    f"Method '{expr.name}' on '{shown(receiver_type.struct_name)}' has a pointer receiver, so it "
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
                f"Method '{expr.name}' on '{shown(receiver_type.struct_name)}' "
                f"expects {len(param_types)} argument(s), got "
                f"{len(expr.args)}",
                expr,
            )
        for i, (arg, expected_type) in enumerate(zip(expr.args, param_types), start=1):
            actual_type = self._check_value_flowing_into_allowing_struct_literal(arg, expected_type)
            if not self._types_compatible(actual_type, expected_type):
                raise SemanticError(
                    f"Argument {i} to method '{expr.name}' on "
                    f"'{shown(receiver_type.struct_name)}' should be "
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
        if name in self.structs:
            raise SemanticError(
                f"'{expr.name}(...)' is a struct literal, which is only "
                f"allowed as a variable's initializer, a plain "
                f"assignment's value, a direct function-call or "
                f"method-call argument, a method-call receiver, a "
                f"direct return value, an array literal's own "
                f"element, an IndexAssign's own element, a "
                f"FieldAssign's own field or base, a field-access "
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
        visible = expr.nid in self.module_set.qualified or self.scope.resolve(expr.name) is not None
        if name not in self.functions or not visible:  # another module's extern needs an import too
            raise SemanticError(f"Call to undeclared function '{shown(name)}'", expr)
        param_types, return_type = self.functions[name]
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
        if arg_type == Type.VOID:
            raise SemanticError(
                "'print' cannot be called with the result of a function "
                "that has no declared return type -- there's no value there to print",
                expr.args[0],
            )
        if arg_type == Type.NONE:
            raise SemanticError(
                "'print' cannot be called with a bare 'none' -- store it "
                "in a pointer-typed variable first (e.g. `*int p = none`), "
                "then print that",
                expr.args[0],
            )
        return Type.VOID

    def check_len_call(self, expr: Call) -> Type:
        """`len(x)` for arrays, slices, str, and dicts."""
        if len(expr.args) != 1:
            raise SemanticError(
                f"'len' expects exactly 1 argument, got {len(expr.args)}",
                expr,
            )
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
        """`bytes(s)`: a new []byte copy of str s. Lowered to a runtime call."""
        if len(expr.args) != 1 or expr.kwargs:
            raise SemanticError(f"bytes() takes exactly one argument, got {len(expr.args)}", expr)
        arg_type = self.check_expr(expr.args[0])
        if arg_type != Type.STR:
            raise SemanticError(f"bytes() takes a str, got {arg_type}", expr)
        self.facts.calls[expr.nid] = ('hornet_bytes', list(expr.args))
        self.functions['hornet_bytes'] = ([Type.STR], _BYTE_SLICE)
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
        if isinstance(expr.value, float) and not expr.value.is_integer():
            raise SemanticError(
                f"'{expr.value}' is not a whole number -- this language has "
                f"no floating-point type; only int and bool exist",
                expr,
            )
        if expr.value > 2**63 - 1:
            raise SemanticError(f"Integer literal {int(expr.value)} is out of range for int (64-bit)", expr)
        return Type.INT

    def check_variable(self, expr: Variable) -> Type:
        t, self.facts.decls[expr.nid] = self._resolve(expr.name, expr)
        if self.facts.decls[expr.nid] is None:
            self.facts.const_refs[expr.nid] = self._const_key(expr.name)
        return t

    def check_is_check(self, expr: IsCheck) -> Type:
        """`NAME is T` / `EXPR is T as NAME`; T must be a variant."""
        variable_type, self.facts.decls[expr.nid] = self._resolve(expr.variable_name, expr)
        if variable_type.kind != TypeKind.SUM:
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
        narrowed_type = self._type(expr.type_name, expr, sums=False)
        sum_type_info = self.sum_types[variable_type.sum_type_name]
        if narrowed_type not in sum_type_info.variants:
            raise SemanticError(
                f"'{expr.type_name}' is not one of {variable_type}'s own "
                f"declared variants ({', '.join(str(v) for v in sum_type_info.variants)})",
                expr,
            )
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
                    f"`not (x == 0)` instead of `not x`)",
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
        """`T(expr)` between integer types; literals range-checked against T."""
        target_type = self._type(expr.target_type, expr)
        if target_type == Type.STR:
            source_type = self.check_expr(expr.expr)
            if source_type not in (Type.UINT8, _BYTE_SLICE):
                raise SemanticError(f"str(...) takes a byte or []byte, got {source_type}", expr)
            return Type.STR
        if target_type == Type.INT64 and self._as_folded_int_literal(expr.expr) is not None:
            self._record_literal_type(expr.expr, Type.INT64)
            source_type = Type.INT64
        else:
            source_type = self.check_expr(expr.expr)
        if target_type not in _INTEGER_TYPES or source_type not in _INTEGER_TYPES:
            raise SemanticError(
                f"Cannot cast {source_type} to {target_type} -- casting "
                f"is only supported between int, int8, uint8, and int32 "
                f"right now",
                expr,
            )
        return target_type

    def _is_comparable_type(self, t: Type) -> bool:
        """Whether `==` is defined for `t`."""
        if t.kind == TypeKind.ARRAY:
            return self._is_comparable_type(t.element_type)
        if t.kind == TypeKind.STRUCT:
            struct_info = self.structs[t.struct_name]
            return all(self._is_comparable_type(field_type) for field_type in struct_info.fields.values())
        if t.kind in (TypeKind.SLICE, TypeKind.SUM, TypeKind.DICT):
            return False
        return True  # INT, BOOL, STR, POINTER

    def check_binary(self, expr: Binary) -> Type:
        if expr.op == BinaryOp.MULTIPLY:
            self._reject_typed_literal_read_as_multiplication(expr)
        left_type = self._check_expr_allowing_struct_literal(expr.left)
        right_type = self._check_expr_allowing_struct_literal(expr.right)
        op = expr.op

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
            self._require_same_integer_type(left_type, right_type, op, expr)
            return Type.BOOL

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
                    if Type.NONE not in self.sum_types[side.sum_type_name].variants:
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
                        f"sum type, or dict, none of which has '==' "
                        f"defined for it yet outside comparing to none",
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
                        f"sum type, or dict, none of which has '==' "
                        f"defined for it yet outside comparing to none",
                        expr,
                    )
                return Type.BOOL

            # Slice, sum, and dict equality is undefined.
            if left_type.kind in (TypeKind.SLICE, TypeKind.VOID, TypeKind.NONE, TypeKind.SUM, TypeKind.DICT) or right_type.kind in (TypeKind.SLICE, TypeKind.VOID, TypeKind.NONE, TypeKind.SUM, TypeKind.DICT):
                raise SemanticError(
                    f"'{op.symbol()}' does not support slice, void, sum "
                    f"type, dict, or none operands, except comparing a "
                    f"slice, pointer, or dict to none",
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
            # `key in dict`, or `value in array/slice`.
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


class ElaborationError(Exception):
    """A tree shape with no typed-tree rule: a compiler bug, since analysis accepted it."""


class _TypedTreeBuilder:
    """Builds the typed tree (typed_ast.py) from a checked program: every implicit operation becomes
    explicit, and values flowing into a slot of a known type go through `convert`."""

    def __init__(self, analyzer: 'SemanticAnalyzer'):
        self.facts = analyzer.facts
        self.consts = analyzer.consts
        self.symbols = analyzer.symbols
        self.structs = analyzer.structs
        self.sum_types = analyzer.sum_types
        self.functions = analyzer.functions
        self.intrinsics = analyzer.intrinsic_original_names  # name -> the intrinsic's own name
        self.externs = analyzer.extern_names
        self.return_type = None

    def ty(self, e) -> Type:
        return self.facts.types[e.nid]

    def constant(self, e) -> typed.Expr:
        """A reference to a constant (`NAME` or `alias.NAME`): its value."""
        const_type, value = self.consts[self.facts.const_refs[e.nid]]
        if const_type == Type.BOOL:
            return typed.BoolLit(Type.BOOL, value)
        if const_type == Type.STR:
            return typed.StrLit(Type.STR, value)
        return typed.IntLit(self.ty(e), value)

    def program(self, functions) -> typed.Program:
        return typed.Program(tuple(self.function(fn) for fn in functions), self.structs, self.sum_types,
                             self.symbols)

    def function(self, fn: syntax.Function) -> typed.Function:
        self.return_type = self.facts.returns[fn.nid]
        return typed.Function(fn.name, tuple(self.facts.symbols[param.nid] for param in fn.params), self.return_type,
                              self.block(fn.body))

    # -- statements

    def block(self, statements) -> tuple:
        out = []
        for stmt in statements or ():
            out.extend(self.statement(stmt))
        return tuple(out)

    def statement(self, s) -> list:
        """Typed statements for one parser statement (a narrowing binding adds its declaration)."""
        if isinstance(s, syntax.VarDecl):
            return [self.declare(s)]
        if isinstance(s, syntax.Assign):
            symbol = self.symbols[self.facts.decls[s.nid]]
            return [typed.Assign(typed.Local(symbol.type, symbol), self.convert(s.value, symbol.type))]
        if isinstance(s, (syntax.IndexAssign, syntax.FieldAssign, syntax.DerefAssign)):
            if isinstance(s, syntax.IndexAssign):
                target = self.index(syntax.Index(array=s.array, index=s.index))
            elif isinstance(s, syntax.FieldAssign):
                target = self.field(syntax.Field(base=s.base, name=s.name))
            else:
                pointer = self.expr(s.pointer)
                target = typed.Deref(pointer.type.element_type, pointer)
            if s.compound_op is not None:
                return [typed.CompoundAssign(target, s.compound_op, self.convert(s.value, target.type))]
            return [typed.Assign(target, self.convert(s.value, target.type))]
        if isinstance(s, syntax.ExprStmt):
            return [typed.ExprStmt(self.expr(s.expr))]
        if isinstance(s, syntax.Return):
            return [typed.Return(None if s.value is None else self.convert(s.value, self.return_type))]
        if isinstance(s, syntax.If):
            return self.if_statement(s)
        if isinstance(s, syntax.Match):
            before, subject = self.narrowing_subject(s.arms[0][0])
            arms = tuple((self.facts.narrowed[check.nid], self.block(body)) for check, body in s.arms)
            return before + [typed.Match(subject, arms, None if s.else_body is None else self.block(s.else_body))]
        if isinstance(s, syntax.While):
            return [typed.While(self.expr(s.condition), self.block(s.body))]
        if isinstance(s, syntax.For):
            init = self.statement(s.init) if s.init is not None else []
            if len(init) > 1:
                raise ElaborationError(f"for-loop initializer elaborated to {len(init)} statements")
            step = self.statement(s.increment)
            return [typed.For(init[0] if init else None, self.expr(s.condition), step[0], self.block(s.body))]
        if isinstance(s, syntax.ForIn):
            iterable = self.expr(s.iterable)
            kind = {TypeKind.ARRAY: 'array', TypeKind.SLICE: 'slice', TypeKind.STR: 'str',
                    TypeKind.DICT: 'dict'}[iterable.type.kind]
            return [typed.ForIn(kind, iterable, tuple(self.facts.for_symbols[s.nid]), self.block(s.body))]
        if isinstance(s, syntax.Break):
            return [typed.Break()]
        if isinstance(s, syntax.Continue):
            return [typed.Continue()]
        raise ElaborationError(f"No elaboration for statement {type(s).__name__}")

    def declare(self, s: syntax.VarDecl) -> typed.Declare:
        symbol = self.facts.symbols[s.nid]
        init = self.zero(symbol.type) if s.init is None else self.convert(s.init, symbol.type)
        return typed.Declare(symbol, init)

    def if_statement(self, s: syntax.If) -> list:
        if not isinstance(s.condition, syntax.IsCheck):
            return [typed.If(self.expr(s.condition), self.block(s.then_body), self.block(s.else_body))]
        before, subject = self.narrowing_subject(s.condition)
        test = typed.TagTest(Type.BOOL, subject, self.facts.narrowed[s.condition.nid])
        return before + [typed.If(test, self.block(s.then_body), self.block(s.else_body))]

    def narrowing_subject(self, check: syntax.IsCheck):
        """(statements to run first, the sum being tested) for `NAME is T` or `EXPR is T as NAME`."""
        binding = self.facts.bindings.get(check.nid)
        if binding is not None:
            symbol = self.facts.symbols[binding.nid]
            declare = typed.Declare(symbol, self.convert(check.subject, symbol.type))
            return [declare], typed.Local(symbol.type, symbol)
        symbol = self.symbols[self.facts.decls[check.nid]]
        return [], typed.Local(symbol.type, symbol)

    # -- conversions

    def zero(self, type_: Type) -> typed.Expr:
        if type_.kind == TypeKind.DICT:
            return typed.NewEmptyDict(type_)
        if type_.kind == TypeKind.SLICE:
            return typed.EmptySlice(type_)
        return typed.ZeroValue(type_)

    def convert(self, e, target: Type) -> typed.Expr:
        """`e` as a value flowing into a slot of type `target`, implicit operations made explicit."""
        if isinstance(e, syntax.Unary) and e.nid in self.facts.boxed:
            return typed.BoxVariant(target, typed.WidenToSum(self.facts.boxed[e.nid], self.expr(e.operand)))
        if isinstance(e, syntax.NoneLiteral):
            if target.kind == TypeKind.POINTER:
                return typed.NoneLit(target)
            if target.kind == TypeKind.SUM:
                return typed.WidenToSum(target, typed.NoneLit(Type.NONE))
            raise ElaborationError(f"`none` flowing into {target}")
        if (target.kind == TypeKind.SLICE and isinstance(e, syntax.ArrayLiteral) and e.type_expr is None):
            elements = tuple(self.convert(x, target.element_type) for x in e.elements)
            return typed.SliceLiteral(target, elements) if elements else typed.EmptySlice(target)
        if target.kind == TypeKind.ARRAY and isinstance(e, syntax.ArrayLiteral) and e.type_expr is None:
            return typed.ArrayLiteral(target, tuple(self.convert(x, target.element_type) for x in e.elements))
        value = self.expr(e)
        if value.type == target:
            return value
        if target.kind == TypeKind.SUM and value.type in self.sum_types[target.sum_type_name].variants:
            return typed.WidenToSum(target, value)
        if value.type.kind == TypeKind.POINTER and value.type.element_type == target:
            return typed.Deref(target, value)  # a method's value receiver called through a pointer
        if (
                isinstance(value, typed.IntLit)
                and target.kind in (TypeKind.INT, TypeKind.INT32, TypeKind.INT8, TypeKind.UINT8)
        ):
            return typed.IntLit(target, value.value)
        raise ElaborationError(f"No conversion from {value.type} to {target} for {e!r}")

    # -- expressions

    def expr(self, e) -> typed.Expr:
        if isinstance(e, syntax.Constant):
            return typed.IntLit(self.ty(e), e.value)
        if isinstance(e, syntax.BoolLiteral):
            return typed.BoolLit(Type.BOOL, e.value)
        if isinstance(e, syntax.StringLiteral):
            return typed.StrLit(Type.STR, e.value)
        if isinstance(e, syntax.ByteLiteral):
            return typed.IntLit(Type.UINT8, e.value)
        if isinstance(e, syntax.NoneLiteral):
            return typed.NoneLit(Type.NONE)
        if isinstance(e, syntax.Variable):
            decl = self.facts.decls[e.nid]
            if decl is None:
                return self.constant(e)
            symbol = self.symbols[decl]
            local = typed.Local(symbol.type, symbol)
            if self.ty(e) != symbol.type:
                return typed.Payload(self.ty(e), local)  # narrowed by an enclosing `is`
            return local
        if isinstance(e, syntax.ArrayLiteral):
            array_type = self.ty(e)
            return typed.ArrayLiteral(array_type, tuple(self.convert(x, array_type.element_type) for x in e.elements))
        if isinstance(e, syntax.DictLiteral):
            d = self.ty(e)
            return typed.DictLiteral(d, tuple((self.convert(k, d.key_type), self.convert(v, d.element_type))
                                              for k, v in e.entries))
        if isinstance(e, syntax.Index):
            return self.index(e)
        if (
                isinstance(e, syntax.Slice) and isinstance(e.array, syntax.ArrayLiteral)
                and e.low is None and e.high is None
        ):
            # A typed slice literal, `[]T[...]`: new storage holding the elements.
            elements = tuple(self.convert(x, self.ty(e).element_type) for x in e.array.elements)
            return typed.SliceLiteral(self.ty(e), elements) if elements else typed.EmptySlice(self.ty(e))
        if isinstance(e, syntax.Slice):
            base = self.expr(e.array)
            kind = {TypeKind.ARRAY: 'array', TypeKind.SLICE: 'slice', TypeKind.STR: 'str'}[base.type.kind]
            low = None if e.low is None else self.expr(e.low)
            high = None if e.high is None else self.expr(e.high)
            return typed.SliceOf(self.ty(e), kind, base, low, high)
        if isinstance(e, syntax.Field):
            return self.constant(e) if e.nid in self.facts.const_refs else self.field(e)
        if isinstance(e, syntax.Call):
            return self.call(e)
        if isinstance(e, syntax.Unary):
            if e.op == UnaryOp.ADDRESS_OF:
                if e.nid in self.facts.boxed:
                    return self.convert(e, self.ty(e))
                operand = self.expr(e.operand)
                return typed.AddressOf(self.ty(e), operand)
            if e.op == UnaryOp.DEREFERENCE:
                pointer = self.expr(e.operand)
                return typed.Deref(pointer.type.element_type, pointer)
            return typed.Unary(self.ty(e), e.op, self.expr(e.operand))
        if isinstance(e, syntax.Cast):
            value = self.expr(e.expr)
            if self.ty(e) == Type.STR:
                return (typed.StrFromByte if value.type == Type.UINT8 else typed.StrFromBytes)(Type.STR, value)
            return typed.IntCast(self.ty(e), value)
        if isinstance(e, syntax.Binary):
            return self.binary(e)
        raise ElaborationError(f"No elaboration for expression {type(e).__name__}")

    def index(self, e: syntax.Index) -> typed.Expr:
        base = self.expr(e.array)
        kind = base.type.kind
        if kind == TypeKind.DICT:
            return typed.DictLookup(base.type.element_type, base, self.convert(e.index, base.type.key_type))
        index = self.convert(e.index, Type.INT) if not isinstance(e.index, syntax.Constant) else self.expr(e.index)
        node = {TypeKind.ARRAY: typed.ArrayIndex, TypeKind.SLICE: typed.SliceIndex, TypeKind.STR: typed.StrIndex}[kind]
        element = Type.UINT8 if kind == TypeKind.STR else base.type.element_type
        return node(element, base, index)

    def field(self, e: syntax.Field) -> typed.FieldAccess:
        base = self.expr(e.base)
        through_pointer = base.type.kind == TypeKind.POINTER
        struct_type = base.type.element_type if through_pointer else base.type
        names = list(self.structs[struct_type.struct_name].fields)
        field_type = self.structs[struct_type.struct_name].fields[e.name]
        return typed.FieldAccess(field_type, base, e.name, names.index(e.name), through_pointer)

    def call(self, e: syntax.Call) -> typed.Expr:
        name, args = self.facts.calls.get(e.nid, (e.name, e.args))
        if name == 'print':
            return typed.Print(Type.VOID, self.expr(args[0]))
        if name == 'len':
            return typed.Len(Type.INT, self.expr(args[0]))
        if name == 'append':
            s = self.expr(args[0])
            return typed.Append(s.type, s, self.convert(args[1], s.type.element_type))
        if name == 'del':
            d = self.expr(args[0])
            return typed.DictDelete(Type.VOID, d, self.convert(args[1], d.type.key_type))
        if name in ('bytes', 'hornet_bytes'):  # analysis rewrites bytes(s) to the runtime function
            return typed.BytesFromStr(_BYTE_SLICE, self.expr(args[0]))
        if name in self.structs:
            field_types = self.structs[name].fields
            if e.kwargs is not None:
                given = dict(e.kwargs)
                values = tuple(self.convert(given[f], ft) if f in given else self.zero(ft)
                               for f, ft in field_types.items())
            else:
                values = tuple(self.convert(a, ft) for a, ft in zip(args, field_types.values()))
            return typed.StructLiteral(Type(TypeKind.STRUCT, struct_name=name), values)
        param_types, return_type = self.functions[name]
        converted = tuple(self.convert(a, pt) for a, pt in zip(args, param_types))
        if name in self.intrinsics:
            intrinsic = {'_raw_ptr': typed.StrRawPtr, '_raw_len': typed.StrRawLen,
                         '_from_raw_parts': typed.StrFromRawParts}[self.intrinsics[name]]
            return intrinsic(return_type, *converted)
        return typed.Call(return_type, name, 'extern' if name in self.externs else 'function', converted)

    def binary(self, e: syntax.Binary) -> typed.Expr:
        op, left_type, right_type = e.op, self.ty(e.left), self.ty(e.right)
        if op == BinaryOp.IN:
            container = self.expr(e.right)
            if container.type.kind == TypeKind.DICT:
                return typed.DictContains(Type.BOOL, self.convert(e.left, container.type.key_type), container)
            return typed.ElementContains(Type.BOOL, self.convert(e.left, container.type.element_type), container)
        if op in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL) and Type.NONE in (left_type, right_type):
            other = e.right if left_type == Type.NONE else e.left
            value = self.expr(other)
            if value.type.kind == TypeKind.SUM:
                test = typed.TagTest(Type.BOOL, value, Type.NONE)
                return test if op == BinaryOp.EQUAL else typed.Unary(Type.BOOL, UnaryOp.NOT, test)
            return typed.Binary(Type.BOOL, op, value, typed.NoneLit(value.type))
        if left_type == Type.STR and op == BinaryOp.ADD:
            return typed.StrConcat(Type.STR, self.expr(e.left), self.expr(e.right))
        if left_type == Type.STR and op in (BinaryOp.EQUAL, BinaryOp.NOT_EQUAL):
            return typed.StrCompare(Type.BOOL, op, self.expr(e.left), self.expr(e.right))
        left, right = self.expr(e.left), self.expr(e.right)
        if left.type != right.type and isinstance(right, typed.IntLit):
            right = typed.IntLit(left.type, right.value)
        elif left.type != right.type and isinstance(left, typed.IntLit):
            left = typed.IntLit(right.type, left.value)
        return typed.Binary(self.ty(e), op, left, right)


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
