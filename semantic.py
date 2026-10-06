"""Semantic analysis: name resolution, type checking, and control-flow checks; its result is the
typed tree (typed_ast.py), which every later stage consumes.

Strict typing: no implicit conversions; integer operands must match exactly (an integer literal
takes the other operand's type). Blocks scope lexically and may shadow. Non-void functions must
return on all paths.

The parser's tree is never changed: checking records what it learns in Facts, keyed by node number,
and _TypedTreeBuilder builds each function's typed tree from those facts at the end of analyze().
"""

import argparse
import dataclasses
import os
from typing import Dict, List, Optional, Set, Tuple

from diagnostics import CompileError
from lexer import lex
from typesys import EnumInfo, StructInfo, SumTypeInfo, Type, TypeKind
from folding import fold_binary_op, fold_cast, fold_unary_op
import parser as syntax
import typed_ast as typed
from scopes import BUILTIN_FUNCTION_NAMES, build_module_set, display_name as shown
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
    DictLiteral,
    DictTypeExpr,
    ExprStmt,
    ExternFunctionDecl,
    Field,
    QualifiedTypeExpr,
    For,
    ForIn,
    Function,
    MethodDef,
    Param,
    If,
    Index,
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
    SliceLiteral,
    SliceTypeExpr,
    StringLiteral,
    EnumDef,
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
    enums: Dict[str, EnumInfo] = None,
) -> Type:
    """Resolve a parsed type expression to a Type. `resolve` maps a name as written in its file (or an
    `alias.Name`) to its declaration's key (see scopes.py)."""
    if isinstance(type_expr, ArrayTypeExpr):
        element = type_from_name(type_expr.element_type, structs, aliases, node, sum_types, resolve, enums)
        size = type_expr.size if isinstance(type_expr.size, int) else _array_sizes[type_expr.nid]
        return Type(TypeKind.ARRAY, element_type=element, size=size)
    if isinstance(type_expr, SliceTypeExpr):
        element = type_from_name(type_expr.element_type, structs, aliases, node, sum_types, resolve, enums)
        return Type(TypeKind.SLICE, element_type=element)
    if isinstance(type_expr, PointerTypeExpr):
        # Pointer-to-pointer is rejected for now.
        pointee = type_from_name(type_expr.pointee_type, structs, aliases, node, sum_types, resolve, enums)
        if pointee.kind == TypeKind.POINTER:
            raise SemanticError(
                "Pointer-to-pointer types aren't supported yet -- "
                "planned for later, once single-level pointers are proven out",
                node,
            )
        return Type(TypeKind.POINTER, element_type=pointee)
    if isinstance(type_expr, DictTypeExpr):
        key_type = type_from_name(type_expr.key_type, structs, aliases, node, sum_types, resolve, enums)
        value_type = type_from_name(type_expr.value_type, structs, aliases, node, sum_types, resolve, enums)
        if key_type not in _VALID_DICT_KEY_TYPES and key_type.kind != TypeKind.ENUM:
            raise SemanticError(
                f"'{key_type}' can't be a dict's own key type -- only "
                f"int, int8, uint8, int32, bool, str, and enums are supported "
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
    if enums is not None and type_expr in enums:
        return Type(TypeKind.ENUM, enum_name=type_expr)
    raise SemanticError(f"Unknown type '{type_expr}'", node)


def _always_ends(statements: List[Node], types: dict, ends: tuple) -> bool:
    """Whether control never runs off the end of `statements`: every path reaches a statement of
    one of the classes `ends`, a call to a `never` function (by `types`, the checked types by node
    number), or a `while true` without a reachable break."""
    for stmt in statements:
        if isinstance(stmt, ends):
            return True
        if isinstance(stmt, ExprStmt) and types.get(stmt.expr.nid) == Type.NEVER:
            return True
        if isinstance(stmt, Match) and all(_always_ends(body, types, ends) for _, body in stmt.arms) and (
                stmt.else_body is None or _always_ends(stmt.else_body, types, ends)):  # exhaustive without an else
            return True
        if isinstance(stmt, If) and stmt.else_body is not None and _always_ends(stmt.then_body, types, ends) and \
                _always_ends(stmt.else_body, types, ends):
            return True
        if isinstance(stmt, While) and isinstance(stmt.condition, BoolLiteral) and stmt.condition.value is True \
                and not contains_reachable_break(stmt.body):
            return True
    return False


def always_returns(statements: List[Node], types: dict) -> bool:
    """Whether every path through `statements` returns (or never finishes)."""
    return _always_ends(statements, types, (Return,))


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


def _file_and_line(node: Node) -> str:
    """`file (line N)` of a declaration, for an error reported somewhere else."""
    return f"{os.path.basename(node.file) if node.file else '<input>'} (line {node.line})"


def always_leaves(statements: List[Node], types: dict) -> bool:
    """Whether every path through `statements` returns, breaks out of or continues the enclosing
    loop, or never finishes."""
    return _always_ends(statements, types, (Return, Break, Continue))


def _conditions_after(stmt: If, types: dict) -> Tuple[List[Node], Optional[Node]]:
    """(the conditions known false, the condition known true) once control passes the `if`/`elif`/
    `else` chain `stmt`. When every branch but one always leaves, that one ran: the conditions before
    it were false, and its own was true (it has none if it is the `else`, written or not)."""
    branches = []  # (condition, body); the `else` last, with no condition
    while True:
        branches.append((stmt.condition, stmt.then_body))
        if stmt.else_body is not None and len(stmt.else_body) == 1 and isinstance(stmt.else_body[0], If):
            stmt = stmt.else_body[0]
            continue
        branches.append((None, stmt.else_body or []))
        break
    staying = [i for i, (_, body) in enumerate(branches) if not always_leaves(body, types)]
    if len(staying) != 1:
        return [], None
    return [condition for condition, _ in branches[:staying[0]]], branches[staying[0]][0]


def _assigned_names(node) -> Set[str]:
    """The names of the variables assigned anywhere under `node` (a statement, or a list of them)."""
    names: Set[str] = set()
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, (list, tuple)):
            stack.extend(n)
        elif isinstance(n, Node):
            if isinstance(n, Assign) and isinstance(n.target, Variable):
                names.add(n.target.name)
            stack.extend(getattr(n, f.name) for f in dataclasses.fields(n))
    return names


def _conjuncts(condition: Node) -> List[Node]:
    """The checks a condition's top-level `and`s join, in order (the condition itself if it has none)."""
    if isinstance(condition, Binary) and condition.op == BinaryOp.AND:
        return _conjuncts(condition.left) + _conjuncts(condition.right)
    return [condition]


def _both(a: dict, b: dict) -> dict:
    """What is known when both of two sets of narrowing facts hold (SemanticAnalyzer._when)."""
    out = dict(a)
    for decl_id, (name, variants) in b.items():
        out[decl_id] = (name, variants & out[decl_id][1]) if decl_id in out else (name, variants)
    return out


def _either(a: dict, b: dict) -> dict:
    """What is known when one of two sets of narrowing facts holds, but not which."""
    return {decl_id: (name, variants | b[decl_id][1]) for decl_id, (name, variants) in a.items() if decl_id in b}


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
    enum_members: dict = dataclasses.field(default_factory=dict)  # `Enum.Member` Field -> (enum's key, index)
    enum_checks: dict = dataclasses.field(default_factory=dict)  # IsCheck on an enum -> (member's index, the enum)
    enum_lens: dict = dataclasses.field(default_factory=dict)  # `len(Enum)` Call -> the number of members
    formats: dict = dataclasses.field(default_factory=dict)  # `format(...)` Call -> its template's text
    enum_ins: dict = dataclasses.field(default_factory=dict)  # `n in Enum` Binary -> the enum's key


class SemanticAnalyzer:
    """Type- and scope-checks a Program."""

    def __init__(self):
        self.scopes: List[Dict[str, Tuple[Type, object]]] = []  # name -> (type, decl id)
        self.loop_depth = 0  # enclosing loop count
        self.functions: Dict[str, tuple] = {}  # name -> (param types, return type)
        self.structs: Dict[str, StructInfo] = {}
        # (struct, method) -> (param types, return type, mangled name)
        self.methods: Dict[Tuple[str, str], Tuple[List[Type], Type, str]] = {}
        self.pointer_receivers: set = set()  # (struct, method) with a `*receiver`
        self.type_aliases: Dict[str, Type] = {}
        self.sum_types: Dict[str, SumTypeInfo] = {}
        self._declared: List[Set[str]] = []  # per scope: the names declared in it (not merely narrowed)
        # Sum variables an `is` check has narrowed here: decl id -> the variants it may still hold.
        self._possible: Dict[int, frozenset] = {}
        self._undo: list = []  # what _restrict changed, undone by _end_region

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
        # Order matters: 0. enums depend on nothing, and constant array sizes are evaluated before any
        # type is resolved.
        self.enums = self._collect_enums(program.enums)
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
            if shown(fn.name) in BUILTIN_FUNCTION_NAMES:
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
            self.functions[fn.name] = (param_types, self._return_type(fn))

        # 4.5. Externs share the function registry.
        self.extern_names = {ext.name for ext in program.extern_functions}
        self._extern_decls = {}  # name -> its declaration, for the duplicate's error
        for ext in program.extern_functions:
            self._enter(ext)
            self.check_extern_function_decl(ext)

        # 4.6. Intrinsics share the function registry.
        for ic in program.intrinsics:
            self._enter(ic)
            self.check_intrinsic_decl(ic)

        self._check_enum_name_collisions(program)

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

        # 6. The typed tree: what later stages consume.
        return _TypedTreeBuilder(self).program(self.all_functions)

    # -- names across modules

    def _enter(self, decl: Node) -> None:
        """Check `decl` (a top-level declaration, under its key) in its own file's scope."""
        self.scope = self.module_set.scope_of[decl.nid]

    def _resolve_type_name(self, name):
        """A type name (or `alias.Name`) as written in the current file -> its declaration's key."""
        if isinstance(name, QualifiedTypeExpr):
            if name.nid not in self.module_set.qualified:  # scopes.py left it: `Enum.Member`
                raise SemanticError(f"'{name.module}.{name.name}' is an enum's member, not a type", name)
            return self.module_set.qualified[name.nid]
        return (self.scope.resolve(name) or name) if isinstance(name, str) else name

    def _type(self, type_expr, node: Node, sums: bool = True) -> Type:
        return type_from_name(type_expr, self.structs, self.type_aliases, node, self.sum_types if sums else None,
                              resolve=self._resolve_type_name, enums=self.enums)

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

    def _collect_enums(self, enum_defs: List[EnumDef]) -> Dict[str, EnumInfo]:
        """The enum registry: each enum's members, distinct and at least one."""
        registry: Dict[str, EnumInfo] = {}
        for ed in enum_defs:
            self._enter(ed)
            if shown(ed.name) in BUILTIN_FUNCTION_NAMES:
                raise SemanticError(f"'{shown(ed.name)}' is a builtin and can't be used as an enum name", ed)
            if ed.name in registry:
                raise SemanticError(f"Enum '{shown(ed.name)}' is already declared", ed)
            names: List[str] = []
            for member in ed.members:
                if member.name in names:
                    raise SemanticError(
                        f"Member '{member.name}' is already declared in enum '{shown(ed.name)}'", member)
                names.append(member.name)
            registry[ed.name] = EnumInfo(name=ed.name, members=names)
        return registry

    def _check_enum_name_collisions(self, program: Program) -> None:
        for ed in program.enums:
            for kind, table in (('function', self.functions), ('struct', self.structs), ('constant', self.const_decls),
                                ('type alias', self.type_aliases), ('sum type', self.sum_types)):
                if ed.name in table:
                    self._enter(ed)
                    raise SemanticError(f"Enum '{shown(ed.name)}' collides with a {kind} of the same name", ed)

    def _check_const_name_collisions(self, program: Program) -> None:
        for name, cd in self.const_decls.items():
            for kind, table in (('function', self.functions), ('struct', self.structs),
                                ('type alias', self.type_aliases), ('sum type', self.sum_types)):
                if name in table:
                    raise SemanticError(f"Constant '{shown(name)}' collides with a {kind} of the same name", cd)
            if shown(name) in BUILTIN_FUNCTION_NAMES:
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
        if const_type not in _INTEGER_TYPES and const_type not in (Type.BOOL, Type.STR) \
                and const_type.kind != TypeKind.ENUM:
            raise SemanticError(
                f"Constant '{shown(name)}' has type {const_type} -- constants must be an integer type, bool, str, "
                f"or an enum", cd)
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
        """Value of an already type-checked constant expression; an enum's is its member's index."""
        t = self.facts.types.get(expr.nid)
        if expr.nid in self.facts.enum_members:  # `Enum.Member`
            return self.facts.enum_members[expr.nid][1]
        if isinstance(expr, Call) and self._callee(expr) in self.enums:  # `Enum(n)`
            value, members = self._const_eval(expr.args[0]), self.enums[t.enum_name].members
            if not 0 <= value < len(members):
                raise SemanticError(
                    f"{value} is not a member of {t} (its members' values are 0 to {len(members) - 1})", expr)
            return value
        if isinstance(expr, (Constant, ByteLiteral)) and isinstance(expr.value, int):
            return expr.value
        if isinstance(expr, (BoolLiteral, StringLiteral)):
            return expr.value
        if isinstance(expr, Call) and expr.nid in self.facts.enum_lens:
            return self.facts.enum_lens[expr.nid]
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
                if expr.op in _ORDERING_OPS:  # each character is a byte, so this is their order
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
        if isinstance(expr, Cast) and t in _INTEGER_TYPES:
            return fold_cast(t, self._const_eval(expr.expr), self.facts.types.get(expr.expr.nid))
        source_type = self.facts.types.get(expr.expr.nid) if isinstance(expr, Cast) else None
        if t == Type.STR and source_type is not None and source_type.kind == TypeKind.ENUM:  # `str(Enum.Member)`
            return self.enums[source_type.enum_name].members[self._const_eval(expr.expr)]
        raise SemanticError(
            "A constant's value must be built from literals, other constants, operators, and integer casts", expr)

    def _resolve_sum_types(
            self, sum_type_defs: List[SumTypeDef], structs: Dict[str, StructInfo]) -> Dict[str, SumTypeInfo]:
        """Resolve sum type variants and check name collisions."""
        registry: Dict[str, SumTypeInfo] = {}
        sum_names = {std.name: None for std in sum_type_defs}
        for std in sum_type_defs:
            if shown(std.name) in BUILTIN_FUNCTION_NAMES:
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
                    # A sum can't be a variant itself (above), but a pointer to one, or a slice or
                    # dict of them, can: the names of the sums are enough to resolve those.
                    variant_type = type_from_name(
                        variant_name, structs, self.type_aliases, std, sum_names, resolve=self._resolve_type_name,
                        enums=self.enums)
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
                return_type = self._return_type(md)
                methods[(sd.name, md.name)] = (param_types, return_type, mangle_method_name(sd.name, md.name))
                if md.receiver_is_pointer:
                    self.pointer_receivers.add((sd.name, md.name))
        return methods

    def _collect_type_aliases(self, alias_defs: List[TypeAlias], structs: Dict[str, StructInfo]) -> Dict[str, Type]:
        """Resolve every alias fully, detecting cycles."""
        # Pass 1: reserve names, rejecting duplicates and collisions.
        seen: Dict[str, TypeAlias] = {}
        for ad in alias_defs:
            if shown(ad.name) in BUILTIN_FUNCTION_NAMES:
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
            if target in self.enums:
                return Type(TypeKind.ENUM, enum_name=target)
            raise SemanticError(
                f"Unknown type '{target}' in a type alias's own target "
                f"-- expected int, bool, str, a struct or enum name, or "
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
            if shown(sd.name) in BUILTIN_FUNCTION_NAMES:
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
                    f.field_type, registry, self.type_aliases, f, sum_names, resolve=self._resolve_type_name,
                    enums=self.enums)
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
        if shown(ext.name) in BUILTIN_FUNCTION_NAMES:
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
        if ext.name in self._extern_decls:
            raise SemanticError(
                f"Extern '{shown(ext.name)}' is already declared in {_file_and_line(self._extern_decls[ext.name])} "
                f"-- declare an extern once and import it where else it is needed", ext)
        if ext.name in self.functions:
            # A function of the entry file: only there is a function's key its bare name, like an extern's.
            function = next((fn for fn in self.all_functions if fn.name == ext.name), None)
            raise SemanticError(
                f"Function '{shown(ext.name)}' has the same name as an extern declared in {_file_and_line(ext)} "
                f"-- rename the function", function if function is not None else ext)
        self._extern_decls[ext.name] = ext

        param_types = [self._type(p.type, p) for p in ext.params]
        return_type = self._return_type(ext)

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
        if shown(ic.name) in BUILTIN_FUNCTION_NAMES:
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
        self.scopes, self._declared = [{}], [set()]
        self._possible, self._undo = {}, []
        self._bindable, self._bound = set(), set()  # `as NAME` checks that may bind here; those that have
        self._assignments = []  # decl ids, as variables are assigned on paths that reach what follows
        self.loop_depth = 0
        # Params are locals; _declare also catches duplicates.
        for p in fn.params:
            param_type = self._type(p.type, p)
            self.facts.types[p.nid] = param_type
            self.facts.symbols[p.nid] = self.symbols.new(p.name, 'param', param_type, p)
            self._declare(p.name, param_type, p, self.facts.symbols[p.nid].id)
        return_type = self._return_type(fn)
        self.facts.returns[fn.nid] = return_type
        self._analyze_block(fn.body, return_type)
        if return_type == Type.NEVER and not _always_ends(fn.body, self.facts.types, ()):
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
        """A def's or extern's return type: VOID when it declares none, NEVER for `never`."""
        if decl.return_type is None:
            return Type.VOID
        return Type.NEVER if decl.return_type == 'never' else self._type(decl.return_type, decl)

    def _push_scope(self) -> None:
        self.scopes.append({})
        self._declared.append(set())

    def _pop_scope(self) -> None:
        self.scopes.pop()
        self._declared.pop()

    def _declare(self, name: str, type_: Type, node: Optional[Node], decl_id) -> None:
        """Declare in the innermost scope; shadowing outer scopes is allowed.
        decl_id is the declaration's Symbol.id."""
        if name in self._declared[-1]:
            raise SemanticError(f"Variable '{shown(name)}' is already declared in this scope", node)
        self._declared[-1].add(name)
        self.scopes[-1][name] = (type_, decl_id)

    # -- narrowing: what `is` checks have established about a sum variable, within a region

    def _possible_variants(self, decl_id) -> frozenset:
        """The variants the sum variable declared as `decl_id` may hold here."""
        if decl_id in self._possible:
            return self._possible[decl_id]
        return frozenset(self.sum_types[self.symbols[decl_id].type.sum_type_name].variants)

    def _restrict(self, name: str, decl_id, possible: frozenset) -> None:
        """From here to the end of the region (_end_region), sum variable `name` holds one of
        `possible`: with one variant left it has that variant's type (`none` has nothing to read,
        so the variable stays its sum). Assigning to it ends that (_forget)."""
        narrowed = len(possible) == 1 and Type.NONE not in possible
        entry = (next(iter(possible)) if narrowed else self.symbols[decl_id].type, decl_id)
        scope = self.scopes[-1]
        self._undo.append((scope, name, scope.get(name), entry, decl_id, self._possible.get(decl_id)))
        scope[name] = entry
        self._possible[decl_id] = possible

    def _end_region(self, mark: int) -> None:
        """Undo every _restrict since `mark` (a length of self._undo)."""
        while len(self._undo) > mark:
            scope, name, previous, entry, decl_id, possible = self._undo.pop()
            if scope.get(name) is entry:  # not since replaced by a declaration of the same name
                if previous is None:
                    del scope[name]
                else:
                    scope[name] = previous
            if possible is None:
                del self._possible[decl_id]
            else:
                self._possible[decl_id] = possible

    def _when(self, condition: Node) -> Tuple[dict, dict]:
        """What `condition` (already checked) says about sum variables when it is true, and when it is
        false: each a dict, decl id -> (name, the variants the variable may then hold). `x is T` (and
        `x == none`) says so directly; `and`, `or`, and `not` combine what their operands say."""
        if isinstance(condition, Unary) and condition.op == UnaryOp.NOT:
            when_true, when_false = self._when(condition.operand)
            return when_false, when_true
        if isinstance(condition, Binary) and condition.op in _LOGICAL_OPS:
            left_true, left_false = self._when(condition.left)
            right_true, right_false = self._when(condition.right)
            if condition.op == BinaryOp.AND:
                return _both(left_true, right_true), _either(left_false, right_false)
            return _either(left_true, right_true), _both(left_false, right_false)
        name = decl_id = variant = None
        if isinstance(condition, IsCheck) and condition.nid in self.facts.narrowed:
            name, decl_id, variant = condition.variable_name, self.facts.decls.get(condition.nid), \
                self.facts.narrowed[condition.nid]
        elif isinstance(condition, Binary) and condition.op in _EQUALITY_OPS:
            for side, other in ((condition.left, condition.right), (condition.right, condition.left)):
                if isinstance(side, Variable) and isinstance(other, NoneLiteral):  # `x == none` is `x is none`
                    name, decl_id, variant = side.name, self.facts.decls.get(side.nid), Type.NONE
        if decl_id is None or self.symbols[decl_id].type.kind != TypeKind.SUM:
            return {}, {}
        variants = frozenset(self.sum_types[self.symbols[decl_id].type.sum_type_name].variants)
        holds, excluded = {decl_id: (name, frozenset({variant}))}, {decl_id: (name, variants - {variant})}
        if isinstance(condition, Binary) and condition.op == BinaryOp.NOT_EQUAL:
            return excluded, holds
        return holds, excluded

    def _apply(self, known: Optional[dict]) -> None:
        """Narrow by `known` (one of _when's results) until the end of the region."""
        for decl_id, (name, variants) in (known or {}).items():
            # An `as NAME` binding ends with its `if`, so the name may be gone, or another variable's.
            entry = next((scope[name] for scope in reversed(self.scopes) if name in scope), None)
            if entry is not None and entry[1] == decl_id:
                self._restrict(name, decl_id, self._possible_variants(decl_id) & variants)

    def _forget(self, decl_id) -> None:
        """The variable declared as `decl_id` has been assigned: nothing is known about it from here
        to the end of the region, whatever was known before."""
        name = self.symbols[decl_id].name
        entry = next((scope[name] for scope in reversed(self.scopes) if name in scope), None)
        if decl_id in self._possible and entry is not None and entry[1] == decl_id:
            self._restrict(name, decl_id, frozenset(self.sum_types[self.symbols[decl_id].type.sum_type_name].variants))

    def _forget_assigned_in(self, loop_syntax) -> None:
        """Before a loop: forget what is known about every variable its body assigns, since the body
        may already have run when its condition and its statements are reached."""
        for name in _assigned_names(loop_syntax):
            entry = next((scope[name] for scope in reversed(self.scopes) if name in scope), None)
            if entry is not None and entry[1] is not None:
                self._forget(entry[1])

    def _analyze_block(self, statements: List[Node], return_type: Type) -> None:
        """Check a block's statements in the current scope. What an `if` or a `while` establishes for
        the code after it narrows the rest of the block; what a statement assigns is forgotten."""
        mark = len(self._undo)
        for stmt in statements:
            first = len(self._assignments)
            self.analyze_statement(stmt, return_type)
            assigned = set(self._assignments[first:])  # here, or in nested blocks that reach what follows
            if isinstance(stmt, If):
                known_false, known_true = _conditions_after(stmt, self.facts.types)
                for condition in known_false:
                    self._apply(self._when(condition)[1])
                if known_true is not None:
                    self._apply(self._when(known_true)[0])
            for decl_id in assigned:
                self._forget(decl_id)
            if isinstance(stmt, While) and not contains_reachable_break(stmt.body):
                self._apply(self._when(stmt.condition)[1])  # the loop ended because its condition was false
        self._end_region(mark)

    def _analyze_body(self, statements: List[Node], return_type: Type, known: Optional[dict] = None) -> None:
        """A block in a scope of its own, narrowed by `known` (one of _when's results)."""
        mark, first = len(self._undo), len(self._assignments)
        self._push_scope()
        self._apply(known)
        self._analyze_block(statements, return_type)
        self._pop_scope()
        self._end_region(mark)
        if always_leaves(statements, self.facts.types):
            del self._assignments[first:]  # what follows the enclosing statement isn't reached from here

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
        enum = self.enums.get(self.scope.resolve(name))
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
        """`target = value` or `target op= value`, to a name, field, element, or pointee."""
        target = stmt.target
        if isinstance(target, Variable):
            if self._const_key(target.name) is not None and not any(target.name in scope for scope in self.scopes):
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
            self._forget(self.facts.decls[target.nid])

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
        binders = [c for c in _conjuncts(stmt.condition) if isinstance(c, IsCheck) and c.binds]
        whole = isinstance(stmt.condition, IsCheck) and stmt.condition.binds
        if binders:
            self._push_scope()
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
        when_true, when_false = self._when(stmt.condition)
        self._analyze_body(stmt.then_body, return_type, when_true)
        if binders and not whole:
            self._pop_scope()  # in the `else`, a check after an `and` may never have run: its name is unset
        if stmt.else_body is not None:
            self._analyze_body(stmt.else_body, return_type, when_false)
        if whole:
            self._pop_scope()

    def analyze_match(self, stmt: 'Match', return_type: Type) -> None:
        """Each arm narrows the subject within its own body, like `if NAME is T:` (the order of checks,
        and so of errors, is that of the equivalent `if`/`else` chain)."""
        first_check = stmt.arms[0][0]
        has_binding = stmt.subject is not None
        if has_binding:
            self._push_scope()
            self._declare_is_binding(first_check)
        self.check_expr(first_check)
        if first_check.nid in self.facts.enum_checks:
            self._analyze_enum_match(stmt, return_type)
            if has_binding:
                self._pop_scope()
            return
        tested = set()
        for i, (check, body) in enumerate(stmt.arms):
            self.check_expr(check)
            if i == 0:
                self._check_match_exhaustiveness(stmt)
            variant = self.facts.narrowed[check.nid]
            tested.add(variant)
            self._analyze_body(body, return_type, self._when(check)[0])
        if stmt.else_body is not None:  # the subject is none of the arms' variants
            decl_id = self.facts.decls[first_check.nid]
            self._analyze_body(stmt.else_body, return_type,
                               {decl_id: (stmt.variable_name, self._possible_variants(decl_id) - tested)})
        if has_binding:
            self._pop_scope()

    def _analyze_enum_match(self, stmt: 'Match', return_type: Type) -> None:
        """`match` on an enum: each arm names a member, and the arms cover every member unless there is
        an `else`. Nothing is narrowed."""
        subject_type = self.facts.enum_checks[stmt.arms[0][0].nid][1]
        members = self.enums[subject_type.enum_name].members
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
        if isinstance(check.subject, Variable) and self.facts.decls[check.subject.nid] in self._possible:
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
        self._declare(check.variable_name, subject_type, check, sym.id)

    def _check_match_exhaustiveness(self, stmt: 'Match') -> None:
        """Reject duplicate arms; require exhaustiveness without an else."""
        subject_name = stmt.variable_name
        subject_type = self.symbols[self.facts.decls[stmt.arms[0][0].nid]].type  # as declared, not as narrowed
        sum_type_info = self.sum_types[subject_type.sum_type_name]

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
        self._forget_assigned_in(stmt.body)
        condition_type = self.check_expr(stmt.condition)
        if condition_type != Type.BOOL:
            raise SemanticError(
                f"'while' condition must be bool, got {condition_type} "
                f"(no implicit int-to-bool conversion -- try `x != 0` "
                f"instead of `x`)",
                stmt.condition,
            )

        self.loop_depth += 1
        self._analyze_body(stmt.body, return_type, self._when(stmt.condition)[0])
        self.loop_depth -= 1

    def analyze_for(self, stmt: For, return_type: Type) -> None:
        """`for init; cond; increment:`; one scope spans all clauses."""
        self._push_scope()
        self.analyze_statement(stmt.init, return_type)
        self._forget_assigned_in([stmt.body, stmt.increment])
        condition_type = self.check_expr(stmt.condition)
        if condition_type != Type.BOOL:
            raise SemanticError(
                f"'for' condition must be bool, got {condition_type} "
                f"(no implicit int-to-bool conversion -- try `x != 0` "
                f"instead of `x`)",
                stmt.condition,
            )
        self.loop_depth += 1
        mark = len(self._undo)
        self._apply(self._when(stmt.condition)[0])  # the condition narrows the body, as a `while`'s does
        self._analyze_block(stmt.body, return_type)
        self._end_region(mark)
        self.loop_depth -= 1
        self.analyze_statement(stmt.increment, return_type)
        self._pop_scope()

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
        self._push_scope()
        symbols = [
            self.symbols.new(name, 'binding', t, stmt) for name, t in zip(stmt.binding_names, binding_types)
        ]
        self.facts.for_symbols[stmt.nid] = symbols
        for name, binding_type, sym in zip(stmt.binding_names, binding_types, symbols):
            self._declare(name, binding_type, stmt, sym.id)
        self._forget_assigned_in(stmt.body)
        self.loop_depth += 1
        self._analyze_block(stmt.body, return_type)
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
                constant_key = ('enum', f"{shown(enum)}.{self.enums[enum].members[index]}")
            if constant_key is not None:
                if constant_key in seen_constant_keys:
                    raise SemanticError(
                        f"Dict literal lists the key {constant_key[1]!r} more than once",
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
        `expected_element_type` when there is one, else the first element's type."""
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
        enum = self._enum_named_by(expr.base)
        if enum is not None:  # `Enum.Member`
            members = self.enums[enum].members
            if expr.name not in members:
                raise SemanticError(
                    f"Enum '{shown(enum)}' has no member '{expr.name}' (its members: {', '.join(members)})", expr)
            self.facts.enum_members[expr.nid] = (enum, members.index(expr.name))
            return Type(TypeKind.ENUM, enum_name=enum)
        return self._check_struct_and_field(expr.base, expr.name)

    def _enum_named_by(self, expr: Node) -> Optional[str]:
        """The key of the enum that `expr` names (`Enum`, or `alias.Enum`), unless a variable has the name."""
        if isinstance(expr, Variable) and not any(expr.name in scope for scope in self.scopes):
            key = self.scope.resolve(expr.name)
        elif isinstance(expr, Field):
            key = self.module_set.qualified.get(expr.nid)
        else:
            return None
        return key if key in self.enums else None

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
        struct_info = self.structs[name]
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
        if self._hidden(receiver_type.struct_name, expr.name):
            raise SemanticError(
                f"Method '{expr.name}' of '{shown(receiver_type.struct_name)}' is not visible outside the module "
                f"that defines the struct -- names starting with '_' are private to their own module", expr)
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
        if name in self.enums:
            return self.check_enum_conversion(expr, name)
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
        if arg_type in (Type.VOID, Type.NEVER):
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
            text = self._const_eval(template)
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
        members = self.enums[enum].members
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
            self.facts.enum_lens[expr.nid] = len(self.enums[enum].members)
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
            narrowed = self.facts.decls[expr.nid] in self._possible
        if variable_type.kind == TypeKind.ENUM:  # `NAME is Member`: an equality test; nothing is narrowed
            members = self.enums[variable_type.enum_name].members
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
        sum_type_info = self.sum_types[variable_type.sum_type_name]
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
            struct_info = self.structs[t.struct_name]
            return all(self._is_comparable_type(field_type) for field_type in struct_info.fields.values())
        if t.kind in (TypeKind.SLICE, TypeKind.SUM, TypeKind.DICT):
            return False
        return True  # integers, bool, str, pointers

    def check_binary(self, expr: Binary) -> Type:
        if expr.op == BinaryOp.MULTIPLY:
            self._reject_typed_literal_read_as_multiplication(expr)
        if expr.op == BinaryOp.IN and self._enum_named_by(expr.right) is not None:
            # `n in Enum`: whether the integer `n` is a member's value (so `Enum(n)` wouldn't panic).
            enum = self._enum_named_by(expr.right)
            left_type = self.check_expr(expr.left)
            if left_type not in _INTEGER_TYPES:
                raise SemanticError(
                    f"'in' with the enum {shown(enum)} on its right tests an integer, got {left_type}", expr.left)
            self.facts.enum_ins[expr.nid] = enum
            return Type.BOOL
        left_type = self._check_expr_allowing_struct_literal(expr.left)
        mark = len(self._undo)
        if expr.op in _LOGICAL_OPS:  # the right side runs only when the left was true (`and`) or false (`or`)
            self._apply(self._when(expr.left)[0 if expr.op == BinaryOp.AND else 1])
        right_type = self._check_expr_allowing_struct_literal(expr.right)
        self._end_region(mark)
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
                if isinstance(expr.right, ArrayLiteral) and expr.right.type_expr is None \
                        and left_type in _NARROW_INT_RANGES:
                    right_type = self.check_array_literal(expr.right, expected_element_type=left_type)
                    self.facts.types[expr.right.nid] = right_type
                elif self._as_folded_int_literal(expr.left) is not None:
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
        self.enums = analyzer.enums
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
        if const_type.kind == TypeKind.ENUM:
            return typed.EnumMember(const_type, self.enums[const_type.enum_name].members[value], value)
        return typed.IntLit(self.ty(e), value)

    def program(self, functions) -> typed.Program:
        return typed.Program(tuple(self.function(fn) for fn in functions), self.structs, self.sum_types,
                             self.symbols, self.enums)

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

    def _at(self, node, source):
        """Give typed `node` the position of parser node `source`, unless it already has one."""
        if isinstance(node, (typed.Expr, typed.Stmt)) and not node.line:
            for name in ('line', 'col', 'file'):
                object.__setattr__(node, name, getattr(source, name))
        return node

    def statement(self, s) -> list:
        """Typed statements for one parser statement (a narrowing binding adds its declaration)."""
        return [self._at(node, s) for node in self._statement(s)]

    def _statement(self, s) -> list:
        if isinstance(s, syntax.VarDecl):
            return [self.declare(s)]
        if isinstance(s, syntax.Assign):
            if isinstance(s.target, syntax.Variable):
                symbol = self.symbols[self.facts.decls[s.target.nid]]
                target = typed.Local(symbol.type, symbol)
            else:
                target = self.expr(s.target)
            value = self.convert(s.value, target.type)
            return [typed.Assign(target, value) if s.op is None else typed.CompoundAssign(target, s.op, value)]
        if isinstance(s, syntax.ExprStmt):
            return [typed.ExprStmt(self.expr(s.expr))]
        if isinstance(s, syntax.Return):
            return [typed.Return(None if s.value is None else self.convert(s.value, self.return_type))]
        if isinstance(s, syntax.If):
            return self.if_statement(s)
        if isinstance(s, syntax.Match):
            before, subject = self.narrowing_subject(s.arms[0][0])
            if s.arms[0][0].nid in self.facts.enum_checks:  # an `if`/`elif` chain of equality tests
                chain = self.block(s.else_body)
                for check, body in reversed(s.arms):
                    chain = (self._at(typed.If(self.enum_test(subject, check), self.block(body), chain), check),)
                return before + list(chain)
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
        if not (isinstance(s.condition, syntax.IsCheck) and s.condition.binds):
            return [typed.If(self.expr(s.condition), self.block(s.then_body), self.block(s.else_body))]
        before, subject = self.narrowing_subject(s.condition)  # declares the `as NAME` binding
        return before + [typed.If(self.is_test(subject, s.condition), self.block(s.then_body),
                                  self.block(s.else_body))]

    def is_test(self, subject: typed.Expr, check: syntax.IsCheck) -> typed.Expr:
        """`subject is T`: a tag test on a sum, an equality test on an enum."""
        if check.nid in self.facts.enum_checks:
            return self.enum_test(subject, check)
        return typed.TagTest(Type.BOOL, subject, self.facts.narrowed[check.nid])

    def enum_test(self, subject: typed.Local, check: syntax.IsCheck) -> typed.Binary:
        """`subject is Member` on an enum: `subject == Enum.Member`. The subject may be a sum's variable
        narrowed to the enum."""
        index, enum = self.facts.enum_checks[check.nid]
        value = subject if subject.type == enum else typed.Payload(enum, subject)
        member = typed.EnumMember(enum, self.enums[enum.enum_name].members[index], index)
        return typed.Binary(Type.BOOL, BinaryOp.EQUAL, value, member)

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
        return self._at(self._convert(e, target), e)

    def _convert(self, e, target: Type) -> typed.Expr:
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
        return self._at(self._expr(e), e)

    def _expr(self, e) -> typed.Expr:
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
                return typed.Payload(self.ty(e), local)  # narrowed by an `is` check
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
        if isinstance(e, syntax.SliceLiteral):
            elements = tuple(self.convert(x, self.ty(e).element_type) for x in e.elements)
            return typed.SliceLiteral(self.ty(e), elements) if elements else typed.EmptySlice(self.ty(e))
        if (
                isinstance(e, syntax.Slice) and isinstance(e.array, syntax.ArrayLiteral)
                and e.low is None and e.high is None
        ):
            # `[...][:]`: new storage holding the elements, like a slice literal.
            elements = tuple(self.convert(x, self.ty(e).element_type) for x in e.array.elements)
            return typed.SliceLiteral(self.ty(e), elements) if elements else typed.EmptySlice(self.ty(e))
        if isinstance(e, syntax.Slice):
            base = self.expr(e.array)
            kind = {TypeKind.ARRAY: 'array', TypeKind.SLICE: 'slice', TypeKind.STR: 'str'}[base.type.kind]
            low = None if e.low is None else self.expr(e.low)
            high = None if e.high is None else self.expr(e.high)
            return typed.SliceOf(self.ty(e), kind, base, low, high)
        if isinstance(e, syntax.Field):
            if e.nid in self.facts.enum_members:
                enum, index = self.facts.enum_members[e.nid]
                return typed.EnumMember(self.ty(e), self.enums[enum].members[index], index)
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
            if self.ty(e) == Type.STR and value.type.kind == TypeKind.ENUM:
                if isinstance(value, typed.EnumMember):  # a member or a constant: its name is known here
                    return typed.StrLit(Type.STR, value.name)
                return typed.EnumName(Type.STR, value)
            if self.ty(e) == Type.STR:
                return (typed.StrFromByte if value.type == Type.UINT8 else typed.StrFromBytes)(Type.STR, value)
            return typed.IntCast(self.ty(e), value)
        if isinstance(e, syntax.Binary):
            return self.binary(e)
        if isinstance(e, syntax.IsCheck):
            if e.variable_name is None:
                return self.is_test(self.expr(e.subject), e)
            if e.binds:  # one of a condition's checks joined by `and` (a whole condition is if_statement's)
                symbol = self.facts.symbols[self.facts.bindings[e.nid].nid]
                return typed.Bind(Type.BOOL, symbol, self.convert(e.subject, symbol.type),
                                  self.is_test(typed.Local(symbol.type, symbol), e))
            symbol = self.symbols[self.facts.decls[e.nid]]
            return self.is_test(typed.Local(symbol.type, symbol), e)
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
            if e.nid in self.facts.enum_lens:
                return typed.IntLit(Type.INT, self.facts.enum_lens[e.nid])
            return typed.Len(Type.INT, self.expr(args[0]))
        if name == 'append':
            s = self.expr(args[0])
            return typed.Append(s.type, s, self.convert(args[1], s.type.element_type))
        if name == 'del':
            d = self.expr(args[0])
            return typed.DictDelete(Type.VOID, d, self.convert(args[1], d.type.key_type))
        if name == 'bytes':
            return typed.BytesFromStr(_BYTE_SLICE, self.expr(args[0]))
        if name == 'panic':
            return typed.Panic(Type.NEVER, self.expr(args[0]))
        if name == 'format':
            template = self.facts.formats[e.nid]
            if len(args) == 1:  # nothing to fill in: the text itself
                return typed.StrLit(Type.STR, typed.format_pieces(template)[0])
            return typed.Format(Type.STR, template, tuple(self.expr(a) for a in args[1:]))
        if name in self.enums:  # `Enum(n)`
            value = self.expr(args[0])
            if isinstance(value, typed.IntLit):  # checked to be a member
                return typed.EnumMember(self.ty(e), self.enums[name].members[value.value], value.value)
            return typed.EnumFromInt(self.ty(e), value)
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
        if e.nid in self.facts.enum_ins:
            enum = self.facts.enum_ins[e.nid]
            return typed.EnumContains(Type.BOOL, self.expr(e.left), Type(TypeKind.ENUM, enum_name=enum))
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
        if left_type == Type.STR and op in _EQUALITY_OPS | _ORDERING_OPS:
            return typed.StrCompare(Type.BOOL, op, self.expr(e.left), self.expr(e.right))
        return typed.Binary(self.ty(e), op, self.expr(e.left), self.expr(e.right))


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
