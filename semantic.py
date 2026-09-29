"""Semantic analysis: name resolution, type checking, and control-flow checks.

Strict typing: no implicit conversions; integer operands must match exactly. Blocks
scope lexically and may shadow. Non-void functions must return on all paths.

Later passes rely on these annotations instead of re-resolving: resolved_type on expressions,
VarDecls, and Params; decl_id on Variable/Assign/IsCheck (id() of the declaring VarDecl/Param,
or (id(ForIn), index)); resolved_return_type, binding_types, narrowed_type; and the registries
stashed on Program.
"""

import argparse
from typing import Dict, List, Optional, Set, Tuple

from diagnostics import CompileError
from lexer import lex
from typesys import StructInfo, SumTypeInfo, Type, TypeKind
from desugar import mangle_method_name
from parser import (
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
    FieldAssign,
    For,
    ForIn,
    Function,
    If,
    Index,
    IndexAssign,
    IntrinsicDecl,
    IsCheck,
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
    'bool': Type.BOOL,
    'str': Type.STR,
}


def type_from_name(
    type_expr,
    structs: Dict[str, StructInfo],
    aliases: Dict[str, Type],
    node: Optional[Node] = None,
    sum_types: Dict[str, SumTypeInfo] = None,
) -> Type:
    """Resolve a parsed type expression to a Type."""
    if isinstance(type_expr, ArrayTypeExpr):
        element = type_from_name(type_expr.element_type, structs, aliases, node, sum_types)
        return Type(TypeKind.ARRAY, element_type=element, size=type_expr.size)
    if isinstance(type_expr, SliceTypeExpr):
        element = type_from_name(type_expr.element_type, structs, aliases, node, sum_types)
        return Type(TypeKind.SLICE, element_type=element)
    if isinstance(type_expr, PointerTypeExpr):
        # Pointer-to-pointer is rejected for now.
        pointee = type_from_name(type_expr.pointee_type, structs, aliases, node, sum_types)
        if pointee.kind == TypeKind.POINTER:
            raise SemanticError(
                "Pointer-to-pointer types aren't supported yet -- "
                "planned for later, once single-level pointers are proven out",
                node,
            )
        return Type(TypeKind.POINTER, element_type=pointee)
    if isinstance(type_expr, DictTypeExpr):
        key_type = type_from_name(type_expr.key_type, structs, aliases, node, sum_types)
        value_type = type_from_name(type_expr.value_type, structs, aliases, node, sum_types)
        if key_type not in _VALID_DICT_KEY_TYPES:
            raise SemanticError(
                f"'{key_type}' can't be a dict's own key type -- only "
                f"int, int8, uint8, int64, bool, and str are supported "
                f"as dict keys right now",
                node,
            )
        return Type(TypeKind.DICT, key_type=key_type, element_type=value_type)
    if type_expr in _TYPE_NAMES:
        return _TYPE_NAMES[type_expr]
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
        if isinstance(stmt, If):
            # Exhaustiveness was already verified by analyze_if.
            if stmt.is_match and _match_always_returns(stmt):
                return True
            if stmt.else_body is not None and always_returns(stmt.then_body) and always_returns(stmt.else_body):
                return True
        if isinstance(stmt, While):
            # `while true` without a reachable break never falls through.
            is_infinite = isinstance(stmt.condition, BoolLiteral) and stmt.condition.value is True
            if is_infinite and not contains_reachable_break(stmt.body):
                return True
    return False


def _match_always_returns(stmt: If) -> bool:
    """Whether every arm of a match chain returns."""
    current = stmt
    for i in range(stmt.match_arm_count):
        if not always_returns(current.then_body):
            return False
        if i < stmt.match_arm_count - 1:
            current = current.else_body[0]
    if current.else_body is None:
        return True
    return always_returns(current.else_body)


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
    return False


# Builtins; see check_call.
_BUILTIN_FUNCTION_NAMES = {'print', 'len', 'append', 'del'}


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

_INTEGER_TYPES = {Type.INT, Type.INT8, Type.UINT8, Type.INT64}
_VALID_DICT_KEY_TYPES = _INTEGER_TYPES | {Type.BOOL, Type.STR}

# Literal ranges for narrow integer types.
_NARROW_INT_RANGES = {
    Type.INT8: (-128, 127),
    Type.UINT8: (0, 255),
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


class SemanticAnalyzer:
    """Type- and scope-checks a Program."""

    def __init__(self):
        self.scopes: List[Dict[str, Tuple[Type, object]]] = []  # name -> (type, decl id)
        self.loop_depth = 0  # enclosing loop count
        self.functions: Dict[str, tuple] = {}  # name -> (param types, return type)
        self.structs: Dict[str, StructInfo] = {}
        self.methods: Dict[Tuple[str, str], Tuple[List[Type], Type, str]] = {}  # (struct, method) -> (param types, return type, mangled name)
        self.type_aliases: Dict[str, Type] = {}
        self.sum_types: Dict[str, SumTypeInfo] = {}
        self._narrowed_names: set = set()  # currently narrowed variable names

    def analyze(self, program: Program) -> None:
        # Order matters: 1. reserve struct names so aliases can target them.
        struct_registry = self._reserve_struct_names(program.structs)

        # 2. Resolve aliases before struct fields, which may use them.
        self.type_aliases = self._collect_type_aliases(program.type_aliases, struct_registry)
        program.type_alias_registry = self.type_aliases

        # 3. Resolve struct fields, then check cycles.
        self.structs = self._resolve_struct_fields(program.structs, struct_registry)
        program.struct_registry = self.structs

        # 3.5. Sum types need resolved structs.
        self.sum_types = self._resolve_sum_types(program.sum_types, self.structs)
        program.sum_type_registry = self.sum_types

        # 3.6. Methods, after struct resolution.
        self.methods = self._collect_methods(program)

        # 4. All signatures before any body, so order doesn't matter.
        self.functions = {}
        self.intrinsic_original_names = {}  # mangled name -> original_name
        for fn in program.functions:
            if fn.name in _BUILTIN_FUNCTION_NAMES:
                raise SemanticError(
                    f"'{fn.name}' is a builtin and can't be redefined as "
                    f"a function",
                    fn,
                )
            if fn.name in self.structs:
                raise SemanticError(
                    f"Function '{fn.name}' collides with a struct of the "
                    f"same name -- struct and function names share one "
                    f"namespace and can never be the same, since "
                    f"'{fn.name}(...)' would otherwise be ambiguous "
                    f"between a call and a struct literal",
                    fn,
                )
            if fn.name in self.type_aliases:
                raise SemanticError(
                    f"Function '{fn.name}' collides with a type alias "
                    f"of the same name -- function and type-alias "
                    f"names share one namespace and can never be the "
                    f"same",
                    fn,
                )
            if fn.name in self.sum_types:
                raise SemanticError(
                    f"Function '{fn.name}' collides with a sum type "
                    f"of the same name -- function and sum-type names "
                    f"share one namespace and can never be the same",
                    fn,
                )
            if fn.name in self.functions:
                raise SemanticError(f"Function '{fn.name}' is already declared", fn)
            param_types = [type_from_name(p.type, self.structs, self.type_aliases, p, self.sum_types) for p in fn.params]
            return_type = Type.VOID if fn.return_type is None else type_from_name(fn.return_type, self.structs, self.type_aliases, fn, self.sum_types)
            self.functions[fn.name] = (param_types, return_type)

        # 4.5. Externs share the function registry.
        for ext in program.extern_functions:
            self.check_extern_function_decl(ext)

        # 4.6. Intrinsics share the function registry.
        for ic in program.intrinsics:
            self.check_intrinsic_decl(ic)
        program.intrinsic_original_names = self.intrinsic_original_names
        program.function_registry = self.functions

        # 5. Check bodies, collecting at most one error per function.
        errors: List[SemanticError] = []
        for fn in program.functions:
            try:
                self.analyze_function(fn)
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

    def _resolve_sum_types(self, sum_type_defs: List[SumTypeDef], structs: Dict[str, StructInfo]) -> Dict[str, SumTypeInfo]:
        """Resolve sum type variants and check name collisions."""
        registry: Dict[str, SumTypeInfo] = {}
        for std in sum_type_defs:
            if std.name in _BUILTIN_FUNCTION_NAMES:
                raise SemanticError(
                    f"'{std.name}' is a builtin and can't be used as a "
                    f"sum type name",
                    std,
                )
            if std.name in structs:
                raise SemanticError(
                    f"Sum type '{std.name}' collides with a struct of "
                    f"the same name -- struct and sum-type names share "
                    f"one namespace and can never be the same",
                    std,
                )
            if std.name in self.type_aliases:
                raise SemanticError(
                    f"Sum type '{std.name}' collides with a type alias "
                    f"of the same name -- type-alias and sum-type names "
                    f"share one namespace and can never be the same",
                    std,
                )
            if std.name in registry:
                raise SemanticError(f"Sum type '{std.name}' is already declared", std)

            resolved_variants: List[Type] = []
            for variant_name in std.variants:
                if any(sd.name == variant_name for sd in sum_type_defs):
                    raise SemanticError(
                        f"Sum type '{std.name}' names '{variant_name}' as "
                        f"a variant, but '{variant_name}' is itself a sum "
                        f"type -- a sum type's variants can't include "
                        f"another sum type yet",
                        std,
                    )
                try:
                    variant_type = type_from_name(variant_name, structs, self.type_aliases, std)
                except SemanticError:
                    # Name what's allowed for a simple typo.
                    if not isinstance(variant_name, str):
                        raise
                    raise SemanticError(
                        f"Sum type '{std.name}' names '{variant_name}' as "
                        f"a variant, but '{variant_name}' isn't a declared "
                        f"struct or a valid scalar/str/array/slice/pointer "
                        f"type",
                        std,
                    )
                if variant_type in resolved_variants:
                    raise SemanticError(
                        f"Sum type '{std.name}' lists '{variant_type}' "
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
            seen_names: Set[str] = set()
            for md in sd.methods:
                if md.name in seen_names:
                    raise SemanticError(
                        f"Method '{md.name}' is already declared on "
                        f"struct '{sd.name}'",
                        md,
                    )
                seen_names.add(md.name)
                param_types = [type_from_name(p.type, self.structs, self.type_aliases, p, self.sum_types) for p in md.params]
                return_type = Type.VOID if md.return_type is None else type_from_name(md.return_type, self.structs, self.type_aliases, md, self.sum_types)
                methods[(sd.name, md.name)] = (param_types, return_type, mangle_method_name(sd.name, md.name))
        return methods

    def _collect_type_aliases(self, alias_defs: List[TypeAlias], structs: Dict[str, StructInfo]) -> Dict[str, Type]:
        """Resolve every alias fully, detecting cycles."""
        # Pass 1: reserve names, rejecting duplicates and collisions.
        seen: Dict[str, TypeAlias] = {}
        for ad in alias_defs:
            if ad.name in _BUILTIN_FUNCTION_NAMES:
                raise SemanticError(
                    f"'{ad.name}' is a builtin and can't be used as a "
                    f"type alias name",
                    ad,
                )
            if ad.name in structs:
                raise SemanticError(
                    f"Type alias '{ad.name}' collides with a struct of "
                    f"the same name -- struct and type-alias names "
                    f"share one namespace and can never be the same",
                    ad,
                )
            if ad.name in seen:
                raise SemanticError(f"Type alias '{ad.name}' is already declared", ad)
            seen[ad.name] = ad

        resolved: Dict[str, Type] = {}
        resolving: Set[str] = set()

        def resolve(name: str) -> Type:
            if name in resolved:
                return resolved[name]
            if name in resolving:
                raise SemanticError(
                    f"Type alias '{name}' is defined in terms of "
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
                return Type(TypeKind.ARRAY, element_type=resolve_target(target.element_type, alias_node), size=target.size)
            if isinstance(target, SliceTypeExpr):
                return Type(TypeKind.SLICE, element_type=resolve_target(target.element_type, alias_node))
            if target in _TYPE_NAMES:
                return _TYPE_NAMES[target]
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
                    f"'{sd.name}' is a builtin and can't be used as a "
                    f"struct name",
                    sd,
                )
            if sd.name in registry:
                raise SemanticError(f"Struct '{sd.name}' is already declared", sd)
            registry[sd.name] = None
        return registry

    def _resolve_struct_fields(self, struct_defs: List[StructDef], registry: Dict[str, StructInfo]) -> Dict[str, StructInfo]:
        """Resolve field types, then reject containment cycles."""
        for sd in struct_defs:
            fields: Dict[str, Type] = {}
            for f in sd.fields:
                if f.name in fields:
                    raise SemanticError(
                        f"Field '{f.name}' is already declared in struct '{sd.name}'",
                        f,
                    )
                fields[f.name] = type_from_name(f.field_type, registry, self.type_aliases, f)
            registry[sd.name] = StructInfo(name=sd.name, fields=fields)

        by_name = {sd.name: sd for sd in struct_defs}
        for sd in struct_defs:
            self._check_struct_contains(sd.name, registry, path=[], by_name=by_name)

        return registry

    def _check_struct_contains(self, name: str, registry: Dict[str, StructInfo], path: List[str], by_name: Dict[str, StructDef]) -> None:
        """DFS for struct containment cycles through fields and arrays (slices and pointers break cycles)."""
        if name in path:
            cycle = ' -> '.join(path + [name])
            raise SemanticError(
                f"Struct '{name}' cannot contain itself, directly or "
                f"transitively: {cycle}",
                by_name[name],
            )
        info = registry[name]
        for field_type in info.fields.values():
            contained = self._directly_embedded_struct_name(field_type)
            if contained is not None:
                self._check_struct_contains(contained, registry, path + [name], by_name)

    @staticmethod
    def _directly_embedded_struct_name(field_type: Type) -> Optional[str]:
        """Struct name embedded by `field_type` directly or via arrays, else None."""
        while field_type.kind == TypeKind.ARRAY:
            field_type = field_type.element_type
        return field_type.struct_name if field_type.kind == TypeKind.STRUCT else None

    @staticmethod
    def _contains_sum_type_at_any_array_depth(t: Type) -> bool:
        """Whether `t` is a sum type or an array of one."""
        while t.kind == TypeKind.ARRAY:
            t = t.element_type
        return t.kind == TypeKind.SUM

    def check_extern_function_decl(self, ext: ExternFunctionDecl) -> None:
        """Validate an extern signature (scalars and pointers only) and register it."""
        if ext.name in _BUILTIN_FUNCTION_NAMES:
            raise SemanticError(
                f"'{ext.name}' is a builtin and can't be redefined as "
                f"an extern function",
                ext,
            )
        if ext.name in self.structs:
            raise SemanticError(
                f"Extern function '{ext.name}' collides with a struct "
                f"of the same name -- struct and function names share "
                f"one namespace and can never be the same, since "
                f"'{ext.name}(...)' would otherwise be ambiguous "
                f"between a call and a struct literal",
                ext,
            )
        if ext.name in self.type_aliases:
            raise SemanticError(
                f"Extern function '{ext.name}' collides with a type "
                f"alias of the same name -- function and type-alias "
                f"names share one namespace and can never be the same",
                ext,
            )
        if ext.name in self.sum_types:
            raise SemanticError(
                f"Extern function '{ext.name}' collides with a sum "
                f"type of the same name -- function and sum-type "
                f"names share one namespace and can never be the same",
                ext,
            )
        if ext.name in self.functions:
            raise SemanticError(f"Function '{ext.name}' is already declared", ext)

        param_types = [type_from_name(p.type, self.structs, self.type_aliases, p, self.sum_types) for p in ext.params]
        return_type = Type.VOID if ext.return_type is None else type_from_name(ext.return_type, self.structs, self.type_aliases, ext, self.sum_types)

        for p, p_type in zip(ext.params, param_types):
            if p_type.kind in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT, TypeKind.SUM, TypeKind.STR):
                raise SemanticError(
                    f"Extern function '{ext.name}''s parameter '{p.name}' has "
                    f"type {p_type} -- only scalar and pointer types are "
                    f"supported in an extern function's signature for now "
                    f"(array/slice/struct/sum/str-typed parameters aren't yet)",
                    p,
                )
        if return_type.kind in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT, TypeKind.SUM, TypeKind.STR):
            raise SemanticError(
                f"Extern function '{ext.name}' returns {return_type} -- "
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
                f"'{ic.name}' is a builtin and can't be redefined as "
                f"an intrinsic",
                ic,
            )
        if ic.name in self.structs:
            raise SemanticError(
                f"Intrinsic '{ic.name}' collides with a struct "
                f"of the same name -- struct and function names share "
                f"one namespace and can never be the same, since "
                f"'{ic.name}(...)' would otherwise be ambiguous "
                f"between a call and a struct literal",
                ic,
            )
        if ic.name in self.type_aliases:
            raise SemanticError(
                f"Intrinsic '{ic.name}' collides with a type "
                f"alias of the same name -- function and type-alias "
                f"names share one namespace and can never be the same",
                ic,
            )
        if ic.name in self.sum_types:
            raise SemanticError(
                f"Intrinsic '{ic.name}' collides with a sum "
                f"type of the same name -- function and sum-type "
                f"names share one namespace and can never be the same",
                ic,
            )
        if ic.name in self.functions:
            raise SemanticError(f"Function '{ic.name}' is already declared", ic)

        param_types = [type_from_name(p.type, self.structs, self.type_aliases, p, self.sum_types) for p in ic.params]
        return_type = Type.VOID if ic.return_type is None else type_from_name(ic.return_type, self.structs, self.type_aliases, ic, self.sum_types)

        self.functions[ic.name] = (param_types, return_type)
        self.intrinsic_original_names[ic.name] = ic.original_name

    def analyze_function(self, fn: Function) -> None:
        self.scopes = [{}]
        self.loop_depth = 0
        # Params are locals; _declare also catches duplicates.
        for p in fn.params:
            p.resolved_type = type_from_name(p.type, self.structs, self.type_aliases, p, self.sum_types)
            self._declare(p.name, p.resolved_type, p, id(p))
        return_type = Type.VOID if fn.return_type is None else type_from_name(fn.return_type, self.structs, self.type_aliases, fn, self.sum_types)
        fn.resolved_return_type = return_type
        for stmt in fn.body:
            self.analyze_statement(stmt, return_type)
        # Void functions may fall off the end.
        if return_type != Type.VOID and not always_returns(fn.body):
            raise SemanticError(
                f"Function '{fn.name}' (declared to return {return_type}) "
                f"does not return a value on all code paths",
                fn,
            )


    def _push_scope(self) -> None:
        self.scopes.append({})

    def _pop_scope(self) -> None:
        self.scopes.pop()

    def _declare(self, name: str, type_: Type, node: Optional[Node], decl_id) -> None:
        """Declare in the innermost scope; shadowing outer scopes is allowed.
        decl_id identifies the storage: id() of a VarDecl/Param, or (id(ForIn), index)."""
        if name in self.scopes[-1]:
            raise SemanticError(f"Variable '{name}' is already declared in this scope", node)
        self.scopes[-1][name] = (type_, decl_id)

    def _resolve(self, name: str, node: Optional[Node] = None) -> Tuple[Type, object]:
        """(type, decl id) of `name`, innermost-first."""
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        raise SemanticError(f"Reference to undeclared variable '{name}'", node)

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
        """Equality, or NONE into slice/pointer/dict, or a variant into its sum type."""
        if value_type == target_type:
            return True
        if value_type == Type.NONE and target_type.kind in (TypeKind.SLICE, TypeKind.POINTER):
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
        if isinstance(expr, ArrayLiteral) and expr.type_expr is None and target_type.kind in (TypeKind.SLICE, TypeKind.ARRAY):
            array_type = self.check_array_literal(expr, expected_element_type=target_type.element_type)
            expr.resolved_type = array_type
            return target_type if target_type.kind == TypeKind.SLICE else array_type
        value_type = self.check_expr(expr)
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
                self._annotate_literal_resolved_type(expr, target_type)
                return target_type
        if value_type == Type.INT and target_type == Type.INT64:
            if self._as_folded_int_literal(expr) is not None:
                self._annotate_literal_resolved_type(expr, target_type)
                return target_type
        return value_type

    def _annotate_literal_resolved_type(self, expr: Node, target_type: Type) -> None:
        """Annotate a literal (and a negated literal's operand) with target_type."""
        expr.resolved_type = target_type
        if isinstance(expr, Unary) and expr.op == UnaryOp.NEGATE and isinstance(expr.operand, Constant):
            expr.operand.resolved_type = target_type

    def _check_expr_allowing_struct_literal(self, expr: Node) -> Type:
        """check_expr, but accepts struct literals."""
        if isinstance(expr, Call) and expr.name in self.structs:
            return self.check_struct_literal(expr)
        return self.check_expr(expr)

    def _check_value_flowing_into_allowing_struct_literal(self, expr: Node, target_type: Type) -> Type:
        """_check_value_flowing_into, but accepts struct literals."""
        if isinstance(expr, Call) and expr.name in self.structs:
            return self.check_struct_literal(expr)
        return self._check_value_flowing_into(expr, target_type)

    def analyze_var_decl(self, stmt: VarDecl) -> None:
        declared_type = type_from_name(stmt.var_type, self.structs, self.type_aliases, stmt, self.sum_types)
        if stmt.init is None and self._contains_sum_type_at_any_array_depth(declared_type):
            # Sum types have no zero value; require an initializer.
            raise SemanticError(
                f"'{stmt.name}' (declared {declared_type}) has no "
                f"initializer -- a sum type has no natural zero value, "
                f"so one is required here",
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
        stmt.resolved_type = declared_type
        self._declare(stmt.name, declared_type, stmt, id(stmt))

    def analyze_assign(self, stmt: Assign) -> None:
        if stmt.name in self._narrowed_names:
            raise SemanticError(
                f"Cannot reassign '{stmt.name}' while it's narrowed by "
                f"an enclosing 'is' check -- assign to a different "
                f"variable instead",
                stmt,
            )
        declared_type, stmt.decl_id = self._resolve(stmt.name, stmt)
        value_type = self._check_value_flowing_into_allowing_struct_literal(stmt.value, declared_type)
        if not self._types_compatible(value_type, declared_type):
            raise SemanticError(
                f"Cannot assign a value of type {value_type} to '{stmt.name}' "
                f"(declared {declared_type})",
                stmt,
            )

    def _check_compound_assign(self, compound_op: BinaryOp, target_type: Type, target_expr_for_check: Node, value_expr: Node, stmt: Node) -> None:
        """Check a compound assignment via a synthetic Binary."""
        if target_type == Type.STR and compound_op == BinaryOp.ADD:
            raise SemanticError(
                f"Compound assignment ('+=') to a str-typed target isn't "
                f"supported yet -- string concatenation has a different "
                f"codegen shape than arithmetic compound assignment; "
                f"write it as a plain '=' instead",
                stmt,
            )
        synthetic = Binary(op=compound_op, left=target_expr_for_check, right=value_expr, line=stmt.line, col=stmt.col, file=stmt.file)
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
            target_expr = Unary(op=UnaryOp.DEREFERENCE, operand=stmt.pointer, line=stmt.line, col=stmt.col, file=stmt.file)
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
            subject_type = self.check_expr(stmt.condition.subject)
            # binding_decl is built once here; see IsCheck.
            if subject_type.kind == TypeKind.SUM:
                stmt.condition.binding_decl = VarDecl(
                    name=stmt.condition.variable_name,
                    var_type=subject_type.sum_type_name,
                    init=stmt.condition.subject,
                    line=stmt.condition.line,
                    col=stmt.condition.col,
                    file=stmt.condition.file,
                )
                stmt.condition.binding_decl.resolved_type = subject_type
            self._declare(stmt.condition.variable_name, subject_type, stmt.condition, id(stmt.condition.binding_decl))

        condition_type = self.check_expr(stmt.condition)
        if condition_type != Type.BOOL:
            raise SemanticError(
                f"'if' condition must be bool, got {condition_type} "
                f"(no implicit int-to-bool conversion -- try `x != 0` "
                f"instead of `x`)",
                stmt.condition,
            )

        if stmt.is_match:
            self._check_match_exhaustiveness(stmt)

        self._push_scope()
        # Narrow the IsCheck variable within then_body only.
        narrowed_name = None
        if isinstance(stmt.condition, IsCheck):
            narrowed_name = stmt.condition.variable_name
            narrowed_type = type_from_name(stmt.condition.type_name, self.structs, self.type_aliases, stmt.condition)
            stmt.condition.narrowed_type = narrowed_type
            self._declare(narrowed_name, narrowed_type, stmt.condition, stmt.condition.decl_id)
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

    def _check_match_exhaustiveness(self, stmt: If) -> None:
        """Reject duplicate arms; require exhaustiveness without an else."""
        subject_name = stmt.condition.variable_name
        subject_type = self._lookup(subject_name, stmt.condition)
        sum_type_info = self.sum_types[subject_type.sum_type_name]

        seen: Dict[Type, IsCheck] = {}
        current = stmt
        for i in range(stmt.match_arm_count):
            arm_condition = current.condition
            arm_type = type_from_name(arm_condition.type_name, self.structs, self.type_aliases, arm_condition)
            if arm_type in seen:
                raise SemanticError(
                    f"'{arm_condition.type_name}' is tested more than once in "
                    f"this match on '{subject_name}'",
                    arm_condition,
                )
            seen[arm_type] = arm_condition
            if i < stmt.match_arm_count - 1:
                current = current.else_body[0]

        if current.else_body is not None:
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
        """`for a[, b] in iterable:` over arrays, slices, and dicts."""
        if not isinstance(stmt.iterable, (Variable, Field, Index, Slice, ArrayLiteral, DictLiteral, Call)):
            raise SemanticError(
                f"'for ... in' requires a variable, field, index, "
                f"slice, or array/dict literal as its own iterable, "
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
        if iterable_type.kind not in (TypeKind.ARRAY, TypeKind.SLICE, TypeKind.DICT):
            raise SemanticError(
                f"'for ... in' requires an array, slice, or dict as its own "
                f"iterable, got {iterable_type}",
                stmt.iterable,
            )
        num_bindings = len(stmt.binding_names)
        if iterable_type.kind == TypeKind.DICT:
            binding_types = [iterable_type.key_type, iterable_type.element_type][:num_bindings]
        else:
            binding_types = [Type.INT, iterable_type.element_type] if num_bindings == 2 else [iterable_type.element_type]
        stmt.binding_types = binding_types
        self._push_scope()
        for i, (name, binding_type) in enumerate(zip(stmt.binding_names, binding_types)):
            self._declare(name, binding_type, stmt, (id(stmt), i))
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
        """Type-check `expr` and set expr.resolved_type."""
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
        expr.resolved_type = result
        return result

    def check_dict_literal(self, expr: DictLiteral) -> Type:
        """`dict[K]V{...}`; keys must be distinct constants."""
        key_type = type_from_name(expr.key_type, self.structs, self.type_aliases, expr, self.sum_types)
        value_type = type_from_name(expr.value_type, self.structs, self.type_aliases, expr, self.sum_types)
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
            declared_type = type_from_name(expr.type_expr, self.structs, self.type_aliases, expr, self.sum_types)
            if len(expr.elements) != declared_type.size:
                raise SemanticError(
                    f"Array literal declares type {declared_type} (size "
                    f"{declared_type.size}), but has {len(expr.elements)} "
                    f"element(s)",
                    expr,
                )
            for i, element in enumerate(expr.elements, start=1):
                element_type = self._check_value_flowing_into_allowing_struct_literal(element, declared_type.element_type)
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
                f"Struct '{base_type.struct_name}' has no field '{field_name}'",
                base_expr,
            )
        return struct_info.fields[field_name]

    def check_struct_literal(self, expr: Call) -> Type:
        """`Name(args)`: positional struct literal; must be exhaustive."""
        struct_info = self.structs[expr.name]
        field_items = list(struct_info.fields.items())
        if expr.kwargs is not None:
            return self._check_named_struct_literal(expr, struct_info, field_items)
        if len(expr.args) != len(field_items):
            field_names = ', '.join(name for name, _ in field_items)
            raise SemanticError(
                f"Struct literal for '{expr.name}' expects "
                f"{len(field_items)} argument(s) (one per field, in "
                f"declaration order: {field_names}), got {len(expr.args)}",
                expr,
            )
        for i, (arg, (field_name, field_type)) in enumerate(zip(expr.args, field_items), start=1):
            arg_type = self._check_value_flowing_into_allowing_struct_literal(arg, field_type)
            if not self._types_compatible(arg_type, field_type):
                raise SemanticError(
                    f"Argument {i} to struct literal '{expr.name}' "
                    f"(field '{field_name}') should be {field_type}, "
                    f"got {arg_type}",
                    arg,
                )
        result = Type(TypeKind.STRUCT, struct_name=expr.name)
        expr.resolved_type = result
        return result

    def _check_named_struct_literal(self, expr: Call, struct_info: StructInfo, field_items: list) -> Type:
        """`Name(f=v, ...)`: named struct literal; omitted fields are zero."""
        field_types = struct_info.fields
        valid_names = ', '.join(name for name, _ in field_items)
        seen = set()
        for field_name, value in expr.kwargs:
            if field_name not in field_types:
                raise SemanticError(
                    f"Struct literal for '{expr.name}' has no field "
                    f"'{field_name}' -- valid fields are: {valid_names}",
                    expr,
                )
            if field_name in seen:
                raise SemanticError(
                    f"Field '{field_name}' specified more than once in "
                    f"struct literal for '{expr.name}'",
                    expr,
                )
            seen.add(field_name)
            value_type = self._check_value_flowing_into_allowing_struct_literal(value, field_types[field_name])
            expected_type = field_types[field_name]
            if not self._types_compatible(value_type, expected_type):
                raise SemanticError(
                    f"Field '{field_name}' of struct literal '{expr.name}' "
                    f"should be {expected_type}, got {value_type}",
                    value,
                )
        result = Type(TypeKind.STRUCT, struct_name=expr.name)
        expr.resolved_type = result
        return result

    def _check_method_call(self, expr: Call) -> Type:
        """Resolve `receiver.name(args)` and rewrite it in place to a mangled call."""
        if expr.kwargs is not None:
            raise SemanticError(
                f"'{expr.name}(...)' uses named arguments, which are not "
                f"supported for method calls",
                expr,
            )
        receiver_type = self._check_expr_allowing_struct_literal(expr.receiver)
        if receiver_type.kind == TypeKind.POINTER and receiver_type.element_type.kind == TypeKind.STRUCT:
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
                f"Struct '{receiver_type.struct_name}' has no method "
                f"'{expr.name}'",
                expr,
            )
        param_types, return_type, mangled_name = self.methods[key]
        if len(expr.args) != len(param_types):
            raise SemanticError(
                f"Method '{expr.name}' on '{receiver_type.struct_name}' "
                f"expects {len(param_types)} argument(s), got "
                f"{len(expr.args)}",
                expr,
            )
        for i, (arg, expected_type) in enumerate(zip(expr.args, param_types), start=1):
            actual_type = self._check_value_flowing_into_allowing_struct_literal(arg, expected_type)
            if not self._types_compatible(actual_type, expected_type):
                raise SemanticError(
                    f"Argument {i} to method '{expr.name}' on "
                    f"'{receiver_type.struct_name}' should be "
                    f"{expected_type}, got {actual_type}",
                    arg,
                )
        expr.args = [expr.receiver] + expr.args
        expr.name = mangled_name
        expr.receiver = None
        expr.resolved_type = return_type
        return return_type

    def check_call(self, expr: Call) -> Type:
        if expr.receiver is not None:
            # Receiver present means method call, checked first.
            return self._check_method_call(expr)
        if expr.name in self.structs:
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
        if expr.name == 'print':
            return self.check_print_call(expr)
        if expr.name == 'len':
            return self.check_len_call(expr)
        if expr.name == 'append':
            return self.check_append_call(expr)
        if expr.name == 'del':
            return self.check_del_call(expr)
        if expr.name not in self.functions:
            raise SemanticError(f"Call to undeclared function '{expr.name}'", expr)
        param_types, return_type = self.functions[expr.name]

        if len(expr.args) != len(param_types):
            raise SemanticError(
                f"Function '{expr.name}' expects {len(param_types)} "
                f"argument(s), got {len(expr.args)}",
                expr,
            )
        for i, (arg, expected_type) in enumerate(zip(expr.args, param_types), start=1):
            actual_type = self._check_value_flowing_into_allowing_struct_literal(arg, expected_type)
            if not self._types_compatible(actual_type, expected_type):
                raise SemanticError(
                    f"Argument {i} to '{expr.name}' should be "
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
                "in a slice-typed variable first (e.g. `[]int s = none`), "
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
        return Type.INT

    def check_variable(self, expr: Variable) -> Type:
        t, expr.decl_id = self._resolve(expr.name, expr)
        return t

    def check_is_check(self, expr: IsCheck) -> Type:
        """`NAME is T` / `EXPR is T as NAME`; T must be a variant."""
        variable_type, expr.decl_id = self._resolve(expr.variable_name, expr)
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
        narrowed_type = type_from_name(expr.type_name, self.structs, self.type_aliases, expr)
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
        operand_type = self._check_expr_allowing_struct_literal(expr.operand)
        if expr.op in (UnaryOp.NEGATE, UnaryOp.COMPLEMENT):
            if operand_type not in _INTEGER_TYPES:
                raise SemanticError(
                    f"'{expr.op.symbol()}' requires an int, int8, "
                    f"uint8, or int64 operand, got {operand_type}",
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
            # Only variables, struct literals, and chains rooted in a variable.
            is_struct_literal = isinstance(expr.operand, Call) and expr.operand.name in self.structs
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
            pointee_type = operand_type.element_type
            if pointee_type.kind == TypeKind.SUM:
                raise SemanticError(
                    f"'*' on a pointer to {pointee_type} (a sum type) "
                    f"isn't supported yet as a value -- write through it "
                    f"with '*p = value', or access a field directly "
                    f"(auto-deref already handles 'p.field')",
                    expr,
                )
            return pointee_type
        raise SemanticError(f"No semantic rule for unary operator: {expr.op}", expr)

    def check_cast(self, expr: Cast) -> Type:
        """`T(expr)` between integer types; literals range-checked against T."""
        target_type = type_from_name(expr.target_type, self.structs, self.type_aliases, expr, self.sum_types)
        if target_type == Type.INT64 and self._as_folded_int_literal(expr.expr) is not None:
            self._annotate_literal_resolved_type(expr.expr, Type.INT64)
            source_type = Type.INT64
        else:
            source_type = self.check_expr(expr.expr)
        if target_type not in _INTEGER_TYPES or source_type not in _INTEGER_TYPES:
            raise SemanticError(
                f"Cannot cast {source_type} to {target_type} -- casting "
                f"is only supported between int, int8, uint8, and int64 "
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
                f"(int, int8, uint8, or int64) or two str operands, "
                f"got {left_type} and {right_type}",
                expr,
            )

        if op in _INT_ONLY_BINARY_OPS:
            return self._require_same_integer_type(left_type, right_type, op, expr)

        if op in _ORDERING_OPS:
            self._require_same_integer_type(left_type, right_type, op, expr)
            return Type.BOOL

        if op in _EQUALITY_OPS:
            # Slices, pointers, and dicts compare to `none`.
            none_vs_nilable = (
                (left_type == Type.NONE and right_type.kind in (TypeKind.SLICE, TypeKind.POINTER, TypeKind.DICT)) or
                (right_type == Type.NONE and left_type.kind in (TypeKind.SLICE, TypeKind.POINTER, TypeKind.DICT))
            )
            if none_vs_nilable:
                return Type.BOOL

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
                f"integer type (int, int8, uint8, or int64), got "
                f"{left_type} and {right_type}",
                node,
            )
        return left_type


# Entry points

def analyze(program: Program) -> None:
    SemanticAnalyzer().analyze(program)


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
