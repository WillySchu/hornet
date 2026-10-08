"""Types as they are written: a parsed type expression to the Type it names.

A name in a type is resolved in the scope of the file it is written in (scopes.py), against the
program's declarations. Both the declaration phase and the body checker need this, each with its own
place in the program, so the scope is given at each call."""

from typing import Dict, Optional

from parser import ArrayTypeExpr, DictTypeExpr, Node, PointerTypeExpr, QualifiedTypeExpr, SliceTypeExpr
from semantic.errors import SemanticError
from typesys import INTEGER_TYPES, EnumInfo, StructInfo, SumTypeInfo, Type, TypeKind

TYPE_NAMES = {
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

VALID_DICT_KEY_TYPES = INTEGER_TYPES | {Type.BOOL, Type.STR}


def type_from_name(
        type_expr,
        structs: Dict[str, StructInfo],
        aliases: Dict[str, Type],
        node: Optional[Node] = None,
        sum_types: Dict[str, SumTypeInfo] = None,
        resolve=None,
        enums: Dict[str, EnumInfo] = None,
        array_sizes: Dict[int, int] = None,
) -> Type:
    """Resolve a parsed type expression to a Type. `resolve` maps a name as written in its file (or an
    `alias.Name`) to its declaration's key (see scopes.py). `array_sizes` (Facts.array_sizes) has the
    value of each `[EXPR]T` size that isn't a plain number."""
    if isinstance(type_expr, ArrayTypeExpr):
        element = type_from_name(type_expr.element_type, structs, aliases, node, sum_types, resolve, enums, array_sizes)
        size = type_expr.size if isinstance(type_expr.size, int) else array_sizes[type_expr.nid]
        return Type(TypeKind.ARRAY, element_type=element, size=size)
    if isinstance(type_expr, SliceTypeExpr):
        element = type_from_name(type_expr.element_type, structs, aliases, node, sum_types, resolve, enums, array_sizes)
        return Type(TypeKind.SLICE, element_type=element)
    if isinstance(type_expr, PointerTypeExpr):
        # Pointer-to-pointer is rejected for now.
        pointee = type_from_name(type_expr.pointee_type, structs, aliases, node, sum_types, resolve, enums, array_sizes)
        if pointee.kind == TypeKind.POINTER:
            raise SemanticError(
                "Pointer-to-pointer types aren't supported yet -- "
                "planned for later, once single-level pointers are proven out",
                node,
            )
        return Type(TypeKind.POINTER, element_type=pointee)
    if isinstance(type_expr, DictTypeExpr):
        key_type = type_from_name(type_expr.key_type, structs, aliases, node, sum_types, resolve, enums, array_sizes)
        value_type = type_from_name(
            type_expr.value_type, structs, aliases, node, sum_types, resolve, enums, array_sizes)
        if key_type not in VALID_DICT_KEY_TYPES and key_type.kind != TypeKind.ENUM:
            raise SemanticError(
                f"'{key_type}' can't be a dict's own key type -- only "
                f"int, int8, uint8, int32, bool, str, and enums are supported "
                f"as dict keys right now",
                node,
            )
        return Type(TypeKind.DICT, key_type=key_type, element_type=value_type)
    if isinstance(type_expr, QualifiedTypeExpr) and resolve is not None:
        type_expr = resolve(type_expr)
    if type_expr in TYPE_NAMES:
        return TYPE_NAMES[type_expr]
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


class TypeResolver:
    """Written types to Types, against `decls` (semantic/declarations.py), which may still be filling."""

    def __init__(self, decls, module_set, array_sizes: Dict[int, int]):
        self.decls = decls
        self.module_set = module_set
        self.array_sizes = array_sizes  # Facts.array_sizes

    def key(self, name, scope):
        """A type name (or `alias.Name`) as written in a file, whose scope is `scope` -> its declaration's key."""
        if isinstance(name, QualifiedTypeExpr):
            if name.nid not in self.module_set.qualified:  # scopes.py left it: `Enum.Member`
                raise SemanticError(f"'{name.module}.{name.name}' is an enum's member, not a type", name)
            return self.module_set.qualified[name.nid]
        return (scope.resolve(name) or name) if isinstance(name, str) else name

    def resolve(self, type_expr, node: Node, scope, sums: bool = True) -> Type:
        decls = self.decls
        return type_from_name(type_expr, decls.structs, decls.type_aliases, node, decls.sum_types if sums else None,
                              resolve=lambda name: self.key(name, scope), enums=decls.enums,
                              array_sizes=self.array_sizes)

    def return_type(self, decl, scope) -> Type:
        """A def's or extern's return type: VOID when it declares none, NEVER for `never`."""
        if decl.return_type is None:
            return Type.VOID
        return Type.NEVER if decl.return_type == 'never' else self.resolve(decl.return_type, decl, scope)
