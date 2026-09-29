"""Stateless type helpers shared by the IR builder mixins."""

from parser import Node, Field, ForIn, Index, Unary, UnaryOp, Variable
from semantic import Type, TypeKind, StructInfo

from ir.errors import IRError

# Composite types: copied through addresses, not held in registers.
COMPOSITE_KINDS = {TypeKind.ARRAY, TypeKind.SLICE, TypeKind.STRUCT, TypeKind.SUM, TypeKind.STR, TypeKind.DICT}


def is_composite_addressable(expr: Node) -> bool:
    """Whether `expr` has an address to copy a composite value from."""
    if isinstance(expr, (Variable, Field, Index)):
        return True
    return isinstance(expr, Unary) and expr.op == UnaryOp.DEREFERENCE

# Sum layout: 4-byte tag, then the largest variant's payload.
SUM_TYPE_TAG_WIDTH = 4


def type_byte_width(t: Type, structs: dict[str, StructInfo], sum_types: dict) -> int:
    """Storage size of `t` in bytes."""
    if t == Type.INT8 or t == Type.UINT8:
        return 1
    if t == Type.INT64:
        return 8
    if t.kind == TypeKind.ARRAY:
        return t.size * type_byte_width(t.element_type, structs, sum_types)
    if t.kind == TypeKind.SLICE:
        return 24
    if t.kind == TypeKind.STR:
        return 16  # {ptr, len}
    if t.kind == TypeKind.STRUCT:
        return sum(type_byte_width(field_type, structs, sum_types) for field_type in structs[t.struct_name].fields.values())
    if t.kind == TypeKind.SUM:
        variant_widths = (
            type_byte_width(variant_type, structs, sum_types)
            for variant_type in sum_types[t.sum_type_name].variants
        )
        return SUM_TYPE_TAG_WIDTH + max(variant_widths)
    if t.kind == TypeKind.POINTER:
        return 8  # pointer
    if t.kind == TypeKind.DICT:
        return 24  # {buckets_ptr, count, capacity}
    return 4  # INT, BOOL


def is_wide_type(t: Type) -> bool:
    """Whether `t` needs 8-byte moves (int64, pointer)."""
    return t in (Type.INT64,) or t.kind == TypeKind.POINTER


def leaf_type(t: Type) -> Type:
    """Innermost non-array element type."""
    while t.kind == TypeKind.ARRAY:
        t = t.element_type
    return t


def type_of(expr: Node) -> Type:
    """expr.resolved_type; IRError if semantic analysis didn't run."""
    if expr.resolved_type is None:
        raise IRError(
            f"{expr!r} has no resolved type -- semantic.analyze() "
            f"must run before codegen (see compile_to_asm)"
        )
    return expr.resolved_type


def for_in_binding_types(stmt: ForIn, iterable_type: Type) -> list:
    """Binding types for a ForIn, matching semantic.analyze_for_in."""
    num_bindings = len(stmt.binding_names)
    if iterable_type.kind == TypeKind.DICT:
        return [iterable_type.key_type, iterable_type.element_type][:num_bindings]
    if num_bindings == 2:
        return [Type.INT, iterable_type.element_type]
    return [iterable_type.element_type]
