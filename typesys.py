"""Semantic types shared by the frontend, IR, and backend."""

from dataclasses import dataclass
from enum import auto, Enum
from typing import Dict, List, Optional


class TypeKind(Enum):
    INT = auto()
    INT8 = auto()
    UINT8 = auto()
    INT32 = auto()
    BOOL = auto()
    STR = auto()
    ARRAY = auto()
    SLICE = auto()
    STRUCT = auto()
    SUM = auto()
    POINTER = auto()
    DICT = auto()
    VOID = auto()
    NEVER = auto()
    NONE = auto()


@dataclass(frozen=True)
class Type:
    """A type. Scalars use kind alone; ARRAY/SLICE/POINTER use element_type (ARRAY also size);
    DICT uses key_type and element_type (value); STRUCT/SUM are nominal by name.
    Frozen for structural equality and hashing.
    """
    kind: TypeKind
    element_type: Optional['Type'] = None  # ARRAY/SLICE/POINTER pointee; DICT value
    size: Optional[int] = None  # ARRAY only
    struct_name: Optional[str] = None  # STRUCT only
    sum_type_name: Optional[str] = None  # SUM only
    key_type: Optional['Type'] = None  # DICT only

    def __str__(self) -> str:
        if self.kind == TypeKind.ARRAY:
            return f"[{self.size}]{self.element_type}"
        if self.kind == TypeKind.SLICE:
            return f"[]{self.element_type}"
        if self.kind == TypeKind.STRUCT:  # as declared, without the module prefix of its key
            return self.struct_name.rsplit('$', 1)[-1]
        if self.kind == TypeKind.SUM:
            return self.sum_type_name.rsplit('$', 1)[-1]
        if self.kind == TypeKind.POINTER:
            return f"*{self.element_type}"
        if self.kind == TypeKind.DICT:
            return f"dict[{self.key_type}]{self.element_type}"
        return self.kind.name.lower()


Type.INT = Type(TypeKind.INT)
Type.INT8 = Type(TypeKind.INT8)
Type.UINT8 = Type(TypeKind.UINT8)
Type.INT64 = Type.INT  # `int64` is another spelling of `int`
Type.INT32 = Type(TypeKind.INT32)
Type.BOOL = Type(TypeKind.BOOL)
Type.STR = Type(TypeKind.STR)
# VOID (no declared return) and NONE (`none`) have no source spelling. NEVER is the return type
# `never`: the function doesn't return, so a call to it has no value and ends its path.
Type.VOID = Type(TypeKind.VOID)
Type.NEVER = Type(TypeKind.NEVER)
Type.NONE = Type(TypeKind.NONE)


@dataclass
class StructInfo:
    """A struct's name and ordered fields; order fixes layout."""
    name: str
    fields: Dict[str, Type]


@dataclass
class SumTypeInfo:
    """A sum type's name and ordered variants; index is the discriminant. Variants are structs, scalars, or str."""
    name: str
    variants: List[Type]


# Sum layout: 4-byte tag, then the largest variant's payload.
SUM_TYPE_TAG_WIDTH = 4


def type_byte_width(t: Type, structs: dict[str, StructInfo], sum_types: dict) -> int:
    """Storage size of `t` in bytes."""
    if t == Type.INT8 or t == Type.UINT8:
        return 1
    if t == Type.INT:
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
    if t.kind == TypeKind.NONE:
        return 0  # a sum type's `none` variant: the tag alone
    if t.kind == TypeKind.DICT:
        return 8  # a pointer to the shared header: copies alias (ir/dicts.py)
    return 4  # INT32, BOOL


def is_wide_type(t: Type) -> bool:
    """Whether `t` needs 8-byte moves (int, pointer)."""
    return t == Type.INT or t.kind == TypeKind.POINTER


def leaf_type(t: Type) -> Type:
    """Innermost non-array element type."""
    while t.kind == TypeKind.ARRAY:
        t = t.element_type
    return t
