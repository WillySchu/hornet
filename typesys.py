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
    ENUM = auto()
    VOID = auto()
    NEVER = auto()
    NONE = auto()


@dataclass(frozen=True)
class Type:
    """A type. Scalars use kind alone; ARRAY/SLICE/POINTER use element_type (ARRAY also size);
    DICT uses key_type and element_type (value); STRUCT/SUM/ENUM are nominal by name.
    Frozen for structural equality and hashing.
    """
    kind: TypeKind
    element_type: Optional['Type'] = None  # ARRAY/SLICE/POINTER pointee; DICT value
    size: Optional[int] = None  # ARRAY only
    struct_name: Optional[str] = None  # STRUCT only
    sum_type_name: Optional[str] = None  # SUM only
    key_type: Optional['Type'] = None  # DICT only
    enum_name: Optional[str] = None  # ENUM only

    def __str__(self) -> str:
        if self.kind == TypeKind.ARRAY:
            return f"[{self.size}]{self.element_type}"
        if self.kind == TypeKind.SLICE:
            return f"[]{self.element_type}"
        if self.kind == TypeKind.STRUCT:  # as declared, without the module prefix of its key
            return self.struct_name.rsplit('$', 1)[-1]
        if self.kind == TypeKind.SUM:
            return self.sum_type_name.rsplit('$', 1)[-1]
        if self.kind == TypeKind.ENUM:
            return self.enum_name.rsplit('$', 1)[-1]
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

INTEGER_TYPES = {Type.INT, Type.INT8, Type.UINT8, Type.INT64, Type.INT32}
BYTE_SLICE = Type(TypeKind.SLICE, element_type=Type.UINT8)  # `[]byte`


@dataclass
class StructInfo:
    """A struct's name and ordered fields; order fixes layout."""
    name: str
    fields: Dict[str, Type]


@dataclass
class SumTypeInfo:
    """A sum type's name and ordered variants (any types but sums; NONE for `none`); index is the
    discriminant."""
    name: str
    variants: List[Type]


@dataclass
class EnumInfo:
    """An enum's name and ordered members; a value is its member's index, held like an int32."""
    name: str
    members: List[str]


# Storage is laid out as C lays out the same fields: each value at an offset its alignment divides,
# the alignment of a scalar being its width and of a str, slice, pointer, or dict its pointer's (8);
# a struct's or an array's that of what it holds; and a size rounded up to the alignment. (So a
# Hornet struct is the C struct with the same fields, and a pointer is never split across words.)

# Sum layout: a 4-byte tag, then the payload at 8, whatever the variants: so a sum's value can be
# copied whole into a sum of other variants (widened or narrowed) with its payload in place.
SUM_TYPE_TAG_WIDTH = 4
SUM_PAYLOAD_OFFSET = 8

# A dict's header (runtime/internal.h's struct hornet_dict): the buckets, count, tombstones, and
# capacity, then where a bucket's key and value are, how far apart buckets are, and whether a bucket
# may hold a pointer.
DICT_HEADER_SIZE = 64
DICT_KEY_OFFSET_FIELD, DICT_VALUE_OFFSET_FIELD, DICT_STRIDE_FIELD, DICT_HOLDS_POINTERS_FIELD = 32, 40, 48, 56


def _rounded(n: int, alignment: int) -> int:
    return -(-n // alignment) * alignment


def type_alignment(t: Type, structs: dict[str, StructInfo], sum_types: dict) -> int:
    """What the offset of `t`'s storage is a multiple of (see above)."""
    if t.kind == TypeKind.ARRAY:
        return type_alignment(t.element_type, structs, sum_types)
    if t.kind == TypeKind.STRUCT:
        return max((type_alignment(f, structs, sum_types) for f in structs[t.struct_name].fields.values()), default=1)
    if t.kind == TypeKind.SUM:
        return max([SUM_TYPE_TAG_WIDTH] + [type_alignment(v, structs, sum_types)
                                           for v in sum_types[t.sum_type_name].variants])
    if t.kind in (TypeKind.STR, TypeKind.SLICE):
        return 8
    if t.kind == TypeKind.NONE:
        return 1
    return type_byte_width(t, structs, sum_types)  # a scalar, a pointer, or a dict


def holds_pointers(t: Type, structs: dict[str, StructInfo], sum_types: dict) -> bool:
    """Whether storage of `t` may hold a pointer: a str's, a slice's, a dict's, or a pointer; or
    something that has one of those in it (a sum, in any of its variants, whichever it holds). Memory
    that holds none is allocated as such (`_atomic`), so that a collector need not look through it.
    (What a str or a slice points to is another matter: a str's bytes, a []int's elements hold none.)"""
    if t.kind in (TypeKind.STR, TypeKind.SLICE, TypeKind.DICT, TypeKind.POINTER):
        return True
    if t.kind == TypeKind.ARRAY:
        return holds_pointers(t.element_type, structs, sum_types)
    if t.kind == TypeKind.STRUCT:
        return any(holds_pointers(f, structs, sum_types) for f in structs[t.struct_name].fields.values())
    if t.kind == TypeKind.SUM:
        return any(holds_pointers(v, structs, sum_types) for v in sum_types[t.sum_type_name].variants)
    return False


def field_offsets(struct_name: str, structs: dict[str, StructInfo], sum_types: dict) -> dict[str, int]:
    """The offset of each of a struct's fields, by name, in declaration order."""
    offsets, end = {}, 0
    for name, field_type in structs[struct_name].fields.items():
        offsets[name] = end = _rounded(end, type_alignment(field_type, structs, sum_types))
        end += type_byte_width(field_type, structs, sum_types)
    return offsets


def dict_bucket_layout(key_type: Type, value_type: Type, structs: dict[str, StructInfo], sum_types: dict) -> tuple:
    """(key offset, value offset, stride) of a dict's buckets: a state byte, then the key, then the
    value, each where its alignment puts it."""
    key_alignment = type_alignment(key_type, structs, sum_types)
    value_alignment = type_alignment(value_type, structs, sum_types)
    key = _rounded(1, key_alignment)
    value = _rounded(key + type_byte_width(key_type, structs, sum_types), value_alignment)
    stride = _rounded(value + type_byte_width(value_type, structs, sum_types), max(key_alignment, value_alignment))
    return key, value, stride


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
        fields = structs[t.struct_name].fields
        if not fields:
            return 0
        last = list(fields)[-1]
        end = field_offsets(t.struct_name, structs, sum_types)[last] + type_byte_width(fields[last], structs, sum_types)
        return _rounded(end, type_alignment(t, structs, sum_types))
    if t.kind == TypeKind.SUM:
        payload = max(type_byte_width(v, structs, sum_types) for v in sum_types[t.sum_type_name].variants)
        return _rounded(SUM_PAYLOAD_OFFSET + payload, type_alignment(t, structs, sum_types))
    if t.kind == TypeKind.POINTER:
        return 8  # pointer
    if t.kind == TypeKind.NONE:
        return 0  # a sum type's `none` variant: the tag alone
    if t.kind == TypeKind.DICT:
        return 8  # a pointer to the shared header: copies alias
    return 4  # INT32, BOOL, ENUM


def is_wide_type(t: Type) -> bool:
    """Whether `t` needs 8-byte moves (int, pointer)."""
    return t == Type.INT or t.kind == TypeKind.POINTER


def leaf_type(t: Type) -> Type:
    """Innermost non-array element type."""
    while t.kind == TypeKind.ARRAY:
        t = t.element_type
    return t
