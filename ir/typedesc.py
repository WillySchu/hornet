"""Type descriptors: static data describing a type, which runtime.c's printer walks."""

from ir.errors import IRError
from typesys import Type, TypeKind, type_byte_width

# Kind tags; the source of truth for runtime/generate_typedesc_header.py.
_TYPEDESC_INT = 0
_TYPEDESC_BOOL = 1
_TYPEDESC_STR = 2
_TYPEDESC_ARRAY = 3
_TYPEDESC_SLICE = 4
_TYPEDESC_STRUCT = 5
_TYPEDESC_INT8 = 6
_TYPEDESC_UINT8 = 7
_TYPEDESC_SUM = 9
_TYPEDESC_POINTER = 10
_TYPEDESC_DICT = 11
_TYPEDESC_INT32 = 12
_TYPEDESC_NONE = 13  # a sum type's `none` variant
_TYPEDESC_ENUM = 14


def type_descriptor(ir_program, t: Type, in_progress: dict = None) -> str:
    """The label of `t`'s descriptor, built once per program (recursive types refer back by label)."""
    cache = ir_program.__dict__.setdefault('_type_descriptor_labels', {})
    if t in cache:
        return cache[t]
    in_progress = {} if in_progress is None else in_progress
    if t in in_progress:
        return in_progress[t]
    ids, structs, sums = ir_program.ids, ir_program.struct_registry, ir_program.sum_type_registry
    label = ids.new_label("typedesc")
    in_progress[t] = label

    def name() -> str:
        name_label = ids.new_label("typedesc_name")
        ir_program.string_literals.append((name_label, str(t)))
        return name_label

    def width(x: Type) -> int:
        return type_byte_width(x, structs, sums)

    scalar_tags = {Type.INT8: _TYPEDESC_INT8, Type.UINT8: _TYPEDESC_UINT8, Type.INT32: _TYPEDESC_INT32}
    if t in scalar_tags:
        words = [scalar_tags[t]]
    elif t.kind == TypeKind.INT:
        words = [_TYPEDESC_INT]
    elif t.kind == TypeKind.BOOL:
        words = [_TYPEDESC_BOOL]
    elif t.kind == TypeKind.STR:
        words = [_TYPEDESC_STR]
    elif t.kind == TypeKind.ARRAY:
        words = [_TYPEDESC_ARRAY, name(), type_descriptor(ir_program, t.element_type, in_progress), t.size,
                 width(t.element_type)]
    elif t.kind == TypeKind.SLICE:
        words = [_TYPEDESC_SLICE, name(), type_descriptor(ir_program, t.element_type, in_progress), width(t.element_type)]
    elif t.kind == TypeKind.STRUCT:
        words, offset = [_TYPEDESC_STRUCT, name(), len(structs[t.struct_name].fields)], 0
        for field_name, field_type in structs[t.struct_name].fields.items():
            field_label = ids.new_label("typedesc_fname")
            ir_program.string_literals.append((field_label, field_name))
            words += [field_label, type_descriptor(ir_program, field_type, in_progress), offset]
            offset += width(field_type)
    elif t.kind == TypeKind.SUM:  # [tag, variant count, variant descriptors...]
        variants = sums[t.sum_type_name].variants
        words = [_TYPEDESC_SUM, len(variants)] + [type_descriptor(ir_program, v, in_progress) for v in variants]
    elif t.kind == TypeKind.NONE:
        words = [_TYPEDESC_NONE]
    elif t.kind == TypeKind.ENUM:  # [tag, name, member count, member names...]
        members = ir_program.enum_registry[t.enum_name].members
        words = [_TYPEDESC_ENUM, name(), len(members)]
        for member in members:
            member_label = ids.new_label("typedesc_member")
            ir_program.string_literals.append((member_label, member))
            words.append(member_label)
    elif t.kind == TypeKind.POINTER:  # printed as an address
        words = [_TYPEDESC_POINTER]
    elif t.kind == TypeKind.DICT:  # [tag, name, key descriptor, key width, value descriptor, value width]
        words = [_TYPEDESC_DICT, name(), type_descriptor(ir_program, t.key_type, in_progress), width(t.key_type),
                 type_descriptor(ir_program, t.element_type, in_progress), width(t.element_type)]
    else:
        raise IRError(f"No type descriptor rule for: {t}")
    ir_program.type_descriptors.append((label, words))
    cache[t] = label
    return label
