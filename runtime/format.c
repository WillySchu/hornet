// print and format(...): a value as text, by its type descriptor (ir/typedesc.py).
#include "internal.h"

void hornet__stringify(void *value_addr, const unsigned char *type_desc, int quote_strings, struct hornet_buf *buf) {
    switch ((int)read_desc_word(type_desc, 0)) {
        case HORNET_TYPEDESC_INT: {
            int64_t value = read_i64(value_addr);
            char digits[32];
            int n = snprintf(digits, sizeof(digits), "%lld", (long long)value);
            hornet_buf_append_bytes(buf, digits, n);
            break;
        }
        case HORNET_TYPEDESC_INT32: {
            int32_t value = read_i32(value_addr);
            char digits[16];
            int n = snprintf(digits, sizeof(digits), "%d", value);
            hornet_buf_append_bytes(buf, digits, n);
            break;
        }
        case HORNET_TYPEDESC_INT8: {
            int8_t value = *(int8_t *)value_addr;
            char digits[16];
            int n = snprintf(digits, sizeof(digits), "%d", (int32_t)value);
            hornet_buf_append_bytes(buf, digits, n);
            break;
        }
        case HORNET_TYPEDESC_UINT8: {
            uint8_t value = *(uint8_t *)value_addr;
            char digits[16];
            int n = snprintf(digits, sizeof(digits), "%d", (int32_t)value);
            hornet_buf_append_bytes(buf, digits, n);
            break;
        }
        case HORNET_TYPEDESC_BOOL: {
            int32_t value = read_i32(value_addr);
            if (value != 0) {
                hornet_buf_append_cstr(buf, "true");
            } else {
                hornet_buf_append_cstr(buf, "false");
            }
            break;
        }
        case HORNET_TYPEDESC_STR: {
            // {ptr, len} descriptor.
            const char *s = (const char *)read_ptr(value_addr);
            int64_t len = read_i64((char *)value_addr + 8);
            if (quote_strings) {
                hornet_buf_append_byte(buf, '\'');
                hornet_buf_append_bytes(buf, s, len);
                hornet_buf_append_byte(buf, '\'');
            } else {
                hornet_buf_append_bytes(buf, s, len);
            }
            break;
        }
        case HORNET_TYPEDESC_ARRAY: {
            // [tag, name, elem_desc, count, elem_width]; data inline.
            const char *name = (const char *)read_desc_word(type_desc, 1);
            const unsigned char *elem_desc = (const unsigned char *)read_desc_word(type_desc, 2);
            int64_t count = (int64_t)read_desc_word(type_desc, 3);
            int64_t elem_width = (int64_t)read_desc_word(type_desc, 4);

            hornet_buf_append_cstr(buf, name);
            hornet_buf_append_byte(buf, '[');
            for (int64_t i = 0; i < count; i++) {
                if (i != 0) {
                    hornet_buf_append_bytes(buf, ", ", 2);
                }
                void *elem_addr = (char *)value_addr + i * elem_width;
                hornet__stringify(elem_addr, elem_desc, 1, buf);
            }
            hornet_buf_append_byte(buf, ']');
            break;
        }
        case HORNET_TYPEDESC_SLICE: {
            // [tag, name, elem_desc, elem_width]
            const char *name = (const char *)read_desc_word(type_desc, 1);
            const unsigned char *elem_desc = (const unsigned char *)read_desc_word(type_desc, 2);
            int64_t elem_width = (int64_t)read_desc_word(type_desc, 3);

            // {ptr, len, cap}, 8 bytes each
            void *base_ptr = read_ptr(value_addr);
            int64_t length = read_i64((char *)value_addr + 8);
            hornet_buf_append_cstr(buf, name);
            hornet_buf_append_byte(buf, '[');
            for (int64_t i = 0; i < length; i++) {
                if (i != 0) {
                    hornet_buf_append_bytes(buf, ", ", 2);
                }
                void *elem_addr = (char *)base_ptr + (int64_t)i * elem_width;
                hornet__stringify(elem_addr, elem_desc, 1, buf);
            }
            hornet_buf_append_byte(buf, ']');
            break;
        }
        case HORNET_TYPEDESC_DICT: {
            // [tag, name, key_desc, key_width, value_desc, value_width]; the bucket layout is the header's.
            const char *name = (const char *)read_desc_word(type_desc, 1);
            const unsigned char *key_desc = (const unsigned char *)read_desc_word(type_desc, 2);
            const unsigned char *value_desc = (const unsigned char *)read_desc_word(type_desc, 4);
            const struct hornet_dict *dict = read_ptr(value_addr);  // a dict value points to its shared header

            hornet_buf_append_cstr(buf, name);
            hornet_buf_append_byte(buf, '{');
            int printed_any = 0;
            for (int64_t i = 0; i < dict->capacity; i++) {
                unsigned char *bucket = (unsigned char *)dict->buckets + i * dict->stride;
                if (bucket[0] != HORNET_DICT_BUCKET_OCCUPIED) {
                    continue;
                }
                if (printed_any) {
                    hornet_buf_append_bytes(buf, ", ", 2);
                }
                printed_any = 1;
                hornet__stringify(bucket + dict->key_offset, key_desc, 1, buf);
                hornet_buf_append_bytes(buf, ": ", 2);
                hornet__stringify(bucket + dict->value_offset, value_desc, 1, buf);
            }
            hornet_buf_append_byte(buf, '}');
            break;
        }
        case HORNET_TYPEDESC_STRUCT: {
            // [tag, name, field_count, (field_name, field_desc, offset)...]
            const char *name = (const char *)read_desc_word(type_desc, 1);
            int64_t field_count = (int64_t)read_desc_word(type_desc, 2);
            hornet_buf_append_cstr(buf, name);
            hornet_buf_append_byte(buf, '(');
            for (int64_t i = 0; i < field_count; i++) {
                if (i != 0) {
                    hornet_buf_append_bytes(buf, ", ", 2);
                }
                int64_t entry_index = 3 + i * 3;
                const char *field_name = (const char *)read_desc_word(type_desc, entry_index);
                const unsigned char *field_type_desc =
                    (const unsigned char *)read_desc_word(type_desc, entry_index + 1);
                int64_t field_offset = (int64_t)read_desc_word(type_desc, entry_index + 2);

                hornet_buf_append_cstr(buf, field_name);
                hornet_buf_append_bytes(buf, ": ", 2);
                void *field_addr = (char *)value_addr + field_offset;
                hornet__stringify(field_addr, field_type_desc, 1, buf);
            }
            hornet_buf_append_byte(buf, ')');
            break;
        }
        case HORNET_TYPEDESC_SUM: {
            // [tag, variant_count, variant_desc...]; prints the active variant.
            int32_t discriminant = read_i32(value_addr);
            const unsigned char *variant_desc =
                (const unsigned char *)read_desc_word(type_desc, 2 + discriminant);
            void *payload_addr = (char *)value_addr + HORNET_SUM_PAYLOAD_OFFSET;
            hornet__stringify(payload_addr, variant_desc, 1, buf);
            break;
        }
        case HORNET_TYPEDESC_NONE:
            hornet_buf_append_cstr(buf, "none");  // a sum type's `none` variant
            break;
        case HORNET_TYPEDESC_ENUM: {
            // [tag, name, member_count, member_name...]; the value is its member's index. Prints
            // `Enum.Member`, as it is written.
            int32_t index = read_i32(value_addr);
            hornet_buf_append_cstr(buf, (const char *)read_desc_word(type_desc, 1));
            hornet_buf_append_byte(buf, '.');
            if (index >= 0 && (uint64_t)index < read_desc_word(type_desc, 2)) {
                hornet_buf_append_cstr(buf, (const char *)read_desc_word(type_desc, 3 + index));
            } else {  // not a member: only memory the program didn't write holds one
                char digits[16];
                int n = snprintf(digits, sizeof(digits), "%d", index);
                hornet_buf_append_bytes(buf, digits, n);
            }
            break;
        }
        case HORNET_TYPEDESC_POINTER: {
            // Prints the address, not the pointee; a pointer to nothing is `none`, as it is written.
            void *value = read_ptr(value_addr);
            if (value == NULL) {
                hornet_buf_append_cstr(buf, "none");
                break;
            }
            char digits[20];
            int n = snprintf(digits, sizeof(digits), "0x%llx", (unsigned long long)(uintptr_t)value);
            hornet_buf_append_bytes(buf, digits, n);
            break;
        }
        default:
            // unreachable
            break;
    }
}

// format(...): the compiled code keeps a hornet_buf in its frame, begins it, and has each piece
// appended. Its first two words are then the str: the bytes are never freed, being the result.
_Static_assert(offsetof(struct hornet_buf, ptr) == 0 && offsetof(struct hornet_buf, len) == 8,
               "a hornet_buf starts with a str's pointer and length");

void hornet_format_begin(struct hornet_buf *buf) {
    buf->cap = 32;
    buf->ptr = hornet_alloc(buf->cap);
    buf->len = 0;
}

void hornet_format_text(struct hornet_buf *buf, const char *ptr, int64_t len) {
    hornet_buf_append_bytes(buf, ptr, len);
}

// A value as print shows it.
void hornet_format_value(struct hornet_buf *buf, void *value_addr, const unsigned char *type_desc) {
    hornet__stringify(value_addr, type_desc, 0, buf);
}

void hornet_print(void *value_addr, const unsigned char *type_desc) {
    struct hornet_buf buf;
    buf.cap = 16;
    buf.ptr = hornet_alloc(buf.cap);
    buf.len = 0;
    hornet__stringify(value_addr, type_desc, 0, &buf);
    hornet_buf_append_byte(&buf, '\n');
    hornet_write_fd(1, buf.ptr, buf.len);
    free(buf.ptr);
}
