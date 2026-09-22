// hornet_stringify/hornet_print -- the Hornet print() runtime.
//
// Direct C port of build_stringify_function/gen_print_call_into in
// codegen/strings.py. Faithful to that implementation's own logic
// (the kind-dispatch structure, the buffer growth POLICY, the
// distinction between an ARRAY's inline-data value and a SLICE's
// {ptr, len, cap} runtime descriptor, the recursive struct-field
// walk) -- but uses realloc/memcpy/snprintf in place of the hand-
// rolled malloc+byte-copy-loop+free and digit-extraction loops that
// existed only because hand-written x86 assembly had no better tool
// available. Nothing here works around a limitation C doesn't have.
//
// Every incoming value in this file is read through a type descriptor
// built by _get_or_build_type_descriptor (codegen/strings.py) -- a
// flat array of 8-byte words, positionally read, whose exact per-kind
// shape is documented at each dispatch case below. That descriptor
// layout, and the {ptr, len, cap} runtime shape of a SLICE value
// itself, are the ABI boundary between this file and the Python
// compiler that emits the data this file reads -- see this repo's own
// notes on that boundary for why careful hand-matching, not
// mechanical generation, is how the two sides stay in sync for now.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "hornet_typedesc_tags.h"

// The print machinery's own growable byte buffer -- distinct from,
// and simpler than, a Hornet-level slice's own {ptr, len, cap}
// descriptor (which is {ptr: 8 bytes, len: 4 bytes, cap: 4 bytes} --
// see hornet_stringify's own SLICE case below): this one is purely
// internal to this file, never exposed to compiled Hornet code, and
// uses a uniform 8-byte width for every field.
struct hornet_buf {
    char *ptr;
    int64_t len;
    int64_t cap;
};

static void hornet_stringify(
    void *value_addr, const unsigned char *type_desc, int quote_strings, struct hornet_buf *buf);

// Ensures at least `additional` more bytes of spare capacity, growing
// via realloc if needed. GROWTH POLICY, preserved exactly from
// gen_buffer_append_bytes_into's own documented formula: needed =
// len + additional; if needed <= cap, no reallocation; otherwise
// grow to new_cap = max(needed, cap*2 if cap < 256 else cap +
// cap/4) -- the full formula, not append()'s own single-element
// simplification, since `additional` can be arbitrarily larger than
// one doubling would produce.
static void hornet_buf_ensure(struct hornet_buf *buf, int64_t additional) {
    int64_t needed = buf->len + additional;
    if (needed <= buf->cap) {
        return;
    }
    int64_t doubled_or_quartered = buf->cap < 256 ? buf->cap * 2 : buf->cap + buf->cap / 4;
    int64_t new_cap = needed > doubled_or_quartered ? needed : doubled_or_quartered;
    buf->ptr = realloc(buf->ptr, (size_t)new_cap);
    buf->cap = new_cap;
}

static void hornet_buf_append_bytes(struct hornet_buf *buf, const char *data, int64_t count) {
    hornet_buf_ensure(buf, count);
    memcpy(buf->ptr + buf->len, data, (size_t)count);
    buf->len += count;
}

static void hornet_buf_append_byte(struct hornet_buf *buf, char byte_value) {
    hornet_buf_append_bytes(buf, &byte_value, 1);
}

static void hornet_buf_append_cstr(struct hornet_buf *buf, const char *s) {
    hornet_buf_append_bytes(buf, s, (int64_t)strlen(s));
}

// Hornet's own struct layout is tightly packed, with no alignment
// padding inserted between fields at all (see _field_offset's own
// docstring in codegen/structs.py: "x86-64 doesn't require aligned
// access") -- so a multi-byte value (an int, a pointer, an int64) can
// genuinely land at a misaligned address once nested inside a struct
// or array. The SAME risk applies to a type descriptor's own words:
// emitter.py emits every string literal (variable-length .asciz)
// before any type descriptor in the same .data block, with no
// .align directive anywhere in between -- so a type descriptor's own
// first word is not guaranteed to start 8-byte-aligned either.
//
// x86 hardware tolerates an unaligned `mov` just fine, which is why
// the old hand-written assembly this file replaces never needed to
// think about any of this -- but an ordinary C pointer dereference
// of a misaligned pointer (`*(int32_t *)p`, or plain array indexing
// on a uint64_t*, which is exactly the same thing) is undefined
// behavior regardless of what the hardware happens to allow, exactly
// what UBSan's own "misaligned address" report caught directly, on a
// self-referential struct test, before this fix. memcpy has no such
// alignment requirement on either side, so every multi-byte read --
// of a Hornet value, or of a type descriptor's own word -- goes
// through one of these helpers instead of a direct dereference or
// array index. A single-byte read (int8/uint8) has no alignment
// requirement at all and keeps using an ordinary dereference.
//
// This is also why a type descriptor is passed around as `const
// unsigned char *` throughout this file, not `const uint64_t *`:
// the latter would make every ordinary array-index expression
// (desc[1], desc[2], ...) silently reintroduce the exact same
// misaligned-access risk this fix closes.
static int32_t read_i32(const void *addr) {
    int32_t value;
    memcpy(&value, addr, sizeof(value));
    return value;
}

static int64_t read_i64(const void *addr) {
    int64_t value;
    memcpy(&value, addr, sizeof(value));
    return value;
}

static void *read_ptr(const void *addr) {
    void *value;
    memcpy(&value, addr, sizeof(value));
    return value;
}

// Reads the `index`-th 8-byte word out of a type descriptor (byte
// offset index*8), matching how _get_or_build_type_descriptor's own
// per-kind field layout is documented at each dispatch case below.
static uint64_t read_desc_word(const unsigned char *desc, int64_t index) {
    uint64_t value;
    memcpy(&value, desc + index * 8, sizeof(value));
    return value;
}

// The recursive core. type_desc[0] is always the kind tag; every
// other field's own position depends on the kind, documented at each
// case below (see _get_or_build_type_descriptor's own docstring in
// codegen/strings.py for the authoritative layout this mirrors).
//
// quote_strings is 0 only for the outermost value of a print() call
// (hornet_print's own, single top-level call below); every recursive
// call this function makes to itself -- into an array/slice element
// or a struct field -- passes 1, so a str value stays unambiguous
// next to its neighbors.
static void hornet_stringify(
    void *value_addr, const unsigned char *type_desc, int quote_strings, struct hornet_buf *buf) {
    switch ((int)read_desc_word(type_desc, 0)) {
        case HORNET_TYPEDESC_INT: {
            int32_t value = read_i32(value_addr);
            char digits[16];
            int n = snprintf(digits, sizeof(digits), "%d", value);
            hornet_buf_append_bytes(buf, digits, n);
            break;
        }
        case HORNET_TYPEDESC_INT8: {
            // value_addr points at a genuinely 1-byte-wide value --
            // a signed narrow read, then formatted as an ordinary
            // int, matching gen_int_to_decimal_into's own contract
            // (called there after a widening MovSX).
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
        case HORNET_TYPEDESC_INT64: {
            int64_t value = read_i64(value_addr);
            char digits[32];
            int n = snprintf(digits, sizeof(digits), "%lld", (long long)value);
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
            // The value itself IS a pointer -- value_addr holds the
            // address of a pointer, not the string's own bytes
            // directly (unlike every other kind here).
            const char *s = (const char *)read_ptr(value_addr);
            int64_t len = (int64_t)strlen(s);
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
            // Descriptor shape: [tag, name_ptr, elem_type_desc_ptr,
            // count, elem_width]. value_addr points directly at the
            // array's own inline data -- no indirection to unwrap,
            // since an array's bytes ARE its storage.
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
                hornet_stringify(elem_addr, elem_desc, 1, buf);
            }
            hornet_buf_append_byte(buf, ']');
            break;
        }
        case HORNET_TYPEDESC_SLICE: {
            // Descriptor shape: [tag, name_ptr, elem_type_desc_ptr,
            // elem_width] -- no count field here, unlike ARRAY: a
            // slice's length is a RUNTIME property of the value, not
            // its static type, so it's read from value_addr's own
            // {ptr, len, cap} descriptor below instead.
            const char *name = (const char *)read_desc_word(type_desc, 1);
            const unsigned char *elem_desc = (const unsigned char *)read_desc_word(type_desc, 2);
            int64_t elem_width = (int64_t)read_desc_word(type_desc, 3);

            // The VALUE's own runtime descriptor: {ptr: 8 bytes,
            // len: 4 bytes, cap: 4 bytes} -- cap is never read here,
            // only ptr and len.
            void *base_ptr = read_ptr(value_addr);
            int32_t length = read_i32((char *)value_addr + 8);
            hornet_buf_append_cstr(buf, name);
            hornet_buf_append_byte(buf, '[');
            for (int32_t i = 0; i < length; i++) {
                if (i != 0) {
                    hornet_buf_append_bytes(buf, ", ", 2);
                }
                void *elem_addr = (char *)base_ptr + (int64_t)i * elem_width;
                hornet_stringify(elem_addr, elem_desc, 1, buf);
            }
            hornet_buf_append_byte(buf, ']');
            break;
        }
        case HORNET_TYPEDESC_STRUCT: {
            // Descriptor shape: [tag, name_ptr, field_count, then
            // field_count fixed-size triples of (field_name_ptr,
            // field_type_desc_ptr, field_byte_offset)] -- field i's
            // triple starts at word index 3 + i*3. Format is
            // `Name(field: value, field: value)` -- parentheses, not
            // ARRAY/SLICE's square brackets, matching struct-literal
            // syntax.
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
                hornet_stringify(field_addr, field_type_desc, 1, buf);
            }
            hornet_buf_append_byte(buf, ')');
            break;
        }
        case HORNET_TYPEDESC_SUM: {
            // Descriptor shape: [tag, variant_count, variant_desc_ptr,
            // variant_desc_ptr, ...] -- no name field, and no wrapper
            // syntax of its own (no brackets, no parens added here):
            // a sum-typed value prints EXACTLY as its active variant
            // would on its own (`Circle(radius: 5)`, never `Shape
            // (Circle(radius: 5))`) -- see _get_or_build_type_
            // descriptor's own SUM case in ir/strings.py for why.
            //
            // The discriminant that picks WHICH variant lives in the
            // VALUE, not the descriptor: a 4-byte int at value_addr's
            // own start (SUM_TYPE_TAG_WIDTH in ir/utils.py --
            // hardcoded here as a plain 4, not generated, matching
            // every other structural layout detail in this file),
            // indexing directly into the variant-pointer array right
            // after variant_count. Always a value this compiler
            // itself wrote (see SumTypeDef's own docstring in
            // parser.py -- there's no user-facing way to construct an
            // out-of-range one), so no bounds check here, matching how
            // a struct's own field_offset or an array's own elem_
            // width is trusted unconditionally too.
            //
            // The recursive call passes 1, matching every other
            // composite case's own children here (ARRAY/SLICE
            // elements, STRUCT fields) -- not, as might seem more
            // "correct" for a value that's meant to print completely
            // transparently, whatever quote_strings this SUM case
            // itself received. That distinction turns out to be
            // unobservable either way: a variant is always a struct
            // (semantic.py's own restriction -- see SumTypeDef's own
            // docstring), and HORNET_TYPEDESC_STRUCT's own case, just
            // above, never reads its OWN incoming quote_strings
            // parameter at all -- only a bare leaf STR value, or a
            // collection recursing toward one, ever consults it. If a
            // variant could ever be something other than a struct,
            // this would need revisiting.
            int32_t discriminant = read_i32(value_addr);
            const unsigned char *variant_desc =
                (const unsigned char *)read_desc_word(type_desc, 2 + discriminant);
            void *payload_addr = (char *)value_addr + 4;
            hornet_stringify(payload_addr, variant_desc, 1, buf);
            break;
        }
        case HORNET_TYPEDESC_POINTER: {
            // Prints the raw address itself, Go-style (0xc0000...),
            // never the pointee's own value -- see _get_or_build_type_
            // descriptor's own POINTER case in ir/strings.py for why
            // there's no recursion here at all, unlike every other
            // composite case above: quote_strings is irrelevant, and
            // the descriptor carries no pointee-type field to recurse
            // through even if it wanted to. value_addr holds the
            // address of the pointer VALUE (an 8-byte address itself),
            // exactly the same "value_addr points at a pointer, not
            // the pointee's own bytes" shape HORNET_TYPEDESC_STR's own
            // case above already has -- read_ptr, not a dereference.
            void *value = read_ptr(value_addr);
            char digits[20];
            int n = snprintf(digits, sizeof(digits), "0x%llx", (unsigned long long)(uintptr_t)value);
            hornet_buf_append_bytes(buf, digits, n);
            break;
        }
        default:
            // Unreachable for any type this compiler ever hands
            // here -- matches build_stringify_function's own
            // identical "Jmp(done_label)" fallthrough for an
            // unrecognized tag.
            break;
    }
}

// The single entry point the compiler's own call-site codegen calls
// for `print(x)`: allocates the initial buffer, stringifies the
// outermost value (quote_strings=0 -- a bare str prints unquoted),
// appends a trailing newline, writes the result to stdout, frees the
// buffer.
void hornet_print(void *value_addr, const unsigned char *type_desc) {
    struct hornet_buf buf;
    buf.cap = 16;
    buf.ptr = malloc((size_t)buf.cap);
    buf.len = 0;
    hornet_stringify(value_addr, type_desc, 0, &buf);
    hornet_buf_append_byte(&buf, '\n');
    write(1, buf.ptr, (size_t)buf.len);
    free(buf.ptr);
}

// The single entry point the compiler's own bounds-check codegen
// calls on failure: prints `msg`, then aborts (SIGABRT) rather than a
// plain exit() -- an out-of-bounds access is a genuine program bug,
// not a normal termination condition, the same "abnormal termination"
// character division by zero's hardware-trapped SIGFPE already has.
// Never returns.
//
// Explicitly calls fflush(NULL) between puts() and abort() -- found
// necessary by testing: abort() terminates via a raw signal, bypassing
// the normal exit() path that would otherwise flush libc's own
// buffered stdio. Without this, the message prints reliably when
// stdout is line-buffered (an interactive terminal) but is silently
// LOST whenever stdout is redirected or piped -- the case for most
// non-interactively run programs.
void hornet_panic(const char *msg) {
    puts(msg);
    fflush(NULL);
    abort();
}

// The single entry point the compiler's own append-growth codegen
// calls: the growth-ONLY half of append(s, value) -- mallocs a fresh
// backing array of new_cap*element_width bytes and copies the
// existing len*element_width bytes over from old_ptr. Writing the
// newly-appended value itself happens as a separate step immediately
// after this call returns, never this function's own concern (see
// IRSliceGrow's own docstring in ir/ir.py); new_cap is likewise
// computed by the CALLER, not here -- capacity-doubling is pure
// policy, with no allocation of its own, so it stays in codegen
// rather than moving in here alongside the actual allocation.
//
// A plain memcpy in place of the hand-rolled movq/movl/movb chunking
// loop this used to be as inline assembly -- correct for ANY element
// type (int/bool/str/array/struct/slice alike), since copying an
// ALREADY-existing, already-valid element is always just "copy
// element_width bytes," with no type-specific construction logic
// needed. Nothing here works around a limitation C doesn't have.
void *hornet_slice_grow(const void *old_ptr, int32_t len, int32_t new_cap, int32_t element_width) {
    void *new_ptr = malloc((size_t)new_cap * (size_t)element_width);
    memcpy(new_ptr, old_ptr, (size_t)len * (size_t)element_width);
    return new_ptr;
}
