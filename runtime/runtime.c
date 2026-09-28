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

// Bucket state byte -- each dict bucket's own first byte. TOMBSTONE
// (a deleted entry) is a real, separate state from EMPTY: linear-
// probed deletion can't just reset a bucket to EMPTY, since that
// would silently break lookup for any OTHER key that happened to
// probe PAST this same bucket during its own insertion (lookup stops
// probing at the first EMPTY slot it finds -- an incorrectly-reset
// bucket would look like "the probe chain ends here", hiding
// whatever key was placed one step further along). Insert instead
// reuses the FIRST tombstone it passes over if the key turns out to
// be new (see hornet_dict_insert_scalar_key's own probing loop);
// lookup, delete, and hornet_stringify's own DICT case all skip PAST
// a tombstone, never stopping there or printing it. Defined here,
// ahead of hornet_stringify below, since a #define (unlike a
// function) has no forward declaration -- it has to already be in
// effect at every one of its own use sites in this file, not just
// the dict-specific functions further down that this shape is really
// about.
#define HORNET_DICT_BUCKET_EMPTY 0
#define HORNET_DICT_BUCKET_OCCUPIED 1
#define HORNET_DICT_BUCKET_TOMBSTONE 2

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
            // A str value is a 16-byte {ptr, len} descriptor now (see
            // ir/strings.py's own module docstring) -- value_addr
            // holds the address of that descriptor, not the string's
            // own bytes directly (unlike every other kind here). ptr
            // at offset 0, len (a plain int32_t, same convention a
            // SLICE's own len/cap already use just below) at offset
            // 8 -- read directly, never via strlen: there is no null
            // terminator anywhere in this scheme to scan for, and an
            // embedded '\0' byte is ordinary content, not a stopping
            // point.
            const char *s = (const char *)read_ptr(value_addr);
            int64_t len = (int64_t)read_i32((char *)value_addr + 8);
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
        case HORNET_TYPEDESC_DICT: {
            // Descriptor shape: [tag, name_ptr, key_type_desc_ptr,
            // key_width, value_type_desc_ptr, value_width].
            // bucket_stride is recomputed here from the two widths,
            // exactly the same formula ir/dicts.py's own construction
            // code already used to build this dict in the first
            // place -- the two sides have to keep agreeing on it, the
            // same ABI-boundary care this whole file's own module
            // docstring already calls out.
            //
            // value_addr points at the dict's own {buckets_ptr, count,
            // capacity} descriptor -- capacity (not count) decides how
            // many bucket SLOTS to walk over, since occupied ones can
            // be scattered anywhere among them; count only decides how
            // many commas to print (skip an empty bucket's own state
            // byte entirely, never recursing into its unwritten key/
            // value bytes).
            const char *name = (const char *)read_desc_word(type_desc, 1);
            const unsigned char *key_desc = (const unsigned char *)read_desc_word(type_desc, 2);
            int64_t key_width = (int64_t)read_desc_word(type_desc, 3);
            const unsigned char *value_desc = (const unsigned char *)read_desc_word(type_desc, 4);
            int64_t value_width = (int64_t)read_desc_word(type_desc, 5);
            int64_t bucket_stride = 1 + key_width + value_width;

            void *buckets = read_ptr(value_addr);
            int64_t capacity = read_i64((char *)value_addr + 16);

            hornet_buf_append_cstr(buf, name);
            hornet_buf_append_byte(buf, '{');
            int printed_any = 0;
            for (int64_t i = 0; i < capacity; i++) {
                unsigned char *bucket = (unsigned char *)buckets + i * bucket_stride;
                if (bucket[0] != HORNET_DICT_BUCKET_OCCUPIED) {
                    continue;
                }
                if (printed_any) {
                    hornet_buf_append_bytes(buf, ", ", 2);
                }
                printed_any = 1;
                hornet_stringify(bucket + 1, key_desc, 1, buf);
                hornet_buf_append_bytes(buf, ": ", 2);
                hornet_stringify(bucket + 1 + key_width, value_desc, 1, buf);
            }
            hornet_buf_append_byte(buf, '}');
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

// FNV-1a 64-bit over an arbitrary byte range. Used to hash a dict's
// own key, whatever kind it is: for a scalar key, ptr/len is a
// scratch stack slot holding the key's own value (materialized there
// specifically so it HAS an address to hash from at all -- an
// ordinary scalar otherwise just lives in a register); for a str key,
// ptr/len is the string's own CONTENT bytes directly (never its own
// {ptr, len} descriptor's address -- two different string views with
// identical content, but different underlying buffers, must hash
// identically, the same correctness requirement _ir_string_compare
// already has for equality). This one function suffices for every
// supported key kind -- there's no per-kind hash logic anywhere else,
// only per-kind ADDRESSING of what to hash (decided in Python, at IR-
// build time, by ir/dicts.py).
//
// Purely an internal, compiler-managed implementation detail -- never
// exposed to Hornet code, so there's no correctness requirement to
// match hash.ht's own (independent, Hornet-level) FNV-1a output.
int64_t hornet_hash_bytes(const void *ptr, int64_t len) {
    uint64_t h = 0xcbf29ce484222325ULL;  // FNV-1a 64-bit offset basis
    const unsigned char *bytes = (const unsigned char *)ptr;
    for (int64_t i = 0; i < len; i++) {
        h ^= bytes[i];
        h *= 0x100000001b3ULL;  // FNV prime
    }
    return (int64_t)h;
}

// Inserts one entry into an open-addressed (linear probing) dict's
// own bucket array, for a SCALAR-keyed dict (int/int8/uint8/int64/
// bool) -- str keys go through hornet_dict_insert_str_key instead,
// since they need content-based hashing/comparison, not a flat byte
// compare. Each bucket is 1 (state byte) + key_width + value_width
// bytes, contiguous, no padding -- bucket_stride is the caller's own
// already-computed total. capacity is ALWAYS a power of two, letting
// probing use a cheap bitwise AND instead of a modulo.
//
// key_ptr/value_ptr point at the actual bytes to store -- key_ptr is
// typically a scratch stack slot's own address (see hornet_hash_
// bytes's own docstring for why a scalar key needs one at all).
//
// The FIRST tombstone passed over during probing is remembered: if
// the key turns out to be genuinely new (an EMPTY slot reached with
// no match), that tombstone is reused instead, whenever one was seen
// -- this is what reclaims a deleted slot's own space.
//
// Returns 0 if an existing entry with the identical key was found and
// overwritten (count unchanged, ordinary last-write-wins, not an
// error). Otherwise a genuinely NEW entry was inserted: 1 into a
// fresh EMPTY slot (count++, tombstones unchanged), or 2 into a
// reused tombstone (count++ AND tombstones--).
//
// No growth check here at all: by the time this runs, the caller
// (ir/dicts.py, or hornet_dict_set_scalar_key's own growth check) has
// already guaranteed capacity leaves enough headroom that an empty
// slot always exists somewhere along the probe sequence -- unbounded
// linear probing is therefore safe, not an infinite-loop risk.
int hornet_dict_insert_scalar_key(
    void *buckets, int64_t capacity, int64_t bucket_stride,
    const void *key_ptr, int64_t key_width,
    const void *value_ptr, int64_t value_width
) {
    int64_t index = hornet_hash_bytes(key_ptr, key_width) & (capacity - 1);
    int64_t tombstone_index = -1;
    while (1) {
        unsigned char *bucket = (unsigned char *)buckets + index * bucket_stride;
        if (bucket[0] == HORNET_DICT_BUCKET_EMPTY) {
            int64_t target = (tombstone_index >= 0) ? tombstone_index : index;
            unsigned char *target_bucket = (unsigned char *)buckets + target * bucket_stride;
            target_bucket[0] = HORNET_DICT_BUCKET_OCCUPIED;
            memcpy(target_bucket + 1, key_ptr, (size_t)key_width);
            memcpy(target_bucket + 1 + key_width, value_ptr, (size_t)value_width);
            return (tombstone_index >= 0) ? 2 : 1;
        }
        if (bucket[0] == HORNET_DICT_BUCKET_TOMBSTONE) {
            if (tombstone_index < 0) {
                tombstone_index = index;
            }
            index = (index + 1) & (capacity - 1);
            continue;
        }
        if (memcmp(bucket + 1, key_ptr, (size_t)key_width) == 0) {
            memcpy(bucket + 1 + key_width, value_ptr, (size_t)value_width);
            return 0;
        }
        index = (index + 1) & (capacity - 1);
    }
}

// hornet_dict_insert_scalar_key's own str-keyed counterpart: key_ptr/
// key_len are the incoming string's own CONTENT bytes (see hornet_
// hash_bytes's own docstring), stored in each bucket's own key region
// as a fresh {ptr, len} pair -- 8 bytes for ptr, then a 32-bit (not
// 64-bit) len, matching str's own existing runtime descriptor layout
// exactly (_ir_write_str_descriptor_into_address's own len field is
// Type.INT, 4 bytes, at offset 8 -- see its own docstring), padded to
// 16 bytes total to match type_byte_width(str) exactly, the width
// bucket_stride already assumes for this key region. key_len itself
// is accepted as a wider int64_t purely because that's this whole
// file's own general convention for a byte count -- SysV's own
// calling convention already zero-extends a 32-bit argument into its
// full 64-bit register on the caller's own side, so no truncation
// risk exists passing one of Hornet's own Type.INT length values in
// here. Length-first comparison, exactly like _ir_string_compare's
// own real-IR version: two different lengths can never be equal, so
// memcmp only ever runs once they're already known equal, never
// risking a read past either buffer's own true size.
//
// Return value and tombstone-reuse: identical tri-state contract as
// hornet_dict_insert_scalar_key's own -- see its own docstring.
int hornet_dict_insert_str_key(
    void *buckets, int64_t capacity, int64_t bucket_stride,
    const void *key_ptr, int64_t key_len,
    const void *value_ptr, int64_t value_width
) {
    // 16 == type_byte_width(str) on the Python side (an 8-byte ptr
    // plus a 4-byte len, padded to 16) -- the fixed width of this
    // bucket's own key region whenever key_type is str, regardless of
    // the actual string's own length.
    const int64_t key_region_width = 16;
    int64_t index = hornet_hash_bytes(key_ptr, key_len) & (capacity - 1);
    int64_t tombstone_index = -1;
    while (1) {
        unsigned char *bucket = (unsigned char *)buckets + index * bucket_stride;
        if (bucket[0] == HORNET_DICT_BUCKET_EMPTY) {
            int64_t target = (tombstone_index >= 0) ? tombstone_index : index;
            unsigned char *target_bucket = (unsigned char *)buckets + target * bucket_stride;
            target_bucket[0] = HORNET_DICT_BUCKET_OCCUPIED;
            *(void **)(target_bucket + 1) = (void *)key_ptr;
            *(int32_t *)(target_bucket + 1 + sizeof(void *)) = (int32_t)key_len;
            memcpy(target_bucket + 1 + key_region_width, value_ptr, (size_t)value_width);
            return (tombstone_index >= 0) ? 2 : 1;
        }
        if (bucket[0] == HORNET_DICT_BUCKET_TOMBSTONE) {
            if (tombstone_index < 0) {
                tombstone_index = index;
            }
            index = (index + 1) & (capacity - 1);
            continue;
        }
        void *stored_ptr = *(void **)(bucket + 1);
        int32_t stored_len = *(int32_t *)(bucket + 1 + sizeof(void *));
        if (stored_len == key_len && memcmp(stored_ptr, key_ptr, (size_t)key_len) == 0) {
            memcpy(bucket + 1 + key_region_width, value_ptr, (size_t)value_width);
            return 0;
        }
        index = (index + 1) & (capacity - 1);
    }
}

// hornet_dict_set_scalar_key/hornet_dict_set_str_key's own shared
// descriptor field accessors -- offsets match ir/dicts.py's own
// _ir_write_dict_literal_into exactly: buckets_ptr at 0, count (a
// plain 32-bit int, not int64 -- padded to the same 8-byte slot) at
// 8, capacity at 16. descriptor is always the dict's own 24-byte
// value's address (from _ir_dict_address on the Python side), never
// a bare buckets pointer -- these functions need to update count and
// (on growth) buckets/capacity too, which a bare buckets pointer
// alone has nowhere to write back through.
static void *dict_buckets(void *descriptor) { return read_ptr(descriptor); }
static int64_t dict_capacity(void *descriptor) { return read_i64((char *)descriptor + 16); }

static void dict_set_buckets(void *descriptor, void *buckets) {
    *(void **)descriptor = buckets;
}
static void dict_set_capacity(void *descriptor, int64_t capacity) {
    *(int64_t *)((char *)descriptor + 16) = capacity;
}
static void dict_bump_count(void *descriptor) {
    int32_t *count = (int32_t *)((char *)descriptor + 8);
    *count += 1;
}

// tombstones lives in what was, before delete existed, 4 bytes of
// pure padding between count (4 bytes, offset 8) and capacity (8
// bytes, offset 16) -- kept there deliberately so the descriptor
// stays exactly 24 bytes, with buckets/count/capacity's own existing
// offsets (0/8/16) completely unchanged; ir/dicts.py's own
// construction code already writes a 0 here explicitly now (see its
// own comment), since this address is otherwise just whatever
// garbage was already on the stack.
//
// count itself keeps meaning exactly what it always has -- LIVE
// entries only, unaffected by tombstones -- since that's what a
// future len() (not yet built) will want to read directly, with no
// adjustment needed. tombstones is tracked separately purely so the
// GROWTH check (see dict_grow_scalar_key_if_needed's own docstring)
// can count it alongside count without needing a full bucket-array
// scan on every single insert.
static int32_t dict_tombstones(void *descriptor) { return read_i32((char *)descriptor + 12); }
static void dict_set_tombstones(void *descriptor, int32_t tombstones) {
    *(int32_t *)((char *)descriptor + 12) = tombstones;
}
static void dict_bump_tombstones(void *descriptor, int32_t delta) {
    dict_set_tombstones(descriptor, dict_tombstones(descriptor) + delta);
}

// Shared growth policy: doubles capacity once count+tombstones would
// exceed 75% of it, rehashing every LIVE (OCCUPIED) bucket -- never a
// tombstone, and never an already-EMPTY one -- into the fresh, zeroed
// array via the identical hornet_dict_insert_*_key this file's own
// literal-construction path already uses (a rehashing insert can
// never find a "matching existing key" -- every key in the old array
// is already distinct -- so reuse is exactly as correct as inserting
// a brand-new key). Counting tombstones alongside live entries here,
// not just live entries alone, is what actually bounds a pathological
// insert/delete/insert/delete... churn pattern: without it, a table
// could fill entirely with tombstones while its own live count stays
// tiny, degrading every future lookup toward scanning the whole
// table before ever reaching an EMPTY slot. A grow-triggered rehash
// starts the new array with ZERO tombstones (nothing dead is ever
// copied over), so dict_set_tombstones(descriptor, 0) at the end is
// always correct regardless of how many tombstones the OLD array had
// accumulated. The stale old buckets array is never freed, matching
// this compiler's own "never free, only leak" story throughout (no
// GC yet).
static void dict_grow_scalar_key_if_needed(void *descriptor, int64_t key_width, int64_t value_width) {
    int64_t capacity = dict_capacity(descriptor);
    int32_t count = read_i32((char *)descriptor + 8);
    int32_t tombstones = dict_tombstones(descriptor);
    if ((count + tombstones + 1) * 4 <= capacity * 3) {
        return;  // (count + tombstones + 1) / capacity <= 0.75, room for one more
    }
    int64_t bucket_stride = 1 + key_width + value_width;
    int64_t new_capacity = capacity * 2;
    void *new_buckets = calloc((size_t)new_capacity, (size_t)bucket_stride);
    void *old_buckets = dict_buckets(descriptor);
    for (int64_t i = 0; i < capacity; i++) {
        unsigned char *bucket = (unsigned char *)old_buckets + i * bucket_stride;
        if (bucket[0] != HORNET_DICT_BUCKET_OCCUPIED) {
            continue;
        }
        hornet_dict_insert_scalar_key(
            new_buckets, new_capacity, bucket_stride,
            bucket + 1, key_width, bucket + 1 + key_width, value_width);
    }
    dict_set_buckets(descriptor, new_buckets);
    dict_set_capacity(descriptor, new_capacity);
    dict_set_tombstones(descriptor, 0);
}

static void dict_grow_str_key_if_needed(void *descriptor, int64_t value_width) {
    const int64_t key_region_width = 16;  // see hornet_dict_insert_str_key's own comment
    int64_t capacity = dict_capacity(descriptor);
    int32_t count = read_i32((char *)descriptor + 8);
    int32_t tombstones = dict_tombstones(descriptor);
    if ((count + tombstones + 1) * 4 <= capacity * 3) {
        return;
    }
    int64_t bucket_stride = 1 + key_region_width + value_width;
    int64_t new_capacity = capacity * 2;
    void *new_buckets = calloc((size_t)new_capacity, (size_t)bucket_stride);
    void *old_buckets = dict_buckets(descriptor);
    for (int64_t i = 0; i < capacity; i++) {
        unsigned char *bucket = (unsigned char *)old_buckets + i * bucket_stride;
        if (bucket[0] != HORNET_DICT_BUCKET_OCCUPIED) {
            continue;
        }
        void *stored_ptr = *(void **)(bucket + 1);
        int32_t stored_len = *(int32_t *)(bucket + 1 + sizeof(void *));
        hornet_dict_insert_str_key(
            new_buckets, new_capacity, bucket_stride,
            stored_ptr, stored_len, bucket + 1 + key_region_width, value_width);
    }
    dict_set_buckets(descriptor, new_buckets);
    dict_set_capacity(descriptor, new_capacity);
    dict_set_tombstones(descriptor, 0);
}

// The stage-2 (`d[key] = value`) counterpart to this file's own
// stage-1 hornet_dict_insert_scalar_key/hornet_dict_insert_str_key,
// used by a dict LITERAL's own fixed, pre-sized construction: grows
// first if needed (see the two functions just above), THEN inserts
// via the identical probe logic, finally updating descriptor's own
// count/tombstones fields itself based on the tri-state result (see
// hornet_dict_insert_scalar_key's own docstring for what each of the
// three return values means) -- the literal-construction path
// instead threads that decision back through IR as a chain of ADDs,
// since it has no persistent descriptor to mutate yet at that point
// in its own construction, and never has tombstones to begin with.
void hornet_dict_set_scalar_key(
    void *descriptor, int64_t key_width, int64_t value_width,
    const void *key_ptr, const void *value_ptr
) {
    dict_grow_scalar_key_if_needed(descriptor, key_width, value_width);
    int64_t bucket_stride = 1 + key_width + value_width;
    int result = hornet_dict_insert_scalar_key(
        dict_buckets(descriptor), dict_capacity(descriptor), bucket_stride,
        key_ptr, key_width, value_ptr, value_width);
    if (result != 0) {
        dict_bump_count(descriptor);
    }
    if (result == 2) {
        dict_bump_tombstones(descriptor, -1);
    }
}

void hornet_dict_set_str_key(
    void *descriptor, int64_t value_width,
    const void *key_ptr, int64_t key_len, const void *value_ptr
) {
    dict_grow_str_key_if_needed(descriptor, value_width);
    const int64_t key_region_width = 16;
    int64_t bucket_stride = 1 + key_region_width + value_width;
    int result = hornet_dict_insert_str_key(
        dict_buckets(descriptor), dict_capacity(descriptor), bucket_stride,
        key_ptr, key_len, value_ptr, value_width);
    if (result != 0) {
        dict_bump_count(descriptor);
    }
    if (result == 2) {
        dict_bump_tombstones(descriptor, -1);
    }
}

// `x = d[key]` -- probes for a matching key, SKIPPING PAST a
// tombstone rather than stopping there (a tombstone means "something
// used to live here, keep looking", not "the probe chain ends here"
// -- see HORNET_DICT_BUCKET_TOMBSTONE's own docstring) -- but never
// writes: returns the matching bucket's own VALUE address for the
// caller to read from, or -- a key genuinely absent, a truly EMPTY
// slot reached with no match along the way -- panics outright (see
// hornet_panic's own unconditional abort()) rather than returning
// anything at all, per this feature's own confirmed design: a missing
// key is a hard error, not a silent zero value.
void *hornet_dict_lookup_scalar_key(void *descriptor, int64_t key_width, int64_t value_width, const void *key_ptr) {
    void *buckets = dict_buckets(descriptor);
    int64_t capacity = dict_capacity(descriptor);
    int64_t bucket_stride = 1 + key_width + value_width;
    int64_t index = hornet_hash_bytes(key_ptr, key_width) & (capacity - 1);
    while (1) {
        unsigned char *bucket = (unsigned char *)buckets + index * bucket_stride;
        if (bucket[0] == HORNET_DICT_BUCKET_EMPTY) {
            hornet_panic("dict lookup: key not found");
        }
        if (bucket[0] == HORNET_DICT_BUCKET_OCCUPIED && memcmp(bucket + 1, key_ptr, (size_t)key_width) == 0) {
            return bucket + 1 + key_width;
        }
        index = (index + 1) & (capacity - 1);
    }
}

void *hornet_dict_lookup_str_key(void *descriptor, int64_t value_width, const void *key_ptr, int64_t key_len) {
    const int64_t key_region_width = 16;
    void *buckets = dict_buckets(descriptor);
    int64_t capacity = dict_capacity(descriptor);
    int64_t bucket_stride = 1 + key_region_width + value_width;
    int64_t index = hornet_hash_bytes(key_ptr, key_len) & (capacity - 1);
    while (1) {
        unsigned char *bucket = (unsigned char *)buckets + index * bucket_stride;
        if (bucket[0] == HORNET_DICT_BUCKET_EMPTY) {
            hornet_panic("dict lookup: key not found");
        }
        if (bucket[0] == HORNET_DICT_BUCKET_OCCUPIED) {
            void *stored_ptr = *(void **)(bucket + 1);
            int32_t stored_len = *(int32_t *)(bucket + 1 + sizeof(void *));
            if (stored_len == key_len && memcmp(stored_ptr, key_ptr, (size_t)key_len) == 0) {
                return bucket + 1 + key_region_width;
            }
        }
        index = (index + 1) & (capacity - 1);
    }
}

// `key in d` -- probes exactly like hornet_dict_lookup_scalar_key's
// own read side (skipping past a tombstone rather than stopping
// there), but never panics and never returns an address: a truly
// EMPTY slot reached with no match found simply means the key is
// ABSENT, an ordinary, fully-expected outcome for a membership test
// (unlike a lookup, where absence is a hard error) -- so this
// returns 0 rather than calling hornet_panic. value_width is only
// ever used to compute bucket_stride correctly; the value's own
// bytes are never read at all, since a membership test has nothing
// to do with the value on either a hit or a miss.
int hornet_dict_contains_scalar_key(void *descriptor, int64_t key_width, int64_t value_width, const void *key_ptr) {
    void *buckets = dict_buckets(descriptor);
    int64_t capacity = dict_capacity(descriptor);
    int64_t bucket_stride = 1 + key_width + value_width;
    int64_t index = hornet_hash_bytes(key_ptr, key_width) & (capacity - 1);
    while (1) {
        unsigned char *bucket = (unsigned char *)buckets + index * bucket_stride;
        if (bucket[0] == HORNET_DICT_BUCKET_EMPTY) {
            return 0;
        }
        if (bucket[0] == HORNET_DICT_BUCKET_OCCUPIED && memcmp(bucket + 1, key_ptr, (size_t)key_width) == 0) {
            return 1;
        }
        index = (index + 1) & (capacity - 1);
    }
}

int hornet_dict_contains_str_key(void *descriptor, int64_t value_width, const void *key_ptr, int64_t key_len) {
    const int64_t key_region_width = 16;
    void *buckets = dict_buckets(descriptor);
    int64_t capacity = dict_capacity(descriptor);
    int64_t bucket_stride = 1 + key_region_width + value_width;
    int64_t index = hornet_hash_bytes(key_ptr, key_len) & (capacity - 1);
    while (1) {
        unsigned char *bucket = (unsigned char *)buckets + index * bucket_stride;
        if (bucket[0] == HORNET_DICT_BUCKET_EMPTY) {
            return 0;
        }
        if (bucket[0] == HORNET_DICT_BUCKET_OCCUPIED) {
            void *stored_ptr = *(void **)(bucket + 1);
            int32_t stored_len = *(int32_t *)(bucket + 1 + sizeof(void *));
            if (stored_len == key_len && memcmp(stored_ptr, key_ptr, (size_t)key_len) == 0) {
                return 1;
            }
        }
        index = (index + 1) & (capacity - 1);
    }
}

// `del(d, key)` -- probes exactly like a lookup (skipping tombstones,
// panicking on a truly EMPTY slot with no match found), but on a
// match, marks the bucket a TOMBSTONE instead of reading its own
// value: decrements count (one fewer LIVE entry) and increments
// tombstones (one more dead slot future inserts may eventually reuse,
// and future growth checks must account for -- see dict_grow_*_key_
// if_needed's own docstring).
void hornet_dict_delete_scalar_key(void *descriptor, int64_t key_width, int64_t value_width, const void *key_ptr) {
    void *buckets = dict_buckets(descriptor);
    int64_t capacity = dict_capacity(descriptor);
    int64_t bucket_stride = 1 + key_width + value_width;
    int64_t index = hornet_hash_bytes(key_ptr, key_width) & (capacity - 1);
    while (1) {
        unsigned char *bucket = (unsigned char *)buckets + index * bucket_stride;
        if (bucket[0] == HORNET_DICT_BUCKET_EMPTY) {
            hornet_panic("dict delete: key not found");
        }
        if (bucket[0] == HORNET_DICT_BUCKET_OCCUPIED && memcmp(bucket + 1, key_ptr, (size_t)key_width) == 0) {
            bucket[0] = HORNET_DICT_BUCKET_TOMBSTONE;
            int32_t *count = (int32_t *)((char *)descriptor + 8);
            *count -= 1;
            dict_bump_tombstones(descriptor, 1);
            return;
        }
        index = (index + 1) & (capacity - 1);
    }
}

void hornet_dict_delete_str_key(void *descriptor, int64_t value_width, const void *key_ptr, int64_t key_len) {
    const int64_t key_region_width = 16;
    void *buckets = dict_buckets(descriptor);
    int64_t capacity = dict_capacity(descriptor);
    int64_t bucket_stride = 1 + key_region_width + value_width;
    int64_t index = hornet_hash_bytes(key_ptr, key_len) & (capacity - 1);
    while (1) {
        unsigned char *bucket = (unsigned char *)buckets + index * bucket_stride;
        if (bucket[0] == HORNET_DICT_BUCKET_EMPTY) {
            hornet_panic("dict delete: key not found");
        }
        if (bucket[0] == HORNET_DICT_BUCKET_OCCUPIED) {
            void *stored_ptr = *(void **)(bucket + 1);
            int32_t stored_len = *(int32_t *)(bucket + 1 + sizeof(void *));
            if (stored_len == key_len && memcmp(stored_ptr, key_ptr, (size_t)key_len) == 0) {
                bucket[0] = HORNET_DICT_BUCKET_TOMBSTONE;
                int32_t *count = (int32_t *)((char *)descriptor + 8);
                *count -= 1;
                dict_bump_tombstones(descriptor, 1);
                return;
            }
        }
        index = (index + 1) & (capacity - 1);
    }
}
