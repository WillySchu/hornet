// Hornet runtime: print, panic, bytes(), slice growth, dict hash tables, and what stdlib/os.ht calls.

// What the system's headers are asked for beyond standard C: POSIX with its extensions (signal
// stacks), and on macOS the system's own (a thread's stack).
#ifndef _WIN32
#define _XOPEN_SOURCE 700
#define _DARWIN_C_SOURCE
#endif

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef _WIN32
#include <io.h>
// Windows opens streams and files in text mode, which rewrites "\n" as "\r\n" on the way out and
// back on the way in. Hornet's I/O is bytes, so everything is binary.
#define HORNET_O_BINARY _O_BINARY
__attribute__((constructor)) static void hornet_binary_standard_streams(void) {
    _setmode(0, _O_BINARY);
    _setmode(1, _O_BINARY);
    _setmode(2, _O_BINARY);
}
#else
#define HORNET_O_BINARY 0
#endif

#include "hornet_typedesc_tags.h"

// ---- A stack overflow is a panic.
//
// Running off the end of the stack is a fault like any bad memory access. What catches it tells
// the two apart, says `panic: stack overflow`, and ends the program as a panic does; any other
// fault is left to crash as it would have. (There is no source position: a fault doesn't say where
// in the program it happened.)

static const char hornet_stack_overflow[] = "panic: stack overflow\n";

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// Windows raises an exception of its own for it. The handler runs on the stack that overflowed, in
// the space SetThreadStackGuarantee keeps back.
static LONG WINAPI hornet_on_exception(EXCEPTION_POINTERS *info) {
    if (info->ExceptionRecord->ExceptionCode == EXCEPTION_STACK_OVERFLOW) {
        DWORD written;
        WriteFile(GetStdHandle(STD_ERROR_HANDLE), hornet_stack_overflow, sizeof hornet_stack_overflow - 1, &written,
                  NULL);
        TerminateProcess(GetCurrentProcess(), 3);  // abort()'s exit code
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

__attribute__((constructor)) static void hornet_catch_stack_overflow(void) {
    ULONG kept_back = 64 * 1024;
    SetThreadStackGuarantee(&kept_back);
    AddVectoredExceptionHandler(1, hornet_on_exception);
}
#else
#include <signal.h>
#include <sys/resource.h>
#ifdef __APPLE__
#include <pthread.h>
#endif

// Elsewhere it is a signal, handled on a stack of its own (the program's has no room left).
static char hornet_signal_stack[64 * 1024];
// The stack's range, as far as it is known: its lowest address is an estimate on Linux.
static char *hornet_stack_low, *hornet_stack_high;
// A frame larger than the protected memory below the stack can step past it, so a fault this far
// below the stack still counts.
#define HORNET_BELOW_STACK ((intptr_t)64 * 1024 * 1024)

static void hornet_on_fault(int sig, siginfo_t *info, void *context) {
    (void)context;
    char *address = (char *)info->si_addr;
    // Memory in the stack's own range doesn't fault unless the stack has run out.
    if (address >= hornet_stack_low - HORNET_BELOW_STACK && address < hornet_stack_high) {
        ssize_t written = write(2, hornet_stack_overflow, sizeof hornet_stack_overflow - 1);
        (void)written;
        abort();
    }
    signal(sig, SIG_DFL);  // not the stack: back to the instruction, to fault again and crash
}

__attribute__((constructor)) static void hornet_catch_stack_overflow(void) {
#ifdef __APPLE__
    hornet_stack_high = (char *)pthread_get_stackaddr_np(pthread_self());
    hornet_stack_low = hornet_stack_high - pthread_get_stacksize_np(pthread_self());
#else
    // The main thread's stack grows down from about here, to the size the system allows it.
    struct rlimit limit;
    char here;
    if (getrlimit(RLIMIT_STACK, &limit) != 0 || limit.rlim_cur == RLIM_INFINITY) {
        return;  // no end to look for
    }
    hornet_stack_high = &here + 4096;
    hornet_stack_low = &here - limit.rlim_cur;
#endif
    stack_t stack;
    memset(&stack, 0, sizeof stack);
    stack.ss_sp = hornet_signal_stack;
    stack.ss_size = sizeof hornet_signal_stack;
    struct sigaction action;
    memset(&action, 0, sizeof action);
    action.sa_sigaction = hornet_on_fault;
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);
    if (sigaltstack(&stack, NULL) == 0) {
        sigaction(SIGSEGV, &action, NULL);
        sigaction(SIGBUS, &action, NULL);  // (macOS reports protected memory this way)
    }
}
#endif

// Bucket state byte. Tombstones keep probe chains intact after deletion.
#define HORNET_DICT_BUCKET_EMPTY 0
#define HORNET_DICT_BUCKET_OCCUPIED 1
#define HORNET_DICT_BUCKET_TOMBSTONE 2

// Growable byte buffer for print.
struct hornet_buf {
    char *ptr;
    int64_t len;
    int64_t cap;
};

static void hornet_stringify(
    void *value_addr, const unsigned char *type_desc, int quote_strings, struct hornet_buf *buf);

// Ensure `additional` spare bytes, growing by doubling.
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

// Unaligned reads: Hornet structs are packed.
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

// `index`-th 8-byte word of a type descriptor.
static uint64_t read_desc_word(const unsigned char *desc, int64_t index) {
    uint64_t value;
    memcpy(&value, desc + index * 8, sizeof(value));
    return value;
}

// Format the value at value_addr per type_desc (type_desc[0] is the kind tag).
static void hornet_stringify(
    void *value_addr, const unsigned char *type_desc, int quote_strings, struct hornet_buf *buf) {
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
                hornet_stringify(elem_addr, elem_desc, 1, buf);
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
                hornet_stringify(elem_addr, elem_desc, 1, buf);
            }
            hornet_buf_append_byte(buf, ']');
            break;
        }
        case HORNET_TYPEDESC_DICT: {
            // [tag, name, key_desc, key_width, value_desc, value_width]
            const char *name = (const char *)read_desc_word(type_desc, 1);
            const unsigned char *key_desc = (const unsigned char *)read_desc_word(type_desc, 2);
            int64_t key_width = (int64_t)read_desc_word(type_desc, 3);
            const unsigned char *value_desc = (const unsigned char *)read_desc_word(type_desc, 4);
            int64_t value_width = (int64_t)read_desc_word(type_desc, 5);
            int64_t bucket_stride = 1 + key_width + value_width;

            void *header = read_ptr(value_addr);  // a dict value points to its shared header
            void *buckets = read_ptr(header);
            int64_t capacity = read_i64((char *)header + 24);

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
                hornet_stringify(field_addr, field_type_desc, 1, buf);
            }
            hornet_buf_append_byte(buf, ')');
            break;
        }
        case HORNET_TYPEDESC_SUM: {
            // [tag, variant_count, variant_desc...]; prints the active variant.
            int32_t discriminant = read_i32(value_addr);
            const unsigned char *variant_desc =
                (const unsigned char *)read_desc_word(type_desc, 2 + discriminant);
            void *payload_addr = (char *)value_addr + 4;
            hornet_stringify(payload_addr, variant_desc, 1, buf);
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
            // Prints the address, not the pointee.
            void *value = read_ptr(value_addr);
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

// print(x) entry point.
// bytes(s): *out = a new []byte copy of the len bytes at ptr. Hornet's calling convention
// passes the result slot first, then the str as (ptr, len).
struct hornet_slice {
    void *ptr;
    int64_t len;
    int64_t cap;
};

void hornet_bytes(struct hornet_slice *out, const char *ptr, int64_t len) {
    out->ptr = malloc(len > 0 ? (size_t)len : 1);
    if (len > 0) {
        memcpy(out->ptr, ptr, (size_t)len);
    }
    out->len = len;
    out->cap = len;
}

// Write all `len` bytes to `fd`, retrying partial writes; -1 on error.
int64_t hornet_write_fd(int64_t fd, const char *ptr, int64_t len) {
    int64_t done = 0;
    while (done < len) {
        ssize_t n = write((int)fd, ptr + done, (size_t)(len - done));
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return -1;
        }
        done += n;
    }
    return done;
}

// open(2) for writing, creating or truncating (open is variadic, so it's called from C).
int64_t hornet_open_write(const char *path) {
    return open(path, O_WRONLY | O_CREAT | O_TRUNC | HORNET_O_BINARY, 0644);
}

// open(2) for reading.
int64_t hornet_open_read(const char *path) {
    return open(path, O_RDONLY | HORNET_O_BINARY);
}

// read(2): up to `len` bytes; 0 at the end, -1 on error. (C's own returns an int on Windows.)
// Standard input goes through stdio, so that this and hornet_read_line share what is buffered.
int64_t hornet_read_fd(int64_t fd, char *ptr, int64_t len) {
    if (fd == 0) {
        size_t n = fread(ptr, 1, (size_t)len, stdin);
        return n == 0 && ferror(stdin) ? -1 : (int64_t)n;
    }
    return read((int)fd, ptr, (unsigned int)len);
}

// The next line of standard input, without its "\n" or "\r\n", in memory the caller keeps; its
// length goes to *len. NULL at the end of the input (*len 0), or on an error (*len -1).
char *hornet_read_line(int64_t *len) {
    size_t cap = 128, n = 0;
    char *line = malloc(cap);
    int c;
    while ((c = getc(stdin)) != EOF && c != '\n') {
        if (n == cap) {
            cap *= 2;
            line = realloc(line, cap);
        }
        line[n++] = (char)c;
    }
    if (c == EOF && n == 0) {
        free(line);
        *len = ferror(stdin) ? -1 : 0;
        return NULL;
    }
    if (n > 0 && line[n - 1] == '\r') {
        n--;
    }
    *len = (int64_t)n;
    return line;
}

// Whether `fd` is a terminal (so a person, not a file or a pipe, is at the other end).
int64_t hornet_is_terminal(int64_t fd) {
    return isatty((int)fd) != 0;
}

int64_t hornet_close_fd(int64_t fd) {
    return close((int)fd);
}

// strerror(errno) for the most recent failed call.
const char *hornet_error_message(void) {
    return strerror(errno);
}

// Flush stdio, then exit.
void hornet_exit(int64_t code) {
    fflush(NULL);
    exit((int)code);
}

void hornet_print(void *value_addr, const unsigned char *type_desc) {
    struct hornet_buf buf;
    buf.cap = 16;
    buf.ptr = malloc((size_t)buf.cap);
    buf.len = 0;
    hornet_stringify(value_addr, type_desc, 0, &buf);
    hornet_buf_append_byte(&buf, '\n');
    hornet_write_fd(1, buf.ptr, buf.len);
    free(buf.ptr);
}

// Print `msg` and abort.
void hornet_panic(const char *msg) {
    fflush(stdout);  // the program's own output first
    fputs(msg, stderr);
    fputc('\n', stderr);
    fflush(stderr);
    abort();
}

// `panic(message)`: print `where: panic: message` and abort. `where` is the call's position, empty
// if it has none; the message is a str, so not NUL-terminated.
// A failed bounds check: the message, then the two numbers that were compared, each named.
static void hornet_panic_with(const char *msg, const char *first_name, int64_t first, const char *second_name,
                              int64_t second) {
    fflush(stdout);  // the program's own output first
    fprintf(stderr, "%s: %s %lld, %s %lld\n", msg, first_name, (long long)first, second_name, (long long)second);
    fflush(stderr);
    abort();
}

// `a[index]`, of an array, slice, or str of `length`.
void hornet_panic_index(const char *msg, int64_t index, int64_t length) {
    hornet_panic_with(msg, "index", index, "length", length);
}

// `a[start:end]`: a bound past the length (of a slice: its capacity), or the two out of order.
void hornet_panic_slice_start(const char *msg, int64_t start, int64_t length) {
    hornet_panic_with(msg, "start", start, "length", length);
}

void hornet_panic_slice_end(const char *msg, int64_t end, int64_t length) {
    hornet_panic_with(msg, "end", end, "length", length);
}

void hornet_panic_slice_start_capacity(const char *msg, int64_t start, int64_t capacity) {
    hornet_panic_with(msg, "start", start, "capacity", capacity);
}

void hornet_panic_slice_end_capacity(const char *msg, int64_t end, int64_t capacity) {
    hornet_panic_with(msg, "end", end, "capacity", capacity);
}

void hornet_panic_slice_order(const char *msg, int64_t start, int64_t end) {
    hornet_panic_with(msg, "start", start, "end", end);
}

void hornet_panic_at(const char *where, const char *msg, int64_t len) {
    fflush(stdout);  // the program's own output first
    if (where[0] != '\0') {
        fputs(where, stderr);
        fputs(": ", stderr);
    }
    fputs("panic: ", stderr);
    fwrite(msg, 1, (size_t)len, stderr);
    fputc('\n', stderr);
    fflush(stderr);
    abort();
}

// malloc new_cap * element_width bytes and copy `len` elements.
void *hornet_slice_grow(const void *old_ptr, int64_t len, int64_t new_cap, int64_t element_width) {
    void *new_ptr = malloc((size_t)new_cap * (size_t)element_width);
    memcpy(new_ptr, old_ptr, (size_t)len * (size_t)element_width);
    return new_ptr;
}

// FNV-1a 64 over bytes. Scalar keys hash their value bytes; str keys hash their content.
int64_t hornet_hash_bytes(const void *ptr, int64_t len) {
    uint64_t h = 0xcbf29ce484222325ULL;  // FNV offset basis
    const unsigned char *bytes = (const unsigned char *)ptr;
    for (int64_t i = 0; i < len; i++) {
        h ^= bytes[i];
        h *= 0x100000001b3ULL;  // FNV prime
    }
    return (int64_t)h;
}

// Insert into a table with room (literals, and after any rehash): returns 0 if it overwrote the
// key, 1 if the key is new, 2 if new in a tombstone's bucket.
int64_t hornet_dict_insert_scalar_key(
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

// Str-keyed insert; the bucket stores the key's {ptr, len}.
int64_t hornet_dict_insert_str_key(
    void *buckets, int64_t capacity, int64_t bucket_stride,
    const void *key_ptr, int64_t key_len,
    const void *value_ptr, int64_t value_width
) {
    // type_byte_width(str)
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
            *(int64_t *)(target_bucket + 1 + sizeof(void *)) = key_len;
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
        int64_t stored_len = read_i64(bucket + 1 + sizeof(void *));
        if (stored_len == key_len && memcmp(stored_ptr, key_ptr, (size_t)key_len) == 0) {
            memcpy(bucket + 1 + key_region_width, value_ptr, (size_t)value_width);
            return 0;
        }
        index = (index + 1) & (capacity - 1);
    }
}

// Descriptor fields (8 bytes each): buckets @0, count @8, tombstones @16, capacity @24.
static void *dict_buckets(void *descriptor) { return read_ptr(descriptor); }
static int64_t dict_capacity(void *descriptor) { return read_i64((char *)descriptor + 24); }

static void dict_set_buckets(void *descriptor, void *buckets) {
    *(void **)descriptor = buckets;
}
static void dict_set_capacity(void *descriptor, int64_t capacity) {
    *(int64_t *)((char *)descriptor + 24) = capacity;
}
static void dict_bump_count(void *descriptor) {
    int64_t *count = (int64_t *)((char *)descriptor + 8);
    *count += 1;
}

static int64_t dict_tombstones(void *descriptor) { return read_i64((char *)descriptor + 16); }
static void dict_set_tombstones(void *descriptor, int64_t tombstones) {
    *(int64_t *)((char *)descriptor + 16) = tombstones;
}
static void dict_bump_tombstones(void *descriptor, int64_t delta) {
    dict_set_tombstones(descriptor, dict_tombstones(descriptor) + delta);
}

// A rehash's capacity: 8 for a dict's first write; unchanged when at most half the buckets would
// be live (tombstones filled the table); otherwise doubled.
static int64_t dict_rehash_capacity(int64_t capacity, int64_t count) {
    if (capacity == 0) {
        return 8;
    }
    return (count + 1) * 2 <= capacity ? capacity : capacity * 2;
}

// Rehash the live buckets into a new table once (count + tombstones + 1) exceeds 75%.
static void dict_grow_scalar_key_if_needed(void *descriptor, int64_t key_width, int64_t value_width) {
    int64_t capacity = dict_capacity(descriptor);
    int64_t count = read_i64((char *)descriptor + 8);
    int64_t tombstones = dict_tombstones(descriptor);
    if ((count + tombstones + 1) * 4 <= capacity * 3) {
        return;
    }
    int64_t bucket_stride = 1 + key_width + value_width;
    int64_t new_capacity = dict_rehash_capacity(capacity, count);
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
    const int64_t key_region_width = 16;  // type_byte_width(str)
    int64_t capacity = dict_capacity(descriptor);
    int64_t count = read_i64((char *)descriptor + 8);
    int64_t tombstones = dict_tombstones(descriptor);
    if ((count + tombstones + 1) * 4 <= capacity * 3) {
        return;
    }
    int64_t bucket_stride = 1 + key_region_width + value_width;
    int64_t new_capacity = dict_rehash_capacity(capacity, count);
    void *new_buckets = calloc((size_t)new_capacity, (size_t)bucket_stride);
    void *old_buckets = dict_buckets(descriptor);
    for (int64_t i = 0; i < capacity; i++) {
        unsigned char *bucket = (unsigned char *)old_buckets + i * bucket_stride;
        if (bucket[0] != HORNET_DICT_BUCKET_OCCUPIED) {
            continue;
        }
        void *stored_ptr = *(void **)(bucket + 1);
        int64_t stored_len = read_i64(bucket + 1 + sizeof(void *));
        hornet_dict_insert_str_key(
            new_buckets, new_capacity, bucket_stride,
            stored_ptr, stored_len, bucket + 1 + key_region_width, value_width);
    }
    dict_set_buckets(descriptor, new_buckets);
    dict_set_capacity(descriptor, new_capacity);
    dict_set_tombstones(descriptor, 0);
}

// `d[k] = v`: grow if needed, then insert.
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

// `d[k]`: address of the value, or NULL if the key is missing (the compiled code panics, with the
// source position). Probes past tombstones.
// The key a lookup or a delete has just failed to find. The compiled code panics as soon as one
// returns "not found" (`k in d` is a different call), and the panic prints the key from here: so
// finding a key costs nothing for it.
static const void *hornet_missing_key;
static struct {
    const void *ptr;
    int64_t len;
} hornet_missing_str_key;

static void hornet_note_missing_key(const void *key_ptr) {
    hornet_missing_key = key_ptr;  // (in the caller's frame, which is still there for the panic)
}

static void hornet_note_missing_str_key(const void *key_ptr, int64_t key_len) {
    hornet_missing_str_key.ptr = key_ptr;
    hornet_missing_str_key.len = key_len;
    hornet_missing_key = &hornet_missing_str_key;
}

// How much of a str key a panic shows.
#define HORNET_PANIC_KEY_BYTES 100

// `msg: key`, the key as print shows one inside a value: `d[k]` or `del(d, k)` without the key.
void hornet_panic_missing_key(const char *msg, const unsigned char *key_desc) {
    struct hornet_buf buf;
    buf.cap = 64;
    buf.ptr = malloc((size_t)buf.cap);
    buf.len = 0;
    hornet_buf_append_cstr(&buf, msg);
    hornet_buf_append_cstr(&buf, ": ");
    if (read_desc_word(key_desc, 0) == HORNET_TYPEDESC_STR && hornet_missing_str_key.len > HORNET_PANIC_KEY_BYTES) {
        char rest[48];
        hornet_buf_append_byte(&buf, '\'');
        hornet_buf_append_bytes(&buf, hornet_missing_str_key.ptr, HORNET_PANIC_KEY_BYTES);
        int n = snprintf(rest, sizeof(rest), "...' (%lld bytes)", (long long)hornet_missing_str_key.len);
        hornet_buf_append_bytes(&buf, rest, n);
    } else {
        hornet_stringify((void *)hornet_missing_key, key_desc, 1, &buf);
    }
    hornet_buf_append_byte(&buf, '\n');
    fflush(stdout);  // the program's own output first
    fwrite(buf.ptr, 1, (size_t)buf.len, stderr);
    fflush(stderr);
    abort();
}

void *hornet_dict_lookup_scalar_key(void *descriptor, int64_t key_width, int64_t value_width, const void *key_ptr) {
    void *buckets = dict_buckets(descriptor);
    int64_t capacity = dict_capacity(descriptor);
    if (capacity == 0) {
        // no table yet: capacity 0 would break the mask.
        hornet_note_missing_key(key_ptr);
        return NULL;
    }
    int64_t bucket_stride = 1 + key_width + value_width;
    int64_t index = hornet_hash_bytes(key_ptr, key_width) & (capacity - 1);
    while (1) {
        unsigned char *bucket = (unsigned char *)buckets + index * bucket_stride;
        if (bucket[0] == HORNET_DICT_BUCKET_EMPTY) {
            hornet_note_missing_key(key_ptr);
            return NULL;
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
    if (capacity == 0) {
        // no table yet
        hornet_note_missing_str_key(key_ptr, key_len);
        return NULL;
    }
    int64_t bucket_stride = 1 + key_region_width + value_width;
    int64_t index = hornet_hash_bytes(key_ptr, key_len) & (capacity - 1);
    while (1) {
        unsigned char *bucket = (unsigned char *)buckets + index * bucket_stride;
        if (bucket[0] == HORNET_DICT_BUCKET_EMPTY) {
            hornet_note_missing_str_key(key_ptr, key_len);
            return NULL;
        }
        if (bucket[0] == HORNET_DICT_BUCKET_OCCUPIED) {
            void *stored_ptr = *(void **)(bucket + 1);
            int64_t stored_len = read_i64(bucket + 1 + sizeof(void *));
            if (stored_len == key_len && memcmp(stored_ptr, key_ptr, (size_t)key_len) == 0) {
                return bucket + 1 + key_region_width;
            }
        }
        index = (index + 1) & (capacity - 1);
    }
}

// `k in d`.
int64_t hornet_dict_contains_scalar_key(void *descriptor, int64_t key_width, int64_t value_width, const void *key_ptr) {
    void *buckets = dict_buckets(descriptor);
    int64_t capacity = dict_capacity(descriptor);
    if (capacity == 0) {
        // no table yet
        return 0;
    }
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

int64_t hornet_dict_contains_str_key(void *descriptor, int64_t value_width, const void *key_ptr, int64_t key_len) {
    const int64_t key_region_width = 16;
    void *buckets = dict_buckets(descriptor);
    int64_t capacity = dict_capacity(descriptor);
    if (capacity == 0) {
        // no table yet
        return 0;
    }
    int64_t bucket_stride = 1 + key_region_width + value_width;
    int64_t index = hornet_hash_bytes(key_ptr, key_len) & (capacity - 1);
    while (1) {
        unsigned char *bucket = (unsigned char *)buckets + index * bucket_stride;
        if (bucket[0] == HORNET_DICT_BUCKET_EMPTY) {
            return 0;
        }
        if (bucket[0] == HORNET_DICT_BUCKET_OCCUPIED) {
            void *stored_ptr = *(void **)(bucket + 1);
            int64_t stored_len = read_i64(bucket + 1 + sizeof(void *));
            if (stored_len == key_len && memcmp(stored_ptr, key_ptr, (size_t)key_len) == 0) {
                return 1;
            }
        }
        index = (index + 1) & (capacity - 1);
    }
}

// `del(d, k)`: mark a tombstone and return 1, or return 0 if the key is missing (the compiled code
// panics, with the source position).
int64_t hornet_dict_delete_scalar_key(void *descriptor, int64_t key_width, int64_t value_width, const void *key_ptr) {
    void *buckets = dict_buckets(descriptor);
    int64_t capacity = dict_capacity(descriptor);
    if (capacity == 0) {
        // no table yet
        hornet_note_missing_key(key_ptr);
        return 0;
    }
    int64_t bucket_stride = 1 + key_width + value_width;
    int64_t index = hornet_hash_bytes(key_ptr, key_width) & (capacity - 1);
    while (1) {
        unsigned char *bucket = (unsigned char *)buckets + index * bucket_stride;
        if (bucket[0] == HORNET_DICT_BUCKET_EMPTY) {
            hornet_note_missing_key(key_ptr);
            return 0;
        }
        if (bucket[0] == HORNET_DICT_BUCKET_OCCUPIED && memcmp(bucket + 1, key_ptr, (size_t)key_width) == 0) {
            bucket[0] = HORNET_DICT_BUCKET_TOMBSTONE;
            int64_t *count = (int64_t *)((char *)descriptor + 8);
            *count -= 1;
            dict_bump_tombstones(descriptor, 1);
            return 1;
        }
        index = (index + 1) & (capacity - 1);
    }
}

int64_t hornet_dict_delete_str_key(void *descriptor, int64_t value_width, const void *key_ptr, int64_t key_len) {
    const int64_t key_region_width = 16;
    void *buckets = dict_buckets(descriptor);
    int64_t capacity = dict_capacity(descriptor);
    if (capacity == 0) {
        // no table yet
        hornet_note_missing_str_key(key_ptr, key_len);
        return 0;
    }
    int64_t bucket_stride = 1 + key_region_width + value_width;
    int64_t index = hornet_hash_bytes(key_ptr, key_len) & (capacity - 1);
    while (1) {
        unsigned char *bucket = (unsigned char *)buckets + index * bucket_stride;
        if (bucket[0] == HORNET_DICT_BUCKET_EMPTY) {
            hornet_note_missing_str_key(key_ptr, key_len);
            return 0;
        }
        if (bucket[0] == HORNET_DICT_BUCKET_OCCUPIED) {
            void *stored_ptr = *(void **)(bucket + 1);
            int64_t stored_len = read_i64(bucket + 1 + sizeof(void *));
            if (stored_len == key_len && memcmp(stored_ptr, key_ptr, (size_t)key_len) == 0) {
                bucket[0] = HORNET_DICT_BUCKET_TOMBSTONE;
                int64_t *count = (int64_t *)((char *)descriptor + 8);
                *count -= 1;
                dict_bump_tombstones(descriptor, 1);
                return 1;
            }
        }
        index = (index + 1) & (capacity - 1);
    }
}

// argv[index]; Hornet lacks pointer indexing. Used by stdlib/os.ht.
char *hornet_argv_get(char **argv, int64_t index) {
    return argv[index];
}
