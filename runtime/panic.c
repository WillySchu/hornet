// Panics: what is said, then abort().
#include "internal.h"

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

// `Enum(value)`, when no member has that value (`count` of them, from 0: it isn't shown).
void hornet_panic_enum_value(const char *msg, int64_t value, int64_t count) {
    (void)count;
    fflush(stdout);  // the program's own output first
    fprintf(stderr, "%s: %lld\n", msg, (long long)value);
    fflush(stderr);
    abort();
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
