// Memory.
//
// Every allocation, the compiled code's and the runtime's own, comes through here, so that being
// refused memory is a panic and not a write through a null pointer. (A system that grants memory it
// doesn't have, as Linux does by default, kills the program when the memory is used: there is no
// refusal to catch then.) Nothing is freed yet but the runtime's own scratch buffers.
#include "internal.h"

// `panic: out of memory (allocating N bytes)`, or `N x M bytes` when their product doesn't fit.
// From a buffer on the stack: this is no time to allocate.
static void hornet_out_of_memory(int64_t count, int64_t size) {
    char message[96];
    int n = size < 0
        ? snprintf(message, sizeof message, "panic: out of memory (allocating %lld bytes)\n", (long long)count)
        : snprintf(message, sizeof message, "panic: out of memory (allocating %lld x %lld bytes)\n",
                   (long long)count, (long long)size);
    fflush(stdout);  // the program's own output first
    ssize_t written = write(2, message, (size_t)n);
    (void)written;
    abort();
}

// `size` bytes, uninitialized. (No bytes is still somewhere to point: malloc(0) may give NULL.)
void *hornet_alloc(int64_t size) {
    void *ptr = size < 0 ? NULL : malloc(size > 0 ? (size_t)size : 1);
    if (ptr == NULL) {
        hornet_out_of_memory(size, -1);
    }
    return ptr;
}

// Whether `count` items of `size` bytes is a number of bytes an int64 holds.
static int hornet_fits(int64_t count, int64_t size) {
    return count >= 0 && size >= 0 && (size == 0 || count <= INT64_MAX / size);
}

// `count` items of `size` bytes, all zero.
void *hornet_alloc_zeroed(int64_t count, int64_t size) {
    if (!hornet_fits(count, size)) {
        hornet_out_of_memory(count, size);
    }
    void *ptr = calloc(count > 0 ? (size_t)count : 1, size > 0 ? (size_t)size : 1);
    if (ptr == NULL) {
        hornet_out_of_memory(count * size, -1);
    }
    return ptr;
}

void *hornet__alloc_items(int64_t count, int64_t size) {
    if (!hornet_fits(count, size)) {
        hornet_out_of_memory(count, size);
    }
    return hornet_alloc(count * size);
}

void *hornet__realloc(void *ptr, int64_t size) {
    void *grown = size < 0 ? NULL : realloc(ptr, size > 0 ? (size_t)size : 1);
    if (grown == NULL) {
        hornet_out_of_memory(size, -1);
    }
    return grown;
}

void hornet__buf_ensure(struct hornet_buf *buf, int64_t additional) {
    int64_t needed = buf->len + additional;
    if (needed <= buf->cap) {
        return;
    }
    int64_t doubled_or_quartered = buf->cap < 256 ? buf->cap * 2 : buf->cap + buf->cap / 4;
    int64_t new_cap = needed > doubled_or_quartered ? needed : doubled_or_quartered;
    buf->ptr = hornet__realloc(buf->ptr, new_cap);
    buf->cap = new_cap;
}
