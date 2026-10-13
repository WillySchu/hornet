// Memory.
//
// Every allocation, the compiled code's and the runtime's own, comes through here, so that being
// refused memory is a panic and not a write through a null pointer. (A system that grants memory it
// doesn't have, as Linux does by default, kills the program when the memory is used: there is no
// refusal to catch then.) Nothing is freed yet but the runtime's own scratch memory.
//
// Memory that may hold a pointer and memory that holds none (`_atomic`) are asked for apart, for
// the collector that will free what nothing points to: it need look for pointers only in the first.
// Both are malloc's until then.
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
static void *hornet_malloc(int64_t size) {
    void *ptr = size < 0 ? NULL : malloc(size > 0 ? (size_t)size : 1);
    if (ptr == NULL) {
        hornet_out_of_memory(size, -1);
    }
    return ptr;
}

// `size` bytes that may hold a pointer, uninitialized.
void *hornet_alloc(int64_t size) {
    return hornet_malloc(size);
}

// `size` bytes that hold no pointer, uninitialized.
void *hornet_alloc_atomic(int64_t size) {
    return hornet_malloc(size);
}

void *hornet__scratch_alloc(int64_t size) {
    return hornet_malloc(size);
}

void hornet__scratch_free(void *ptr) {
    free(ptr);
}

// Whether `count` items of `size` bytes is a number of bytes an int64 holds.
static int hornet_fits(int64_t count, int64_t size) {
    return count >= 0 && size >= 0 && (size == 0 || count <= INT64_MAX / size);
}

// `count` items of `size` bytes, all zero.
static void *hornet_calloc(int64_t count, int64_t size) {
    if (!hornet_fits(count, size)) {
        hornet_out_of_memory(count, size);
    }
    void *ptr = calloc(count > 0 ? (size_t)count : 1, size > 0 ? (size_t)size : 1);
    if (ptr == NULL) {
        hornet_out_of_memory(count * size, -1);
    }
    return ptr;
}

// `count` items of `size` bytes, all zero, that may hold a pointer.
void *hornet_alloc_zeroed(int64_t count, int64_t size) {
    return hornet_calloc(count, size);
}

// `count` items of `size` bytes, all zero, that hold no pointer.
void *hornet_alloc_zeroed_atomic(int64_t count, int64_t size) {
    return hornet_calloc(count, size);
}

void *hornet__alloc_items(int64_t count, int64_t size, int holds_pointers) {
    if (!hornet_fits(count, size)) {
        hornet_out_of_memory(count, size);
    }
    return holds_pointers ? hornet_alloc(count * size) : hornet_alloc_atomic(count * size);
}

static void *hornet_realloc(void *ptr, int64_t size) {
    void *grown = size < 0 ? NULL : realloc(ptr, size > 0 ? (size_t)size : 1);
    if (grown == NULL) {
        hornet_out_of_memory(size, -1);
    }
    return grown;
}

void *hornet__realloc_atomic(void *ptr, int64_t size) {
    return hornet_realloc(ptr, size);
}

void *hornet__scratch_realloc(void *ptr, int64_t size) {
    return hornet_realloc(ptr, size);
}

void hornet__buf_ensure(struct hornet_buf *buf, int64_t additional) {
    int64_t needed = buf->len + additional;
    if (needed <= buf->cap) {
        return;
    }
    int64_t doubled_or_quartered = buf->cap < 256 ? buf->cap * 2 : buf->cap + buf->cap / 4;
    int64_t new_cap = needed > doubled_or_quartered ? needed : doubled_or_quartered;
    buf->ptr = buf->scratch ? hornet__scratch_realloc(buf->ptr, new_cap) : hornet__realloc_atomic(buf->ptr, new_cap);
    buf->cap = new_cap;
}
