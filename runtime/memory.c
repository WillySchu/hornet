// Memory.
//
// Every allocation, the compiled code's and the runtime's own, comes through here, so that being
// refused memory is a panic and not a write through a null pointer. (A system that grants memory it
// doesn't have, as Linux does by default, kills the program when the memory is used: there is no
// refusal to catch then.)
//
// Memory that may hold a pointer and memory that holds none (`_atomic`) are asked for apart, for
// the collector, which frees what nothing points to: it need look for pointers only in the first.
// Both are the heap's (heap.c); the runtime's scratch memory is malloc's.
#include "internal.h"

// From a buffer on the stack: this is no time to allocate.
void hornet__out_of_memory(int64_t count, int64_t size) {
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
        hornet__out_of_memory(size, -1);
    }
    return ptr;
}

// `size` bytes that may hold a pointer: all zero, though the caller may not count on it.
void *hornet_alloc(int64_t size) {
    return hornet__heap_alloc(size, 1, 0);
}

// `size` bytes that hold no pointer, uninitialized.
void *hornet_alloc_atomic(int64_t size) {
    return hornet__heap_alloc(size, 0, 0);
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

// `count` items of `size` bytes, all zero, that may hold a pointer.
void *hornet_alloc_zeroed(int64_t count, int64_t size) {
    if (!hornet_fits(count, size)) {
        hornet__out_of_memory(count, size);
    }
    return hornet__heap_alloc(count * size, 1, 1);
}

// `count` items of `size` bytes, all zero, that hold no pointer.
void *hornet_alloc_zeroed_atomic(int64_t count, int64_t size) {
    if (!hornet_fits(count, size)) {
        hornet__out_of_memory(count, size);
    }
    return hornet__heap_alloc(count * size, 0, 1);
}

void *hornet__alloc_items(int64_t count, int64_t size, int holds_pointers) {
    if (!hornet_fits(count, size)) {
        hornet__out_of_memory(count, size);
    }
    return holds_pointers ? hornet_alloc(count * size) : hornet_alloc_atomic(count * size);
}

static void *hornet_realloc(void *ptr, int64_t size) {
    void *grown = size < 0 ? NULL : realloc(ptr, size > 0 ? (size_t)size : 1);
    if (grown == NULL) {
        hornet__out_of_memory(size, -1);
    }
    return grown;
}

// A new object, with what fits of the old one's contents (which stays where it was, until no longer
// pointed to). Unless the old one's slot already has room.
void *hornet__realloc_atomic(void *ptr, int64_t size) {
    struct hornet_heap_object old;
    if (ptr == NULL || !hornet__heap_find((uintptr_t)ptr, &old)) {
        return hornet__heap_alloc(size, 0, 0);
    }
    if (size >= 0 && size <= old.size && old.start == ptr) {
        return ptr;
    }
    char *grown = hornet__heap_alloc(size, 0, 0);
    memcpy(grown, ptr, (size_t)(old.size < size ? old.size : size));
    return grown;
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
