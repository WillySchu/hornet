// Slices: bytes(s), growing, and copying elements.
#include "internal.h"

// bytes(s): *out = a new []byte copy of the len bytes at ptr. Hornet's calling convention
// passes the result slot first, then the str as (ptr, len).

void hornet_bytes(struct hornet_slice *out, const char *ptr, int64_t len) {
    out->ptr = hornet_alloc(len);
    if (len > 0) {
        memcpy(out->ptr, ptr, (size_t)len);
    }
    out->len = len;
    out->cap = len;
}


// Room for new_cap elements of element_width bytes, with the first `len` copied in.
void *hornet_slice_grow(const void *old_ptr, int64_t len, int64_t new_cap, int64_t element_width) {
    void *new_ptr = hornet__alloc_items(new_cap, element_width);
    memcpy(new_ptr, old_ptr, (size_t)len * (size_t)element_width);
    return new_ptr;
}

// `count` elements of element_width bytes from `src` to element `at` of `dst`, as if read before any
// is written: the two may overlap (append(xs[:1], xs...)).
void hornet_copy_items(void *dst, int64_t at, const void *src, int64_t count, int64_t element_width) {
    if (count > 0) {
        memmove((char *)dst + at * element_width, src, (size_t)count * (size_t)element_width);
    }
}
