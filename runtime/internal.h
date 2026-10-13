// Declarations the runtime's files share, and what each includes first. Not for the compiled code:
// what it calls is each file's own `hornet_` functions. A name here that is a function's starts with
// `hornet__`, which a program's own functions, and the C functions it declares `extern`, don't.
#ifndef HORNET_INTERNAL_H
#define HORNET_INTERNAL_H

// What the system's headers are asked for beyond standard C: POSIX with its extensions (signal
// stacks), and on macOS the system's own (a thread's stack).
#ifndef _WIN32
#define _XOPEN_SOURCE 700
#define _DARWIN_C_SOURCE
#endif

#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef _WIN32
#include <winsock2.h>  // (before windows.h, which would otherwise bring in the older winsock.h)
#include <ws2tcpip.h>
#include <io.h>
// Files are opened binary: Hornet's I/O is bytes (see start.c).
#define HORNET_O_BINARY _O_BINARY
#else
#include <netdb.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#define HORNET_O_BINARY 0
#endif

#include "hornet_typedesc_tags.h"

// ---- The start of a program (start.c).

// Ready to catch a stack overflow (stack_posix.c, stack_windows.c).
void hornet__catch_stack_overflow(void);

// ---- A stack overflow (stack.c).

// The first part of what is said: before the function is looked for, so that it is said whatever
// becomes of the looking.
#define HORNET_STACK_OVERFLOW "panic: stack overflow"

// The program's function that `address` is in, or NULL.
const char *hornet__function_at(const char *address);

// The rest of what is said: " in 'function'\n", or only the newline if there is no function to name.
// Its length. (Nothing here allocates.)
size_t hornet__stack_overflow_rest(char *text, size_t room, const char *function);

// ---- Memory (memory.c).
//
// Memory a program's values are kept in comes in two kinds: what may hold a pointer, and what is
// known to hold none (`_atomic`: bytes, numbers). The runtime's own scratch memory, which no value
// is ever kept in, is a third, and is freed when it is done with.

void *hornet_alloc(int64_t size);
void *hornet_alloc_atomic(int64_t size);
void *hornet_alloc_zeroed(int64_t count, int64_t size);
void *hornet_alloc_zeroed_atomic(int64_t count, int64_t size);

// `count` items of `size` bytes, uninitialized; as hornet_alloc if they may hold a pointer, else as
// hornet_alloc_atomic.
void *hornet__alloc_items(int64_t count, int64_t size, int holds_pointers);

// Grow memory from hornet_alloc_atomic to `size` bytes, keeping what it held: a str being built.
void *hornet__realloc_atomic(void *ptr, int64_t size);

// `panic: out of memory (allocating N bytes)`, or `N x M bytes` when `size` isn't negative. No
// allocation: this is no time for it.
void hornet__out_of_memory(int64_t count, int64_t size);

// ---- The heap (heap.c), where the program's values are.

// `size` bytes (not negative). All zero if it may hold pointers or `zeroed`, else uninitialized.
void *hornet__heap_alloc(int64_t size, int holds_pointers, int zeroed);

// The object an address is in.
struct hornet_heap_object {
    char *start;
    int64_t size;  // a small object's slot's, which may be more than was asked for
    int holds_pointers;
};

// Whether `address` is in an object on the heap (anywhere in it), and if so which.
int hornet__heap_find(uintptr_t address, struct hornet_heap_object *found);

// Free what nothing reaches from the stack and the registers. (The heap runs it as it needs to.)
void hornet__gc_collect(void);

struct hornet_heap_stats {
    int64_t collections;
    int64_t pages;        // the heap's 64 KB pages now
    int64_t live_bytes;   // in objects (their slots) that the last collection kept
    int64_t freed_bytes;  // by every collection so far
};
void hornet__heap_stats(struct hornet_heap_stats *stats);

// The runtime's own scratch memory.
void *hornet__scratch_alloc(int64_t size);
void *hornet__scratch_realloc(void *ptr, int64_t size);
void hornet__scratch_free(void *ptr);

// Where a sum's payload is: after its 4-byte tag, at 8 (typesys.py's SUM_PAYLOAD_OFFSET).
#define HORNET_SUM_PAYLOAD_OFFSET 8

// A dict's header, which every copy of the dict points to (typesys.py's DICT_HEADER_SIZE and its
// fields). The compiled code writes the bucket layout when it makes the dict: the runtime goes by it.
struct hornet_dict {
    void *buckets;
    int64_t count, tombstones, capacity;
    int64_t key_offset, value_offset;  // where in a bucket, after its state byte, the key and value are
    int64_t stride;  // how far apart buckets are
    int64_t holds_pointers;  // whether a bucket may hold one (in its key or value): how to allocate them
};
_Static_assert(sizeof(struct hornet_dict) == 64 && offsetof(struct hornet_dict, key_offset) == 32,
               "a dict's header is as typesys.py has it");

// Bucket state byte. Tombstones keep probe chains intact after deletion.
#define HORNET_DICT_BUCKET_EMPTY 0
#define HORNET_DICT_BUCKET_OCCUPIED 1
#define HORNET_DICT_BUCKET_TOMBSTONE 2

// Growable byte buffer for print.
struct hornet_buf {
    char *ptr;
    int64_t len;
    int64_t cap;
    int64_t scratch;  // 0: the bytes become a str (from hornet_alloc_atomic); 1: the runtime's scratch memory
};

// Ensure `additional` spare bytes, growing by doubling (as the buffer's kind of memory).
void hornet__buf_ensure(struct hornet_buf *buf, int64_t additional);

static inline void hornet_buf_append_bytes(struct hornet_buf *buf, const char *data, int64_t count) {
    hornet__buf_ensure(buf, count);
    memcpy(buf->ptr + buf->len, data, (size_t)count);
    buf->len += count;
}

static inline void hornet_buf_append_byte(struct hornet_buf *buf, char byte_value) {
    hornet_buf_append_bytes(buf, &byte_value, 1);
}

static inline void hornet_buf_append_cstr(struct hornet_buf *buf, const char *s) {
    hornet_buf_append_bytes(buf, s, (int64_t)strlen(s));
}

// A slice's header, as the compiled code keeps one.
struct hornet_slice {
    void *ptr;
    int64_t len;
    int64_t cap;
};

// Reads of what a value holds, wherever it is. (Hornet lays values out as C would, aligned; these
// don't depend on it.)
static inline int32_t read_i32(const void *addr) {
    int32_t value;
    memcpy(&value, addr, sizeof(value));
    return value;
}

static inline int64_t read_i64(const void *addr) {
    int64_t value;
    memcpy(&value, addr, sizeof(value));
    return value;
}

static inline void *read_ptr(const void *addr) {
    void *value;
    memcpy(&value, addr, sizeof(value));
    return value;
}

// `index`-th 8-byte word of a type descriptor.
static inline uint64_t read_desc_word(const unsigned char *desc, int64_t index) {
    uint64_t value;
    memcpy(&value, desc + index * 8, sizeof(value));
    return value;
}

// ---- Formatting (format.c).

// Format the value at value_addr per type_desc (type_desc[0] is the kind tag).
void hornet__stringify(void *value_addr, const unsigned char *type_desc, int quote_strings, struct hornet_buf *buf);

void hornet_print(void *value_addr, const unsigned char *type_desc);

int64_t hornet_write_fd(int64_t fd, const char *ptr, int64_t len);

// ---- The system's errors (system.c).
//
// The calls differ between POSIX and Windows, and so do their errors: errno there, GetLastError()
// here. hornet_system_error says why the most recent call that noted one failed, either way.

#ifdef _WIN32
void hornet__system_failed_with(DWORD code);
#else
void hornet__system_failed_with(int code);
#endif

// The most recent call's own error.
void hornet__system_failed(void);

#endif
