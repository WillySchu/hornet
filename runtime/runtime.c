// Hornet runtime: print, panic, bytes(), slice growth, dict hash tables, and what stdlib/os.ht calls.

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
// Windows opens streams and files in text mode, which rewrites "\n" as "\r\n" on the way out and
// back on the way in. Hornet's I/O is bytes, so everything is binary.
#define HORNET_O_BINARY _O_BINARY
__attribute__((constructor)) static void hornet_binary_standard_streams(void) {
    _setmode(0, _O_BINARY);
    _setmode(1, _O_BINARY);
    _setmode(2, _O_BINARY);
}
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

// ---- Which of the program's functions an address is in.
//
// The compiler writes a table of the functions it compiled: where each starts and ends, and its
// name (backend/common/function_table.py). A program's table takes the place of this empty one,
// which is here for a runtime linked with no program.

typedef struct {
    const char *start, *end, *name;
} hornet_function;

__attribute__((weak)) const hornet_function hornet_functions[] = {{NULL, NULL, NULL}};

static const char *hornet_function_at(const char *address) {
    // (Through a pointer whose target the compiler can't see: the table is as long as the
    // program's, not as this one.)
    const hornet_function *volatile table = hornet_functions;
    for (const hornet_function *f = table; f->start != NULL; f++) {
        if (address >= f->start && address < f->end) {
            return f->name;
        }
    }
    return NULL;
}

// ---- A stack overflow is a panic.
//
// Running off the end of the stack is a fault like any bad memory access. What catches it tells
// the two apart, says `panic: stack overflow in 'f'`, and ends the program as a panic does; any
// other fault is left to crash as it would have. (There is no source position: a fault doesn't say
// where in the program it happened. The function is the program's own that was running: the one
// the fault was in, or the nearest that called the C the fault was in. It goes unnamed if that
// can't be found.)

// The message is written in two parts: this, before the function is looked for, so that it is said
// whatever becomes of the looking; then the rest.
static const char hornet_stack_overflow[] = "panic: stack overflow";

// The rest: " in 'function'\n", or only the newline if there is no function to name. Its length.
// (Nothing here allocates.)
static size_t hornet_stack_overflow_rest(char *text, size_t room, const char *function) {
    size_t length = 0;
    if (function != NULL) {
        static const char in[] = " in '";
        memcpy(text, in, sizeof in - 1);
        length = sizeof in - 1;
        while (*function != '\0' && length < room - 2) {
            text[length++] = *function++;
        }
        text[length++] = '\'';
    }
    text[length++] = '\n';
    return length;
}

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// The program's function that was running at `fault`: the registers there are unwound, a frame at a
// time, until the address is in one. (A frame with no unwind table of its own, like a stack probe's,
// has only its return address on the stack.)
static const char *hornet_running_function(const CONTEXT *fault) {
    static CONTEXT at;
    at = *fault;
    for (int frame = 0; frame < 64 && at.Rip != 0; frame++) {
        // Past the first, the address is where a call returns to: the call is just before it.
        const char *name = hornet_function_at((const char *)at.Rip - (frame == 0 ? 0 : 1));
        if (name != NULL) {
            return name;
        }
        DWORD64 image;
        RUNTIME_FUNCTION *unwind = RtlLookupFunctionEntry(at.Rip, &image, NULL);
        if (unwind == NULL) {
            DWORD64 returns_to;
            if (!ReadProcessMemory(GetCurrentProcess(), (void *)at.Rsp, &returns_to, sizeof returns_to, NULL)) {
                return NULL;
            }
            at.Rip = returns_to;
            at.Rsp += 8;
        } else {
            void *handler_data;
            DWORD64 establisher;
            RtlVirtualUnwind(UNW_FLAG_NHANDLER, image, at.Rip, unwind, &at, &handler_data, &establisher, NULL);
        }
    }
    return NULL;
}

// Windows raises an exception of its own for it. The handler runs on the stack that overflowed, in
// the space SetThreadStackGuarantee keeps back: enough to say so and end the program, but the
// looking for the function is done on a stack of its own, since how much is kept back isn't sure.
static char hornet_handler_stack[64 * 1024];
static const CONTEXT *hornet_fault;
static const char *hornet_function_found;

static void hornet_find_function(void) {
    hornet_function_found = hornet_running_function(hornet_fault);
}

// Call `work` with the stack pointer in hornet_handler_stack (leaving the 32 bytes a callee may use
// above its return address, 16-byte aligned), and come back to this stack.
static void hornet_on_handler_stack(void (*work)(void)) {
    char *top = hornet_handler_stack + sizeof hornet_handler_stack - 64;
    __asm__ volatile("mov %%rsp, %%rbx\n\t"
                     "mov %[top], %%rsp\n\t"
                     "call *%[work]\n\t"
                     "mov %%rbx, %%rsp"
                     :
                     : [top] "r"(top), [work] "r"(work)
                     : "rbx", "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "memory", "cc");
}

static LONG WINAPI hornet_on_exception(EXCEPTION_POINTERS *info) {
    if (info->ExceptionRecord->ExceptionCode == EXCEPTION_STACK_OVERFLOW) {
        static char rest[256];
        HANDLE standard_error = GetStdHandle(STD_ERROR_HANDLE);
        DWORD written;
        WriteFile(standard_error, hornet_stack_overflow, sizeof hornet_stack_overflow - 1, &written, NULL);
        hornet_fault = info->ContextRecord;
        hornet_on_handler_stack(hornet_find_function);
        size_t length = hornet_stack_overflow_rest(rest, sizeof rest, hornet_function_found);
        WriteFile(standard_error, rest, (DWORD)length, &written, NULL);
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
#else
#include <unwind.h>
#endif

// Elsewhere it is a signal, handled on a stack of its own (the program's has no room left).
static char hornet_signal_stack[64 * 1024];
// The stack's range, as far as it is known: its lowest address is an estimate on Linux.
static char *hornet_stack_low, *hornet_stack_high;
// A frame larger than the protected memory below the stack can step past it, so a fault this far
// below the stack still counts.
#define HORNET_BELOW_STACK ((intptr_t)64 * 1024 * 1024)

#ifdef __APPLE__
// macOS's unwinder ends its walk at the first frame that has no unwind table, without reporting
// it, and the program's functions have none. So there the frames are followed by hand, from the
// registers at the fault. Every function keeps a frame pointer, the program's and the system's
// alike: at it are the caller's frame pointer, then the address the function returns to.

// An address of code as a frame holds it. (On arm64 a system function may have signed the one it
// returns to, in the bits above an address's 47.)
#ifdef __aarch64__
#define HORNET_CODE_ADDRESS(held) ((held) & (uintptr_t)0x00007fffffffffff)
#else
#define HORNET_CODE_ADDRESS(held) (held)
#endif

// The program's function that was running at a fault at address `fault`: the one `pc` is in; or
// the one that had just called (`link`, the register a call leaves its return address in, or 0);
// or, going up the frames from `frame`, the nearest that called the C the fault was in.
static const char *hornet_function_up_the_frames(uintptr_t pc, uintptr_t link, uintptr_t frame, uintptr_t fault) {
    const char *name = hornet_function_at((const char *)pc);
    if (name == NULL && link != 0) {
        name = hornet_function_at((const char *)HORNET_CODE_ADDRESS(link) - 1);  // (the call is just before it)
    }
    for (int depth = 0; name == NULL && depth < 256; depth++) {
        // What a frame pointer points at is two words in the stack, above the fault. Anything else
        // is no frame pointer, and nothing is read through it.
        if (frame % sizeof(uintptr_t) != 0 || frame <= fault || frame < (uintptr_t)hornet_stack_low
            || frame + 2 * sizeof(uintptr_t) > (uintptr_t)hornet_stack_high) {
            break;
        }
        const uintptr_t *held = (const uintptr_t *)frame;
        name = hornet_function_at((const char *)HORNET_CODE_ADDRESS(held[1]) - 1);
        if (held[0] <= frame) {
            break;  // the caller's is always further up
        }
        frame = held[0];
    }
    return name;
}

static const char *hornet_running_function(void *context, const char *fault) {
    const ucontext_t *at = (const ucontext_t *)context;
#if defined(__x86_64__)
    uintptr_t pc = (uintptr_t)at->uc_mcontext->__ss.__rip, frame = (uintptr_t)at->uc_mcontext->__ss.__rbp, link = 0;
#elif defined(__darwin_arm_thread_state64_get_pc)
    uintptr_t pc = (uintptr_t)__darwin_arm_thread_state64_get_pc(at->uc_mcontext->__ss);
    uintptr_t frame = (uintptr_t)__darwin_arm_thread_state64_get_fp(at->uc_mcontext->__ss);
    uintptr_t link = (uintptr_t)__darwin_arm_thread_state64_get_lr(at->uc_mcontext->__ss);
#else
    uintptr_t pc = (uintptr_t)at->uc_mcontext->__ss.__pc, frame = (uintptr_t)at->uc_mcontext->__ss.__fp;
    uintptr_t link = (uintptr_t)at->uc_mcontext->__ss.__lr;
#endif
    return hornet_function_up_the_frames(pc, link, frame, (uintptr_t)fault);
}
#else
// One frame of the walk back from the signal's handler: if it is in one of the program's
// functions, that is the one, and the walk ends. (GCC's unwinder reports a frame that has no
// unwind table, as the program's have none, before ending there.)
static _Unwind_Reason_Code hornet_note_function(struct _Unwind_Context *frame, void *found) {
    // The frame the signal interrupted is at the instruction that faulted; any other's address is
    // where a call returns to, and the call is just before it.
    int at_instruction = 0;
    const char *address = (const char *)_Unwind_GetIPInfo(frame, &at_instruction);
    const char *name = hornet_function_at(at_instruction ? address : address - 1);
    if (name == NULL) {
        return _URC_NO_REASON;
    }
    *(const char **)found = name;
    return _URC_END_OF_STACK;
}

// The program's function that was running when the signal came: found by unwinding from here,
// through the signal's own frame and any C the program had called.
static const char *hornet_running_function(void *context, const char *fault) {
    (void)context;
    (void)fault;
    const char *name = NULL;
    _Unwind_Backtrace(hornet_note_function, &name);
    return name;
}
#endif

static void hornet_on_fault(int sig, siginfo_t *info, void *context) {
    char *address = (char *)info->si_addr;
    // Memory in the stack's own range doesn't fault unless the stack has run out.
    if (address >= hornet_stack_low - HORNET_BELOW_STACK && address < hornet_stack_high) {
        static char rest[256];
        ssize_t written = write(2, hornet_stack_overflow, sizeof hornet_stack_overflow - 1);
        size_t length = hornet_stack_overflow_rest(rest, sizeof rest, hornet_running_function(context, address));
        written = write(2, rest, length);
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

// ---- Memory.
//
// Every allocation, the compiled code's and the runtime's own, comes through here, so that being
// refused memory is a panic and not a write through a null pointer. (A system that grants memory it
// doesn't have, as Linux does by default, kills the program when the memory is used: there is no
// refusal to catch then.) Nothing is freed yet but the runtime's own scratch buffers.

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

// `count` items of `size` bytes, uninitialized.
static void *hornet_alloc_items(int64_t count, int64_t size) {
    if (!hornet_fits(count, size)) {
        hornet_out_of_memory(count, size);
    }
    return hornet_alloc(count * size);
}

// realloc, for the runtime's own growing buffers.
static void *hornet_realloc(void *ptr, int64_t size) {
    void *grown = size < 0 ? NULL : realloc(ptr, size > 0 ? (size_t)size : 1);
    if (grown == NULL) {
        hornet_out_of_memory(size, -1);
    }
    return grown;
}

// Ensure `additional` spare bytes, growing by doubling.
static void hornet_buf_ensure(struct hornet_buf *buf, int64_t additional) {
    int64_t needed = buf->len + additional;
    if (needed <= buf->cap) {
        return;
    }
    int64_t doubled_or_quartered = buf->cap < 256 ? buf->cap * 2 : buf->cap + buf->cap / 4;
    int64_t new_cap = needed > doubled_or_quartered ? needed : doubled_or_quartered;
    buf->ptr = hornet_realloc(buf->ptr, new_cap);
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

// print(x) entry point.
// bytes(s): *out = a new []byte copy of the len bytes at ptr. Hornet's calling convention
// passes the result slot first, then the str as (ptr, len).
struct hornet_slice {
    void *ptr;
    int64_t len;
    int64_t cap;
};

void hornet_bytes(struct hornet_slice *out, const char *ptr, int64_t len) {
    out->ptr = hornet_alloc(len);
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
    char *line = hornet_alloc((int64_t)cap);
    int c;
    while ((c = getc(stdin)) != EOF && c != '\n') {
        if (n == cap) {
            cap *= 2;
            line = hornet_realloc(line, (int64_t)cap);
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

// ---- TCP connections, for stdlib/net.ht.
//
// A socket is an int on POSIX and a SOCKET on Windows, where it isn't a file descriptor either:
// reading, writing, and closing one have calls of their own. Hornet sees an int64 everywhere.

#ifdef _WIN32
typedef SOCKET hornet_socket;
#define HORNET_NO_SOCKET INVALID_SOCKET
#define hornet_socket_close closesocket
#else
typedef int hornet_socket;
#define HORNET_NO_SOCKET (-1)
#define hornet_socket_close close
#endif

// Sending to a connection the other side has closed raises SIGPIPE, which would end the program
// without a word; an error from the call is wanted instead. Linux asks per send, macOS per socket
// (SO_NOSIGPIPE, below), and Windows has no such signal.
#ifdef MSG_NOSIGNAL
#define HORNET_SEND_FLAGS MSG_NOSIGNAL
#else
#define HORNET_SEND_FLAGS 0
#endif

static char hornet_net_error_text[160] = "";

// Why the most recent hornet_tcp_* call failed.
const char *hornet_net_error(void) {
    return hornet_net_error_text;
}

static void hornet_net_set_error(const char *text) {
    snprintf(hornet_net_error_text, sizeof hornet_net_error_text, "%s", text);
}

// Record the system's reason for the socket call that just failed.
static void hornet_net_failed(void) {
#ifdef _WIN32
    int code = WSAGetLastError();
    if (code == WSAETIMEDOUT) {
        hornet_net_set_error("timed out");
        return;
    }
    char text[128];
    DWORD n = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, (DWORD)code, 0,
                             text, sizeof text, NULL);
    while (n > 0 && (text[n - 1] == '\n' || text[n - 1] == '\r' || text[n - 1] == '.' || text[n - 1] == ' ')) {
        text[--n] = '\0';
    }
    if (n == 0) {
        snprintf(text, sizeof text, "Winsock error %d", code);
    }
    hornet_net_set_error(text);
#else
    // A socket with a timeout reports one as "try again".
    hornet_net_set_error(errno == EAGAIN || errno == EWOULDBLOCK ? "timed out" : strerror(errno));
#endif
}

// Connect to `host` (a name or an address) on `port`, trying each address the name has. A positive
// `timeout_ms` bounds each later read and write (and, on Linux, the connecting itself). The socket,
// or -1.
int64_t hornet_tcp_connect(const char *host, int64_t port, int64_t timeout_ms) {
#ifdef _WIN32
    static int started = 0;
    WSADATA data;
    if (!started && WSAStartup(MAKEWORD(2, 2), &data) != 0) {
        hornet_net_set_error("Winsock is not available");
        return -1;
    }
    started = 1;
#endif
    char service[16];
    snprintf(service, sizeof service, "%d", (int)port);
    struct addrinfo hints, *found = NULL;
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    int rc = getaddrinfo(host, service, &hints, &found);
    if (rc != 0) {
#ifdef _WIN32
        WSASetLastError(rc);
        hornet_net_failed();
#else
        hornet_net_set_error(rc == EAI_SYSTEM ? strerror(errno) : gai_strerror(rc));
#endif
        return -1;
    }
    hornet_socket s = HORNET_NO_SOCKET;
    hornet_net_set_error("no address");
    for (struct addrinfo *a = found; a != NULL; a = a->ai_next) {
        s = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
        if (s == HORNET_NO_SOCKET) {
            hornet_net_failed();
            continue;
        }
        if (timeout_ms > 0) {
#ifdef _WIN32
            DWORD timeout = (DWORD)timeout_ms;
#else
            struct timeval timeout = {(time_t)(timeout_ms / 1000), (suseconds_t)(timeout_ms % 1000 * 1000)};
#endif
            setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, (const char *)&timeout, sizeof timeout);
            setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, (const char *)&timeout, sizeof timeout);
        }
#ifdef SO_NOSIGPIPE
        int on = 1;
        setsockopt(s, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof on);
#endif
        if (connect(s, a->ai_addr, (socklen_t)a->ai_addrlen) == 0) {
            break;
        }
#ifndef _WIN32
        if (errno == EINPROGRESS) {  // what a connect that outlasts SO_SNDTIMEO reports
            errno = EAGAIN;
        }
#endif
        hornet_net_failed();
        hornet_socket_close(s);
        s = HORNET_NO_SOCKET;
    }
    freeaddrinfo(found);
    return s == HORNET_NO_SOCKET ? -1 : (int64_t)s;
}

// Up to `len` bytes from a connection: how many arrived, 0 once the other side has closed, -1 on
// an error.
int64_t hornet_tcp_read(int64_t sock, char *ptr, int64_t len) {
    for (;;) {
        int64_t n = recv((hornet_socket)sock, ptr, len > 1 << 30 ? 1 << 30 : (int)len, 0);
        if (n >= 0) {
            return n;
        }
#ifndef _WIN32
        if (errno == EINTR) {
            continue;
        }
#endif
        hornet_net_failed();
        return -1;
    }
}

// Send all `len` bytes to a connection; -1 on an error.
int64_t hornet_tcp_write(int64_t sock, const char *ptr, int64_t len) {
    int64_t done = 0;
    while (done < len) {
        int64_t left = len - done;
        int64_t n = send((hornet_socket)sock, ptr + done, left > 1 << 30 ? 1 << 30 : (int)left, HORNET_SEND_FLAGS);
        if (n < 0) {
#ifndef _WIN32
            if (errno == EINTR) {
                continue;
            }
#endif
            hornet_net_failed();
            return -1;
        }
        done += n;
    }
    return done;
}

int64_t hornet_tcp_close(int64_t sock) {
    return hornet_socket_close((hornet_socket)sock);
}

// Flush stdio, then exit.
void hornet_exit(int64_t code) {
    fflush(NULL);
    exit((int)code);
}

// ---- Other programs, and the file system, for stdlib/process.ht and stdlib/os.ht.
//
// The calls differ between POSIX and Windows, and so do their errors: errno there, GetLastError()
// here. hornet_system_error says why the most recent of these calls failed, either way.

static char hornet_system_error_text[200] = "";

const char *hornet_system_error(void) {
    return hornet_system_error_text;
}

#ifdef _WIN32
static void hornet_system_failed_with(DWORD code) {
    char *text = hornet_system_error_text;
    DWORD n = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, NULL, code, 0, text,
                             sizeof hornet_system_error_text, NULL);
    while (n > 0 && (text[n - 1] == '\n' || text[n - 1] == '\r' || text[n - 1] == '.' || text[n - 1] == ' ')) {
        text[--n] = '\0';
    }
    if (n == 0) {
        snprintf(text, sizeof hornet_system_error_text, "Windows error %lu", (unsigned long)code);
    }
}

static void hornet_system_failed(void) {
    hornet_system_failed_with(GetLastError());
}
#else
static void hornet_system_failed_with(int code) {
    snprintf(hornet_system_error_text, sizeof hornet_system_error_text, "%s", strerror(code));
}

static void hornet_system_failed(void) {
    hornet_system_failed_with(errno);
}
#endif

#ifdef _WIN32
// `arg` as one argument of a command line, as Windows programs' C runtimes split one: quoted if it
// has a space, a tab, or a quote (or is empty), a quote escaped by a backslash, and the backslashes
// before a quote, or before the closing one, doubled.
static void hornet_quote_argument(struct hornet_buf *line, const char *arg) {
    if (*arg != '\0' && strpbrk(arg, " \t\"") == NULL) {
        hornet_buf_append_cstr(line, arg);
        return;
    }
    hornet_buf_append_byte(line, '"');
    for (const char *p = arg;; p++) {
        int64_t backslashes = 0;
        while (*p == '\\') {
            backslashes++;
            p++;
        }
        int64_t doubled = *p == '"' || *p == '\0' ? backslashes * 2 : backslashes;
        for (int64_t i = 0; i < doubled; i++) {
            hornet_buf_append_byte(line, '\\');
        }
        if (*p == '\0') {
            break;
        }
        if (*p == '"') {
            hornet_buf_append_byte(line, '\\');
        }
        hornet_buf_append_byte(line, *p);
    }
    hornet_buf_append_byte(line, '"');
}

// A new file in the temporary directory, removed when the last handle to it is closed; one the
// child inherits.
static HANDLE hornet_temporary_file(void) {
    char dir[MAX_PATH + 1], path[MAX_PATH + 1];
    SECURITY_ATTRIBUTES inherited = {sizeof inherited, NULL, TRUE};
    if (GetTempPathA(sizeof dir, dir) == 0 || GetTempFileNameA(dir, "hor", 0, path) == 0) {
        return INVALID_HANDLE_VALUE;
    }
    return CreateFileA(path, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                       &inherited, CREATE_ALWAYS, FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE, NULL);
}
#else
// A new file in $TMPDIR (or /tmp), already unlinked: it goes when its descriptors do. -1 on failure.
static int hornet_temporary_file(void) {
    const char *dir = getenv("TMPDIR");
    char path[4096];
    snprintf(path, sizeof path, "%s/hornet-XXXXXX", dir != NULL && *dir != '\0' ? dir : "/tmp");
    int fd = mkstemp(path);
    if (fd >= 0) {
        unlink(path);
        fcntl(fd, F_SETFD, FD_CLOEXEC);
    }
    return fd;
}
#endif

// Run the program `args` names (found on PATH unless it has a directory), with `count` arguments
// one after another in `args`, each ending in a zero byte (the first is the program); wait for it.
// Its input is empty, and what it writes to its standard output and error goes to files, read from
// the start through *out_fd and *err_fd: the caller reads and closes them. The exit code, with
// *signal 0, or the signal that ended it (POSIX) in *signal; -1 if it couldn't be started.
int64_t hornet_process_run(const char *args, int64_t count, int64_t *out_fd, int64_t *err_fd, int64_t *signal) {
    *signal = 0;
    *out_fd = *err_fd = -1;
#ifdef _WIN32
    struct hornet_buf line = {0};
    const char *arg = args;
    for (int64_t i = 0; i < count; i++) {
        if (i > 0) {
            hornet_buf_append_byte(&line, ' ');
        }
        hornet_quote_argument(&line, arg);
        arg += strlen(arg) + 1;
    }
    hornet_buf_append_byte(&line, '\0');
    SECURITY_ATTRIBUTES inherited = {sizeof inherited, NULL, TRUE};
    HANDLE out = hornet_temporary_file(), err = hornet_temporary_file();
    HANDLE in = CreateFileA("NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &inherited, OPEN_EXISTING, 0,
                            NULL);
    if (out == INVALID_HANDLE_VALUE || err == INVALID_HANDLE_VALUE || in == INVALID_HANDLE_VALUE) {
        hornet_system_failed();
        if (out != INVALID_HANDLE_VALUE) CloseHandle(out);
        if (err != INVALID_HANDLE_VALUE) CloseHandle(err);
        if (in != INVALID_HANDLE_VALUE) CloseHandle(in);
        return -1;
    }
    STARTUPINFOA startup = {0};
    startup.cb = sizeof startup;
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = in;
    startup.hStdOutput = out;
    startup.hStdError = err;
    PROCESS_INFORMATION process;
    BOOL started = CreateProcessA(NULL, line.ptr, NULL, NULL, TRUE, 0, NULL, NULL, &startup, &process);
    if (!started) {
        hornet_system_failed();
    }
    CloseHandle(in);
    if (!started) {
        CloseHandle(out);
        CloseHandle(err);
        return -1;
    }
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(process.hProcess, &code);
    CloseHandle(process.hProcess);
    CloseHandle(process.hThread);
    SetFilePointer(out, 0, NULL, FILE_BEGIN);
    SetFilePointer(err, 0, NULL, FILE_BEGIN);
    *out_fd = _open_osfhandle((intptr_t)out, _O_RDONLY | _O_BINARY);
    *err_fd = _open_osfhandle((intptr_t)err, _O_RDONLY | _O_BINARY);
    return (int64_t)code;
#else
    char **argv = hornet_alloc((count + 1) * (int64_t)sizeof(char *));
    const char *arg = args;
    for (int64_t i = 0; i < count; i++) {
        argv[i] = (char *)arg;
        arg += strlen(arg) + 1;
    }
    argv[count] = NULL;
    int out = hornet_temporary_file(), err = hornet_temporary_file(), in = open("/dev/null", O_RDONLY | O_CLOEXEC);
    // The child reports an exec that failed down this pipe; one that worked closes it (O_CLOEXEC).
    int report[2] = {-1, -1};
    if (out < 0 || err < 0 || in < 0 || pipe(report) != 0) {
        hornet_system_failed();
        goto failed;
    }
    fcntl(report[0], F_SETFD, FD_CLOEXEC);
    fcntl(report[1], F_SETFD, FD_CLOEXEC);
    fflush(NULL);  // (what is buffered would otherwise be written twice, once by each process)
    pid_t pid = fork();
    if (pid < 0) {
        hornet_system_failed();
        goto failed;
    }
    if (pid == 0) {
        if (dup2(in, 0) >= 0 && dup2(out, 1) >= 0 && dup2(err, 2) >= 0) {
            execvp(argv[0], argv);
        }
        int why = errno;
        ssize_t written = write(report[1], &why, sizeof why);
        (void)written;
        _exit(127);
    }
    close(report[1]);
    report[1] = -1;
    int why;
    ssize_t got;
    do {
        got = read(report[0], &why, sizeof why);
    } while (got < 0 && errno == EINTR);
    int status;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    if (got == (ssize_t)sizeof why) {
        hornet_system_failed_with(why);
        goto failed;
    }
    close(report[0]);
    close(in);
    lseek(out, 0, SEEK_SET);
    lseek(err, 0, SEEK_SET);
    *out_fd = out;
    *err_fd = err;
    if (WIFSIGNALED(status)) {
        *signal = WTERMSIG(status);
        return 0;
    }
    return WEXITSTATUS(status);
failed:
    if (out >= 0) close(out);
    if (err >= 0) close(err);
    if (in >= 0) close(in);
    if (report[0] >= 0) close(report[0]);
    if (report[1] >= 0) close(report[1]);
    return -1;
#endif
}

// getenv: NULL when `name` isn't set.
const char *hornet_get_env(const char *name) {
    return getenv(name);
}

int64_t hornet_process_id(void) {
#ifdef _WIN32
    return (int64_t)GetCurrentProcessId();
#else
    return (int64_t)getpid();
#endif
}

// A new directory in the temporary directory, its path in memory the caller keeps; NULL on failure.
char *hornet_make_temp_dir(void) {
#ifdef _WIN32
    char dir[MAX_PATH + 1];
    if (GetTempPathA(sizeof dir, dir) == 0) {
        hornet_system_failed();
        return NULL;
    }
    char *path = hornet_alloc(MAX_PATH + 64);
    for (unsigned attempt = 0; attempt < 1000; attempt++) {
        snprintf(path, MAX_PATH + 64, "%shornet-%lu-%lu-%u", dir, (unsigned long)GetCurrentProcessId(),
                 (unsigned long)GetTickCount(), attempt);
        if (CreateDirectoryA(path, NULL)) {
            return path;
        }
        if (GetLastError() != ERROR_ALREADY_EXISTS) {
            break;
        }
    }
    hornet_system_failed();
    return NULL;
#else
    const char *dir = getenv("TMPDIR");
    char *path = hornet_alloc(4096);
    snprintf(path, 4096, "%s/hornet-XXXXXX", dir != NULL && *dir != '\0' ? dir : "/tmp");
    if (mkdtemp(path) == NULL) {
        hornet_system_failed();
        return NULL;
    }
    return path;
#endif
}

static int hornet_is_directory(const char *path) {
#ifdef _WIN32
    DWORD attributes = GetFileAttributesA(path);
    return attributes != INVALID_FILE_ATTRIBUTES && (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
#else
    struct stat info;
    return stat(path, &info) == 0 && S_ISDIR(info.st_mode);
#endif
}

// Make the directory `path`, and those it is in, as needed: 0, or -1 (with what is made kept).
int64_t hornet_make_dirs(const char *path) {
    size_t n = strlen(path);
    char *partial = hornet_alloc((int64_t)n + 1);
    memcpy(partial, path, n + 1);
    for (size_t i = 1; i <= n; i++) {
        if (i < n && partial[i] != '/' && partial[i] != '\\') {
            continue;
        }
        char kept = partial[i];
        partial[i] = '\0';
        int made;
#ifdef _WIN32
        made = (i == 2 && partial[1] == ':') || CreateDirectoryA(partial, NULL)  // (`C:` is no directory to make)
               || GetLastError() == ERROR_ALREADY_EXISTS;
#else
        made = mkdir(partial, 0777) == 0 || errno == EEXIST;
#endif
        if (!made || !hornet_is_directory(partial)) {
            if (made) {  // (it exists, but is something else)
#ifdef _WIN32
                hornet_system_failed_with(ERROR_DIRECTORY);
#else
                hornet_system_failed_with(ENOTDIR);
#endif
            } else {
                hornet_system_failed();
            }
            return -1;
        }
        partial[i] = kept;
    }
    return 0;
}

int64_t hornet_path_exists(const char *path) {
#ifdef _WIN32
    return GetFileAttributesA(path) != INVALID_FILE_ATTRIBUTES;
#else
    struct stat info;
    return stat(path, &info) == 0;
#endif
}

// Rename `from` to `to`, replacing what `to` was: 0, or -1.
int64_t hornet_rename(const char *from, const char *to) {
#ifdef _WIN32
    if (!MoveFileExA(from, to, MOVEFILE_REPLACE_EXISTING | MOVEFILE_COPY_ALLOWED)) {
#else
    if (rename(from, to) != 0) {
#endif
        hornet_system_failed();
        return -1;
    }
    return 0;
}

// Remove the file, or the empty directory, `path`: 0, or -1.
int64_t hornet_remove(const char *path) {
#ifdef _WIN32
    if (!(hornet_is_directory(path) ? RemoveDirectoryA(path) : DeleteFileA(path))) {
#else
    if (remove(path) != 0) {
#endif
        hornet_system_failed();
        return -1;
    }
    return 0;
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
    hornet_stringify(value_addr, type_desc, 0, buf);
}

void hornet_print(void *value_addr, const unsigned char *type_desc) {
    struct hornet_buf buf;
    buf.cap = 16;
    buf.ptr = hornet_alloc(buf.cap);
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

// Room for new_cap elements of element_width bytes, with the first `len` copied in.
void *hornet_slice_grow(const void *old_ptr, int64_t len, int64_t new_cap, int64_t element_width) {
    void *new_ptr = hornet_alloc_items(new_cap, element_width);
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
    void *new_buckets = hornet_alloc_zeroed(new_capacity, bucket_stride);
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
    void *new_buckets = hornet_alloc_zeroed(new_capacity, bucket_stride);
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
    buf.ptr = hornet_alloc(buf.cap);
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
