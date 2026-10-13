// A stack overflow on POSIX systems, macOS and Linux (see stack.c).
#include "internal.h"

#ifndef _WIN32
#include <signal.h>
#include <sys/resource.h>
#ifdef __APPLE__
#include <pthread.h>
#else
#include <unwind.h>
#endif

// Elsewhere it is a signal, handled on a stack of its own (the program's has no room left).
static char hornet_signal_stack[64 * 1024];
// The stack's range, as far as it is known: its lowest address is an estimate on Linux. (Kept as
// numbers: as pointers, reaching past the stack or comparing with a fault's address would be
// undefined behaviour, which an optimizing compiler may act on.)
static uintptr_t hornet_stack_low, hornet_stack_high;
// A frame larger than the protected memory below the stack can step past it, so a fault this far
// below the stack still counts.
#define HORNET_BELOW_STACK ((uintptr_t)64 * 1024 * 1024)

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
    const char *name = hornet__function_at((const char *)pc);
    if (name == NULL && link != 0) {
        name = hornet__function_at((const char *)HORNET_CODE_ADDRESS(link) - 1);  // (the call is just before it)
    }
    for (int depth = 0; name == NULL && depth < 256; depth++) {
        // What a frame pointer points at is two words in the stack, above the fault. Anything else
        // is no frame pointer, and nothing is read through it.
        if (frame % sizeof(uintptr_t) != 0 || frame <= fault || frame < hornet_stack_low
            || frame + 2 * sizeof(uintptr_t) > hornet_stack_high) {
            break;
        }
        const uintptr_t *held = (const uintptr_t *)frame;
        name = hornet__function_at((const char *)HORNET_CODE_ADDRESS(held[1]) - 1);
        if (held[0] <= frame) {
            break;  // the caller's is always further up
        }
        frame = held[0];
    }
    return name;
}

static const char *hornet_running_function(void *context, uintptr_t fault) {
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
    return hornet_function_up_the_frames(pc, link, frame, fault);
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
    const char *name = hornet__function_at(at_instruction ? address : address - 1);
    if (name == NULL) {
        return _URC_NO_REASON;
    }
    *(const char **)found = name;
    return _URC_END_OF_STACK;
}

// The program's function that was running when the signal came: found by unwinding from here,
// through the signal's own frame and any C the program had called.
static const char *hornet_running_function(void *context, uintptr_t fault) {
    (void)context;
    (void)fault;
    const char *name = NULL;
    _Unwind_Backtrace(hornet_note_function, &name);
    return name;
}
#endif

static void hornet_on_fault(int sig, siginfo_t *info, void *context) {
    uintptr_t address = (uintptr_t)info->si_addr;
    // Memory in the stack's own range doesn't fault unless the stack has run out.
    if (address < hornet_stack_high && address + HORNET_BELOW_STACK >= hornet_stack_low) {
        static char rest[256];
        ssize_t written = write(2, HORNET_STACK_OVERFLOW, sizeof HORNET_STACK_OVERFLOW - 1);
        size_t length = hornet__stack_overflow_rest(rest, sizeof rest, hornet_running_function(context, address));
        written = write(2, rest, length);
        (void)written;
        abort();
    }
    signal(sig, SIG_DFL);  // not the stack: back to the instruction, to fault again and crash
}

void hornet__catch_stack_overflow(void) {
#ifdef __APPLE__
    hornet_stack_high = (uintptr_t)pthread_get_stackaddr_np(pthread_self());
    hornet_stack_low = hornet_stack_high - pthread_get_stacksize_np(pthread_self());
#else
    // The main thread's stack grows down from about here, to the size the system allows it.
    struct rlimit limit;
    char here;
    if (getrlimit(RLIMIT_STACK, &limit) != 0 || limit.rlim_cur == RLIM_INFINITY) {
        return;  // no end to look for
    }
    uintptr_t at = (uintptr_t)&here;
    hornet_stack_high = at + 4096;
    hornet_stack_low = limit.rlim_cur < at ? at - limit.rlim_cur : 0;
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
