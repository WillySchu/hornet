// Safepoints: where the compiled code checks whether the runtime asks the running task to stop (the
// checks are placed by optimize/safepoints.py, for now only with HORNET_SAFEPOINTS set). A check reads
// hornet_preempt and, if it is set, calls hornet__safepoint_stub, which keeps every register: so the
// check moves nothing in the code around it, and a collection there finds what the registers held on
// the stack.
//
// Nothing stops a task yet. HORNET_SAFEPOINT_STRESS=N keeps the flag set, so that every check calls
// in, and collects at every Nth: to show a register the stub doesn't keep, or one the collector
// doesn't find.
#include "internal.h"
#include <stdatomic.h>

// (A line of its own: read by every check, written seldom.)
_Alignas(64) _Atomic uint8_t hornet_preempt;
static int64_t hornet_stress_every, hornet_stress_count;

__attribute__((constructor)) static void hornet_safepoint_settings(void) {
    const char *stress = getenv("HORNET_SAFEPOINT_STRESS");
    hornet_stress_every = stress != NULL ? strtoll(stress, NULL, 10) : 0;
    if (hornet_stress_every > 0) {
        atomic_store_explicit(&hornet_preempt, 1, memory_order_relaxed);
    }
}

void hornet__safepoint(void);
void hornet__safepoint(void) {
    if (hornet_stress_every <= 0) {
        atomic_store_explicit(&hornet_preempt, 0, memory_order_relaxed);
        return;
    }
    if (++hornet_stress_count % hornet_stress_every == 0) {
        hornet__gc_collect();
    }
}

#ifdef __APPLE__
#define HORNET_SYMBOL(name) "_" #name
#else
#define HORNET_SYMBOL(name) #name
#endif
// What each step does to the stack, for an unwinder walking up from a fault in what the stub calls
// (stack_posix.c's and stack_windows.c's, naming the function a stack overflow was in): DWARF's call
// frame information elsewhere, Windows' own unwind codes there.
#ifdef _WIN32
#define HORNET_CFI(text) ""
#define HORNET_SEH(text) "    " text "\n"
#else
#define HORNET_CFI(text) "    " text "\n"
#define HORNET_SEH(text) ""
#endif

// hornet__safepoint_stub: hornet__safepoint, keeping the registers C may change. (The others C keeps
// itself. The flags needn't be kept: the check has just set them, and nothing reads them after it.)
#if defined(__x86_64__)
// Called with %rsp 16-byte aligned, as at any call in the compiled code: nine pushes realign it, and
// 32 bytes more are Windows' shadow space (unneeded elsewhere, and harmless).
__asm__(
    ".text\n"
    ".globl " HORNET_SYMBOL(hornet__safepoint_stub) "\n"
    HORNET_SEH(".seh_proc " HORNET_SYMBOL(hornet__safepoint_stub))
    HORNET_SYMBOL(hornet__safepoint_stub) ":\n"
    HORNET_CFI(".cfi_startproc")
    "    pushq %rax\n" HORNET_CFI(".cfi_adjust_cfa_offset 8") HORNET_SEH(".seh_pushreg %rax")
    "    pushq %rcx\n" HORNET_CFI(".cfi_adjust_cfa_offset 8") HORNET_SEH(".seh_pushreg %rcx")
    "    pushq %rdx\n" HORNET_CFI(".cfi_adjust_cfa_offset 8") HORNET_SEH(".seh_pushreg %rdx")
    "    pushq %rsi\n" HORNET_CFI(".cfi_adjust_cfa_offset 8") HORNET_SEH(".seh_pushreg %rsi")
    "    pushq %rdi\n" HORNET_CFI(".cfi_adjust_cfa_offset 8") HORNET_SEH(".seh_pushreg %rdi")
    "    pushq %r8\n" HORNET_CFI(".cfi_adjust_cfa_offset 8") HORNET_SEH(".seh_pushreg %r8")
    "    pushq %r9\n" HORNET_CFI(".cfi_adjust_cfa_offset 8") HORNET_SEH(".seh_pushreg %r9")
    "    pushq %r10\n" HORNET_CFI(".cfi_adjust_cfa_offset 8") HORNET_SEH(".seh_pushreg %r10")
    "    pushq %r11\n" HORNET_CFI(".cfi_adjust_cfa_offset 8") HORNET_SEH(".seh_pushreg %r11")
    "    subq $32, %rsp\n" HORNET_CFI(".cfi_adjust_cfa_offset 32") HORNET_SEH(".seh_stackalloc 32")
    HORNET_SEH(".seh_endprologue")
    "    call " HORNET_SYMBOL(hornet__safepoint) "\n"
    "    addq $32, %rsp\n" HORNET_CFI(".cfi_adjust_cfa_offset -32")
    "    popq %r11\n" HORNET_CFI(".cfi_adjust_cfa_offset -8")
    "    popq %r10\n" HORNET_CFI(".cfi_adjust_cfa_offset -8")
    "    popq %r9\n" HORNET_CFI(".cfi_adjust_cfa_offset -8")
    "    popq %r8\n" HORNET_CFI(".cfi_adjust_cfa_offset -8")
    "    popq %rdi\n" HORNET_CFI(".cfi_adjust_cfa_offset -8")
    "    popq %rsi\n" HORNET_CFI(".cfi_adjust_cfa_offset -8")
    "    popq %rdx\n" HORNET_CFI(".cfi_adjust_cfa_offset -8")
    "    popq %rcx\n" HORNET_CFI(".cfi_adjust_cfa_offset -8")
    "    popq %rax\n" HORNET_CFI(".cfi_adjust_cfa_offset -8")
    "    ret\n"
    HORNET_CFI(".cfi_endproc")
    HORNET_SEH(".seh_endproc"));
#elif defined(__aarch64__)
// x0-x17 and the link register (x18 is the platform's on macOS, and C doesn't change it).
__asm__(
    ".text\n"
    ".globl " HORNET_SYMBOL(hornet__safepoint_stub) "\n"
    ".p2align 2\n"
    HORNET_SYMBOL(hornet__safepoint_stub) ":\n"
    HORNET_CFI(".cfi_startproc")
    "    sub sp, sp, #160\n"
    HORNET_CFI(".cfi_def_cfa_offset 160")
    "    stp x0, x1, [sp, #0]\n    stp x2, x3, [sp, #16]\n    stp x4, x5, [sp, #32]\n"
    "    stp x6, x7, [sp, #48]\n    stp x8, x9, [sp, #64]\n    stp x10, x11, [sp, #80]\n"
    "    stp x12, x13, [sp, #96]\n    stp x14, x15, [sp, #112]\n    stp x16, x17, [sp, #128]\n"
    "    str x30, [sp, #144]\n"
    HORNET_CFI(".cfi_offset x30, -16")
    "    bl " HORNET_SYMBOL(hornet__safepoint) "\n"
    "    ldp x0, x1, [sp, #0]\n    ldp x2, x3, [sp, #16]\n    ldp x4, x5, [sp, #32]\n"
    "    ldp x6, x7, [sp, #48]\n    ldp x8, x9, [sp, #64]\n    ldp x10, x11, [sp, #80]\n"
    "    ldp x12, x13, [sp, #96]\n    ldp x14, x15, [sp, #112]\n    ldp x16, x17, [sp, #128]\n"
    "    ldr x30, [sp, #144]\n"
    "    add sp, sp, #160\n"
    HORNET_CFI(".cfi_def_cfa_offset 0")
    "    ret\n"
    HORNET_CFI(".cfi_endproc"));
#endif
