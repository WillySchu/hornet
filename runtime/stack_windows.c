// A stack overflow on Windows (see stack.c).
#include "internal.h"

#ifdef _WIN32
// The program's function that was running at `fault`: the registers there are unwound, a frame at a
// time, until the address is in one. (A frame with no unwind table of its own, like a stack probe's,
// has only its return address on the stack.)
static const char *hornet_running_function(const CONTEXT *fault) {
    static CONTEXT at;
    at = *fault;
    for (int frame = 0; frame < 64 && at.Rip != 0; frame++) {
        // Past the first, the address is where a call returns to: the call is just before it.
        const char *name = hornet__function_at((const char *)at.Rip - (frame == 0 ? 0 : 1));
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
        WriteFile(standard_error, HORNET_STACK_OVERFLOW, sizeof HORNET_STACK_OVERFLOW - 1, &written, NULL);
        hornet_fault = info->ContextRecord;
        hornet_on_handler_stack(hornet_find_function);
        size_t length = hornet__stack_overflow_rest(rest, sizeof rest, hornet_function_found);
        WriteFile(standard_error, rest, (DWORD)length, &written, NULL);
        TerminateProcess(GetCurrentProcess(), 3);  // abort()'s exit code
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

void hornet__catch_stack_overflow(void) {
    ULONG kept_back = 64 * 1024;
    SetThreadStackGuarantee(&kept_back);
    AddVectoredExceptionHandler(1, hornet_on_exception);
}
#endif
