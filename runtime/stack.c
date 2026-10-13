// ---- A stack overflow is a panic.
//
// Running off the end of the stack is a fault like any bad memory access. What catches it tells
// the two apart, says `panic: stack overflow in 'f'`, and ends the program as a panic does; any
// other fault is left to crash as it would have. (There is no source position: a fault doesn't say
// where in the program it happened. The function is the program's own that was running: the one
// the fault was in, or the nearest that called the C the fault was in. It goes unnamed if that
// can't be found.)
//
// What is common to the systems: which of the program's functions an address is in, and what is
// said. Each system catches it its own way (stack_posix.c, stack_windows.c).
#include "internal.h"

// The compiler writes a table of the functions it compiled: where each starts and ends, and its
// name (backend/common/function_table.py). A program's table takes the place of this empty one,
// which is here for a runtime linked with no program.

typedef struct {
    const char *start, *end, *name;
} hornet_function;

__attribute__((weak)) const hornet_function hornet_functions[] = {{NULL, NULL, NULL}};

const char *hornet__function_at(const char *address) {
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

size_t hornet__stack_overflow_rest(char *text, size_t room, const char *function) {
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
