// The start and end of a program: what runs before main, its arguments, and exit.
#include "internal.h"

#ifdef _WIN32
// Windows opens streams and files in text mode, which rewrites "\n" as "\r\n" on the way out and
// back on the way in. Hornet's I/O is bytes, so everything is binary.
static void hornet_binary_standard_streams(void) {
    _setmode(0, _O_BINARY);
    _setmode(1, _O_BINARY);
    _setmode(2, _O_BINARY);
}
#endif

// Before main. (This file is linked into every program whole: nothing calls into it, so taken from
// the runtime's archive, as the others are, it would be left out.)
__attribute__((constructor)) static void hornet_start(void) {
#ifdef _WIN32
    hornet_binary_standard_streams();
#endif
    hornet__catch_stack_overflow();
}

// argv[index]; Hornet lacks pointer indexing. Used by stdlib/os.ht.
char *hornet_argv_get(char **argv, int64_t index) {
    return argv[index];
}

// Flush stdio, then exit.
void hornet_exit(int64_t code) {
    fflush(NULL);
    exit((int)code);
}
