// Running another program, on Windows (see process_posix.c).
#include "internal.h"

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

// Run the program `args` names (found on PATH unless it has a directory), with `count` arguments
// one after another in `args`, each ending in a zero byte (the first is the program); wait for it.
// Its input is empty, and what it writes to its standard output and error goes to files, read from
// the start through *out_fd and *err_fd: the caller reads and closes them. The exit code, with
// *signal 0, or the signal that ended it (POSIX) in *signal; -1 if it couldn't be started.
int64_t hornet_process_run(const char *args, int64_t count, int64_t *out_fd, int64_t *err_fd, int64_t *signal) {
    *signal = 0;
    *out_fd = *err_fd = -1;
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
        hornet__system_failed();
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
        hornet__system_failed();
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
}
#endif
