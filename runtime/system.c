// What the system says: why a call failed, the environment, and the process.
#include "internal.h"

static char hornet_system_error_text[200] = "";

const char *hornet_system_error(void) {
    return hornet_system_error_text;
}

#ifdef _WIN32
void hornet__system_failed_with(DWORD code) {
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

void hornet__system_failed(void) {
    hornet__system_failed_with(GetLastError());
}
#else
void hornet__system_failed_with(int code) {
    snprintf(hornet_system_error_text, sizeof hornet_system_error_text, "%s", strerror(code));
}

void hornet__system_failed(void) {
    hornet__system_failed_with(errno);
}
#endif

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
