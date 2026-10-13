// Files and standard streams, for stdlib/os.ht.
#include "internal.h"

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
            line = hornet__realloc(line, (int64_t)cap);
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
