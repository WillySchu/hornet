// Running another program, on POSIX systems (see process_windows.c).
#include "internal.h"

#ifndef _WIN32
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

// Run the program `args` names (found on PATH unless it has a directory), with `count` arguments
// one after another in `args`, each ending in a zero byte (the first is the program); wait for it.
// Its input is empty, and what it writes to its standard output and error goes to files, read from
// the start through *out_fd and *err_fd: the caller reads and closes them. The exit code, with
// *signal 0, or the signal that ended it (POSIX) in *signal; -1 if it couldn't be started.
int64_t hornet_process_run(const char *args, int64_t count, int64_t *out_fd, int64_t *err_fd, int64_t *signal) {
    *signal = 0;
    *out_fd = *err_fd = -1;
    char **argv = hornet__scratch_alloc((count + 1) * (int64_t)sizeof(char *));
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
        hornet__system_failed();
        goto failed;
    }
    fcntl(report[0], F_SETFD, FD_CLOEXEC);
    fcntl(report[1], F_SETFD, FD_CLOEXEC);
    fflush(NULL);  // (what is buffered would otherwise be written twice, once by each process)
    pid_t pid = fork();
    if (pid < 0) {
        hornet__system_failed();
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
        hornet__system_failed_with(why);
        goto failed;
    }
    close(report[0]);
    close(in);
    lseek(out, 0, SEEK_SET);
    lseek(err, 0, SEEK_SET);
    *out_fd = out;
    *err_fd = err;
    hornet__scratch_free(argv);
    if (WIFSIGNALED(status)) {
        *signal = WTERMSIG(status);
        return 0;
    }
    return WEXITSTATUS(status);
failed:
    hornet__scratch_free(argv);
    if (out >= 0) close(out);
    if (err >= 0) close(err);
    if (in >= 0) close(in);
    if (report[0] >= 0) close(report[0]);
    if (report[1] >= 0) close(report[1]);
    return -1;
}
#endif
