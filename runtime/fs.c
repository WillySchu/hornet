// The file system: directories, and moving and removing files, for stdlib/os.ht.
#include "internal.h"

// A new directory in the temporary directory, its path in memory the caller keeps; NULL on failure.
char *hornet_make_temp_dir(void) {
#ifdef _WIN32
    char dir[MAX_PATH + 1];
    if (GetTempPathA(sizeof dir, dir) == 0) {
        hornet__system_failed();
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
    hornet__system_failed();
    return NULL;
#else
    const char *dir = getenv("TMPDIR");
    char *path = hornet_alloc(4096);
    snprintf(path, 4096, "%s/hornet-XXXXXX", dir != NULL && *dir != '\0' ? dir : "/tmp");
    if (mkdtemp(path) == NULL) {
        hornet__system_failed();
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
                hornet__system_failed_with(ERROR_DIRECTORY);
#else
                hornet__system_failed_with(ENOTDIR);
#endif
            } else {
                hornet__system_failed();
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
        hornet__system_failed();
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
        hornet__system_failed();
        return -1;
    }
    return 0;
}
