/*
 * malloc_trace.c
 *
 * A small LD_PRELOAD (Linux) / DYLD_INSERT_LIBRARIES (macOS) shim that
 * intercepts every malloc() and free() call made by a running process,
 * without needing to modify or rebuild that process at all.
 *
 * This exists specifically to verify Hornet's string memory management
 * (see codegen.py's STRINGS section, and _gen_free_if_fresh_concat) --
 * the compiler's own generated code calls the real libc malloc/free
 * directly, in hand-written assembly, so there's no Python-level hook
 * to inspect. This shim sits *underneath* that: it wraps the actual
 * malloc/free symbols the dynamic linker resolves, transparently, so
 * any compiled Hornet binary can be run under it with no changes to
 * that binary at all.
 *
 * A NOTE ON THE ~4KB ALLOCATION YOU'LL SEE IF THE PROGRAM CALLS print()
 * -------------------------------------------------------------------------
 * The first time a traced program calls print() with an int or bool
 * argument (printf) or a str argument (puts), you'll see one malloc()
 * around 4096 bytes that Hornet's own codegen never asked for and that
 * never gets free()'d. That's glibc allocating its internal stdio
 * output buffer the first time anything is actually written to stdout
 * -- completely standard behavior for *any* C program using buffered
 * I/O, not a Hornet bug or a real leak. It'll show up as part of
 * "still-leaked" in the summary below; don't mistake it for something
 * this compiler's codegen is responsible for. Confirmed by running the
 * same program with and without a print() call: the 4096-byte
 * allocation only appears in the version that prints something.
 *
 * ONE MORE THING WORTH KNOWING: WHAT HAPPENS ON A REAL CRASH
 * ---------------------------------------------------------------
 * This shim still calls the *real* free() after logging an invalid
 * one, rather than skipping it -- on purpose, so a traced run behaves
 * identically to an untraced one and isn't hiding or changing the
 * actual bug's real-world consequences. That means if the invalid free
 * is bad enough that glibc's own allocator detects the corruption (a
 * double-free, for instance), the process can still abort right after
 * this shim's own diagnostic prints -- and since that's an abnormal
 * termination, the SUMMARY block at the very end (via a destructor)
 * may never get printed at all. That's expected, not a shim bug: the
 * "*** INVALID FREE ***" line itself, printed the moment it happens
 * and naming the exact pointer, is the actionable diagnostic either
 * way, whether or not the program lives long enough afterward to also
 * print a final summary.
 *
 * WHAT IT CHECKS
 * ----------------
 * For every free(ptr) call, it checks whether `ptr` is currently a
 * live, previously-malloc'd pointer this shim has actually seen handed
 * out. If not, that's a real bug -- a double-free, or a free() on some
 * address that was never malloc'd in the first place (e.g. a string
 * literal living in static .data, which must never be freed). This is
 * the check that actually matters: a program can produce completely
 * correct *output* while still corrupting the heap or double-freeing,
 * since neither of those necessarily crashes immediately or predictably.
 *
 * It also reports, at process exit, how many allocations were never
 * freed. Note that a nonzero leak count is *expected* right now, not
 * necessarily a bug -- Hornet's current freeing logic (see
 * _gen_free_if_fresh_concat) only reclaims a concatenation's operand
 * when that operand is itself a fresh, unnamed intermediate result. A
 * concatenation stored in a named variable is never freed, on purpose,
 * since deciding whether that's actually safe needs real escape
 * analysis this compiler doesn't do. So: "invalid frees" should always
 * be 0. "still-leaked" being nonzero is normal and expected for any
 * program that stores concatenation results in variables -- what
 * matters there is whether the *count* matches what you'd predict by
 * hand for a given program, not whether it's zero.
 *
 * BUILDING
 * ---------
 * Linux:
 *     gcc -shared -fPIC -o malloc_trace.so malloc_trace.c -ldl
 *
 * macOS:
 *     gcc -shared -fPIC -o malloc_trace.dylib malloc_trace.c
 *     (dlsym/dlopen are already part of libSystem on macOS; there's no
 *     separate libdl to link against the way Linux needs -ldl.)
 *
 * USING IT DIRECTLY
 * -------------------
 * Linux:
 *     LD_PRELOAD=./malloc_trace.so ./your_compiled_hornet_binary
 *
 * macOS:
 *     DYLD_INSERT_LIBRARIES=./malloc_trace.dylib \
 *     DYLD_FORCE_FLAT_NAMESPACE=1 \
 *     ./your_compiled_hornet_binary
 *
 *     (DYLD_FORCE_FLAT_NAMESPACE=1 is required on macOS -- without it,
 *     each library gets its own symbol namespace, and this shim's
 *     malloc/free would never actually get called in place of the
 *     real ones.)
 *
 * Easier: use trace_memory.sh in this same directory, which builds
 * this shim if needed, runs the full Hornet pipeline (lex -> parse ->
 * analyze -> codegen -> assemble -> link) on a .ht file, and runs the
 * result under this shim automatically, on either platform.
 *
 * A NOTE OF HONESTY ABOUT THE macOS PATH
 * -----------------------------------------
 * This was developed and verified entirely on Linux (this environment
 * doesn't have macOS available to test against). The macOS-specific
 * pieces above (DYLD_INSERT_LIBRARIES, DYLD_FORCE_FLAT_NAMESPACE, the
 * lack of -ldl) reflect how macOS's dynamic linker is documented to
 * behave, but haven't actually been run and confirmed the way the
 * Linux path has. If it doesn't work as described, that's the first
 * place to look.
 *
 * LIMITATIONS
 * ------------
 * - Only intercepts malloc/free, since that's the entire allocation
 *   surface Hornet's codegen currently uses (see gen_string_concat_into
 *   -- it only ever calls malloc, never calloc/realloc). If Hornet ever
 *   starts calling those too, extend this shim the same way: get the
 *   real symbol via dlsym, wrap it, track it.
 * - The live-allocation table is a fixed-size array (see MAX_TRACKED
 *   below) for simplicity, not a real hash table or a dynamically
 *   growable one. It's sized generously for the kind of small test
 *   programs this is meant for; if you ever exceed it, this shim prints
 *   a clear warning rather than silently corrupting anything or
 *   crashing, so you'll know to raise the limit rather than trusting a
 *   wrong report.
 * - Not thread-safe. Hornet doesn't have threads, so this was never a
 *   concern here, but if that ever changes, this shim would need real
 *   locking around the tracking table.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>

static void *(*real_malloc)(size_t) = NULL;
static void (*real_free)(void *) = NULL;

#define MAX_TRACKED 100000

typedef struct {
    void *ptr;
    size_t size;
} live_alloc_t;

static live_alloc_t live[MAX_TRACKED];
static int live_count = 0;
static int overflowed = 0;

static long malloc_calls = 0;
static long free_calls = 0;
static long invalid_frees = 0;

static int initialized = 0;

static void init(void) {
    if (initialized) return;
    real_malloc = dlsym(RTLD_NEXT, "malloc");
    real_free = dlsym(RTLD_NEXT, "free");
    initialized = 1;
}

void *malloc(size_t size) {
    init();
    void *p = real_malloc(size);
    malloc_calls++;

    if (live_count < MAX_TRACKED) {
        live[live_count].ptr = p;
        live[live_count].size = size;
        live_count++;
    } else if (!overflowed) {
        fprintf(stderr,
            "[malloc_trace] *** WARNING: exceeded MAX_TRACKED (%d) live "
            "allocations -- tracking is now INCOMPLETE. Raise MAX_TRACKED "
            "in malloc_trace.c and rebuild if you need to trace a program "
            "this large. ***\n", MAX_TRACKED);
        overflowed = 1;
    }

    fprintf(stderr, "[malloc_trace] malloc(%zu) = %p\n", size, p);
    return p;
}

void free(void *ptr) {
    init();
    if (ptr == NULL) {
        real_free(ptr);
        return;
    }

    free_calls++;
    int found = -1;
    for (int i = 0; i < live_count; i++) {
        if (live[i].ptr == ptr) {
            found = i;
            break;
        }
    }

    if (found < 0) {
        fprintf(stderr,
            "[malloc_trace] *** INVALID FREE: free(%p) -- this pointer is "
            "not a currently-live malloc'd allocation this shim tracked. "
            "This is either a double-free, or a free() on something that "
            "was never malloc'd (e.g. a static string literal). ***\n",
            ptr);
        invalid_frees++;
    } else {
        fprintf(stderr, "[malloc_trace] free(%p) -- OK, was a live %zu-byte allocation\n",
                ptr, live[found].size);
        /* swap-remove: order doesn't matter for this table */
        live[found] = live[live_count - 1];
        live_count--;
    }

    real_free(ptr);
}

__attribute__((destructor))
static void report(void) {
    size_t leaked_bytes = 0;
    for (int i = 0; i < live_count; i++) {
        leaked_bytes += live[i].size;
    }

    fprintf(stderr, "\n");
    fprintf(stderr, "[malloc_trace] ================ SUMMARY ================\n");
    fprintf(stderr, "[malloc_trace] malloc calls:     %ld\n", malloc_calls);
    fprintf(stderr, "[malloc_trace] free calls:       %ld\n", free_calls);
    fprintf(stderr, "[malloc_trace] still-leaked:     %d allocations, %zu bytes\n",
            live_count, leaked_bytes);
    fprintf(stderr, "[malloc_trace] invalid frees:    %ld%s\n", invalid_frees,
            invalid_frees > 0 ? "   <-- REAL BUG, this should always be 0" : "");
    fprintf(stderr, "[malloc_trace] ===========================================\n");
}
