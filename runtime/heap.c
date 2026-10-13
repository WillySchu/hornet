// The heap: where the program's values live, in memory this file manages itself rather than one
// malloc call per value, so that any address can be traced to the object it is in, and objects
// nothing points to can be found and freed: a collector.
//
// Memory comes in 64 KB pages, each aligned to its size, so that an address's page is the address
// with its low 16 bits cleared. A small object goes in a page of objects all of one size class (8
// bytes to 32 KB, about a quarter apart), where its start follows from any address inside it by
// arithmetic; a larger one has a run of pages to itself. A table finds the record of any page from
// its address. Pages of objects that hold pointers and of objects that hold none are kept apart.
//
// The collector is conservative: it can't tell a pointer from a number that looks like one, so
// takes every word that is the address of (anywhere in) an object as keeping it, starting from the
// stack and the registers (there are no other roots: Hornet has no globals) and following the words
// of each object found that may hold pointers. What none of them reach is freed. It runs when an
// allocation finds that as much has been allocated since it last ran as was alive then (8 MB at
// least), and before memory is refused.
//
// HORNET_GC_STRESS=1 makes it run before every allocation (=N: every Nth), and HORNET_GC_POISON=1
// fills what it frees with 0xdb: both to make a missed pointer show.
//
// A pointer just past the end of an object (xs[len(xs):]) doesn't keep it: the slice it is in has
// no capacity left, so nothing can be read or written through it.
#include "internal.h"
#ifdef _WIN32
#include <malloc.h>
#elif defined(__APPLE__)
#include <pthread.h>
#endif

#define HORNET_PAGE_SHIFT 16
#define HORNET_PAGE_SIZE ((uintptr_t)1 << HORNET_PAGE_SHIFT)

// The size classes: 8 to 64 by 8, then four to each doubling, to 32 KB. Each a multiple of 8, the
// most any value's alignment needs.
static const uint32_t hornet_classes[] = {
    8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768,
    896, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096, 5120, 6144, 7168, 8192, 10240, 12288,
    14336, 16384, 20480, 24576, 28672, 32768,
};
#define HORNET_CLASS_COUNT ((int)(sizeof hornet_classes / sizeof hornet_classes[0]))
#define HORNET_LARGEST_SMALL 32768

// A page's record, kept apart from the page. A large object's pages all share one.
struct hornet_page {
    uintptr_t start;           // where its objects start: the page, or the large object
    uint32_t object_size;      // its size class; 0 for a large object
    uint32_t slots;            // a small page's: how many objects it holds
    uint32_t fresh;            // of those, how many have been handed out at some time (from the start)
    uint32_t live;             // how many are allocated now
    int64_t large_size;        // a large object's size, as asked for
    int holds_pointers;        // whether its objects may hold a pointer
    int large_marked;          // a large object's mark
    void *free;                // a small page's freed slots: each's first word is the next
    struct hornet_page *next;  // the next page of its class with room, while it is on that list
    uint64_t *allocated;       // a small page's: which slots hold an object
    uint64_t *marked;          // ... and which the collector has found reachable
    uint64_t bits[];           // (where those two are)
};

// The table, from a page's number (its address >> 16) to its record: the number's high 16 bits pick
// a block of the table, made when first needed, and its low 16 bits the record in it. (Addresses go
// up to 48 bits.)
static struct hornet_page **hornet_page_table[(size_t)1 << 16];

// The lowest address the heap has used, and one past the highest: most words that aren't pointers
// to it are told apart by these alone.
static uintptr_t hornet_heap_low = UINTPTR_MAX, hornet_heap_high = 0;

// Every page record, for the sweep (a large object's once).
static struct hornet_page **hornet_pages;
static size_t hornet_page_count, hornet_page_room;

// The pages of each class with room, for objects that may hold pointers [1] and that hold none [0];
// the first is the one being filled.
static struct hornet_page *hornet_with_room[2][HORNET_CLASS_COUNT];

// Allocated since the collector last ran, and how much makes it run again.
static int64_t hornet_allocated_since, hornet_collect_at = 8 << 20;
static struct hornet_heap_stats hornet_stats;

// HORNET_GC_STRESS and HORNET_GC_POISON, read at the first allocation.
static int hornet_settings_read, hornet_poison;
static int64_t hornet_stress_every, hornet_stress_count;

// The collector's own scratch: objects found but not yet looked through.
static struct hornet_heap_object *hornet_mark_stack;
static size_t hornet_mark_count, hornet_mark_room;

static void *hornet_scratch_grow(void *array, size_t *room, size_t item, int64_t size_for_message) {
    size_t new_room = *room == 0 ? 256 : *room * 2;
    void *grown = realloc(array, new_room * item);
    if (grown == NULL) {
        hornet__out_of_memory(size_for_message, -1);
    }
    *room = new_room;
    return grown;
}

// Pages from the C library, aligned to their size (`size` a multiple of HORNET_PAGE_SIZE); NULL if
// there is no memory.
static void *hornet_get_pages(size_t size) {
#ifdef _WIN32
    return _aligned_malloc(size, HORNET_PAGE_SIZE);
#else
    void *memory;  // (posix_memalign rather than C11's aligned_alloc: older macOS lacks the latter)
    return posix_memalign(&memory, HORNET_PAGE_SIZE, size) == 0 ? memory : NULL;
#endif
}

static void hornet_free_pages(void *memory) {
#ifdef _WIN32
    _aligned_free(memory);
#else
    free(memory);
#endif
}

// Have the table give `page` (NULL: no page) for each of the `count` pages from `address`.
static void hornet_record_pages(uintptr_t address, size_t count, struct hornet_page *page,
                                int64_t size_for_message) {
    for (size_t i = 0; i < count; i++) {
        uintptr_t number = (address >> HORNET_PAGE_SHIFT) + i;
        struct hornet_page **block = hornet_page_table[number >> 16];
        if (block == NULL) {
            block = calloc((size_t)1 << 16, sizeof *block);
            if (block == NULL) {
                hornet__out_of_memory(size_for_message, -1);
            }
            hornet_page_table[number >> 16] = block;
        }
        block[number & 0xFFFF] = page;
    }
    if (page != NULL && address < hornet_heap_low) {
        hornet_heap_low = address;
    }
    if (page != NULL && address + count * HORNET_PAGE_SIZE > hornet_heap_high) {
        hornet_heap_high = address + count * HORNET_PAGE_SIZE;
    }
}

// A record for `pages` pages from `start`, with room for the marks of `slots` objects.
static struct hornet_page *hornet_new_record(uintptr_t start, size_t pages, uint32_t slots,
                                             int64_t size_for_message) {
    size_t words = (slots + 63) / 64;
    struct hornet_page *page = calloc(1, sizeof *page + 2 * words * sizeof(uint64_t));
    if (page == NULL) {
        hornet__out_of_memory(size_for_message, -1);
    }
    page->start = start;
    page->slots = slots;
    page->allocated = page->bits;
    page->marked = page->bits + words;
    if (hornet_page_count == hornet_page_room) {
        hornet_pages = hornet_scratch_grow(hornet_pages, &hornet_page_room, sizeof *hornet_pages, size_for_message);
    }
    hornet_pages[hornet_page_count++] = page;
    hornet_record_pages(start, pages, page, size_for_message);
    hornet_stats.pages += pages;
    return page;
}

// The size class for `size` bytes (at most HORNET_LARGEST_SMALL): the smallest that holds it.
static int hornet_class_of(int64_t size) {
    if (size <= 64) {
        return size <= 8 ? 0 : (int)((size + 7) / 8) - 1;
    }
    int low = 8, high = HORNET_CLASS_COUNT - 1;  // hornet_classes[high] >= size, hornet_classes[low - 1] < size
    while (low < high) {
        int middle = (low + high) / 2;
        if (hornet_classes[middle] >= size) {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    return low;
}

static int hornet_bit(const uint64_t *bits, uint32_t index) {
    return (int)((bits[index / 64] >> (index % 64)) & 1);
}

static void hornet_set_bit(uint64_t *bits, uint32_t index, int value) {
    if (value) {
        bits[index / 64] |= (uint64_t)1 << (index % 64);
    } else {
        bits[index / 64] &= ~((uint64_t)1 << (index % 64));
    }
}

static void hornet_read_settings(void) {
    hornet_settings_read = 1;
    const char *stress = getenv("HORNET_GC_STRESS");
    hornet_stress_every = stress != NULL ? strtoll(stress, NULL, 10) : 0;
    const char *poison = getenv("HORNET_GC_POISON");
    hornet_poison = poison != NULL && strcmp(poison, "0") != 0;
}

// Take a page of a class's objects from the C library: NULL if there is no memory.
static struct hornet_page *hornet_new_small_page(int class, int holds_pointers, int64_t size_for_message) {
    void *memory = hornet_get_pages(HORNET_PAGE_SIZE);
    if (memory == NULL) {
        return NULL;
    }
    struct hornet_page *page =
        hornet_new_record((uintptr_t)memory, 1, (uint32_t)(HORNET_PAGE_SIZE / hornet_classes[class]), size_for_message);
    page->object_size = hornet_classes[class];
    page->holds_pointers = holds_pointers;
    return page;
}

// A slot of the class (an object's room) from a page with room, or a new page; NULL if there is no
// memory for one.
static void *hornet_take_slot(int class, int holds_pointers, int64_t size_for_message) {
    struct hornet_page **with_room = &hornet_with_room[holds_pointers][class];
    while (*with_room != NULL && (*with_room)->free == NULL && (*with_room)->fresh == (*with_room)->slots) {
        *with_room = (*with_room)->next;  // (full: off the list)
    }
    struct hornet_page *page = *with_room;
    if (page == NULL) {
        page = hornet_new_small_page(class, holds_pointers, size_for_message);
        if (page == NULL) {
            return NULL;
        }
        *with_room = page;
    }
    uint32_t index;
    char *slot;
    if (page->free != NULL) {
        slot = page->free;
        memcpy(&page->free, slot, sizeof page->free);
        index = (uint32_t)(((uintptr_t)slot - page->start) / page->object_size);
    } else {
        index = page->fresh++;
        slot = (char *)page->start + (size_t)index * page->object_size;
    }
    hornet_set_bit(page->allocated, index, 1);
    page->live++;
    return slot;
}

static void *hornet_take_large(int64_t size, int holds_pointers) {
    size_t pages = ((size_t)size + HORNET_PAGE_SIZE - 1) >> HORNET_PAGE_SHIFT;
    void *memory = hornet_get_pages(pages << HORNET_PAGE_SHIFT);
    if (memory == NULL) {
        return NULL;
    }
    struct hornet_page *page = hornet_new_record((uintptr_t)memory, pages, 0, size);
    page->large_size = size;
    page->holds_pointers = holds_pointers;
    page->live = 1;
    return memory;
}

void *hornet__heap_alloc(int64_t size, int holds_pointers, int zeroed) {
    if (size < 0 || (uint64_t)size > (uint64_t)PTRDIFF_MAX - HORNET_PAGE_SIZE) {
        hornet__out_of_memory(size, -1);
    }
    if (!hornet_settings_read) {
        hornet_read_settings();
    }
    int small = size <= HORNET_LARGEST_SMALL;
    int class = small ? hornet_class_of(size) : 0;
    int64_t span = small ? (int64_t)hornet_classes[class] : size;  // what the collector would look through
    if (hornet_allocated_since + span > hornet_collect_at
        || (hornet_stress_every > 0 && ++hornet_stress_count % hornet_stress_every == 0)) {
        hornet__gc_collect();
    }
    holds_pointers = holds_pointers != 0;
    void *object;
    for (int asked = 0;; asked++) {  // (one place each is taken from, for the compiler to inline)
        object = small ? hornet_take_slot(class, holds_pointers, size) : hornet_take_large(size, holds_pointers);
        if (object != NULL) {
            break;
        }
        if (asked == 1) {
            hornet__out_of_memory(size, -1);
        }
        hornet__gc_collect();  // refused: collect, and ask once more
    }
    hornet_allocated_since += span;
    // Memory that may hold pointers is all zero, so that what it held before is never taken for one.
    if (holds_pointers) {
        memset(object, 0, (size_t)span);
    } else if (zeroed) {
        memset(object, 0, (size_t)size);
    }
    return object;
}

static struct hornet_page *hornet_page_of(uintptr_t address) {
    if (address < hornet_heap_low || address >= hornet_heap_high) {
        return NULL;
    }
    uintptr_t number = address >> HORNET_PAGE_SHIFT;
    struct hornet_page **block = hornet_page_table[(number >> 16) & 0xFFFF];
    return (number >> 32) == 0 && block != NULL ? block[number & 0xFFFF] : NULL;
}

// The page `address` is in, and its object's index there (0 for a large object): 0 if it isn't in
// an object that is allocated.
static int hornet_locate(uintptr_t address, struct hornet_page **found_page, uint32_t *found_index) {
    struct hornet_page *page = hornet_page_of(address);
    if (page == NULL) {
        return 0;
    }
    uintptr_t offset = address - page->start;
    if (page->object_size == 0) {
        *found_page = page;
        *found_index = 0;
        return offset < (uintptr_t)page->large_size;  // (not the rest of its last page)
    }
    uint32_t index = (uint32_t)(offset / page->object_size);
    *found_page = page;
    *found_index = index;
    return index < page->slots && hornet_bit(page->allocated, index);
}

int hornet__heap_find(uintptr_t address, struct hornet_heap_object *found) {
    struct hornet_page *page;
    uint32_t index;
    if (!hornet_locate(address, &page, &index)) {
        return 0;
    }
    if (page->object_size == 0) {
        *found = (struct hornet_heap_object){(char *)page->start, page->large_size, page->holds_pointers};
    } else {
        *found = (struct hornet_heap_object){(char *)page->start + (size_t)index * page->object_size,
                                             page->object_size, page->holds_pointers};
    }
    return 1;
}

// ---- Collecting.

// If `word` is the address of (anywhere in) an object not yet found, mark it found; and if it may
// hold pointers, keep it to be looked through.
static void hornet_mark_word(uintptr_t word) {
    struct hornet_page *page;
    uint32_t index;
    if (!hornet_locate(word, &page, &index)) {
        return;
    }
    if (page->object_size == 0) {
        if (page->large_marked) {
            return;
        }
        page->large_marked = 1;
    } else {
        if (hornet_bit(page->marked, index)) {
            return;
        }
        hornet_set_bit(page->marked, index, 1);
    }
    if (page->holds_pointers) {
        if (hornet_mark_count == hornet_mark_room) {
            hornet_mark_stack =
                hornet_scratch_grow(hornet_mark_stack, &hornet_mark_room, sizeof *hornet_mark_stack, 0);
        }
        struct hornet_heap_object object = {(char *)page->start, page->large_size, 1};
        if (page->object_size != 0) {
            object = (struct hornet_heap_object){(char *)page->start + (size_t)index * page->object_size,
                                                 page->object_size, 1};
        }
        hornet_mark_stack[hornet_mark_count++] = object;
    }
}

// Each word from `low` to `high`, as hornet_mark_word. (Reading what isn't an object of the
// program's: the address sanitizer would object to the stack's.)
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define HORNET_NO_SANITIZE __attribute__((no_sanitize("address")))
#endif
#endif
#if !defined(HORNET_NO_SANITIZE) && defined(__SANITIZE_ADDRESS__)
#define HORNET_NO_SANITIZE __attribute__((no_sanitize_address))
#endif
#ifndef HORNET_NO_SANITIZE
#define HORNET_NO_SANITIZE
#endif

HORNET_NO_SANITIZE static void hornet_mark_range(uintptr_t low, uintptr_t high) {
    for (uintptr_t at = (low + 7) & ~(uintptr_t)7; at + sizeof(uintptr_t) <= high; at += sizeof(uintptr_t)) {
        hornet_mark_word(*(const uintptr_t *)at);
    }
}

// The top of the stack: above every frame of the program's.
static uintptr_t hornet_stack_top(void) {
#ifdef _WIN32
    ULONG_PTR low, high;
    GetCurrentThreadStackLimits(&low, &high);
    return (uintptr_t)high;
#elif defined(__APPLE__)
    return (uintptr_t)pthread_get_stackaddr_np(pthread_self());
#else
    extern void *__libc_stack_end;  // (glibc's: where the stack was when the program started)
    return (uintptr_t)__libc_stack_end;
#endif
}

// The stack, from this function's frame up: below its caller's, where the registers were put.
__attribute__((noinline)) HORNET_NO_SANITIZE static void hornet_mark_the_stack(void) {
    volatile uintptr_t here = 0;
    hornet_mark_range((uintptr_t)&here, hornet_stack_top());
}

static void hornet_mark_from_roots(void) {
    __builtin_unwind_init();  // the registers that callers expect kept, into this frame, where they are scanned
    hornet_mark_the_stack();
    while (hornet_mark_count > 0) {
        struct hornet_heap_object object = hornet_mark_stack[--hornet_mark_count];
        hornet_mark_range((uintptr_t)object.start, (uintptr_t)object.start + (uintptr_t)object.size);
    }
}

// Free each object not marked, and clear the marks; give back pages with nothing left. The pages
// with room are listed afresh.
static void hornet_sweep(void) {
    int64_t live_bytes = 0;
    memset(hornet_with_room, 0, sizeof hornet_with_room);
    size_t kept = 0;
    for (size_t i = 0; i < hornet_page_count; i++) {
        struct hornet_page *page = hornet_pages[i];
        if (page->object_size == 0) {
            if (page->large_marked) {
                page->large_marked = 0;
                live_bytes += page->large_size;
                hornet_pages[kept++] = page;
                continue;
            }
            size_t pages = ((size_t)page->large_size + HORNET_PAGE_SIZE - 1) >> HORNET_PAGE_SHIFT;
            if (hornet_poison) {
                memset((void *)page->start, 0xdb, (size_t)page->large_size);
            }
            hornet_record_pages(page->start, pages, NULL, 0);
            hornet_free_pages((void *)page->start);
            hornet_stats.pages -= pages;
            hornet_stats.freed_bytes += page->large_size;
            free(page);
            continue;
        }
        for (uint32_t word = 0; word * 64 < page->fresh; word++) {
            uint64_t garbage = page->allocated[word] & ~page->marked[word];
            page->allocated[word] &= page->marked[word];
            page->marked[word] = 0;
            while (garbage != 0) {
                int bit = __builtin_ctzll(garbage);
                garbage &= garbage - 1;
                char *slot = (char *)page->start + (size_t)(word * 64 + (uint32_t)bit) * page->object_size;
                if (hornet_poison) {
                    memset(slot, 0xdb, page->object_size);
                }
                memcpy(slot, &page->free, sizeof page->free);
                page->free = slot;
                page->live--;
                hornet_stats.freed_bytes += page->object_size;
            }
        }
        if (page->live == 0) {  // nothing left on it: back to the C library
            hornet_record_pages(page->start, 1, NULL, 0);
            hornet_free_pages((void *)page->start);
            hornet_stats.pages -= 1;
            free(page);
            continue;
        }
        live_bytes += (int64_t)page->live * page->object_size;
        if (page->free != NULL || page->fresh < page->slots) {
            int class = hornet_class_of(page->object_size);
            page->next = hornet_with_room[page->holds_pointers][class];
            hornet_with_room[page->holds_pointers][class] = page;
        }
        hornet_pages[kept++] = page;
    }
    hornet_page_count = kept;
    hornet_stats.live_bytes = live_bytes;
    hornet_collect_at = live_bytes > (8 << 20) ? live_bytes : (8 << 20);
    hornet_allocated_since = 0;
}

void hornet__gc_collect(void) {
    hornet_mark_from_roots();
    hornet_sweep();
    hornet_stats.collections++;
}

void hornet__heap_stats(struct hornet_heap_stats *stats) {
    *stats = hornet_stats;
}
