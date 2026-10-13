// The heap: where the program's values live, in memory this file manages itself rather than one
// malloc call per value, so that any address can be traced to the object it is in (as a collector
// must, for a word that may be a pointer to anywhere inside an object).
//
// Memory comes in 64 KB pages, each aligned to its size, so that an address's page is the address
// with its low 16 bits cleared. A small object goes in a page of objects all of one size class (8
// bytes to 32 KB, about a quarter apart), where its start follows from any address inside it by
// arithmetic; a larger one has a run of pages to itself. A table finds the record of any page from
// its address. Pages of objects that hold pointers and of objects that hold none are kept apart.
//
// Nothing is freed yet: an object stays until the program ends.
#include "internal.h"
#ifdef _WIN32
#include <malloc.h>
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
    uintptr_t start;         // where its objects start: the page, or the large object
    uint32_t object_size;    // its size class; 0 for a large object
    uint32_t used;           // of a small page's slots, how many have been handed out (from the start)
    int64_t large_size;      // a large object's size, as asked for
    int holds_pointers;      // whether its objects may hold a pointer
};

// The table, from a page's number (its address >> 16) to its record: the number's high 16 bits pick
// a block of the table, made when first needed, and its low 16 bits the record in it. (Addresses go
// up to 48 bits.)
static struct hornet_page **hornet_page_table[(size_t)1 << 16];

// The lowest address in the heap, and one past the highest: most words that aren't pointers to it
// are told apart by these alone.
static uintptr_t hornet_heap_low = UINTPTR_MAX, hornet_heap_high = 0;

// The page each class is filling, for objects that may hold pointers [1] and that hold none [0].
static struct hornet_page *hornet_filling[2][HORNET_CLASS_COUNT];

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

static struct hornet_page *hornet_new_record(int64_t size_for_message) {
    struct hornet_page *page = malloc(sizeof *page);
    if (page == NULL) {
        hornet__out_of_memory(size_for_message, -1);
    }
    return page;
}

// Have the table give `page` for each of the `count` pages from `address`.
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
    if (address < hornet_heap_low) {
        hornet_heap_low = address;
    }
    if (address + count * HORNET_PAGE_SIZE > hornet_heap_high) {
        hornet_heap_high = address + count * HORNET_PAGE_SIZE;
    }
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

static void *hornet_large(int64_t size, int holds_pointers) {
    if ((uint64_t)size > (uint64_t)PTRDIFF_MAX - HORNET_PAGE_SIZE) {
        hornet__out_of_memory(size, -1);
    }
    size_t pages = ((size_t)size + HORNET_PAGE_SIZE - 1) >> HORNET_PAGE_SHIFT;
    void *memory = hornet_get_pages(pages << HORNET_PAGE_SHIFT);
    if (memory == NULL) {
        hornet__out_of_memory(size, -1);
    }
    struct hornet_page *page = hornet_new_record(size);
    *page = (struct hornet_page){(uintptr_t)memory, 0, 0, size, holds_pointers};
    hornet_record_pages((uintptr_t)memory, pages, page, size);
    return memory;
}

void *hornet__heap_alloc(int64_t size, int holds_pointers, int zeroed) {
    if (size < 0) {
        hornet__out_of_memory(size, -1);
    }
    void *object;
    int64_t span = size;  // what a collector would look through: the whole of a small object's slot
    if (size > HORNET_LARGEST_SMALL) {
        object = hornet_large(size, holds_pointers);
    } else {
        int class = hornet_class_of(size);
        uint32_t object_size = hornet_classes[class];
        struct hornet_page *page = hornet_filling[holds_pointers != 0][class];
        if (page == NULL || page->used == HORNET_PAGE_SIZE / object_size) {
            void *memory = hornet_get_pages(HORNET_PAGE_SIZE);
            if (memory == NULL) {
                hornet__out_of_memory(size, -1);
            }
            page = hornet_new_record(size);
            *page = (struct hornet_page){(uintptr_t)memory, object_size, 0, 0, holds_pointers};
            hornet_record_pages((uintptr_t)memory, 1, page, size);
            hornet_filling[holds_pointers != 0][class] = page;
        }
        object = (char *)page->start + (size_t)page->used * object_size;
        page->used++;
        span = object_size;
    }
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

int hornet__heap_find(uintptr_t address, struct hornet_heap_object *found) {
    struct hornet_page *page = hornet_page_of(address);
    if (page == NULL) {
        return 0;
    }
    uintptr_t offset = address - page->start;
    if (page->object_size == 0) {
        if (offset >= (uintptr_t)page->large_size) {
            return 0;  // past the object, in the rest of its last page
        }
        *found = (struct hornet_heap_object){(char *)page->start, page->large_size, page->holds_pointers};
        return 1;
    }
    uintptr_t index = offset / page->object_size;
    if (index >= page->used) {
        return 0;  // a slot not handed out
    }
    *found = (struct hornet_heap_object){(char *)page->start + index * page->object_size, page->object_size,
                                         page->holds_pointers};
    return 1;
}
