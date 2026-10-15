// White-box tests of the runtime: hand-built type descriptors, shaped as ir/typedesc.py emits them,
// formatted through hornet__stringify and hornet_print. Compiled with the runtime's own files, and
// with what they share (internal.h) to reach past the entry points.
#include "internal.h"

#ifdef _WIN32
#define pipe(fds) _pipe(fds, 65536, _O_BINARY)  // what MinGW has for POSIX's pipe(2)
#endif

#include <assert.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static int g_failures = 0;

#define CHECK_STR(actual, expected)                                                        \
    do {                                                                                    \
        if (strcmp((actual), (expected)) != 0) {                                            \
            fprintf(stderr, "FAIL %s:%d: expected %s, got %s\n", __FILE__, __LINE__,        \
                    (expected), (actual));                                                  \
            g_failures++;                                                                   \
        }                                                                                    \
    } while (0)

#define CHECK(condition)                                                                    \
    do {                                                                                    \
        if (!(condition)) {                                                                 \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);            \
            g_failures++;                                                                   \
        }                                                                                   \
    } while (0)

// Stringifies value_addr/type_desc with the given quote_strings into
// a fresh buffer and returns a null-terminated copy of the result --
// caller must free() it.
static char *stringify_to_cstr(void *value_addr, const uint64_t *type_desc, int quote_strings) {
    struct hornet_buf buf;
    buf.cap = 4;  // deliberately tiny, to force growth on almost every test
    buf.ptr = hornet__scratch_alloc(buf.cap);
    buf.len = 0;
    buf.scratch = 1;
    hornet__stringify(value_addr, (const unsigned char *)type_desc, quote_strings, &buf);
    char *result = malloc((size_t)buf.len + 1);
    memcpy(result, buf.ptr, (size_t)buf.len);
    result[buf.len] = '\0';
    hornet__scratch_free(buf.ptr);
    return result;
}

static void test_int(void) {
    int64_t v;
    uint64_t desc[] = {HORNET_TYPEDESC_INT};

    v = 42;
    char *s = stringify_to_cstr(&v, desc, 0);
    CHECK_STR(s, "42");
    free(s);

    v = -17;
    s = stringify_to_cstr(&v, desc, 0);
    CHECK_STR(s, "-17");
    free(s);

    v = 0;
    s = stringify_to_cstr(&v, desc, 0);
    CHECK_STR(s, "0");
    free(s);

    v = INT64_MIN;
    s = stringify_to_cstr(&v, desc, 0);
    CHECK_STR(s, "-9223372036854775808");
    free(s);
}

static void test_int8_uint8(void) {
    int8_t v8;
    uint64_t desc8[] = {HORNET_TYPEDESC_INT8};
    v8 = -56;  // 0xC8: the sign must be extended
    char *s = stringify_to_cstr(&v8, desc8, 0);
    CHECK_STR(s, "-56");
    free(s);

    v8 = 127;
    s = stringify_to_cstr(&v8, desc8, 0);
    CHECK_STR(s, "127");
    free(s);

    v8 = -128;
    s = stringify_to_cstr(&v8, desc8, 0);
    CHECK_STR(s, "-128");
    free(s);

    uint8_t vu8 = 200;
    uint64_t descu8[] = {HORNET_TYPEDESC_UINT8};
    s = stringify_to_cstr(&vu8, descu8, 0);
    CHECK_STR(s, "200");
    free(s);
}

static void test_int32(void) {
    int32_t v;
    uint64_t desc[] = {HORNET_TYPEDESC_INT32};

    v = INT32_MAX;
    char *s = stringify_to_cstr(&v, desc, 0);
    CHECK_STR(s, "2147483647");
    free(s);

    v = INT32_MIN;
    s = stringify_to_cstr(&v, desc, 0);
    CHECK_STR(s, "-2147483648");
    free(s);
}

static void test_bool(void) {
    int32_t v;
    uint64_t desc[] = {HORNET_TYPEDESC_BOOL};

    v = 1;
    char *s = stringify_to_cstr(&v, desc, 0);
    CHECK_STR(s, "true");
    free(s);

    v = 0;
    s = stringify_to_cstr(&v, desc, 0);
    CHECK_STR(s, "false");
    free(s);
}

static void test_enum(void) {
    // [tag, name, member_count, member_name...]; the value is a member's index, 4 bytes.
    uint64_t desc[] = {HORNET_TYPEDESC_ENUM, (uint64_t)"Color", 2, (uint64_t)"Red", (uint64_t)"Green"};
    int32_t v = 1;
    char *s = stringify_to_cstr(&v, desc, 0);
    CHECK_STR(s, "Color.Green");
    free(s);

    v = 7;  // not a member
    s = stringify_to_cstr(&v, desc, 0);
    CHECK_STR(s, "Color.7");
    free(s);
}

static void test_str(void) {
    // str: 16-byte {ptr, int64 len}; no NUL terminator.
    uint64_t desc[] = {HORNET_TYPEDESC_STR};

    struct {
        const char *ptr;
        int64_t len;
    } str_value = {"hello", 5};

    char *s = stringify_to_cstr(&str_value, desc, 0);
    CHECK_STR(s, "hello");
    free(s);

    s = stringify_to_cstr(&str_value, desc, 1);
    CHECK_STR(s, "'hello'");
    free(s);

    struct {
        const char *ptr;
        int64_t len;
    } empty_value = {"", 0};
    s = stringify_to_cstr(&empty_value, desc, 0);
    CHECK_STR(s, "");
    free(s);

    // An embedded '\0' byte is ordinary string content now, not a
    // terminator: len alone decides where the string ends, so this
    // must print all 11 bytes, not stop at the 5th.
    struct {
        const char *ptr;
        int64_t len;
    } embedded_null_value = {"hello\0world", 11};
    s = stringify_to_cstr(&embedded_null_value, desc, 0);
    if (memcmp(s, "hello\0world", 11) != 0 || strlen(s) != 5) {
        fprintf(
            stderr,
            "FAIL %s:%d: expected 11 bytes 'hello\\0world', got %zu bytes\n",
            __FILE__,
            __LINE__,
            strlen(s)
        );
        g_failures++;
    }
    free(s);
}

static void test_array(void) {
    // [3]int -- descriptor: [tag, name, elem_desc, count, elem_width]
    uint64_t elem_desc[] = {HORNET_TYPEDESC_INT};
    uint64_t arr_desc[] = {HORNET_TYPEDESC_ARRAY, (uint64_t)"[3]int", (uint64_t)elem_desc, 3, 8};
    int64_t values[3] = {1, 2, 3};
    char *s = stringify_to_cstr(values, arr_desc, 0);
    CHECK_STR(s, "[3]int[1, 2, 3]");
    free(s);

    // Zero-length array
    uint64_t empty_arr_desc[] = {HORNET_TYPEDESC_ARRAY, (uint64_t)"[0]int", (uint64_t)elem_desc, 0, 8};
    s = stringify_to_cstr(values, empty_arr_desc, 0);
    CHECK_STR(s, "[0]int[]");
    free(s);
}

static void test_slice(void) {
    // []int -- descriptor: [tag, name, elem_desc, elem_width]. Value
    // is {ptr, int64 len, int64 cap}.
    uint64_t elem_desc[] = {HORNET_TYPEDESC_INT};
    uint64_t slice_desc[] = {HORNET_TYPEDESC_SLICE, (uint64_t)"[]int", (uint64_t)elem_desc, 8};

    int64_t backing[2] = {10, 20};
    struct {
        void *ptr;
        int64_t len;
        int64_t cap;
    } slice_value = {backing, 2, 2};
    char *s = stringify_to_cstr(&slice_value, slice_desc, 0);
    CHECK_STR(s, "[]int[10, 20]");
    free(s);

    // nil/empty slice
    struct {
        void *ptr;
        int64_t len;
        int64_t cap;
    } nil_slice = {NULL, 0, 0};
    s = stringify_to_cstr(&nil_slice, slice_desc, 0);
    CHECK_STR(s, "[]int[]");
    free(s);
}

static void test_struct(void) {
    // struct Point: int x; int y -- descriptor: [tag, name,
    // field_count, (fname, ftype, foffset) x field_count]
    uint64_t int_desc[] = {HORNET_TYPEDESC_INT};
    uint64_t point_desc[] = {
        HORNET_TYPEDESC_STRUCT, (uint64_t)"Point", 2,
        (uint64_t)"x", (uint64_t)int_desc, 0,
        (uint64_t)"y", (uint64_t)int_desc, 8,
    };
    struct {
        int64_t x;
        int64_t y;
    } point = {1, 2};
    char *s = stringify_to_cstr(&point, point_desc, 0);
    CHECK_STR(s, "Point(x: 1, y: 2)");
    free(s);
}

static void test_nested_array_of_structs(void) {
    uint64_t int_desc[] = {HORNET_TYPEDESC_INT};
    uint64_t point_desc[] = {
        HORNET_TYPEDESC_STRUCT, (uint64_t)"Point", 2,
        (uint64_t)"x", (uint64_t)int_desc, 0,
        (uint64_t)"y", (uint64_t)int_desc, 8,
    };
    uint64_t arr_desc[] = {HORNET_TYPEDESC_ARRAY, (uint64_t)"[2]Point", (uint64_t)point_desc, 2, 16};
    struct {
        int64_t x, y;
    } points[2] = {{1, 2}, {3, 4}};
    char *s = stringify_to_cstr(points, arr_desc, 0);
    CHECK_STR(s, "[2]Point[Point(x: 1, y: 2), Point(x: 3, y: 4)]");
    free(s);
}

static void test_struct_with_slice_field(void) {
    uint64_t int_desc[] = {HORNET_TYPEDESC_INT};
    uint64_t slice_desc[] = {HORNET_TYPEDESC_SLICE, (uint64_t)"[]int", (uint64_t)int_desc, 8};
    uint64_t box_desc[] = {
        HORNET_TYPEDESC_STRUCT, (uint64_t)"Box", 1,
        (uint64_t)"values", (uint64_t)slice_desc, 0,
    };
    int64_t backing[3] = {7, 8, 9};
    struct {
        void *ptr;
        int64_t len;
        int64_t cap;
    } slice_value = {backing, 3, 3};
    struct {
        __typeof__(slice_value) values;
    } box = {slice_value};
    char *s = stringify_to_cstr(&box, box_desc, 0);
    CHECK_STR(s, "Box(values: []int[7, 8, 9])");
    free(s);
}

static void test_nested_struct_field(void) {
    uint64_t int_desc[] = {HORNET_TYPEDESC_INT};
    uint64_t point_desc[] = {
        HORNET_TYPEDESC_STRUCT, (uint64_t)"Point", 2,
        (uint64_t)"x", (uint64_t)int_desc, 0,
        (uint64_t)"y", (uint64_t)int_desc, 8,
    };
    uint64_t wrapper_desc[] = {
        HORNET_TYPEDESC_STRUCT, (uint64_t)"Wrapper", 1,
        (uint64_t)"p", (uint64_t)point_desc, 0,
    };
    struct {
        int64_t x, y;
    } wrapper = {5, 6};
    char *s = stringify_to_cstr(&wrapper, wrapper_desc, 0);
    CHECK_STR(s, "Wrapper(p: Point(x: 5, y: 6))");
    free(s);
}

// Fields of different alignments, padded between as C pads them: the descriptor's offsets are C's.
static void test_padded_struct(void) {
    uint64_t int8_desc[] = {HORNET_TYPEDESC_INT8};
    uint64_t int_desc[] = {HORNET_TYPEDESC_INT};
    uint64_t int32_desc[] = {HORNET_TYPEDESC_INT32};
    struct padded {
        int8_t a;
        int64_t b;
        int32_t c;
    } value = {1, 2, 3};
    uint64_t desc[] = {
        HORNET_TYPEDESC_STRUCT, (uint64_t)"Padded", 3,
        (uint64_t)"a", (uint64_t)int8_desc, offsetof(struct padded, a),
        (uint64_t)"b", (uint64_t)int_desc, offsetof(struct padded, b),
        (uint64_t)"c", (uint64_t)int32_desc, offsetof(struct padded, c),
    };
    char *s = stringify_to_cstr(&value, desc, 0);
    CHECK_STR(s, "Padded(a: 1, b: 2, c: 3)");
    free(s);
}

// A sum's payload is at HORNET_SUM_PAYLOAD_OFFSET, past its tag and padding.
static void test_sum_payload(void) {
    uint64_t int_desc[] = {HORNET_TYPEDESC_INT};
    uint64_t int8_desc[] = {HORNET_TYPEDESC_INT8};
    uint64_t sum_desc[] = {HORNET_TYPEDESC_SUM, 2, (uint64_t)int8_desc, (uint64_t)int_desc};
    struct {
        int32_t tag;
        int64_t payload;
    } value = {1, -5};
    _Static_assert(HORNET_SUM_PAYLOAD_OFFSET == 8, "the payload is where C puts an int64 after an int32");
    char *s = stringify_to_cstr(&value, sum_desc, 0);
    CHECK_STR(s, "-5");
    free(s);
}

// A dict printed by its header's bucket layout: an int8 key at 1, an int value at 8, 16 apart.
static void test_dict_bucket_layout(void) {
    uint64_t int8_desc[] = {HORNET_TYPEDESC_INT8};
    uint64_t int_desc[] = {HORNET_TYPEDESC_INT};
    uint64_t dict_desc[] = {HORNET_TYPEDESC_DICT, (uint64_t)"dict[int8]int", (uint64_t)int8_desc, 1,
                            (uint64_t)int_desc, 8};
    struct {
        uint8_t state;
        int8_t key;
        int64_t value;
    } buckets[4] = {{0, 0, 0}, {HORNET_DICT_BUCKET_OCCUPIED, 1, 10}, {HORNET_DICT_BUCKET_TOMBSTONE, 9, 90},
                    {HORNET_DICT_BUCKET_OCCUPIED, 2, 20}};
    struct hornet_dict header = {buckets, 2, 1, 4, 1, 8, sizeof buckets[0], 0};
    struct hornet_dict *value = &header;
    char *s = stringify_to_cstr(&value, dict_desc, 0);
    CHECK_STR(s, "dict[int8]int{1: 10, 2: 20}");
    free(s);
}

// The heap: every address inside an object leads to the object, and no other address does.
static void test_heap_finds_each_object_from_inside_it(void) {
    static const int64_t sizes[] = {0, 1, 7, 8, 9, 24, 40, 64, 65, 100, 1000, 4097, 20000, 32768};
    for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; i++) {
        for (int holds = 0; holds <= 1; holds++) {
            int64_t size = sizes[i];
            char *object = hornet__heap_alloc(size, holds, 0);
            struct hornet_heap_object found;
            CHECK((uintptr_t)object % 8 == 0);
            CHECK(hornet__heap_find((uintptr_t)object, &found) && found.start == object);
            CHECK(found.size >= size && found.size < (size < 8 ? 8 : size) * 5 / 4 + 8);  // its class, a quarter apart
            CHECK(found.holds_pointers == holds);
            CHECK(hornet__heap_find((uintptr_t)object + (uintptr_t)found.size - 1, &found) && found.start == object);
            if (holds) {
                int zero = 1;  // the whole slot, so that nothing it held before looks like a pointer
                for (int64_t j = 0; j < found.size; j++) {
                    zero &= object[j] == 0;
                }
                CHECK(zero);
            }
        }
    }
}

// Each size rounds up to the smallest class that holds it: the classes are the sizes that round to
// themselves, and a size's room changes only past one of them.
static void test_heap_rounds_each_size_up_to_its_class(void) {
    int ok = hornet__heap_slot_size(0) == 8;
    for (int64_t size = 1; size <= 32768; size++) {
        int64_t room = hornet__heap_slot_size(size), before = hornet__heap_slot_size(size - 1);
        ok &= room >= size && room % 8 == 0 && hornet__heap_slot_size(room) == room;
        ok &= room == before || before == size - 1;
        ok &= room < (size < 8 ? 8 : size) * 5 / 4 + 8;  // a quarter apart
    }
    ok &= hornet__heap_slot_size(32769) == 32769;
    CHECK(ok);
}

// From its first byte and its last, each object of each class on a page or more of them. (The slot
// an address is in is found by multiplying, not dividing: the pages' ends are where that would err.)
static void test_heap_finds_each_object_of_a_full_page(void) {
    int ok = 1;
    for (int64_t size = 8; size <= 32768; size = hornet__heap_slot_size(size + 1)) {
        int64_t count = 65536 / size + 2;
        char **volatile objects = hornet__heap_alloc(count * (int64_t)sizeof(char *), 1, 0);
        for (int64_t i = 0; i < count; i++) {
            objects[i] = hornet__heap_alloc(size, 0, 0);
        }
        for (int64_t i = 0; i < count; i++) {
            struct hornet_heap_object first, last;
            ok &= hornet__heap_find((uintptr_t)objects[i], &first) && first.start == objects[i];
            ok &= hornet__heap_find((uintptr_t)objects[i] + (uintptr_t)size - 1, &last) && last.start == objects[i];
            ok &= first.size == size;
        }
    }
    CHECK(ok);
}

static void test_heap_tells_a_slot_not_handed_out(void) {
    char *object = hornet__heap_alloc(20000, 1, 0);  // (a class few others use: 20480 bytes, 3 to a page)
    struct hornet_heap_object found;
    CHECK(hornet__heap_find((uintptr_t)object, &found) && found.size == 20480);
    struct hornet_heap_object after;
    int next = hornet__heap_find((uintptr_t)object + 20480, &after);
    CHECK(!next || after.start == object + 20480);  // the next slot: someone else's object, or none yet
    char *second = hornet__heap_alloc(20000, 1, 0);
    CHECK(hornet__heap_find((uintptr_t)second + 100, &found) && found.start == second);
}

static void test_heap_large_objects(void) {
    int64_t size = 3 * 65536 + 1000;  // four pages, the last mostly unused
    char *object = hornet__heap_alloc(size, 0, 1);
    struct hornet_heap_object found;
    CHECK((uintptr_t)object % 65536 == 0);
    CHECK(object[0] == 0 && object[size - 1] == 0);
    for (int64_t at = 0; at < size; at += 65536 / 2) {
        int in = hornet__heap_find((uintptr_t)object + (uintptr_t)at, &found);
        CHECK(in && found.start == object && found.size == size);
    }
    CHECK(hornet__heap_find((uintptr_t)object + (uintptr_t)size - 1, &found) && found.start == object);
    CHECK(!hornet__heap_find((uintptr_t)object + (uintptr_t)size, &found));  // the rest of its last page
    CHECK(!found.holds_pointers);
}

static int g_static_value;

static void test_heap_finds_nothing_elsewhere(void) {
    int local = 0;
    struct hornet_heap_object found;
    void *from_malloc = malloc(64);
    CHECK(!hornet__heap_find(0, &found));
    CHECK(!hornet__heap_find(1, &found));
    CHECK(!hornet__heap_find(UINTPTR_MAX, &found));
    CHECK(!hornet__heap_find((uintptr_t)&local, &found));
    CHECK(!hornet__heap_find((uintptr_t)&g_static_value, &found));
    CHECK(!hornet__heap_find((uintptr_t)from_malloc, &found));
    CHECK(!hornet__heap_find((uintptr_t)"a literal", &found));
    free(from_malloc);
}

static void test_heap_grows_a_str_s_bytes_keeping_them(void) {
    char *text = hornet_alloc_atomic(5);
    memcpy(text, "hello", 5);
    char *same = hornet__realloc_atomic(text, 8);  // (its slot is 8 bytes already)
    CHECK(same == text);
    char *grown = hornet__realloc_atomic(text, 100000);
    CHECK(grown != text && memcmp(grown, "hello", 5) == 0);
    struct hornet_heap_object found;
    CHECK(hornet__heap_find((uintptr_t)grown + 99999, &found) && found.start == grown && !found.holds_pointers);
}

// ---- The collector. An address a test checks later is kept as its complement, which doesn't look
// like a pointer, so that keeping it doesn't keep the object; garbage is made in functions of its
// own, whose frames are gone (and below the stack scanned) by the time the collector runs.

struct node {
    struct node *next;
    int64_t value;
};

// That garbage was freed: not checked under the address sanitizer, whose stack frames keep words of
// earlier frames that no clearing reaches, and that the collector, being conservative, can't tell
// from pointers. (What it checks there is the other half: that nothing kept was freed.)
#if defined(__SANITIZE_ADDRESS__)
#define HORNET_UNDER_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define HORNET_UNDER_ASAN 1
#endif
#endif
#ifdef HORNET_UNDER_ASAN
#define CHECK_FREED(condition) ((void)0)
#else
#define CHECK_FREED(condition) CHECK(condition)
#endif

static int alive(uintptr_t complement) {
    struct hornet_heap_object found;
    return hornet__heap_find(~complement, &found);
}

// Zero the stack below here, where the frames of the functions that made the garbage were: the
// collector's own frames go there, and may leave some of it as it was, for it to read. (In a program
// that keeps an object a while longer; in a test, it would be a failure.)
__attribute__((noinline)) static void clear_stack(void) {
    volatile char below[64 * 1024];
    for (size_t i = 0; i < sizeof below; i++) {  // (a memset of it, never read, could be left out)
        below[i] = 0;
    }
}

static void collect(void) {
    clear_stack();
    hornet__gc_collect();
}

__attribute__((noinline)) static uintptr_t make_garbage_list(int count) {
    struct node *head = NULL;
    for (int i = 0; i < count; i++) {
        struct node *node = hornet__heap_alloc(sizeof *node, 1, 0);
        node->next = head;
        node->value = i;
        head = node;
    }
    return ~(uintptr_t)head;
}

__attribute__((noinline)) static struct node *make_list(int count) {
    struct node *head = NULL;
    for (int i = 0; i < count; i++) {
        struct node *node = hornet__heap_alloc(sizeof *node, 1, 0);
        node->next = head;
        node->value = i;
        head = node;
    }
    return head;
}

static void test_gc_keeps_what_the_stack_reaches_and_frees_the_rest(void) {
    struct node *volatile kept = make_list(10000);
    uintptr_t dropped = make_garbage_list(10000);
    struct hornet_heap_stats before, after;
    hornet__heap_stats(&before);
    collect();
    hornet__heap_stats(&after);
    CHECK(after.collections == before.collections + 1);
    CHECK_FREED(!alive(dropped));
    CHECK_FREED(after.freed_bytes - before.freed_bytes >= 9990 * 16);
    int64_t expected = 9999, intact = 1;
    for (struct node *node = kept; node != NULL; node = node->next) {
        intact &= node->value == expected--;
    }
    CHECK(intact && expected == -1);
}

__attribute__((noinline)) static uintptr_t hide_in_atomic(char **atomic) {
    struct node *node = hornet__heap_alloc(sizeof *node, 1, 0);
    *atomic = hornet__heap_alloc(sizeof node, 0, 0);
    memcpy(*atomic, &node, sizeof node);  // the only pointer to it, where the collector doesn't look
    return ~(uintptr_t)node;
}

static void test_gc_doesnt_look_through_what_holds_no_pointers(void) {
    char *volatile atomic;
    uintptr_t node = hide_in_atomic((char **)&atomic);
    collect();
    CHECK(alive(~(uintptr_t)atomic));
    CHECK_FREED(!alive(node));
}

__attribute__((noinline)) static char *middle_of_new(int64_t size, uintptr_t *start) {
    char *object = hornet__heap_alloc(size, 0, 1);
    *start = ~(uintptr_t)object;
    object[size / 2] = 42;
    return object + size / 2;
}

static void test_gc_keeps_an_object_pointed_to_inside(void) {
    uintptr_t small_start, large_start;
    char *volatile small = middle_of_new(1000, &small_start);
    char *volatile large = middle_of_new(200000, &large_start);
    collect();
    CHECK(alive(small_start) && alive(large_start));
    CHECK(*small == 42 && *large == 42);
}

__attribute__((noinline)) static uintptr_t make_garbage_large(void) {
    return ~(uintptr_t)hornet__heap_alloc(3 * 65536, 1, 0);
}

static void test_gc_gives_back_a_large_object_s_pages(void) {
    uintptr_t large = make_garbage_large();
    struct hornet_heap_stats before, after;
    hornet__heap_stats(&before);
    collect();
    hornet__heap_stats(&after);
    CHECK_FREED(!alive(large) && !alive(~(~large + 65536 * 2)));  // (from its first page, and its third)
    CHECK_FREED(after.pages <= before.pages - 3);
}

__attribute__((noinline)) static void churn(int64_t bytes) {
    for (int64_t made = 0; made < bytes; made += 64) {
        int64_t *object = hornet__heap_alloc(64, 1, 0);
        object[0] = made;
    }
}

static void test_gc_reuses_what_it_frees(void) {
    struct hornet_heap_stats before, after;
    collect();
    hornet__heap_stats(&before);
    churn((int64_t)200 << 20);  // 200 MB, nothing kept: collections as it goes keep the heap small
    hornet__heap_stats(&after);
    CHECK(after.collections >= before.collections + 10);
    CHECK_FREED(after.pages < before.pages + 64 * 16 * 2);  // (a collection's worth of pages, 8 MB, twice over)
    collect();
    hornet__heap_stats(&after);
    CHECK(after.empty_pages > 0 && after.empty_pages <= 16);  // kept for reuse: 1 MB, as little is alive
}

__attribute__((noinline)) static uintptr_t make_garbage_bytes(void) {
    char *object = hornet__heap_alloc(64, 0, 0);
    memset(object, 'x', 64);
    return ~(uintptr_t)object;
}

static void test_gc_poisons_what_it_frees(void) {
    uintptr_t freed = make_garbage_bytes();
    collect();
    const unsigned char *bytes = (const unsigned char *)~freed;
    int poisoned = !alive(freed);
    for (int i = 8; i < 64; i++) {  // (the first word links it to the page's other free slots)
        poisoned &= bytes[i] == 0xdb;
    }
    CHECK_FREED(poisoned);
}

static void test_buffer_growth_stress(void) {
    // A large array forces many reallocations through the real
    // growth policy (starting from stringify_to_cstr's own
    // deliberately tiny 4-byte initial capacity) -- exactly the kind
    // of stress that would surface a corrupted-pointer or lost-byte
    // bug in the realloc/memcpy path.
    enum { N = 5000 };
    uint64_t elem_desc[] = {HORNET_TYPEDESC_INT};
    uint64_t arr_desc[] = {HORNET_TYPEDESC_ARRAY, (uint64_t)"[5000]int", (uint64_t)elem_desc, N, 8};
    static int64_t values[N];
    for (int i = 0; i < N; i++) {
        values[i] = i;
    }
    char *s = stringify_to_cstr(values, arr_desc, 0);

    // Build the expected string independently, via ordinary sprintf,
    // rather than re-deriving it from the same code under test.
    char *expected = malloc(N * 8 + 64);
    int pos = sprintf(expected, "[5000]int[");
    for (int i = 0; i < N; i++) {
        pos += sprintf(expected + pos, i == 0 ? "%d" : ", %d", i);
    }
    pos += sprintf(expected + pos, "]");
    CHECK_STR(s, expected);
    free(expected);
    free(s);
}

static void test_hornet_print_end_to_end(void) {
    // Redirect stdout to a pipe, call hornet_print, read back what
    // was actually written -- confirms the full path (malloc,
    // stringify, newline append, write, free), not just the
    // stringify logic in isolation.
    int pipefd[2];
    assert(pipe(pipefd) == 0);
    int saved_stdout = dup(1);
    dup2(pipefd[1], 1);
    close(pipefd[1]);

    int64_t v = 123;
    uint64_t desc[] = {HORNET_TYPEDESC_INT};
    hornet_print(&v, (const unsigned char *)desc);

    fflush(stdout);
    dup2(saved_stdout, 1);
    close(saved_stdout);

    char buf[64] = {0};
    ssize_t n = read(pipefd[0], buf, sizeof(buf) - 1);
    close(pipefd[0]);
    assert(n >= 0);
    buf[n] = '\0';
    CHECK_STR(buf, "123\n");
}

static void test_self_referential_struct(void) {
    // struct Node: int value; []Node children. Its descriptor contains a cycle (the slice's element
    // descriptor is the struct's own), as type_descriptor (ir/typedesc.py) emits for a
    // self-referential type; the printer has to follow it by recursion.
    uint64_t int_desc[] = {HORNET_TYPEDESC_INT};

    // children_slice_desc's element descriptor (index 2) is filled in below, once node_desc exists.
    static uint64_t children_slice_desc[4];
    static uint64_t node_desc[6];
    children_slice_desc[0] = HORNET_TYPEDESC_SLICE;
    children_slice_desc[1] = (uint64_t)"[]Node";
    children_slice_desc[2] = (uint64_t)node_desc;
    children_slice_desc[3] = 32;  // Node width: 8-byte int + 24-byte slice

    node_desc[0] = HORNET_TYPEDESC_STRUCT;
    node_desc[1] = (uint64_t)"Node";
    node_desc[2] = 2;
    node_desc[3] = (uint64_t)"value";
    node_desc[4] = (uint64_t)int_desc;
    node_desc[5] = 0;
    // (children's own triple appended below, since C's own static
    // initializer can't easily mix compile-time-sized arrays like
    // this inline -- functionally identical to declaring all 9 words
    // up front.)
    static uint64_t node_desc_full[9];
    memcpy(node_desc_full, node_desc, sizeof(node_desc));
    node_desc_full[6] = (uint64_t)"children";
    node_desc_full[7] = (uint64_t)children_slice_desc;
    node_desc_full[8] = 8;  // "children" offset, after the int
    children_slice_desc[2] = (uint64_t)node_desc_full;  // complete the cycle

    // As Hornet lays the struct out: as C does.
    struct node_value {
        int64_t value;
        void *children_ptr;
        int64_t children_len;
        int64_t children_cap;
    };

    // A small, finite tree: root(1, [leaf(2, []), leaf(3, [])])
    struct node_value leaf2 = {2, NULL, 0, 0};
    struct node_value leaf3 = {3, NULL, 0, 0};
    struct node_value children[2] = {leaf2, leaf3};
    struct node_value root = {1, children, 2, 2};

    char *s = stringify_to_cstr(&root, node_desc_full, 0);
    CHECK_STR(
        s,
        "Node(value: 1, children: []Node[Node(value: 2, children: []Node[]), "
        "Node(value: 3, children: []Node[])])");
    free(s);
}

// Under the address sanitizer, locals stay on the stack, where the collector looks for them (it would
// otherwise move them elsewhere, to catch their use after their function returns).
const char *__asan_default_options(void);
const char *__asan_default_options(void) {
    return "detect_stack_use_after_return=0";
}

int main(void) {
    // Before anything is allocated, when the heap reads its settings. (Not stressed, whatever the
    // tests are run with: these collect when they mean to, and allocate too much to collect each time.)
#ifdef _WIN32
    _putenv("HORNET_GC_POISON=1");
    _putenv("HORNET_GC_STRESS=0");
#else
    setenv("HORNET_GC_POISON", "1", 1);
    setenv("HORNET_GC_STRESS", "0", 1);
#endif
    test_int();
    test_int8_uint8();
    test_int32();
    test_bool();
    test_enum();
    test_str();
    test_array();
    test_slice();
    test_struct();
    test_nested_array_of_structs();
    test_struct_with_slice_field();
    test_nested_struct_field();
    test_self_referential_struct();
    test_padded_struct();
    test_heap_finds_each_object_from_inside_it();
    test_heap_tells_a_slot_not_handed_out();
    test_heap_large_objects();
    test_heap_finds_nothing_elsewhere();
    test_heap_grows_a_str_s_bytes_keeping_them();
    test_heap_rounds_each_size_up_to_its_class();
    test_heap_finds_each_object_of_a_full_page();
    test_gc_keeps_what_the_stack_reaches_and_frees_the_rest();
    test_gc_doesnt_look_through_what_holds_no_pointers();
    test_gc_keeps_an_object_pointed_to_inside();
    test_gc_gives_back_a_large_object_s_pages();
    test_gc_reuses_what_it_frees();
    test_gc_poisons_what_it_frees();
    test_sum_payload();
    test_dict_bucket_layout();
    test_buffer_growth_stress();
    test_hornet_print_end_to_end();

    if (g_failures == 0) {
        printf("all tests passed\n");
        return 0;
    } else {
        printf("%d test(s) failed\n", g_failures);
        return 1;
    }
}
