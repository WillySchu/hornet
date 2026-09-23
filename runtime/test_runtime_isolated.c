// Standalone isolation test for runtime.c -- hand-builds type
// descriptors matching EXACTLY what codegen/strings.py's own
// _get_or_build_type_descriptor would emit for each shape, then
// exercises hornet_stringify/hornet_print directly and checks the
// resulting bytes. Not wired into the Hornet compiler at all -- pure
// C, verifying this file in isolation before it's ever linked into a
// compiled Hornet program.
//
// #include's runtime.c directly (rather than linking against it) so
// this can call the static, internal hornet_stringify/hornet_buf_*
// functions directly for precise, white-box checks, not just
// hornet_print's own external, end-to-end behavior.
#include "runtime.c"

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

// Stringifies value_addr/type_desc with the given quote_strings into
// a fresh buffer and returns a null-terminated copy of the result --
// caller must free() it.
static char *stringify_to_cstr(void *value_addr, const uint64_t *type_desc, int quote_strings) {
    struct hornet_buf buf;
    buf.cap = 4;  // deliberately tiny, to force growth on almost every test
    buf.ptr = malloc((size_t)buf.cap);
    buf.len = 0;
    hornet_stringify(value_addr, (const unsigned char *)type_desc, quote_strings, &buf);
    char *result = malloc((size_t)buf.len + 1);
    memcpy(result, buf.ptr, (size_t)buf.len);
    result[buf.len] = '\0';
    free(buf.ptr);
    return result;
}

static void test_int(void) {
    int32_t v;
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

    v = INT32_MIN;
    s = stringify_to_cstr(&v, desc, 0);
    CHECK_STR(s, "-2147483648");
    free(s);
}

static void test_int8_uint8(void) {
    int8_t v8;
    uint64_t desc8[] = {HORNET_TYPEDESC_INT8};
    v8 = -56;  // 0xC8, the sign-extension case already reasoned through in the old assembly
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

static void test_int64(void) {
    int64_t v;
    uint64_t desc[] = {HORNET_TYPEDESC_INT64};

    v = INT64_MAX;
    char *s = stringify_to_cstr(&v, desc, 0);
    CHECK_STR(s, "9223372036854775807");
    free(s);

    v = INT64_MIN;
    s = stringify_to_cstr(&v, desc, 0);
    CHECK_STR(s, "-9223372036854775808");
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

static void test_str(void) {
    // A str value is a 16-byte {ptr, len} descriptor now (see ir/
    // strings.py's own module docstring): ptr (8 bytes), len (an
    // ordinary int32_t, in its own 8-byte slot -- same layout a
    // SLICE's own len/cap fields already use just above, minus cap).
    // No null terminator anywhere in this scheme -- len alone decides
    // how many bytes get read, exactly like it already does for a
    // slice's own backing storage.
    uint64_t desc[] = {HORNET_TYPEDESC_STR};

    struct {
        const char *ptr;
        int32_t len;
    } str_value = {"hello", 5};

    char *s = stringify_to_cstr(&str_value, desc, 0);
    CHECK_STR(s, "hello");
    free(s);

    s = stringify_to_cstr(&str_value, desc, 1);
    CHECK_STR(s, "'hello'");
    free(s);

    struct {
        const char *ptr;
        int32_t len;
    } empty_value = {"", 0};
    s = stringify_to_cstr(&empty_value, desc, 0);
    CHECK_STR(s, "");
    free(s);

    // An embedded '\0' byte is ordinary string content now, not a
    // terminator: len alone decides where the string ends, so this
    // must print all 11 bytes, not stop at the 5th.
    struct {
        const char *ptr;
        int32_t len;
    } embedded_null_value = {"hello\0world", 11};
    s = stringify_to_cstr(&embedded_null_value, desc, 0);
    if (memcmp(s, "hello\0world", 11) != 0 || strlen(s) != 5) {
        fprintf(stderr, "FAIL %s:%d: expected 11 bytes 'hello\\0world', got %zu bytes\n", __FILE__, __LINE__, strlen(s));
        g_failures++;
    }
    free(s);
}

static void test_array(void) {
    // [3]int -- descriptor: [tag, name, elem_desc, count, elem_width]
    uint64_t elem_desc[] = {HORNET_TYPEDESC_INT};
    uint64_t arr_desc[] = {HORNET_TYPEDESC_ARRAY, (uint64_t)"[3]int", (uint64_t)elem_desc, 3, 4};
    int32_t values[3] = {1, 2, 3};
    char *s = stringify_to_cstr(values, arr_desc, 0);
    CHECK_STR(s, "[3]int[1, 2, 3]");
    free(s);

    // Zero-length array
    uint64_t empty_arr_desc[] = {HORNET_TYPEDESC_ARRAY, (uint64_t)"[0]int", (uint64_t)elem_desc, 0, 4};
    s = stringify_to_cstr(values, empty_arr_desc, 0);
    CHECK_STR(s, "[0]int[]");
    free(s);
}

static void test_slice(void) {
    // []int -- descriptor: [tag, name, elem_desc, elem_width]. Value
    // is a runtime {ptr, len, cap} triple: ptr (8 bytes), len (4
    // bytes), cap (4 bytes).
    uint64_t elem_desc[] = {HORNET_TYPEDESC_INT};
    uint64_t slice_desc[] = {HORNET_TYPEDESC_SLICE, (uint64_t)"[]int", (uint64_t)elem_desc, 4};

    int32_t backing[2] = {10, 20};
    struct {
        void *ptr;
        int32_t len;
        int32_t cap;
    } slice_value = {backing, 2, 2};
    char *s = stringify_to_cstr(&slice_value, slice_desc, 0);
    CHECK_STR(s, "[]int[10, 20]");
    free(s);

    // nil/empty slice
    struct {
        void *ptr;
        int32_t len;
        int32_t cap;
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
        (uint64_t)"y", (uint64_t)int_desc, 4,
    };
    struct {
        int32_t x;
        int32_t y;
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
        (uint64_t)"y", (uint64_t)int_desc, 4,
    };
    uint64_t arr_desc[] = {HORNET_TYPEDESC_ARRAY, (uint64_t)"[2]Point", (uint64_t)point_desc, 2, 8};
    struct {
        int32_t x, y;
    } points[2] = {{1, 2}, {3, 4}};
    char *s = stringify_to_cstr(points, arr_desc, 0);
    CHECK_STR(s, "[2]Point[Point(x: 1, y: 2), Point(x: 3, y: 4)]");
    free(s);
}

static void test_struct_with_slice_field(void) {
    uint64_t int_desc[] = {HORNET_TYPEDESC_INT};
    uint64_t slice_desc[] = {HORNET_TYPEDESC_SLICE, (uint64_t)"[]int", (uint64_t)int_desc, 4};
    uint64_t box_desc[] = {
        HORNET_TYPEDESC_STRUCT, (uint64_t)"Box", 1,
        (uint64_t)"values", (uint64_t)slice_desc, 0,
    };
    int32_t backing[3] = {7, 8, 9};
    struct {
        void *ptr;
        int32_t len;
        int32_t cap;
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
        (uint64_t)"y", (uint64_t)int_desc, 4,
    };
    uint64_t wrapper_desc[] = {
        HORNET_TYPEDESC_STRUCT, (uint64_t)"Wrapper", 1,
        (uint64_t)"p", (uint64_t)point_desc, 0,
    };
    struct {
        int32_t x, y;
    } wrapper = {5, 6};
    char *s = stringify_to_cstr(&wrapper, wrapper_desc, 0);
    CHECK_STR(s, "Wrapper(p: Point(x: 5, y: 6))");
    free(s);
}

static void test_buffer_growth_stress(void) {
    // A large array forces many reallocations through the real
    // growth policy (starting from stringify_to_cstr's own
    // deliberately tiny 4-byte initial capacity) -- exactly the kind
    // of stress that would surface a corrupted-pointer or lost-byte
    // bug in the realloc/memcpy path.
    enum { N = 5000 };
    uint64_t elem_desc[] = {HORNET_TYPEDESC_INT};
    uint64_t arr_desc[] = {HORNET_TYPEDESC_ARRAY, (uint64_t)"[5000]int", (uint64_t)elem_desc, N, 4};
    static int32_t values[N];
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

    int32_t v = 123;
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
    // struct Node: int value; []Node children -- the type descriptor
    // itself contains a pointer CYCLE here (node_desc's own
    // "children" field points back at node_desc), matching exactly
    // what _get_or_build_type_descriptor builds for a genuinely
    // self-referential struct type (see its own docstring for why
    // reserving the label before recursing is what makes this
    // representable as a finite amount of static data at all). This
    // is the specific case build_stringify_function's own docstring
    // names as the reason real recursion, not per-call-site inlining,
    // was needed in the first place -- worth confirming the C port
    // walks a cyclic descriptor correctly, not just a tree-shaped one.
    uint64_t int_desc[] = {HORNET_TYPEDESC_INT};

    // children_slice_desc's own elem_desc field (index 2) is filled
    // in below, once node_desc exists to point back at -- mirroring
    // the compile-time order _get_or_build_type_descriptor builds
    // these in (reserve the label, THEN recurse).
    static uint64_t children_slice_desc[4];
    static uint64_t node_desc[6];
    children_slice_desc[0] = HORNET_TYPEDESC_SLICE;
    children_slice_desc[1] = (uint64_t)"[]Node";
    children_slice_desc[2] = (uint64_t)node_desc;
    children_slice_desc[3] = 20;  // a Node value's own byte width (4-byte int value + 16-byte slice descriptor)

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
    node_desc_full[8] = 4;  // "children" field's own byte offset (after the int value)
    children_slice_desc[2] = (uint64_t)node_desc_full;  // complete the cycle

    // __attribute__((packed)): Hornet's own struct layout is tightly
    // packed with no alignment padding at all (see _field_offset's
    // own docstring) -- an ordinary C struct would insert 4 bytes of
    // padding before children_ptr to align it, landing it at offset 8
    // rather than the offset 4 this test's own descriptor says, which
    // is exactly the mismatch that crashed here before this fix (not
    // a bug in runtime.c -- a bug in this test not reproducing
    // Hornet's own layout faithfully).
    struct __attribute__((packed)) node_value {
        int32_t value;
        void *children_ptr;
        int32_t children_len;
        int32_t children_cap;
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

int main(void) {
    test_int();
    test_int8_uint8();
    test_int64();
    test_bool();
    test_str();
    test_array();
    test_slice();
    test_struct();
    test_nested_array_of_structs();
    test_struct_with_slice_field();
    test_nested_struct_field();
    test_self_referential_struct();
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
