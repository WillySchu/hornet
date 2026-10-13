// hornet_print called the way compiled programs call it: the runtime as build.py builds it, linked
// with this, behind an extern declaration. (test_runtime_isolated.c includes internal.h instead,
// to reach what the runtime's files share.)
#include <stdint.h>
#include <string.h>

extern void hornet_print(void *value_addr, const unsigned char *type_desc);

// Matches HORNET_TYPEDESC_INT's own value directly (0) rather than
// including hornet_typedesc_tags.h, specifically to confirm this
// test exercises hornet_print as an opaque, externally-linked
// function -- not by reaching into the runtime's own headers.
#define TYPEDESC_INT 0

int main(void) {
    int64_t value = 42;
    unsigned char desc[8];
    uint64_t tag = TYPEDESC_INT;
    memcpy(desc, &tag, sizeof(tag));
    hornet_print(&value, desc);
    return 0;
}
