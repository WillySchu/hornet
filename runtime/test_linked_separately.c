// hornet_print called the way compiled programs call it: runtime.c compiled and linked as a
// separate object, behind an extern declaration. (test_runtime_isolated.c #includes runtime.c
// instead, to reach its static functions.)
#include <stdint.h>
#include <string.h>

extern void hornet_print(void *value_addr, const unsigned char *type_desc);

// Matches HORNET_TYPEDESC_INT's own value directly (0) rather than
// including hornet_typedesc_tags.h, specifically to confirm this
// test exercises hornet_print as an opaque, externally-linked
// function -- not by reaching into runtime.c's own build artifacts.
#define TYPEDESC_INT 0

int main(void) {
    int64_t value = 42;
    unsigned char desc[8];
    uint64_t tag = TYPEDESC_INT;
    memcpy(desc, &tag, sizeof(tag));
    hornet_print(&value, desc);
    return 0;
}
