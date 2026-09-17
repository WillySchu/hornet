// Verifies hornet_print works correctly when runtime.c is compiled
// and linked as a genuinely SEPARATE object file, called via an
// ordinary extern declaration -- unlike test_runtime_isolated.c,
// which #includes runtime.c directly to reach its static internals
// for white-box testing. This is closer to how the real compiler
// will eventually call it: an external `call hornet_print`, nothing
// about runtime.c's own internal implementation visible or assumed.
#include <stdint.h>
#include <string.h>

extern void hornet_print(void *value_addr, const unsigned char *type_desc);

// Matches HORNET_TYPEDESC_INT's own value directly (0) rather than
// including hornet_typedesc_tags.h, specifically to confirm this
// test exercises hornet_print as an opaque, externally-linked
// function -- not by reaching into runtime.c's own build artifacts.
#define TYPEDESC_INT 0

int main(void) {
    int32_t value = 42;
    unsigned char desc[8];
    uint64_t tag = TYPEDESC_INT;
    memcpy(desc, &tag, sizeof(tag));
    hornet_print(&value, desc);
    return 0;
}
