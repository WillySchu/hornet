"""A dict's capacity follows its live entries: inserts and deletes that leave few entries rehash
at the same capacity, and only a table that is really filling up doubles."""
import shutil
import subprocess

import pytest

from build import c_compiler, executable_name, link_flags, runtime_objects
from tests.targets import each_runnable_target, run_binary
from typesys import Type, dict_bucket_layout

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not available")

# A dict header is {buckets, count, tombstones, capacity, key offset, value offset, stride, holds pointers}
# (typesys.py's dict_bucket_layout gives the bucket layout); prints `count capacity` per case.
_PROGRAM = r"""
#include <stdint.h>
#include <stdio.h>

void hornet_dict_set_scalar_key(void *d, int64_t key_width, int64_t value_width, const void *key, const void *value);
void hornet_dict_delete_scalar_key(void *d, int64_t key_width, int64_t value_width, const void *key);
void *hornet_dict_lookup_scalar_key(void *d, int64_t key_width, int64_t value_width, const void *key);
void hornet_dict_set_str_key(void *d, int64_t value_width, const void *key, int64_t key_len, const void *value);
void hornet_dict_delete_str_key(void *d, int64_t value_width, const void *key, int64_t key_len);
void *hornet_dict_lookup_str_key(void *d, int64_t value_width, const void *key, int64_t key_len);

int main(void) {
    // Three entries stay; a hundred thousand others come and go.
    int64_t scalar[8] = {0, 0, 0, 0, 8, 16, 24, 0};  // an int key and an int value
    for (int64_t i = -3; i < 100000; i++) {
        hornet_dict_set_scalar_key(scalar, 8, 8, &i, &i);
        if (i >= 0) {
            hornet_dict_delete_scalar_key(scalar, 8, 8, &i);
        }
    }
    int64_t kept = -2;
    printf("%lld %lld %lld\n", (long long)scalar[1], (long long)scalar[3],
           (long long)*(int64_t *)hornet_dict_lookup_scalar_key(scalar, 8, 8, &kept));

    int64_t str[8] = {0, 0, 0, 0, 8, 24, 32, 1};  // a str key and an int value
    int64_t seven = 7;
    hornet_dict_set_str_key(str, 8, "keep", 4, &seven);
    for (int64_t i = 0; i < 100000; i++) {
        char key[24];
        int n = snprintf(key, sizeof key, "%lld", (long long)i);
        hornet_dict_set_str_key(str, 8, key, n, &i);
        hornet_dict_delete_str_key(str, 8, key, n);
    }
    printf("%lld %lld %lld\n", (long long)str[1], (long long)str[3],
           (long long)*(int64_t *)hornet_dict_lookup_str_key(str, 8, "keep", 4));

    // A thousand live entries still grow the table.
    int64_t full[8] = {0, 0, 0, 0, 8, 16, 24, 0};
    for (int64_t i = 0; i < 1000; i++) {
        hornet_dict_set_scalar_key(full, 8, 8, &i, &i);
    }
    printf("%lld %lld\n", (long long)full[1], (long long)full[3]);
    return 0;
}
"""


@each_runnable_target
def test_capacity_follows_live_entries(target, tmp_path):
    source, binary = tmp_path / "churn.c", tmp_path / executable_name("churn", target)
    source.write_text(_PROGRAM)
    subprocess.run([*c_compiler(target), "-std=c11", str(source), *map(str, runtime_objects(target)), "-o", str(binary),
                    *link_flags(target)],
                   check=True, capture_output=True, text=True)
    result = run_binary(target, [binary], capture_output=True, text=True)
    assert (result.returncode, result.stdout) == (0, "3 8 -2\n1 8 7\n1000 2048\n")


def test_the_layout_the_program_gives_is_typesys_s():
    assert dict_bucket_layout(Type.INT, Type.INT, {}, {}) == (8, 16, 24)
    assert dict_bucket_layout(Type.STR, Type.INT, {}, {}) == (8, 24, 32)
