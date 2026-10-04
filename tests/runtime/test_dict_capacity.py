"""A dict's capacity follows its live entries: inserts and deletes that leave few entries rehash
at the same capacity, and only a table that is really filling up doubles."""
import shutil
import subprocess

import pytest

from build import c_compiler, executable_name, runtime_object
from tests.targets import each_runnable_target, run_binary

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="gcc not available")

# A dict header is {buckets, count, tombstones, capacity}; prints `count capacity` per case.
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
    int64_t scalar[4] = {0};
    for (int64_t i = -3; i < 100000; i++) {
        hornet_dict_set_scalar_key(scalar, 8, 8, &i, &i);
        if (i >= 0) {
            hornet_dict_delete_scalar_key(scalar, 8, 8, &i);
        }
    }
    int64_t kept = -2;
    printf("%lld %lld %lld\n", (long long)scalar[1], (long long)scalar[3],
           (long long)*(int64_t *)hornet_dict_lookup_scalar_key(scalar, 8, 8, &kept));

    int64_t str[4] = {0};
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
    int64_t full[4] = {0};
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
    subprocess.run([*c_compiler(target), "-std=c11", str(source), str(runtime_object(target)), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    result = run_binary(target, [binary], capture_output=True, text=True)
    assert (result.returncode, result.stdout) == (0, "3 8 -2\n1 8 7\n1000 2048\n")
