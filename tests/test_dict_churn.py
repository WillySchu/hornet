"""Dicts whose entries come and go: the table is rehashed at the same capacity (see
tests/runtime/test_dict_capacity.py), which keeps its entries and still ends a `for ... in` over it."""

from tests.test_compiler import GCC_SKIP, assert_program_panics, assert_program_stdout


@GCC_SKIP
def test_entries_survive_inserts_and_deletes_of_other_keys():
    assert_program_stdout(
        "def int main():\n"
        "    dict[int]int d\n"
        "    dict[str]int s\n"
        "    d[-1] = 10\n"
        "    d[-2] = 20\n"
        "    s['keep'] = 7\n"
        "    for int i = 0; i < 1000; i += 1:\n"
        "        d[i] = i\n"
        "        del(d, i)\n"
        "        str k = str(byte(i % 90 + 33))\n"
        "        s[k] = i\n"
        "        del(s, k)\n"
        "    int total = 0\n"
        "    for k, v in d:\n"
        "        total += k + v\n"
        "    print(len(d))\n"
        "    print(total)\n"
        "    print(-1 in d and -2 in d)\n"
        "    print(999 in d)\n"
        "    print(len(s))\n"
        "    print(s['keep'])\n"
        "    print('!' in s)\n"
        "    return 0\n",
        "2\n27\ntrue\nfalse\n1\n7\nfalse\n",
    )


@GCC_SKIP
def test_a_rehash_at_the_same_capacity_ends_iteration():
    assert_program_panics(
        "def int main():\n"
        "    dict[int]int d\n"
        "    d[-1] = 1\n"
        "    d[-2] = 2\n"
        "    d[-3] = 3\n"
        "    for k in d:\n"
        "        for int i = 0; i < 8; i += 1:\n"
        "            d[i] = i\n"
        "            del(d, i)\n"
        "    return 0\n",
        "for ... in: dict's own buckets were reallocated (e.g. by an insert that triggered growth) during iteration",
    )
