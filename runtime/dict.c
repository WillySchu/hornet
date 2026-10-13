// Dictionaries: hash tables, open addressing with linear probing.
#include "internal.h"

// FNV-1a 64 over bytes. Scalar keys hash their value bytes; str keys hash their content.
int64_t hornet_hash_bytes(const void *ptr, int64_t len) {
    uint64_t h = 0xcbf29ce484222325ULL;  // FNV offset basis
    const unsigned char *bytes = (const unsigned char *)ptr;
    for (int64_t i = 0; i < len; i++) {
        h ^= bytes[i];
        h *= 0x100000001b3ULL;  // FNV prime
    }
    return (int64_t)h;
}

// Insert into a table with room (literals, and after any rehash): returns 0 if it overwrote the
// key, 1 if the key is new, 2 if new in a tombstone's bucket.
int64_t hornet_dict_insert_scalar_key(
    void *buckets, int64_t capacity, int64_t bucket_stride,
    const void *key_ptr, int64_t key_width,
    const void *value_ptr, int64_t value_width
) {
    int64_t index = hornet_hash_bytes(key_ptr, key_width) & (capacity - 1);
    int64_t tombstone_index = -1;
    while (1) {
        unsigned char *bucket = (unsigned char *)buckets + index * bucket_stride;
        if (bucket[0] == HORNET_DICT_BUCKET_EMPTY) {
            int64_t target = (tombstone_index >= 0) ? tombstone_index : index;
            unsigned char *target_bucket = (unsigned char *)buckets + target * bucket_stride;
            target_bucket[0] = HORNET_DICT_BUCKET_OCCUPIED;
            memcpy(target_bucket + 1, key_ptr, (size_t)key_width);
            memcpy(target_bucket + 1 + key_width, value_ptr, (size_t)value_width);
            return (tombstone_index >= 0) ? 2 : 1;
        }
        if (bucket[0] == HORNET_DICT_BUCKET_TOMBSTONE) {
            if (tombstone_index < 0) {
                tombstone_index = index;
            }
            index = (index + 1) & (capacity - 1);
            continue;
        }
        if (memcmp(bucket + 1, key_ptr, (size_t)key_width) == 0) {
            memcpy(bucket + 1 + key_width, value_ptr, (size_t)value_width);
            return 0;
        }
        index = (index + 1) & (capacity - 1);
    }
}

// Str-keyed insert; the bucket stores the key's {ptr, len}.
int64_t hornet_dict_insert_str_key(
    void *buckets, int64_t capacity, int64_t bucket_stride,
    const void *key_ptr, int64_t key_len,
    const void *value_ptr, int64_t value_width
) {
    // type_byte_width(str)
    const int64_t key_region_width = 16;
    int64_t index = hornet_hash_bytes(key_ptr, key_len) & (capacity - 1);
    int64_t tombstone_index = -1;
    while (1) {
        unsigned char *bucket = (unsigned char *)buckets + index * bucket_stride;
        if (bucket[0] == HORNET_DICT_BUCKET_EMPTY) {
            int64_t target = (tombstone_index >= 0) ? tombstone_index : index;
            unsigned char *target_bucket = (unsigned char *)buckets + target * bucket_stride;
            target_bucket[0] = HORNET_DICT_BUCKET_OCCUPIED;
            *(void **)(target_bucket + 1) = (void *)key_ptr;
            *(int64_t *)(target_bucket + 1 + sizeof(void *)) = key_len;
            memcpy(target_bucket + 1 + key_region_width, value_ptr, (size_t)value_width);
            return (tombstone_index >= 0) ? 2 : 1;
        }
        if (bucket[0] == HORNET_DICT_BUCKET_TOMBSTONE) {
            if (tombstone_index < 0) {
                tombstone_index = index;
            }
            index = (index + 1) & (capacity - 1);
            continue;
        }
        void *stored_ptr = *(void **)(bucket + 1);
        int64_t stored_len = read_i64(bucket + 1 + sizeof(void *));
        if (stored_len == key_len && memcmp(stored_ptr, key_ptr, (size_t)key_len) == 0) {
            memcpy(bucket + 1 + key_region_width, value_ptr, (size_t)value_width);
            return 0;
        }
        index = (index + 1) & (capacity - 1);
    }
}

// Descriptor fields (8 bytes each): buckets @0, count @8, tombstones @16, capacity @24.
static void *dict_buckets(void *descriptor) { return read_ptr(descriptor); }
static int64_t dict_capacity(void *descriptor) { return read_i64((char *)descriptor + 24); }

static void dict_set_buckets(void *descriptor, void *buckets) {
    *(void **)descriptor = buckets;
}
static void dict_set_capacity(void *descriptor, int64_t capacity) {
    *(int64_t *)((char *)descriptor + 24) = capacity;
}
static void dict_bump_count(void *descriptor) {
    int64_t *count = (int64_t *)((char *)descriptor + 8);
    *count += 1;
}

static int64_t dict_tombstones(void *descriptor) { return read_i64((char *)descriptor + 16); }
static void dict_set_tombstones(void *descriptor, int64_t tombstones) {
    *(int64_t *)((char *)descriptor + 16) = tombstones;
}
static void dict_bump_tombstones(void *descriptor, int64_t delta) {
    dict_set_tombstones(descriptor, dict_tombstones(descriptor) + delta);
}

// A rehash's capacity: 8 for a dict's first write; unchanged when at most half the buckets would
// be live (tombstones filled the table); otherwise doubled.
static int64_t dict_rehash_capacity(int64_t capacity, int64_t count) {
    if (capacity == 0) {
        return 8;
    }
    return (count + 1) * 2 <= capacity ? capacity : capacity * 2;
}

// Rehash the live buckets into a new table once (count + tombstones + 1) exceeds 75%.
static void dict_grow_scalar_key_if_needed(void *descriptor, int64_t key_width, int64_t value_width) {
    int64_t capacity = dict_capacity(descriptor);
    int64_t count = read_i64((char *)descriptor + 8);
    int64_t tombstones = dict_tombstones(descriptor);
    if ((count + tombstones + 1) * 4 <= capacity * 3) {
        return;
    }
    int64_t bucket_stride = 1 + key_width + value_width;
    int64_t new_capacity = dict_rehash_capacity(capacity, count);
    void *new_buckets = hornet_alloc_zeroed(new_capacity, bucket_stride);
    void *old_buckets = dict_buckets(descriptor);
    for (int64_t i = 0; i < capacity; i++) {
        unsigned char *bucket = (unsigned char *)old_buckets + i * bucket_stride;
        if (bucket[0] != HORNET_DICT_BUCKET_OCCUPIED) {
            continue;
        }
        hornet_dict_insert_scalar_key(
            new_buckets, new_capacity, bucket_stride,
            bucket + 1, key_width, bucket + 1 + key_width, value_width);
    }
    dict_set_buckets(descriptor, new_buckets);
    dict_set_capacity(descriptor, new_capacity);
    dict_set_tombstones(descriptor, 0);
}

static void dict_grow_str_key_if_needed(void *descriptor, int64_t value_width) {
    const int64_t key_region_width = 16;  // type_byte_width(str)
    int64_t capacity = dict_capacity(descriptor);
    int64_t count = read_i64((char *)descriptor + 8);
    int64_t tombstones = dict_tombstones(descriptor);
    if ((count + tombstones + 1) * 4 <= capacity * 3) {
        return;
    }
    int64_t bucket_stride = 1 + key_region_width + value_width;
    int64_t new_capacity = dict_rehash_capacity(capacity, count);
    void *new_buckets = hornet_alloc_zeroed(new_capacity, bucket_stride);
    void *old_buckets = dict_buckets(descriptor);
    for (int64_t i = 0; i < capacity; i++) {
        unsigned char *bucket = (unsigned char *)old_buckets + i * bucket_stride;
        if (bucket[0] != HORNET_DICT_BUCKET_OCCUPIED) {
            continue;
        }
        void *stored_ptr = *(void **)(bucket + 1);
        int64_t stored_len = read_i64(bucket + 1 + sizeof(void *));
        hornet_dict_insert_str_key(
            new_buckets, new_capacity, bucket_stride,
            stored_ptr, stored_len, bucket + 1 + key_region_width, value_width);
    }
    dict_set_buckets(descriptor, new_buckets);
    dict_set_capacity(descriptor, new_capacity);
    dict_set_tombstones(descriptor, 0);
}

// `d[k] = v`: grow if needed, then insert.
void hornet_dict_set_scalar_key(
    void *descriptor, int64_t key_width, int64_t value_width,
    const void *key_ptr, const void *value_ptr
) {
    dict_grow_scalar_key_if_needed(descriptor, key_width, value_width);
    int64_t bucket_stride = 1 + key_width + value_width;
    int result = hornet_dict_insert_scalar_key(
        dict_buckets(descriptor), dict_capacity(descriptor), bucket_stride,
        key_ptr, key_width, value_ptr, value_width);
    if (result != 0) {
        dict_bump_count(descriptor);
    }
    if (result == 2) {
        dict_bump_tombstones(descriptor, -1);
    }
}

void hornet_dict_set_str_key(
    void *descriptor, int64_t value_width,
    const void *key_ptr, int64_t key_len, const void *value_ptr
) {
    dict_grow_str_key_if_needed(descriptor, value_width);
    const int64_t key_region_width = 16;
    int64_t bucket_stride = 1 + key_region_width + value_width;
    int result = hornet_dict_insert_str_key(
        dict_buckets(descriptor), dict_capacity(descriptor), bucket_stride,
        key_ptr, key_len, value_ptr, value_width);
    if (result != 0) {
        dict_bump_count(descriptor);
    }
    if (result == 2) {
        dict_bump_tombstones(descriptor, -1);
    }
}

// `d[k]`: address of the value, or NULL if the key is missing (the compiled code panics, with the
// source position). Probes past tombstones.
// The key a lookup or a delete has just failed to find. The compiled code panics as soon as one
// returns "not found" (`k in d` is a different call), and the panic prints the key from here: so
// finding a key costs nothing for it.
static const void *hornet_missing_key;
static struct {
    const void *ptr;
    int64_t len;
} hornet_missing_str_key;

static void hornet_note_missing_key(const void *key_ptr) {
    hornet_missing_key = key_ptr;  // (in the caller's frame, which is still there for the panic)
}

static void hornet_note_missing_str_key(const void *key_ptr, int64_t key_len) {
    hornet_missing_str_key.ptr = key_ptr;
    hornet_missing_str_key.len = key_len;
    hornet_missing_key = &hornet_missing_str_key;
}

// How much of a str key a panic shows.
#define HORNET_PANIC_KEY_BYTES 100

// `msg: key`, the key as print shows one inside a value: `d[k]` or `del(d, k)` without the key.
void hornet_panic_missing_key(const char *msg, const unsigned char *key_desc) {
    struct hornet_buf buf;
    buf.cap = 64;
    buf.ptr = hornet_alloc(buf.cap);
    buf.len = 0;
    hornet_buf_append_cstr(&buf, msg);
    hornet_buf_append_cstr(&buf, ": ");
    if (read_desc_word(key_desc, 0) == HORNET_TYPEDESC_STR && hornet_missing_str_key.len > HORNET_PANIC_KEY_BYTES) {
        char rest[48];
        hornet_buf_append_byte(&buf, '\'');
        hornet_buf_append_bytes(&buf, hornet_missing_str_key.ptr, HORNET_PANIC_KEY_BYTES);
        int n = snprintf(rest, sizeof(rest), "...' (%lld bytes)", (long long)hornet_missing_str_key.len);
        hornet_buf_append_bytes(&buf, rest, n);
    } else {
        hornet__stringify((void *)hornet_missing_key, key_desc, 1, &buf);
    }
    hornet_buf_append_byte(&buf, '\n');
    fflush(stdout);  // the program's own output first
    fwrite(buf.ptr, 1, (size_t)buf.len, stderr);
    fflush(stderr);
    abort();
}

void *hornet_dict_lookup_scalar_key(void *descriptor, int64_t key_width, int64_t value_width, const void *key_ptr) {
    void *buckets = dict_buckets(descriptor);
    int64_t capacity = dict_capacity(descriptor);
    if (capacity == 0) {
        // no table yet: capacity 0 would break the mask.
        hornet_note_missing_key(key_ptr);
        return NULL;
    }
    int64_t bucket_stride = 1 + key_width + value_width;
    int64_t index = hornet_hash_bytes(key_ptr, key_width) & (capacity - 1);
    while (1) {
        unsigned char *bucket = (unsigned char *)buckets + index * bucket_stride;
        if (bucket[0] == HORNET_DICT_BUCKET_EMPTY) {
            hornet_note_missing_key(key_ptr);
            return NULL;
        }
        if (bucket[0] == HORNET_DICT_BUCKET_OCCUPIED && memcmp(bucket + 1, key_ptr, (size_t)key_width) == 0) {
            return bucket + 1 + key_width;
        }
        index = (index + 1) & (capacity - 1);
    }
}

void *hornet_dict_lookup_str_key(void *descriptor, int64_t value_width, const void *key_ptr, int64_t key_len) {
    const int64_t key_region_width = 16;
    void *buckets = dict_buckets(descriptor);
    int64_t capacity = dict_capacity(descriptor);
    if (capacity == 0) {
        // no table yet
        hornet_note_missing_str_key(key_ptr, key_len);
        return NULL;
    }
    int64_t bucket_stride = 1 + key_region_width + value_width;
    int64_t index = hornet_hash_bytes(key_ptr, key_len) & (capacity - 1);
    while (1) {
        unsigned char *bucket = (unsigned char *)buckets + index * bucket_stride;
        if (bucket[0] == HORNET_DICT_BUCKET_EMPTY) {
            hornet_note_missing_str_key(key_ptr, key_len);
            return NULL;
        }
        if (bucket[0] == HORNET_DICT_BUCKET_OCCUPIED) {
            void *stored_ptr = *(void **)(bucket + 1);
            int64_t stored_len = read_i64(bucket + 1 + sizeof(void *));
            if (stored_len == key_len && memcmp(stored_ptr, key_ptr, (size_t)key_len) == 0) {
                return bucket + 1 + key_region_width;
            }
        }
        index = (index + 1) & (capacity - 1);
    }
}

// `k in d`.
int64_t hornet_dict_contains_scalar_key(void *descriptor, int64_t key_width, int64_t value_width, const void *key_ptr) {
    void *buckets = dict_buckets(descriptor);
    int64_t capacity = dict_capacity(descriptor);
    if (capacity == 0) {
        // no table yet
        return 0;
    }
    int64_t bucket_stride = 1 + key_width + value_width;
    int64_t index = hornet_hash_bytes(key_ptr, key_width) & (capacity - 1);
    while (1) {
        unsigned char *bucket = (unsigned char *)buckets + index * bucket_stride;
        if (bucket[0] == HORNET_DICT_BUCKET_EMPTY) {
            return 0;
        }
        if (bucket[0] == HORNET_DICT_BUCKET_OCCUPIED && memcmp(bucket + 1, key_ptr, (size_t)key_width) == 0) {
            return 1;
        }
        index = (index + 1) & (capacity - 1);
    }
}

int64_t hornet_dict_contains_str_key(void *descriptor, int64_t value_width, const void *key_ptr, int64_t key_len) {
    const int64_t key_region_width = 16;
    void *buckets = dict_buckets(descriptor);
    int64_t capacity = dict_capacity(descriptor);
    if (capacity == 0) {
        // no table yet
        return 0;
    }
    int64_t bucket_stride = 1 + key_region_width + value_width;
    int64_t index = hornet_hash_bytes(key_ptr, key_len) & (capacity - 1);
    while (1) {
        unsigned char *bucket = (unsigned char *)buckets + index * bucket_stride;
        if (bucket[0] == HORNET_DICT_BUCKET_EMPTY) {
            return 0;
        }
        if (bucket[0] == HORNET_DICT_BUCKET_OCCUPIED) {
            void *stored_ptr = *(void **)(bucket + 1);
            int64_t stored_len = read_i64(bucket + 1 + sizeof(void *));
            if (stored_len == key_len && memcmp(stored_ptr, key_ptr, (size_t)key_len) == 0) {
                return 1;
            }
        }
        index = (index + 1) & (capacity - 1);
    }
}

// `del(d, k)`: mark a tombstone and return 1, or return 0 if the key is missing (the compiled code
// panics, with the source position).
int64_t hornet_dict_delete_scalar_key(void *descriptor, int64_t key_width, int64_t value_width, const void *key_ptr) {
    void *buckets = dict_buckets(descriptor);
    int64_t capacity = dict_capacity(descriptor);
    if (capacity == 0) {
        // no table yet
        hornet_note_missing_key(key_ptr);
        return 0;
    }
    int64_t bucket_stride = 1 + key_width + value_width;
    int64_t index = hornet_hash_bytes(key_ptr, key_width) & (capacity - 1);
    while (1) {
        unsigned char *bucket = (unsigned char *)buckets + index * bucket_stride;
        if (bucket[0] == HORNET_DICT_BUCKET_EMPTY) {
            hornet_note_missing_key(key_ptr);
            return 0;
        }
        if (bucket[0] == HORNET_DICT_BUCKET_OCCUPIED && memcmp(bucket + 1, key_ptr, (size_t)key_width) == 0) {
            bucket[0] = HORNET_DICT_BUCKET_TOMBSTONE;
            int64_t *count = (int64_t *)((char *)descriptor + 8);
            *count -= 1;
            dict_bump_tombstones(descriptor, 1);
            return 1;
        }
        index = (index + 1) & (capacity - 1);
    }
}

int64_t hornet_dict_delete_str_key(void *descriptor, int64_t value_width, const void *key_ptr, int64_t key_len) {
    const int64_t key_region_width = 16;
    void *buckets = dict_buckets(descriptor);
    int64_t capacity = dict_capacity(descriptor);
    if (capacity == 0) {
        // no table yet
        hornet_note_missing_str_key(key_ptr, key_len);
        return 0;
    }
    int64_t bucket_stride = 1 + key_region_width + value_width;
    int64_t index = hornet_hash_bytes(key_ptr, key_len) & (capacity - 1);
    while (1) {
        unsigned char *bucket = (unsigned char *)buckets + index * bucket_stride;
        if (bucket[0] == HORNET_DICT_BUCKET_EMPTY) {
            hornet_note_missing_str_key(key_ptr, key_len);
            return 0;
        }
        if (bucket[0] == HORNET_DICT_BUCKET_OCCUPIED) {
            void *stored_ptr = *(void **)(bucket + 1);
            int64_t stored_len = read_i64(bucket + 1 + sizeof(void *));
            if (stored_len == key_len && memcmp(stored_ptr, key_ptr, (size_t)key_len) == 0) {
                bucket[0] = HORNET_DICT_BUCKET_TOMBSTONE;
                int64_t *count = (int64_t *)((char *)descriptor + 8);
                *count -= 1;
                dict_bump_tombstones(descriptor, 1);
                return 1;
            }
        }
        index = (index + 1) & (capacity - 1);
    }
}
