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

// A bucket: a state byte, then the key and the value where the dict's header says (key_offset,
// value_offset), `stride` bytes from the next. A str key is held as its {ptr, len}.

// Insert into `buckets`, a table of `capacity` laid out as `dict` says, with room: 0 if it overwrote
// the key, 1 if the key is new, 2 if new in a tombstone's bucket. (The probe loops keep the layout
// in locals: the runtime is built without optimization, and these are the hot path.)
static int64_t dict_insert_scalar(void *buckets, int64_t capacity, const struct hornet_dict *dict,
                                  const void *key_ptr, int64_t key_width, const void *value_ptr, int64_t value_width) {
    int64_t stride = dict->stride, key_offset = dict->key_offset, value_offset = dict->value_offset;
    int64_t index = hornet_hash_bytes(key_ptr, key_width) & (capacity - 1);
    int64_t tombstone_index = -1;
    while (1) {
        unsigned char *bucket = (unsigned char *)buckets + index * stride;
        if (bucket[0] == HORNET_DICT_BUCKET_EMPTY) {
            int64_t at = tombstone_index >= 0 ? tombstone_index : index;
            unsigned char *target = (unsigned char *)buckets + at * stride;
            target[0] = HORNET_DICT_BUCKET_OCCUPIED;
            memcpy(target + key_offset, key_ptr, (size_t)key_width);
            memcpy(target + value_offset, value_ptr, (size_t)value_width);
            return (tombstone_index >= 0) ? 2 : 1;
        }
        if (bucket[0] == HORNET_DICT_BUCKET_TOMBSTONE) {
            if (tombstone_index < 0) {
                tombstone_index = index;
            }
        } else if (memcmp(bucket + key_offset, key_ptr, (size_t)key_width) == 0) {
            memcpy(bucket + value_offset, value_ptr, (size_t)value_width);
            return 0;
        }
        index = (index + 1) & (capacity - 1);
    }
}

// A str key as its bucket holds it.
struct dict_str_key {
    const void *ptr;
    int64_t len;
};

static int64_t dict_insert_str(void *buckets, int64_t capacity, const struct hornet_dict *dict,
                               const void *key_ptr, int64_t key_len, const void *value_ptr, int64_t value_width) {
    int64_t stride = dict->stride, key_offset = dict->key_offset, value_offset = dict->value_offset;
    int64_t index = hornet_hash_bytes(key_ptr, key_len) & (capacity - 1);
    int64_t tombstone_index = -1;
    while (1) {
        unsigned char *bucket = (unsigned char *)buckets + index * stride;
        if (bucket[0] == HORNET_DICT_BUCKET_EMPTY) {
            int64_t at = tombstone_index >= 0 ? tombstone_index : index;
            unsigned char *target = (unsigned char *)buckets + at * stride;
            struct dict_str_key key = {key_ptr, key_len};
            target[0] = HORNET_DICT_BUCKET_OCCUPIED;
            memcpy(target + key_offset, &key, sizeof key);
            memcpy(target + value_offset, value_ptr, (size_t)value_width);
            return (tombstone_index >= 0) ? 2 : 1;
        }
        if (bucket[0] == HORNET_DICT_BUCKET_TOMBSTONE) {
            if (tombstone_index < 0) {
                tombstone_index = index;
            }
        } else {
            struct dict_str_key stored;
            memcpy(&stored, bucket + key_offset, sizeof stored);
            if (stored.len == key_len && memcmp(stored.ptr, key_ptr, (size_t)key_len) == 0) {
                memcpy(bucket + value_offset, value_ptr, (size_t)value_width);
                return 0;
            }
        }
        index = (index + 1) & (capacity - 1);
    }
}

// Insert into a dict whose table has room (a literal's): as dict_insert_scalar. The caller counts.
int64_t hornet_dict_insert_scalar_key(void *descriptor, const void *key_ptr, int64_t key_width,
                                      const void *value_ptr, int64_t value_width) {
    struct hornet_dict *dict = descriptor;
    return dict_insert_scalar(dict->buckets, dict->capacity, dict, key_ptr, key_width, value_ptr, value_width);
}

int64_t hornet_dict_insert_str_key(void *descriptor, const void *key_ptr, int64_t key_len,
                                   const void *value_ptr, int64_t value_width) {
    struct hornet_dict *dict = descriptor;
    return dict_insert_str(dict->buckets, dict->capacity, dict, key_ptr, key_len, value_ptr, value_width);
}

// A rehash's capacity: 8 for a dict's first write; unchanged when at most half the buckets would
// be live (tombstones filled the table); otherwise doubled.
static int64_t dict_rehash_capacity(int64_t capacity, int64_t count) {
    if (capacity == 0) {
        return 8;
    }
    return (count + 1) * 2 <= capacity ? capacity : capacity * 2;
}

// Rehash the live buckets into a new table once (count + tombstones + 1) exceeds 75%. A str key is
// given as -1 for `key_width`.
static void dict_grow_if_needed(struct hornet_dict *dict, int64_t key_width, int64_t value_width) {
    if ((dict->count + dict->tombstones + 1) * 4 <= dict->capacity * 3) {
        return;
    }
    int64_t new_capacity = dict_rehash_capacity(dict->capacity, dict->count);
    void *new_buckets = hornet_alloc_zeroed(new_capacity, dict->stride);
    for (int64_t i = 0; i < dict->capacity; i++) {
        unsigned char *bucket = (unsigned char *)dict->buckets + i * dict->stride;
        if (bucket[0] != HORNET_DICT_BUCKET_OCCUPIED) {
            continue;
        }
        if (key_width < 0) {
            struct dict_str_key key;
            memcpy(&key, bucket + dict->key_offset, sizeof key);
            dict_insert_str(new_buckets, new_capacity, dict, key.ptr, key.len, bucket + dict->value_offset,
                            value_width);
        } else {
            dict_insert_scalar(new_buckets, new_capacity, dict, bucket + dict->key_offset, key_width,
                               bucket + dict->value_offset, value_width);
        }
    }
    dict->buckets = new_buckets;
    dict->capacity = new_capacity;
    dict->tombstones = 0;
}

// What an insert into a dict's own table did to its count and tombstones.
static void dict_note_insert(struct hornet_dict *dict, int64_t result) {
    if (result != 0) {
        dict->count += 1;
    }
    if (result == 2) {
        dict->tombstones -= 1;
    }
}

// `d[k] = v`: grow if needed, then insert.
void hornet_dict_set_scalar_key(
    void *descriptor, int64_t key_width, int64_t value_width,
    const void *key_ptr, const void *value_ptr
) {
    struct hornet_dict *dict = descriptor;
    dict_grow_if_needed(dict, key_width, value_width);
    dict_note_insert(dict, dict_insert_scalar(dict->buckets, dict->capacity, dict, key_ptr, key_width, value_ptr,
                                              value_width));
}

void hornet_dict_set_str_key(
    void *descriptor, int64_t value_width,
    const void *key_ptr, int64_t key_len, const void *value_ptr
) {
    struct hornet_dict *dict = descriptor;
    dict_grow_if_needed(dict, -1, value_width);
    dict_note_insert(dict, dict_insert_str(dict->buckets, dict->capacity, dict, key_ptr, key_len, value_ptr,
                                           value_width));
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

// The bucket holding the key, or NULL. (The probe loops keep the layout in locals: the runtime is
// built without optimization, and these are the hot path.)
static unsigned char *dict_find_scalar(const struct hornet_dict *dict, int64_t key_width, const void *key_ptr) {
    int64_t capacity = dict->capacity;
    if (capacity == 0) {
        return NULL;  // no table yet: capacity 0 would break the mask
    }
    unsigned char *buckets = dict->buckets;
    int64_t stride = dict->stride, key_offset = dict->key_offset;
    int64_t index = hornet_hash_bytes(key_ptr, key_width) & (capacity - 1);
    while (1) {
        unsigned char *bucket = buckets + index * stride;
        if (bucket[0] == HORNET_DICT_BUCKET_EMPTY) {
            return NULL;
        }
        if (bucket[0] == HORNET_DICT_BUCKET_OCCUPIED && memcmp(bucket + key_offset, key_ptr, (size_t)key_width) == 0) {
            return bucket;
        }
        index = (index + 1) & (capacity - 1);
    }
}

static unsigned char *dict_find_str(const struct hornet_dict *dict, const void *key_ptr, int64_t key_len) {
    int64_t capacity = dict->capacity;
    if (capacity == 0) {
        return NULL;
    }
    unsigned char *buckets = dict->buckets;
    int64_t stride = dict->stride, key_offset = dict->key_offset;
    int64_t index = hornet_hash_bytes(key_ptr, key_len) & (capacity - 1);
    while (1) {
        unsigned char *bucket = buckets + index * stride;
        if (bucket[0] == HORNET_DICT_BUCKET_EMPTY) {
            return NULL;
        }
        if (bucket[0] == HORNET_DICT_BUCKET_OCCUPIED) {
            struct dict_str_key stored;
            memcpy(&stored, bucket + key_offset, sizeof stored);
            if (stored.len == key_len && memcmp(stored.ptr, key_ptr, (size_t)key_len) == 0) {
                return bucket;
            }
        }
        index = (index + 1) & (capacity - 1);
    }
}

void *hornet_dict_lookup_scalar_key(void *descriptor, int64_t key_width, int64_t value_width, const void *key_ptr) {
    (void)value_width;
    const struct hornet_dict *dict = descriptor;
    unsigned char *bucket = dict_find_scalar(dict, key_width, key_ptr);
    if (bucket == NULL) {
        hornet_note_missing_key(key_ptr);
        return NULL;
    }
    return bucket + dict->value_offset;
}

void *hornet_dict_lookup_str_key(void *descriptor, int64_t value_width, const void *key_ptr, int64_t key_len) {
    (void)value_width;
    const struct hornet_dict *dict = descriptor;
    unsigned char *bucket = dict_find_str(dict, key_ptr, key_len);
    if (bucket == NULL) {
        hornet_note_missing_str_key(key_ptr, key_len);
        return NULL;
    }
    return bucket + dict->value_offset;
}

// `k in d`.
int64_t hornet_dict_contains_scalar_key(void *descriptor, int64_t key_width, int64_t value_width, const void *key_ptr) {
    (void)value_width;
    return dict_find_scalar(descriptor, key_width, key_ptr) != NULL;
}

int64_t hornet_dict_contains_str_key(void *descriptor, int64_t value_width, const void *key_ptr, int64_t key_len) {
    (void)value_width;
    return dict_find_str(descriptor, key_ptr, key_len) != NULL;
}

// `del(d, k)`: mark a tombstone and return 1, or return 0 if the key is missing (the compiled code
// panics, with the source position).
static int64_t dict_delete(struct hornet_dict *dict, unsigned char *bucket) {
    if (bucket == NULL) {
        return 0;
    }
    bucket[0] = HORNET_DICT_BUCKET_TOMBSTONE;
    dict->count -= 1;
    dict->tombstones += 1;
    return 1;
}

int64_t hornet_dict_delete_scalar_key(void *descriptor, int64_t key_width, int64_t value_width, const void *key_ptr) {
    (void)value_width;
    int64_t deleted = dict_delete(descriptor, dict_find_scalar(descriptor, key_width, key_ptr));
    if (!deleted) {
        hornet_note_missing_key(key_ptr);
    }
    return deleted;
}

int64_t hornet_dict_delete_str_key(void *descriptor, int64_t value_width, const void *key_ptr, int64_t key_len) {
    (void)value_width;
    int64_t deleted = dict_delete(descriptor, dict_find_str(descriptor, key_ptr, key_len));
    if (!deleted) {
        hornet_note_missing_str_key(key_ptr, key_len);
    }
    return deleted;
}
