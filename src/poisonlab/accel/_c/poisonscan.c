#include "poisonscan.h"

#include <stdlib.h>
#include <string.h>

#define FNV_OFFSET 0xCBF29CE484222325ULL
#define FNV_PRIME 0x100000001B3ULL
#define GOLDEN 0x9E3779B97F4A7C15ULL
#define MAX_GRAM_TABLE (1u << 22)
#define MAX_LOCAL_TABLE (1u << 22)
#define MAX_NGRAM_SIZE 8
#define MAX_BUCKETS (1 << 30)

typedef struct {
    uint64_t key;
    int32_t used;
    int32_t n;
    int32_t count;
    int32_t target_count;
    int32_t doc_count;
    int32_t last_doc;
    int32_t first_doc;
    int32_t first_pos;
} GramEntry;

typedef struct {
    uint64_t key;
    int32_t stamp;
    int32_t index;
    float value;
} LocalEntry;

static uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static uint32_t next_pow2(uint32_t value) {
    uint32_t result = 8;
    while (result < value && result < (1u << 31)) {
        result <<= 1;
    }
    return result;
}

static int is_token_byte(uint8_t c) {
    return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '\'' || c >= 0x80;
}

static int32_t tokenize_doc(const uint8_t *doc, int64_t length, uint64_t **buffer,
                            int64_t *capacity) {
    int64_t count = 0;
    int64_t index = 0;
    while (index < length) {
        uint8_t c = doc[index];
        if (c >= 'A' && c <= 'Z') {
            c = (uint8_t)(c + 32);
        }
        if (!is_token_byte(c)) {
            index++;
            continue;
        }
        uint64_t hash = FNV_OFFSET;
        while (index < length) {
            uint8_t current = doc[index];
            if (current >= 'A' && current <= 'Z') {
                current = (uint8_t)(current + 32);
            }
            if (!is_token_byte(current)) {
                break;
            }
            hash ^= (uint64_t)current;
            hash *= FNV_PRIME;
            index++;
        }
        if (count >= *capacity) {
            int64_t grown = (*capacity) * 2;
            uint64_t *resized = (uint64_t *)realloc(*buffer, (size_t)grown * sizeof(uint64_t));
            if (resized == NULL) {
                return PLSC_ERR_ALLOC;
            }
            *buffer = resized;
            *capacity = grown;
        }
        (*buffer)[count++] = hash;
    }
    return (int32_t)count;
}

static uint64_t gram_hash(const uint64_t *hashes, int32_t n) {
    uint64_t value = FNV_OFFSET ^ ((uint64_t)n * GOLDEN);
    for (int32_t i = 0; i < n; i++) {
        value ^= hashes[i];
        value *= FNV_PRIME;
    }
    return value;
}

int32_t plsc_abi_version(void) { return PLSC_ABI_VERSION; }

int32_t plsc_token_count(const uint8_t *blob, const int64_t *offsets, int32_t n_docs,
                         int32_t *out_counts) {
    if (blob == NULL || offsets == NULL || out_counts == NULL || n_docs < 0) {
        return PLSC_ERR_ARGS;
    }
    int64_t capacity = 256;
    uint64_t *tokens = (uint64_t *)malloc((size_t)capacity * sizeof(uint64_t));
    if (tokens == NULL) {
        return PLSC_ERR_ALLOC;
    }
    for (int32_t doc = 0; doc < n_docs; doc++) {
        int32_t count = tokenize_doc(blob + offsets[doc], offsets[doc + 1] - offsets[doc], &tokens,
                                     &capacity);
        if (count < 0) {
            free(tokens);
            return count;
        }
        out_counts[doc] = count;
    }
    free(tokens);
    return n_docs;
}

int32_t plsc_featurize(const uint8_t *blob, const int64_t *offsets, int32_t n_docs, int32_t max_n,
                       int32_t buckets, int32_t *out_index, float *out_value, int32_t *out_doc_start,
                       int32_t out_cap) {
    if (blob == NULL || offsets == NULL || out_index == NULL || out_value == NULL ||
        out_doc_start == NULL || n_docs < 0 || max_n < 1 || max_n > MAX_NGRAM_SIZE || buckets < 2 ||
        buckets > MAX_BUCKETS || out_cap < 0) {
        return PLSC_ERR_ARGS;
    }
    int64_t capacity = 256;
    uint64_t *tokens = (uint64_t *)malloc((size_t)capacity * sizeof(uint64_t));
    if (tokens == NULL) {
        return PLSC_ERR_ALLOC;
    }
    uint32_t local_size = 1024;
    LocalEntry *local = (LocalEntry *)calloc(local_size, sizeof(LocalEntry));
    if (local == NULL) {
        free(tokens);
        return PLSC_ERR_ALLOC;
    }
    int32_t written = 0;
    int32_t stamp = 0;
    for (int32_t doc = 0; doc < n_docs; doc++) {
        out_doc_start[doc] = written;
        int32_t count = tokenize_doc(blob + offsets[doc], offsets[doc + 1] - offsets[doc], &tokens,
                                     &capacity);
        if (count < 0) {
            free(tokens);
            free(local);
            return count;
        }
        int64_t grams = 0;
        for (int32_t n = 1; n <= max_n && n <= count; n++) {
            grams += count - n + 1;
        }
        if (grams > (int64_t)(MAX_LOCAL_TABLE / 4)) {
            free(tokens);
            free(local);
            return PLSC_ERR_ALLOC;
        }
        uint32_t needed = next_pow2((uint32_t)(grams * 4 + 16));
        if (needed > local_size) {
            LocalEntry *resized = (LocalEntry *)calloc(needed, sizeof(LocalEntry));
            if (resized == NULL) {
                free(tokens);
                free(local);
                return PLSC_ERR_ALLOC;
            }
            free(local);
            local = resized;
            local_size = needed;
            stamp = 0;
        }
        stamp++;
        uint32_t mask = local_size - 1;
        for (int32_t n = 1; n <= max_n && n <= count; n++) {
            for (int32_t start = 0; start + n <= count; start++) {
                uint64_t key = gram_hash(tokens + start, n);
                uint32_t slot = (uint32_t)(splitmix64(key)) & mask;
                while (1) {
                    LocalEntry *entry = &local[slot];
                    if (entry->stamp != stamp) {
                        entry->stamp = stamp;
                        entry->key = key;
                        entry->value = 1.0f;
                        if (written >= out_cap) {
                            free(tokens);
                            free(local);
                            return PLSC_ERR_CAPACITY;
                        }
                        entry->index = written;
                        out_index[written] = (int32_t)(key % (uint64_t)buckets);
                        out_value[written] = 1.0f;
                        written++;
                        break;
                    }
                    if (entry->key == key) {
                        entry->value += 1.0f;
                        out_value[entry->index] = entry->value;
                        break;
                    }
                    slot = (slot + 1) & mask;
                }
            }
        }
    }
    out_doc_start[n_docs] = written;
    free(tokens);
    free(local);
    return written;
}

int32_t plsc_gram_stats(const uint8_t *blob, const int64_t *offsets, int32_t n_docs,
                        const int32_t *flags, int32_t max_n, int32_t min_count, PlscGram *out,
                        int32_t out_cap) {
    if (blob == NULL || offsets == NULL || flags == NULL || out == NULL || n_docs < 0 ||
        max_n < 1 || max_n > MAX_NGRAM_SIZE || out_cap < 0) {
        return PLSC_ERR_ARGS;
    }
    int64_t total_bytes = offsets[n_docs] - offsets[0];
    uint64_t estimate = (uint64_t)(total_bytes / 5 + 16) * (uint64_t)max_n;
    uint64_t wanted = estimate * 2;
    if (wanted > MAX_GRAM_TABLE) {
        wanted = MAX_GRAM_TABLE;
    }
    uint32_t table_size = next_pow2((uint32_t)wanted);
    GramEntry *table = (GramEntry *)calloc(table_size, sizeof(GramEntry));
    if (table == NULL) {
        return PLSC_ERR_ALLOC;
    }
    int64_t capacity = 256;
    uint64_t *tokens = (uint64_t *)malloc((size_t)capacity * sizeof(uint64_t));
    if (tokens == NULL) {
        free(table);
        return PLSC_ERR_ALLOC;
    }
    uint32_t mask = table_size - 1;
    uint32_t filled = 0;
    uint32_t limit = table_size - (table_size >> 2);
    for (int32_t doc = 0; doc < n_docs; doc++) {
        int32_t count = tokenize_doc(blob + offsets[doc], offsets[doc + 1] - offsets[doc], &tokens,
                                     &capacity);
        if (count < 0) {
            free(tokens);
            free(table);
            return count;
        }
        int32_t is_target = flags[doc] ? 1 : 0;
        for (int32_t n = 1; n <= max_n && n <= count; n++) {
            for (int32_t start = 0; start + n <= count; start++) {
                uint64_t key = gram_hash(tokens + start, n);
                uint32_t slot = (uint32_t)(splitmix64(key ^ 0x5bf03635U)) & mask;
                while (1) {
                    GramEntry *entry = &table[slot];
                    if (!entry->used) {
                        if (filled >= limit) {
                            break;
                        }
                        entry->used = 1;
                        entry->key = key;
                        entry->n = n;
                        entry->count = 1;
                        entry->target_count = is_target;
                        entry->doc_count = 1;
                        entry->last_doc = doc;
                        entry->first_doc = doc;
                        entry->first_pos = start;
                        filled++;
                        break;
                    }
                    if (entry->key == key) {
                        entry->count += 1;
                        entry->target_count += is_target;
                        if (entry->last_doc != doc) {
                            entry->last_doc = doc;
                            entry->doc_count += 1;
                        }
                        break;
                    }
                    slot = (slot + 1) & mask;
                }
            }
        }
    }
    int32_t written = 0;
    for (uint32_t index = 0; index < table_size; index++) {
        GramEntry *entry = &table[index];
        if (!entry->used || entry->count < min_count) {
            continue;
        }
        if (written >= out_cap) {
            free(tokens);
            free(table);
            return PLSC_ERR_CAPACITY;
        }
        out[written].key = entry->key;
        out[written].n = entry->n;
        out[written].count = entry->count;
        out[written].target_count = entry->target_count;
        out[written].doc_count = entry->doc_count;
        out[written].first_doc = entry->first_doc;
        out[written].first_pos = entry->first_pos;
        written++;
    }
    free(tokens);
    free(table);
    return written;
}

int32_t plsc_minhash(const uint8_t *blob, const int64_t *offsets, int32_t n_docs, int32_t shingle_n,
                     int32_t num_perm, uint64_t *out_sig) {
    if (blob == NULL || offsets == NULL || out_sig == NULL || n_docs < 0 || shingle_n < 1 ||
        shingle_n > MAX_NGRAM_SIZE || num_perm < 1 || num_perm > 4096) {
        return PLSC_ERR_ARGS;
    }
    int64_t capacity = 256;
    uint64_t *tokens = (uint64_t *)malloc((size_t)capacity * sizeof(uint64_t));
    if (tokens == NULL) {
        return PLSC_ERR_ALLOC;
    }
    uint64_t *seeds = (uint64_t *)malloc((size_t)num_perm * sizeof(uint64_t));
    if (seeds == NULL) {
        free(tokens);
        return PLSC_ERR_ALLOC;
    }
    for (int32_t p = 0; p < num_perm; p++) {
        seeds[p] = splitmix64((uint64_t)p + 0x1234567ULL);
    }
    for (int32_t doc = 0; doc < n_docs; doc++) {
        uint64_t *signature = out_sig + (int64_t)doc * num_perm;
        for (int32_t p = 0; p < num_perm; p++) {
            signature[p] = UINT64_MAX;
        }
        int32_t count = tokenize_doc(blob + offsets[doc], offsets[doc + 1] - offsets[doc], &tokens,
                                     &capacity);
        if (count < 0) {
            free(tokens);
            free(seeds);
            return count;
        }
        int32_t window = shingle_n <= count ? shingle_n : count;
        if (window < 1) {
            continue;
        }
        for (int32_t start = 0; start + window <= count; start++) {
            uint64_t key = gram_hash(tokens + start, window);
            for (int32_t p = 0; p < num_perm; p++) {
                uint64_t value = splitmix64(key ^ seeds[p]);
                if (value < signature[p]) {
                    signature[p] = value;
                }
            }
        }
    }
    free(tokens);
    free(seeds);
    return n_docs;
}
