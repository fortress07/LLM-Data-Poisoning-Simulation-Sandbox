#ifndef POISONSCAN_H
#define POISONSCAN_H

#include <stdint.h>

#if defined(_WIN32)
#define PLSC_API __declspec(dllexport)
#else
#define PLSC_API __attribute__((visibility("default")))
#endif

#define PLSC_ABI_VERSION 4
#define PLSC_ERR_CAPACITY (-1)
#define PLSC_ERR_ALLOC (-2)
#define PLSC_ERR_ARGS (-3)

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t key;
    int32_t n;
    int32_t count;
    int32_t target_count;
    int32_t doc_count;
    int32_t first_doc;
    int32_t first_pos;
} PlscGram;

PLSC_API int32_t plsc_abi_version(void);

PLSC_API int32_t plsc_featurize(const uint8_t *blob, const int64_t *offsets, int32_t n_docs,
                                int32_t max_n, int32_t buckets, int32_t *out_index,
                                float *out_value, int32_t *out_doc_start, int32_t out_cap);

PLSC_API int32_t plsc_gram_stats(const uint8_t *blob, const int64_t *offsets, int32_t n_docs,
                                 const int32_t *flags, int32_t max_n, int32_t min_count,
                                 PlscGram *out, int32_t out_cap);

PLSC_API int32_t plsc_minhash(const uint8_t *blob, const int64_t *offsets, int32_t n_docs,
                              int32_t shingle_n, int32_t num_perm, uint64_t *out_sig);

PLSC_API int32_t plsc_token_count(const uint8_t *blob, const int64_t *offsets, int32_t n_docs,
                                  int32_t *out_counts);

#ifdef __cplusplus
}
#endif

#endif
