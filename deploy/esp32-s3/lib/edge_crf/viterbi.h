#ifndef EDGE_CRF_VITERBI_H
#define EDGE_CRF_VITERBI_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int edge_crf_viterbi_decode(
    const float *emissions,
    const float *transitions,
    size_t seq_len,
    size_t num_tags,
    int *out_tags);

#ifdef __cplusplus
}
#endif

#endif
