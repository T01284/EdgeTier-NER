#ifndef EDGE_CRF_VITERBI_H
#define EDGE_CRF_VITERBI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Linear-chain CRF Viterbi decode.
 *
 * @param emissions      [seq_len * num_tags] row-major token emissions
 * @param transitions    [num_tags * num_tags] transition[i][j] = i -> j
 * @param seq_len        valid sequence length
 * @param num_tags       number of labels
 * @param out_tags       output buffer [seq_len]
 * @return 0 on success
 */
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
