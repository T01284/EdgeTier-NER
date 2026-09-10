#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Allocate TemporalCNN scratch buffers (PSRAM preferred). Call once at boot. */
bool edge_blocks_init(void);

/**
 * Run BN-folded TemporalCNN + linear classifier.
 * word_features_ct: int8 NCT layout [C * T], dequantized with feature_exp.
 * emissions_out: [T * num_tags] float logits.
 */
bool edge_blocks_emit(
    const int8_t *word_features_ct,
    float *emissions_out,
    size_t seq_len,
    size_t num_tags);

#ifdef __cplusplus
}
#endif
