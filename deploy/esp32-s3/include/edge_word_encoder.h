#pragma once

#include "edge_char_embed.h"
#include "edge_config.h"
#include "edge_word_conv.h"

#include <stddef.h>
#include <stdint.h>

#include <stdint.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void edge_word_features_from_input_ids(
    const float *input_ids,
    int8_t *word_features_out,
    size_t seq_len);

void edge_word_features_t0_int8(
    const float *input_ids,
    int8_t *token0_channels_out,
    size_t num_channels);

#ifdef __cplusplus
}
#endif
