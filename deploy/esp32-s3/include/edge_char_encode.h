#pragma once

#include "edge_config.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

size_t edge_encode_sentence(
    const char *text,
    float *input_ids,
    uint8_t *mask,
    size_t max_len,
    size_t max_word_len);

#ifdef __cplusplus
}
#endif
