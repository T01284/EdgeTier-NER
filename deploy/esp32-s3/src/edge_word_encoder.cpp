#include "edge_word_encoder.h"

#include "edge_char_embed.h"
#include "edge_char_table.h"
#include "edge_config.h"
#include "edge_word_conv.h"

#include <cmath>
#include <cstring>

static int8_t quant_word_feature(float value) {
    const float scaled = value / std::ldexp(1.0f, EDGE_WORD_FEATURE_EXP);
    const int rounded = static_cast<int>(std::lround(scaled));
    if (rounded > 127) {
        return 127;
    }
    if (rounded < -128) {
        return -128;
    }
    return static_cast<int8_t>(rounded);
}

static bool token_is_all_pad(const float *input_ids, size_t token_idx) {
    const size_t base = token_idx * EDGE_MAX_WORD_LEN;
    for (size_t w = 0; w < EDGE_MAX_WORD_LEN; ++w) {
        if (static_cast<int>(input_ids[base + w]) != EDGE_CHAR_PAD_ID) {
            return false;
        }
    }
    return true;
}

static void compute_token_features(
    const float *input_ids,
    size_t token_idx,
    const float embed_scale,
    float *token_embed,
    int8_t *out_features_ct) {
    for (size_t c = 0; c < EDGE_CHAR_EMBED_DIM; ++c) {
        for (size_t w = 0; w < EDGE_MAX_WORD_LEN; ++w) {
            const size_t char_idx =
                static_cast<size_t>(input_ids[token_idx * EDGE_MAX_WORD_LEN + w]);
            const size_t vocab_idx =
                char_idx < EDGE_CHAR_VOCAB_SIZE ? char_idx : (EDGE_CHAR_VOCAB_SIZE - 1);
            token_embed[c * EDGE_MAX_WORD_LEN + w] =
                static_cast<float>(kEdgeCharEmbedInt8[vocab_idx * EDGE_CHAR_EMBED_DIM + c]) *
                embed_scale;
        }
    }

    for (size_t oc = 0; oc < EDGE_NUM_FILTERS; ++oc) {
        float max_value = 0.0f;
        for (size_t w = 0; w < EDGE_MAX_WORD_LEN; ++w) {
            float acc = kEdgeWordConvBias[oc];
            for (size_t k = 0; k < EDGE_WORD_CONV_KERNEL; ++k) {
                const int wi = static_cast<int>(w) + static_cast<int>(k) - 1;
                if (wi < 0 || wi >= static_cast<int>(EDGE_MAX_WORD_LEN)) {
                    continue;
                }
                const float *embed_col = token_embed + static_cast<size_t>(wi);
                const size_t weight_base =
                    (oc * EDGE_CHAR_EMBED_DIM * EDGE_WORD_CONV_KERNEL) + k;
                for (size_t ic = 0; ic < EDGE_CHAR_EMBED_DIM; ++ic) {
                    acc += embed_col[ic * EDGE_MAX_WORD_LEN] *
                           kEdgeWordConvWeight[weight_base + ic * EDGE_WORD_CONV_KERNEL];
                }
            }
            const float relu = acc > 0.0f ? acc : 0.0f;
            if (relu > max_value) {
                max_value = relu;
            }
        }
        out_features_ct[oc] = quant_word_feature(max_value);
    }
}

void edge_word_features_from_input_ids(
    const float *input_ids,
    int8_t *word_features_out,
    size_t seq_len) {
    if (input_ids == nullptr || word_features_out == nullptr) {
        return;
    }

    (void)seq_len;
    const float embed_scale = std::ldexp(1.0f, EDGE_CHAR_EMBED_INPUT_EXP);
    static float token_embed[EDGE_CHAR_EMBED_DIM * EDGE_MAX_WORD_LEN];
    static int8_t pad_features[EDGE_NUM_FILTERS];
    static bool pad_features_ready = false;

    if (!pad_features_ready) {
        float pad_ids[EDGE_MAX_WORD_LEN];
        for (size_t w = 0; w < EDGE_MAX_WORD_LEN; ++w) {
            pad_ids[w] = static_cast<float>(EDGE_CHAR_PAD_ID);
        }
        compute_token_features(pad_ids, 0, embed_scale, token_embed, pad_features);
        pad_features_ready = true;
    }

    for (size_t t = 0; t < EDGE_MAX_SEQ_LEN; ++t) {
        if (token_is_all_pad(input_ids, t)) {
            for (size_t oc = 0; oc < EDGE_NUM_FILTERS; ++oc) {
                word_features_out[oc * EDGE_MAX_SEQ_LEN + t] = pad_features[oc];
            }
            continue;
        }

        int8_t token_features[EDGE_NUM_FILTERS];
        compute_token_features(input_ids, t, embed_scale, token_embed, token_features);
        for (size_t oc = 0; oc < EDGE_NUM_FILTERS; ++oc) {
            word_features_out[oc * EDGE_MAX_SEQ_LEN + t] = token_features[oc];
        }
    }
}

void edge_word_features_t0_int8(
    const float *input_ids,
    int8_t *token0_channels_out,
    size_t num_channels) {
    if (input_ids == nullptr || token0_channels_out == nullptr || num_channels == 0) {
        return;
    }

    const size_t copy_channels =
        num_channels < EDGE_NUM_FILTERS ? num_channels : EDGE_NUM_FILTERS;
    const float embed_scale = std::ldexp(1.0f, EDGE_CHAR_EMBED_INPUT_EXP);
    static float token_embed[EDGE_CHAR_EMBED_DIM * EDGE_MAX_WORD_LEN];
    int8_t token_features[EDGE_NUM_FILTERS];

    if (token_is_all_pad(input_ids, 0)) {
        static int8_t pad_features[EDGE_NUM_FILTERS];
        static bool pad_features_ready = false;
        if (!pad_features_ready) {
            float pad_ids[EDGE_MAX_WORD_LEN];
            for (size_t w = 0; w < EDGE_MAX_WORD_LEN; ++w) {
                pad_ids[w] = static_cast<float>(EDGE_CHAR_PAD_ID);
            }
            compute_token_features(pad_ids, 0, embed_scale, token_embed, pad_features);
            pad_features_ready = true;
        }
        for (size_t oc = 0; oc < copy_channels; ++oc) {
            token0_channels_out[oc] = pad_features[oc];
        }
        return;
    }

    compute_token_features(input_ids, 0, embed_scale, token_embed, token_features);
    for (size_t oc = 0; oc < copy_channels; ++oc) {
        token0_channels_out[oc] = token_features[oc];
    }
}
