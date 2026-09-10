#include "edge_blocks_infer.h"

#include "edge_blocks_weights.h"
#include "edge_char_embed.h"
#include "edge_config.h"
#include "edge_word_conv.h"

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_task_wdt.h"

#include <cmath>
#include <cstring>

static const char *TAG = "edge_blocks";

// Stacked TemporalCNN receptive field half-width: sum_i (K-1)*d_i = 2+4+8 = 14.
static constexpr size_t kEdgeBlocksContext = 14;

static float *g_buf_a = nullptr;
static float *g_buf_b = nullptr;
static bool g_ready = false;

static float *alloc_feature_buf() {
    float *p = static_cast<float *>(heap_caps_malloc(
        EDGE_BLOCKS_NUM_FILTERS * EDGE_MAX_SEQ_LEN * sizeof(float),
        MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (p == nullptr) {
        p = static_cast<float *>(heap_caps_malloc(
            EDGE_BLOCKS_NUM_FILTERS * EDGE_MAX_SEQ_LEN * sizeof(float),
            MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    }
    return p;
}

bool edge_blocks_init(void) {
    if (g_ready) {
        return true;
    }
    g_buf_a = alloc_feature_buf();
    g_buf_b = alloc_feature_buf();
    if (g_buf_a == nullptr || g_buf_b == nullptr) {
        ESP_LOGE(TAG, "failed to allocate TemporalCNN scratch");
        g_ready = false;
        return false;
    }
    g_ready = true;
    ESP_LOGI(TAG, "firmware FP32 TemporalCNN ready (blocks=%d)", EDGE_NUM_BLOCKS);
    return true;
}

static void conv_block_nct(
    const float *x,
    float *y,
    int block_idx,
    size_t seq_len) {
    const int dil = kEdgeBlockDilations[block_idx];
    const int pad = ((EDGE_BLOCK_KERNEL - 1) / 2) * dil;
    const float *w_base =
        kEdgeBlockWeight +
        block_idx * EDGE_BLOCKS_NUM_FILTERS * EDGE_BLOCKS_NUM_FILTERS * EDGE_BLOCK_KERNEL;
    const float *b_base = kEdgeBlockBias + block_idx * EDGE_BLOCKS_NUM_FILTERS;

    for (size_t oc = 0; oc < EDGE_BLOCKS_NUM_FILTERS; ++oc) {
        if ((oc & 7u) == 0u) {
            esp_task_wdt_reset();
        }
        const float *w_oc =
            w_base + oc * EDGE_BLOCKS_NUM_FILTERS * EDGE_BLOCK_KERNEL;
        const float bias = b_base[oc];
        float *y_oc = y + oc * EDGE_MAX_SEQ_LEN;
        for (size_t t = 0; t < seq_len; ++t) {
            float acc = bias;
            for (int k = 0; k < EDGE_BLOCK_KERNEL; ++k) {
                const int ti = static_cast<int>(t) + k * dil - pad;
                if (ti < 0 || ti >= static_cast<int>(seq_len)) {
                    continue;
                }
                const size_t t_idx = static_cast<size_t>(ti);
                for (size_t ic = 0; ic < EDGE_BLOCKS_NUM_FILTERS; ++ic) {
                    acc += x[ic * EDGE_MAX_SEQ_LEN + t_idx] *
                           w_oc[ic * EDGE_BLOCK_KERNEL + k];
                }
            }
            y_oc[t] = acc > 0.0f ? acc : 0.0f;
        }
    }

    for (size_t oc = 0; oc < EDGE_BLOCKS_NUM_FILTERS; ++oc) {
        float *y_oc = y + oc * EDGE_MAX_SEQ_LEN;
        const float *x_oc = x + oc * EDGE_MAX_SEQ_LEN;
        for (size_t t = 0; t < seq_len; ++t) {
            y_oc[t] += x_oc[t];
        }
    }
}

bool edge_blocks_emit(
    const int8_t *word_features_ct,
    float *emissions_out,
    size_t seq_len,
    size_t num_tags) {
    if (!g_ready || word_features_ct == nullptr || emissions_out == nullptr) {
        return false;
    }
    if (seq_len == 0 || seq_len > EDGE_MAX_SEQ_LEN) {
        return false;
    }
    if (num_tags != EDGE_BLOCKS_NUM_TAGS) {
        ESP_LOGE(TAG, "num_tags mismatch %u vs %d", (unsigned)num_tags, EDGE_BLOCKS_NUM_TAGS);
        return false;
    }

    // Truncate padded timeline to tokens + TemporalCNN context (exact for 0..seq_len-1).
    size_t T = seq_len + kEdgeBlocksContext;
    if (T > EDGE_MAX_SEQ_LEN) {
        T = EDGE_MAX_SEQ_LEN;
    }
    const float feature_scale = std::ldexp(1.0f, EDGE_WORD_FEATURE_EXP);
    for (size_t c = 0; c < EDGE_BLOCKS_NUM_FILTERS; ++c) {
        for (size_t t = 0; t < T; ++t) {
            g_buf_a[c * EDGE_MAX_SEQ_LEN + t] =
                static_cast<float>(word_features_ct[c * EDGE_MAX_SEQ_LEN + t]) * feature_scale;
        }
    }

    float *src = g_buf_a;
    float *dst = g_buf_b;
    for (int b = 0; b < EDGE_NUM_BLOCKS; ++b) {
        conv_block_nct(src, dst, b, T);
        float *tmp = src;
        src = dst;
        dst = tmp;
        esp_task_wdt_reset();
    }

    for (size_t t = 0; t < seq_len; ++t) {
        for (size_t tag = 0; tag < num_tags; ++tag) {
            float acc = kEdgeClassifierBias[tag];
            const float *row = kEdgeClassifierWeight + tag * EDGE_BLOCKS_NUM_FILTERS;
            for (size_t c = 0; c < EDGE_BLOCKS_NUM_FILTERS; ++c) {
                acc += row[c] * src[c * EDGE_MAX_SEQ_LEN + t];
            }
            emissions_out[t * num_tags + tag] = acc;
        }
    }
    return true;
}
