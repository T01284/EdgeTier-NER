#pragma once

#include "edge_config.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "viterbi.h"

#ifdef __cplusplus
}
#endif

struct EdgeSampleInput {
    const char *id;
    const char *text;
};

struct EdgeBenchmarkResult {
    const char *sample_id;
    uint32_t seq_len;
    uint32_t latency_us;
    uint32_t free_heap_bytes;
    uint32_t free_psram_bytes;
    int tag_ids[EDGE_MAX_SEQ_LEN];
};

struct EdgeLayerSummary {
    char name[48];
    uint32_t size;
    int8_t exponent;
    uint8_t found;
    float max_abs;
    float mean;
    float samples[16];
    uint8_t num_samples;
};

#ifdef __cplusplus

namespace edge {

class InferenceEngine {
public:
    bool init();
    bool run_emissions(const float *input_ids, const uint8_t *mask, size_t seq_len, float *emissions_out);
    // Timed variant: embed_us = word-embed + int8 pack; espdl_us = model.run + dequant.
    // On firmware-block builds, espdl_us measures TemporalCNN emit (same role as ESP-DL).
    bool run_emissions_timed(
        const float *input_ids,
        const uint8_t *mask,
        size_t seq_len,
        float *emissions_out,
        uint32_t *embed_us_out,
        uint32_t *espdl_us_out);
    bool run_golden_emissions(float *emissions_out, size_t tokens, size_t out_tags);
    bool dump_layer_summaries(
        const float *input_ids,
        const char *const *layer_names,
        size_t num_layers,
        EdgeLayerSummary *out,
        size_t out_capacity);
    bool dump_word_features_int8(
        const float *input_ids,
        int8_t *out,
        size_t capacity,
        size_t *written);
    bool crf_transitions(float *transitions_out, size_t capacity) const;
    size_t num_tags() const { return num_tags_; }
    bool espdl_selftest_passed() const { return espdl_selftest_passed_; }

private:
    size_t num_tags_ = 4;
    bool ready_ = false;
    bool espdl_selftest_passed_ = false;
};

}  // namespace edge

#endif
