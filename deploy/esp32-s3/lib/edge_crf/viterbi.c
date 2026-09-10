#include "viterbi.h"

#include <float.h>
#include <stdlib.h>
#include <string.h>

int edge_crf_viterbi_decode(
    const float *emissions,
    const float *transitions,
    size_t seq_len,
    size_t num_tags,
    int *out_tags) {
    if (!emissions || !transitions || !out_tags || seq_len == 0 || num_tags == 0) {
        return -1;
    }

    const size_t state_count = num_tags;
    const size_t total = seq_len * state_count;

    float *dp = (float *)malloc(total * sizeof(float));
    int *back = (int *)malloc(total * sizeof(int));
    if (!dp || !back) {
        free(dp);
        free(back);
        return -2;
    }

    for (size_t t = 0; t < state_count; ++t) {
        dp[t] = emissions[t];
        back[t] = -1;
    }

    for (size_t pos = 1; pos < seq_len; ++pos) {
        const float *emit = emissions + pos * state_count;
        for (size_t cur = 0; cur < state_count; ++cur) {
            float best = -FLT_MAX;
            int best_prev = 0;
            for (size_t prev = 0; prev < state_count; ++prev) {
                const size_t prev_idx = (pos - 1) * state_count + prev;
                const float score = dp[prev_idx] + transitions[prev * state_count + cur] + emit[cur];
                if (score > best) {
                    best = score;
                    best_prev = (int)prev;
                }
            }
            const size_t idx = pos * state_count + cur;
            dp[idx] = best;
            back[idx] = best_prev;
        }
    }

    float best_final = -FLT_MAX;
    int last = 0;
    for (size_t t = 0; t < state_count; ++t) {
        const float val = dp[(seq_len - 1) * state_count + t];
        if (val > best_final) {
            best_final = val;
            last = (int)t;
        }
    }

    out_tags[seq_len - 1] = last;
    for (size_t pos = seq_len - 1; pos-- > 0;) {
        const size_t idx = (pos + 1) * state_count + (size_t)out_tags[pos + 1];
        out_tags[pos] = back[idx];
    }

    free(dp);
    free(back);
    return 0;
}
