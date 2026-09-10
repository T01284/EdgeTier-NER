#include "edge_inference.h"

#include "edge_config.h"
#include "esp_log.h"
#include "esp_timer.h"

#include <cmath>
#include <cstring>
#include <vector>

#if defined(EDGE_USE_FIRMWARE_BLOCKS)
#include "edge_blocks_infer.h"
#include "edge_char_embed.h"
#include "edge_crf_transitions.h"
#include "edge_word_conv.h"
#include "edge_word_encoder.h"
#endif

#if defined(EDGE_USE_ESP_DL) && EDGE_HAS_ESPDL_MODEL
#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"
#include "edge_char_embed.h"
#include "edge_crf_transitions.h"
#include "edge_model_data.h"
#include "edge_word_conv.h"
#include "edge_word_encoder.h"
#include "fbs_model.hpp"
#endif

static const char *TAG = "edge_infer";

namespace edge {

#if defined(EDGE_USE_FIRMWARE_BLOCKS)
static int8_t g_fw_word_features[EDGE_NUM_FILTERS * EDGE_MAX_SEQ_LEN];
#endif

#if defined(EDGE_USE_ESP_DL) && EDGE_HAS_ESPDL_MODEL
static dl::Model *g_model = nullptr;
static dl::TensorBase *g_model_input = nullptr;
static dl::TensorBase *g_model_output = nullptr;
static size_t g_output_elems = 0;
static float g_output_scale = 1.0f;
static dl::dtype_t g_output_dtype = dl::DATA_TYPE_INT8;
static std::vector<int> g_input_shape;
static std::vector<int> g_output_shape;
static std::vector<int> g_input_axis_offset;
static std::vector<int> g_output_axis_offset;
static bool g_espdl_test_passed = false;
static int8_t g_last_word_features[EDGE_NUM_FILTERS * EDGE_MAX_SEQ_LEN];

static dl::TensorBase *pick_tensor(
    std::map<std::string, dl::TensorBase *> &tensors,
    const char *preferred) {
    if (preferred != nullptr) {
        auto it = tensors.find(preferred);
        if (it != tensors.end()) {
            return it->second;
        }
    }
    return tensors.empty() ? nullptr : tensors.begin()->second;
}

static int input_tensor_index(int c, int t) {
    if (g_input_axis_offset.size() >= 3) {
        return c * g_input_axis_offset[1] + t * g_input_axis_offset[2];
    }
    if (g_input_axis_offset.size() == 2) {
        return c * g_input_axis_offset[0] + t * g_input_axis_offset[1];
    }
    return c * EDGE_MAX_SEQ_LEN + t;
}

static int output_tensor_index(int t, int tag) {
    if (g_output_axis_offset.size() >= 3) {
        return t * g_output_axis_offset[1] + tag * g_output_axis_offset[2];
    }
    if (g_output_axis_offset.size() == 2) {
        return t * g_output_axis_offset[0] + tag * g_output_axis_offset[1];
    }
    return t * EDGE_NUM_TAGS + tag;
}

static bool write_model_input_int8(const float *input_ids) {
    int8_t *dst = g_model_input->get_element_ptr<int8_t>();
    if (dst == nullptr) {
        ESP_LOGE(TAG, "model input int8 pointer is null");
        return false;
    }

    static int8_t word_features[EDGE_NUM_FILTERS * EDGE_MAX_SEQ_LEN];
    edge_word_features_from_input_ids(input_ids, word_features, EDGE_MAX_SEQ_LEN);
    std::memcpy(
        g_last_word_features,
        word_features,
        sizeof(g_last_word_features));

    // ESP-DL flatbuffers store / consume word_features in TC order (t*C+c),
    // matching the embedded golden vector. Logical shape remains [1,C,T] for
    // ONNX, but on-device axis_offset CT indexing desynchronizes Conv from PPQ.
    // Write TC linear so runtime sees the same bytes PPQ golden used.
    const int size = static_cast<int>(g_model_input->get_size());
    for (size_t t = 0; t < EDGE_MAX_SEQ_LEN; ++t) {
        for (size_t c = 0; c < EDGE_NUM_FILTERS; ++c) {
            const int idx = static_cast<int>(t * EDGE_NUM_FILTERS + c);
            if (idx < 0 || idx >= size) {
                ESP_LOGE(TAG, "input index oob c=%u t=%u idx=%d", (unsigned)c, (unsigned)t, idx);
                return false;
            }
            dst[idx] = word_features[c * EDGE_MAX_SEQ_LEN + t];
        }
    }
    return true;
}

static bool read_emissions_int8(float *emissions_out, size_t out_tags, size_t tokens) {
    const int8_t *src_i8 = g_model_output->get_element_ptr<int8_t>();
    const int16_t *src_i16 = g_model_output->get_element_ptr<int16_t>();
    if (g_output_dtype == dl::DATA_TYPE_INT16 ? (src_i16 == nullptr) : (src_i8 == nullptr)) {
        ESP_LOGE(TAG, "model output pointer is null");
        return false;
    }

    for (size_t t = 0; t < tokens; ++t) {
        for (size_t tag = 0; tag < out_tags; ++tag) {
            const int idx = output_tensor_index(static_cast<int>(t), static_cast<int>(tag));
            if (idx < 0 || idx >= static_cast<int>(g_model_output->get_size())) {
                continue;
            }
            float val = 0.0f;
            if (g_output_dtype == dl::DATA_TYPE_INT16) {
                val = static_cast<float>(src_i16[idx]) * g_output_scale;
            } else {
                val = static_cast<float>(src_i8[idx]) * g_output_scale;
            }
            emissions_out[t * out_tags + tag] = val;
        }
    }
    return true;
}

static void fill_layer_summary(dl::TensorBase *tensor, const char *name, EdgeLayerSummary *out) {
    std::memset(out, 0, sizeof(*out));
    std::snprintf(out->name, sizeof(out->name), "%s", name);
    if (tensor == nullptr) {
        out->found = 0;
        return;
    }

    const float scale = DL_SCALE(tensor->get_exponent());
    const int size = static_cast<int>(tensor->get_size());
    out->size = static_cast<uint32_t>(size);
    out->exponent = static_cast<int8_t>(tensor->get_exponent());
    out->found = 1;

    float sum = 0.0f;
    float max_abs = 0.0f;
    const int sample_count = size < 16 ? size : 16;
    out->num_samples = static_cast<uint8_t>(sample_count > 0 ? sample_count : 0);

    for (int i = 0; i < size; ++i) {
        float value = 0.0f;
        if (tensor->get_dtype() == dl::DATA_TYPE_INT16) {
            value = static_cast<float>(tensor->get_element<int16_t>(i)) * scale;
        } else if (tensor->get_dtype() == dl::DATA_TYPE_FLOAT) {
            value = tensor->get_element<float>(i);
        } else {
            value = static_cast<float>(tensor->get_element<int8_t>(i)) * scale;
        }
        if (i < sample_count) {
            out->samples[i] = value;
        }
        sum += value;
        const float abs_value = std::fabs(value);
        if (abs_value > max_abs) {
            max_abs = abs_value;
        }
    }
    out->mean = size > 0 ? (sum / static_cast<float>(size)) : 0.0f;
    out->max_abs = max_abs;
}

static dl::TensorBase *resolve_layer_tensor(const char *layer_name) {
    if (layer_name == nullptr) {
        return nullptr;
    }
    if (std::strcmp(layer_name, "embedded") == 0 ||
        std::strcmp(layer_name, "word_features") == 0) {
        return g_model_input;
    }
    if (std::strcmp(layer_name, "emissions") == 0) {
        return g_model_output;
    }
    return g_model->get_intermediate(layer_name);
}
#endif

bool InferenceEngine::init() {
    num_tags_ = EDGE_NUM_TAGS;
    espdl_selftest_passed_ = false;
    ready_ = false;

#if defined(EDGE_USE_FIRMWARE_BLOCKS)
    if (!edge_blocks_init()) {
        ESP_LOGE(TAG, "firmware TemporalCNN init failed");
        return false;
    }
    ESP_LOGI(TAG, "inference path: firmware FP32 TemporalCNN (+ int8 word encoder)");
    ready_ = true;
#endif

#if defined(EDGE_USE_ESP_DL) && EDGE_HAS_ESPDL_MODEL
    ESP_LOGI(TAG, "loading espdl model (%u bytes)", (unsigned)EDGE_ESPDL_MODEL_SIZE);
    g_model = new dl::Model(
        (const char *)kEdgeEspdlModel,
        fbs::MODEL_LOCATION_IN_FLASH_RODATA,
        128 * 1024,
        dl::MEMORY_MANAGER_GREEDY,
        nullptr,
        true);
    if (g_model == nullptr) {
        ESP_LOGE(TAG, "Model allocator returned null");
        ready_ = false;
        return ready_;
    }

    auto &inputs = g_model->get_inputs();
    auto &outputs = g_model->get_outputs();
    g_model_input = pick_tensor(inputs, "word_features");
    if (g_model_input == nullptr) {
        g_model_input = pick_tensor(inputs, "embedded");
    }
    g_model_output = pick_tensor(outputs, "emissions");
    if (g_model_input == nullptr || g_model_output == nullptr) {
        ESP_LOGE(TAG, "model missing inputs/outputs");
        ready_ = false;
        return ready_;
    }

    g_input_shape = g_model_input->get_shape();
    g_output_shape = g_model_output->get_shape();
    g_input_axis_offset = g_model_input->axis_offset;
    g_output_axis_offset = g_model_output->axis_offset;
    g_output_scale = DL_SCALE(g_model_output->get_exponent());
    g_output_dtype = g_model_output->get_dtype();
    g_output_elems = 1;
    for (int dim : g_output_shape) {
        g_output_elems *= static_cast<size_t>(dim);
    }

    const char *input_name = inputs.begin()->first.c_str();
    const char *output_name = outputs.begin()->first.c_str();
    for (const auto &entry : inputs) {
        if (entry.second == g_model_input) {
            input_name = entry.first.c_str();
            break;
        }
    }
    for (const auto &entry : outputs) {
        if (entry.second == g_model_output) {
            output_name = entry.first.c_str();
            break;
        }
    }

    ESP_LOGI(TAG,
             "ESP-DL loaded | input=%s dtype=%d exp=%d shape=[%d,%d,%d] in_off=[%d,%d,%d] | "
             "output=%s exp=%d out_elems=%u out_off=[%d,%d,%d] tags=%u",
             input_name,
             (int)g_model_input->get_dtype(),
             g_model_input->get_exponent(),
             g_input_shape.size() > 0 ? g_input_shape[0] : -1,
             g_input_shape.size() > 1 ? g_input_shape[1] : -1,
             g_input_shape.size() > 2 ? g_input_shape[2] : -1,
             g_input_axis_offset.size() > 0 ? g_input_axis_offset[0] : -1,
             g_input_axis_offset.size() > 1 ? g_input_axis_offset[1] : -1,
             g_input_axis_offset.size() > 2 ? g_input_axis_offset[2] : -1,
             output_name,
             g_model_output->get_exponent(),
             (unsigned)g_output_elems,
             g_output_axis_offset.size() > 0 ? g_output_axis_offset[0] : -1,
             g_output_axis_offset.size() > 1 ? g_output_axis_offset[1] : -1,
             g_output_axis_offset.size() > 2 ? g_output_axis_offset[2] : -1,
             (unsigned)num_tags_);

    const esp_err_t test_rc = g_model->test();
    g_espdl_test_passed = (test_rc == ESP_OK);
    espdl_selftest_passed_ = g_espdl_test_passed;
    ESP_LOGI(TAG, "embedded espdl self-test: %s", g_espdl_test_passed ? "PASS" : "FAIL");
#if !defined(EDGE_USE_FIRMWARE_BLOCKS)
    ready_ = true;
#endif
#elif defined(EDGE_USE_STUB)
    ESP_LOGW(TAG, "STUB mode — synthetic emissions (no .espdl loaded)");
    num_tags_ = 4;
    espdl_selftest_passed_ = false;
    ready_ = true;
#else
#if !defined(EDGE_USE_FIRMWARE_BLOCKS)
    ESP_LOGE(TAG, "No inference backend selected");
    espdl_selftest_passed_ = false;
    ready_ = false;
#endif
#endif
    return ready_;
}

bool InferenceEngine::run_emissions(
    const float *input_ids,
    const uint8_t *mask,
    size_t seq_len,
    float *emissions_out) {
    return run_emissions_timed(input_ids, mask, seq_len, emissions_out, nullptr, nullptr);
}

bool InferenceEngine::run_emissions_timed(
    const float *input_ids,
    const uint8_t *mask,
    size_t seq_len,
    float *emissions_out,
    uint32_t *embed_us_out,
    uint32_t *espdl_us_out) {
    if (!ready_ || !input_ids || !mask || !emissions_out) {
        return false;
    }

#if defined(EDGE_USE_FIRMWARE_BLOCKS)
    (void)mask;
    const uint64_t t_embed0 = esp_timer_get_time();
    edge_word_features_from_input_ids(input_ids, g_fw_word_features, EDGE_MAX_SEQ_LEN);
    const uint64_t t_embed1 = esp_timer_get_time();
    const size_t tokens = seq_len == 0 ? EDGE_MAX_SEQ_LEN : seq_len;
    const uint64_t t_infer0 = esp_timer_get_time();
    const bool ok = edge_blocks_emit(g_fw_word_features, emissions_out, tokens, num_tags_);
    const uint64_t t_infer1 = esp_timer_get_time();
    if (embed_us_out) {
        *embed_us_out = (uint32_t)(t_embed1 - t_embed0);
    }
    if (espdl_us_out) {
        *espdl_us_out = (uint32_t)(t_infer1 - t_infer0);
    }
    return ok;
#elif defined(EDGE_USE_ESP_DL) && EDGE_HAS_ESPDL_MODEL
    (void)seq_len;
    const uint64_t t_embed0 = esp_timer_get_time();
    if (!write_model_input_int8(input_ids)) {
        return false;
    }
    const uint64_t t_embed1 = esp_timer_get_time();

    const uint64_t t_infer0 = esp_timer_get_time();
    g_model->run();
    const size_t out_tags = num_tags_;
    const size_t seq_cap = g_output_elems / out_tags;
    const size_t tokens = seq_cap < EDGE_MAX_SEQ_LEN ? seq_cap : EDGE_MAX_SEQ_LEN;
    const bool ok = read_emissions_int8(emissions_out, out_tags, tokens);
    const uint64_t t_infer1 = esp_timer_get_time();

    if (embed_us_out) {
        *embed_us_out = (uint32_t)(t_embed1 - t_embed0);
    }
    if (espdl_us_out) {
        *espdl_us_out = (uint32_t)(t_infer1 - t_infer0);
    }
    return ok;
#else
    (void)embed_us_out;
    (void)espdl_us_out;
    for (size_t t = 0; t < seq_len; ++t) {
        const float word_signal = mask[t] ? 1.0f : 0.0f;
        for (size_t tag = 0; tag < num_tags_; ++tag) {
            emissions_out[t * num_tags_ + tag] =
                mask[t] ? word_signal + (float)tag * 0.001f : -1e9f;
        }
    }
    return true;
#endif
}

static bool load_golden_word_features_tensor(dl::TensorBase *golden_tensor) {
    int8_t *dst = g_model_input->get_element_ptr<int8_t>();
    const int8_t *src = golden_tensor->get_element_ptr<int8_t>();
    if (dst == nullptr || src == nullptr) {
        return false;
    }
    // Golden is already flattened TC (t*C+c); match write_model_input_int8.
    const size_t nbytes = static_cast<size_t>(g_model_input->get_size()) * sizeof(int8_t);
    std::memcpy(dst, src, nbytes);
    return true;
}

bool InferenceEngine::run_golden_emissions(float *emissions_out, size_t tokens, size_t out_tags) {
    if (!ready_ || !emissions_out || tokens == 0 || out_tags == 0) {
        return false;
    }
#if defined(EDGE_USE_ESP_DL) && EDGE_HAS_ESPDL_MODEL
    fbs::FbsModel *fbs = g_model->get_fbs_model();
    if (fbs == nullptr) {
        return false;
    }
    fbs->load_map();
    dl::TensorBase *test_input = fbs->get_test_input_tensor("word_features");
    if (test_input == nullptr) {
        test_input = fbs->get_test_input_tensor("embedded");
    }
    if (test_input == nullptr || !load_golden_word_features_tensor(test_input)) {
        if (test_input != nullptr) {
            delete test_input;
        }
        fbs->clear_map();
        return false;
    }
    delete test_input;

    g_model->run();

    const size_t seq_cap = g_output_elems / out_tags;
    const size_t read_tokens = tokens < seq_cap ? tokens : seq_cap;
    const bool ok = read_emissions_int8(emissions_out, out_tags, read_tokens);
    fbs->clear_map();
    return ok;
#else
    (void)tokens;
    (void)out_tags;
    return false;
#endif
}

bool InferenceEngine::dump_layer_summaries(
    const float *input_ids,
    const char *const *layer_names,
    size_t num_layers,
    EdgeLayerSummary *out,
    size_t out_capacity) {
    if (!ready_ || !input_ids || !layer_names || !out || out_capacity == 0) {
        return false;
    }
#if defined(EDGE_USE_ESP_DL) && EDGE_HAS_ESPDL_MODEL
    if (num_layers > out_capacity) {
        num_layers = out_capacity;
    }
    if (!write_model_input_int8(input_ids)) {
        return false;
    }
    g_model->run();

    for (size_t i = 0; i < num_layers; ++i) {
        dl::TensorBase *tensor = resolve_layer_tensor(layer_names[i]);
        fill_layer_summary(tensor, layer_names[i], &out[i]);
    }
    return true;
#else
    (void)input_ids;
    (void)layer_names;
    (void)num_layers;
    (void)out;
    (void)out_capacity;
    return false;
#endif
}

bool InferenceEngine::dump_word_features_int8(
    const float *input_ids,
    int8_t *out,
    size_t capacity,
    size_t *written) {
    if (!ready_ || !input_ids || !out || capacity == 0 || written == nullptr) {
        return false;
    }
#if defined(EDGE_USE_FIRMWARE_BLOCKS) || (defined(EDGE_USE_ESP_DL) && EDGE_HAS_ESPDL_MODEL)
    static int8_t word_features[EDGE_NUM_FILTERS * EDGE_MAX_SEQ_LEN];
    edge_word_features_from_input_ids(input_ids, word_features, EDGE_MAX_SEQ_LEN);
#if defined(EDGE_USE_ESP_DL) && EDGE_HAS_ESPDL_MODEL
    std::memcpy(
        g_last_word_features,
        word_features,
        sizeof(g_last_word_features));
#endif
    const size_t ct_bytes = EDGE_NUM_FILTERS * EDGE_MAX_SEQ_LEN;
    const size_t copy_ct = capacity < ct_bytes ? capacity : ct_bytes;
    std::memcpy(out, word_features, copy_ct);
    *written = copy_ct;
    return true;
#else
    (void)input_ids;
    (void)out;
    (void)capacity;
    (void)written;
    return false;
#endif
}

bool InferenceEngine::crf_transitions(float *transitions_out, size_t capacity) const {
    if (!transitions_out || capacity < num_tags_ * num_tags_) {
        return false;
    }
#if defined(EDGE_USE_FIRMWARE_BLOCKS) || (defined(EDGE_USE_ESP_DL) && EDGE_HAS_ESPDL_MODEL)
    memcpy(transitions_out, kEdgeCrfTransitions, num_tags_ * num_tags_ * sizeof(float));
    return true;
#else
    for (size_t i = 0; i < num_tags_; ++i) {
        for (size_t j = 0; j < num_tags_; ++j) {
            transitions_out[i * num_tags_ + j] = (i == j) ? 0.1f : 0.0f;
        }
    }
    return true;
#endif
}

}  // namespace edge
