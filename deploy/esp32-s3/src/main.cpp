#include "edge_inference.h"

#include "edge_char_encode.h"
#include "edge_config.h"
#include "edge_word_encoder.h"
#if EDGE_HAS_ESPDL_MODEL
#include "edge_build_info.h"
#else
#define EDGE_ARTIFACT_TAG "stub"
#define EDGE_ESPDL_SHA256 "none"
#define EDGE_ESPDL_SHA256_PREFIX "none"
#define EDGE_GIT_COMMIT "none"
#endif
#include "esp_log.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static const char *TAG = "edge_benchmark";

#ifndef EDGE_UART_LINE_MAX
#define EDGE_UART_LINE_MAX 1024
#endif

static const EdgeSampleInput kSamples[] = {
    {"en-01", "John works in Paris"},
    {"en-02", "Mary visited London yesterday"},
    {"en-03", "Google opened an office in Berlin"},
};

static void log_memory(const char *stage) {
    ESP_LOGI(TAG, "%s | heap=%u psram=%u",
             stage,
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
}

static bool run_one_sentence(
    edge::InferenceEngine &engine,
    const char *text,
    uint32_t *latency_us_out,
    std::vector<int> *tag_ids_out,
    float *em_debug_out = nullptr,
    size_t em_debug_tokens = 0,
    size_t em_debug_tags = 0,
    uint32_t *embed_us_out = nullptr,
    uint32_t *espdl_us_out = nullptr,
    uint32_t *viterbi_us_out = nullptr) {
    const size_t num_tags = engine.num_tags();
    const size_t plane = EDGE_MAX_SEQ_LEN * EDGE_MAX_WORD_LEN;
    std::vector<float> input_ids(plane, 0.0f);
    std::vector<float> emissions(EDGE_MAX_SEQ_LEN * EDGE_MAX_TAGS, 0.0f);
    std::vector<float> transitions(num_tags * num_tags, 0.0f);
    if (!engine.crf_transitions(transitions.data(), transitions.size())) {
        return false;
    }

    uint8_t mask[EDGE_MAX_SEQ_LEN] = {0};
    const uint64_t t_enc0 = esp_timer_get_time();
    const size_t len = edge_encode_sentence(
        text,
        input_ids.data(),
        mask,
        EDGE_MAX_SEQ_LEN,
        EDGE_MAX_WORD_LEN);
    const uint64_t t_enc1 = esp_timer_get_time();

    const int repeat =
#if defined(EDGE_UART_EVAL)
        1;
#else
        EDGE_BENCHMARK_REPEAT;
#endif
    uint32_t embed_inner_us = 0;
    uint32_t espdl_us = 0;
    for (int r = 0; r < repeat; ++r) {
        uint32_t e_us = 0;
        uint32_t i_us = 0;
        if (!engine.run_emissions_timed(
                input_ids.data(),
                mask,
                len,
                emissions.data(),
                &e_us,
                &i_us)) {
            return false;
        }
        embed_inner_us = e_us;
        espdl_us = i_us;
    }

    tag_ids_out->assign(len, 0);
    const uint64_t t_vit0 = esp_timer_get_time();
    edge_crf_viterbi_decode(
        emissions.data(),
        transitions.data(),
        len,
        num_tags,
        tag_ids_out->data());
    const uint64_t t_vit1 = esp_timer_get_time();

    if (em_debug_out != nullptr && em_debug_tokens > 0 && em_debug_tags > 0) {
        const size_t copy_tokens = len < em_debug_tokens ? len : em_debug_tokens;
        for (size_t t = 0; t < copy_tokens; ++t) {
            for (size_t tag = 0; tag < em_debug_tags; ++tag) {
                em_debug_out[t * em_debug_tags + tag] = emissions[t * num_tags + tag];
            }
        }
    }

    const uint32_t encode_us = (uint32_t)(t_enc1 - t_enc0);
    const uint32_t embed_us = encode_us + embed_inner_us;
    const uint32_t viterbi_us = (uint32_t)(t_vit1 - t_vit0);
    if (embed_us_out) {
        *embed_us_out = embed_us;
    }
    if (espdl_us_out) {
        *espdl_us_out = espdl_us;
    }
    if (viterbi_us_out) {
        *viterbi_us_out = viterbi_us;
    }
    // End-to-end: encode + embed/quant + ESP-DL (or FW blocks) + Viterbi.
    *latency_us_out = embed_us + espdl_us + viterbi_us;
    return true;
}

static void run_smoke_benchmark(edge::InferenceEngine &engine) {
    const size_t plane = EDGE_MAX_SEQ_LEN * EDGE_MAX_WORD_LEN;
    std::vector<float> input_ids(plane, 0.0f);
    std::vector<float> emissions(EDGE_MAX_SEQ_LEN * EDGE_MAX_TAGS, 0.0f);
    uint8_t mask[EDGE_MAX_SEQ_LEN] = {0};

    for (int w = 0; w < EDGE_BENCHMARK_WARMUP; ++w) {
        const size_t len = edge_encode_sentence(
            kSamples[0].text,
            input_ids.data(),
            mask,
            EDGE_MAX_SEQ_LEN,
            EDGE_MAX_WORD_LEN);
        engine.run_emissions(input_ids.data(), mask, len, emissions.data());
    }

    for (size_t si = 0; si < sizeof(kSamples) / sizeof(kSamples[0]); ++si) {
        const EdgeSampleInput &sample = kSamples[si];
        uint32_t latency_us = 0;
        std::vector<int> tag_ids;
        if (!run_one_sentence(engine, sample.text, &latency_us, &tag_ids)) {
            ESP_LOGE(TAG, "inference failed for %s", sample.id);
            continue;
        }

        std::string tag_str;
        for (size_t i = 0; i < tag_ids.size(); ++i) {
            if (i > 0) {
                tag_str += ',';
            }
            tag_str += std::to_string(tag_ids[i]);
        }

        ESP_LOGI(TAG,
                 "sample=%s len=%u latency_us=%u tags=%s heap=%u psram=%u",
                 sample.id,
                 (unsigned)tag_ids.size(),
                 (unsigned)latency_us,
                 tag_str.c_str(),
                 (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT),
                 (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    }

    ESP_LOGI(TAG, "benchmark complete");
}

#if defined(EDGE_UART_EVAL)

#if defined(EDGE_UART_GPIO)
#include "driver/gpio.h"
#include "driver/uart.h"
#ifndef EDGE_UART_NUM
#define EDGE_UART_NUM UART_NUM_1
#endif
#ifndef EDGE_UART_TX_GPIO
#define EDGE_UART_TX_GPIO 17
#endif
#ifndef EDGE_UART_RX_GPIO
#define EDGE_UART_RX_GPIO 18
#endif
#ifndef EDGE_UART_BAUD
#define EDGE_UART_BAUD 115200
#endif
#else
#include "driver/usb_serial_jtag.h"
#endif

static bool read_line_host(char *buf, size_t max_len) {
    size_t n = 0;
    uint8_t ch = 0;
    while (n + 1 < max_len) {
#if defined(EDGE_UART_GPIO)
        const int r = uart_read_bytes(EDGE_UART_NUM, &ch, 1, pdMS_TO_TICKS(100));
#else
        const int r = usb_serial_jtag_read_bytes(&ch, 1, pdMS_TO_TICKS(100));
#endif
        if (r <= 0) {
            continue;
        }
        if (ch == '\n' || ch == '\r') {
            if (n == 0) {
                continue;
            }
            buf[n] = '\0';
            return true;
        }
        buf[n++] = static_cast<char>(ch);
    }
    buf[n] = '\0';
    return n > 0;
}

static void write_host_line(const char *text) {
#if defined(EDGE_UART_GPIO)
    uart_write_bytes(EDGE_UART_NUM, text, std::strlen(text));
#else
    usb_serial_jtag_write_bytes(
        reinterpret_cast<const uint8_t *>(text),
        std::strlen(text),
        pdMS_TO_TICKS(1000));
#endif
}

static const char *kLayerDumpNames[] = {
    "word_features",
    "/blocks.0/relu/Relu_output_0",
    "/blocks.0/Add_output_0",
    "/blocks.1/relu/Relu_output_0",
    "/blocks.1/Add_output_0",
    "/blocks.2/relu/Relu_output_0",
    "/blocks.2/Add_output_0",
    "/classifier/MatMul_output_0",
    "emissions",
};

static bool write_layer_dump(edge::InferenceEngine &engine, const char *text) {
    const size_t plane = EDGE_MAX_SEQ_LEN * EDGE_MAX_WORD_LEN;
    std::vector<float> input_ids(plane, 0.0f);
    uint8_t mask[EDGE_MAX_SEQ_LEN] = {0};
    const size_t len = edge_encode_sentence(
        text,
        input_ids.data(),
        mask,
        EDGE_MAX_SEQ_LEN,
        EDGE_MAX_WORD_LEN);
    (void)len;

    const size_t num_layers = sizeof(kLayerDumpNames) / sizeof(kLayerDumpNames[0]);
    std::vector<EdgeLayerSummary> layers(num_layers);
    if (!engine.dump_layer_summaries(
            input_ids.data(),
            kLayerDumpNames,
            num_layers,
            layers.data(),
            layers.size())) {
        return false;
    }

    for (size_t i = 0; i < num_layers; ++i) {
        const EdgeLayerSummary &layer = layers[i];
        std::string sample_str;
        for (uint8_t s = 0; s < layer.num_samples; ++s) {
            if (s > 0) {
                sample_str += ',';
            }
            char buf[24];
            std::snprintf(buf, sizeof(buf), "%.5f", layer.samples[s]);
            sample_str += buf;
        }

        char line[512];
        std::snprintf(
            line,
            sizeof(line),
            "LYR name=%s found=%u size=%u exp=%d max=%.5f mean=%.5f samp=%s\n",
            layer.name,
            (unsigned)layer.found,
            (unsigned)layer.size,
            (int)layer.exponent,
            layer.max_abs,
            layer.mean,
            sample_str.c_str());
        write_host_line(line);
    }
    return true;
}

static void emit_ready_line(edge::InferenceEngine &engine) {
    char ready_line[256];
    std::snprintf(
        ready_line,
        sizeof(ready_line),
#if defined(EDGE_USE_FIRMWARE_BLOCKS)
#if defined(EDGE_UART_GPIO)
        "READY board=%s tag=%s espdl_sha=%s espdl_test=%s infer=fw_fp32 "
        "uart=GPIO TX=%d RX=%d baud=%d\n",
#else
        "READY board=%s tag=%s espdl_sha=%s espdl_test=%s infer=fw_fp32 uart=USB_JTAG\n",
#endif
#else
#if defined(EDGE_UART_GPIO)
        "READY board=%s tag=%s espdl_sha=%s espdl_test=%s infer=espdl "
        "uart=GPIO TX=%d RX=%d baud=%d\n",
#else
        "READY board=%s tag=%s espdl_sha=%s espdl_test=%s infer=espdl uart=USB_JTAG\n",
#endif
#endif
        EDGE_BOARD_NAME,
        EDGE_ARTIFACT_TAG,
        EDGE_ESPDL_SHA256_PREFIX,
        engine.espdl_selftest_passed() ? "PASS" : "FAIL"
#if defined(EDGE_UART_GPIO)
        ,
        EDGE_UART_TX_GPIO,
        EDGE_UART_RX_GPIO,
        EDGE_UART_BAUD
#endif
    );
    write_host_line(ready_line);
}

static void run_uart_eval_loop(edge::InferenceEngine &engine) {
#if defined(EDGE_UART_GPIO)
    uart_config_t uart_config = {};
    uart_config.baud_rate = EDGE_UART_BAUD;
    uart_config.data_bits = UART_DATA_8_BITS;
    uart_config.parity = UART_PARITY_DISABLE;
    uart_config.stop_bits = UART_STOP_BITS_1;
    uart_config.flow_ctrl = UART_HW_FLOWCTRL_DISABLE;
    uart_config.source_clk = UART_SCLK_DEFAULT;
    ESP_ERROR_CHECK(uart_driver_install(EDGE_UART_NUM, 4096, 4096, 0, nullptr, 0));
    ESP_ERROR_CHECK(uart_param_config(EDGE_UART_NUM, &uart_config));
    ESP_ERROR_CHECK(uart_set_pin(
        EDGE_UART_NUM,
        EDGE_UART_TX_GPIO,
        EDGE_UART_RX_GPIO,
        UART_PIN_NO_CHANGE,
        UART_PIN_NO_CHANGE));
    ESP_LOGI(
        TAG,
        "host UART on GPIO TX=%d RX=%d baud=%d (USB can be power-only)",
        EDGE_UART_TX_GPIO,
        EDGE_UART_RX_GPIO,
        EDGE_UART_BAUD);
#else
    usb_serial_jtag_driver_config_t usb_cfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    if (usb_serial_jtag_driver_install(&usb_cfg) != ESP_OK) {
        ESP_LOGW(TAG, "usb_serial_jtag_driver_install returned non-OK (may already be installed)");
    }
#endif
    vTaskDelay(pdMS_TO_TICKS(50));

    emit_ready_line(engine);
    ESP_LOGI(TAG,
             "READY board=%s tag=%s espdl_sha=%s git=%s",
             EDGE_BOARD_NAME,
             EDGE_ARTIFACT_TAG,
             EDGE_ESPDL_SHA256_PREFIX,
             EDGE_GIT_COMMIT);

    static char line[EDGE_UART_LINE_MAX];
    static float wordfeat_ids[EDGE_MAX_SEQ_LEN * EDGE_MAX_WORD_LEN];
    while (read_line_host(line, sizeof(line))) {
        const size_t n = std::strlen(line);
        if (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) {
            line[n - 1] = '\0';
        }

        if (line[0] == '\0') {
            continue;
        }
        if (std::strcmp(line, "PING") == 0) {
            write_host_line("PONG\n");
            emit_ready_line(engine);
            continue;
        }
        const char *wordfeat_text = nullptr;
        if (std::strncmp(line, "WORDFEAT\t", 9) == 0) {
            wordfeat_text = line + 9;
        } else if (std::strncmp(line, "WORDFEAT ", 9) == 0) {
            wordfeat_text = line + 9;
        }
        if (wordfeat_text != nullptr) {
            const char *text = wordfeat_text;
            uint8_t mask[EDGE_MAX_SEQ_LEN] = {0};
            for (size_t i = 0; i < EDGE_MAX_SEQ_LEN * EDGE_MAX_WORD_LEN; ++i) {
                wordfeat_ids[i] = 0.0f;
            }
            (void)edge_encode_sentence(
                text,
                wordfeat_ids,
                mask,
                EDGE_MAX_SEQ_LEN,
                EDGE_MAX_WORD_LEN);

            int8_t t0c_vals[16] = {0};
            edge_word_features_t0_int8(wordfeat_ids, t0c_vals, 16);

            char t0c[192] = {0};
            size_t t0c_len = 0;
            for (size_t c = 0; c < 16; ++c) {
                if (c > 0) {
                    t0c_len += (size_t)std::snprintf(t0c + t0c_len, sizeof(t0c) - t0c_len, ",");
                }
                t0c_len += (size_t)std::snprintf(
                    t0c + t0c_len,
                    sizeof(t0c) - t0c_len,
                    "%d",
                    (int)t0c_vals[c]);
            }

            char out[256];
            std::snprintf(out, sizeof(out), "OKFEAT t0c=%s\n", t0c);
            write_host_line(out);
            continue;
        }
        if (std::strcmp(line, "GOLDEN") == 0) {
            const size_t num_tags = engine.num_tags();
            std::vector<float> emissions(EDGE_MAX_SEQ_LEN * num_tags, 0.0f);
            std::vector<float> transitions(num_tags * num_tags, 0.0f);
            if (!engine.crf_transitions(transitions.data(), transitions.size()) ||
                !engine.run_golden_emissions(emissions.data(), 4, num_tags)) {
                write_host_line("ERR golden_failed\n");
                continue;
            }
            std::vector<int> tag_ids(4, 0);
            edge_crf_viterbi_decode(
                emissions.data(),
                transitions.data(),
                4,
                num_tags,
                tag_ids.data());
            std::string tag_str;
            for (size_t i = 0; i < tag_ids.size(); ++i) {
                if (i > 0) {
                    tag_str += ',';
                }
                tag_str += std::to_string(tag_ids[i]);
            }
            char out[256];
            std::snprintf(
                out,
                sizeof(out),
                "OKGOLDEN em0=%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f tags=%s\n",
                emissions[0],
                emissions[1],
                emissions[2],
                emissions[3],
                emissions[4],
                emissions[5],
                emissions[6],
                emissions[7],
                tag_str.c_str());
            write_host_line(out);
            continue;
        }
        if (std::strncmp(line, "LAYERDUMP\t", 10) == 0) {
            if (!write_layer_dump(engine, line + 10)) {
                write_host_line("ERR layerdump_failed\n");
                continue;
            }
            write_host_line("OKLAYERS\n");
            continue;
        }
        if (std::strcmp(line, "LAYERDUMP") == 0) {
            if (!write_layer_dump(engine, kSamples[0].text)) {
                write_host_line("ERR layerdump_failed\n");
                continue;
            }
            write_host_line("OKLAYERS\n");
            continue;
        }
        if (std::strcmp(line, "QUIT") == 0) {
            write_host_line("BYE\n");
            break;
        }

        const char *text = nullptr;
        bool debug_mode = false;
        if (std::strncmp(line, "EVALDBG\t", 8) == 0) {
            text = line + 8;
            debug_mode = true;
        } else if (std::strncmp(line, "EVALDBG ", 8) == 0) {
            text = line + 8;
            debug_mode = true;
        } else if (std::strncmp(line, "EVAL\t", 5) == 0) {
            text = line + 5;
        } else if (std::strncmp(line, "EVAL ", 5) == 0) {
            text = line + 5;
        } else {
            write_host_line("ERR unknown_command\n");
            continue;
        }

        uint32_t latency_us = 0;
        uint32_t embed_us = 0;
        uint32_t espdl_us = 0;
        uint32_t viterbi_us = 0;
        std::vector<int> tag_ids;
        const size_t num_tags = engine.num_tags();
        float em_debug[4 * EDGE_MAX_TAGS] = {0.0f};
        const size_t em_debug_tokens = debug_mode ? 4 : 0;
        if (!run_one_sentence(
                engine,
                text,
                &latency_us,
                &tag_ids,
                debug_mode ? em_debug : nullptr,
                em_debug_tokens,
                debug_mode ? num_tags : 0,
                &embed_us,
                &espdl_us,
                &viterbi_us)) {
            write_host_line("ERR inference_failed\n");
            continue;
        }

        std::string tag_str;
        for (size_t i = 0; i < tag_ids.size(); ++i) {
            if (i > 0) {
                tag_str += ',';
            }
            tag_str += std::to_string(tag_ids[i]);
        }

        if (debug_mode) {
            char em0_str[160] = {0};
            size_t offset = 0;
            const size_t em0_cap = sizeof(em0_str);
            for (size_t tag = 0; tag < num_tags; ++tag) {
                if (tag > 0 && offset + 1 < em0_cap) {
                    em0_str[offset++] = ',';
                }
                const int written = std::snprintf(
                    em0_str + offset,
                    em0_cap - offset,
                    "%.5f",
                    em_debug[tag]);
                if (written <= 0) {
                    break;
                }
                offset += static_cast<size_t>(written);
                if (offset >= em0_cap) {
                    break;
                }
            }

            char dbg[384];
            std::snprintf(
                dbg,
                sizeof(dbg),
                "OKDBG latency_us=%u embed_us=%u espdl_us=%u viterbi_us=%u "
                "len=%u em0=%s tags=%s\n",
                (unsigned)latency_us,
                (unsigned)embed_us,
                (unsigned)espdl_us,
                (unsigned)viterbi_us,
                (unsigned)tag_ids.size(),
                em0_str,
                tag_str.c_str());
            write_host_line(dbg);
            continue;
        }

        char out[640];
        std::snprintf(
            out,
            sizeof(out),
            "OK latency_us=%u embed_us=%u espdl_us=%u viterbi_us=%u "
            "len=%u tags=%s heap=%u psram=%u\n",
            (unsigned)latency_us,
            (unsigned)embed_us,
            (unsigned)espdl_us,
            (unsigned)viterbi_us,
            (unsigned)tag_ids.size(),
            tag_str.c_str(),
            (unsigned)heap_caps_get_free_size(MALLOC_CAP_8BIT),
            (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
        write_host_line(out);
    }
}

#endif

extern "C" void app_main(void) {
    ESP_LOGI(TAG,
             "EdgeFS NER ESP32-S3 benchmark start (%s) tag=%s espdl=%s",
             EDGE_BOARD_NAME,
             EDGE_ARTIFACT_TAG,
             EDGE_ESPDL_SHA256_PREFIX);
    log_memory("boot");

    edge::InferenceEngine engine;
    if (!engine.init()) {
        ESP_LOGE(TAG, "InferenceEngine init failed");
        return;
    }

#if defined(EDGE_UART_EVAL)
    run_uart_eval_loop(engine);
#else
    run_smoke_benchmark(engine);
#endif

    log_memory("done");
}
