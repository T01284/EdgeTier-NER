# ESP-DL component integration (PlatformIO)

ESP-DL 官方以 ESP-IDF component 形式提供，PlatformIO 接入方式：

## 推荐步骤

1. 克隆 ESP-DL 到本目录（或 git submodule）：
   ```bash
   git clone --depth 1 https://github.com/espressif/esp-dl.git esp-dl
   ```

2. 在 `platformio.ini` 的 `[env:esp32-s3-n16r8-espdl]` 中启用：
   ```ini
   lib_extra_dirs = lib
   build_flags =
       -D EDGE_USE_ESP_DL=1
       -D EDGE_HAS_ESPDL_MODEL=1
   board_build.embed_txtfiles =
       models/edgefs_model.espdl
   ```

3. 在 `src/inference_engine.cpp` 中调用 ESP-DL Model API 加载嵌入的 `.espdl`。

4. 导出前核对算子支持：
   https://github.com/espressif/esp-dl/blob/master/docs/en/operator_support_state.md

## 注意事项

- BatchNorm 若不受支持：训练侧做 BN folding（施工手册 Step 2）
- CRF Viterbi 保持在 `lib/edge_crf/`，不放入 ESP-DL 图
- INT16 优先，INT8 作为体积/延迟对照
