# ESP32-S3 (PlatformIO) — Edge NER test harness

ESP32-S3 **DevKitC-1 N16R8**（8 MB Flash + 8 MB PSRAM）真机 benchmark，对齐论文 Table 3 MCU 列。

本仓库在实测板卡上已完成 ESP-DL INT8 推理（post-embed 图 + 固件侧 INT8 embedding 查表）。`models/edgefs_model.espdl` 与 `include/edge_*.h` **已纳入 git**，克隆后可直接 `pio run -e esp32-s3-n16r8-espdl` 烧录；完整量化流程见 [`../REPRODUCE.md`](../REPRODUCE.md)。

## 版本标记（论文复现）

每次更新 `.espdl` 或固件资产后运行：

```powershell
.venv\Scripts\python.exe scripts\write_esp32_artifact_manifest.py
# 可选：同步打 git tag
.venv\Scripts\python.exe scripts\write_esp32_artifact_manifest.py --git-tag
```

产出：

| 文件 | 说明 |
|------|------|
| `models/artifact_manifest.json` | 全量 SHA256、checkpoint、git commit、量化参数 |
| `include/edge_build_info.h` | 固件启动日志中的 `EDGE_ARTIFACT_TAG` |

**论文 Table 3 当前 artifact tag**：`edgefs-esp32-e9849bc4`（git tag 同名）。复现时在 manifest 中核对 `primary_espdl_sha256` 与 `edgefs_model.espdl` 一致即可。

开发工具：**PlatformIO**（`framework = espidf`）

## 目录

```
esp32-s3/
├── platformio.ini      # stub / espdl 两套环境
├── partitions.csv
├── include/            # 板级与推理 API 头文件
├── src/                # main + InferenceEngine
├── lib/
│   ├── edge_crf/       # Viterbi 解码（与 deploy/shared/crf 保持同步）
│   └── esp_dl/         # ESP-DL 组件接入说明
├── models/             # edgefs_model.espdl / .onnx（已纳入 git）
├── scripts/            # build / upload / monitor
└── test/               # 可选 native 单测
```

## 环境 A：Stub（无需 .espdl，先验证 PIO 流水线）

```powershell
cd deploy/esp32-s3
pio run -e esp32-s3-n16r8-stub
pio run -e esp32-s3-n16r8-stub -t upload
pio device monitor
```

串口输出每句 `latency_us`、`heap`、`psram` 空闲值。

## 环境 B：ESP-DL（量化模型就绪后）

1. 在 Linux/GPU 主机用 ESP-PPQ 导出 `edgefs_model.espdl`
2. 复制到 `models/edgefs_model.espdl`
3. 按 `lib/esp_dl/README.md` 接入 ESP-DL 组件
4. 切换环境：

```powershell
pio run -e esp32-s3-n16r8-espdl -t upload
```

## 硬件要求

| 项 | 规格（实测） |
|----|------|
| 芯片 | ESP32-S3 v0.2 |
| Flash | 8MB |
| 内部 SRAM | 512KB |
| 外部 PSRAM | 8 MB（输入缓冲分配于 SPIRAM） |
| ESP-IDF | ≥ 5.3（PlatformIO espressif32 6.x 自带） |

## 与 Pi 3B 对齐

- 测试句与 `deploy/shared/sample_inputs.json` 一致（当前内嵌于 `src/main.cpp`）
- 神经网络只跑 **emissions**；CRF Viterbi 在设备端 C 实现（不进 .espdl 图）
- 正式实验记录：**每句延迟**、**峰值内部 SRAM**（本板无 PSRAM）

## 常用命令

```powershell
.\scripts\pio_build.ps1
.\scripts\pio_upload.ps1
.\scripts\pio_monitor.ps1
```

## UART 全量 eval（CoNLL test）

默认 COM 口在 `platformio.ini`（当前 `COM32`）。变更端口：

```powershell
# 方式 1：命令行
.venv\Scripts\python.exe scripts\esp32_uart_eval.py --port COM3

# 方式 2：临时改 platformio.ini 的 upload_port / monitor_port
```

**断线续跑**（1722 句全量 eval）：

```powershell
# 从第 N 句继续（0-based）
.venv\Scripts\python.exe scripts\esp32_uart_eval.py --port COM32 --start 500 --skip-upload

# 使用 checkpoint 文件恢复
.venv\Scripts\python.exe scripts\esp32_uart_eval.py --resume-checkpoint --skip-upload
```

中间结果写入 `deploy/results/esp32_full_eval.checkpoint.json`（默认每 50 句）。

### GPIO UART（功耗测量：USB 只供电）

评测协议可迁到 **UART1 GPIO**，USB 口仅供电/烧录：

```text
ESP32-S3 TX  GPIO17  --->  TTL 适配器 RX
ESP32-S3 RX  GPIO18  <---  TTL 适配器 TX
GND          GND     ----  TTL GND
波特率 115200 8N1；TTL 不要再供 5V（板已由 USB 供电）
```

```powershell
pio run -e esp32-s3-n16r8-espdl-uart-pure-gpio -t upload --upload-port <USB烧录口>
# 之后协议走 TTL 口，例如 COM23：
.venv\Scripts\python.exe scripts\esp32_uart_eval.py --port COM23 --pio-env esp32-s3-n16r8-espdl-uart-pure-gpio --skip-upload
```

READY 行会带 `uart=GPIO TX=17 RX=18 baud=115200`。

**artifact 版本**：eval JSON 应含 `artifact_tag`（与 `models/artifact_manifest.json` 一致）。对已有结果回填：

```powershell
.venv\Scripts\python.exe scripts\patch_esp32_eval_artifact.py
```

烧录后串口 `READY` 行会输出 `tag=` 与 `espdl_sha=` 前缀，便于与论文复现 tag 核对。

## 下一步（等 ESP-PPQ 模型）

- [x] `InferenceEngine` 加载 `models/edgefs_model.espdl`
- [x] 从 `deploy_manifest.json` 烧录 CRF transitions
- [ ] 对比 Pi / ESP32 标签一致性（Pi F1 70.2 vs ESP32 9.9 量化差距已记录）
