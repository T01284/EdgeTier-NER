# 实验复现指南

本文档说明如何在另一台机器上复现 **Table 3 ESP32-S3 ESP-DL** 真机 benchmark。仓库已纳入以下**可复现资产**（无需重新训练即可烧录）：

| 路径 | 说明 |
|------|------|
| `outputs/runs/conll2003_full/best.pt` | Char-CNN-CRF 训练检查点（~725 KB） |
| `outputs/export/conll2003_full/deploy/` | ONNX + 词表 + CRF 导出包 |
| `deploy/esp32-s3/models/edgefs_model.espdl` | INT8 量化模型（~343 KB） |
| `deploy/esp32-s3/models/artifact_manifest.json` | 部署资产版本清单（SHA256 + git tag） |
| `deploy/esp32-s3/include/edge_build_info.h` | 固件内嵌 artifact tag（启动日志可核对） |
| `deploy/esp32-s3/models/edgefs_model.onnx` | post-embed ONNX（固件侧查表 embedding） |
| `deploy/esp32-s3/include/edge_*.h` | 由 `gen_esp32_assets.py` 生成的 C 头文件 |
| `deploy/pi3b/assets/deploy/` | Pi 3B ONNX Runtime 同款部署包 |
| `deploy/results/espdl_test.json` | 最近一次真机 benchmark 记录 |

> **依赖组件**：`deploy/esp32-s3/managed_components/` 与 `.pio/` 仍在 gitignore 中；首次 `pio run` 会按 `dependencies.lock` 自动拉取 ESP-DL。

---

## 路径 A：仅复现 ESP32 烧录（最快）

适合已有 PlatformIO + ESP-IDF 的环境，**不需要** Python 虚拟环境。

```powershell
cd deploy/esp32-s3
pio run -e esp32-s3-n16r8-espdl -t upload --upload-port COM3
pio device monitor -p COM3
```

硬件：**ESP32-S3 DevKitC-1 N16R8**（8 MB Flash + 8 MB PSRAM）。串口应输出每句 `latency_us` 与 BIO 标签。

---

## 路径 B：从检查点完整重跑量化 → 资产 → 烧录

```powershell
# 1. Python 环境
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 2. 导出 ONNX（若 deploy 包已存在可跳过）
.venv\Scripts\python.exe scripts\export_model.py ^
  --checkpoint outputs/runs/conll2003_full/best.pt ^
  --output-dir outputs/export/conll2003_full

# 3. 量化 post-embed 图为 .espdl
.venv\Scripts\python.exe scripts\quantize_to_espdl.py --backend torch

# 4. 主机侧校验（可选）
.venv\Scripts\python.exe scripts\validate_espdl_export.py

# 5. 生成 C 头文件（char 表、INT8 embed、CRF、espdl 数组）+ artifact manifest
.venv\Scripts\python.exe scripts\gen_esp32_assets.py
# 或单独刷新版本标记：
.venv\Scripts\python.exe scripts\write_esp32_artifact_manifest.py --git-tag

# 6. 编译烧录
cd deploy\esp32-s3
pio run -e esp32-s3-n16r8-espdl -t upload --upload-port COM3
```

---

## 路径 C：一键自动化测试（主机 + 串口）

```powershell
.venv\Scripts\python.exe scripts\run_full_deploy_test.py ^
  --esp32-port COM3 ^
  --esp32-env esp32-s3-n16r8-espdl
```

加 `--skip-esp32-upload` 可跳过烧录，只解析已有串口日志。

---

## Pi 3B 对齐测试

```bash
cd deploy/pi3b
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_benchmark.py
```

资产目录：`deploy/pi3b/assets/deploy/`（与 `outputs/export/conll2003_full/deploy/` 同源）。

---

## 重新同步部署包（训练机 → 仓库）

在训练机更新检查点或导出后，运行：

```powershell
.venv\Scripts\python.exe scripts\export_model.py --checkpoint outputs/runs/conll2003_full/best.pt --output-dir outputs/export/conll2003_full
.venv\Scripts\python.exe scripts\quantize_to_espdl.py --backend torch
.venv\Scripts\python.exe scripts\gen_esp32_assets.py
Copy-Item -Recurse outputs\export\conll2003_full\deploy\* deploy\pi3b\assets\deploy\
git add outputs/ deploy/esp32-s3/models/ deploy/esp32-s3/include/ deploy/pi3b/assets/deploy/ deploy/results/
```

远程 GPU 训练：复制 `configs/remote/gpu_server.yaml.example` → `gpu_server.yaml` 并填入密钥（**勿提交** `gpu_server.yaml`）。
