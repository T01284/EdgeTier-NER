# Raspberry Pi 3 Model B — Edge NER test harness

ONNX Runtime 推理 + Python CRF Viterbi 解码，用于测量论文部署表中 Pi 3B 延迟、峰值内存与 F1。

钉扎 bundle：`assets/deploy/` 与仓库根目录
`outputs/export/conll2003_full_fulltestb_s42/deploy/` **哈希一致**（论文 Pi INT16 路径）。

## 目录

```
pi3b/
├── edgefs_pi/          # 推理与 benchmark Python 包
├── scripts/            # CLI 入口
├── config/             # benchmark 配置
├── assets/deploy/      # 从训练机复制的 deploy bundle（gitignore）
└── results/            # benchmark JSON 输出（gitignore）
```

## 1. 准备 deploy bundle（训练机）

```powershell
python scripts/export_model.py --checkpoint outputs/runs/toy_full/best.pt
```

将 `outputs/export/deploy/` 整目录复制到 Pi：

```bash
scp -r outputs/export/deploy pi@<pi-ip>:~/EdgeFS_NER/deploy/pi3b/assets/
```

## 2. Pi 上安装

```bash
cd deploy/pi3b
chmod +x scripts/setup_pi.sh
./scripts/setup_pi.sh
source .venv/bin/activate
```

## 3. 单句推理

```bash
python scripts/run_inference.py \
  --deploy-dir assets/deploy \
  --text "John works in Paris"
```

## 4. Benchmark（延迟 + RSS）

```bash
python scripts/run_benchmark.py \
  --deploy-dir assets/deploy \
  --output results/pi3b_benchmark.json
```

输出字段：
- `latency_ms_mean`：均句延迟
- `peak_rss_mb`：进程峰值 RSS（`resource.getrusage`）
- 每句 `tags`：便于与 ESP32 对比

## 注意事项

- 使用 **64-bit Raspberry Pi OS**，安装 `onnxruntime` CPU 版即可
- ONNX 仅导出 **emissions**；CRF 在 Pi 上独立 Viterbi 解码（与 ESP32 一致）
- 正式实验前先用 `shared/sample_inputs.json` 与 ESP32 对齐输入
