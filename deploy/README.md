# Edge deployment test harnesses

跨平台真机测试代码，对应论文 Table 3（延迟 / 峰值内存 / 可行性）。

**在其他设备复现实验**：见 [`REPRODUCE.md`](REPRODUCE.md)（检查点、ONNX、`.espdl`、生成头文件与 benchmark 结果均已纳入 git）。

```
deploy/
├── shared/                 # Pi 与 ESP32 共用的测试输入、CRF 参考实现
├── pi3b/                   # Raspberry Pi 3 Model B（ONNX Runtime + Python）
└── esp32-s3/               # ESP32-S3 DevKitC-1 N8（8MB Flash，PlatformIO + ESP-IDF）
```

## 部署资源包（训练机导出）

在训练机 / GPU 服务器上：

```powershell
python scripts/export_model.py --checkpoint outputs/runs/toy_full/best.pt
```

生成 `outputs/export/deploy/`：

| 文件 | 用途 |
|------|------|
| `model_emissions.onnx` | 神经网络前向（emissions） |
| `char_vocab.json` | 字符词表 |
| `tags.json` | BIO 标签表 |
| `crf_transitions.npy` | CRF 转移矩阵（Viterbi 解码） |
| `deploy_manifest.json` | 元数据索引 |

将 `deploy/` 目录复制到 Pi；将 `.espdl`（量化后）复制到 ESP32 `models/`。

## 快速入口

- Pi 3B：`deploy/pi3b/README.md`
- ESP32-S3：`deploy/esp32-s3/README.md`

## 统一测试句

`shared/sample_inputs.json` 中的句子应在 Pi 与 ESP32 上使用相同输入，便于对比延迟与标签一致性。
