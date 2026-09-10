# 云端实验进度（AutoDL）

> 最后更新：**2026-07-13 19:00 UTC+8**（已启动 `gpu_paper_gap` 批次）  
> 远程路径：`/root/autodl-tmp/EdgeFS_NER`  
> 刷新命令：`python scripts/check_experiment_progress.py` / `python scripts/_pull_metrics_summary.py`

---

## 流水线状态

| 日志 | 状态 | 完成时间 (UTC) |
|------|------|----------------|
| `logs/gpu_experiments.log` | **已完成** | 2026-07-12 14:03:48 |
| `logs/gpu_retrain_iob_fix.log` | **已完成** | 2026-07-13 10:01:27 |
| `logs/gpu_paper_gap.log` | **运行中** | P0 补洞批次（few-shot 基线 / CLUENER 中文 BERT / OntoNotes 核对 / 消融） |
| 当前 GPU 训练进程 | 见 `nohup_paper_gap.out` | `run_gpu_paper_gap.sh` |

---

## 全量训练 test F1（%）

### CoNLL-2003

| Run | F1 | 备注 |
|-----|-----|------|
| **conll2003_full (Ours)** | **82.37** | 论文主结果 |
| conll2003_bilstm | 81.76 | baseline |
| conll2003_bilstm_v3 | 79.24 | baseline |
| conll2003_bert | 85.88 | baseline |
| conll2003_distilbert | 85.87 | baseline |
| conll2003_tinybert | 84.23 | baseline |
| conll2003_distill | 70.94 | 蒸馏消融（非最终 Ours） |
| conll2003_ablation_wordlevel | 77.00 | 词级消融 |

### OntoNotes 5.0

| Run | F1 |
|-----|-----|
| ontonotes_full (Ours) | 79.11 |
| ontonotes_bilstm | 80.83 |
| ontonotes_bert | 87.39 |
| ontonotes_distilbert | 86.30 |
| ontonotes_tinybert | 81.63 |

### CLUENER

| Run | F1 | 备注 |
|-----|-----|------|
| cluener_full (Ours) | 67.76 | |
| cluener_bilstm | 67.14 | |
| cluener_bert | **28.29** | ⚠️ 异常偏低，需排查 |
| cluener_distilbert | **27.35** | ⚠️ 异常偏低，需排查 |
| cluener_tinybert | **26.94** | ⚠️ 异常偏低，需排查 |

---

## 小样本（5 seeds，test F1 %）

| Run | mean ± std |
|-----|------------|
| conll2003_fewshot_5w1s | 29.10 ± 1.58 |
| conll2003_fewshot_5w5s | 29.03 ± 2.43 |
| conll2003_fewshot_proto_only | 29.81 ± 1.10 |
| conll2003_fewshot_contrastive_only | 29.51 ± 1.17 |
| ontonotes_fewshot_5w1s | 28.79 ± 1.62 |
| ontonotes_fewshot_5w5s | 30.84 ± 1.98 |

**未在云端发现：** `cluener_fewshot_*`（若论文需要 CLUENER 小样本，需补跑）

---

## 与论文待办（P0）对照

| 项 | 云端状态 | 下一步 |
|----|----------|--------|
| P0-1 填 Table 1–4 | 数据已在远程 | `remote_pull_all.py` → 更新 `paper/tables/*.tex` |
| P0-2 CoNLL baseline | ✅ 已有 BiLSTM/BERT/DistilBERT/TinyBERT | Ours 82.37 vs BERT 85.88（掉 ~3.5 pt） |
| P0-3 few-shot | ✅ CoNLL + OntoNotes 完成 | 填 Table 2；分数偏低需讨论/调参 |
| P0-4 ESP32 全量 F1 | 不依赖 GPU | 本地/真机补跑 |
| CLUENER BERT 基线 | ⚠️ ~28% F1 | 检查中文分词、标签对齐、预训练模型加载 |

---

## 关键观察（给写论文用）

1. **Trade-off 可写了**：Ours CoNLL 82.37 vs BERT 85.88，换 343KB MCU 可部署。
2. **Ours 未低于 BiLSTM**（82.37 vs 81.76），掉点主要在 transformer 档。
3. **Few-shot ~29%** 远低于全量 82%——需确认 episode 设置、是否只训了 meta 未微调、或评估协议问题。
4. **CLUENER transformer 基线崩溃**——投稿前必须修或从表内剔除并说明。
5. **导出被 skip**（`== skip export ==`）——若需更新 ONNX/ESP32 权重，需在远程重跑 `export_model.py`。

---

## 本地同步

```powershell
# 拉取全部 metrics / fewshot_summary
python scripts/remote_pull_all.py   # 或扩展 run 列表后重跑

# 刷新本文件
python scripts/_pull_metrics_summary.py | Tee-Object experiments/cloud_progress_latest.txt
```

原始输出备份：`experiments/cloud_progress_latest.txt`
