# Experiment run registry

> 详见 **`experiments/CLOUD_PROGRESS.md`**（云端 AutoDL 最新进度与 F1 汇总）。

| Run ID | Config | Machine | Status | Test F1 | Metrics path |
|--------|--------|---------|--------|---------|--------------|
| toy_full | configs/train/full_toy.yaml | local CPU | done | — | outputs/runs/toy_full/metrics.json |
| conll2003_full | full_conll2003_autodl.yaml | AutoDL GPU | **done** | 82.37 | remote: outputs/runs/conll2003_full/metrics.json |
| conll2003_bilstm | full_conll2003_bilstm_autodl.yaml | AutoDL GPU | done | 81.76 | remote |
| conll2003_bert | full_conll2003_bert_autodl.yaml | AutoDL GPU | done | 85.88 | remote |
| conll2003_distilbert | full_conll2003_distilbert_autodl.yaml | AutoDL GPU | done | 85.87 | remote |
| conll2003_tinybert | full_conll2003_tinybert_autodl.yaml | AutoDL GPU | done | 84.23 | remote |
| conll2003_fewshot_5w1s | fewshot_conll2003_5w1s_autodl.yaml | AutoDL GPU | done | 29.10±1.58 | remote |
| conll2003_fewshot_5w5s | fewshot_conll2003_5w5s_autodl.yaml | AutoDL GPU | done | 29.03±2.43 | remote |
| ontonotes_full | full_ontonotes_autodl.yaml | AutoDL GPU | done | 79.11 | remote |
| cluener_full | full_cluener_autodl.yaml | AutoDL GPU | done | 67.76 | remote |
| cluener_bert | full_cluener_bert_autodl.yaml | AutoDL GPU | done ⚠️ | 28.29 | remote — 待排查 |

**云端流水线：** `gpu_paper_gap` 批次于 2026-07-13 18:48 UTC+8 启动（P0 补洞）。监控：`tail -f /root/autodl-tmp/EdgeFS_NER/logs/gpu_paper_gap.log`；本地：`python scripts/_pull_metrics_summary.py`。
