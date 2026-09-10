# Cloud evidence snapshot (2026-07-14)

Pulled from AutoDL via `scripts/_pull_evidence_pack.py`.
Manifest: `deploy/results/cloud_evidence_manifest.json` (35 metric/log files).
Local mirrors: `outputs/runs/*/metrics.json|fewshot_summary.json`, `outputs/logs/gpu_paper_gap*.log`.

## Full-data test F1 (%)

| Run | F1 |
|-----|-----|
| conll2003_full | 82.37 |
| conll2003_bilstm | 81.76 |
| conll2003_bert | 85.88 |
| conll2003_distilbert | 85.87 |
| conll2003_tinybert | 84.23 |
| ontonotes_full | 73.78 |
| ontonotes_bilstm | 75.70 |
| ontonotes_bert | 83.77 |
| ontonotes_distilbert | 83.14 |
| ontonotes_tinybert | 72.26 |
| cluener_full | 67.59 |
| cluener_bilstm | 66.97 |
| cluener_bert | 75.24 |
| cluener_distilbert | 71.93 (mDistilBERT) |
| cluener_tinybert | 71.01 |
| conll2003_ablation_wordlevel | 76.22 |
| conll2003_distill | 70.69 |

## Few-shot (mean±std, n=5; all **4-way**)

| Run | mean±std |
|-----|----------|
| Ours CoNLL 1/5-shot (*5w* name / n_way=4) | 29.27±1.79 / 30.86±2.04 |
| Ours OntoNotes 4w1s/4w5s | 12.10±2.45 / 14.60±2.71 |
| ProtoBERT 4w1s/4w5s | 58.80±2.05 / 61.27±1.58 |
| NNShot 4w1s/4w5s | 10.83±2.38 / 20.57±1.21 |
| StructShot 4w1s/4w5s | 59.12±0.97 / 60.97±1.92 |

## Paper fixes applied with this pull

- Removed CLUENER "under debugging"; aligned OntoNotes 73.8 / CLUENER 67.6
- Clarified 82.4→70.2 as train/deploy snapshot mismatch (ckpt re-eval ≈70.3)
- Removed ablation "re-evaluation in progress"; reported 76.2 / 70.7 training F1
- Unified few-shot as 4-way (legacy `*5w*` dirs are aliases)
