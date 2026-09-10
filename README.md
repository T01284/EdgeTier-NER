# EdgeTier-NER

Train-once Char-CNN-CRF export for on-device named entity recognition.  
Targets **Raspberry Pi 3B+** (ONNX Runtime) and **ESP32-S3** (ESP-DL / INT8).

> The internal Python package path `src/edgefs/` and firmware artifact name
> `edgefs_model.espdl` predate the framework rename to **EdgeTier-NER**; they
> are kept unchanged so the validated MCU path does not need a firmware
> re-verification pass.

**Repository:** [github.com/T01284/EdgeTier-NER](https://github.com/T01284/EdgeTier-NER)  
**License:** MIT (see `LICENSE`)

The LaTeX manuscript is maintained separately and is **not** in this repository.

## What is here

| Path | Purpose |
|------|---------|
| `src/edgefs/` | Models, data loaders, training losses, ONNX/ESP export |
| `configs/` | Dataset, model, and training YAML |
| `scripts/` | `train_full.py`, `train_fewshot.py`, `export_model.py`, edge eval helpers |
| `deploy/pi3b/` | Pi harness + pinned INT16 ONNX deploy bundle (**fulltestb s42**) |
| `deploy/esp32-s3/` | PlatformIO firmware + identity-skip INT8 `.espdl` |
| `outputs/runs/conll2003_full_fulltestb_s42/` | Pinned CoNLL checkpoint for the deploy path |
| `outputs/export/conll2003_full_fulltestb_s42/` | Same deploy bundle as `deploy/pi3b/assets/deploy/` (hashes match) |
| `deploy/results/` | Paper-cited metric summaries only (see that folder's README) |
| `data/toy/` | Tiny CoNLL for CPU smoke tests |
| `data/processed_remote/cluener/` | Open CLUENER processed splits |
| `tests/` | `pytest` smoke |

## Pinned environment versions

| Component | Version used for paper numbers |
|-----------|--------------------------------|
| Python | 3.10+ |
| PyTorch | ≥2.1 (see `requirements/base.txt`) |
| ONNX Runtime | ≥1.16 (`requirements/export.txt`; Pi bench used ORT 1.27 on-device) |
| PlatformIO `espressif32` | `^6.13.0` (`deploy/esp32-s3/platformio.ini`) |
| ESP-IDF | via PlatformIO / ESP-IDF 5.x board support in that platform release |
| esp-dl | `espressif/esp-dl` **^3.3.4** (`deploy/esp32-s3/src/idf_component.yml`; paper cites the Espressif repo accessed 2026-08-27) |

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements/export.txt
```

## Dataset licensing (read before preparing data)

| Dataset | In this repo? | How to reproduce |
|---------|---------------|------------------|
| **Toy** | Yes (`data/toy/`) | Local smoke only |
| **CLUENER** | Yes (`data/processed_remote/cluener/`) | Open Chinese NER set |
| **CoNLL-2003** | **No processed splits** | Obtain the official distribution, then `python scripts/prepare_datasets.py --config configs/dataset/conll2003.yaml` |
| **OntoNotes 5.0** | **No processed splits** | LDC2013T19 — obtain from LDC, then prepare via `configs/dataset/ontonotes.yaml` |

## Reproduce (local, no GPU)

```powershell
python scripts/prepare_datasets.py --config configs/dataset/toy.yaml
python scripts/train_full.py --config configs/train/full_toy.yaml --device cpu
pytest -q tests/test_smoke.py
```

## CoNLL deploy path

Pinned artifacts (do not mix with other export folders):

- Checkpoint: `outputs/runs/conll2003_full_fulltestb_s42/best.pt`
- Export / Pi assets: `outputs/export/conll2003_full_fulltestb_s42/deploy/` ≡ `deploy/pi3b/assets/deploy/`
- Host ONNX F1 on the truncated deploy split: `deploy/results/host_deploy_eval_fulltestb_s42.json` (**69.1**)
- Pi board F1 / latency / RSS: `deploy/results/pi3b_full_test.json` (**54.7±13.7 ms** mean±std; paper energy uses 54.7 ms)
- ESP32 stage latency: `deploy/results/esp32_stage_latency_full.json` (**1719±910 ms**; infer **615±2.6 ms**)
- Board power: `deploy/results/pi3b_power_summary.json`, `esp32_power_summary.json` → wall energy **181 mJ** (Pi) / **774 mJ** (ESP32), bounds **402 ms** / **12.5 sent/s**

Checkpoint `metrics.json` reports the **official eng.testb** score for that seed (~72.7 F1). The paper host ceiling DistilBERT-CRF is **86.6** (`fulltestb_transformers_summary.json`); compact encoder multi-seed mean is **71.4±0.9**. Deploy F1 on the truncated UART/Pi split is lower (~69.1) by design.

### Label vocabulary

Deploy `tags.json` has **8** tags (no `B-PER`): the vocabulary is induced from the prepared CoNLL split. Person mentions appear as `I-PER` in that encoding. Entity-level metrics use seqeval IOB2 on decoded tags. See `deploy/results/README.md`.

Re-export from the pinned checkpoint:

```powershell
python scripts/export_model.py --checkpoint outputs/runs/conll2003_full_fulltestb_s42/best.pt
```

Other datasets / ablations / few-shot: use matching YAML under `configs/train/` and retrain.

## Edge benchmarks

- **Pi 3B+:** `deploy/pi3b/README.md` — assets already include the s42 `model_emissions.onnx`. Set `PI_HOST` / `PI_PASSWORD` (see `.env.example`).
- **ESP32-S3:** `deploy/esp32-s3/README.md` — flash `esp32-s3-n16r8-espdl-uart-pure`, then `scripts/esp32_uart_eval.py`. The ESP32 ONNX/`.espdl` graph is the identity-skip rewrite and will **not** match the Pi ONNX hash.

Paper number → JSON map: `deploy/results/README.md`.

## Remote GPU (optional)

Copy `.env.example` → `.env`, create `configs/remote/gpu_server.yaml` from the example, then use `scripts/remote_*.py`.

## Publishing snapshot

```powershell
.\scripts\publish-to-github.ps1
```
