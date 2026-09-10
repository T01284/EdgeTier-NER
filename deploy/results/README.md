# Paper evidence logs

Only summary JSON cited by the manuscript tables and figures is kept in the
public repository. Intermediate smoke runs, checkpoints, probe dumps, and
abandoned configurations are intentionally omitted.

## Deploy path identity

The Pi INT16 numbers (69.1 F1 @ 54.7±13.7 ms) come from the **same** export
bundle:

`outputs/export/conll2003_full_fulltestb_s42/deploy/`

which is **byte-identical** to `deploy/pi3b/assets/deploy/` (including
`model_emissions.onnx` and `crf_transitions.npy`). Host ONNX evaluation on
that bundle is `host_deploy_eval_fulltestb_s42.json` (69.1 F1); board timing
and on-device F1 are in `pi3b_full_test.json`.

Board latency varies run-to-run (~±2%, thermal/scheduler); F1 is bit-identical
across runs.

ESP32 `deploy/esp32-s3/models/model_emissions.onnx` differs by design: it is
the identity-skip / densified MCU rewrite, not the Pi ORT graph.

| Manuscript use | Artifact | What it supports |
|----------------|----------|------------------|
| Deployment table | `pi3b_full_test.json`, `host_deploy_eval_fulltestb_s42.json`, `esp32_stage_latency_full.json`, `espdl_test.json`, `fulltestb_transformers_summary.json` | Pi/ESP32 latency; DistilBERT host F1 86.6; OOM/Unsup. |
| Stage latency | `esp32_stage_latency_full.json` | Stage means/std on ESP32 pure INT8 path |
| Host ceiling / gap | `fulltestb_multiseed_summary.json`, `fulltestb_transformers_summary.json` | Ours 71.4±0.9; DistilBERT 86.6 |

## Label vocabulary note

Deploy `tags.json` lists **8** labels (no `B-PER`). The tag set is induced from
the prepared CoNLL split (`TagSet.from_sentences`); person entities in that
encoding appear under `I-PER`. Entity-level scores still use seqeval **IOB2**
scheme on decoded tags. This is the deployed SKU; it is not a silent omission
of a ninth class at export time.

Few-shot results are reproduced from `configs/train/fewshot_*.yaml` via
`scripts/train_fewshot.py`; no intermediate episode dumps are shipped.
