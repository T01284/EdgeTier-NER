#!/usr/bin/env python3
"""Host-driven ESP32-S3 UART eval: full CoNLL test F1 + latency on device."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from edgefs.data.conll import read_conll
from edgefs.evaluation.metrics import compute_ner_metrics
from write_esp32_artifact_manifest import load_manifest

RESPONSE = re.compile(
    r"OK latency_us=(?P<lat>\d+)"
    r"(?: embed_us=(?P<embed>\d+) espdl_us=(?P<espdl>\d+) viterbi_us=(?P<viterbi>\d+))?"
    r" len=(?P<len>\d+) tags=(?P<tags>[\d,]+) "
    r"heap=(?P<heap>\d+) psram=(?P<psram>\d+)"
)
ESP32_DIR = ROOT / "deploy" / "esp32-s3"
DEFAULT_TAGS = ROOT / "outputs" / "export" / "conll2003_full" / "deploy" / "tags.json"
DEFAULT_PIO_ENV = "esp32-s3-n16r8-espdl-uart-pure"


def load_tags(tags_path: Path) -> list[str]:
    obj = json.loads(tags_path.read_text(encoding="utf-8"))
    return obj["tags"]


def upload_firmware(port: str, env: str = DEFAULT_PIO_ENV) -> None:
    cmd = ["pio", "run", "-e", env, "-t", "upload", "--upload-port", port]
    proc = subprocess.run(cmd, cwd=str(ESP32_DIR), capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(f"upload failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-2000:]}")


def wait_ready(ser, timeout_s: float = 90.0) -> None:
    deadline = time.time() + timeout_s
    ser.reset_input_buffer()
    ser.write(b"PING\r\n")
    ser.flush()
    while time.time() < deadline:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if not line:
            continue
        if "READY board=" in line:
            print(f"Device ready: {line}", flush=True)
            # Drain any duplicate READY/PONG lines left in the USB buffer.
            ser.reset_input_buffer()
            time.sleep(0.05)
            ser.reset_input_buffer()
            return
        if line == "PONG":
            ser.write(b"PING\r\n")
            ser.flush()
    raise TimeoutError("ESP32 did not emit READY")


def send_eval(ser, text: str, timeout_s: float = 90.0) -> dict:
    ser.reset_input_buffer()
    payload = f"EVAL\t{text}\r\n".encode("utf-8")
    ser.write(payload)
    ser.flush()
    t0 = time.time()
    deadline = t0 + timeout_s
    while time.time() < deadline:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if not line:
            continue
        if line.startswith("ERR"):
            raise RuntimeError(line)
        # Ignore stale READY left in USB buffer right after send; later READY = reboot.
        if line.startswith("READY board="):
            if time.time() - t0 < 2.0:
                continue
            raise TimeoutError("device rebooted mid-eval (READY seen)")
        m = RESPONSE.search(line)
        if m:
            tag_ids = [int(x) for x in m.group("tags").split(",") if x]
            out = {
                "latency_us": int(m.group("lat")),
                "len": int(m.group("len")),
                "tag_ids": tag_ids,
                "free_heap_bytes": int(m.group("heap")),
                "free_psram_bytes": int(m.group("psram")),
            }
            if m.group("embed") is not None:
                out["embed_us"] = int(m.group("embed"))
                out["espdl_us"] = int(m.group("espdl"))
                out["viterbi_us"] = int(m.group("viterbi"))
            return out
    raise TimeoutError(f"no response for sentence: {text[:80]!r}")


def reconnect_device(port: str):
    import serial

    ser = serial.Serial(port, 115200, timeout=2.0)
    ser.setDTR(False)
    ser.setRTS(True)
    time.sleep(0.1)
    ser.setRTS(False)
    time.sleep(6.0)
    ser.reset_input_buffer()
    wait_ready(ser, timeout_s=90.0)
    return ser


def send_eval_with_retry(port: str, ser, text: str) -> tuple[dict | None, object]:
    import serial

    for attempt in range(3):
        try:
            if ser is None or not getattr(ser, "is_open", False):
                print(f"  reconnecting serial before attempt {attempt + 1}", flush=True)
                ser = reconnect_device(port)
            return send_eval(ser, text), ser
        except TimeoutError:
            print(f"  timeout attempt {attempt + 1}/3; reconnecting", flush=True)
            try:
                ser.close()
            except Exception:
                pass
            ser = reconnect_device(port)
        except serial.SerialException as exc:
            print(f"  serial error attempt {attempt + 1}/3: {exc}; reconnecting", flush=True)
            try:
                ser.close()
            except Exception:
                pass
            ser = reconnect_device(port)
    return None, ser


ESP32_MAX_TOKENS = 64  # Longer sequences can hang on-device (5/1722 sentences affected)


def run_eval(
    port: str,
    conll_path: Path,
    tags_path: Path,
    skip_upload: bool = False,
    max_sentences: int | None = None,
    start_index: int = 0,
    checkpoint_every: int = 50,
    checkpoint_path: Path | None = None,
    resume_checkpoint: bool = False,
    pio_env: str = DEFAULT_PIO_ENV,
) -> dict:
    import serial

    if not skip_upload:
        print(f"Uploading UART eval firmware ({pio_env}) ...")
        upload_firmware(port, env=pio_env)

    tag_names = load_tags(tags_path)
    sentences = read_conll(conll_path)

    y_true: list[list[str]] = []
    y_pred: list[list[str]] = []
    latencies_ms: list[float] = []
    embed_us: list[int] = []
    espdl_us: list[int] = []
    viterbi_us: list[int] = []
    samples: list[dict] = []
    skipped: list[int] = []

    if resume_checkpoint and checkpoint_path and checkpoint_path.exists():
        prior = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        start_index = max(start_index, int(prior.get("completed", 0)))
        y_true = list(prior.get("y_true", []))
        y_pred = list(prior.get("y_pred", []))
        latencies_ms = list(prior.get("latencies_ms", []))
        embed_us = list(prior.get("embed_us", []))
        espdl_us = list(prior.get("espdl_us", []))
        viterbi_us = list(prior.get("viterbi_us", []))
        print(
            f"Resuming from checkpoint completed={start_index} "
            f"(loaded {len(y_true)} sentences)"
        )

    if start_index:
        sentences = sentences[start_index:]
    if max_sentences is not None:
        sentences = sentences[:max_sentences]

    ser = serial.Serial(port, 115200, timeout=2.0)
    ser.setDTR(False)
    ser.setRTS(True)
    time.sleep(0.1)
    ser.setRTS(False)
    time.sleep(6.0)
    ser.reset_input_buffer()
    wait_ready(ser, timeout_s=90.0)

    t0 = time.perf_counter()
    truncated_count = 0
    for idx, sent in enumerate(sentences):
        global_idx = start_index + idx
        tokens = sent.tokens
        gold = sent.tags
        if len(tokens) > ESP32_MAX_TOKENS:
            tokens = tokens[:ESP32_MAX_TOKENS]
            gold = gold[:ESP32_MAX_TOKENS]
            truncated_count += 1
        text = " ".join(tokens)
        if (global_idx + 1) % 5 == 0 or idx < 3:
            print(f"  … sending [{global_idx + 1}/1722] tokens={len(tokens)}", flush=True)
        try:
            result, ser = send_eval_with_retry(port, ser, text)
        except TimeoutError as exc:
            skipped.append(global_idx)
            print(f"  skip sentence_index={global_idx}: {exc}")
            continue
        if result is None:
            skipped.append(global_idx)
            print(f"  skip sentence_index={global_idx}: no response after retries")
            continue
        pred = [tag_names[i] for i in result["tag_ids"]]
        gold = gold[: len(pred)]
        y_true.append(gold)
        y_pred.append(pred)
        latencies_ms.append(result["latency_us"] / 1000.0)
        if "embed_us" in result:
            embed_us.append(result["embed_us"])
            espdl_us.append(result["espdl_us"])
            viterbi_us.append(result["viterbi_us"])
        if idx < 3 or (global_idx + 1) % 25 == 0:
            rec = {
                "sentence_id": global_idx,
                "latency_ms": latencies_ms[-1],
                "num_tokens": len(sent.tokens),
            }
            if "embed_us" in result:
                rec.update(
                    {
                        "embed_us": result["embed_us"],
                        "espdl_us": result["espdl_us"],
                        "viterbi_us": result["viterbi_us"],
                    }
                )
            samples.append(rec)
        if (global_idx + 1) % 25 == 0:
            stage = ""
            if "embed_us" in result:
                stage = (
                    f" embed={result['embed_us']/1000:.1f}"
                    f" espdl={result['espdl_us']/1000:.1f}"
                    f" vit={result['viterbi_us']/1000:.1f}"
                )
            print(
                f"  [{global_idx + 1}/1722] last_latency_ms={latencies_ms[-1]:.1f}{stage}",
                flush=True,
            )
        if checkpoint_path and ((global_idx + 1) % checkpoint_every == 0 or (global_idx + 1) % 5 == 0):
            partial = {
                "start_index": 0,
                "completed": global_idx + 1,
                "latencies_ms": latencies_ms,
                "embed_us": embed_us,
                "espdl_us": espdl_us,
                "viterbi_us": viterbi_us,
                "y_true": y_true,
                "y_pred": y_pred,
            }
            checkpoint_path.write_text(json.dumps(partial), encoding="utf-8")
            if (global_idx + 1) % checkpoint_every == 0:
                print(f"  checkpoint saved at {global_idx + 1}", flush=True)

    try:
        ser.write(b"QUIT\n")
    except Exception:
        pass
    try:
        ser.close()
    except Exception:
        pass

    metrics = compute_ner_metrics(y_true, y_pred)
    ordered = sorted(latencies_ms)
    artifact = load_manifest(ESP32_DIR / "models")

    def _mean_ms(us_list: list[int]) -> float | None:
        if not us_list:
            return None
        return (sum(us_list) / len(us_list)) / 1000.0

    stage = None
    if embed_us:
        emb_ms = _mean_ms(embed_us)
        esp_ms = _mean_ms(espdl_us)
        vit_ms = _mean_ms(viterbi_us)
        total = (emb_ms or 0.0) + (esp_ms or 0.0) + (vit_ms or 0.0)
        stage = {
            "n": len(embed_us),
            "embed_ms_mean": emb_ms,
            "espdl_ms_mean": esp_ms,
            "viterbi_ms_mean": vit_ms,
            "share_embed": round((emb_ms or 0.0) / total, 4) if total else None,
            "share_espdl": round((esp_ms or 0.0) / total, 4) if total else None,
            "share_viterbi": round((vit_ms or 0.0) / total, 4) if total else None,
            "embed_us_per_sentence": embed_us,
            "espdl_us_per_sentence": espdl_us,
            "viterbi_us_per_sentence": viterbi_us,
        }

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": "esp32_s3_int8_uart",
        "pio_env": pio_env,
        "artifact_manifest": artifact,
        "artifact_tag": (artifact or {}).get("artifact_tag"),
        "espdl_sha256": (artifact or {}).get("primary_espdl_sha256"),
        "port": port,
        "conll_path": str(conll_path),
        "num_sentences": len(y_true),
        "truncated_over_max_tokens": truncated_count,
        "esp32_max_tokens": ESP32_MAX_TOKENS,
        "start_index": 0 if resume_checkpoint else start_index,
        "resumed_from": start_index if resume_checkpoint or start_index else 0,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "metrics": metrics,
        "latency_ms_mean": sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0,
        "latency_ms_p50": ordered[len(ordered) // 2] if ordered else 0.0,
        "latency_ms_p95": ordered[int(len(ordered) * 0.95)] if ordered else 0.0,
        "latency_ms_per_sentence": latencies_ms,
        "stage_latency": stage,
        "skipped_sentence_ids": skipped,
        "y_true": y_true,
        "y_pred": y_pred,
        "sample_records": samples,
        "host_reference_f1": 0.702,
        "f1_drop_vs_host": round(0.702 - metrics["f1"], 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ESP32 UART full CoNLL eval")
    parser.add_argument("--port", default="COM32")
    parser.add_argument(
        "--conll-test",
        default=str(ROOT / "data/processed_remote/conll2003/test.conll"),
    )
    parser.add_argument("--tags", default=str(DEFAULT_TAGS))
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--pio-env", default=DEFAULT_PIO_ENV)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--start", type=int, default=0, help="Resume from sentence index")
    parser.add_argument("--resume-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument(
        "--output",
        default=str(ROOT / "deploy/results/esp32_full_eval.json"),
    )
    args = parser.parse_args()

    summary = run_eval(
        port=args.port,
        conll_path=Path(args.conll_test),
        tags_path=Path(args.tags),
        skip_upload=args.skip_upload,
        max_sentences=args.max_sentences,
        start_index=args.start,
        checkpoint_every=args.checkpoint_every,
        checkpoint_path=Path(args.output).with_suffix(".checkpoint.json"),
        resume_checkpoint=args.resume_checkpoint,
        pio_env=args.pio_env,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
