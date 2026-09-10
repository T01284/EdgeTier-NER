#!/usr/bin/env python3
"""Watch gpu remainder batch: pull metrics, update paper tables, optional shutdown."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_ssh import connect, run

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "outputs" / "logs" / "remainder_monitor.log"
BASE = "/root/autodl-tmp/EdgeFS_NER"
REMOTE_LOG = f"{BASE}/logs/gpu_paper_gap_remainder.log"
DONE = "=== remainder done"

TARGETS = [
    ("cluener_distilbert", "metrics.json"),
    ("cluener_tinybert", "metrics.json"),
    ("conll2003_ablation_wordlevel", "metrics.json"),
    ("conll2003_distill", "metrics.json"),
]

# Also refresh these for table backfill
EXTRA_PULL = [
    "ontonotes_full",
    "ontonotes_bilstm",
    "ontonotes_bert",
    "ontonotes_distilbert",
    "ontonotes_tinybert",
    "cluener_full",
    "cluener_bilstm",
    "cluener_bert",
    "conll2003_fewshot_5w1s",
    "conll2003_fewshot_5w5s",
    "ontonotes_fewshot_4w1s",
    "ontonotes_fewshot_4w5s",
    "conll2003_fewshot_protobert_4w1s",
    "conll2003_fewshot_protobert_4w5s",
    "conll2003_fewshot_nnshot_4w1s",
    "conll2003_fewshot_nnshot_4w5s",
    "conll2003_fewshot_structshot_4w1s",
    "conll2003_fewshot_structshot_4w5s",
]


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def pull_run(sftp, name: str) -> None:
    remote = f"{BASE}/outputs/runs/{name}"
    local = ROOT / "outputs" / "runs" / name
    local.mkdir(parents=True, exist_ok=True)
    try:
        for entry in sftp.listdir_attr(remote):
            if entry.filename.endswith((".json", ".log")) or entry.filename == "metrics.json":
                if entry.st_mode & 0o40000:
                    continue
                rpath = f"{remote}/{entry.filename}"
                lpath = local / entry.filename
                try:
                    sftp.get(rpath, str(lpath))
                except Exception as exc:
                    log(f"skip {rpath}: {exc}")
    except FileNotFoundError:
        log(f"missing remote run {name}")


def targets_done(client) -> tuple[int, list[str]]:
    pending = []
    done = 0
    for name, fname in TARGETS:
        path = f"{BASE}/outputs/runs/{name}/{fname}"
        _, out, _ = run(client, f"test -f {path} && echo YES || echo NO", timeout=12)
        if out.strip() == "YES":
            done += 1
        else:
            pending.append(name)
    return done, pending


def remainder_done(client) -> bool:
    _, out, _ = run(client, f"grep -cF {DONE!r} {REMOTE_LOG} 2>/dev/null || true", timeout=15)
    try:
        return int((out or "0").strip()) > 0
    except ValueError:
        return False


def fill_tables() -> None:
    log("collecting metrics + updating tables")
    for script in ("collect_gpu_metrics.py",):
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        log(f"{script} exit={proc.returncode}")
        if proc.stdout:
            log(proc.stdout.splitlines()[-3:][0] if proc.stdout.splitlines() else "")
        if proc.returncode != 0 and proc.stderr:
            log(proc.stderr[-400:])

    # Also rewrite paper/data/experiment_results.json from all_gpu_metrics
    metrics_path = ROOT / "deploy" / "results" / "all_gpu_metrics.json"
    if not metrics_path.exists():
        return
    data = json.loads(metrics_path.read_text(encoding="utf-8"))

    def pct(ds: str, run: str) -> float | None:
        m = data.get("full", {}).get(ds, {}).get(run)
        if not m or "f1" not in m:
            return None
        return round(float(m["f1"]) * 100, 1)

    def few(key: str) -> dict | None:
        m = data.get("fewshot", {}).get(key)
        if not m:
            return None
        return {"mean": round(m["mean_f1"] * 100, 2), "std": round(m["std_f1"] * 100, 2)}

    summary = {
        "pulled_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "source": "outputs/runs/*/metrics.json (remote remainder + paper-gap)",
        "cluener_distilbert_backbone": "distilbert-base-multilingual-cased",
        "main_results_test_f1_percent": {
            "BiLSTM-CRF": {
                "conll2003": pct("conll2003", "conll2003_bilstm"),
                "ontonotes": pct("ontonotes", "ontonotes_bilstm"),
                "cluener": pct("cluener", "cluener_bilstm"),
            },
            "BERT-base-CRF": {
                "conll2003": pct("conll2003", "conll2003_bert"),
                "ontonotes": pct("ontonotes", "ontonotes_bert"),
                "cluener": pct("cluener", "cluener_bert"),
            },
            "DistilBERT-CRF": {
                "conll2003": pct("conll2003", "conll2003_distilbert"),
                "ontonotes": pct("ontonotes", "ontonotes_distilbert"),
                "cluener": pct("cluener", "cluener_distilbert"),
            },
            "TinyBERT": {
                "conll2003": pct("conll2003", "conll2003_tinybert"),
                "ontonotes": pct("ontonotes", "ontonotes_tinybert"),
                "cluener": pct("cluener", "cluener_tinybert"),
            },
            "Ours": {
                "conll2003": pct("conll2003", "conll2003_full"),
                "ontonotes": pct("ontonotes", "ontonotes_full"),
                "cluener": pct("cluener", "cluener_full"),
            },
        },
        "fewshot_f1_percent": {
            "conll2003_5w1s": few("conll2003_5w1s"),
            "conll2003_5w5s": few("conll2003_5w5s"),
            "ontonotes_4w1s": few("ontonotes_4w1s") or few("ontonotes_5w1s"),
            "ontonotes_4w5s": few("ontonotes_4w5s") or few("ontonotes_5w5s"),
            "protobert_4w1s": few("conll2003_protobert_4w1s"),
            "protobert_4w5s": few("conll2003_protobert_4w5s"),
            "nnshot_4w1s": few("conll2003_nnshot_4w1s"),
            "nnshot_4w5s": few("conll2003_nnshot_4w5s"),
            "structshot_4w1s": few("conll2003_structshot_4w1s"),
            "structshot_4w5s": few("conll2003_structshot_4w5s"),
        },
        "ablation_test_f1_percent": {
            "wordlevel": round(data["ablation"]["wordlevel"]["f1"] * 100, 1)
            if data.get("ablation", {}).get("wordlevel", {}).get("f1") is not None
            else None,
            "distill": round(data["ablation"]["distill"]["f1"] * 100, 1)
            if data.get("ablation", {}).get("distill", {}).get("f1") is not None
            else None,
        },
    }
    out = ROOT / "paper" / "data" / "experiment_results.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"wrote {out}")


def update_ablation_table() -> None:
    metrics_path = ROOT / "deploy" / "results" / "all_gpu_metrics.json"
    if not metrics_path.exists():
        return
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    abl = data.get("ablation", {})
    t4 = ROOT / "paper" / "tables" / "table4-ablation.tex"
    text = t4.read_text(encoding="utf-8")
    if abl.get("wordlevel", {}).get("f1") is not None:
        w = f"{abl['wordlevel']['f1'] * 100:.1f}"
        text = re.sub(
            r"\$-\$ compact representation & ---",
            f"$-$ compact representation & {w}",
            text,
            count=1,
        )
    if abl.get("distill", {}).get("f1") is not None:
        d = f"{abl['distill']['f1'] * 100:.1f}"
        text = re.sub(
            r"\$-\$ distillation & ---",
            f"$-$ distillation & {d}",
            text,
            count=1,
        )
    t4.write_text(text, encoding="utf-8")
    log(f"updated {t4}")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-sec", type=int, default=300)
    parser.add_argument("--no-shutdown", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    log("remainder monitor started")
    while True:
        try:
            client = connect()
        except Exception as exc:
            log(f"SSH fail: {exc}")
            if args.once:
                return 1
            time.sleep(args.poll_sec)
            continue

        try:
            done_n, pending = targets_done(client)
            _, ps, _ = run(
                client,
                "ps aux | grep '[t]rain_full.py' | head -1",
                timeout=12,
            )
            train = bool(ps.strip())
            cfg = ""
            m = re.search(r"--config\s+(\S+)", ps or "")
            if m:
                cfg = Path(m.group(1)).name
            _, epoch_info, _ = run(
                client,
                "tr '\\r' '\\n' < "
                f"{REMOTE_LOG} | grep -E 'epoch [0-9]+:' | tail -1 "
                "| sed -E 's/.*epoch ([0-9]+):[[:space:]]*([0-9]+)%.*/epoch \\1 \\2%/' "
                "|| echo 'epoch ?'",
                timeout=20,
            )
            job = cfg or (pending[0] if pending else "none")
            log(
                f"{done_n}/4 targets | train={'yes' if train else 'no'} "
                f"| job={job} | {(epoch_info or '').strip()} | pending={pending}"
            )

            finished = remainder_done(client) or done_n == 4
            if finished:
                log("remainder complete — pulling & filling tables")
                sftp = client.open_sftp()
                for name, _ in TARGETS:
                    pull_run(sftp, name)
                for name in EXTRA_PULL:
                    pull_run(sftp, name)
                sftp.close()
                fill_tables()
                update_ablation_table()
                if not args.no_shutdown:
                    log("sending remote shutdown")
                    run(client, "shutdown -h now", timeout=20)
                log("monitor exiting")
                return 0
        finally:
            client.close()

        if args.once:
            return 0
        time.sleep(args.poll_sec)


if __name__ == "__main__":
    raise SystemExit(main())
