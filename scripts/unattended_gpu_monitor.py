#!/usr/bin/env python3
"""Unattended local watchdog: wait, verify, restart remote pipeline, periodic pull."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_ssh import connect, run

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "outputs" / "logs" / "local_monitor.log"
REMOTE_LOG = "/root/autodl-tmp/EdgeFS_NER/logs/gpu_experiments.log"
REMOTE_SCRIPT = "/root/EdgeFS_NER/scripts/run_gpu_experiments.sh"
REMOTE_NOHUP_LOG = "/root/autodl-tmp/EdgeFS_NER/logs/nohup_experiments.out"
PHASE1_MARKER = (
    "/root/autodl-tmp/EdgeFS_NER/outputs/runs/"
    "conll2003_fewshot_contrastive_only/fewshot_summary.json"
)
PHASE2_DONE_MARKER = "=== GPU experiments done"


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def remote_file_exists(client, path: str) -> bool:
    code, _, _ = run(client, f"bash -lc 'test -f {path!r} && echo yes'", timeout=20)
    return code == 0


def remote_train_running(client) -> bool:
    _, out, _ = run(
        client,
        "bash -lc 'ps aux | grep -E \"train_full|train_fewshot\" | grep -v grep'",
        timeout=20,
    )
    return bool(out.strip())


def remote_pipeline_running(client) -> bool:
    _, out, _ = run(
        client,
        "bash -lc 'ps aux | grep run_gpu_experiments | grep -v grep'",
        timeout=20,
    )
    return bool(out.strip())


def remote_log_tail(client, n: int = 5) -> str:
    _, out, _ = run(client, f"bash -lc 'tail -{n} {REMOTE_LOG} 2>/dev/null'", timeout=20)
    return out.strip()


def remote_phase2_done(client) -> bool:
    marker = "=== GPU experiments done"
    cmd = f"tail -100 {REMOTE_LOG} 2>/dev/null | grep -cF {marker!r} || true"
    _, out, _ = run(client, f"bash -lc {cmd!r}", timeout=20)
    try:
        return int(out.strip() or "0") > 0
    except ValueError:
        return False


def restart_remote_pipeline(client) -> None:
    run(
        client,
        "bash -lc 'pkill -f run_gpu_experiments.sh || true; "
        "pkill -f train_full.py || true; pkill -f train_fewshot.py || true'",
        timeout=30,
    )
    time.sleep(3)
    cmd = (
        "cd /root/EdgeFS_NER && source .venv/bin/activate && "
        "pip install -e . -q && "
        f"chmod +x {REMOTE_SCRIPT} && "
        "mkdir -p /root/autodl-tmp/EdgeFS_NER/logs && "
        f"nohup bash {REMOTE_SCRIPT} > {REMOTE_NOHUP_LOG} 2>&1 & echo STARTED"
    )
    code, out, err = run(client, f"bash -lc {cmd!r}", timeout=120)
    log(f"restart pipeline: code={code} out={out.strip()} err={err.strip()}")


def pull_results() -> int:
    log("pulling remote results ...")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "remote_pull_all.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    tail = (proc.stdout or proc.stderr).splitlines()[-3:]
    log("pull done: " + " | ".join(tail))
    return proc.returncode


def verify_phase1() -> bool:
    log("verifying phase 1 checkpoints ...")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "verify_experiment_steps.py"), "--phase", "1"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    for line in (proc.stdout or "").splitlines():
        log(line)
    ok = proc.returncode == 0
    log(f"phase 1 verify: {'PASS' if ok else 'FAIL'}")
    return ok


def phase2_already_started(client) -> bool:
    ontonotes_metrics = "/root/autodl-tmp/EdgeFS_NER/outputs/runs/ontonotes_full/metrics.json"
    if remote_file_exists(client, ontonotes_metrics):
        return True
    _, out, _ = run(
        client,
        "bash -lc 'ps aux | grep train_full | grep ontonotes | grep -v grep'",
        timeout=20,
    )
    if out.strip():
        return True
    _, out, _ = run(
        client,
        f"bash -lc 'grep -E \"full_ontonotes|cluener_full\" {REMOTE_LOG} | tail -1'",
        timeout=20,
    )
    return bool(out.strip())


def wait_phase1(client, poll_sec: int, timeout_sec: int) -> bool:
    log(f"waiting for phase 1 marker: {PHASE1_MARKER}")
    if remote_file_exists(client, PHASE1_MARKER):
        log("phase 1 marker already present")
        if phase2_already_started(client):
            log("phase 2 already started; will not interrupt running training")
            return True
        log("waiting for active training to stop before pausing pipeline shell")
        while remote_train_running(client):
            time.sleep(5)
        run(
            client,
            "bash -lc 'pkill -f run_gpu_experiments.sh || true'",
            timeout=20,
        )
        time.sleep(10)
        return True

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if remote_file_exists(client, PHASE1_MARKER):
            log("phase 1 marker found")
            if phase2_already_started(client):
                log("phase 2 already started; will not interrupt running training")
                return True
            log("waiting for active training to stop before pausing pipeline shell")
            while remote_train_running(client):
                time.sleep(5)
            run(
                client,
                "bash -lc 'pkill -f run_gpu_experiments.sh || true'",
                timeout=20,
            )
            time.sleep(10)
            return True
        if remote_train_running(client):
            tail = remote_log_tail(client, 1)
            log(f"training ... {tail[:120]}")
        else:
            log("no train process (between jobs or idle)")
        time.sleep(poll_sec)
    log("timeout waiting for phase 1")
    return False


def monitor_phase2(
    client,
    poll_sec: int,
    pull_every_sec: int,
    stall_sec: int,
    max_restarts: int,
) -> None:
    log("entering phase 2 unattended monitor")
    last_progress = time.time()
    last_pull = 0.0
    restarts = 0
    last_tail = ""

    while True:
        if remote_phase2_done(client):
            log("pipeline completed (experiments done)")
            pull_results()
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "verify_experiment_steps.py")],
                cwd=ROOT,
            )
            return

        tail = remote_log_tail(client, 3)
        if tail != last_tail:
            last_progress = time.time()
            last_tail = tail
            log(f"progress: {tail[:200]}")

        train_on = remote_train_running(client)
        pipe_on = remote_pipeline_running(client)

        if not train_on and not pipe_on:
            elapsed = time.time() - last_progress
            if elapsed > stall_sec and restarts < max_restarts:
                log(f"stall detected ({elapsed:.0f}s), restarting ({restarts + 1}/{max_restarts})")
                restart_remote_pipeline(client)
                restarts += 1
                last_progress = time.time()
            elif restarts >= max_restarts:
                log("max restarts reached; manual intervention needed")
                return

        if time.time() - last_pull >= pull_every_sec:
            pull_results()
            last_pull = time.time()

        time.sleep(poll_sec)


def main() -> int:
    parser = argparse.ArgumentParser(description="Unattended GPU experiment monitor")
    parser.add_argument("--wait-phase1", action="store_true", help="Wait for contrastive few-shot to finish")
    parser.add_argument("--skip-wait", action="store_true", help="Skip wait; verify immediately")
    parser.add_argument("--poll-sec", type=int, default=60, help="Remote status poll interval")
    parser.add_argument("--wait-timeout-sec", type=int, default=1800, help="Max wait for phase 1")
    parser.add_argument("--pull-every-sec", type=int, default=7200, help="Pull interval during phase 2")
    parser.add_argument("--stall-sec", type=int, default=900, help="Restart if idle this long")
    parser.add_argument("--max-restarts", type=int, default=5, help="Max auto restarts in phase 2")
    parser.add_argument("--no-restart", action="store_true", help="Monitor only; do not restart pipeline")
    args = parser.parse_args()

    client = connect()
    try:
        if args.wait_phase1 and not args.skip_wait:
            if not wait_phase1(client, args.poll_sec, args.wait_timeout_sec):
                return 1
            # Brief settle time after last few-shot episode.
            time.sleep(15)

        pull_results()
        if not verify_phase1():
            log("phase 1 verification failed; not starting unattended phase 2")
            return 1

        phase2_ok = "/root/autodl-tmp/EdgeFS_NER/logs/phase2_approved"
        run(client, f"bash -lc 'mkdir -p /root/autodl-tmp/EdgeFS_NER/logs && touch {phase2_ok}'", timeout=15)
        log(f"created phase 2 approval marker: {phase2_ok}")

        log("checking remote pipeline state ...")
        if remote_phase2_done(client):
            log("phase 2 already complete")
            pull_results()
            return 0

        if not remote_train_running(client) and not remote_pipeline_running(client):
            if args.no_restart:
                log("pipeline idle; --no-restart set, exiting")
                return 0
            log("starting phase 2 pipeline (nohup)")
            restart_remote_pipeline(client)
            time.sleep(20)
        elif phase2_already_started(client):
            log("phase 2 training in progress; monitor-only mode")
        else:
            log("pipeline already running; attaching monitor")

        monitor_phase2(
            client,
            poll_sec=args.poll_sec,
            pull_every_sec=args.pull_every_sec,
            stall_sec=args.stall_sec,
            max_restarts=args.max_restarts,
        )
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
