#!/usr/bin/env python3
"""Sync deploy/pi3b to Raspberry Pi 3 Model B Plus and run inference + benchmark."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]
PI3B = ROOT / "deploy" / "pi3b"
SHARED = ROOT / "deploy" / "shared" / "sample_inputs.json"
TEST_CONLL = ROOT / "data" / "processed" / "conll2003" / "test.conll"
if not TEST_CONLL.exists():
    TEST_CONLL = ROOT / "data" / "processed_remote" / "conll2003" / "test.conll"

HOST = os.environ.get("PI_HOST", "").strip()
USER = os.environ.get("PI_USER", "rpi").strip()
PASSWORD = os.environ.get("PI_PASSWORD", "")
REMOTE_ROOT = os.environ.get("PI_REMOTE_ROOT", "/home/rpi/EdgeTier-NER/deploy/pi3b")
REMOTE_DATA = os.environ.get(
    "PI_REMOTE_DATA", "/home/rpi/EdgeTier-NER/data/processed/conll2003"
)
SKIP_DIRS = {".venv", "__pycache__", ".git"}


def ensure_remote_dir(sftp: paramiko.SFTPClient, path: str) -> None:
    parts = path.strip("/").split("/")
    cur = ""
    for part in parts:
        cur += "/" + part
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def sync_tree(sftp: paramiko.SFTPClient, local_root: Path, remote_root: str) -> int:
    uploaded = 0
    for local in local_root.rglob("*"):
        if any(part in SKIP_DIRS for part in local.parts):
            continue
        rel = local.relative_to(local_root).as_posix()
        remote = f"{remote_root}/{rel}"
        if local.is_dir():
            try:
                sftp.stat(remote)
            except FileNotFoundError:
                ensure_remote_dir(sftp, remote)
        else:
            ensure_remote_dir(sftp, os.path.dirname(remote))
            sftp.put(str(local), remote)
            uploaded += 1
    return uploaded


def run_remote(client: paramiko.SSHClient, command: str, timeout: int = 600) -> tuple[int, str, str]:
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    code = stdout.channel.recv_exit_status()
    return code, out, err


def pull_results(client: paramiko.SSHClient, full: bool = False) -> Path:
    sftp = client.open_sftp()
    if full:
        remote_json = f"{REMOTE_ROOT}/results/pi3b_full_eval.json"
        local = ROOT / "deploy" / "pi3b" / "results" / "pi3b_full_eval.json"
    else:
        remote_json = f"{REMOTE_ROOT}/results/pi3b_benchmark.json"
        local = ROOT / "deploy" / "pi3b" / "results" / "pi3b_benchmark.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote_json, str(local))
    sftp.close()

    hardware: dict[str, str] = {}
    for key, cmd in {
        "board": "cat /proc/device-tree/model 2>/dev/null | tr -d '\\0'",
        "memory": "free -h | head -2",
        "cpu_cores": "nproc",
    }.items():
        _, out, _ = client.exec_command(cmd)
        hardware[key] = out.read().decode().strip()

    summary = json.loads(local.read_text(encoding="utf-8"))
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": f"{USER}@{HOST}",
        "hardware_probe": hardware,
        "ok": True,
        "summary": summary,
    }
    report_path = ROOT / "deploy" / "results" / ("pi3b_full_test.json" if full else "pi3b_test.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Pulled {local}")
    print(f"Wrote {report_path}")
    for key, val in hardware.items():
        print(f"{key}: {val}")
    return local


def main(full: bool = False) -> int:
    if not HOST or not PASSWORD:
        raise RuntimeError(
            "Set PI_HOST and PI_PASSWORD in the environment (see .env.example)."
        )
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"Connecting {USER}@{HOST} ...")
    client.connect(HOST, username=USER, password=PASSWORD, timeout=15, allow_agent=False, look_for_keys=False)

    for cmd in ["uname -a", "python3 --version", f"ls -la {REMOTE_ROOT}/assets/deploy 2>/dev/null || echo NO_ASSETS"]:
        _, out, err = run_remote(client, cmd)
        print(f"$ {cmd}\n{out or err}")

    sftp = client.open_sftp()
    uploaded = sync_tree(sftp, PI3B, REMOTE_ROOT)
    shared_remote = str(Path(REMOTE_ROOT).parent / "shared")
    ensure_remote_dir(sftp, shared_remote)
    sftp.put(str(SHARED), f"{shared_remote}/sample_inputs.json")
    uploaded += 1

    if full:
        ensure_remote_dir(sftp, REMOTE_DATA)
        if TEST_CONLL.exists():
            sftp.put(str(TEST_CONLL), f"{REMOTE_DATA}/test.conll")
            uploaded += 1

    sftp.close()
    print(f"Uploaded {uploaded} files")

    if full:
        setup_cmd = f"""
set -e
cd {REMOTE_ROOT}
if [ ! -d .venv ]; then python3 -m venv .venv; fi
source .venv/bin/activate
pip install --upgrade pip --default-timeout=180 --retries 5 -q
pip install -r requirements.txt --default-timeout=180 --retries 5
python -c "import onnxruntime; print('onnxruntime', onnxruntime.__version__)"
echo '=== full eval ==='
PYTHONPATH={REMOTE_ROOT} \\
  python scripts/run_full_eval.py \\
    --deploy-dir assets/deploy \\
    --conll-test {REMOTE_DATA}/test.conll \\
    --output results/pi3b_full_eval.json
cat results/pi3b_full_eval.json
"""
        timeout = 7200
    else:
        setup_cmd = f"""
set -e
cd {REMOTE_ROOT}
if [ ! -d .venv ]; then python3 -m venv .venv; fi
source .venv/bin/activate
pip install --upgrade pip --default-timeout=180 --retries 5 -q
pip install -r requirements.txt --default-timeout=180 --retries 5
python -c "import onnxruntime; print('onnxruntime', onnxruntime.__version__)"
echo '=== inference ==='
python scripts/run_inference.py --deploy-dir assets/deploy --text "john works in paris"
echo '=== benchmark ==='
python scripts/run_benchmark.py --deploy-dir assets/deploy --output results/pi3b_benchmark.json
cat results/pi3b_benchmark.json
"""
        timeout = 900

    code, out, err = run_remote(client, setup_cmd, timeout=timeout)
    print("=== remote output ===")
    print(out)
    if err:
        print("=== remote stderr ===")
        print(err)
    if code == 0:
        pull_results(client, full=full)
    client.close()
    return code


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pull-only", action="store_true", help="Only pull benchmark JSON from Pi")
    parser.add_argument("--full", action="store_true", help="Run full CoNLL test eval on Pi")
    args = parser.parse_args()
    if args.pull_only:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(HOST, username=USER, password=PASSWORD, timeout=15, allow_agent=False, look_for_keys=False)
        pull_results(client, full=args.full)
        client.close()
        sys.exit(0)
    sys.exit(main(full=args.full))
