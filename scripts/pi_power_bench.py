#!/usr/bin/env python3
"""Remote Pi power-measurement harness: idle hold then continuous ONNX+Viterbi load.

Prints clear phase banners for hand-read USB / supply power meters.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import paramiko

HOST = os.environ.get("PI_HOST", "").strip()
USER = os.environ.get("PI_USER", "rpi").strip()
PASSWORD = os.environ.get("PI_PASSWORD", "")
REMOTE_ROOT = os.environ.get("PI_REMOTE_ROOT", "/home/rpi/EdgeTier-NER/deploy/pi3b")
REMOTE_PY = f"{REMOTE_ROOT}/.venv/bin/python"


def banner(title: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n  {title}\n{line}", flush=True)


def connect() -> paramiko.SSHClient:
    if not HOST or not PASSWORD:
        raise RuntimeError(
            "Set PI_HOST and PI_PASSWORD in the environment (see .env.example)."
        )
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(
        HOST,
        username=USER,
        password=PASSWORD,
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
        allow_agent=False,
        look_for_keys=False,
    )
    return c


def run(c: paramiko.SSHClient, cmd: str, timeout: int = 60) -> tuple[str, str, int]:
    _, o, e = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode("utf-8", errors="replace")
    err = e.read().decode("utf-8", errors="replace")
    rc = o.channel.recv_exit_status()
    return out, err, rc


def remote_probe(c: paramiko.SSHClient) -> None:
    out, err, rc = run(
        c,
        "hostname; uptime; "
        f"test -x {REMOTE_PY} && echo HAS_VENV || echo NO_VENV; "
        f"test -f {REMOTE_ROOT}/assets/deploy/model_emissions.onnx && echo HAS_ONNX || echo NO_ONNX",
    )
    print(out, flush=True)
    if "NO_VENV" in out or "NO_ONNX" in out or rc != 0:
        raise RuntimeError(f"Pi probe failed:\n{out}\n{err}")


def hold_idle(seconds: float) -> None:
    banner(f"PHASE IDLE — read Pi meter NOW ({seconds:.0f}s)")
    print(
        "Pi is up; no EdgeFS inference. Prefer reading after load average settles.\n"
        "Note idle power (W / mW / or V+A). Say what rail you measure (5V USB / PSU).",
        flush=True,
    )
    t0 = time.time()
    c = connect()
    try:
        while time.time() - t0 < seconds:
            left = seconds - (time.time() - t0)
            out, _, _ = run(c, "uptime; cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || true")
            temp = ""
            for line in out.splitlines():
                if line.strip().isdigit():
                    temp = f" temp={int(line.strip())/1000:.1f}C"
            print(f"  idle … {left:5.1f}s left | {out.splitlines()[0] if out else '?'}{temp}", flush=True)
            time.sleep(min(5.0, max(0.5, left)))
    finally:
        c.close()
    banner("PHASE IDLE — DONE (write down idle reading)")


def run_active(seconds: float, text: str) -> dict:
    banner(f"PHASE ACTIVE — read Pi meter NOW ({seconds:.0f}s continuous infer)")
    print(f"Sentence: {text!r}", flush=True)

    # Remote loop: keep one process warm to avoid restart overhead.
    remote_script = f"""
import json, time, sys
sys.path.insert(0, '{REMOTE_ROOT}')
from edgefs_pi.onnx_runner import DeployAssets, OnnxNERRunner
runner = OnnxNERRunner(DeployAssets.load('{REMOTE_ROOT}/assets/deploy'))
text = {text!r}
deadline = time.time() + {seconds}
n = 0
lat = []
# warmup
for _ in range(3):
    runner.predict(text)
while time.time() < deadline:
    t0 = time.perf_counter()
    runner.predict(text)
    lat.append((time.perf_counter() - t0) * 1000.0)
    n += 1
    if n % 20 == 0:
        left = deadline - time.time()
        print(f'PROGRESS n={{n}} last={{lat[-1]:.1f}}ms left={{left:.0f}}s', flush=True)
mean = sum(lat)/len(lat) if lat else 0.0
print(json.dumps({{'n': n, 'latency_ms_mean': mean, 'latency_ms_p50': sorted(lat)[len(lat)//2] if lat else 0}}))
"""
    c = connect()
    try:
        cmd = f"cd {REMOTE_ROOT} && {REMOTE_PY} - <<'PY'\n{remote_script}\nPY"
        # Stream stdout so local banners stay aligned with remote progress.
        transport = c.get_transport()
        assert transport is not None
        chan = transport.open_session()
        chan.set_combine_stderr(True)
        chan.exec_command(cmd)
        buf = ""
        summary = None
        while True:
            if chan.recv_ready():
                chunk = chan.recv(4096).decode("utf-8", errors="replace")
                buf += chunk
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("PROGRESS"):
                        print(f"  {line}", flush=True)
                    elif line.startswith("{"):
                        import json

                        summary = json.loads(line)
                        print(f"  summary {summary}", flush=True)
                    else:
                        print(f"  remote: {line}", flush=True)
            if chan.exit_status_ready():
                # drain
                while chan.recv_ready():
                    buf += chan.recv(4096).decode("utf-8", errors="replace")
                break
            time.sleep(0.2)
        rc = chan.recv_exit_status()
        if summary is None:
            for line in buf.splitlines():
                if line.strip().startswith("{"):
                    import json

                    summary = json.loads(line.strip())
        if rc != 0 and summary is None:
            raise RuntimeError(f"active load failed rc={rc}\n{buf}")
    finally:
        c.close()

    banner("PHASE ACTIVE — DONE (write down active reading)")
    print(f"Active summary: {summary}", flush=True)
    return summary or {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--idle-s", type=float, default=45.0)
    ap.add_argument("--active-s", type=float, default=90.0)
    ap.add_argument(
        "--text",
        default="John works in Paris and Mary visited London yesterday",
    )
    ap.add_argument("--skip-idle", action="store_true")
    ap.add_argument("--skip-active", action="store_true")
    args = ap.parse_args()

    banner("CONNECT / PROBE")
    c = connect()
    try:
        remote_probe(c)
    finally:
        c.close()

    if not args.skip_idle:
        hold_idle(args.idle_s)
    if not args.skip_active:
        run_active(args.active_s, args.text)

    banner("REPORT BACK")
    print(
        "Please reply with:\n"
        "  1) Pi idle power\n"
        "  2) Pi active power (same units)\n"
        "  3) meter point (USB 5V / official PSU / inline) and model\n"
        "Optional: Pi still cooler idle after longer settle — we can re-run idle-only.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
