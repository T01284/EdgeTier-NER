#!/usr/bin/env python3
"""ESP32 power-measurement harness: idle hold then continuous EVAL load.

Prints clear phase banners so a USB power meter can be read by hand.
Usage:
  python scripts/esp32_power_bench.py --port COM32 --idle-s 30 --active-s 60
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from esp32_uart_eval import reconnect_device, send_eval  # noqa: E402


def banner(title: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n  {title}\n{line}", flush=True)


def hold_idle(ser, seconds: float) -> None:
    banner(f"PHASE IDLE — read meter NOW ({seconds:.0f}s hold)")
    print("Device is READY; no EVAL traffic. Note idle power (mW or mA@V).", flush=True)
    t0 = time.time()
    while time.time() - t0 < seconds:
        left = seconds - (time.time() - t0)
        print(f"  idle … {left:5.1f}s left", flush=True)
        # Keep link soft-alive without inference load.
        try:
            ser.write(b"PING\r\n")
            ser.flush()
            deadline = time.time() + 1.0
            while time.time() < deadline:
                line = ser.readline().decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if line in ("PONG",) or line.startswith("READY"):
                    break
        except Exception as exc:
            print(f"  ping warn: {exc}", flush=True)
        time.sleep(max(0.0, min(4.0, left - 0.05)))
    banner("PHASE IDLE — DONE (write down idle reading)")


def run_active(ser, port: str, seconds: float, text: str) -> dict:
    banner(f"PHASE ACTIVE — read meter NOW ({seconds:.0f}s continuous EVAL)")
    print(f"Sentence: {text!r}", flush=True)
    t0 = time.time()
    n = 0
    lat_ms: list[float] = []
    errors = 0
    while time.time() - t0 < seconds:
        try:
            r = send_eval(ser, text, timeout_s=90.0)
            n += 1
            lat_ms.append(r["latency_us"] / 1000.0)
            if n % 5 == 0 or n <= 2:
                left = seconds - (time.time() - t0)
                print(
                    f"  active n={n} last={lat_ms[-1]:.0f}ms left={left:.0f}s",
                    flush=True,
                )
        except Exception as exc:
            errors += 1
            print(f"  active error: {exc}; reconnecting", flush=True)
            try:
                ser.close()
            except Exception:
                pass
            ser = reconnect_device(port)
    mean = sum(lat_ms) / len(lat_ms) if lat_ms else 0.0
    banner("PHASE ACTIVE — DONE (write down active reading)")
    print(f"Sentences={n} errors={errors} latency_ms_mean={mean:.1f}", flush=True)
    return {
        "n": n,
        "errors": errors,
        "latency_ms_mean": mean,
        "ser": ser,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM32")
    ap.add_argument("--idle-s", type=float, default=30.0)
    ap.add_argument("--active-s", type=float, default=90.0)
    ap.add_argument(
        "--text",
        default="John works in Paris and Mary visited London yesterday",
        help="Fixed sentence for continuous active load",
    )
    args = ap.parse_args()

    banner("CONNECT / READY")
    ser = reconnect_device(args.port)
    hold_idle(ser, args.idle_s)
    result = run_active(ser, args.port, args.active_s, args.text)
    ser = result.pop("ser")
    try:
        ser.write(b"QUIT\n")
        ser.close()
    except Exception:
        pass

    banner("REPORT BACK")
    print(
        "Please reply with:\n"
        "  1) idle power (W / mW / or V+A)\n"
        "  2) active power (same units)\n"
        "  3) meter model / what the reading includes (USB 5V vs 3V3)\n"
        f"Active load: n={result['n']}, mean_lat={result['latency_ms_mean']:.1f} ms",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
