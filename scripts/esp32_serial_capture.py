#!/usr/bin/env python3
"""Capture ESP32-S3 benchmark lines from serial port."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(
    r"sample=(?P<id>\S+) len=(?P<len>\d+) latency_us=(?P<lat>\d+) "
    r"tags(?:\[0\])?=(?P<tag>-?\d+(?:,-?\d+)*) heap=(?P<heap>\d+) psram=(?P<psram>\d+)"
)


def capture(port: str, seconds: float = 15.0, reset: bool = True) -> dict:
    import serial

    lines: list[str] = []
    ser = serial.Serial(port, 115200, timeout=0.5)
    if reset:
        ser.setDTR(False)
        ser.setRTS(True)
        time.sleep(0.1)
        ser.setRTS(False)
        time.sleep(0.5)
    deadline = time.time() + seconds
    while time.time() < deadline:
        raw = ser.readline()
        if raw:
            lines.append(raw.decode("utf-8", errors="replace").rstrip())
    ser.close()

    samples = []
    for line in lines:
        m = PATTERN.search(line)
        if m:
            samples.append(
                {
                    "id": m.group("id"),
                    "len": int(m.group("len")),
                    "latency_us": int(m.group("lat")),
                    "tag0": int(m.group("tag").split(",")[0]),
                    "free_heap_bytes": int(m.group("heap")),
                    "free_psram_bytes": int(m.group("psram")),
                }
            )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "port": port,
        "ok": len(samples) >= 1,
        "samples": samples,
        "serial_lines": lines[-50:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM23")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--output", default=str(ROOT / "deploy" / "results" / "esp32_serial_capture.json"))
    args = parser.parse_args()
    report = capture(args.port, args.seconds)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
