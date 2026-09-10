#!/usr/bin/env python3
"""Compare ESP32 UART word_features int8 vs host firmware mirror."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_word_features import firmware_mirror_int8
from esp32_embed import embed_exponent
from esp32_int8_parity import encode_sentence, load_model

FEAT_RESPONSE = re.compile(r"OKFEAT t0c=(?P<t0c>[-\d,]+)")


def parse_int_list(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x]


def uart_wait_ready(ser, timeout_s: float = 90.0) -> str:
    """Wait for device READY; send PING to elicit READY if host connected late."""
    deadline = time.time() + timeout_s
    ready_line = ""
    ping_sent = False
    while time.time() < deadline:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if line and "READY board=" in line:
            return line
        if not line and not ping_sent and (deadline - time.time()) < (timeout_s - 2.0):
            ser.write(b"PING\r\n")
            ser.flush()
            ping_sent = True
            continue
        if line == "PONG" and not ready_line:
            ser.write(b"PING\r\n")
            ser.flush()
    raise TimeoutError("device not READY")


def uart_wordfeat(port: str, text: str, baud: int = 115200) -> dict:
    import serial

    with serial.Serial(port, baudrate=baud, timeout=2.0) as ser:
        ser.setDTR(False)
        ser.setRTS(True)
        time.sleep(0.1)
        ser.setRTS(False)
        time.sleep(6.0)
        ser.reset_input_buffer()
        ser.write(b"PING\r\n")
        ser.flush()

        ready_line = uart_wait_ready(ser, timeout_s=90.0)

        ser.write(f"WORDFEAT\t{text}\r\n".encode("utf-8"))
        ser.flush()
        deadline = time.time() + 120.0
        while time.time() < deadline:
            line = ser.readline().decode("utf-8", errors="replace").strip()
            if not line:
                continue
            if line.startswith("ERR"):
                raise RuntimeError(line)
            m = FEAT_RESPONSE.search(line)
            if m:
                return {
                    "ready_line": ready_line,
                    "t0c": parse_int_list(m.group("t0c")),
                }
        raise TimeoutError("no OKFEAT response")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM32")
    parser.add_argument("--text", default="John works in Paris")
    parser.add_argument("--output", default="deploy/results/esp32_wordfeat_parity.json")
    parser.add_argument("--skip-uart", action="store_true")
    args = parser.parse_args()

    deploy = ROOT / "deploy" / "esp32-s3" / "models"
    vocab = json.loads((deploy / "char_vocab.json").read_text(encoding="utf-8"))
    model = load_model(ROOT / "outputs/runs/conll2003_full/best.pt")
    embed_exp = embed_exponent(deploy)
    feature_exp = -5
    for line in (deploy / "edgefs_model.info").read_text(encoding="utf-8").splitlines():
        if "word_features[INT8" in line and "exponents:" in line:
            part = line.split("exponents:")[1].strip()
            feature_exp = int(part[part.index("[") + 1 : part.index("]")].split(",")[0])

    char_ids = encode_sentence(args.text, vocab)
    mirror = firmware_mirror_int8(model, char_ids, feature_exp, embed_exp)
    host_t0c = mirror[:, 0].tolist()[:16]
    host_ct0 = mirror.reshape(-1)[:16].tolist()

    result = {
        "text": args.text,
        "feature_exp": feature_exp,
        "embed_exp": embed_exp,
        "host_t0c": host_t0c,
        "host_ct0": host_ct0,
    }

    if not args.skip_uart:
        uart = uart_wordfeat(args.port, args.text)
        result["uart"] = uart
        dev_t0c = uart["t0c"]
        t0c_diff = np.abs(np.array(host_t0c[: len(dev_t0c)], dtype=np.int16) - np.array(dev_t0c, dtype=np.int16))
        result["parity"] = {
            "t0c_max_diff": int(t0c_diff.max()) if len(t0c_diff) else None,
            "t0c_match": bool(t0c_diff.max() == 0) if len(t0c_diff) else False,
        }

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
