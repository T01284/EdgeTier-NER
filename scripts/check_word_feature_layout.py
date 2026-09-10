#!/usr/bin/env python3
"""Check espdl golden word_features layout vs firmware mirror."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_word_features import firmware_mirror_int8
from esp32_embed import embed_exponent
from esp32_int8_parity import encode_sentence, load_model


def parse_golden_word_features(info_path: Path) -> np.ndarray:
    text = info_path.read_text(encoding="utf-8")
    block = text.split("test inputs value:\n", 1)[1].split("test outputs value:", 1)[0]
    nums = [int(x) for x in re.findall(r"-?\d+", block.split("value:", 1)[1])]
    return np.array(nums[: 128 * 128], dtype=np.int8)


def main() -> None:
    deploy = ROOT / "deploy" / "esp32-s3" / "models"
    vocab = json.loads((deploy / "char_vocab.json").read_text(encoding="utf-8"))
    model = load_model(ROOT / "outputs/runs/conll2003_full/best.pt")
    text = "John works in Paris"
    char_ids = encode_sentence(text, vocab)
    feature_exp = -5
    embed_exp = embed_exponent(deploy)
    mirror = firmware_mirror_int8(model, char_ids, feature_exp, embed_exp)  # [C, T]
    golden = parse_golden_word_features(deploy / "edgefs_model.info")

    layouts = {
        "CT_c128+t": mirror.reshape(-1),
        "TC_t128+c": mirror.T.reshape(-1),
    }
    report = {"text": text, "layouts": {}}
    for name, arr in layouts.items():
        diff = np.abs(golden.astype(np.int16) - arr.astype(np.int16))
        report["layouts"][name] = {
            "max_diff": int(diff.max()),
            "mean_diff": float(diff.mean()),
            "nonzero": int(np.count_nonzero(diff)),
        }

    report["golden_head"] = golden[:16].tolist()
    report["mirror_c0_t0_7"] = mirror[0, :8].tolist()
    report["mirror_c0_7_t0"] = mirror[:8, 0].tolist()
    report["golden_idx128_131"] = golden[128:132].tolist()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
