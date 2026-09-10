"""ONNX Runtime inference wrapper for Pi 3B."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edgefs_pi.char_vocab import CharVocab
from edgefs_pi.crf_decode import viterbi_decode


@dataclass
class DeployAssets:
    onnx_path: Path
    char_vocab: CharVocab
    tags: list[str]
    transitions: np.ndarray
    max_len: int
    max_word_len: int

    @classmethod
    def load(cls, deploy_dir: str | Path) -> DeployAssets:
        deploy_dir = Path(deploy_dir)
        tags = json.loads((deploy_dir / "tags.json").read_text(encoding="utf-8"))["tags"]
        transitions = np.load(deploy_dir / "crf_transitions.npy")
        cfg = json.loads((deploy_dir / "model_config.json").read_text(encoding="utf-8"))
        max_len = int(cfg.get("max_len", 128))
        max_word_len = int(cfg.get("max_word_len", 20))
        return cls(
            onnx_path=deploy_dir / "model_emissions.onnx",
            char_vocab=CharVocab.load(deploy_dir / "char_vocab.json"),
            tags=tags,
            transitions=transitions,
            max_len=max_len,
            max_word_len=max_word_len,
        )


class OnnxNERRunner:
    def __init__(self, assets: DeployAssets) -> None:
        import onnxruntime as ort

        self.assets = assets
        self.session = ort.InferenceSession(
            str(assets.onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [i.name for i in self.session.get_inputs()]

    def _session_inputs(self, input_ids: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
        feeds: dict[str, np.ndarray] = {"input_ids": input_ids}
        if "mask" in self.input_names:
            feeds["mask"] = mask
        return feeds

    def predict(self, text: str) -> list[tuple[str, str]]:
        input_ids, mask = self._encode_text(text)
        seq_len = int(mask.sum())
        outputs = self.session.run(
            None,
            self._session_inputs(input_ids, mask),
        )
        emissions = outputs[0][0][:seq_len]
        tag_ids = viterbi_decode(emissions, self.assets.transitions)
        tokens = text.split()[:seq_len]
        return list(zip(tokens, [self.assets.tags[i] for i in tag_ids]))

    def _encode_text(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        words = text.split()[: self.assets.max_len]
        input_ids = np.zeros(
            (1, self.assets.max_len, self.assets.max_word_len),
            dtype=np.int64,
        )
        mask = np.zeros((1, self.assets.max_len), dtype=np.bool_)
        cv = self.assets.char_vocab
        for i, word in enumerate(words):
            mask[0, i] = True
            for j, ch in enumerate(word[: self.assets.max_word_len]):
                input_ids[0, i, j] = cv.char2id.get(ch.lower(), cv.unk_id)
        return input_ids, mask
