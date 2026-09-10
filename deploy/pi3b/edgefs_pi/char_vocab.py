"""Character vocabulary loader for edge inference."""

from __future__ import annotations

import json
from pathlib import Path


class CharVocab:
    def __init__(self, char2id: dict[str, int], pad_id: int, unk_id: int) -> None:
        self.char2id = char2id
        self.pad_id = pad_id
        self.unk_id = unk_id

    @classmethod
    def load(cls, path: str | Path) -> CharVocab:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(char2id=obj["char2id"], pad_id=obj["pad_id"], unk_id=obj["unk_id"])

    def encode(self, text: str, max_len: int) -> tuple[list[int], list[bool]]:
        chars = list(text[:max_len])
        ids = [self.char2id.get(c, self.unk_id) for c in chars]
        mask = [True] * len(ids)
        pad = max_len - len(ids)
        if pad > 0:
            ids.extend([self.pad_id] * pad)
            mask.extend([False] * pad)
        return ids, mask
