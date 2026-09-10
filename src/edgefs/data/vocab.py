from __future__ import annotations

from dataclasses import dataclass

from edgefs.data.schema import NERSentence


PAD = "<PAD>"
UNK = "<UNK>"


@dataclass
class CharVocab:
    char2id: dict[str, int]
    pad_id: int
    unk_id: int

    @classmethod
    def build(
        cls,
        sentences: list[NERSentence],
        min_freq: int = 1,
        max_size: int = 5000,
    ) -> CharVocab:
        freq: dict[str, int] = {}
        for sent in sentences:
            for token in sent.tokens:
                for ch in token:
                    freq[ch] = freq.get(ch, 0) + 1
        chars = [PAD, UNK]
        ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        for ch, count in ranked:
            if count < min_freq:
                continue
            if len(chars) >= max_size:
                break
            chars.append(ch)
        char2id = {c: i for i, c in enumerate(chars)}
        return cls(char2id=char2id, pad_id=char2id[PAD], unk_id=char2id[UNK])

    @classmethod
    def build_byte(cls) -> CharVocab:
        """Fixed UTF-8 byte alphabet (|B|=256) plus PAD/UNK."""
        char2id = {PAD: 0, UNK: 1}
        for b in range(256):
            char2id[f"b{b:03d}"] = b + 2
        return cls(char2id=char2id, pad_id=0, unk_id=1)

    @staticmethod
    def token_to_byte_keys(token: str) -> list[str]:
        return [f"b{b:03d}" for b in token.encode("utf-8")]

    @property
    def is_byte(self) -> bool:
        return "b000" in self.char2id and "b255" in self.char2id

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.char2id.get(t, self.unk_id) for t in tokens]

    def __len__(self) -> int:
        return len(self.char2id)

    def save(self, path: str) -> None:
        import json
        from pathlib import Path

        Path(path).write_text(
            json.dumps(
                {
                    "char2id": self.char2id,
                    "pad_id": self.pad_id,
                    "unk_id": self.unk_id,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str) -> CharVocab:
        import json
        from pathlib import Path

        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(char2id=obj["char2id"], pad_id=obj["pad_id"], unk_id=obj["unk_id"])
