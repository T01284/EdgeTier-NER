from __future__ import annotations

from dataclasses import dataclass

from edgefs.data.schema import NERSentence

PAD = "<PAD>"
UNK = "<UNK>"


@dataclass
class WordVocab:
    word2id: dict[str, int]
    pad_id: int
    unk_id: int

    @classmethod
    def build(
        cls,
        sentences: list[NERSentence],
        min_freq: int = 1,
        max_size: int = 50000,
    ) -> WordVocab:
        freq: dict[str, int] = {}
        for sent in sentences:
            for token in sent.tokens:
                freq[token] = freq.get(token, 0) + 1
        words = [PAD, UNK]
        ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        for word, count in ranked:
            if count < min_freq:
                continue
            if len(words) >= max_size:
                break
            words.append(word)
        word2id = {w: i for i, w in enumerate(words)}
        return cls(word2id=word2id, pad_id=word2id[PAD], unk_id=word2id[UNK])

    def __len__(self) -> int:
        return len(self.word2id)

    def save(self, path: str) -> None:
        import json
        from pathlib import Path

        Path(path).write_text(
            json.dumps(
                {
                    "word2id": self.word2id,
                    "pad_id": self.pad_id,
                    "unk_id": self.unk_id,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str) -> WordVocab:
        import json
        from pathlib import Path

        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(word2id=obj["word2id"], pad_id=obj["pad_id"], unk_id=obj["unk_id"])
