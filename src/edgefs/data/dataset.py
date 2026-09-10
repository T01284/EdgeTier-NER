from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from edgefs.data.prepare import NERCorpus
from edgefs.data.schema import NERSentence


@dataclass
class Batch:
    input_ids: torch.Tensor
    word_ids: torch.Tensor
    tags: torch.Tensor
    mask: torch.Tensor


class NERDataset(Dataset):
    def __init__(
        self,
        sentences: list[NERSentence],
        char2id: dict[str, int],
        tag2id: dict[str, int],
        pad_id: int,
        unk_id: int,
        max_len: int = 128,
        max_word_len: int = 20,
        word2id: dict[str, int] | None = None,
        word_pad_id: int = 0,
        word_unk_id: int = 1,
        encoding: str = "char",
    ) -> None:
        self.sentences = sentences
        self.char2id = char2id
        self.tag2id = tag2id
        self.pad_id = pad_id
        self.unk_id = unk_id
        self.max_len = max_len
        self.max_word_len = max_word_len
        self.word2id = word2id or {}
        self.word_pad_id = word_pad_id
        self.word_unk_id = word_unk_id
        self.encoding = encoding

    def __len__(self) -> int:
        return len(self.sentences)

    def _encode_word(self, token: str) -> list[int]:
        if self.encoding == "byte":
            from edgefs.data.vocab import CharVocab

            keys = CharVocab.token_to_byte_keys(token)[: self.max_word_len]
            ids = [self.char2id.get(k, self.unk_id) for k in keys]
        else:
            chars = list(token[: self.max_word_len])
            ids = [self.char2id.get(ch, self.unk_id) for ch in chars]
        pad_len = self.max_word_len - len(ids)
        if pad_len > 0:
            ids.extend([self.pad_id] * pad_len)
        return ids

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sent = self.sentences[idx]
        tokens = sent.tokens[: self.max_len]
        tags = sent.tags[: self.max_len]
        input_ids = [self._encode_word(token) for token in tokens]
        word_ids = [
            self.word2id.get(token, self.word_unk_id) if self.word2id else self.word_unk_id
            for token in tokens
        ]
        tag_ids = [self.tag2id[t] for t in tags]
        length = len(input_ids)
        pad_len = self.max_len - length
        if pad_len > 0:
            empty_word = [self.pad_id] * self.max_word_len
            input_ids.extend([empty_word] * pad_len)
            word_ids.extend([self.word_pad_id] * pad_len)
            tag_ids += [self.tag2id["O"]] * pad_len
        mask = [1] * length + [0] * pad_len
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "word_ids": torch.tensor(word_ids, dtype=torch.long),
            "tags": torch.tensor(tag_ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.bool),
        }


def collate_batch(items: list[dict[str, torch.Tensor]]) -> Batch:
    return Batch(
        input_ids=torch.stack([x["input_ids"] for x in items]),
        word_ids=torch.stack([x["word_ids"] for x in items]),
        tags=torch.stack([x["tags"] for x in items]),
        mask=torch.stack([x["mask"] for x in items]),
    )


def build_datasets(
    corpus: NERCorpus,
    max_len: int = 128,
    max_word_len: int = 20,
    encoding: str | None = None,
) -> tuple[NERDataset, NERDataset, NERDataset]:
    tag2id = corpus.tagset.tag2id
    cv = corpus.char_vocab
    wv = corpus.word_vocab
    if encoding is None:
        encoding = "byte" if cv.is_byte else "char"
    common = {
        "char2id": cv.char2id,
        "tag2id": tag2id,
        "pad_id": cv.pad_id,
        "unk_id": cv.unk_id,
        "max_len": max_len,
        "max_word_len": max_word_len,
        "encoding": encoding,
    }
    if wv is not None:
        common.update(
            {
                "word2id": wv.word2id,
                "word_pad_id": wv.pad_id,
                "word_unk_id": wv.unk_id,
            }
        )
    train = NERDataset(corpus.train, **common)
    dev = NERDataset(corpus.dev, **common)
    test = NERDataset(corpus.test, **common)
    return train, dev, test
