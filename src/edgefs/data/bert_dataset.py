from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from edgefs.data.prepare import NERCorpus
from edgefs.data.schema import NERSentence
from edgefs.utils.hf_mirror import use_hf_mirror


@dataclass
class BertBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    tags: torch.Tensor
    mask: torch.Tensor
    word_mask: torch.Tensor


class BertWordDataset(Dataset):
    """Word-level NER with HuggingFace tokenizer (first subword per word)."""

    def __init__(
        self,
        sentences: list[NERSentence],
        tokenizer_name: str,
        tag2id: dict[str, int],
        max_len: int = 128,
    ) -> None:
        self.sentences = sentences
        use_hf_mirror()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tag2id = tag2id
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sent = self.sentences[idx]
        tokens = sent.tokens[: self.max_len]
        tags = sent.tags[: self.max_len]
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_len + 2,
            return_attention_mask=True,
        )
        word_ids = encoding.word_ids(batch_index=0)
        tag_ids = []
        prev_word = None
        for word_id in word_ids:
            if word_id is None:
                tag_ids.append(self.tag2id["O"])
            elif word_id != prev_word:
                tag_ids.append(self.tag2id[tags[word_id]])
            else:
                tag_ids.append(self.tag2id[tags[word_id]])
            prev_word = word_id

        length = len(encoding["input_ids"])
        pad_len = self.max_len + 2 - length
        input_ids = encoding["input_ids"] + [self.tokenizer.pad_token_id] * pad_len
        attention_mask = encoding["attention_mask"] + [0] * pad_len
        tag_ids = tag_ids + [self.tag2id["O"]] * pad_len

        # CRF runs over all non-pad tokens (incl. [CLS]/[SEP]); eval uses first subword only.
        crf_mask = attention_mask
        word_mask: list[int] = []
        prev_word = None
        for word_id in word_ids:
            if word_id is None:
                word_mask.append(0)
            elif word_id != prev_word:
                word_mask.append(1)
            else:
                word_mask.append(0)
            if word_id is not None:
                prev_word = word_id
        word_mask = word_mask + [0] * pad_len

        seq_len = self.max_len + 2
        return {
            "input_ids": torch.tensor(input_ids[:seq_len], dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask[:seq_len], dtype=torch.long),
            "tags": torch.tensor(tag_ids[:seq_len], dtype=torch.long),
            "mask": torch.tensor(crf_mask[:seq_len], dtype=torch.bool),
            "word_mask": torch.tensor(word_mask[:seq_len], dtype=torch.bool),
        }


def collate_bert_batch(items: list[dict[str, torch.Tensor]]) -> BertBatch:
    return BertBatch(
        input_ids=torch.stack([x["input_ids"] for x in items]),
        attention_mask=torch.stack([x["attention_mask"] for x in items]),
        tags=torch.stack([x["tags"] for x in items]),
        mask=torch.stack([x["mask"] for x in items]),
        word_mask=torch.stack([x["word_mask"] for x in items]),
    )


def build_bert_datasets(
    corpus: NERCorpus,
    tokenizer_name: str,
    max_len: int = 128,
) -> tuple[BertWordDataset, BertWordDataset, BertWordDataset]:
    common = {
        "tokenizer_name": tokenizer_name,
        "tag2id": corpus.tagset.tag2id,
        "max_len": max_len,
    }
    return (
        BertWordDataset(corpus.train, **common),
        BertWordDataset(corpus.dev, **common),
        BertWordDataset(corpus.test, **common),
    )
