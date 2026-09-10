"""Minimal CoNLL reader for Pi deploy eval (no full edgefs package required)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

STANDARD_NER_TAGS = frozenset(
    {
        "O",
        "B-LOC",
        "B-MISC",
        "B-ORG",
        "B-PER",
        "I-LOC",
        "I-MISC",
        "I-ORG",
        "I-PER",
    }
)


def normalize_ner_tag(tag: str) -> str:
    return tag if tag in STANDARD_NER_TAGS else "O"


@dataclass(frozen=True)
class NERSentence:
    tokens: list[str]
    tags: list[str]


def read_conll(path: str | Path, lowercase: bool = False) -> list[NERSentence]:
    path = Path(path)
    sentences: list[NERSentence] = []
    tokens: list[str] = []
    tags: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                if tokens:
                    sentences.append(NERSentence(tokens=tokens, tags=tags))
                    tokens, tags = [], []
                continue
            if line.startswith("-DOCSTART-"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                token, tag = parts[0], parts[-1]
            elif len(parts) >= 2:
                token, tag = parts[0], parts[-1]
            else:
                continue
            if lowercase:
                token = token.lower()
            tokens.append(token)
            tags.append(normalize_ner_tag(tag))

    if tokens:
        sentences.append(NERSentence(tokens=tokens, tags=tags))
    return sentences
