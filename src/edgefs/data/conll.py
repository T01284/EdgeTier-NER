from __future__ import annotations

from pathlib import Path

from edgefs.data.schema import NERSentence, normalize_ner_tag


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
                # CoNLL-2003: WORD POS CHUNK NER
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


def write_conll(path: str | Path, sentences: list[NERSentence]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sent in sentences:
            for token, tag in zip(sent.tokens, sent.tags):
                f.write(f"{token} {tag}\n")
            f.write("\n")
