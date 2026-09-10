from __future__ import annotations

from dataclasses import dataclass

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


_CO_NLL_ENTITY_TYPES = frozenset({"LOC", "MISC", "ORG", "PER"})


def normalize_ner_tag(tag: str) -> str:
    """Map leaked chunk tags (e.g. I-NP) to O; keep dataset-specific B-/I- entity tags."""
    if tag in STANDARD_NER_TAGS or tag == "O":
        return tag
    if len(tag) > 2 and tag[0] in "BI" and tag[1] == "-":
        etype = tag[2:]
        if etype in _CO_NLL_ENTITY_TYPES:
            return tag
        # CLUENER etc.: entity types like company, name (lowercase)
        if etype and etype[0].islower():
            return tag
        return "O"
    return "O"


@dataclass(frozen=True)
class NERSentence:
    tokens: list[str]
    tags: list[str]

    def __post_init__(self) -> None:
        if len(self.tokens) != len(self.tags):
            raise ValueError("tokens and tags length mismatch")


@dataclass
class TagSet:
    tags: list[str]

    @property
    def tag2id(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.tags)}

    @property
    def id2tag(self) -> dict[int, str]:
        return {i: t for i, t in enumerate(self.tags)}

    @classmethod
    def from_sentences(cls, sentences: list[NERSentence], include_o: bool = True) -> TagSet:
        tags = set()
        for sent in sentences:
            tags.update(sent.tags)
        ordered = sorted(tags)
        if include_o and "O" in ordered:
            ordered.remove("O")
            ordered = ["O"] + ordered
        return cls(tags=ordered)
