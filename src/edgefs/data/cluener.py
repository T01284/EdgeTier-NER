from __future__ import annotations

import json
from pathlib import Path

from edgefs.data.schema import NERSentence


def read_cluener_json(path: str | Path) -> list[NERSentence]:
    """Read CLUENER JSON lines (public zip or legacy GitHub layout)."""
    path = Path(path)
    sentences: list[NERSentence] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" in obj:
                text = obj["text"]
                spans = obj.get("label") or {}
            else:
                text = next(iter(obj.keys()))
                spans = obj[text]
            if not isinstance(spans, dict):
                spans = {}
            tags = ["O"] * len(text)
            for label, entities in spans.items():
                if isinstance(entities, dict):
                    interval_groups = entities.values()
                else:
                    interval_groups = [entities]
                for intervals in interval_groups:
                    for start, end in intervals:
                        tags[start] = f"B-{label}"
                        for i in range(start + 1, end + 1):
                            tags[i] = f"I-{label}"
            tokens = list(text)
            sentences.append(NERSentence(tokens=tokens, tags=tags))
    return sentences
