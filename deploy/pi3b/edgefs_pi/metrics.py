"""Minimal IOB2 entity-level metrics without seqeval dependency."""

from __future__ import annotations


def _to_entities(tags: list[str]) -> set[tuple[str, int, int]]:
    entities: set[tuple[str, int, int]] = set()
    start = None
    ent_type = None
    for i, tag in enumerate(tags):
        if tag == "O" or tag == "":
            if start is not None:
                entities.add((ent_type or "MISC", start, i))
                start = None
                ent_type = None
            continue
        if "-" not in tag:
            continue
        prefix, typ = tag.split("-", 1)
        if prefix == "B":
            if start is not None:
                entities.add((ent_type or "MISC", start, i))
            start = i
            ent_type = typ
        elif prefix == "I":
            if start is None:
                start = i
                ent_type = typ
            elif ent_type != typ:
                entities.add((ent_type or "MISC", start, i))
                start = i
                ent_type = typ
        else:
            if start is not None:
                entities.add((ent_type or "MISC", start, i))
            start = None
            ent_type = None
    if start is not None:
        entities.add((ent_type or "MISC", start, len(tags)))
    return entities


def compute_ner_metrics(
    y_true: list[list[str]],
    y_pred: list[list[str]],
) -> dict[str, float]:
    tp = fp = fn = 0
    for gold_tags, pred_tags in zip(y_true, y_pred):
        n = min(len(gold_tags), len(pred_tags))
        gold = _to_entities(gold_tags[:n])
        pred = _to_entities(pred_tags[:n])
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
