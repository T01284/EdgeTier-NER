from __future__ import annotations

from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.scheme import IOB1, IOB2

_SCHEMES = {"IOB1": IOB1, "IOB2": IOB2}


def compute_ner_metrics(
    y_true: list[list[str]],
    y_pred: list[list[str]],
    scheme: str = "IOB2",
) -> dict[str, float]:
    tag_scheme = _SCHEMES.get(scheme.upper(), IOB2)
    return {
        "precision": float(precision_score(y_true, y_pred, scheme=tag_scheme)),
        "recall": float(recall_score(y_true, y_pred, scheme=tag_scheme)),
        "f1": float(f1_score(y_true, y_pred, scheme=tag_scheme)),
    }


def decode_predictions(id2tag: dict[int, str], pred_ids: list[list[int]]) -> list[list[str]]:
    return [[id2tag[i] for i in seq] for seq in pred_ids]
