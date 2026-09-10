from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from edgefs.data.bert_dataset import BertWordDataset, collate_bert_batch
from edgefs.data.schema import NERSentence


@torch.no_grad()
def collect_word_token_reprs(
    model: torch.nn.Module,
    sentences: list[NERSentence],
    tag2id: dict[str, int],
    tokenizer_name: str,
    device: torch.device,
    max_len: int = 128,
    batch_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten word-level BERT hidden states and gold tag ids from support sentences."""
    if not sentences:
        empty = torch.empty(0, device=device)
        return empty, empty

    ds = BertWordDataset(sentences, tokenizer_name, tag2id, max_len=max_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_bert_batch)
    model.eval()
    reprs: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch in loader:
        batch = type(batch)(
            input_ids=batch.input_ids.to(device),
            attention_mask=batch.attention_mask.to(device),
            tags=batch.tags.to(device),
            mask=batch.mask.to(device),
            word_mask=batch.word_mask.to(device),
        )
        hidden = model.encode(batch.input_ids, batch.attention_mask)
        valid = batch.word_mask.bool()
        reprs.append(hidden[valid])
        labels.append(batch.tags[valid])
    return torch.cat(reprs, dim=0), torch.cat(labels, dim=0)


def nearest_neighbor_predict(
    query_repr: torch.Tensor,
    support_repr: torch.Tensor,
    support_labels: torch.Tensor,
    k: int = 1,
) -> torch.Tensor:
    """Assign each query token the majority label among k nearest support tokens."""
    if support_repr.numel() == 0 or query_repr.numel() == 0:
        return torch.zeros(query_repr.size(0), dtype=torch.long, device=query_repr.device)

    q = F.normalize(query_repr, dim=-1)
    s = F.normalize(support_repr, dim=-1)
    sim = torch.matmul(q, s.T)
    k = min(k, s.size(0))
    _, nn_idx = sim.topk(k, dim=1)
    nn_labels = support_labels[nn_idx]
    preds = []
    for row in nn_labels:
        vals, counts = torch.unique(row, return_counts=True)
        preds.append(vals[counts.argmax()])
    return torch.stack(preds)


def estimate_support_transitions(
    sentences: list[NERSentence],
    tag2id: dict[str, int],
    num_tags: int,
    device: torch.device,
) -> torch.Tensor:
    """Empirical bigram transition counts aligned with torchcrf layout [to, from]."""
    trans = torch.zeros(num_tags, num_tags, device=device)
    for sent in sentences:
        ids = [tag2id[t] for t in sent.tags]
        for prev_id, next_id in zip(ids, ids[1:]):
            trans[next_id, prev_id] += 1.0
    col_sum = trans.sum(dim=0, keepdim=True).clamp_min(1.0)
    return trans / col_sum


def structural_transition_loss(
    model_trans: torch.Tensor,
    support_trans: torch.Tensor,
) -> torch.Tensor:
    """Align learned CRF transitions with support-set structure (StructShot)."""
    return F.mse_loss(model_trans, support_trans)
