from __future__ import annotations

import torch
import torch.nn.functional as F


def prototypical_loss(
    token_repr: torch.Tensor,
    tags: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 10.0,
) -> torch.Tensor:
    """Token-level prototypical loss over observed tag ids in batch."""
    valid = mask.bool()
    repr_flat = token_repr[valid]
    tag_flat = tags[valid]
    unique_tags = torch.unique(tag_flat)
    if unique_tags.numel() <= 1:
        return token_repr.sum() * 0.0

    prototypes = []
    for tag_id in unique_tags:
        proto = repr_flat[tag_flat == tag_id].mean(dim=0)
        prototypes.append(proto)
    prototypes = torch.stack(prototypes, dim=0)
    logits = -torch.cdist(repr_flat, prototypes) * temperature
    label_map = {int(t.item()): i for i, t in enumerate(unique_tags)}
    targets = torch.tensor([label_map[int(t.item())] for t in tag_flat], device=tags.device)
    return F.cross_entropy(logits, targets)


def supervised_contrastive_loss(
    token_repr: torch.Tensor,
    tags: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    valid = mask.bool()
    feats = F.normalize(token_repr[valid], dim=-1)
    labels = tags[valid]
    if feats.size(0) < 2:
        return token_repr.sum() * 0.0

    sim = torch.matmul(feats, feats.T) / temperature
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T).float()
    diag = torch.eye(feats.size(0), device=feats.device)
    pos_mask = pos_mask - diag
    exp_sim = torch.exp(sim - sim.max(dim=1, keepdim=True).values) * (1 - diag)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    pos_count = pos_mask.sum(dim=1)
    valid_rows = pos_count > 0
    if not valid_rows.any():
        return token_repr.sum() * 0.0
    loss = -(pos_mask[valid_rows] * log_prob[valid_rows]).sum(dim=1) / pos_count[valid_rows]
    return loss.mean()


def align_emissions_for_distill(
    student_emissions: torch.Tensor,
    teacher_emissions: torch.Tensor,
    student_tags: list[str],
    teacher_tags: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    if student_tags == teacher_tags:
        return student_emissions, teacher_emissions

    student_map = {tag: idx for idx, tag in enumerate(student_tags)}
    aligned_teacher = torch.zeros_like(student_emissions)
    for teacher_idx, tag in enumerate(teacher_tags):
        student_idx = student_map.get(tag)
        if student_idx is None:
            continue
        aligned_teacher[..., student_idx] = teacher_emissions[..., teacher_idx]
    return student_emissions, aligned_teacher


def align_transition_matrix(
    student_trans: torch.Tensor,
    teacher_trans: torch.Tensor,
    student_tags: list[str],
    teacher_tags: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    if student_tags == teacher_tags:
        return student_trans, teacher_trans

    student_map = {tag: idx for idx, tag in enumerate(student_tags)}
    aligned_teacher = torch.zeros_like(student_trans)
    for from_t_idx, from_tag in enumerate(teacher_tags):
        from_s_idx = student_map.get(from_tag)
        if from_s_idx is None:
            continue
        for to_t_idx, to_tag in enumerate(teacher_tags):
            to_s_idx = student_map.get(to_tag)
            if to_s_idx is None:
                continue
            aligned_teacher[to_s_idx, from_s_idx] = teacher_trans[to_t_idx, from_t_idx]
    return student_trans, aligned_teacher


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    if student_logits.shape[-1] != teacher_logits.shape[-1]:
        raise ValueError(
            "student/teacher emission width mismatch: "
            f"{student_logits.shape[-1]} vs {teacher_logits.shape[-1]}"
        )
    valid = mask.unsqueeze(-1).float()
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(s, t, reduction="none").sum(-1, keepdim=True)
    return (kl * valid).sum() / valid.sum().clamp_min(1.0) * (alpha * temperature**2)


def transition_distill_loss(
    student_trans: torch.Tensor,
    teacher_trans: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(student_trans, teacher_trans)
