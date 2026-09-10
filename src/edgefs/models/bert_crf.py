from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel

from edgefs.utils.hf_mirror import use_hf_mirror


@dataclass
class BertCRFConfig:
    pretrained_name: str
    num_tags: int
    dropout: float = 0.1
    freeze_backbone: bool = False


class BertCRF(nn.Module):
    """Pretrained transformer + linear emissions + linear-chain CRF."""

    def __init__(self, config: BertCRFConfig) -> None:
        super().__init__()
        use_hf_mirror()
        self.config = config
        self.bert = AutoModel.from_pretrained(config.pretrained_name)
        if config.freeze_backbone:
            for p in self.bert.parameters():
                p.requires_grad = False
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(hidden, config.num_tags)
        self.crf = CRF(config.num_tags, batch_first=True)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.dropout(out.last_hidden_state)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tags: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if mask is None:
            mask = attention_mask.bool()
        emissions = self.classifier(self.encode(input_ids, attention_mask))
        result: dict[str, torch.Tensor] = {"emissions": emissions}
        if tags is not None:
            result["loss"] = -self.crf(emissions, tags, mask=mask, reduction="mean")
        return result

    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask: torch.Tensor,
    ) -> list[list[int]]:
        emissions = self.classifier(self.encode(input_ids, attention_mask))
        return self.crf.decode(emissions, mask=mask)

    def transition_matrix(self) -> torch.Tensor:
        return self.crf.transitions.detach().clone()
