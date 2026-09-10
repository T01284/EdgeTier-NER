from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchcrf import CRF


@dataclass
class CharCNNCRFConfig:
    vocab_size: int
    num_tags: int
    embed_dim: int = 64
    num_filters: int = 128
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4)
    dropout: float = 0.1
    use_residual: bool = True
    max_word_len: int = 20


class CharWordEncoder(nn.Module):
    """Encode each token from its character sequence via 1-D convolution."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_filters: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            embed_dim,
            num_filters,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        # char_ids: [B, T, W]
        batch_size, seq_len, _word_len = char_ids.shape
        flat = char_ids.view(batch_size * seq_len, -1)
        x = self.embedding(flat).transpose(1, 2)
        x = torch.relu(self.conv(x))
        x = x.max(dim=2).values
        x = self.dropout(x)
        return x.view(batch_size, seq_len, -1)


class ConvBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        use_residual: bool,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.use_residual = use_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn(self.conv(x)))
        if self.use_residual:
            return out + x
        return out


class CharCNNCRF(nn.Module):
    """MCU-friendly char-level CNN + linear-chain CRF."""

    def __init__(self, config: CharCNNCRFConfig) -> None:
        super().__init__()
        self.config = config
        self.word_encoder = CharWordEncoder(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            num_filters=config.num_filters,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
        )
        self.blocks = nn.ModuleList(
            [
                ConvBlock(
                    channels=config.num_filters,
                    kernel_size=config.kernel_size,
                    dilation=d,
                    use_residual=config.use_residual,
                )
                for d in config.dilations
            ]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.num_filters, config.num_tags)
        self.crf = CRF(config.num_tags, batch_first=True)

    def encode(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # input_ids: [B, T, W]
        x = self.word_encoder(input_ids).transpose(1, 2)  # [B, C, T]
        for block in self.blocks:
            x = block(x)
        x = self.dropout(x.transpose(1, 2))  # [B, T, C]
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        tags: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if mask is None:
            mask = input_ids[:, :, 0].ne(0)
        emissions = self.classifier(self.encode(input_ids, mask))
        out: dict[str, torch.Tensor] = {"emissions": emissions}
        if tags is not None and mask is not None:
            out["loss"] = -self.crf(emissions, tags, mask=mask, reduction="mean")
        return out

    def decode(self, input_ids: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        emissions = self.classifier(self.encode(input_ids, mask))
        return self.crf.decode(emissions, mask=mask)

    def transition_matrix(self) -> torch.Tensor:
        return self.crf.transitions.detach().clone()
