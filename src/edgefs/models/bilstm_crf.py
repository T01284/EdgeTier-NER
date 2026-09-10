from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchcrf import CRF

from edgefs.models.char_cnn_crf import CharWordEncoder


@dataclass
class BiLSTMCRFConfig:
    vocab_size: int
    num_tags: int
    embed_dim: int = 100
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.5
    freeze_embeddings: bool = False


@dataclass
class BiLSTMCharCRFConfig:
    word_vocab_size: int
    char_vocab_size: int
    num_tags: int
    embed_dim: int = 100
    char_embed_dim: int = 64
    char_num_filters: int = 128
    char_kernel_size: int = 3
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.5
    freeze_embeddings: bool = False


class _BiLSTMEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, seq_len: int) -> torch.Tensor:
        lengths = mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.clamp_min(1),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=seq_len,
        )
        return self.dropout(x)


class BiLSTMCRF(nn.Module):
    """GloVe + BiLSTM + linear-chain CRF (standard CoNLL-2003 strong baseline)."""

    def __init__(
        self,
        config: BiLSTMCRFConfig,
        pretrained_embeddings: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embed_dim,
            padding_idx=0,
        )
        if pretrained_embeddings is not None:
            if pretrained_embeddings.shape != self.embedding.weight.shape:
                raise ValueError("Pretrained embedding shape mismatch")
            self.embedding.weight.data.copy_(pretrained_embeddings)
        self.embedding.weight.requires_grad = not config.freeze_embeddings

        self.encoder = _BiLSTMEncoder(
            input_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        self.classifier = nn.Linear(config.hidden_dim, config.num_tags)
        self.crf = CRF(config.num_tags, batch_first=True)

    def encode(self, word_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embedding(word_ids)
        return self.encoder(x, mask, word_ids.size(1))

    def forward(
        self,
        word_ids: torch.Tensor,
        tags: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        char_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if mask is None:
            mask = word_ids.ne(0)
        emissions = self.classifier(self.encode(word_ids, mask))
        out: dict[str, torch.Tensor] = {"emissions": emissions}
        if tags is not None:
            out["loss"] = -self.crf(emissions, tags, mask=mask, reduction="mean")
        return out

    def decode(self, word_ids: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        emissions = self.classifier(self.encode(word_ids, mask))
        return self.crf.decode(emissions, mask=mask)

    def transition_matrix(self) -> torch.Tensor:
        return self.crf.transitions.detach().clone()


class BiLSTMCharCRF(nn.Module):
    """GloVe + char-CNN + BiLSTM + CRF (strong CoNLL baseline with OOV robustness)."""

    def __init__(
        self,
        config: BiLSTMCharCRFConfig,
        pretrained_embeddings: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            config.word_vocab_size,
            config.embed_dim,
            padding_idx=0,
        )
        if pretrained_embeddings is not None:
            if pretrained_embeddings.shape != self.embedding.weight.shape:
                raise ValueError("Pretrained embedding shape mismatch")
            self.embedding.weight.data.copy_(pretrained_embeddings)
        self.embedding.weight.requires_grad = not config.freeze_embeddings

        self.char_encoder = CharWordEncoder(
            vocab_size=config.char_vocab_size,
            embed_dim=config.char_embed_dim,
            num_filters=config.char_num_filters,
            kernel_size=config.char_kernel_size,
            dropout=config.dropout,
        )
        lstm_input_dim = config.embed_dim + config.char_num_filters
        self.encoder = _BiLSTMEncoder(
            input_dim=lstm_input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        self.classifier = nn.Linear(config.hidden_dim, config.num_tags)
        self.crf = CRF(config.num_tags, batch_first=True)

    def encode(
        self,
        word_ids: torch.Tensor,
        char_ids: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        word_repr = self.embedding(word_ids)
        char_repr = self.char_encoder(char_ids)
        x = torch.cat([word_repr, char_repr], dim=-1)
        return self.encoder(x, mask, word_ids.size(1))

    def forward(
        self,
        word_ids: torch.Tensor,
        tags: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        char_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if char_ids is None:
            raise ValueError("BiLSTMCharCRF requires char_ids")
        if mask is None:
            mask = word_ids.ne(0)
        emissions = self.classifier(self.encode(word_ids, char_ids, mask))
        out: dict[str, torch.Tensor] = {"emissions": emissions}
        if tags is not None:
            out["loss"] = -self.crf(emissions, tags, mask=mask, reduction="mean")
        return out

    def decode(
        self,
        word_ids: torch.Tensor,
        mask: torch.Tensor,
        char_ids: torch.Tensor,
    ) -> list[list[int]]:
        emissions = self.classifier(self.encode(word_ids, char_ids, mask))
        return self.crf.decode(emissions, mask=mask)

    def transition_matrix(self) -> torch.Tensor:
        return self.crf.transitions.detach().clone()
