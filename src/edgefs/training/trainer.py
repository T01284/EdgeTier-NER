from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from edgefs.data.dataset import Batch, NERDataset, build_datasets, collate_batch
from edgefs.data.bert_dataset import BertBatch, build_bert_datasets, collate_bert_batch
from edgefs.data.prepare import NERCorpus
from edgefs.evaluation.metrics import compute_ner_metrics
from edgefs.models.char_cnn_crf import CharCNNCRF, CharCNNCRFConfig
from edgefs.training.losses import (
    align_emissions_for_distill,
    align_transition_matrix,
    distillation_loss,
    prototypical_loss,
    supervised_contrastive_loss,
    transition_distill_loss,
)
from edgefs.utils.seed import set_seed


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_len: int = 128
    max_word_len: int = 20
    device: str = "cpu"
    output_dir: str = "outputs/runs/default"
    seed: int = 42
    lambda_proto: float = 0.5
    lambda_contrastive: float = 0.1
    lambda_struct: float = 0.0
    proto_temperature: float = 10.0
    nn_k: int = 1
    support_only: bool = False
    distill_alpha: float = 0.5
    distill_temperature: float = 2.0
    lambda_transition: float = 0.1
    lr_scheduler: str | None = None
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    grad_clip_norm: float | None = None
    early_stop_patience: int | None = None
    min_epochs: int = 1
    eval_scheme: str = "IOB2"
    log_test_each_epoch: bool = False


class FullTrainer:
    def __init__(
        self,
        corpus: NERCorpus,
        train_cfg: TrainConfig,
        model: nn.Module | None = None,
        model_cfg: CharCNNCRFConfig | None = None,
        model_type: str = "char_cnn_crf",
        teacher: nn.Module | None = None,
        teacher_model_type: str | None = None,
        teacher_tagset: list[str] | None = None,
        bert_tokenizer_name: str | None = None,
    ) -> None:
        set_seed(train_cfg.seed)
        self.corpus = corpus
        self.train_cfg = train_cfg
        self.model_type = model_type
        self.is_bert = model_type == "bert_crf"
        self.device = torch.device(train_cfg.device)
        if model is None:
            if model_cfg is None:
                raise ValueError("Either model or model_cfg must be provided")
            model = CharCNNCRF(model_cfg)
        self.model = model.to(self.device)
        self.teacher = teacher
        self.teacher_model_type = teacher_model_type
        self.teacher_tagset = teacher_tagset
        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        self.scheduler = None
        if train_cfg.lr_scheduler == "plateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=train_cfg.scheduler_factor,
                patience=train_cfg.scheduler_patience,
            )
        if bert_tokenizer_name:
            self.train_ds, self.dev_ds, self.test_ds = build_bert_datasets(
                corpus,
                bert_tokenizer_name,
                max_len=train_cfg.max_len,
            )
            self._collate_fn = collate_bert_batch
        else:
            self.train_ds, self.dev_ds, self.test_ds = build_datasets(
                corpus,
                max_len=train_cfg.max_len,
                max_word_len=train_cfg.max_word_len,
            )
            self._collate_fn = collate_batch
        self.output_dir = Path(train_cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _loader(self, ds: NERDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.train_cfg.batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
        )

    def _to_device(self, batch: Batch | BertBatch) -> Batch | BertBatch:
        if self.is_bert:
            assert isinstance(batch, BertBatch)
            return BertBatch(
                input_ids=batch.input_ids.to(self.device),
                attention_mask=batch.attention_mask.to(self.device),
                tags=batch.tags.to(self.device),
                mask=batch.mask.to(self.device),
                word_mask=batch.word_mask.to(self.device),
            )
        return Batch(
            input_ids=batch.input_ids.to(self.device),
            word_ids=batch.word_ids.to(self.device),
            tags=batch.tags.to(self.device),
            mask=batch.mask.to(self.device),
        )

    def _features(self, batch: Batch | BertBatch) -> torch.Tensor:
        if self.is_bert:
            return batch.input_ids  # type: ignore[union-attr]
        if self.model_type == "bilstm_crf":
            return batch.word_ids  # type: ignore[union-attr]
        return batch.input_ids  # type: ignore[union-attr]

    def _forward(self, batch: Batch | BertBatch) -> dict[str, torch.Tensor]:
        batch = self._to_device(batch)
        if self.is_bert:
            return self.model(
                batch.input_ids,
                batch.attention_mask,
                tags=batch.tags,
                mask=batch.mask,
            )
        if self.model_type == "bilstm_char_crf":
            return self.model(
                batch.word_ids,
                tags=batch.tags,
                mask=batch.mask,
                char_ids=batch.input_ids,
            )
        features = self._features(batch)
        return self.model(features, tags=batch.tags, mask=batch.mask)

    def _decode_batch(self, batch: Batch | BertBatch) -> list[list[int]]:
        batch = self._to_device(batch)
        if self.is_bert:
            return self.model.decode(batch.input_ids, batch.attention_mask, batch.mask)
        if self.model_type == "bilstm_char_crf":
            return self.model.decode(batch.word_ids, batch.mask, batch.input_ids)
        features = self._features(batch)
        return self.model.decode(features, batch.mask)

    def train(self) -> dict[str, float]:
        train_loader = self._loader(self.train_ds, shuffle=True)
        best_f1 = -1.0
        best_epoch = 0
        best_dev_metrics: dict[str, float] = {}
        best_path = self.output_dir / "best.pt"
        stale_epochs = 0

        for epoch in range(1, self.train_cfg.epochs + 1):
            self.model.train()
            total_loss = 0.0
            for batch in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
                batch = self._to_device(batch)
                loss = self._step(batch)
                self.optimizer.zero_grad()
                loss.backward()
                if self.train_cfg.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.train_cfg.grad_clip_norm,
                    )
                self.optimizer.step()
                total_loss += float(loss.item())

            metrics = self.evaluate(self.dev_ds)
            if self.scheduler is not None:
                self.scheduler.step(metrics["f1"])
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_epoch = epoch
                best_dev_metrics = dict(metrics)
                stale_epochs = 0
                torch.save(
                    {
                        "model_state": self.model.state_dict(),
                        "model_config": self.model.config.__dict__,
                        "model_type": self.model_type,
                        "tagset": self.corpus.tagset.tags,
                        "epoch": epoch,
                        "dev_metrics": best_dev_metrics,
                        "char_vocab_path": str(self.output_dir / "char_vocab.json"),
                        "word_vocab_path": str(self.output_dir / "word_vocab.json"),
                    },
                    best_path,
                )
            else:
                stale_epochs += 1
            lr = self.optimizer.param_groups[0]["lr"]
            msg = (
                f"epoch={epoch} loss={total_loss/len(train_loader):.4f} lr={lr:.6f} "
                f"dev_f1={metrics['f1']:.4f} dev_p={metrics['precision']:.4f} dev_r={metrics['recall']:.4f}"
            )
            if self.train_cfg.log_test_each_epoch:
                test_metrics = self.evaluate(self.test_ds)
                msg += (
                    f" test_f1={test_metrics['f1']:.4f} test_p={test_metrics['precision']:.4f} "
                    f"test_r={test_metrics['recall']:.4f}"
                )
            print(msg)
            if (
                self.train_cfg.early_stop_patience is not None
                and epoch >= self.train_cfg.min_epochs
                and stale_epochs >= self.train_cfg.early_stop_patience
            ):
                print(
                    f"early_stop at epoch={epoch} "
                    f"(no dev_f1 improvement for {stale_epochs} epochs, "
                    f"best={best_f1:.4f} at epoch={best_epoch})"
                )
                break

        self.model.load_state_dict(torch.load(best_path, map_location=self.device)["model_state"])
        test_metrics = self.evaluate(self.test_ds)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.corpus.char_vocab.save(self.output_dir / "char_vocab.json")
        if self.corpus.word_vocab is not None:
            self.corpus.word_vocab.save(self.output_dir / "word_vocab.json")
        summary = {
            "eval_scheme": self.train_cfg.eval_scheme,
            "best_epoch": best_epoch,
            "best_dev": best_dev_metrics,
            "test": test_metrics,
        }
        (self.output_dir / "metrics.json").write_text(
            __import__("json").dumps(summary, indent=2),
            encoding="utf-8",
        )
        return test_metrics

    def _step(self, batch: Batch) -> torch.Tensor:
        out = self._forward(batch)
        loss = out["loss"]
        if self.teacher is not None:
            with torch.no_grad():
                if self.teacher_model_type == "bert_crf":
                    teacher_out = self.teacher(
                        batch.input_ids,
                        batch.attention_mask,
                        mask=batch.mask,
                    )
                elif self.teacher_model_type == "bilstm_char_crf":
                    teacher_out = self.teacher(
                        batch.word_ids,
                        tags=batch.tags,
                        mask=batch.mask,
                        char_ids=batch.input_ids,
                    )
                elif self.teacher_model_type == "bilstm_crf":
                    teacher_out = self.teacher(
                        batch.word_ids,
                        tags=batch.tags,
                        mask=batch.mask,
                    )
                else:
                    features = self._features(batch)
                    teacher_out = self.teacher(features, mask=batch.mask)
            student_tags = self.corpus.tagset.tags
            teacher_tags = self.teacher_tagset or student_tags
            student_emissions, teacher_emissions = align_emissions_for_distill(
                out["emissions"],
                teacher_out["emissions"],
                student_tags,
                teacher_tags,
            )
            loss = loss + distillation_loss(
                student_emissions,
                teacher_emissions,
                batch.mask,
                temperature=self.train_cfg.distill_temperature,
                alpha=self.train_cfg.distill_alpha,
            )
            student_trans, teacher_trans = align_transition_matrix(
                self.model.transition_matrix(),
                self.teacher.transition_matrix(),
                student_tags,
                teacher_tags,
            )
            loss = loss + transition_distill_loss(
                student_trans,
                teacher_trans,
            ) * self.train_cfg.lambda_transition
        return loss

    @torch.no_grad()
    def evaluate(self, dataset: NERDataset) -> dict[str, float]:
        self.model.eval()
        loader = self._loader(dataset, shuffle=False)
        id2tag = self.corpus.tagset.id2tag
        y_true, y_pred = [], []
        for batch in loader:
            pred_ids = self._decode_batch(batch)
            batch = self._to_device(batch)
            for i, seq_pred in enumerate(pred_ids):
                if self.is_bert:
                    assert isinstance(batch, BertBatch)
                    eval_mask = batch.word_mask[i].bool()
                    gold = [
                        id2tag[int(t)]
                        for t, keep in zip(batch.tags[i].tolist(), eval_mask.tolist())
                        if keep
                    ]
                    pred = [
                        id2tag[p]
                        for p, keep in zip(seq_pred, eval_mask.tolist())
                        if keep
                    ]
                else:
                    length = int(batch.mask[i].sum().item())
                    gold = [id2tag[int(t)] for t in batch.tags[i, :length].tolist()]
                    pred = [id2tag[p] for p in seq_pred[:length]]
                y_true.append(gold)
                y_pred.append(pred)
        return compute_ner_metrics(
            y_true,
            y_pred,
            scheme=self.train_cfg.eval_scheme,
        )


class FewShotTrainer(FullTrainer):
    def _step(self, batch: Batch) -> torch.Tensor:
        batch = self._to_device(batch)
        if self.model_type == "bilstm_char_crf":
            token_repr = self.model.encode(batch.word_ids, batch.input_ids, batch.mask)
            out = self.model(
                batch.word_ids,
                tags=batch.tags,
                mask=batch.mask,
                char_ids=batch.input_ids,
            )
        else:
            features = self._features(batch)
            token_repr = self.model.encode(features, batch.mask)
            out = self.model(features, tags=batch.tags, mask=batch.mask)
        loss = out["loss"]
        loss = loss + self.train_cfg.lambda_proto * prototypical_loss(
            token_repr, batch.tags, batch.mask
        )
        loss = loss + self.train_cfg.lambda_contrastive * supervised_contrastive_loss(
            token_repr, batch.tags, batch.mask
        )
        return loss
