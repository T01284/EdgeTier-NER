from __future__ import annotations

from dataclasses import replace

import torch
from torch.utils.data import DataLoader

from edgefs.data.bert_dataset import BertBatch, BertWordDataset, collate_bert_batch
from edgefs.data.prepare import NERCorpus
from edgefs.data.schema import NERSentence
from edgefs.evaluation.metrics import compute_ner_metrics
from edgefs.fewshot.baselines import (
    collect_word_token_reprs,
    estimate_support_transitions,
    nearest_neighbor_predict,
    structural_transition_loss,
)
from edgefs.models.bert_crf import BertCRF, BertCRFConfig
from edgefs.models.char_cnn_crf import CharCNNCRF, CharCNNCRFConfig
from edgefs.training.losses import prototypical_loss, supervised_contrastive_loss
from edgefs.training.trainer import FewShotTrainer, TrainConfig


def build_char_model(corpus: NERCorpus, model_cfg_raw: dict) -> tuple[CharCNNCRF, CharCNNCRFConfig]:
    model_cfg = CharCNNCRFConfig(
        vocab_size=len(corpus.char_vocab),
        num_tags=len(corpus.tagset.tags),
        embed_dim=model_cfg_raw.get("embed_dim", 64),
        num_filters=model_cfg_raw.get("num_filters", 128),
        kernel_size=model_cfg_raw.get("kernel_size", 3),
        dilations=tuple(model_cfg_raw.get("dilations", [1, 2, 4])),
        dropout=model_cfg_raw.get("dropout", 0.1),
        use_residual=model_cfg_raw.get("use_residual", True),
        max_word_len=int(model_cfg_raw.get("max_word_len", 20)),
    )
    return CharCNNCRF(model_cfg), model_cfg


def build_bert_model(corpus: NERCorpus, model_cfg_raw: dict) -> tuple[BertCRF, BertCRFConfig]:
    model_cfg = BertCRFConfig(
        pretrained_name=model_cfg_raw.get("pretrained_name", "bert-base-cased"),
        num_tags=len(corpus.tagset.tags),
        dropout=float(model_cfg_raw.get("dropout", 0.1)),
        freeze_backbone=bool(model_cfg_raw.get("freeze_backbone", False)),
    )
    return BertCRF(model_cfg), model_cfg


class OursFewShotTrainer(FewShotTrainer):
    """Char-CNN-CRF + prototypical + contrastive (default Ours few-shot)."""

    def _step(self, batch) -> torch.Tensor:
        batch = self._to_device(batch)
        if self.model_type == "bilstm_char_crf":
            token_repr = self.model.encode(batch.word_ids, batch.input_ids, batch.mask)
            out = self.model(
                batch.word_ids,
                tags=batch.tags,
                mask=batch.mask,
                char_ids=batch.input_ids,
            )
        elif self.is_bert:
            token_repr = self.model.encode(batch.input_ids, batch.attention_mask)
            out = self.model(
                batch.input_ids,
                batch.attention_mask,
                tags=batch.tags,
                mask=batch.mask,
            )
        else:
            features = self._features(batch)
            token_repr = self.model.encode(features, batch.mask)
            out = self.model(features, tags=batch.tags, mask=batch.mask)
        loss = out["loss"]
        loss = loss + self.train_cfg.lambda_proto * prototypical_loss(
            token_repr,
            batch.tags,
            batch.mask,
            temperature=self.train_cfg.proto_temperature,
        )
        loss = loss + self.train_cfg.lambda_contrastive * supervised_contrastive_loss(
            token_repr, batch.tags, batch.mask
        )
        return loss


class ProtoBertFewShotTrainer(OursFewShotTrainer):
    """BERT-CRF episodic fine-tuning with prototypical token loss (Fritzler-style ProtoBERT)."""

    def __init__(
        self,
        corpus: NERCorpus,
        train_cfg: TrainConfig,
        model: BertCRF,
        model_cfg: BertCRFConfig,
        tokenizer_name: str,
    ) -> None:
        super().__init__(
            corpus,
            train_cfg,
            model=model,
            model_cfg=None,
            model_type="bert_crf",
            bert_tokenizer_name=tokenizer_name,
        )


class StructShotFewShotTrainer(ProtoBertFewShotTrainer):
    """BERT-CRF + prototypical loss + support-structure transition regularization."""

    def __init__(
        self,
        corpus: NERCorpus,
        train_cfg: TrainConfig,
        model: BertCRF,
        model_cfg: BertCRFConfig,
        tokenizer_name: str,
        support_sentences: list[NERSentence],
    ) -> None:
        super().__init__(corpus, train_cfg, model, model_cfg, tokenizer_name)
        self.support_sentences = support_sentences
        self._support_trans = estimate_support_transitions(
            support_sentences,
            self.corpus.tagset.tag2id,
            len(self.corpus.tagset.tags),
            self.device,
        )

    def _step(self, batch) -> torch.Tensor:
        loss = super()._step(batch)
        if self.train_cfg.lambda_struct > 0:
            loss = loss + self.train_cfg.lambda_struct * structural_transition_loss(
                self.model.transition_matrix(),
                self._support_trans,
            )
        return loss


class NNShotFewShotTrainer(ProtoBertFewShotTrainer):
    """Support-only BERT fine-tune + token-level nearest-neighbor decoding at test time."""

    def __init__(
        self,
        corpus: NERCorpus,
        train_cfg: TrainConfig,
        model: BertCRF,
        model_cfg: BertCRFConfig,
        tokenizer_name: str,
        support_sentences: list[NERSentence],
    ) -> None:
        train_corpus = replace(corpus, train=support_sentences)
        super().__init__(train_corpus, train_cfg, model, model_cfg, tokenizer_name)
        self.support_sentences = support_sentences
        self.tokenizer_name = tokenizer_name

    @torch.no_grad()
    def evaluate(self, dataset) -> dict[str, float]:
        support_repr, support_labels = collect_word_token_reprs(
            self.model,
            self.support_sentences,
            self.corpus.tagset.tag2id,
            self.tokenizer_name,
            self.device,
            max_len=self.train_cfg.max_len,
            batch_size=self.train_cfg.batch_size,
        )
        self.model.eval()
        loader = DataLoader(
            dataset,
            batch_size=self.train_cfg.batch_size,
            shuffle=False,
            collate_fn=collate_bert_batch,
        )
        id2tag = self.corpus.tagset.id2tag
        y_true, y_pred = [], []
        for batch in loader:
            batch = self._to_device(batch)
            assert isinstance(batch, BertBatch)
            hidden = self.model.encode(batch.input_ids, batch.attention_mask)
            for i in range(hidden.size(0)):
                eval_mask = batch.word_mask[i].bool()
                gold = [
                    id2tag[int(t)]
                    for t, keep in zip(batch.tags[i].tolist(), eval_mask.tolist())
                    if keep
                ]
                q_repr = hidden[i][eval_mask]
                if q_repr.numel() == 0:
                    continue
                pred_ids = nearest_neighbor_predict(
                    q_repr,
                    support_repr,
                    support_labels,
                    k=self.train_cfg.nn_k,
                )
                pred = [id2tag[int(t)] for t in pred_ids.tolist()]
                y_true.append(gold)
                y_pred.append(pred)
        return compute_ner_metrics(y_true, y_pred, scheme=self.train_cfg.eval_scheme)


def build_fewshot_trainer(
    method: str,
    corpus: NERCorpus,
    train_cfg: TrainConfig,
    model_cfg_raw: dict,
    support_sentences: list[NERSentence],
):
    method = method.lower()
    if method in ("ours", "char", "default"):
        model, model_cfg = build_char_model(corpus, model_cfg_raw)
        return OursFewShotTrainer(
            corpus,
            train_cfg,
            model=model,
            model_cfg=model_cfg,
            model_type="char_cnn_crf",
        )

    model, bert_cfg = build_bert_model(corpus, model_cfg_raw)
    tokenizer_name = model_cfg_raw.get("pretrained_name", "bert-base-cased")

    if method == "protobert":
        return ProtoBertFewShotTrainer(corpus, train_cfg, model, bert_cfg, tokenizer_name)
    if method == "structshot":
        return StructShotFewShotTrainer(
            corpus, train_cfg, model, bert_cfg, tokenizer_name, support_sentences
        )
    if method == "nnshot":
        return NNShotFewShotTrainer(
            corpus, train_cfg, model, bert_cfg, tokenizer_name, support_sentences
        )
    raise ValueError(f"Unknown few-shot method: {method}")
