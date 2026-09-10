"""Smoke tests runnable on local CPU without GPU or dataset downloads."""

from edgefs.data.prepare import prepare_toy
from edgefs.models.char_cnn_crf import CharCNNCRF, CharCNNCRFConfig
from edgefs.training.trainer import FullTrainer, TrainConfig


def test_toy_prepare_and_train(tmp_path):
    processed = tmp_path / "toy"
    corpus = prepare_toy(processed)
    assert len(corpus.train) > 0
    assert "O" in corpus.tagset.tags

    model_cfg = CharCNNCRFConfig(
        vocab_size=len(corpus.char_vocab),
        num_tags=len(corpus.tagset.tags),
        max_word_len=20,
    )
    train_cfg = TrainConfig(
        epochs=1,
        batch_size=2,
        device="cpu",
        output_dir=str(tmp_path / "run"),
        max_word_len=20,
    )
    trainer = FullTrainer(corpus, train_cfg, model_cfg=model_cfg)
    metrics = trainer.train()
    assert "f1" in metrics
