from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from edgefs.data.cluener import read_cluener_json
from edgefs.data.conll import read_conll, write_conll
from edgefs.data.schema import NERSentence, TagSet
from edgefs.data.vocab import CharVocab
from edgefs.data.word_vocab import WordVocab


@dataclass
class NERCorpus:
    train: list[NERSentence]
    dev: list[NERSentence]
    test: list[NERSentence]
    tagset: TagSet
    char_vocab: CharVocab
    word_vocab: WordVocab | None = None


def load_processed_corpus(processed_dir: str | Path) -> NERCorpus:
    processed_dir = Path(processed_dir)
    train = read_conll(processed_dir / "train.conll")
    dev = read_conll(processed_dir / "dev.conll")
    test = read_conll(processed_dir / "test.conll")
    tagset = TagSet.from_sentences(train + dev + test)
    char_vocab = CharVocab.load(processed_dir / "char_vocab.json")
    word_vocab_path = processed_dir / "word_vocab.json"
    if word_vocab_path.exists():
        word_vocab = WordVocab.load(word_vocab_path)
    else:
        word_vocab = WordVocab.build(train)
        word_vocab.save(word_vocab_path)
    return NERCorpus(
        train=train, dev=dev, test=test, tagset=tagset, char_vocab=char_vocab, word_vocab=word_vocab
    )


def save_processed_corpus(corpus: NERCorpus, processed_dir: str | Path) -> None:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    write_conll(processed_dir / "train.conll", corpus.train)
    write_conll(processed_dir / "dev.conll", corpus.dev)
    write_conll(processed_dir / "test.conll", corpus.test)
    corpus.char_vocab.save(processed_dir / "char_vocab.json")
    if corpus.word_vocab is not None:
        corpus.word_vocab.save(processed_dir / "word_vocab.json")
    (processed_dir / "tags.txt").write_text(
        "\n".join(corpus.tagset.tags), encoding="utf-8"
    )


def prepare_conll2003(
    raw_dir: str | Path,
    processed_dir: str | Path,
    lowercase: bool = False,
) -> NERCorpus:
    """Download CoNLL-2003 and export unified CoNLL files."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    from edgefs.data.conll2003_download import load_conll2003_splits

    splits = load_conll2003_splits(raw_dir, lowercase=lowercase)
    print("Loaded CoNLL-2003 via direct download")

    tagset = TagSet.from_sentences(splits["train"] + splits["dev"] + splits["test"])
    char_vocab = CharVocab.build(splits["train"])
    word_vocab = WordVocab.build(splits["train"])
    corpus = NERCorpus(
        train=splits["train"],
        dev=splits["dev"],
        test=splits["test"],
        tagset=tagset,
        char_vocab=char_vocab,
        word_vocab=word_vocab,
    )
    save_processed_corpus(corpus, processed_dir)
    return corpus


def prepare_cluener(
    raw_dir: str | Path,
    processed_dir: str | Path,
    encoding: str = "char",
) -> NERCorpus:
    """Prepare CLUENER from local JSON lines (auto-download if missing)."""
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    _ensure_cluener_raw(raw_dir)
    train = read_cluener_json(raw_dir / "train.json")
    dev = read_cluener_json(raw_dir / "dev.json")
    test = read_cluener_json(raw_dir / "test.json")
    if not any(any(tag != "O" for tag in sent.tags) for sent in test):
        print("CLUENER test.json has no gold labels; using dev.json for test evaluation")
        test = dev
    tagset = TagSet.from_sentences(train + dev + test)
    if encoding == "byte":
        char_vocab = CharVocab.build_byte()
    else:
        char_vocab = CharVocab.build(train)
    word_vocab = WordVocab.build(train)
    corpus = NERCorpus(
        train=train, dev=dev, test=test, tagset=tagset, char_vocab=char_vocab, word_vocab=word_vocab
    )
    save_processed_corpus(corpus, processed_dir)
    return corpus


def _cluener_raw_ready(raw_dir: Path) -> bool:
    train = raw_dir / "train.json"
    if not train.exists() or train.stat().st_size == 0:
        return False
    try:
        read_cluener_json(train)
        return True
    except Exception:
        return False


def _ensure_cluener_raw(raw_dir: Path) -> None:
    if _cluener_raw_ready(raw_dir):
        return
    import io
    import urllib.request
    import zipfile

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_urls = (
        "https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip",
        "https://ghfast.top/https://storage.googleapis.com/cluebenchmark/tasks/cluener_public.zip",
    )
    last_err: Exception | None = None
    for url in zip_urls:
        try:
            print(f"Downloading CLUENER archive from {url}")
            with urllib.request.urlopen(url, timeout=120) as resp:
                archive = zipfile.ZipFile(io.BytesIO(resp.read()))
            for name in ("train.json", "dev.json", "test.json"):
                try:
                    archive.extract(name, raw_dir)
                except KeyError as exc:
                    raise RuntimeError(f"CLUENER archive missing {name}") from exc
            print("CLUENER raw files ready")
            return
        except Exception as exc:
            last_err = exc
            print(f"  failed: {exc}")
    raise RuntimeError(f"CLUENER download failed: {last_err}") from last_err


def _hf_rows_to_sentences(rows) -> list[NERSentence]:
    sentences: list[NERSentence] = []
    for row in rows:
        tokens = row["tokens"]
        tags = row.get("tags") or row.get("ner_tags")
        if tags and isinstance(tags[0], int):
            tag_names = row.get("tag_names")
            if tag_names:
                id2tag = {i: t for i, t in enumerate(tag_names)}
            else:
                id2tag = {
                    0: "O",
                    1: "B-PER",
                    2: "I-PER",
                    3: "B-ORG",
                    4: "I-ORG",
                    5: "B-LOC",
                    6: "I-LOC",
                    7: "B-MISC",
                    8: "I-MISC",
                }
            tags = [id2tag[i] for i in tags]
        sentences.append(NERSentence(tokens=tokens, tags=list(tags)))
    return sentences


def _read_tner_jsonl(path: Path, id2tag: dict[int, str]) -> list[NERSentence]:
    sentences: list[NERSentence] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        tokens = row["tokens"]
        tags = row.get("tags") or row.get("ner_tags")
        if tags and isinstance(tags[0], int):
            tags = [id2tag[i] for i in tags]
        sentences.append(NERSentence(tokens=tokens, tags=list(tags)))
    return sentences


def _load_tner_ontonotes_raw(raw_dir: Path) -> tuple[list[NERSentence], list[NERSentence], list[NERSentence]]:
    dataset_dir = raw_dir / "dataset"
    label_path = dataset_dir / "label.json"
    if not label_path.exists():
        raise FileNotFoundError(
            f"Missing {label_path}. Place tner/ontonotes5 JSON shards under {dataset_dir} "
            "or run scripts/download_hf_assets.py locally and upload to the server."
        )
    tag2id = json.loads(label_path.read_text(encoding="utf-8"))
    id2tag = {idx: tag for tag, idx in tag2id.items()}
    train: list[NERSentence] = []
    for shard in sorted(dataset_dir.glob("train*.json")):
        train.extend(_read_tner_jsonl(shard, id2tag))
    dev = _read_tner_jsonl(dataset_dir / "valid.json", id2tag)
    test = _read_tner_jsonl(dataset_dir / "test.json", id2tag)
    return train, dev, test


def _download_tner_ontonotes(raw_dir: Path) -> None:
    """Download tner/ontonotes5 JSON shards (datasets>=3 no longer runs ontonotes5.py)."""
    from huggingface_hub import hf_hub_download

    from edgefs.utils.hf_mirror import use_hf_mirror

    dataset_dir = raw_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    files = [
        "label.json",
        "train00.json",
        "train01.json",
        "train02.json",
        "train03.json",
        "valid.json",
        "test.json",
    ]
    endpoints = [use_hf_mirror(), None]
    last_err: Exception | None = None
    for endpoint in endpoints:
        if endpoint is None:
            import os

            os.environ.pop("HF_ENDPOINT", None)
            os.environ.pop("HUGGINGFACE_HUB_ENDPOINT", None)
            print("Retrying OntoNotes download via huggingface.co")
        else:
            print(f"Downloading OntoNotes via {endpoint}")
        try:
            for name in files:
                dest = dataset_dir / name
                if dest.exists() and dest.stat().st_size > 0:
                    continue
                path = hf_hub_download(
                    repo_id="tner/ontonotes5",
                    filename=f"dataset/{name}",
                    repo_type="dataset",
                    local_dir=str(raw_dir),
                )
                print(f"  {name} -> {path}")
            return
        except Exception as exc:
            last_err = exc
            print(f"  failed: {exc}")
    raise RuntimeError("OntoNotes download failed; use scripts/download_hf_assets.py + upload_hf_assets.py") from last_err


def prepare_ontonotes(raw_dir: str | Path, processed_dir: str | Path) -> NERCorpus:
    """Prepare OntoNotes 5.0 English NER from tner/ontonotes5 JSON shards."""
    processed_dir = Path(processed_dir)
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not (raw_dir / "dataset" / "label.json").exists():
        _download_tner_ontonotes(raw_dir)
    train, dev, test = _load_tner_ontonotes_raw(raw_dir)
    tagset = TagSet.from_sentences(train + dev + test)
    char_vocab = CharVocab.build(train)
    word_vocab = WordVocab.build(train)
    corpus = NERCorpus(
        train=train, dev=dev, test=test, tagset=tagset, char_vocab=char_vocab, word_vocab=word_vocab
    )
    save_processed_corpus(corpus, processed_dir)
    return corpus


def prepare_toy(processed_dir: str | Path) -> NERCorpus:
    """Tiny English corpus for local smoke tests without downloads."""
    train = read_conll(Path(__file__).resolve().parents[3] / "data" / "toy" / "train.conll")
    dev = read_conll(Path(__file__).resolve().parents[3] / "data" / "toy" / "dev.conll")
    test = read_conll(Path(__file__).resolve().parents[3] / "data" / "toy" / "test.conll")
    tagset = TagSet.from_sentences(train + dev + test)
    char_vocab = CharVocab.build(train)
    word_vocab = WordVocab.build(train)
    corpus = NERCorpus(
        train=train, dev=dev, test=test, tagset=tagset, char_vocab=char_vocab, word_vocab=word_vocab
    )
    save_processed_corpus(corpus, processed_dir)
    return corpus


def _hf_to_sentence(row: dict) -> NERSentence:
    tokens = row["tokens"]
    tags = row["ner_tags"]
    if isinstance(tags[0], int):
        id2tag = {
            0: "O",
            1: "B-PER",
            2: "I-PER",
            3: "B-ORG",
            4: "I-ORG",
            5: "B-LOC",
            6: "I-LOC",
            7: "B-MISC",
            8: "I-MISC",
        }
        tags = [id2tag[i] for i in tags]
    return NERSentence(tokens=tokens, tags=tags)
