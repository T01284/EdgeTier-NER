from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from edgefs.data.conll import read_conll

CONLL2003_URLS = {
    "train": "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.train",
    "dev": "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testa",
    "test": "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003/eng.testb",
}

CONLL2003_ZIP = "https://data.deepai.org/conll2003.zip"


def _download_file(url: str, dest: Path) -> None:
    print(f"Downloading {dest.name} from {url}")
    urlretrieve(url, dest)


def _download_zip(raw_dir: Path) -> dict[str, Path]:
    zip_path = raw_dir / "conll2003.zip"
    _download_file(CONLL2003_ZIP, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir / "zip_extract")
    base = raw_dir / "zip_extract"
    candidates = {
        "train": list(base.rglob("eng.train")),
        "dev": list(base.rglob("eng.testa")),
        "test": list(base.rglob("eng.testb")),
    }
    paths: dict[str, Path] = {}
    for split, found in candidates.items():
        if not found:
            raise FileNotFoundError(f"Missing {split} in {CONLL2003_ZIP}")
        paths[split] = found[0]
    return paths


def download_conll2003_raw(raw_dir: str | Path) -> dict[str, Path]:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for split, url in CONLL2003_URLS.items():
        dest = raw_dir / f"{split}.txt"
        if dest.exists() and dest.stat().st_size > 0:
            paths[split] = dest
            continue
        try:
            _download_file(url, dest)
            paths[split] = dest
        except Exception as exc:
            print(f"Direct download failed for {split}: {exc}")
            return _download_zip(raw_dir)

    return paths


def load_conll2003_splits(raw_dir: str | Path, lowercase: bool = False):
    from edgefs.data.schema import NERSentence

    paths = download_conll2003_raw(raw_dir)
    return {
        "train": read_conll(paths["train"], lowercase=lowercase),
        "dev": read_conll(paths["dev"], lowercase=lowercase),
        "test": read_conll(paths["test"], lowercase=lowercase),
    }
