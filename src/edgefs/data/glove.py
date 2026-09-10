from __future__ import annotations

import gzip
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch

from edgefs.data.word_vocab import WordVocab

GLOVE_DIM_FILES = {
    50: "glove.6B.50d.txt",
    100: "glove.6B.100d.txt",
    200: "glove.6B.200d.txt",
    300: "glove.6B.300d.txt",
}
GLOVE_6B_100D_FILE = GLOVE_DIM_FILES[100]
MIN_GLOVE_BYTES = {
    50: 50_000_000,
    100: 100_000_000,
    200: 200_000_000,
    300: 300_000_000,
}
GLOVE_HF_ZIP_URL = "https://hf-mirror.com/stanfordnlp/glove/resolve/main/glove.6B.zip"
GLOVE_HF_ZIP_FALLBACK = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
GLOVE_STANFORD_ZIP_URL = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
GLOVE_GZ_URL = "https://hf-mirror.com/datasets/SLU-CSCI4750/glove.6B.100d.txt/resolve/main/glove.6B.100d.txt.gz"


def _download_file(url: str, dest: Path) -> None:
    print(f"Downloading {dest.name} from {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if _try_wget(url, dest):
        return
    urlretrieve(url, dest)


def _try_wget(url: str, dest: Path) -> bool:
    if shutil.which("wget") is None:
        return False
    import subprocess

    cmd = ["wget", "-c", "-O", str(dest), url]
    print("Using wget:", " ".join(cmd))
    result = subprocess.run(cmd, check=False)
    return result.returncode == 0 and dest.exists() and dest.stat().st_size > 0


def _extract_from_zip(zip_path: Path, glove_path: Path, dim: int) -> None:
    member_name = GLOVE_DIM_FILES[dim]
    with zipfile.ZipFile(zip_path) as zf:
        if member_name in zf.namelist():
            member = member_name
        else:
            suffix = f"{dim}d.txt"
            members = [n for n in zf.namelist() if n.endswith(suffix)]
            if not members:
                raise FileNotFoundError(f"No {dim}d GloVe file inside {zip_path}")
            member = members[0]
        with zf.open(member) as src, glove_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)


def _ensure_glove_zip(embed_dir: Path) -> Path:
    zip_path = embed_dir / "glove.6B.zip"
    if zip_path.exists() and zip_path.stat().st_size < 100_000_000:
        print(f"Removing incomplete archive: {zip_path}")
        zip_path.unlink()

    if not zip_path.exists() or zip_path.stat().st_size < 100_000_000:
        for zip_url in (GLOVE_HF_ZIP_URL, GLOVE_HF_ZIP_FALLBACK, GLOVE_STANFORD_ZIP_URL):
            try:
                _download_file(zip_url, zip_path)
                if zip_path.stat().st_size > 100_000_000:
                    print(f"Downloaded GloVe zip from {zip_url}")
                    break
            except Exception as exc:
                print(f"Zip mirror failed ({zip_url}): {exc}")
                zip_path.unlink(missing_ok=True)

    if not zip_path.exists() or zip_path.stat().st_size < 100_000_000:
        raise RuntimeError("Failed to download GloVe archive from all mirrors")
    return zip_path


def ensure_glove(embed_dir: str | Path, dim: int = 100) -> Path:
    if dim not in GLOVE_DIM_FILES:
        raise ValueError(f"Unsupported GloVe dim: {dim}. Choose from {sorted(GLOVE_DIM_FILES)}")

    embed_dir = Path(embed_dir)
    embed_dir.mkdir(parents=True, exist_ok=True)
    glove_path = embed_dir / GLOVE_DIM_FILES[dim]
    min_bytes = MIN_GLOVE_BYTES[dim]
    if glove_path.exists() and glove_path.stat().st_size > min_bytes:
        print(f"Using cached GloVe: {glove_path}")
        return glove_path

    if dim == 100:
        gz_path = embed_dir / "glove.6B.100d.txt.gz"
        try:
            _download_file(GLOVE_GZ_URL, gz_path)
            with gzip.open(gz_path, "rb") as src, glove_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            if glove_path.stat().st_size > min_bytes:
                print(f"Downloaded GloVe from {GLOVE_GZ_URL}")
                return glove_path
        except Exception as exc:
            print(f"GloVe gz mirror failed: {exc}")
            glove_path.unlink(missing_ok=True)

    zip_path = _ensure_glove_zip(embed_dir)
    _extract_from_zip(zip_path, glove_path, dim)
    print(f"Ready: {glove_path} ({glove_path.stat().st_size // (1024 * 1024)} MB)")
    return glove_path


def ensure_glove_100d(embed_dir: str | Path) -> Path:
    return ensure_glove(embed_dir, dim=100)


def load_glove_vectors(glove_path: str | Path) -> dict[str, np.ndarray]:
    glove_path = Path(glove_path)
    vectors: dict[str, np.ndarray] = {}
    with glove_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < 3:
                continue
            word = parts[0]
            vectors[word] = np.asarray(parts[1:], dtype=np.float32)
    return vectors


def _glove_mean_vector(glove_path: Path, embed_dim: int) -> np.ndarray:
    total = np.zeros(embed_dim, dtype=np.float64)
    count = 0
    with glove_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != embed_dim + 1:
                continue
            total += np.asarray(parts[1:], dtype=np.float64)
            count += 1
    if count == 0:
        raise RuntimeError(f"No GloVe vectors found in {glove_path}")
    return (total / count).astype(np.float32)


def build_embedding_matrix(
    word_vocab: WordVocab,
    glove_path: str | Path,
    embed_dim: int = 100,
) -> torch.Tensor:
    glove_path = Path(glove_path)
    glove_mean = _glove_mean_vector(glove_path, embed_dim)
    matrix = np.tile(glove_mean, (len(word_vocab), 1))
    matrix[word_vocab.pad_id] = 0.0
    matrix[word_vocab.unk_id] = glove_mean

    found = 0
    with glove_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != embed_dim + 1:
                continue
            word = parts[0]
            word_id = word_vocab.word2id.get(word)
            if word_id is None:
                continue
            matrix[word_id] = np.asarray(parts[1:], dtype=np.float32)
            found += 1

    if found == 0:
        raise RuntimeError(f"No GloVe vectors matched vocabulary from {glove_path}")
    print(f"Matched {found}/{len(word_vocab)} words in GloVe (UNK/missing use mean vector)")
    return torch.tensor(matrix)
