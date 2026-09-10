#!/usr/bin/env python3
"""SSH helpers for remote GPU server (reads configs/remote/gpu_server.yaml + .env)."""

from __future__ import annotations

import os
import re
import shlex
import tarfile
import tempfile
from pathlib import Path

import yaml

try:
    import paramiko
except ImportError as exc:
    raise SystemExit("Install paramiko: pip install paramiko") from exc


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def load_remote_config() -> dict:
    root = repo_root()
    load_dotenv(root / ".env")
    cfg_path = root / "configs" / "remote" / "gpu_server.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["password"] = os.environ.get("REMOTE_SSH_PASSWORD", "")
    cfg["host"] = cfg.get("host") or os.environ.get("REMOTE_SSH_HOST")
    cfg["user"] = cfg.get("user") or os.environ.get("REMOTE_SSH_USER", "root")
    cfg["port"] = int(cfg.get("port") or os.environ.get("REMOTE_SSH_PORT", "22"))
    cfg["project_dir"] = cfg.get("project_dir") or os.environ.get(
        "REMOTE_PROJECT_DIR", "/root/EdgeFS_NER"
    )
    return cfg


def connect(cfg: dict | None = None) -> paramiko.SSHClient:
    cfg = cfg or load_remote_config()
    if not cfg.get("password"):
        raise RuntimeError("REMOTE_SSH_PASSWORD missing in .env")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=cfg["host"],
        port=int(cfg["port"]),
        username=cfg["user"],
        password=cfg["password"],
        timeout=120,
        banner_timeout=120,
        allow_agent=False,
        look_for_keys=False,
    )
    return client


def run(client: paramiko.SSHClient, command: str, timeout: int = 600) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    exit_code = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return exit_code, out, err


EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "outputs",
    "paper/build",
    "paper/templates/elsarticle-src",
    "deploy/esp32-s3/.pio",
    "__pycache__",
    ".pytest_cache",
}


def should_skip(rel: str) -> bool:
    parts = Path(rel).parts
    if not parts:
        return False
    if parts[0] in EXCLUDE_DIRS:
        return True
    if "node_modules" in parts:
        return True
    return False


def sync_project(client: paramiko.SSHClient, cfg: dict) -> None:
    root = repo_root()
    remote_dir = cfg["project_dir"]
    run(client, f"mkdir -p {remote_dir}")

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tar_path = Path(tmp.name)

    try:
        with tarfile.open(tar_path, "w:gz") as tar:
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                rel = path.relative_to(root).as_posix()
                if should_skip(rel):
                    continue
                if rel.startswith("data/raw/") or rel.startswith("data/processed/"):
                    continue
                if rel.endswith(".pt") or rel.endswith(".pth") or rel.endswith(".onnx"):
                    continue
                tar.add(path, arcname=rel)

        sftp = client.open_sftp()
        remote_tar = f"{remote_dir}/.edgefs_sync.tar.gz"
        sftp.put(str(tar_path), remote_tar)
        sftp.close()
        code, out, err = run(
            client,
            f"cd {remote_dir} && tar -xzf .edgefs_sync.tar.gz && rm -f .edgefs_sync.tar.gz",
            timeout=300,
        )
        if code != 0:
            raise RuntimeError(f"Remote extract failed: {err or out}")
    finally:
        tar_path.unlink(missing_ok=True)


def setup_remote_env(client: paramiko.SSHClient, cfg: dict) -> None:
    remote_dir = cfg["project_dir"]
    py = cfg.get("python", "python3")
    commands = " && ".join(
        [
            f"cd {remote_dir}",
            f"{py} -m venv .venv",
            "source .venv/bin/activate",
            "pip install --upgrade pip setuptools wheel",
            "pip install -e .",
            "pip install -r requirements/gpu.txt",
            'python -c "import torch; print(\'torch\', torch.__version__, \'cuda\', torch.cuda.is_available())"',
            "nvidia-smi || true",
        ]
    )
    code, out, err = run(client, f"bash -lc {shlex.quote(commands)}", timeout=1800)
    print(out)
    if err:
        print(err)
    if code != 0:
        raise RuntimeError(f"Remote setup failed with exit code {code}")
