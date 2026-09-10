#!/usr/bin/env python3
"""Fast upload of A–E batch files + nohup start (skip full pip setup)."""

from __future__ import annotations

import argparse
import sys
import tarfile
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_ssh import connect, load_remote_config, run

ROOT = Path(__file__).resolve().parents[1]
BASE = "/root/autodl-tmp/EdgeFS_NER"
INCLUDE_PREFIXES = (
    "scripts/",
    "src/",
    "configs/",
    "requirements/",
    "pyproject.toml",
    "setup.cfg",
    "setup.py",
)


def should_include(rel: str) -> bool:
    return any(rel == p or rel.startswith(p) for p in INCLUDE_PREFIXES)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume-bcde",
        action="store_true",
        help="Resume from Stage B (skip completed Stage A)",
    )
    args = parser.parse_args()

    cfg = load_remote_config()
    client = connect(cfg)
    remote_dir = cfg["project_dir"]
    script = (
        "run_gpu_abcde_resume_bcde.sh" if args.resume_bcde else "run_gpu_abcde.sh"
    )
    try:
        run(client, f"mkdir -p {remote_dir} {BASE}/logs {BASE}/deploy/results", timeout=30)

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tar_path = Path(tmp.name)
        try:
            with tarfile.open(tar_path, "w:gz") as tar:
                for path in ROOT.rglob("*"):
                    if not path.is_file():
                        continue
                    rel = path.relative_to(ROOT).as_posix()
                    if not should_include(rel) or "__pycache__" in rel:
                        continue
                    tar.add(path, arcname=rel)
            size_mb = tar_path.stat().st_size / (1024 * 1024)
            print(f"Uploading {size_mb:.1f} MB ...")
            sftp = client.open_sftp()
            remote_tar = f"{remote_dir}/.edgefs_abcde_sync.tar.gz"
            sftp.put(str(tar_path), remote_tar)
            sftp.close()
            code, out, err = run(
                client,
                f"cd {remote_dir} && tar -xzf .edgefs_abcde_sync.tar.gz && rm -f .edgefs_abcde_sync.tar.gz",
                timeout=120,
            )
            if code != 0:
                print(err or out)
                return code
            print("Upload OK")
        finally:
            tar_path.unlink(missing_ok=True)

        code, out, err = run(
            client,
            f"bash -lc 'cd {remote_dir} && source .venv/bin/activate && "
            f"pip install -e . -q && chmod +x scripts/run_gpu_abcde.sh "
            f"scripts/run_gpu_abcde_resume_bcde.sh'",
            timeout=180,
        )
        print(out or err)

        run(client, f"rm -f {BASE}/logs/gpu_abcde.DONE", timeout=15)
        code, out, err = run(
            client,
            f"bash -lc 'nohup bash {remote_dir}/scripts/{script} "
            f"> {BASE}/logs/nohup_abcde.out 2>&1 & echo STARTED:$!'",
            timeout=30,
        )
        print(out or err)
        time.sleep(12)
        _, out, _ = run(
            client,
            f"bash -lc 'pgrep -af run_gpu_abcde || true; "
            f"tail -40 {BASE}/logs/nohup_abcde.out 2>/dev/null; "
            f"tail -20 {BASE}/logs/gpu_abcde.log 2>/dev/null'",
            timeout=60,
        )
        print("--- status ---")
        print(out or "(empty)")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
