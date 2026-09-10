#!/usr/bin/env python3
"""Upload test script and run HF mirror test on remote."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_ssh import connect, run

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/EdgeFS_NER"

c = connect()
sftp = c.open_sftp()
sftp.put(str(ROOT / "scripts" / "test_hf_mirror.py"), f"{REMOTE}/scripts/test_hf_mirror.py")
sftp.close()

cmd = (
    "cd /root/EdgeFS_NER && source .venv/bin/activate && "
    "export HF_ENDPOINT=https://hf-mirror.com "
    "HUGGINGFACE_HUB_ENDPOINT=https://hf-mirror.com "
    "HF_HUB_ENABLE_HF_TRANSFER=0 && "
    "python scripts/test_hf_mirror.py"
)
_, out, err = run(c, f"bash -lc {cmd!r}", timeout=300)
print(out or err)
c.close()
