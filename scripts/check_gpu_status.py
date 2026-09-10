#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote_ssh import connect, run

c = connect()
for cmd in [
    "grep -E 'epoch 30|BiLSTM|few-shot|pipeline done' /root/autodl-tmp/EdgeFS_NER/logs/nohup.out | tail -10",
    "ls -la /root/autodl-tmp/EdgeFS_NER/outputs/runs/*/metrics.json 2>/dev/null || echo NO_METRICS",
    "ps aux | grep 'train_full\\|train_fewshot' | grep -v grep | head -3",
]:
    _, out, err = run(c, f"bash -lc {cmd!r}", timeout=30)
    print("===", cmd.split()[0], "===")
    print(out or err)
c.close()
