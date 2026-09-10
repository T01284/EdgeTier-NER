#!/usr/bin/env python3
"""Sync repo and start A–E GPU batch on remote (non-blocking, fast upload)."""

from __future__ import annotations

import sys
from pathlib import Path

# Delegate to the fast path (full sync_project + pip reinstall is too slow).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from start_abcde_fast import main

if __name__ == "__main__":
    raise SystemExit(main())
