#!/usr/bin/env python3
"""Download and extract GloVe 6B 100d vectors."""

from __future__ import annotations

import argparse

from edgefs.data.glove import ensure_glove


def main() -> None:
    parser = argparse.ArgumentParser(description="Download GloVe 6B vectors")
    parser.add_argument("--dir", default="data/embeddings")
    parser.add_argument("--dim", type=int, default=100, choices=[50, 100, 200, 300])
    args = parser.parse_args()
    path = ensure_glove(args.dir, dim=args.dim)
    print(f"Ready: {path}")


if __name__ == "__main__":
    main()
