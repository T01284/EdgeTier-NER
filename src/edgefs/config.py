from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path

    def get(self, key: str, default: Any = None) -> Any:
        if isinstance(self.raw, dict):
            return self.raw.get(key, default)
        return default

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    @property
    def name(self) -> str:
        return str(self.raw.get("name", self.path.stem))


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config(raw=raw, path=path)


def merge_configs(base: Config, override: Config) -> Config:
    merged = _deep_merge(dict(base.raw), dict(override.raw))
    return Config(raw=merged, path=override.path)


def _deep_merge(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for key, value in b.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
