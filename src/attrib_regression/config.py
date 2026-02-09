from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML config as a dictionary."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dict_to_namespace(d: Any) -> Any:
    """Recursively convert dicts to SimpleNamespace for dotted attribute access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_dict_to_namespace(item) for item in d]
    return d


def load_rba_config(path: str | Path) -> SimpleNamespace:
    """Load YAML config and return a nested SimpleNamespace (cfg.data.path, etc.)."""
    raw = load_config(path)
    return _dict_to_namespace(raw)
