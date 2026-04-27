from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DEFAULT_CONFIG: dict[str, Any] = {
    "priority_weight": 0.35,
    "demand_weight": 0.30,
    "urgency_weight": 0.25,
    "affinity_weight": 0.10,
    "delta_threshold": 0.025,
}

_CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def load_config() -> dict[str, Any]:
    if not _CONFIG_PATH.exists():
        return dict(_DEFAULT_CONFIG)
    try:
        raw = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return dict(_DEFAULT_CONFIG)
    merged = dict(_DEFAULT_CONFIG)
    merged.update({k: raw.get(k) for k in _DEFAULT_CONFIG.keys() if k in raw})
    return merged


def save_config(cfg: dict[str, Any]) -> None:
    data = dict(_DEFAULT_CONFIG)
    data.update(cfg or {})
    _CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_config() -> dict[str, Any]:
    return load_config()


def get_delta_threshold() -> float:
    cfg = load_config()
    try:
        return float(cfg.get("delta_threshold", _DEFAULT_CONFIG["delta_threshold"]))
    except (TypeError, ValueError):
        return float(_DEFAULT_CONFIG["delta_threshold"])


def get_weight(name: str) -> float:
    cfg = load_config()
    try:
        return float(cfg.get(name, _DEFAULT_CONFIG[name]))
    except (TypeError, ValueError, KeyError):
        return float(_DEFAULT_CONFIG.get(name, 0.0))
