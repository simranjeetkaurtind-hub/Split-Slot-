"""Job-opening dict helpers: validation, safe parsing, identity map (no deep copies)."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


REQUIRED_JO_KEYS = frozenset(
    {
        "id",
        "priority",
        "days_remaining",
        "total_duration",
        "active_demand",
        "project",
        "type",
        "slots_allocated",
        "tech_stack",
        "level",
        "initial_demand",
    }
)


def safe_int(val: Any, default: int = 0) -> int:
    if val is None:
        return default
    try:
        if pd.isna(val):
            return default
    except Exception:
        pass
    try:
        if hasattr(val, "item"):
            try:
                val = val.item()
            except Exception:
                pass
        if isinstance(val, float) and math.isnan(val):
            return default
        if isinstance(val, str) and val.strip() == "":
            return default
        return int(val)
    except (TypeError, ValueError, OverflowError):
        return default


def assert_jo_dict(jo: Any) -> dict:
    assert isinstance(jo, dict), "JO must be a dictionary"
    missing = REQUIRED_JO_KEYS - jo.keys()
    if missing:
        raise ValueError(f"JO missing keys: {sorted(missing)}")
    return jo


def jo_id(jo: dict) -> str:
    assert_jo_dict(jo)
    return str(jo["id"]).strip()


def normalize_jo_numbers(jo: dict) -> None:
    """
    initial_demand = pipeline size (fixed).
    active_demand = remaining slots to fill (decrements each assignment).
    If initial_demand omitted, infer from active_demand at ingest.
    """
    assert_jo_dict(jo)
    jo["days_remaining"] = safe_int(jo.get("days_remaining"), 0)
    jo["total_duration"] = safe_int(jo.get("total_duration"), 0)
    jo["slots_allocated"] = safe_int(jo.get("slots_allocated"), 0)
    ad = safe_int(jo.get("active_demand"), 0)
    init = safe_int(jo.get("initial_demand"), 0)
    if init <= 0:
        init = ad
    jo["initial_demand"] = max(0, init)
    jo["active_demand"] = min(max(0, ad), jo["initial_demand"]) if jo["initial_demand"] else max(0, ad)
    if str(jo.get("tech_stack", "")).strip() == "":
        jo["tech_stack"] = "*"
    if str(jo.get("level", "")).strip() == "":
        jo["level"] = "*"


def jos_list_to_map(job_openings: list[dict]) -> dict[str, dict]:
    """
    Build id → JO map using the SAME dict objects (no copy / no deepcopy).
    Mutations in allocation update these dicts and the original list references.
    """
    out: dict[str, dict] = {}
    for jo in job_openings:
        assert_jo_dict(jo)
        normalize_jo_numbers(jo)
        jid = jo_id(jo)
        if jid in out:
            raise ValueError(f"Duplicate JO id: {jid}")
        out[jid] = jo
    return out
