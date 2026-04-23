"""Scoring: dynamic per slot. demand_score = active_demand/initial_demand; saturation = slots_allocated/initial_demand."""

from __future__ import annotations

from typing import Any

from jo_utils import assert_jo_dict, safe_int
from models import JOType


def priority_level(priority: Any) -> int:
    if hasattr(priority, "item"):
        try:
            priority = priority.item()
        except Exception:
            pass
    if isinstance(priority, bool):
        priority = str(priority)
    if isinstance(priority, (int, float)):
        return max(0, min(3, int(priority)))
    m = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    return m.get(str(priority).strip().upper(), 3)


def priority_score_from_priority(priority: Any) -> float:
    pl = priority_level(priority)
    return 1.0 / (pl + 1)


def urgency_score(days_remaining: int, total_duration: int) -> float:
    if total_duration <= 0:
        return 1.0
    r = max(0.0, min(1.0, days_remaining / total_duration))
    return 1.0 - r


def demand_score(jo: dict) -> float:
    assert_jo_dict(jo)
    init = max(safe_int(jo.get("initial_demand"), 0), 1)
    curr = max(0, safe_int(jo.get("active_demand"), 0))
    return min(1.0, curr / init)


def base_score(jo: dict) -> float:
    assert_jo_dict(jo)
    ps = priority_score_from_priority(jo["priority"])
    us = urgency_score(safe_int(jo.get("days_remaining"), 0), safe_int(jo.get("total_duration"), 0))
    ds = demand_score(jo)
    return 0.40 * ps + 0.25 * us + 0.25 * ds


def affinity_score_slot(jo: dict, slot_project: str) -> float:
    assert_jo_dict(jo)
    return 1.0 if str(jo["project"]).strip() == str(slot_project).strip() else 0.3


def score_breakdown(jo: dict, slot_project: str) -> dict[str, float]:
    assert_jo_dict(jo)
    ps = priority_score_from_priority(jo["priority"])
    us = urgency_score(safe_int(jo.get("days_remaining"), 0), safe_int(jo.get("total_duration"), 0))
    ds = demand_score(jo)
    base = 0.40 * ps + 0.25 * us + 0.25 * ds
    aff = affinity_score_slot(jo, slot_project)
    final = base + 0.10 * aff
    sl = float(safe_int(jo.get("slots_allocated"), 0))
    ad = float(safe_int(jo.get("active_demand"), 0))
    return {
        "priority_score": ps,
        "urgency_score": us,
        "demand_score": ds,
        "base_score": base,
        "affinity_score": aff,
        "final_score": final,
        "slots_allocated": sl,
        "active_demand": ad,
        "saturation_ratio": saturation_ratio(jo),
    }


def final_score(jo: dict, slot_project: str) -> float:
    return score_breakdown(jo, slot_project)["final_score"]


def saturation_threshold_for_jo(jo: dict) -> float:
    """
    Saturation caps (slots_allocated / initial_demand must not exceed this when assigning):
    - ELTP (any priority) → 0.50
    - LATERAL P0 / P1 → 0.75
    - LATERAL P2 / P3 → 0.50 (P2 saturates faster than P0/P1)

    """
    assert_jo_dict(jo)
    if str(jo.get("type", "")).strip().upper() == JOType.ELTP.value:
        return 0.50
    pl = priority_level(jo["priority"])
    if pl in (0, 1):
        return 0.75
    return 0.50


def saturation_ratio(jo: dict) -> float:
    assert_jo_dict(jo)
    init = safe_int(jo.get("initial_demand"), 0)
    if init <= 0:
        return 1.0
    return safe_int(jo.get("slots_allocated"), 0) / init


def get_saturation_band(jo: dict) -> tuple[str, str, str, float]:
    """
    Returns (band_name, band_label, split_rule, dominant_share).
    Band labels:
    - P0/P1: BELOW LOW (<45%), LOW (≥45%), MID (≥65%), HIGH (≥80%)
    - P2/P3/ELTP: BELOW LOW (<25%), LOW (≥25%), HIGH (≥50%)
    """
    assert_jo_dict(jo)
    ratio = saturation_ratio(jo)
    pl = priority_level(jo["priority"])
    if pl in (0, 1):
        if ratio < 0.45:
            return ("BELOW LOW", "BELOW LOW", "GREEDY (100%)", 1.0)
        if ratio < 0.65:
            return ("LOW", "LOW (≥ 45%)", "80:20", 0.80)
        if ratio < 0.80:
            return ("MID", "MID (≥ 65%)", "60:40", 0.60)
        return ("HIGH", "HIGH (≥ 80%)", "30:70", 0.30)
    if ratio < 0.25:
        return ("BELOW LOW", "BELOW LOW", "GREEDY (100%)", 1.0)
    if ratio < 0.50:
        return ("LOW", "LOW (≥ 25%)", "80:20", 0.80)
    return ("HIGH", "HIGH (≥ 50%)", "50:50", 0.50)


def get_split_ratio(jo: dict) -> tuple[float, float]:
    """
    Progressive saturation band: (dominant_share, competing_share) from priority + saturation_ratio(jo).
    ELTP follows the P2/P3 bands. Shares sum to 1.0. Used for batch cap splits.
    """
    _, _, _, dominant_share = get_saturation_band(jo)
    return (dominant_share, 1.0 - dominant_share)


# Backward-compatible name
saturation_band_split = get_split_ratio


def saturation_pct(jo: dict) -> float:
    return 100.0 * saturation_ratio(jo)


def is_saturated(jo: dict) -> bool:
    assert_jo_dict(jo)
    init = safe_int(jo.get("initial_demand"), 0)
    if init <= 0:
        return True
    return saturation_ratio(jo) >= saturation_threshold_for_jo(jo) - 1e-9


def can_assign_without_exceeding_saturation(jo: dict) -> bool:
    assert_jo_dict(jo)
    if safe_int(jo.get("active_demand"), 0) <= 0:
        return False
    init = safe_int(jo.get("initial_demand"), 0)
    if init <= 0:
        return False
    next_slots = safe_int(jo.get("slots_allocated"), 0) + 1
    next_ratio = next_slots / init
    return next_ratio <= saturation_threshold_for_jo(jo) + 1e-9
