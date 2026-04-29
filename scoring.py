"""Scoring: dynamic per slot. demand_score = active_demand/initial_demand; saturation = slots_allocated/initial_demand."""

from __future__ import annotations

from typing import Any

from config import get_weight
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


def compute_urgency(
    jo: dict,
    days_remaining: int,
    total_duration: int,
    demand_score_value: float,
) -> float:
    if total_duration <= 0:
        base_urgency = 1.0
    else:
        ratio = max(0.0, min(1.0, days_remaining / total_duration))
        base_urgency = 1.0 - ratio
    jo_type = str(jo.get("type", jo.get("jo_type", ""))).strip().upper()
    if jo_type == JOType.ELTP.value:
        beta = 0.35
        adjusted = min(1.0, base_urgency + beta * demand_score_value)
        return adjusted
    return base_urgency


def demand_score(jo: dict) -> float:
    assert_jo_dict(jo)
    init = max(safe_int(jo.get("initial_demand"), 0), 1)
    curr = max(0, safe_int(jo.get("active_demand"), 0))
    return min(1.0, curr / init)


def base_score(jo: dict) -> float:
    assert_jo_dict(jo)
    ps = priority_score_from_priority(jo["priority"])
    ds = demand_score(jo)
    us = compute_urgency(
        jo,
        safe_int(jo.get("days_remaining"), 0),
        safe_int(jo.get("total_duration"), 0),
        ds,
    )
    return (
        get_weight("priority_weight") * ps
        + get_weight("urgency_weight") * us
        + get_weight("demand_weight") * ds
    )


def affinity_score_slot(jo: dict, slot_project: str) -> float:
    assert_jo_dict(jo)
    return 1.0 if str(jo["project"]).strip() == str(slot_project).strip() else 0.3


def score_breakdown(jo: dict, slot_project: str) -> dict[str, float]:
    assert_jo_dict(jo)
    ps = priority_score_from_priority(jo["priority"])
    ds = demand_score(jo)
    us = compute_urgency(
        jo,
        safe_int(jo.get("days_remaining"), 0),
        safe_int(jo.get("total_duration"), 0),
        ds,
    )
    base = (
        get_weight("priority_weight") * ps
        + get_weight("urgency_weight") * us
        + get_weight("demand_weight") * ds
    )
    aff = affinity_score_slot(jo, slot_project)
    final = base + get_weight("affinity_weight") * aff
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

    P0: greedy <45%; then 80:20 / 60:40 / 40:60.
    P1: greedy <35%; then 70:30 / 60:40 / 40:60.
    P2 lateral: greedy <25%; then 30:70 / 40:60 / 20:80.
    ELTP: greedy <30%; then 80:20 / 60:40 / 80:20.
    """
    assert_jo_dict(jo)
    ratio = saturation_ratio(jo)
    pl = priority_level(jo["priority"])
    jo_type = str(jo.get("type", jo.get("jo_type", ""))).strip().upper()

    if pl == 0:
        if ratio < 0.45:
            return ("BELOW LOW", "<45%", "GREEDY", 1.0)
        if ratio < 0.65:
            return ("LOW", "45–65%", "80:20", 0.80)
        if ratio < 0.80:
            return ("MID", "65–80%", "60:40", 0.60)
        return ("HIGH", "≥80%", "40:60", 0.40)

    if pl == 1:
        if ratio < 0.35:
            return ("BELOW LOW", "<35%", "GREEDY", 1.0)
        if ratio < 0.55:
            return ("LOW", "35–55%", "70:30", 0.70)
        if ratio < 0.70:
            return ("MID", "55–70%", "60:40", 0.60)
        return ("HIGH", "≥70%", "40:60", 0.40)

    if jo_type == JOType.ELTP.value:
        if ratio < 0.30:
            return ("BELOW LOW", "<30%", "GREEDY", 1.0)
        if ratio < 0.55:
            return ("LOW", "30–55%", "80:20", 0.80)
        if ratio < 0.70:
            return ("MID", "55–70%", "60:40", 0.60)
        return ("TOP", "≥70%", "80:20", 0.80)

    if ratio < 0.25:
        return ("BELOW LOW", "<25%", "GREEDY", 1.0)
    if ratio < 0.65:
        return ("LOW", "25–65%", "30:70", 0.30)
    if ratio < 0.80:
        return ("MID", "65–80%", "40:60", 0.40)
    return ("TOP", "≥80%", "20:80", 0.20)


def get_split_ratio(jo: dict) -> tuple[float, float]:
    """
    Progressive saturation band: (dominant_share, competing_share) from priority + saturation_ratio(jo).
    ELTP uses its own ladder; lateral P2/P3 use the lateral ladder. Shares sum to 1.0.
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
