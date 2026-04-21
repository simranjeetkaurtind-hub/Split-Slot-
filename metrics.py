"""Summary metrics: demand coverage, lateral fairness, and Streamlit-friendly table styling."""

from __future__ import annotations

from typing import Any

import pandas as pd

from jo_utils import assert_jo_dict, safe_int
from models import JOType


def demand_fulfilled_pct(jo: dict) -> float:
    assert_jo_dict(jo)
    init = safe_int(jo.get("initial_demand"), 0)
    if init <= 0:
        return 100.0
    return 100.0 * safe_int(jo.get("slots_allocated"), 0) / init


def lateral_ratios(jos: dict[str, dict]) -> dict[str, float]:
    """ratio = slots_allocated / initial_demand for each lateral JO."""
    out: dict[str, float] = {}
    for jid, j in jos.items():
        if str(j.get("type", "")).strip().upper() != JOType.LATERAL.value:
            continue
        init = safe_int(j.get("initial_demand"), 0)
        if init <= 0:
            out[jid] = 0.0
        else:
            out[jid] = safe_int(j.get("slots_allocated"), 0) / init
    return out


def fairness_lateral_range(jos: dict[str, dict]) -> float:
    """
    fairness = 1 - (max(ratios) - min(ratios)) over lateral JOs.
    ratio = slots_allocated / initial_demand.
    """
    ratios = list(lateral_ratios(jos).values())
    if len(ratios) < 2:
        return 1.0
    spread = max(ratios) - min(ratios)
    return max(0.0, 1.0 - spread)


def fairness_lateral(jos: dict[str, dict]) -> float:
    return fairness_lateral_range(jos)


def _role_row_styles(row: pd.Series) -> list[str]:
    """Dark theme row backgrounds + light text; stronger tint for Dominant / Competing."""
    r = row.get("role", "")
    if r == "Dominant":
        return ["background-color: #1a3d2e; color: #e8fff1; font-weight: 600"] * len(row)
    if r == "Competing":
        return ["background-color: #3d3518; color: #fff6e0; font-weight: 500"] * len(row)
    return ["background-color: #2a2f36; color: #eceff4"] * len(row)


def build_metrics_styled_table(df: pd.DataFrame) -> Any:
    """
    Dark table body, light text, subtle bars on numeric columns, header row darker.
    Intended for st.dataframe(styler).
    """
    styler = df.style.apply(_role_row_styles, axis=1)
    styler = styler.set_table_styles(
        [
            {
                "selector": "th",
                "props": [
                    ("background-color", "#0d0f12"),
                    ("color", "#f8fafc"),
                    ("font-weight", "700"),
                    ("border", "1px solid #2a3140"),
                    ("padding", "0.45rem 0.55rem"),
                ],
            },
            {
                "selector": "td",
                "props": [("border", "1px solid #343b47"), ("padding", "0.4rem 0.5rem")],
            },
        ]
    )
    demand_col = "📉 demand fulfilled %"
    sat_col = "🔥 saturation %"
    if demand_col in df.columns:
        try:
            styler = styler.bar(
                subset=[demand_col],
                color="#2f8f56",
                vmin=0,
                vmax=100,
                width=75,
                align="zero",
            )
        except TypeError:
            styler = styler.bar(subset=[demand_col], color="#2f8f56", vmin=0, vmax=100)
    if sat_col in df.columns:
        try:
            styler = styler.bar(
                subset=[sat_col],
                color="#c2783d",
                vmin=0,
                vmax=100,
                width=75,
                align="zero",
            )
        except TypeError:
            styler = styler.bar(subset=[sat_col], color="#c2783d", vmin=0, vmax=100)
    return styler
