"""
Phase coverage scenario: delta > DELTA_EQUAL so Equal (50–50) never applies;
dominant saturates after enough slots → later batches use 60–40.

Does not change engine rules — only supplies JOs/slots and validates batch_debug.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from engine import (
    Phase,
    _apply_next_tier_caps,
    _caps_case_b_dominant_saturated,
    _lateral_sort_key,
    _top_next_pool_sizes,
    run_allocation,
)
from scoring import base_score
from models import Slot


def scenario_greedy_to_6040_jos() -> list[dict]:
    """
    Clear dominant (P0, high urgency, strong demand score) vs weaker P1 (low urgency).
    Base delta > 0.040. initial_demand=8 → 75% sat cap = 6 slots on dominant; two greedy
    batches of 4 fill 8 slot attempts but dominant stops at 6; batch 3 sees saturated dominant → 60–40.
    """
    return [
        {
            "id": "JO-P0-DOM",
            "priority": 0,
            "days_remaining": 2,
            "total_duration": 30,
            "initial_demand": 8,
            "active_demand": 8,
            "project": "PanelProj",
            "type": "LATERAL",
            "tech_stack": "*",
            "level": "*",
            "slots_allocated": 0,
        },
        {
            "id": "JO-P1-CMP",
            "priority": 1,
            "days_remaining": 28,
            "total_duration": 30,
            "initial_demand": 8,
            "active_demand": 8,
            "project": "PanelProj",
            "type": "LATERAL",
            "tech_stack": "*",
            "level": "*",
            "slots_allocated": 0,
        },
        {
            "id": "JO-F-ELTP",
            "priority": 2,
            "days_remaining": 20,
            "total_duration": 60,
            "initial_demand": 20,
            "active_demand": 20,
            "project": "Bench",
            "type": "ELTP",
            "tech_stack": "*",
            "level": "*",
            "slots_allocated": 0,
        },
    ]


def scenario_greedy_to_6040_slots() -> list[Slot]:
    """12 slots, all match lateral tech/project; engine batches by batch_size (4) → 3 batches."""
    return [
        Slot(
            slot_id=f"S{i:02d}",
            panel_id=f"P{i:02d}",
            project="PanelProj",
            tech_stack="*",
            level="*",
            batch_id="1",
        )
        for i in range(1, 13)
    ]


def print_batch_phase_report(result) -> None:
    print("")
    print("=== Batch phase summary (Greedy -> 60-40 scenario) ===")
    for bd in result.batch_debug:
        dom_id = bd.dominant_id or "-"
        sat_dom = float("nan")
        if dom_id != "-" and dom_id in bd.saturation_pct:
            sat_dom = bd.saturation_pct[dom_id]
        print(f"Batch ID:    {bd.batch_index}")
        print(f"Delta:       {bd.delta_value}")
        print(f"Dominant JO: {dom_id}")
        print(f"Saturation:  {dom_id} @ batch start = {sat_dom:.2f}% (lateral pool snapshot)")
        print(f"Phase:       {bd.phase}")
        print("---")
    print("")


def test_equal_delta_both_top_saturated_forces_60_40() -> None:
    """When |Δ| ≤ 0.04 but top1 and top2 are both at saturation, phase must be 60–40 (not 50–50)."""
    jos = [
        {
            "id": "JO-P0-A",
            "priority": "P0",
            "days_remaining": 5,
            "total_duration": 30,
            "initial_demand": 8,
            "active_demand": 2,
            "project": "P",
            "type": "LATERAL",
            "tech_stack": "*",
            "level": "*",
            "slots_allocated": 6,
        },
        {
            "id": "JO-P0-B",
            "priority": "P0",
            "days_remaining": 5,
            "total_duration": 30,
            "initial_demand": 8,
            "active_demand": 2,
            "project": "P",
            "type": "LATERAL",
            "tech_stack": "*",
            "level": "*",
            "slots_allocated": 6,
        },
        {
            "id": "JO-P2-LOW",
            "priority": "P2",
            "days_remaining": 20,
            "total_duration": 30,
            "initial_demand": 10,
            "active_demand": 10,
            "project": "P",
            "type": "LATERAL",
            "tech_stack": "*",
            "level": "*",
            "slots_allocated": 0,
        },
        {
            "id": "ELTP1",
            "priority": "P2",
            "days_remaining": 20,
            "total_duration": 60,
            "initial_demand": 20,
            "active_demand": 20,
            "project": "P",
            "type": "ELTP",
            "tech_stack": "*",
            "level": "*",
            "slots_allocated": 0,
        },
    ]
    slots = [
        Slot(
            slot_id=f"S{i:02d}",
            panel_id="P1",
            project="P",
            tech_stack="*",
            level="*",
            batch_id="1",
        )
        for i in range(1, 5)
    ]
    buf = io.StringIO()
    with redirect_stdout(buf):
        res = run_allocation(jos, slots, batch_size=4)
    assert res.batch_debug[0].phase == Phase.SPLIT_60_40, (
        f"expected 60–40 when both top JOs saturated with small Δ, got {res.batch_debug[0].phase}"
    )


def _lateral_jo(
    jid: str,
    priority: str,
    days_remaining: int,
    active_demand: int,
    slots_allocated: int,
) -> dict:
    return {
        "id": jid,
        "priority": priority,
        "days_remaining": days_remaining,
        "total_duration": 30,
        "initial_demand": 10,
        "active_demand": active_demand,
        "project": "P",
        "type": "LATERAL",
        "tech_stack": "*",
        "level": "*",
        "slots_allocated": slots_allocated,
    }


def test_next_tier_caps_greedy_when_top_of_remainder_dominates() -> None:
    """Three+ JOs below top two: next_pool goes to ranked[2] only if |score[2]-score[3]| > 0.04."""
    ranked = sorted(
        [
            _lateral_jo("TOPA", "P0", 1, 10, 0),
            _lateral_jo("TOPB", "P0", 5, 10, 0),
            _lateral_jo("C", "P1", 10, 10, 0),
            _lateral_jo("D", "P2", 28, 10, 0),
            _lateral_jo("E", "P2", 28, 10, 0),
        ],
        key=_lateral_sort_key,
    )
    assert abs(base_score(ranked[2]) - base_score(ranked[3])) > 0.04
    caps = {"TOPA": 2, "TOPB": 1}
    _apply_next_tier_caps(caps, ranked, 1)
    assert caps["C"] == 1
    assert caps.get("D", 0) == 0
    assert caps.get("E", 0) == 0


def test_next_tier_caps_equal_split_between_top_two_of_remainder() -> None:
    """When the first two in ranked[2:] tie on base score, next_pool splits ~50–50 between them."""
    ranked = sorted(
        [
            _lateral_jo("TOPA", "P0", 1, 10, 0),
            _lateral_jo("TOPB", "P0", 5, 10, 0),
            _lateral_jo("D", "P2", 28, 10, 0),
            _lateral_jo("E", "P2", 28, 10, 0),
        ],
        key=_lateral_sort_key,
    )
    assert abs(base_score(ranked[2]) - base_score(ranked[3])) <= 0.04
    caps = {"TOPA": 2, "TOPB": 1}
    _apply_next_tier_caps(caps, ranked, 4)
    assert caps["D"] == 2
    assert caps["E"] == 2


def test_case_b_top_pool_split_includes_second_ranked_when_three_plus_laterals() -> None:
    """Case B must not assign the entire top pool to ranked[0] and 0 to ranked[1] when P2 exists."""
    batch_n = 4
    top_pool, next_pool = _top_next_pool_sizes(batch_n)
    ranked = [
        {
            "id": "JO-P0",
            "priority": "P0",
            "days_remaining": 1,
            "total_duration": 30,
            "initial_demand": 10,
            "active_demand": 10,
            "project": "P",
            "type": "LATERAL",
            "tech_stack": "*",
            "level": "*",
            "slots_allocated": 6,
        },
        {
            "id": "JO-P1",
            "priority": "P1",
            "days_remaining": 20,
            "total_duration": 30,
            "initial_demand": 10,
            "active_demand": 10,
            "project": "P",
            "type": "LATERAL",
            "tech_stack": "*",
            "level": "*",
            "slots_allocated": 5,
        },
        {
            "id": "JO-P2",
            "priority": "P2",
            "days_remaining": 20,
            "total_duration": 30,
            "initial_demand": 10,
            "active_demand": 10,
            "project": "P",
            "type": "LATERAL",
            "tech_stack": "*",
            "level": "*",
            "slots_allocated": 0,
        },
    ]
    caps = _caps_case_b_dominant_saturated(ranked, batch_n)
    assert caps["JO-P1"] > 0, "second-ranked by score must have top-pool cap (not starved vs next tier)"
    assert caps["JO-P0"] + caps["JO-P1"] == top_pool
    assert caps.get("JO-P2", 0) == next_pool


def test_60_40_phase_is_exercised() -> None:
    jos = scenario_greedy_to_6040_jos()
    slots = scenario_greedy_to_6040_slots()
    buf = io.StringIO()
    with redirect_stdout(buf):
        res = run_allocation(jos, slots, batch_size=4)
    print_batch_phase_report(res)
    phases = [bd.phase for bd in res.batch_debug]
    assert Phase.GREEDY in phases, f"expected Greedy phase in sequence, got {phases}"
    assert any(bd.phase == Phase.SPLIT_60_40 for bd in res.batch_debug), (
        "60-40 phase not triggered"
    )


if __name__ == "__main__":
    test_equal_delta_both_top_saturated_forces_60_40()
    print("OK: equal delta + both top saturated -> 60-40.")
    test_case_b_top_pool_split_includes_second_ranked_when_three_plus_laterals()
    print("OK: Case B splits top pool across JO-1 and JO-2 when three+ laterals.")
    test_next_tier_caps_greedy_when_top_of_remainder_dominates()
    print("OK: next tier greedy when highest remainder JO clearly leads.")
    test_next_tier_caps_equal_split_between_top_two_of_remainder()
    print("OK: next tier 50-50 when top two of remainder tie on score.")
    test_60_40_phase_is_exercised()
    print("OK: Greedy and 60-40 both observed in batch_debug.")
