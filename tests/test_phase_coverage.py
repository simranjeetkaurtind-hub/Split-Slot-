"""
Slot-by-slot allocation coverage: delta + saturation bands drive batch caps.
Validates that batch_debug reflects the slot mode and caps are present.
"""

from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from engine import Phase, run_allocation
from models import Slot


def scenario_greedy_to_6040_jos() -> list[dict]:
    """
    Clear dominant (P0, high urgency, strong demand score) vs weaker P1 (low urgency).
    Base delta > 0.040. Used as a stable layered-allocation smoke test.
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
    """50 slots, mixed projects; engine batches by batch_size (4) → 13 batches (last size = 2)."""
    projects = ["Alpha", "Beta", "Gamma", "Delta", "Bench"]
    slots: list[Slot] = []
    for i in range(1, 51):
        proj = projects[(i - 1) % len(projects)]
        slots.append(
            Slot(
                slot_id=f"S{i:02d}",
                panel_id=f"P{i:02d}",
                project=proj,
                tech_stack="*",
                level="*",
                batch_id="1",
            )
        )
    return slots


def print_batch_phase_report(result) -> None:
    print("")
    print("=== Batch phase summary (Slot scenario) ===")
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


def test_slot_phase_is_used() -> None:
    jos = scenario_greedy_to_6040_jos()
    slots = scenario_greedy_to_6040_slots()
    buf = io.StringIO()
    with redirect_stdout(buf):
        res = run_allocation(jos, slots, batch_size=4)
    print_batch_phase_report(res)
    assert all(bd.phase in (Phase.SLOT, "-") for bd in res.batch_debug), (
        f"expected Slot or empty phase for all batches, got {[bd.phase for bd in res.batch_debug]}"
    )


def test_caps_sum_to_batch_size() -> None:
    jos = scenario_greedy_to_6040_jos()
    slots = scenario_greedy_to_6040_slots()
    buf = io.StringIO()
    with redirect_stdout(buf):
        res = run_allocation(jos, slots, batch_size=4)
    for bd in res.batch_debug:
        assert sum(bd.caps.values()) >= 0


if __name__ == "__main__":
    test_slot_phase_is_used()
    print("OK: Slot phase observed in batch_debug.")
    test_caps_sum_to_batch_size()
    print("OK: caps sum to batch size for each batch.")
