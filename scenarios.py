from __future__ import annotations

from typing import Any

from models import Slot


def _make_slots(total: int, projects: list[str]) -> list[Slot]:
    slots: list[Slot] = []
    for i in range(1, total + 1):
        proj = projects[(i - 1) % len(projects)]
        slots.append(
            Slot(
                slot_id=f"S{i:03d}",
                panel_id=f"P{i:03d}",
                project=proj,
                tech_stack="*",
                level="*",
                batch_id="1",
            )
        )
    return slots


def scenario_definitions() -> list[dict[str, Any]]:
    return [
        {
            "id": "same_priority_diff_demand",
            "name": "Same Priority – Demand Driven",
            "jos": [
                _jo("JO-A", 1, 30, 30, "Alpha"),
                _jo("JO-B", 1, 30, 15, "Beta"),
                _jo("JO-C", 1, 30, 5, "Gamma"),
            ],
            "slots": _make_slots(30, ["Alpha", "Beta", "Gamma"]),
            "batch_sizes": [5, 5, 5, 5, 5, 5],
        },
        {
            "id": "same_priority_equal_demand",
            "name": "Same Priority – Equal Demand (Delta)",
            "jos": [
                _jo("JO-A", 1, 30, 20, "Alpha"),
                _jo("JO-B", 1, 30, 20, "Beta"),
                _jo("JO-C", 1, 30, 20, "Gamma"),
            ],
            "slots": _make_slots(30, ["Alpha", "Beta", "Gamma"]),
            "batch_sizes": [5, 5, 5, 5, 5, 5],
        },
        {
            "id": "scarce_slots",
            "name": "Scarce Slots – High Competition",
            "jos": [
                _jo("JO-A", 0, 40, 40, "Alpha"),
                _jo("JO-B", 1, 30, 30, "Beta"),
                _jo("JO-C", 1, 20, 20, "Gamma"),
                _jo("JO-D", 2, 10, 10, "Delta"),
                _jo("JO-ELTP", 2, 100, 100, "Bench", jo_type="ELTP"),
            ],
            "slots": _make_slots(20, ["Alpha", "Beta", "Gamma", "Delta"]),
            "batch_sizes": [5, 5, 5, 5],
        },
        {
            "id": "abundant_slots",
            "name": "Abundant Slots – ELTP Filler",
            "jos": [
                _jo("JO-A", 0, 20, 20, "Alpha"),
                _jo("JO-B", 1, 15, 15, "Beta"),
                _jo("JO-C", 2, 10, 10, "Gamma"),
                _jo("JO-ELTP", 2, 100, 100, "Bench", jo_type="ELTP"),
            ],
            "slots": _make_slots(100, ["Alpha", "Beta", "Gamma"]),
            "batch_sizes": [5] * 20,
        },
        {
            "id": "mixed_priority_demand",
            "name": "Mixed Priority + Demand",
            "jos": [
                _jo("JO-A", 0, 20, 20, "Alpha"),
                _jo("JO-B", 0, 15, 15, "Beta"),
                _jo("JO-C", 1, 15, 15, "Gamma"),
                _jo("JO-D", 1, 10, 10, "Delta"),
                _jo("JO-E", 2, 10, 10, "Alpha"),
                _jo("JO-ELTP", 2, 50, 50, "Bench", jo_type="ELTP"),
            ],
            "slots": _make_slots(40, ["Alpha", "Beta", "Gamma", "Delta"]),
            "batch_sizes": [5] * 8,
        },
        {
            "id": "saturation_band_test",
            "name": "Saturation Band Test (45/65/80, 25/50)",
            "jos": [
                _jo("JO-A", 0, 20, 11, "Alpha"),
                _jo("JO-B", 0, 20, 7, "Beta"),
                _jo("JO-C", 0, 20, 4, "Gamma"),
                _jo("JO-D", 2, 20, 15, "Delta"),
                _jo("JO-E", 2, 20, 10, "Alpha"),
            ],
            "slots": _make_slots(20, ["Alpha", "Beta", "Gamma", "Delta"]),
            "batch_sizes": [5, 5, 5, 5],
        },
        {
            "id": "p2_vs_p1_demand",
            "name": "High Demand P2 vs Low Demand P1",
            "jos": [
                _jo("JO-A", 1, 10, 10, "Alpha"),
                _jo("JO-B", 2, 50, 50, "Beta"),
                _jo("JO-ELTP", 2, 100, 100, "Bench", jo_type="ELTP"),
            ],
            "slots": _make_slots(20, ["Alpha", "Beta"]),
            "batch_sizes": [5, 5, 5, 5],
        },
        {
            "id": "all_p2_competition",
            "name": "All P2 Competition",
            "jos": [
                _jo("JO-A", 2, 20, 20, "Alpha"),
                _jo("JO-B", 2, 20, 20, "Beta"),
                _jo("JO-C", 2, 10, 10, "Gamma"),
                _jo("JO-ELTP", 2, 40, 40, "Bench", jo_type="ELTP"),
            ],
            "slots": _make_slots(40, ["Alpha", "Beta", "Gamma"]),
            "batch_sizes": [5] * 8,
        },
        {
            "id": "small_batch_edge",
            "name": "Small Batch Stability",
            "jos": [
                _jo("JO-A", 0, 10, 10, "Alpha"),
                _jo("JO-B", 1, 10, 10, "Beta"),
                _jo("JO-C", 2, 10, 10, "Gamma"),
            ],
            "slots": _make_slots(15, ["Alpha", "Beta", "Gamma"]),
            "batch_sizes": [1] * 15,
        },
        {
            "id": "single_dominant",
            "name": "Single Dominant JO (Cap Test)",
            "jos": [
                _jo("JO-A", 0, 50, 50, "Alpha"),
                _jo("JO-B", 1, 5, 5, "Beta"),
                _jo("JO-C", 2, 5, 5, "Gamma"),
            ],
            "slots": _make_slots(20, ["Alpha", "Beta", "Gamma"]),
            "batch_sizes": [5, 5, 5, 5],
        },
    ]


def _jo(
    jid: str, priority: int, days_remaining: int, demand: int, project: str, *, jo_type: str = "LATERAL"
) -> dict[str, Any]:
    total_duration = _default_total_duration(priority, jo_type)
    days_remaining = max(1, total_duration - 3)
    return {
        "id": jid,
        "priority": priority,
        "days_remaining": days_remaining,
        "total_duration": total_duration,
        "initial_demand": demand,
        "active_demand": demand,
        "project": project,
        "type": jo_type,
        "tech_stack": "*",
        "level": "*",
        "slots_allocated": 0,
    }


def _default_total_duration(priority: int, jo_type: str) -> int:
    if str(jo_type).strip().upper() == "ELTP":
        return 140
    if priority == 0:
        return 15
    if priority == 1:
        return 45
    return 90
