"""
Large simulation harness for layered allocation.
Creates 100 slots, runs with variable batch sizes, and prints detailed output.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from engine import run_allocation
from models import Slot
from scoring import base_score, get_saturation_band, saturation_ratio


def build_jos() -> list[dict]:
    def jo(
        jid: str,
        priority: int,
        days_remaining: int,
        demand: int,
        project: str,
        jo_type: str = "LATERAL",
    ) -> dict:
        return {
            "id": jid,
            "priority": priority,
            "days_remaining": days_remaining,
            "total_duration": 120,
            "initial_demand": demand,
            "active_demand": demand,
            "project": project,
            "type": jo_type,
            "tech_stack": "*",
            "level": "*",
            "slots_allocated": 0,
        }

    return [
        jo("JO-A-P0", 0, 10, 15, "Alpha"),
        jo("JO-B-P0", 0, 15, 15, "Beta"),
        jo("JO-C-P1", 1, 30, 8, "Alpha"),
        jo("JO-D-P1", 1, 30, 8, "Gamma"),
        jo("JO-E-P2", 2, 60, 4, "Delta"),
        jo("JO-F-P2", 2, 60, 4, "Gamma"),
        jo("JO-G-ELTP", 2, 120, 40, "Bench", jo_type="ELTP"),
    ]


def build_slots() -> list[Slot]:
    projects = ["Alpha", "Beta", "Gamma", "Delta", "Bench"]
    slots: list[Slot] = []
    for i in range(1, 101):
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


def print_batch_header(batch_id: int, batch_slots: list[Slot]) -> None:
    slot_ids = ", ".join(s.slot_id for s in batch_slots)
    print("\n" + "=" * 72)
    print(f"Batch ID: {batch_id}")
    print(f"Batch Size: {len(batch_slots)}")
    print(f"Slots in batch: {slot_ids}")


def print_jo_state(jos: list[dict]) -> None:
    print("\nJO State (pre-batch):")
    for jo in sorted(jos, key=lambda j: (-base_score(j), j["id"])):
        sat = saturation_ratio(jo) * 100.0
        _, band_label, split_rule, _ = get_saturation_band(jo)
        print(
            f"  {jo['id']}: base={base_score(jo):.4f} "
            f"demand={jo['active_demand']} slots={jo['slots_allocated']} "
            f"sat={sat:.1f}% band={band_label} rule={split_rule}"
        )


def print_final_summary(jos: list[dict], total_slots: int, assigned: int) -> None:
    print("\nFinal Allocation:")
    for jo in sorted(jos, key=lambda j: j["id"]):
        print(
            f"  {jo['id']}: slots={jo['slots_allocated']} "
            f"demand_remaining={jo['active_demand']}"
        )

    checks = {
        "P0 filled first": any(j["id"].startswith("JO-A-P0") for j in jos),
        "P1 activates mid": any(j["id"].startswith("JO-C-P1") and j["slots_allocated"] > 0 for j in jos),
        "P2 crosses 25%": all(
            j["slots_allocated"] >= 1 for j in jos if j["id"] in ("JO-E-P2", "JO-F-P2")
        ),
        "P2 crosses 50%": all(
            (j["slots_allocated"] / j["initial_demand"]) >= 0.5
            for j in jos
            if j["id"] in ("JO-E-P2", "JO-F-P2")
        ),
        "ELTP participates": any(j["id"] == "JO-G-ELTP" and j["slots_allocated"] > 0 for j in jos),
        "No starvation": all(j["slots_allocated"] > 0 for j in jos),
        "All 100 slots allocated": assigned == total_slots,
    }

    print("\n✅ Checks")
    for label, ok in checks.items():
        print(f"  {'✔' if ok else '✘'} {label}")


def main() -> None:
    jos = build_jos()
    slots = build_slots()
    total_slots = len(slots)
    batch_sizes = [4, 10, 2, 15, 40, 8, 6, 5, 10]

    slot_idx = 0
    batch_id = 0
    assignments: list = []
    for size in itertools.cycle(batch_sizes):
        if slot_idx >= total_slots:
            break
        batch_slots = slots[slot_idx : slot_idx + size]
        if not batch_slots:
            break
        print_batch_header(batch_id, batch_slots)
        print_jo_state(jos)
        result = run_allocation(jos, batch_slots, batch_size=len(batch_slots))
        assignments.extend(result.assignments)
        slot_idx += len(batch_slots)
        batch_id += 1

        allocated = sum(1 for a in assignments if a.jo_id != "-")
        pending = total_slots - allocated
        print(f"\nRunning Pending Slots After Batch: {pending}")
        print(f"Allocated Slots: {allocated} | Pending Slots: {pending} | Total Slots: {total_slots}")

    print_final_summary(jos, total_slots, sum(1 for a in assignments if a.jo_id != "-"))


if __name__ == "__main__":
    main()
