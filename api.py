from __future__ import annotations

import copy
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import get_config, save_config
from engine import run_allocation
from jo_utils import assert_jo_dict, normalize_jo_numbers
from models import Slot
from scoring import base_score, get_saturation_band
from scenarios import scenario_definitions

app = FastAPI(title="Slot Allocation Simulator API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConfigUpdate(BaseModel):
    priority_weight: float | None = None
    demand_weight: float | None = None
    urgency_weight: float | None = None
    affinity_weight: float | None = None
    delta_threshold: float | None = None


class SlotPayload(BaseModel):
    slot_id: str
    panel_id: str = ""
    project: str
    tech_stack: str = "*"
    level: str = "*"
    batch_id: str = ""


class SimulateRequest(BaseModel):
    scenario_id: str | None = None
    jos: list[dict[str, Any]] | None = None
    slots: list[SlotPayload] | None = None
    batch_sizes: list[int] | None = Field(default=None, min_length=1)


def _slot_from_payload(p: SlotPayload) -> Slot:
    return Slot(
        slot_id=p.slot_id,
        panel_id=p.panel_id,
        project=p.project,
        tech_stack=p.tech_stack,
        level=p.level,
        batch_id=p.batch_id,
    )


def _normalize_jo_payload(jo: dict[str, Any]) -> dict[str, Any]:
    if not jo.get("id"):
        raise HTTPException(status_code=400, detail="JO is missing required id.")
    base = {
        "priority": 3,
        "days_remaining": 0,
        "total_duration": 0,
        "initial_demand": 0,
        "active_demand": 0,
        "project": "",
        "type": "LATERAL",
        "tech_stack": "*",
        "level": "*",
        "slots_allocated": 0,
    }
    base.update(jo)
    assert_jo_dict(base)
    normalize_jo_numbers(base)
    return base


def _run_dynamic(jos: list[dict], slots: list[Slot], batch_sizes: list[int]) -> dict[str, Any]:
    total_slots = len(slots)
    used_sizes: list[int] = []
    assignments = []
    batch_debug = []
    breakdown: dict[tuple[str, str], dict[str, float]] = {}
    final_matrix: dict[tuple[str, str], float] = {}
    base_initial = {jo["id"]: base_score(jo) for jo in jos}
    dom_id_initial = None
    comp_id_initial = None
    slot_idx = 0
    batch_offset = 0
    cumulative: dict[str, int] = {}

    while slot_idx < total_slots:
        size = batch_sizes[len(used_sizes) % len(batch_sizes)]
        batch_slots = slots[slot_idx : slot_idx + size]
        if not batch_slots:
            break
        result = run_allocation(jos, batch_slots, batch_size=len(batch_slots))
        if dom_id_initial is None:
            dom_id_initial = result.lateral_dominant_id_initial
            comp_id_initial = result.lateral_competing_id_initial
        assignments.extend(result.assignments)
        breakdown.update(result.score_breakdown_by_slot_jo)
        final_matrix.update(result.jo_final_score_matrix)

        for bd in result.batch_debug:
            bd.batch_index = batch_offset + bd.batch_index
            cumulative.update(bd.cumulative_slots_assigned)
            bd.cumulative_slots_assigned = dict(cumulative)
            batch_debug.append(bd)

        slot_idx += len(batch_slots)
        used_sizes.append(len(batch_slots))
        batch_offset += len(result.batch_debug)

    return {
        "assignments": assignments,
        "batch_debug": batch_debug,
        "jo_snapshots_end": {jo["id"]: jo for jo in jos},
        "jo_base_scores_initial": base_initial,
        "score_breakdown_by_slot_jo": breakdown,
        "jo_final_score_matrix": final_matrix,
        "lateral_dominant_id_initial": dom_id_initial,
        "lateral_competing_id_initial": comp_id_initial,
        "batch_sizes_used": used_sizes,
    }


def _explanations(payload: dict[str, Any]) -> list[dict[str, Any]]:
    explanations: list[dict[str, Any]] = []
    breakdown = payload["score_breakdown_by_slot_jo"]
    jo_end = payload["jo_snapshots_end"]
    for a in payload["assignments"]:
        if a.jo_id == "-":
            continue
        slot_scores = {
            jid: data
            for (slot_id, jid), data in breakdown.items()
            if slot_id == a.slot_id
        }
        if not slot_scores:
            continue
        ranked = sorted(slot_scores.items(), key=lambda kv: kv[1]["final_score"], reverse=True)
        top = ranked[0]
        runner = ranked[1] if len(ranked) > 1 else None
        band_label, _, split_rule, _ = get_saturation_band(jo_end[a.jo_id])
        top_breakdown = top[1]
        runner_breakdown = runner[1] if runner else None
        delta = (
            abs(top_breakdown["base_score"] - runner_breakdown["base_score"])
            if runner_breakdown
            else None
        )
        reason = (
            f"Slot {a.slot_id} assigned to {a.jo_id} because highest score "
            f"({top_breakdown['final_score']:.2f}"
            + (f" vs {runner_breakdown['final_score']:.2f}" if runner_breakdown else "")
            + "). "
            f"Base score {top_breakdown['base_score']:.2f}"
            + (f" (delta {delta:.3f})" if delta is not None else "")
            + f"; demand score {top_breakdown['demand_score']:.2f}; "
            f"urgency {top_breakdown['urgency_score']:.2f}; "
            f"priority {top_breakdown['priority_score']:.2f}; "
            f"affinity {top_breakdown['affinity_score']:.2f}. "
            f"Band {band_label} with split rule {split_rule}."
        )
        explanations.append(
            {
                "slot_id": a.slot_id,
                "jo_id": a.jo_id,
                "reason": reason,
            }
        )
    return explanations


@app.get("/config")
def read_config() -> dict[str, Any]:
    return get_config()


@app.post("/config")
def update_config(payload: ConfigUpdate) -> dict[str, Any]:
    current = get_config()
    data = current | {k: v for k, v in payload.model_dump().items() if v is not None}
    save_config(data)
    return data


@app.get("/scenarios")
def list_scenarios() -> list[dict[str, Any]]:
    return scenario_definitions()


@app.post("/simulate")
def simulate(req: SimulateRequest) -> dict[str, Any]:
    scenarios = {s["id"]: s for s in scenario_definitions()}
    if req.scenario_id:
        scenario = scenarios.get(req.scenario_id)
        if not scenario:
            raise HTTPException(status_code=404, detail="Scenario not found.")
        jos = copy.deepcopy(scenario["jos"])
        slots = copy.deepcopy(scenario["slots"])
        batch_sizes = req.batch_sizes or scenario.get("batch_sizes", [4])
    else:
        if not req.jos or not req.slots:
            raise HTTPException(status_code=400, detail="jos and slots are required.")
        jos = [_normalize_jo_payload(j) for j in req.jos]
        slots = [_slot_from_payload(p) for p in req.slots]
        batch_sizes = req.batch_sizes or [4]

    result = _run_dynamic(jos, slots, batch_sizes)
    result["explanations"] = _explanations(result)
    return result
