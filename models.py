"""Data models for the Slot Allocation Simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class JOType(str, Enum):
    LATERAL = "LATERAL"
    ELTP = "ELTP"


@dataclass
class Slot:
    slot_id: str
    panel_id: str
    project: str
    tech_stack: str
    level: str
    batch_id: str = ""


@dataclass
class SlotAssignment:
    slot_id: str
    jo_id: str
    allocation_type: str
    reason: str
    final_score: float | None = None
    affinity_match: bool | None = None


@dataclass
class BatchDebugInfo:
    batch_index: int
    slot_ids: list[str]
    lateral_base_scores: dict[str, float]
    delta_value: float | None
    phase: str
    caps: dict[str, int]
    saturation_pct: dict[str, float]
    saturation_threshold_pct: dict[str, float]
    dominant_id: str | None
    next_id: str | None
    lateral_assigned_this_batch: dict[str, int]
    eltp_assigned_this_batch: dict[str, int]
    cumulative_slots_assigned: dict[str, int]
    lateral_ratios_after_batch: dict[str, float]
    fairness_lateral_batch: float
    debug_log_lines: list[str] = field(default_factory=list)


@dataclass
class SimulationResult:
    assignments: list[SlotAssignment]
    jo_snapshots_end: dict[str, dict]
    batch_debug: list[BatchDebugInfo]
    jo_base_scores_initial: dict[str, float]
    score_breakdown_by_slot_jo: dict[tuple[str, str], dict[str, float]]
    jo_final_score_matrix: dict[tuple[str, str], float]
    lateral_dominant_id_initial: str | None = None
    lateral_competing_id_initial: str | None = None
