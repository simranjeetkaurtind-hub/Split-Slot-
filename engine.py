"""
Slot-driven allocation (base score + delta + saturation-band caps).

- Rank candidates by **base score** (desc); priority is already in the score.
- Allocate **slot-by-slot**: delta chooses split; caps stop over-allocation.
- Assignment is argmax(final_score) within the eligible set and caps.
"""

from __future__ import annotations

import math
import sys

from jo_utils import assert_jo_dict, jos_list_to_map, safe_int
from metrics import fairness_lateral_range, lateral_ratios
from models import (
    BatchDebugInfo,
    JOType,
    SimulationResult,
    Slot,
    SlotAssignment,
)
from config import get_delta_threshold
from scoring import (
    base_score,
    final_score,
    get_saturation_band,
    is_saturated,
    priority_level,
    saturation_pct,
    score_breakdown,
    saturation_ratio,
)

DELTA_EQUAL = 0.040
BATCH_SIZE = 4


def _delta_threshold() -> float:
    return get_delta_threshold()


class Phase:
    SLOT = "Slot-by-slot"


def _match_tech_level(jo: dict, slot: Slot) -> bool:
    assert_jo_dict(jo)
    ts = str(jo["tech_stack"]).strip()
    lv = str(jo["level"]).strip()
    st = slot.tech_stack.strip()
    sl = slot.level.strip()
    ts_ok = ts == "*" or ts.lower() == st.lower()
    lv_ok = lv == "*" or lv.lower() == sl.lower()
    return ts_ok and lv_ok


def _lateral_sort_key(jo: dict) -> tuple[float, str]:
    """Descending base score; tie-break by id for stable ordering."""
    return (-base_score(jo), str(jo["id"]))


def _lateral_eligible_for_batch(jos: dict[str, dict], batch_slots: list[Slot]) -> list[dict]:
    """
    Laterals that have active demand and match at least one slot's tech/level in this batch.
    Used for ranking and phase/caps; each slot still applies _match_tech_level per assignment (intentional).
    """
    out: list[dict] = []
    for jo in jos.values():
        assert_jo_dict(jo)
        jo_type = str(jo.get("type", "")).strip().upper()
        if jo_type not in (JOType.LATERAL.value, JOType.ELTP.value):
            continue
        if safe_int(jo.get("active_demand"), 0) <= 0:
            continue
        if any(_match_tech_level(jo, s) for s in batch_slots):
            out.append(jo)
    return out


def _build_batch_caps(jos: dict[str, dict], batch_n: int) -> dict[str, int]:
    caps: dict[str, int] = {}
    for jid, jo in jos.items():
        _, _, _, share = get_saturation_band(jo)
        caps[jid] = int(math.floor(batch_n * share))
    return caps


def _compute_phase_and_caps(
    ranked: list[dict],
    batch_n: int,
) -> tuple[
    str,
    dict[str, int],
    float | None,
    str | None,
    str | None,
    float | None,
    float | None,
    bool,
    bool,
]:
    """
    Lateral/ELTP JOs. Base score only for delta (no affinity). Caps are derived
    from saturation bands once per batch.

    Returns (phase, caps, delta, dominant_id, next_id, top_bs, second_bs,
             top1_sat, top2_sat).
    """
    if not ranked:
        return ("-", {}, None, None, None, None, None, False, False)
    top1 = ranked[0]
    assert_jo_dict(top1)
    dominant_id = str(top1["id"])
    b0 = base_score(top1)
    top1_sat = is_saturated(top1)
    if len(ranked) == 1:
        caps = _build_batch_caps({dominant_id: top1}, batch_n)
        return (
            Phase.SLOT,
            caps,
            None,
            dominant_id,
            None,
            b0,
            None,
            top1_sat,
            False,
        )

    top2 = ranked[1]
    assert_jo_dict(top2)
    next_id = str(top2["id"])
    b1 = base_score(top2)
    delta = abs(b0 - b1)
    top2_sat = is_saturated(top2)

    caps = _build_batch_caps({str(j["id"]): j for j in ranked}, batch_n)

    return (
        Phase.SLOT,
        caps,
        delta,
        dominant_id,
        next_id,
        b0,
        b1,
        top1_sat,
        top2_sat,
    )


def _lateral_jo_ids(jos: dict[str, dict]) -> list[str]:
    return [
        jid
        for jid, j in jos.items()
        if str(j.get("type", "")).strip().upper() in (JOType.LATERAL.value, JOType.ELTP.value)
    ]


def _ensure_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _saturation_display_pct(jo: dict) -> str:
    assert_jo_dict(jo)
    if safe_int(jo.get("slots_allocated"), 0) == 0:
        return "INACTIVE"
    return f"{saturation_pct(jo):.1f}%"


def _cap_remaining_batch(jo: dict, caps: dict[str, int], lateral_counts: dict[str, int]) -> int:
    jid = str(jo["id"])
    return max(0, caps.get(jid, 0) - lateral_counts.get(jid, 0))


def _lateral_valid_for_slot(
    jo: dict,
    slot: Slot,
    caps: dict[str, int],
    lateral_counts: dict[str, int],
    *,
    enforce_positive_batch_cap: bool,
) -> tuple[bool, str]:
    assert_jo_dict(jo)
    jo_type = str(jo.get("type", "")).strip().upper()
    if jo_type not in (JOType.LATERAL.value, JOType.ELTP.value):
        return False, "not eligible type"
    if not _match_tech_level(jo, slot):
        return False, "tech/level mismatch"
    if safe_int(jo.get("active_demand"), 0) <= 0:
        return False, "active_demand is 0"
    init = safe_int(jo.get("initial_demand"), 0)
    if init <= 0:
        return False, "initial_demand is 0"
    jid = str(jo["id"])
    batch_cap = caps.get(jid, 0)
    if enforce_positive_batch_cap and batch_cap <= 0:
        return False, "no batch cap reserved for this JO"
    if batch_cap > 0 and lateral_counts.get(jid, 0) >= batch_cap:
        return False, "batch cap exhausted"
    return True, "OK"


def _equal_group_from_ranked(ranked: list[dict]) -> list[dict]:
    if not ranked:
        return []
    b0 = base_score(ranked[0])
    return [j for j in ranked if abs(base_score(j) - b0) <= _delta_threshold()]


def _pick_best_lateral_for_slot(
    phase_label: str,
    slot_idx: int,
    batch_n: int,
    slot: Slot,
    jos: dict[str, dict],
    candidate_ids: list[str],
    caps: dict[str, int],
    allocated_in_batch: dict[str, int],
) -> tuple[dict | None, str, list[str]]:
    """
    Slot-by-slot allocation. Delta controls split; saturation bands control caps.
    """
    log: list[str] = [f"Slot {slot.slot_id} (batch position {slot_idx + 1}/{batch_n}):", ""]
    log.append("Routing: slot-by-slot")

    def score_for(jo: dict) -> float:
        return final_score(jo, slot.project)

    def describe_candidate(jo: dict) -> str:
        bd = score_breakdown(jo, slot.project)
        aff = bd["affinity_score"]
        return (
            f"{jo['id']} → base {bd['base_score']:.2f} + 0.10×{aff:.1f} = {bd['final_score']:.2f}"
        )

    ranked = sorted(
        [jos[jid] for jid in candidate_ids if jid in jos],
        key=lambda j: (-base_score(j), str(j["id"])),
    )
    ranked = [j for j in ranked if _match_tech_level(j, slot) and safe_int(j.get("active_demand"), 0) > 0]
    eligible = [
        j
        for j in ranked
        if allocated_in_batch.get(str(j["id"]), 0) < caps.get(str(j["id"]), 0)
    ]

    log.append("Candidates (eligible):")
    if not eligible:
        log.append("  (none)")
        return None, "none", log
    for jo in eligible:
        line = describe_candidate(jo)
        log.append(f"  {line}")

    j1 = eligible[0]
    j2 = eligible[1] if len(eligible) > 1 else None
    chosen = j1
    if j2 is not None:
        delta = abs(base_score(j1) - base_score(j2))
        band_name, _, _, _ = get_saturation_band(j1)
        force_equal = band_name == "HIGH" and priority_level(j1.get("priority")) >= 2
        if force_equal or delta <= _delta_threshold():
            first, second = (j1, j2) if slot_idx % 2 == 0 else (j2, j1)
            if allocated_in_batch.get(str(first["id"]), 0) < caps.get(str(first["id"]), 0):
                chosen = first
            elif allocated_in_batch.get(str(second["id"]), 0) < caps.get(str(second["id"]), 0):
                chosen = second
        else:
            chosen = j1

    log.append(f"Selected: {chosen['id']} (slot-by-slot choice)")
    return chosen, "slot", log


def can_assign_lateral_explain(
    jo: dict,
    slot: Slot,
    caps: dict[str, int],
    lateral_counts: dict[str, int],
) -> tuple[bool, str]:
    jid = str(jo["id"])
    return _lateral_valid_for_slot(
        jo,
        slot,
        caps,
        lateral_counts,
        enforce_positive_batch_cap=(caps.get(jid, 0) > 0),
    )


def can_assign_lateral(
    jo: dict,
    slot: Slot,
    caps: dict[str, int],
    lateral_counts: dict[str, int],
) -> bool:
    return can_assign_lateral_explain(jo, slot, caps, lateral_counts)[0]


def _lateral_jos_sorted_by_base_score(jos: dict[str, dict], lateral_ids: list[str]) -> list[dict]:
    rows = [jos[jid] for jid in lateral_ids]
    rows.sort(key=_lateral_sort_key)
    return rows


def _try_lateral_fallback_score_order(
    jos: dict[str, dict],
    lateral_ids: list[str],
    slot: Slot,
    caps: dict[str, int],
    lateral_counts: dict[str, int],
) -> dict | None:
    """
    After slot routing fails: try candidates in descending base score (eligible first).
    Priority is reflected in base_score — no separate tier ordering.
    """
    ordered_ids = sorted(lateral_ids, key=lambda jid: _lateral_sort_key(jos[jid]))
    for jid in ordered_ids:
        jo = jos[jid]
        if can_assign_lateral(jo, slot, caps, lateral_counts):
            return jo
    return None


def _lateral_assignment_reason(phase: str, affinity_match: bool, pick_kind: str) -> str:
    parts = [f"Assignment: best final_score after slot routing ({pick_kind})"]
    if phase == Phase.SLOT:
        parts.append("slot-by-slot caps")
    if affinity_match:
        parts.append("project match (affinity)")
    return " | ".join(parts)


def _print_jo_state(label: str, jo: dict) -> None:
    assert_jo_dict(jo)
    print(f"{label}:")
    print(f"  Slots Allocated: {safe_int(jo.get('slots_allocated'), 0)}")
    print(f"  Demand Remaining: {safe_int(jo.get('active_demand'), 0)}")
    print(f"  Saturation: {_saturation_display_pct(jo)}")


def _print_delta_block(
    top_id: str | None,
    top_bs: float | None,
    second_id: str | None,
    second_bs: float | None,
    delta_val: float | None,
) -> None:
    if top_id is not None and top_bs is not None:
        print(f"Top JO: {top_id} ({round(top_bs, 4)})")
    if second_id is not None and second_bs is not None:
        print(f"Second JO: {second_id} ({round(second_bs, 4)})")
    print("")
    if delta_val is not None and top_bs is not None and second_bs is not None:
        print(
            f"Delta = |{round(top_bs, 4)} - {round(second_bs, 4)}| = {round(delta_val, 4)}"
        )
        print(f"Threshold = {_delta_threshold():.3f}")
        if delta_val <= _delta_threshold():
            print("|Δ| ≤ 0.040 (equal tier); slot split alternates 50–50.")
        else:
            print("\u2192 |Δ| > 0.040 \u2192 dominant gets the slot.")
    else:
        print("Delta: n/a (single lateral competitor)")
        print(f"Threshold = {_delta_threshold():.3f}")


def _lateral_jo_roster_lines(sorted_lateral_jos: list[dict], ref_project: str) -> list[str]:
    """Full score visibility at batch start (affinity uses ref_project for display only)."""
    lines: list[str] = ["Candidate JO roster (scores at batch start):", ""]
    for jo in sorted_lateral_jos:
        assert_jo_dict(jo)
        jid = str(jo["id"])
        bd = score_breakdown(jo, ref_project)
        lines.append(f"{jid}:")
        lines.append(f"  Priority Score: {bd['priority_score']:.4f}")
        lines.append(f"  Urgency Score: {bd['urgency_score']:.4f}")
        lines.append(f"  Demand Score: {bd['demand_score']:.4f}")
        lines.append(f"  Base Score: {bd['base_score']:.4f}")
        lines.append(f"  Active Demand: {int(bd['active_demand'])}")
        lines.append(f"  Slots Allocated: {int(bd['slots_allocated'])}")
        lines.append(f"  Saturation: {_saturation_display_pct(jo)}")
        lines.append("")
    return lines


def _build_batch_debug_log(
    bi: int,
    dom_id: str | None,
    next_id: str | None,
    top_bs: float | None,
    second_bs: float | None,
    delta_val: float | None,
    lateral_sat_lines: list[str],
    phase_label: str,
    roster_lines: list[str],
    caps: dict[str, int],
    demand_at_batch_start: dict[str, int],
    lateral_batch_counts: dict[str, int],
    batch_size: int,
    total_lateral: int,
    leftover: int,
    total_slots: int,
    allocated_slots: int,
    pending_slots: int,
    band_map: dict[str, tuple[str, str]],
    eltp_tally: dict[str, int],
    ratios: dict[str, float],
    fairness: float,
    top1_sat: bool = False,
    top2_sat: bool = False,
) -> list[str]:
    lines = [f"Batch ID: {bi}", ""]
    if dom_id and top_bs is not None:
        lines.append(f"Top JO: {dom_id} ({round(top_bs, 4)})")
    if next_id and second_bs is not None:
        lines.append(f"Second JO: {next_id} ({round(second_bs, 4)})")
    lines.append("")
    if delta_val is not None and top_bs is not None and second_bs is not None:
        lines.append(
            f"Delta = |{round(top_bs, 4)} - {round(second_bs, 4)}| = {round(delta_val, 4)}"
        )
        lines.append(f"Threshold = {_delta_threshold():.3f}")
        lines.append(
            "Slot rule: |Δ|≤0.04 → alternate 50–50 between top two; else greedy to #1."
        )
    else:
        lines.append("Delta: n/a (single or no competitor)")
        lines.append(f"Threshold = {_delta_threshold():.3f}")
    lines.append("")
    lines.append("Saturation (candidates, before batch):")
    lines.extend(lateral_sat_lines)
    lines.append("")
    lines.extend(roster_lines)
    lines.append("Slot Decision:")
    lines.append(f"Mode: {phase_label}")
    lines.append(f"Top1 Sat: {top1_sat}, Top2 Sat: {top2_sat}")
    lines.append(f"Delta: {delta_val}")
    lines.append("")
    lines.append("Batch cap vs demand (effective_cap = min(batch_cap, active_demand) at batch start):")
    for jid in sorted(caps.keys()):
        tcap = caps[jid]
        if tcap <= 0:
            continue
        dem = demand_at_batch_start.get(jid, 0)
        eff = min(tcap, dem)
        lines.append(f"  {jid}:")
        lines.append(f"    Batch cap (theoretical): {tcap}")
        lines.append(f"    Remaining demand (batch start): {dem}")
        lines.append(f"    Effective cap: {eff}")
        unusable = max(0, tcap - eff)
        if unusable > 0:
            lines.append(f"    Theoretical cap not usable (demand < cap): {unusable}")
    lines.append("")
    lines.append("Band / Cap / Allocated:")
    for jid in sorted(caps.keys()):
        band_label, split_rule = band_map.get(jid, ("-", "-"))
        cap_val = caps.get(jid, 0)
        allocated = lateral_batch_counts.get(jid, 0)
        status = "BLOCKED" if cap_val <= 0 or allocated >= cap_val else "ACTIVE"
        lines.append(
            f"  {jid}: Band {band_label} | Cap {cap_val} | Allocated {allocated} → {status}"
        )
    lines.append("")
    lines.append("Allocation (this batch, demand-constrained):")
    printed_caps: set[str] = set()
    for jid in sorted(caps.keys()):
        if caps[jid] <= 0:
            continue
        printed_caps.add(jid)
        a = lateral_batch_counts.get(jid, 0)
        tcap = caps[jid]
        dem0 = demand_at_batch_start.get(jid, 0)
        eff0 = min(tcap, dem0)
        lines.append(f"  {jid}: assigned = {a} (max effective was {eff0})")
        if a < eff0:
            lines.append(
                "    (below effective max - saturation, tech/level mismatch on some slots, "
                "or batch cap routed slots to other laterals in base-score order)"
            )
    for jid in sorted(lateral_batch_counts.keys()):
        a = lateral_batch_counts.get(jid, 0)
        if a <= 0 or jid in printed_caps:
            continue
        lines.append(
            f"  {jid}: assigned = {a} (advisory batch cap 0; slots taken after reserved-cap laterals could not)"
        )
    lines.append(f"Batch size: {batch_size}")
    lines.append(f"Total slots: {total_slots}")
    lines.append(f"Allocated slots: {allocated_slots}")
    lines.append(f"Pending slots: {pending_slots}")
    lines.append(f"Running Pending Slots After Batch: {pending_slots}")
    lines.append(f"Leftover (batch_size - assigned): {leftover}")
    lines.append("")
    lines.append("Lateral ratios (after batch, slots / initial_demand):")
    for jid in sorted(ratios.keys()):
        lines.append(f"  {jid}: {ratios[jid]:.4f}")
    lines.append("")
    lines.append(f"Fairness (lateral, 1 - spread): {fairness:.4f}")
    return lines


def run_allocation(
    job_openings: list[dict],
    slots: list[Slot],
    batch_size: int = BATCH_SIZE,
) -> SimulationResult:
    _ensure_utf8_stdout()
    for jo in job_openings:
        assert_jo_dict(jo)
    jos: dict[str, dict] = jos_list_to_map(job_openings)
    jo_base_initial = {jid: base_score(jos[jid]) for jid in jos}
    lat_ids_init = _lateral_jo_ids(jos)
    lat_ranked = sorted(lat_ids_init, key=lambda jid: (-jo_base_initial[jid], jid))
    lateral_dom_init = lat_ranked[0] if lat_ranked else None
    lateral_comp_init = lat_ranked[1] if len(lat_ranked) > 1 else None

    assignments: list[SlotAssignment] = []
    final_matrix: dict[tuple[str, str], float] = {}
    breakdown: dict[tuple[str, str], dict[str, float]] = {}

    slot_list = list(slots)
    total_slots = len(slot_list)
    batches: list[list[Slot]] = []
    for i in range(0, len(slot_list), batch_size):
        batches.append(slot_list[i : i + batch_size])
    batch_queue = list(batches)

    batch_debug: list[BatchDebugInfo] = []
    cumulative: dict[str, int] = {jid: 0 for jid in jos}

    batch_index = 0
    while batch_queue:
        batch_slots = batch_queue.pop(0)
        total_demand = sum(safe_int(j.get("active_demand"), 0) for j in jos.values())
        if total_demand <= 0:
            break
        n = len(batch_slots)
        if n <= 0:
            continue
        lateral_ids = _lateral_jo_ids(jos)

        eligible_batch = _lateral_eligible_for_batch(jos, batch_slots)
        eligible_batch.sort(key=_lateral_sort_key)

        (
            phase_label,
            caps,
            delta_val,
            dom_id,
            next_id,
            top_bs,
            second_bs,
            top1_sat,
            top2_sat,
        ) = _compute_phase_and_caps(eligible_batch, n)
        caps = _build_batch_caps({jid: jos[jid] for jid in lateral_ids}, n)
        for jid in lateral_ids:
            caps.setdefault(jid, 0)
        allocated_in_batch: dict[str, int] = {jid: 0 for jid in lateral_ids}
        demand_at_batch_start: dict[str, int] = {
            jid: safe_int(jos[jid]["active_demand"], 0) for jid in lateral_ids
        }

        sat_map = {jid: saturation_pct(jos[jid]) for jid in jos}
        thresh_map: dict[str, float] = {}
        base_snapshot = {jid: base_score(jos[jid]) for jid in lateral_ids}
        lateral_batch_counts: dict[str, int] = {jid: 0 for jid in lateral_ids}
        eltp_batch_counts: dict[str, int] = {}

        lateral_sat_lines = []
        band_map: dict[str, tuple[str, str]] = {}
        for jid in sorted(lateral_ids):
            j = jos[jid]
            sp = saturation_ratio(j) * 100.0
            _, band_label, split_rule, _ = get_saturation_band(j)
            band_map[jid] = (band_label, split_rule)
            lateral_sat_lines.append(
                f"  {jid} = {sp:.1f}% | Band: {band_label} | Split Rule: {split_rule}"
            )
            lateral_sat_lines.append(
                f"    Cap: {caps.get(jid, 0)} | Allocated: {allocated_in_batch.get(jid, 0)}"
            )

        print("\n" + "=" * 60)
        print(f"Batch ID: {batch_index}")
        print("- Candidate pool: ranking, delta, caps (ELTP included) -")
        _print_delta_block(dom_id, top_bs, next_id, second_bs, delta_val)
        print("")
        print("Saturation (candidate JOs, start of batch):")
        for line in lateral_sat_lines:
            print(line)
        print("")
        print("Slot Decision:")
        print(f"Mode: {phase_label}")
        print(f"Top1 Sat: {top1_sat}, Top2 Sat: {top2_sat}")
        print(f"Delta: {delta_val}")
        print("")
        print("Batch cap vs demand (effective_cap = min(batch_cap, active_demand)):")
        for ck, cv in sorted(caps.items()):
            if cv <= 0:
                continue
            dem = demand_at_batch_start.get(ck, 0)
            eff = min(cv, dem)
            print(
                f"  {jos[ck]['id']} => Demand: {dem}, "
                f"Cap (theoretical): {cv}, Effective: {eff}"
            )
        print("")

        sorted_lateral_jos = _lateral_jos_sorted_by_base_score(jos, lateral_ids)
        ref_proj = batch_slots[0].project if batch_slots else ""
        roster_lines = _lateral_jo_roster_lines(sorted_lateral_jos, ref_proj)
        for ln in roster_lines:
            print(ln)
        batch_slot_assignments: list[SlotAssignment | None] = [None] * n

        for idx, slot in enumerate(batch_slots):
            print("")
            print(f"Slot ID: {slot.slot_id}")
            print("- Per-slot scores (each candidate vs this slot) -")
            for jo in sorted_lateral_jos:
                jid = str(jo["id"])
                bd = score_breakdown(jos[jid], slot.project)
                breakdown[(slot.slot_id, jid)] = bd
                final_matrix[(slot.slot_id, jid)] = bd["final_score"]
                print(f"{jid}:")
                print(f"  Base: {bd['base_score']:.4f}")
                print(f"  Affinity: {bd['affinity_score']:.1f}")
                print(f"  Final: {bd['final_score']:.4f}")
                sat_line = (
                    "INACTIVE"
                    if int(bd["slots_allocated"]) == 0
                    else f"{bd['saturation_ratio'] * 100:.1f}%"
                )
                print(
                    f"  (P={bd['priority_score']:.4f} U={bd['urgency_score']:.4f} D={bd['demand_score']:.4f} | "
                    f"slots={int(bd['slots_allocated'])} demand={int(bd['active_demand'])} "
                    f"sat={sat_line})"
                )
                print("")

            print("---- SLOT TRACE (slot route → filter → argmax final_score) ----")
            picked, pick_kind, trace_lines = _pick_best_lateral_for_slot(
                phase_label,
                idx,
                n,
                slot,
                jos,
                lateral_ids,
                caps,
                allocated_in_batch,
            )
            for line in trace_lines:
                print(line)
            if phase_label == Phase.SLOT:
                print("Slot-by-slot batch: delta controls split, caps enforce saturation bands.")

            assigned_jo_id: str | None = None
            last_lateral_fail_reason = "No lateral JO could take this slot"

            if picked is not None:
                jid = str(picked["id"])
                allocated_in_batch[jid] += 1
                lateral_batch_counts[jid] += 1
                cumulative[jid] += 1
                assigned_jo = jos[jid]
                assigned_jo["slots_allocated"] = safe_int(assigned_jo.get("slots_allocated"), 0) + 1
                assigned_jo["active_demand"] = max(0, safe_int(assigned_jo.get("active_demand"), 0) - 1)
                fs = final_score(assigned_jo, slot.project)
                aff = str(assigned_jo["project"]).strip() == slot.project.strip()
                reason = _lateral_assignment_reason(phase_label, aff, pick_kind)
                sat_after = saturation_ratio(assigned_jo)
                print(f"  Saturation after assign: {sat_after:.4f}")
                _print_jo_state("Updated", assigned_jo)
                assigned_jo_id = jid
                batch_slot_assignments[idx] = SlotAssignment(
                    slot_id=slot.slot_id,
                    jo_id=jid,
                    allocation_type=phase_label,
                    reason=reason,
                    final_score=fs,
                    affinity_match=aff,
                )
            else:
                last_lateral_fail_reason = trace_lines[-1] if trace_lines else last_lateral_fail_reason

            if assigned_jo_id is None:
                fb = _try_lateral_fallback_score_order(jos, lateral_ids, slot, caps, allocated_in_batch)
                if fb is not None:
                    jid = str(fb["id"])
                    print(
                        "Structured fallback (base score desc, first can_assign): "
                        f"slot routing had no winner → assigning {jid}"
                    )
                    allocated_in_batch[jid] += 1
                    lateral_batch_counts[jid] += 1
                    cumulative[jid] += 1
                    assigned_jo = jos[jid]
                    assigned_jo["slots_allocated"] = safe_int(assigned_jo.get("slots_allocated"), 0) + 1
                    assigned_jo["active_demand"] = max(0, safe_int(assigned_jo.get("active_demand"), 0) - 1)
                    fs = final_score(assigned_jo, slot.project)
                    aff = str(assigned_jo["project"]).strip() == slot.project.strip()
                    reason = "Fallback: structured score-order (after slot route)"
                    sat_after = saturation_ratio(assigned_jo)
                    print(f"  Saturation after assign: {sat_after:.4f}")
                    _print_jo_state("Updated", assigned_jo)
                    assigned_jo_id = jid
                    batch_slot_assignments[idx] = SlotAssignment(
                        slot_id=slot.slot_id,
                        jo_id=jid,
                        allocation_type=phase_label,
                        reason=reason,
                        final_score=fs,
                        affinity_match=aff,
                    )

            if assigned_jo_id is None:
                any_demand = any(safe_int(jos[jid].get("active_demand"), 0) > 0 for jid in lateral_ids)
                still_eligible = any(
                    can_assign_lateral(jos[jid], slot, caps, allocated_in_batch) for jid in lateral_ids
                )
                if any_demand and not still_eligible:
                    print("Note: demand > 0 but no candidate can_assign this slot (caps/tech).")
                if still_eligible:
                    print("ERROR: candidate available (can_assign) but none assigned (logic bug).")
                print("No candidate eligible → unassigned")
                batch_slot_assignments[idx] = SlotAssignment(
                    slot_id=slot.slot_id,
                    jo_id="-",
                    allocation_type="Unassigned",
                    reason=last_lateral_fail_reason,
                    final_score=None,
                    affinity_match=None,
                )

        total_lateral_assigned = sum(allocated_in_batch.values())
        leftover = n - total_lateral_assigned
        print("")
        print("- Allocation result vs demand -")
        printed_result: set[str] = set()
        for ck, cv in sorted(caps.items()):
            if cv <= 0:
                continue
            printed_result.add(ck)
            jo = jos[ck]
            assigned = lateral_batch_counts.get(ck, 0)
            dem0 = demand_at_batch_start.get(ck, 0)
            eff0 = min(cv, dem0)
            gap = max(0, cv - eff0)
            print(f"  {jo['id']}: assigned = {assigned}")
            print(f"    Batch cap (theoretical): {cv} | Demand @ batch start: {dem0} | Effective cap: {eff0}")
            if gap > 0:
                print(f"    Unfulfilled batch-cap slots due to demand ceiling: {gap} (cannot give more than demand)")
            if assigned < eff0:
                print(
                    "    Assigned below effective cap => limited by tech mismatch / saturation / cap for this batch."
                )
        for ck in sorted(lateral_batch_counts.keys()):
            assigned = lateral_batch_counts.get(ck, 0)
            if assigned <= 0 or ck in printed_result:
                continue
            jo = jos[ck]
            print(f"  {jo['id']}: assigned = {assigned} (advisory batch cap 0; saturation/overflow path)")
        print(f"Batch size: {n} | Total assigned: {total_lateral_assigned}")
        print(f"Leftover = batch_size - assigned = {n} - {total_lateral_assigned} = {leftover}")

        assert all(a is not None for a in batch_slot_assignments)
        assignments.extend(batch_slot_assignments)  # type: ignore[arg-type]
        allocated_slots = sum(1 for a in assignments if a.jo_id != "-")
        pending_slots = total_slots - allocated_slots
        print(f"Pending Slots: {pending_slots}")
        print(f"Allocated Slots: {allocated_slots}")
        print(f"Total Slots: {total_slots}")

        print("")

        lat_rat = lateral_ratios(jos)
        fair_b = fairness_lateral_range(jos)

        log_lines = _build_batch_debug_log(
            batch_index,
            dom_id,
            next_id,
            top_bs,
            second_bs,
            delta_val,
            lateral_sat_lines,
            phase_label,
            roster_lines,
            caps,
            demand_at_batch_start,
            dict(lateral_batch_counts),
            n,
            total_lateral_assigned,
            leftover,
            total_slots,
            allocated_slots,
            pending_slots,
            band_map,
            {k: v for k, v in eltp_batch_counts.items() if v > 0},
            lat_rat,
            fair_b,
            top1_sat,
            top2_sat,
        )
        print("")
        print("- Batch summary -")
        for line in log_lines:
            print(line)

        batch_debug.append(
            BatchDebugInfo(
                batch_index=batch_index,
                slot_ids=[s.slot_id for s in batch_slots],
                lateral_base_scores=base_snapshot,
                delta_value=delta_val,
                phase=phase_label,
                caps=dict(caps),
                saturation_pct=sat_map,
                saturation_threshold_pct=thresh_map,
                dominant_id=dom_id,
                next_id=next_id,
                lateral_assigned_this_batch={k: v for k, v in lateral_batch_counts.items() if v > 0},
                eltp_assigned_this_batch={k: v for k, v in eltp_batch_counts.items() if v > 0},
                cumulative_slots_assigned=dict(cumulative),
                lateral_ratios_after_batch=dict(lat_rat),
                fairness_lateral_batch=fair_b,
                debug_log_lines=log_lines,
            )
        )

        batch_index += 1
        if leftover > 0:
            unassigned_slots = [
                slot
                for slot, assignment in zip(batch_slots, batch_slot_assignments)
                if assignment is not None and assignment.jo_id == "-"
            ]
            if unassigned_slots:
                remaining_demand = sum(
                    safe_int(j.get("active_demand"), 0) for j in jos.values()
                )
                if remaining_demand <= 0:
                    break
                can_take = any(
                    can_assign_lateral(jos[jid], slot, caps, allocated_in_batch)
                    for slot in unassigned_slots
                    for jid in lateral_ids
                )
                if not can_take:
                    break
                if remaining_demand > 0:
                    batch_queue.insert(0, unassigned_slots)

    return SimulationResult(
        assignments=assignments,
        jo_snapshots_end=dict(jos),
        batch_debug=batch_debug,
        jo_base_scores_initial=jo_base_initial,
        score_breakdown_by_slot_jo=breakdown,
        jo_final_score_matrix=final_matrix,
        lateral_dominant_id_initial=lateral_dom_init,
        lateral_competing_id_initial=lateral_comp_init,
    )
