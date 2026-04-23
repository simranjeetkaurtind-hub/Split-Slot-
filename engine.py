"""
Slot-driven allocation (base score + delta + saturation-band caps).

- Rank candidates by **base score** (desc); priority is already in the score.
- Layered allocation per batch:
  - Delta chooses competitors at each layer (JO-1 vs JO-2).
  - Saturation bands (`get_split_ratio`) decide top-pool size for that layer.
  - Delta decides split within the pool (equal → 50–50; else greedy).
  - Remaining slots repeat the same logic on the remaining JOs.
- Assignment is argmax(final_score) within the layer’s candidate set and caps.
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
from scoring import (
    base_score,
    final_score,
    get_saturation_band,
    is_saturated,
    get_split_ratio,
    saturation_pct,
    score_breakdown,
    saturation_ratio,
)

DELTA_EQUAL = 0.040
BATCH_SIZE = 4


class Phase:
    LAYERED = "Layered"
    GREEDY = "Greedy"


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


def _top_next_pool_sizes(batch_n: int, dominant_share: float = 0.6) -> tuple[int, int]:
    """
    dominant_share: fraction of batch going to dominant+competing top pool.
    Defaults to 0.6 when no band overrides.
    """
    top_pool = int(math.floor(batch_n * dominant_share))
    return top_pool, batch_n - top_pool


def _next_tier_top_two(ranked: list[dict]) -> list[dict]:
    """
    After global rank (by base score), the next tier is everyone below JO-1 & JO-2.
    Take the top two **within that remainder** — delta logic applies only to that pair.
    """
    if len(ranked) <= 2:
        return []
    rest = ranked[2:]
    if not rest:
        return []
    if len(rest) == 1:
        return [rest[0]]
    return [rest[0], rest[1]]


def _apply_next_tier_caps(caps: dict[str, int], ranked: list[dict], next_pool: int) -> None:
    """
    Allocate next_pool among the top two competitors in ranked[2:] (by base score).
    |Δ| ≤ 0.04 → split ~50–50 between them; else higher score gets all next_pool (greedy).
    Lower-ranked JOs (rest[2:]) get no reserved next-tier cap.
    """
    if next_pool <= 0:
        return
    if len(ranked) <= 2:
        return
    pair = _next_tier_top_two(ranked)
    if not pair:
        return
    if len(pair) == 1:
        jid = str(pair[0]["id"])
        caps[jid] = caps.get(jid, 0) + next_pool
        return
    a, b = pair[0], pair[1]
    assert_jo_dict(a)
    assert_jo_dict(b)
    ida, idb = str(a["id"]), str(b["id"])
    d = abs(base_score(a) - base_score(b))
    if d <= DELTA_EQUAL:
        ca = int(math.ceil(next_pool / 2))
        cb = next_pool - ca
    else:
        ca, cb = next_pool, 0
    caps[ida] = caps.get(ida, 0) + ca
    caps[idb] = caps.get(idb, 0) + cb


def _caps_case_a_equal_both_saturated(
    ranked: list[dict], batch_n: int, dominant_share: float = 0.6
) -> dict[str, int]:
    """
    Legacy helper (pre-layered). Retained for reference; layered mode no longer uses this path.
    """
    top_pool, next_pool = _top_next_pool_sizes(batch_n, dominant_share=dominant_share)
    id0 = str(ranked[0]["id"])
    id1 = str(ranked[1]["id"])
    c0 = int(math.ceil(top_pool / 2))
    c1 = top_pool - c0
    if len(ranked) == 2:
        return {id0: c0, id1: c1 + next_pool}
    caps: dict[str, int] = {id0: c0, id1: c1}
    _apply_next_tier_caps(caps, ranked, next_pool)
    return caps


def _caps_case_b_dominant_saturated(
    ranked: list[dict], batch_n: int, dominant_share: float = 0.6
) -> dict[str, int]:
    """
    Legacy helper (pre-layered). Retained for reference; layered mode no longer uses this path.
    """
    top_pool, next_pool = _top_next_pool_sizes(batch_n, dominant_share=dominant_share)
    id0 = str(ranked[0]["id"])
    id1 = str(ranked[1]["id"])
    if len(ranked) == 2:
        return {id0: top_pool, id1: next_pool}
    c0 = int(math.ceil(top_pool / 2))
    c1 = top_pool - c0
    caps: dict[str, int] = {id0: c0, id1: c1}
    _apply_next_tier_caps(caps, ranked, next_pool)
    return caps


def _build_layer_specs(ranked: list[dict], batch_n: int) -> tuple[list[dict], dict[str, int]]:
    """
    Build layered cap specs for a batch:
    - layer candidates come from the top-ranked remaining JOs
    - delta decides whether JO-2 competes in the layer
    - saturation (get_split_ratio) decides top_pool_slots size
    """
    layers: list[dict] = []
    remaining_slots = batch_n
    remaining = list(ranked)
    caps_total: dict[str, int] = {str(j["id"]): 0 for j in ranked}

    while remaining_slots > 0 and remaining:
        if len(remaining) == 1:
            jo = remaining[0]
            jid = str(jo["id"])
            layer_caps = {jid: remaining_slots}
            layers.append(
                {
                    "slot_count": remaining_slots,
                    "candidate_ids": [jid],
                    "caps": layer_caps,
                    "delta": None,
                }
            )
            caps_total[jid] = caps_total.get(jid, 0) + remaining_slots
            break

        top1, top2 = remaining[0], remaining[1]
        assert_jo_dict(top1)
        assert_jo_dict(top2)
        id0 = str(top1["id"])
        id1 = str(top2["id"])
        _, _, split_rule, dom_share = get_saturation_band(top1)
        delta = abs(base_score(top1) - base_score(top2))
        if dom_share == 1.0:
            if delta <= DELTA_EQUAL:
                c0 = int(math.ceil(remaining_slots / 2))
                c1 = remaining_slots - c0
                layer_caps = {id0: c0, id1: c1}
                candidate_ids = [id0, id1]
                split_rule = "50:50 (delta in greedy band)"
            else:
                layer_caps = {id0: remaining_slots}
                candidate_ids = [id0]
            layers.append(
                {
                    "slot_count": remaining_slots,
                    "candidate_ids": candidate_ids,
                    "caps": layer_caps,
                    "delta": delta,
                    "split_rule": split_rule,
                }
            )
            for jid, count in layer_caps.items():
                caps_total[jid] = caps_total.get(jid, 0) + count
            remaining_slots = 0
            break
        top_pool_slots = int(math.floor(remaining_slots * dom_share))
        top_pool_slots = max(0, min(top_pool_slots, remaining_slots))

        if top_pool_slots == 0:
            remove_ids = {id0} if delta > DELTA_EQUAL else {id0, id1}
            remaining = [j for j in remaining if str(j["id"]) not in remove_ids]
            continue
        if delta <= DELTA_EQUAL:
            c0 = int(math.ceil(top_pool_slots / 2))
            c1 = top_pool_slots - c0
            layer_caps = {id0: c0, id1: c1}
            candidate_ids = [id0, id1]
            remove_ids = {id0, id1}
        else:
            layer_caps = {id0: top_pool_slots}
            candidate_ids = [id0]
            remove_ids = {id0}

        layers.append(
            {
                "slot_count": top_pool_slots,
                "candidate_ids": candidate_ids,
                "caps": layer_caps,
                "delta": delta,
                "split_rule": split_rule,
            }
        )
        for jid, count in layer_caps.items():
            caps_total[jid] = caps_total.get(jid, 0) + count

        remaining_slots -= top_pool_slots
        remaining = [j for j in remaining if str(j["id"]) not in remove_ids]

    return layers, caps_total


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
    int | None,
    int | None,
    list[dict],
]:
    """
    Lateral/ELTP JOs. Base score only for delta (no affinity). Layering is driven by `get_split_ratio`
    (dominant saturation) and delta at each layer.

    Returns (phase, caps, delta, dominant_id, next_id, top_bs, second_bs,
             top1_sat, top2_sat, top_pool, next_pool, layer_specs).
    """
    if not ranked:
        return ("-", {}, None, None, None, None, None, False, False, None, None, [])
    top1 = ranked[0]
    assert_jo_dict(top1)
    dominant_id = str(top1["id"])
    b0 = base_score(top1)
    top1_sat = is_saturated(top1)
    if len(ranked) == 1:
        layers, caps = _build_layer_specs(ranked, batch_n)
        top_pool_dbg = layers[0]["slot_count"] if layers else None
        next_pool_dbg = batch_n - top_pool_dbg if top_pool_dbg is not None else None
        return (
            Phase.LAYERED,
            caps,
            None,
            dominant_id,
            None,
            b0,
            None,
            top1_sat,
            False,
            top_pool_dbg,
            next_pool_dbg,
            layers,
        )

    top2 = ranked[1]
    assert_jo_dict(top2)
    next_id = str(top2["id"])
    b1 = base_score(top2)
    delta = abs(b0 - b1)
    top2_sat = is_saturated(top2)

    layers, caps = _build_layer_specs(ranked, batch_n)
    top_pool_dbg = layers[0]["slot_count"] if layers else None
    next_pool_dbg = batch_n - top_pool_dbg if top_pool_dbg is not None else None

    return (
        Phase.LAYERED,
        caps,
        delta,
        dominant_id,
        next_id,
        b0,
        b1,
        top1_sat,
        top2_sat,
        top_pool_dbg,
        next_pool_dbg,
        layers,
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
    return [j for j in ranked if abs(base_score(j) - b0) <= DELTA_EQUAL]


def _pick_best_lateral_for_slot(
    phase_label: str,
    ranked: list[dict],
    slot_idx: int,
    batch_n: int,
    slot: Slot,
    caps: dict[str, int],
    lateral_counts: dict[str, int],
    layer_specs: list[dict],
) -> tuple[dict | None, str, list[str]]:
    """
    Layered routing by batch index. Each layer defines its own candidate set based on
    ranking + delta + saturation splits. Highest final_score wins within the layer's caps.
    """
    log: list[str] = [f"Slot {slot.slot_id} (batch position {slot_idx + 1}/{batch_n}):", ""]
    if not layer_specs:
        log.append("Routing: no layer specs available.")
        return None, "none", log

    cumulative = 0
    layer_idx = 0
    for i, layer in enumerate(layer_specs):
        cumulative += int(layer["slot_count"])
        if slot_idx < cumulative:
            layer_idx = i
            break
    log.append(f"Routing: layer {layer_idx + 1}/{len(layer_specs)}")

    def score_for(jo: dict) -> float:
        return final_score(jo, slot.project)

    def describe_candidate(jo: dict) -> str:
        bd = score_breakdown(jo, slot.project)
        aff = bd["affinity_score"]
        return (
            f"{jo['id']} → base {bd['base_score']:.2f} + 0.10×{aff:.1f} = {bd['final_score']:.2f}"
        )

    def valid_ok(pool: list[dict]) -> list[dict]:
        return [
            jo
            for jo in pool
            if _lateral_valid_for_slot(
                jo,
                slot,
                caps,
                lateral_counts,
                enforce_positive_batch_cap=True,
            )[0]
        ]

    pick_kind = "layer"
    for idx in range(layer_idx, len(layer_specs)):
        layer = layer_specs[idx]
        candidate_ids = set(layer["candidate_ids"])
        primary = [jo for jo in ranked if str(jo["id"]) in candidate_ids]

        log.append(f"Candidates (layer {idx + 1}):")
        if not primary:
            log.append("  (none)")
        for jo in primary:
            ok, why = _lateral_valid_for_slot(
                jo,
                slot,
                caps,
                lateral_counts,
                enforce_positive_batch_cap=True,
            )
            line = describe_candidate(jo)
            if not ok:
                line += f"  [not eligible: {why}]"
            log.append(f"  {line}")

        log.append("Caps remaining (batch reservation):")
        for jo in primary:
            jid = str(jo["id"])
            br = _cap_remaining_batch(jo, caps, lateral_counts) if caps.get(jid, 0) > 0 else 0
            log.append(f"  {jid} → {br}")

        valid_p = valid_ok(primary)
        if valid_p:
            best = max(valid_p, key=score_for)
            log.append(f"Selected: {best['id']} (best final_score in allowed set)")
            return best, pick_kind, log

    log.append("No valid candidate in any layer.")
    return None, "none", log


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
    After layer routing fails: try candidates in descending base score (eligible first).
    Priority is reflected in base_score — no separate tier ordering.
    """
    ordered_ids = sorted(lateral_ids, key=lambda jid: _lateral_sort_key(jos[jid]))
    for jid in ordered_ids:
        jo = jos[jid]
        if can_assign_lateral(jo, slot, caps, lateral_counts):
            return jo
    return None


def _lateral_assignment_reason(phase: str, affinity_match: bool, pick_kind: str) -> str:
    parts = [f"Assignment: best final_score after layered routing ({pick_kind})"]
    if phase == Phase.LAYERED:
        parts.append("layered split caps")
    elif phase == Phase.GREEDY:
        parts.append("greedy cap on dominant")
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
        print("Threshold = 0.040")
        if delta_val <= DELTA_EQUAL:
            print("|Δ| ≤ 0.040 (equal tier); layer split will be 50–50 for that pool.")
        else:
            print("\u2192 |Δ| > 0.040 \u2192 dominant gets full pool for that layer.")
    else:
        print("Delta: n/a (single lateral competitor)")
        print("Threshold = 0.040")


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
    eltp_tally: dict[str, int],
    ratios: dict[str, float],
    fairness: float,
    top1_sat: bool = False,
    top2_sat: bool = False,
    top_pool_dbg: int | None = None,
    next_pool_dbg: int | None = None,
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
        lines.append("Threshold = 0.040")
        lines.append(
            "Layer rule: delta chooses competitors; saturation band (dominant) sizes the pool; "
            "|Δ|≤0.04 → 50–50 inside the pool, else dominant gets the pool."
        )
    else:
        lines.append("Delta: n/a (single or no competitor)")
        lines.append("Threshold = 0.040")
    lines.append("")
    lines.append("Saturation (candidates, before batch):")
    lines.extend(lateral_sat_lines)
    lines.append("")
    lines.extend(roster_lines)
    lines.append("Layer Decision:")
    lines.append(f"Mode: {phase_label}")
    lines.append(f"Top1 Sat: {top1_sat}, Top2 Sat: {top2_sat}")
    lines.append(f"Delta: {delta_val}")
    if phase_label == Phase.LAYERED and top_pool_dbg is not None and next_pool_dbg is not None:
        lines.append("Top Pool Split:")
        lines.append(f"Top Pool: {top_pool_dbg}")
        lines.append(f"Remaining Pool: {next_pool_dbg}")
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

    batch_debug: list[BatchDebugInfo] = []
    cumulative: dict[str, int] = {jid: 0 for jid in jos}

    for bi, batch_slots in enumerate(batches):
        n = len(batch_slots)
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
            top_pool_dbg,
            next_pool_dbg,
            layer_specs,
        ) = _compute_phase_and_caps(eligible_batch, n)
        for jid in lateral_ids:
            caps.setdefault(jid, 0)
        lateral_counts: dict[str, int] = {jid: 0 for jid in lateral_ids}
        demand_at_batch_start: dict[str, int] = {
            jid: safe_int(jos[jid]["active_demand"], 0) for jid in lateral_ids
        }

        sat_map = {jid: saturation_pct(jos[jid]) for jid in jos}
        thresh_map: dict[str, float] = {}
        base_snapshot = {jid: base_score(jos[jid]) for jid in lateral_ids}
        lateral_batch_counts: dict[str, int] = {jid: 0 for jid in lateral_ids}
        eltp_batch_counts: dict[str, int] = {}

        lateral_sat_lines = []
        for jid in sorted(lateral_ids):
            j = jos[jid]
            sp = saturation_ratio(j) * 100.0
            _, band_label, split_rule, _ = get_saturation_band(j)
            lateral_sat_lines.append(
                f"  {jid} = {sp:.1f}% | Band: {band_label} | Split Rule: {split_rule}"
            )

        print("\n" + "=" * 60)
        print(f"Batch ID: {bi}")
        print("- Candidate pool: ranking, delta, caps (ELTP included) -")
        _print_delta_block(dom_id, top_bs, next_id, second_bs, delta_val)
        print("")
        print("Saturation (candidate JOs, start of batch):")
        for line in lateral_sat_lines:
            print(line)
        print("")
        print("Layer Decision:")
        print(f"Mode: {phase_label}")
        print(f"Top1 Sat: {top1_sat}, Top2 Sat: {top2_sat}")
        print(f"Delta: {delta_val}")
        if phase_label == Phase.LAYERED and top_pool_dbg is not None and next_pool_dbg is not None:
            print("Top Pool Split:")
            print(f"Top Pool: {top_pool_dbg}")
            print(f"Remaining Pool: {next_pool_dbg}")
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

            print("---- SLOT TRACE (layer route → filter → argmax final_score) ----")
            picked, pick_kind, trace_lines = _pick_best_lateral_for_slot(
                phase_label,
                eligible_batch,
                idx,
                n,
                slot,
                caps,
                lateral_counts,
                layer_specs,
            )
            for line in trace_lines:
                print(line)
            if phase_label == Phase.LAYERED:
                print(
                    "Layered batch: delta picks competitors, saturation bands size the pool for each layer."
                )

            assigned_jo_id: str | None = None
            last_lateral_fail_reason = "No lateral JO could take this slot"

            if picked is not None:
                jid = str(picked["id"])
                lateral_counts[jid] += 1
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
                fb = _try_lateral_fallback_score_order(jos, lateral_ids, slot, caps, lateral_counts)
                if fb is not None:
                    jid = str(fb["id"])
                    print(
                        "Structured fallback (base score desc, first can_assign): "
                        f"layer routing had no winner → assigning {jid}"
                    )
                    lateral_counts[jid] += 1
                    lateral_batch_counts[jid] += 1
                    cumulative[jid] += 1
                    assigned_jo = jos[jid]
                    assigned_jo["slots_allocated"] = safe_int(assigned_jo.get("slots_allocated"), 0) + 1
                    assigned_jo["active_demand"] = max(0, safe_int(assigned_jo.get("active_demand"), 0) - 1)
                    fs = final_score(assigned_jo, slot.project)
                    aff = str(assigned_jo["project"]).strip() == slot.project.strip()
                    reason = "Fallback: structured score-order (after layer route)"
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
                    can_assign_lateral(jos[jid], slot, caps, lateral_counts) for jid in lateral_ids
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

        total_lateral_assigned = sum(lateral_counts.values())
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
            bi,
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
            {k: v for k, v in eltp_batch_counts.items() if v > 0},
            lat_rat,
            fair_b,
            top1_sat,
            top2_sat,
            top_pool_dbg,
            next_pool_dbg,
        )
        print("")
        print("- Batch summary -")
        for line in log_lines:
            print(line)

        batch_debug.append(
            BatchDebugInfo(
                batch_index=bi,
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
