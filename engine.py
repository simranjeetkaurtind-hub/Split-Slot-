"""
Slot-driven allocation (base score, priority via scoring, saturation share caps, fairness pools).

- Rank laterals by **base score** (desc); priority is already in the score — no separate tier sort.
  Delta = |JO-1 − JO-2| for the top pair; next tier uses the top two among **remaining** JOs by score.
- Phase: equal tier → 50–50 full batch; dominant → Greedy until saturation policy; after saturation
  → top pool = ceil(60% of N) and next pool = remainder (Case A: equal+both saturated ~50–50 in top;
  Case B: top pool shared between JO-1 & JO-2; next tier (40%) to top two among ranked[2:] using delta rules.
- Each slot is routed to **top pool** (only JO-1 & JO-2) or **next pool** (those two remainder leaders per caps), then
  argmax(final_score) among eligible — not first-fit order.
- Lifetime share cap (75% / 50%) throttles Greedy/50–50; in **60–40** phase it is not applied in
  `_lateral_valid_for_slot` (batch caps express the split; avoids double-blocking top pool).
- ELTP only when no lateral can take a slot.
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
    is_saturated,
    saturation_pct,
    saturation_threshold_for_jo,
    score_breakdown,
    saturation_ratio,
)

DELTA_EQUAL = 0.040
BATCH_SIZE = 4


class Phase:
    GREEDY = "Greedy"
    SPLIT_60_40 = "60–40"
    SPLIT_50_50 = "50–50"


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
        if str(jo["type"]).strip().upper() != JOType.LATERAL.value or safe_int(jo.get("active_demand"), 0) <= 0:
            continue
        if any(_match_tech_level(jo, s) for s in batch_slots):
            out.append(jo)
    return out


def _top_next_pool_sizes(batch_n: int) -> tuple[int, int]:
    top_pool = int(math.ceil(batch_n * 0.6))
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


def _caps_case_a_equal_both_saturated(ranked: list[dict], batch_n: int) -> dict[str, int]:
    """
    Case A: delta ≤ 0.04, both saturated — top pool shared ~50–50; next pool to next tier (or JO-2 if only two JOs).

    IMPORTANT: These batch caps are the throttle. Top-pool slots must be filled only by JO-1 and JO-2
    (see _pick_best_lateral_for_slot); saturation ratio must not zero out their eligibility separately.
    """
    top_pool, next_pool = _top_next_pool_sizes(batch_n)
    id0 = str(ranked[0]["id"])
    id1 = str(ranked[1]["id"])
    c0 = int(math.ceil(top_pool / 2))
    c1 = top_pool - c0
    if len(ranked) == 2:
        return {id0: c0, id1: c1 + next_pool}
    caps: dict[str, int] = {id0: c0, id1: c1}
    _apply_next_tier_caps(caps, ranked, next_pool)
    return caps


def _caps_case_b_dominant_saturated(ranked: list[dict], batch_n: int) -> dict[str, int]:
    """
    Case B: 60–40 phase — top pool shared between JO-1 and JO-2; 40% tier to top two among ranked[2:] (delta).

    When len(ranked) > 2, never give ranked[1] batch cap 0 while top-pool routing uses [JO-1, JO-2] —
    that starves the second-highest **score** lateral while lower ranks still receive next-tier caps.
    Split the top pool ~50–50 between the top pair (same split as Case A's top portion).
    """
    top_pool, next_pool = _top_next_pool_sizes(batch_n)
    id0 = str(ranked[0]["id"])
    id1 = str(ranked[1]["id"])
    if len(ranked) == 2:
        return {id0: top_pool, id1: next_pool}
    c0 = int(math.ceil(top_pool / 2))
    c1 = top_pool - c0
    caps: dict[str, int] = {id0: c0, id1: c1}
    _apply_next_tier_caps(caps, ranked, next_pool)
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
    int | None,
    int | None,
]:
    """
    Lateral JOs only. Base score only for delta (no affinity).

    Order matters: both tops saturated → Case A/B; JO-1 saturated and JO-2 not → Case B (not 50–50);
    then delta ≤ 0.04 → 50–50 (only reached when JO-1 is not saturated); then Greedy; no stray 50–50
    when JO-1 is saturated.

    Returns (phase, caps, delta, dominant_id, next_id, top_bs, second_bs,
             top1_sat, top2_sat, top_pool, next_pool).
    """
    if not ranked:
        return ("-", {}, None, None, None, None, None, False, False, None, None)
    top1 = ranked[0]
    assert_jo_dict(top1)
    dominant_id = str(top1["id"])
    b0 = base_score(top1)
    if len(ranked) == 1:
        return (
            Phase.GREEDY,
            {dominant_id: batch_n},
            None,
            dominant_id,
            None,
            b0,
            None,
            is_saturated(top1),
            False,
            None,
            None,
        )

    top2 = ranked[1]
    assert_jo_dict(top2)
    next_id = str(top2["id"])
    b1 = base_score(top2)
    delta = abs(b0 - b1)
    top1_sat = is_saturated(top1)
    top2_sat = is_saturated(top2)

    top_pool_dbg: int | None = None
    next_pool_dbg: int | None = None

    if top1_sat and top2_sat:
        phase = Phase.SPLIT_60_40
        if delta <= DELTA_EQUAL:
            caps = _caps_case_a_equal_both_saturated(ranked, batch_n)
        else:
            caps = _caps_case_b_dominant_saturated(ranked, batch_n)
        top_pool_dbg, next_pool_dbg = _top_next_pool_sizes(batch_n)
    elif top1_sat and not top2_sat:
        # JO-1 saturated, JO-2 not — Case B regardless of delta (cannot stay on 50–50 when JO-1 is saturated).
        phase = Phase.SPLIT_60_40
        caps = _caps_case_b_dominant_saturated(ranked, batch_n)
        top_pool_dbg, next_pool_dbg = _top_next_pool_sizes(batch_n)
    elif delta <= DELTA_EQUAL:
        phase = Phase.SPLIT_50_50
        equal_group = [j for j in ranked if abs(base_score(j) - b0) <= DELTA_EQUAL]
        n_equal = len(equal_group)
        base_share = batch_n // n_equal
        remainder = batch_n % n_equal
        caps = {}
        for i, j in enumerate(equal_group):
            assert_jo_dict(j)
            caps[str(j["id"])] = base_share + (1 if i < remainder else 0)
        for j in ranked:
            assert_jo_dict(j)
            caps.setdefault(str(j["id"]), 0)
        next_id = str(equal_group[1]["id"]) if len(equal_group) > 1 else None
        b1 = base_score(equal_group[1]) if len(equal_group) > 1 else None
    elif not top1_sat:
        phase = Phase.GREEDY
        caps = {dominant_id: batch_n}
        for j in ranked[1:]:
            assert_jo_dict(j)
            caps[str(j["id"])] = 0

    for j in ranked:
        assert_jo_dict(j)
        caps.setdefault(str(j["id"]), 0)

    # Invariant: 50–50 only when delta ≤ 0.04 and JO-1 not saturated — enforced by branch order above.

    return (
        phase,
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
    )


def _lateral_jo_ids(jos: dict[str, dict]) -> list[str]:
    return [jid for jid, j in jos.items() if str(j.get("type", "")).strip().upper() == JOType.LATERAL.value]


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


def _max_slots_under_saturation_lifetime(jo: dict) -> int:
    """Max slots at or below saturation ratio (P0/P1 75%, P2/ELTP 50%)."""
    assert_jo_dict(jo)
    init = safe_int(jo.get("initial_demand"), 0)
    if init <= 0:
        return 0
    thr = saturation_threshold_for_jo(jo)
    return min(init, int(math.floor(thr * init + 1e-12)))


def _slots_remaining_under_share_cap(jo: dict) -> int:
    now = safe_int(jo.get("slots_allocated"), 0)
    return max(0, _max_slots_under_saturation_lifetime(jo) - now)


def _lateral_valid_for_slot(
    jo: dict,
    slot: Slot,
    caps: dict[str, int],
    lateral_counts: dict[str, int],
    *,
    enforce_positive_batch_cap: bool,
    skip_lifetime_share_cap: bool = False,
) -> tuple[bool, str]:
    assert_jo_dict(jo)
    if str(jo.get("type", "")).strip().upper() != JOType.LATERAL.value:
        return False, "not lateral"
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
    # In 60–40 phase, lifetime ratio is expressed by batch caps only — do not double-block top/next pools.
    if not skip_lifetime_share_cap and _slots_remaining_under_share_cap(jo) <= 0:
        return False, "lifetime share cap (Greedy/50–50 throttle)"
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
) -> tuple[dict | None, str, list[str]]:
    """
    Step 7: Route slot to top pool (JO-1/JO-2) or next pool (remainder leaders) by batch index; argmax final_score.
    No first-fit scan — only candidates allowed for that pool, then best score.
    """
    log: list[str] = [f"Slot {slot.slot_id} (batch position {slot_idx + 1}/{batch_n}):", ""]
    if phase_label == Phase.SPLIT_60_40:
        top_pool_sz, _ = _top_next_pool_sizes(batch_n)
        in_top = slot_idx < top_pool_sz
        log.append(
            f"Routing: {'Top pool (60% tier) — JO-1 & JO-2 only' if in_top else 'Next pool (40% tier) — next competing JO(s)'}"
        )
    else:
        in_top = True
        log.append(
            "Routing: Greedy (JO-1 only)"
            if phase_label == Phase.GREEDY
            else "Routing: 50–50 equal group (no top/next pool split by slot index)"
        )

    def score_for(jo: dict) -> float:
        return final_score(jo, slot.project)

    def describe_candidate(jo: dict) -> str:
        bd = score_breakdown(jo, slot.project)
        aff = bd["affinity_score"]
        return (
            f"{jo['id']} → base {bd['base_score']:.2f} + 0.10×{aff:.1f} = {bd['final_score']:.2f}"
        )

    skip_share = phase_label == Phase.SPLIT_60_40

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
                skip_lifetime_share_cap=skip_share,
            )[0]
        ]

    primary: list[dict] = []
    if phase_label == Phase.GREEDY:
        primary = ranked[:1]
    elif phase_label == Phase.SPLIT_50_50:
        primary = _equal_group_from_ranked(ranked)
    elif phase_label == Phase.SPLIT_60_40:
        # Strict pools: top tier = global JO-1 & JO-2; next tier = top two among ranked[2:] (score + delta caps).
        if in_top:
            primary = ranked[:2] if len(ranked) >= 2 else ranked[:1]
        else:
            primary = list(_next_tier_top_two(ranked))
    else:
        primary = ranked[:1]

    log.append("Candidates (restricted pool):")
    if not primary:
        log.append("  (none)")
    for jo in primary:
        ok, why = _lateral_valid_for_slot(
            jo,
            slot,
            caps,
            lateral_counts,
            enforce_positive_batch_cap=True,
            skip_lifetime_share_cap=skip_share,
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
    pick_kind = "pool"

    if not valid_p:
        log.append("No valid candidate in pool — no cross-pool / overflow bypass (strict).")
        feas = [
            jo
            for jo in primary
            if _match_tech_level(jo, slot) and safe_int(jo.get("active_demand"), 0) > 0
        ]
        if not feas:
            log.append("True infeasibility (tech mismatch or zero demand for pool JOs) → ELTP path.")
        else:
            log.append(
                "Pool JOs have tech+demand but none passed validation (e.g. batch cap exhausted) → ELTP path."
            )
        return None, "none", log

    best = max(valid_p, key=score_for)
    log.append(f"Selected: {best['id']} (best final_score in allowed set)")
    return best, pick_kind, log


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
        skip_lifetime_share_cap=False,
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
    After strict pool routing fails: try laterals in descending base score (eligible first).
    Priority is reflected in base_score — no separate tier ordering.
    """
    ordered_ids = sorted(lateral_ids, key=lambda jid: _lateral_sort_key(jos[jid]))
    for jid in ordered_ids:
        jo = jos[jid]
        if can_assign_lateral(jo, slot, caps, lateral_counts):
            return jo
    return None


def _lateral_assignment_reason(phase: str, affinity_match: bool, pick_kind: str) -> str:
    parts = [f"Lateral: best final_score after top/next pool routing ({pick_kind})"]
    if phase == Phase.SPLIT_50_50:
        parts.append("50–50 phase caps")
    elif phase == Phase.SPLIT_60_40:
        parts.append("60–40 phase caps (top pool + next pool)")
    else:
        parts.append("Greedy phase cap on dominant")
    if affinity_match:
        parts.append("project match (affinity)")
    return " | ".join(parts)


def _make_eltp_picker(jos: dict[str, dict]):
    rr = [0]

    def next_eltp() -> str | None:
        alive = sorted(
            jid
            for jid, j in jos.items()
            if str(j["type"]).strip().upper() == JOType.ELTP.value and safe_int(j.get("active_demand"), 0) > 0
        )
        if not alive:
            return None
        idx = rr[0] % len(alive)
        rr[0] += 1
        return alive[idx]

    return next_eltp


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
            print("|Δ| ≤ 0.040 (equal tier); final phase also depends on top1/top2 saturation — see Phase Decision.")
        else:
            print("\u2192 |Δ| > 0.040 \u2192 Greedy if top1 not saturated, else 60–40")
    else:
        print("Delta: n/a (single lateral competitor)")
        print("Threshold = 0.040")


def _lateral_jo_roster_lines(sorted_lateral_jos: list[dict], ref_project: str) -> list[str]:
    """Full score visibility at batch start (affinity uses ref_project for display only)."""
    lines: list[str] = ["Lateral JO roster (scores at batch start):", ""]
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
            "Phase rule: both top saturated → controlled split (Case A/B); "
            "else |Δ|≤0.04 → 50–50; else Greedy if top1 free; else Case B split."
        )
    else:
        lines.append("Delta: n/a (single or no lateral competitor)")
        lines.append("Threshold = 0.040")
    lines.append("")
    lines.append("Saturation (lateral, before batch):")
    lines.extend(lateral_sat_lines)
    lines.append("")
    lines.extend(roster_lines)
    lines.append("Phase Decision:")
    lines.append(f"Phase: {phase_label}")
    lines.append(f"Top1 Sat: {top1_sat}, Top2 Sat: {top2_sat}")
    lines.append(f"Delta: {delta_val}")
    if phase_label == Phase.SPLIT_60_40 and top_pool_dbg is not None and next_pool_dbg is not None:
        lines.append("60–40 Split:")
        lines.append(f"Top Pool: {top_pool_dbg}")
        lines.append(f"Next Pool: {next_pool_dbg}")
    lines.append("")
    lines.append("Batch cap vs demand (lateral) - effective_cap = min(batch_cap, active_demand) at batch start:")
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
    lines.append("Lateral allocation (this batch, demand-constrained):")
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
    lines.append(f"Total lateral assigned: {total_lateral}")
    lines.append(f"Leftover (batch_size - lateral assigned, => ELTP if > 0): {leftover}")
    lines.append("")
    lines.append("ELTP allocation (this batch, leftover only):")
    if not any(v > 0 for v in eltp_tally.values()):
        lines.append("  (none)")
    else:
        for jid in sorted(eltp_tally.keys()):
            if eltp_tally[jid] > 0:
                lines.append(f"  {jid} => {eltp_tally[jid]}")
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
    batches: list[list[Slot]] = []
    for i in range(0, len(slot_list), batch_size):
        batches.append(slot_list[i : i + batch_size])

    batch_debug: list[BatchDebugInfo] = []
    pick_eltp = _make_eltp_picker(jos)
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
        ) = _compute_phase_and_caps(eligible_batch, n)
        for jid in lateral_ids:
            caps.setdefault(jid, 0)
        lateral_counts: dict[str, int] = {jid: 0 for jid in lateral_ids}
        demand_at_batch_start: dict[str, int] = {
            jid: safe_int(jos[jid]["active_demand"], 0) for jid in lateral_ids
        }

        sat_map = {jid: saturation_pct(jos[jid]) for jid in jos}
        thresh_map = {jid: 100.0 * saturation_threshold_for_jo(jos[jid]) for jid in jos}
        base_snapshot = {jid: base_score(jos[jid]) for jid in lateral_ids}
        lateral_batch_counts: dict[str, int] = {jid: 0 for jid in lateral_ids}
        eltp_batch_counts: dict[str, int] = {
            jid: 0
            for jid, j in jos.items()
            if str(j.get("type", "")).strip().upper() == JOType.ELTP.value
        }

        lateral_sat_lines = []
        for jid in sorted(lateral_ids):
            j = jos[jid]
            thr = 100.0 * saturation_threshold_for_jo(j)
            if safe_int(j.get("slots_allocated"), 0) == 0:
                lateral_sat_lines.append(f"  {jid} = INACTIVE (threshold {thr:.0f}%)")
            else:
                sp = saturation_ratio(j) * 100.0
                lateral_sat_lines.append(f"  {jid} = {sp:.1f}% ACTIVE (threshold {thr:.0f}%)")

        print("\n" + "=" * 60)
        print(f"Batch ID: {bi}")
        print("- Lateral pool only: ranking, delta, caps (ELTP excluded) -")
        _print_delta_block(dom_id, top_bs, next_id, second_bs, delta_val)
        print("")
        print("Saturation (lateral JOs, start of batch):")
        for line in lateral_sat_lines:
            print(line)
        print("")
        print("Phase Decision:")
        print(f"Phase: {phase_label}")
        print(f"Top1 Sat: {top1_sat}, Top2 Sat: {top2_sat}")
        print(f"Delta: {delta_val}")
        if phase_label == Phase.SPLIT_60_40 and top_pool_dbg is not None and next_pool_dbg is not None:
            print("60–40 Split:")
            print(f"Top Pool: {top_pool_dbg}")
            print(f"Next Pool: {next_pool_dbg}")
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
        eltp_slots_this_batch = 0

        for idx, slot in enumerate(batch_slots):
            print("")
            print(f"Slot ID: {slot.slot_id}")
            print("- Per-slot scores (each lateral vs this slot) -")
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

            print("---- SLOT TRACE (pool route → filter → argmax final_score) ----")
            picked, pick_kind, trace_lines = _pick_best_lateral_for_slot(
                phase_label,
                eligible_batch,
                idx,
                n,
                slot,
                caps,
                lateral_counts,
            )
            for line in trace_lines:
                print(line)
            if phase_label == Phase.SPLIT_60_40:
                print(
                    "60–40 batch: top two share top pool; next pool = top two among remainder by score/delta (caps)."
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
                        f"pool routing had no winner → assigning {jid}"
                    )
                    lateral_counts[jid] += 1
                    lateral_batch_counts[jid] += 1
                    cumulative[jid] += 1
                    assigned_jo = jos[jid]
                    assigned_jo["slots_allocated"] = safe_int(assigned_jo.get("slots_allocated"), 0) + 1
                    assigned_jo["active_demand"] = max(0, safe_int(assigned_jo.get("active_demand"), 0) - 1)
                    fs = final_score(assigned_jo, slot.project)
                    aff = str(assigned_jo["project"]).strip() == slot.project.strip()
                    reason = (
                        "Lateral: structured score-order fallback (after pool route; before ELTP)"
                    )
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
                    print("Note: lateral demand > 0 but no lateral can_assign this slot (caps/tech/share).")
                if still_eligible:
                    print("ERROR: lateral available (can_assign) but none assigned (logic bug).")
                print("No lateral eligible → assigning to ELTP")
                eid = pick_eltp()
                if eid is not None:
                    eltp_slots_this_batch += 1
                    eltp_batch_counts[eid] = eltp_batch_counts[eid] + 1
                    cumulative[eid] += 1
                    assigned_jo = jos[eid]
                    assigned_jo["slots_allocated"] = safe_int(assigned_jo.get("slots_allocated"), 0) + 1
                    assigned_jo["active_demand"] = max(0, safe_int(assigned_jo.get("active_demand"), 0) - 1)
                    print(f"  Assigned JO: {eid} (ELTP — no lateral eligible for this slot)")
                    _print_jo_state("Updated", assigned_jo)
                    batch_slot_assignments[idx] = SlotAssignment(
                        slot_id=slot.slot_id,
                        jo_id=eid,
                        allocation_type="ELTP",
                        reason="ELTP (no lateral could assign; infeasibility fallback)",
                        final_score=None,
                        affinity_match=None,
                    )
                else:
                    print(f"  Unassigned - no ELTP capacity ({last_lateral_fail_reason})")
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
        print("- Lateral result vs demand -")
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
        print(f"Batch size: {n} | Total lateral assigned: {total_lateral_assigned}")
        print(f"Leftover = batch_size - lateral assigned = {n} - {total_lateral_assigned} = {leftover}")
        if leftover > 0:
            print(
                "  => ELTP only for slots where every lateral failed can_assign (not a priority shortcut)."
            )

        assert all(a is not None for a in batch_slot_assignments)
        assignments.extend(batch_slot_assignments)  # type: ignore[arg-type]

        print("")
        print(f"ELTP assigned (this batch): {eltp_slots_this_batch}")
        print("(ELTP only if no lateral could take the slot after full base-score order scan.)")

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
