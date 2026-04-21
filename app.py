"""
Slot Allocation Simulator — Streamlit UI.
Run: streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from engine import BATCH_SIZE, DELTA_EQUAL, run_allocation
from jo_utils import assert_jo_dict, safe_int
from metrics import build_metrics_styled_table, demand_fulfilled_pct, fairness_lateral
from models import Slot
from scoring import base_score

st.set_page_config(page_title="Slot Allocation Simulator", layout="wide")


def default_jos() -> list[dict]:
    return [
        {
            # JO-A: P0. High urgency. initial_demand=8 → sat cap = 6 slots (75%).
            # Dominates early. Saturates by end of Batch 2.
            "id": "JO-A-P0",
            "priority": 0,
            "days_remaining": 3,
            "total_duration": 30,
            "initial_demand": 8,
            "active_demand": 8,
            "project": "Alpha",
            "type": "LATERAL",
            "tech_stack": "*",
            "level": "*",
            "slots_allocated": 0,
        },
        {
            # JO-B: P0. Same priority as JO-A. days_remaining=4 vs JO-A's 3.
            # Initial base scores: JO-A=0.875, JO-B=0.858. Delta=0.017 ≤ 0.040.
            # This triggers 50-50 Equal in Batch 1 immediately.
            # initial_demand=8 → sat cap = 6 slots (75%).
            "id": "JO-B-P0",
            "priority": 0,
            "days_remaining": 4,
            "total_duration": 30,
            "initial_demand": 8,
            "active_demand": 8,
            "project": "Beta",
            "type": "LATERAL",
            "tech_stack": "*",
            "level": "*",
            "slots_allocated": 0,
        },
        {
            # JO-C: P1. Competes after JO-A and JO-B saturate.
            # initial_demand=8 → sat cap = 6 slots (75%).
            "id": "JO-C-P1",
            "priority": 1,
            "days_remaining": 5,
            "total_duration": 30,
            "initial_demand": 8,
            "active_demand": 8,
            "project": "Alpha",
            "type": "LATERAL",
            "tech_stack": "*",
            "level": "*",
            "slots_allocated": 0,
        },
        {
            # JO-D: P2. Low saturation cap (50% of 6 = 3 slots max).
            # Receives overflow once P0/P1 JOs are capped out.
            "id": "JO-D-P2",
            "priority": 2,
            "days_remaining": 6,
            "total_duration": 30,
            "initial_demand": 6,
            "active_demand": 6,
            "project": "Gamma",
            "type": "LATERAL",
            "tech_stack": "*",
            "level": "*",
            "slots_allocated": 0,
        },
        {
            # JO-E: P2. Same priority and same urgency as JO-D (both days_remaining=6) so base scores stay tied.
            # initial_demand=6 → sat cap = 3 slots (50%).
            "id": "JO-E-P2",
            "priority": 2,
            "days_remaining": 6,
            "total_duration": 30,
            "initial_demand": 6,
            "active_demand": 6,
            "project": "Delta",
            "type": "LATERAL",
            "tech_stack": "*",
            "level": "*",
            "slots_allocated": 0,
        },
        {
            # JO-F: ELTP. Large demand (30). Only receives slots when ALL
            # laterals have either exhausted demand or hit saturation.
            # With laterals capped at 6+6+6+3+3=24 total and 28 slots,
            # the last 4 slots must fall to ELTP.
            "id": "JO-F-ELTP",
            "priority": 2,
            "days_remaining": 20,
            "total_duration": 60,
            "initial_demand": 30,
            "active_demand": 30,
            "project": "Bench",
            "type": "ELTP",
            "tech_stack": "*",
            "level": "*",
            "slots_allocated": 0,
        },
    ]


def default_slots() -> list[Slot]:
    raw = [
        # Batch 1 — 50-50 Equal (JO-A and JO-B both P0, delta=0.017 ≤ 0.040)
        {"slot_id": "S01", "panel": "P01", "project": "Alpha", "batch": 1},
        {"slot_id": "S02", "panel": "P02", "project": "Beta", "batch": 1},
        {"slot_id": "S03", "panel": "P03", "project": "Alpha", "batch": 1},
        {"slot_id": "S04", "panel": "P04", "project": "Beta", "batch": 1},
        # Batch 2 — 50-50 Equal again (both still unsaturated, delta still ≤ 0.040)
        {"slot_id": "S05", "panel": "P05", "project": "Alpha", "batch": 2},
        {"slot_id": "S06", "panel": "P06", "project": "Beta", "batch": 2},
        {"slot_id": "S07", "panel": "P07", "project": "Alpha", "batch": 2},
        {"slot_id": "S08", "panel": "P08", "project": "Beta", "batch": 2},
        # Batch 3 — 50-50 then saturation gate mid-batch
        {"slot_id": "S09", "panel": "P09", "project": "Alpha", "batch": 3},
        {"slot_id": "S10", "panel": "P10", "project": "Beta", "batch": 3},
        {"slot_id": "S11", "panel": "P11", "project": "Alpha", "batch": 3},
        {"slot_id": "S12", "panel": "P12", "project": "Beta", "batch": 3},
        # Batch 4 — 60-40 phase (dominant saturated; competing JO gets remainder)
        {"slot_id": "S13", "panel": "P13", "project": "Alpha", "batch": 4},
        {"slot_id": "S14", "panel": "P14", "project": "Beta", "batch": 4},
        {"slot_id": "S15", "panel": "P15", "project": "Gamma", "batch": 4},
        {"slot_id": "S16", "panel": "P16", "project": "Delta", "batch": 4},
        # Batch 5 — 60-40 continues until P0/P1 are all saturated
        {"slot_id": "S17", "panel": "P17", "project": "Alpha", "batch": 5},
        {"slot_id": "S18", "panel": "P18", "project": "Beta", "batch": 5},
        {"slot_id": "S19", "panel": "P19", "project": "Gamma", "batch": 5},
        {"slot_id": "S20", "panel": "P20", "project": "Delta", "batch": 5},
        # Batch 6 — P2 50-50 Equal (JO-D-P2 and JO-E-P2 top two, delta ≤ 0.040)
        {"slot_id": "S21", "panel": "P21", "project": "Gamma", "batch": 6},
        {"slot_id": "S22", "panel": "P22", "project": "Delta", "batch": 6},
        {"slot_id": "S23", "panel": "P23", "project": "Gamma", "batch": 6},
        {"slot_id": "S24", "panel": "P24", "project": "Delta", "batch": 6},
        # Batch 7 — ELTP fallback (all laterals saturated or demand = 0)
        {"slot_id": "S25", "panel": "P25", "project": "Bench", "batch": 7},
        {"slot_id": "S26", "panel": "P26", "project": "Bench", "batch": 7},
        {"slot_id": "S27", "panel": "P27", "project": "Bench", "batch": 7},
        {"slot_id": "S28", "panel": "P28", "project": "Bench", "batch": 7},
    ]
    out: list[Slot] = []
    for row in raw:
        out.append(
            Slot(
                slot_id=row["slot_id"],
                panel_id=row["panel"],
                project=row["project"],
                tech_stack="*",
                level="*",
                batch_id=str(row["batch"]),
            )
        )
    return out


def _coerce_priority(val) -> str | int:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 3
    if isinstance(val, (float, int)) and not isinstance(val, bool):
        if isinstance(val, float) and val != int(val):
            return str(val).strip()
        return int(val)
    return str(val).strip()


def jos_from_df(df: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    for _, r in df.iterrows():
        jid = str(r.get("id") or "").strip()
        if not jid:
            continue
        ad = safe_int(r.get("active_demand"), 0)
        ini = safe_int(r.get("initial_demand"), ad)
        jo = {
            "id": jid,
            "priority": _coerce_priority(r.get("priority")),
            "days_remaining": safe_int(r.get("days_remaining"), 0),
            "total_duration": safe_int(r.get("total_duration"), 0),
            "initial_demand": ini,
            "active_demand": ad,
            "project": str(r.get("project") or "").strip(),
            "type": str(r.get("type") or "LATERAL").strip().upper(),
            "tech_stack": str(r.get("tech_stack") or "*").strip() or "*",
            "level": str(r.get("level") or "*").strip() or "*",
            "slots_allocated": safe_int(r.get("slots_allocated"), 0),
        }
        assert_jo_dict(jo)
        out.append(jo)
    if not out:
        raise ValueError("No valid job openings: each row needs a non-empty id.")
    return out


def slots_from_df(df: pd.DataFrame) -> list[Slot]:
    df = df.copy()
    if "batch" in df.columns:
        df["_b"] = df["batch"].apply(lambda x: safe_int(x, 0))
        df = df.sort_values(by=["_b", "slot_id"], na_position="last")
    elif "batch_id" in df.columns:
        df["_b"] = df["batch_id"].apply(lambda x: safe_int(x, 0))
        df = df.sort_values(by=["_b", "slot_id"], na_position="last")

    out: list[Slot] = []
    for _, r in df.iterrows():
        sid = str(r.get("slot_id") or "").strip()
        if not sid:
            continue
        panel = r.get("panel_id")
        if panel is None or (isinstance(panel, float) and pd.isna(panel)):
            panel = r.get("panel")
        batch_val = r.get("batch_id")
        if batch_val is None or (isinstance(batch_val, float) and pd.isna(batch_val)):
            batch_val = r.get("batch")
        out.append(
            Slot(
                slot_id=sid,
                panel_id=str(panel or "").strip(),
                project=str(r.get("project") or "").strip(),
                tech_stack=str(r.get("tech_stack") or "*").strip() or "*",
                level=str(r.get("level") or "*").strip() or "*",
                batch_id=str(safe_int(batch_val, 0)),
            )
        )
    if not out:
        raise ValueError("No valid slots: each row needs a non-empty slot_id.")
    return out


def format_slot_breakdown(res, slot_id: str) -> str:
    lines: list[str] = [f"Slot {slot_id}:", ""]
    jids = sorted(res.jo_snapshots_end.keys())
    for jid in jids:
        key = (slot_id, jid)
        if key not in res.score_breakdown_by_slot_jo:
            continue
        b = res.score_breakdown_by_slot_jo[key]
        lines.append(f"{jid}:")
        lines.append(f"  Base: {b['base_score']:.4f}")
        lines.append(f"  Affinity: {b['affinity_score']:.1f}")
        lines.append(f"  Final: {b['final_score']:.4f}")
        lines.append(f"  Priority Score: {b['priority_score']:.6f}")
        lines.append(f"  Urgency Score:  {b['urgency_score']:.6f}")
        lines.append(f"  Demand Score:   {b['demand_score']:.6f}  (active / initial demand)")
        lines.append(f"  Active Demand (at slot): {int(b['active_demand'])}")
        lines.append(f"  Slots Allocated (at slot): {int(b['slots_allocated'])}")
        sat_disp = "INACTIVE" if int(b["slots_allocated"]) == 0 else f"{100.0 * b['saturation_ratio']:.1f}%"
        lines.append(f"  Saturation: {sat_disp}")
        lines.append("")
    return "\n".join(lines).rstrip()


st.title("Slot Allocation Simulator")
st.caption(
    "Per slot: try every lateral in base-score order (can_assign: demand, cap, saturation, tech/level; "
    "affinity is not a gate) → only if none can take the slot, assign ELTP. "
    f"Delta equal threshold = {DELTA_EQUAL}; batch size = {BATCH_SIZE}. "
    "Final score = base + 0.10 × affinity (preference only). "
    "Check the terminal for ROW / batch / assignment prints."
)

tab_inputs, tab_batch, tab_alloc, tab_scores, tab_metrics = st.tabs(
    ["1. Inputs", "2. Batch debug", "3. Allocation", "4. Score breakdown", "5. Metrics"]
)

with tab_inputs:
    st.subheader("Job openings")
    df_jo = pd.DataFrame(default_jos())
    edited_jo = st.data_editor(
        df_jo,
        num_rows="dynamic",
        use_container_width=True,
        key="jo_editor",
    )

    st.subheader("Slots")
    df_sl = pd.DataFrame(
        [
            {
                "slot_id": s.slot_id,
                "panel": s.panel_id,
                "project": s.project,
                "batch": safe_int(s.batch_id, 0) if s.batch_id else 0,
                "tech_stack": s.tech_stack,
                "level": s.level,
            }
            for s in default_slots()
        ]
    )
    edited_sl = st.data_editor(
        df_sl,
        num_rows="dynamic",
        use_container_width=True,
        key="slot_editor",
    )

    batch_n = st.number_input("Slots per batch", min_value=1, max_value=32, value=BATCH_SIZE, step=1)

    run = st.button("Run simulation", type="primary")

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if run:
    try:
        jos = jos_from_df(edited_jo)
        slots = slots_from_df(edited_sl)
        st.session_state.last_result = run_allocation(jos, slots, batch_size=int(batch_n))
        st.session_state.jo_input_df = edited_jo.copy()
        st.session_state.slot_input_df = edited_sl.copy()
    except Exception as e:
        st.error(f"Simulation error: {e}")
        import traceback

        st.code(traceback.format_exc())
        st.session_state.last_result = None

res = st.session_state.last_result

with tab_batch:
    st.subheader("Batch logs (mandatory transparency)")
    if not res:
        st.info("Run the simulation from the Inputs tab.")
    else:
        for bd in res.batch_debug:
            with st.expander(
                f"Batch {bd.batch_index + 1} — slots: {', '.join(bd.slot_ids)}",
                expanded=bd.batch_index == 0,
            ):
                st.code("\n".join(bd.debug_log_lines), language=None)
                st.markdown("**Saturation threshold % (rule)**")
                st.json({k: round(v, 2) for k, v in bd.saturation_threshold_pct.items()})
                st.markdown("**Allocation this batch — lateral**")
                st.json(bd.lateral_assigned_this_batch or {})
                st.markdown("**Allocation this batch — ELTP**")
                st.json(bd.eltp_assigned_this_batch or {})
                st.markdown("**Cumulative slots assigned**")
                st.json(dict(sorted(bd.cumulative_slots_assigned.items())))
                st.markdown("**Lateral ratios after batch (slots / initial_demand)**")
                st.json({k: round(v, 4) for k, v in bd.lateral_ratios_after_batch.items()})
                st.metric("Fairness (lateral, end of batch)", f"{bd.fairness_lateral_batch:.4f}")

with tab_alloc:
    st.subheader("Slot -> assignment")
    if not res:
        st.info("Run the simulation from the Inputs tab.")
    else:
        rows = []
        for a in res.assignments:
            rows.append(
                {
                    "slot_id": a.slot_id,
                    "assigned_jo": a.jo_id,
                    "type": a.allocation_type,
                    "reason": a.reason,
                    "score": a.final_score,
                    "affinity": a.affinity_match,
                }
            )
        df_alloc = pd.DataFrame(rows)
        try:
            df_ad = df_alloc.copy()
            if df_ad["score"].notna().any():
                df_ad["score"] = pd.to_numeric(df_ad["score"], errors="coerce").round(1)
            st.dataframe(df_ad, use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(df_alloc, use_container_width=True, hide_index=True)
        eltp_used = df_alloc["type"].eq("ELTP").any()
        lateral_demand_left = any(
            str(j.get("type", "")).strip().upper() == "LATERAL"
            and safe_int(j.get("active_demand"), 0) > 0
            for j in res.jo_snapshots_end.values()
        )
        if eltp_used and lateral_demand_left:
            st.warning(
                "If ELTP used while lateral demand exists -> logic issue (verify caps, tech/level, saturation per slot)."
            )

with tab_scores:
    st.subheader("Full score calculation (every JO × every slot)")
    if not res:
        st.info("Run the simulation from the Inputs tab.")
    else:
        slot_ids = list(dict.fromkeys(a.slot_id for a in res.assignments))
        for sid in slot_ids:
            with st.expander(f"Slot {sid}", expanded=False):
                st.code(format_slot_breakdown(res, sid), language=None)

        st.subheader("Base score — initial vs end-of-run")
        end_bs = pd.Series({jid: base_score(j) for jid, j in res.jo_snapshots_end.items()}).sort_values(ascending=False)
        ini_bs = pd.Series(res.jo_base_scores_initial).sort_values(ascending=False)
        st.dataframe(
            pd.DataFrame({"base_initial": ini_bs, "base_final": end_bs}),
            use_container_width=True,
        )

with tab_metrics:
    st.subheader("Saturation & demand")
    if not res:
        st.info("Run the simulation from the Inputs tab.")
    else:
        end = res.jo_snapshots_end
        dom_id = res.lateral_dominant_id_initial
        comp_id = res.lateral_competing_id_initial
        mrows = []
        for jid, j in end.items():
            slots_al = safe_int(j.get("slots_allocated"), 0)
            sat_status = "INACTIVE" if slots_al == 0 else "ACTIVE"
            is_lat = str(j.get("type", "")).strip().upper() == "LATERAL"
            is_dom = bool(is_lat and dom_id and jid == dom_id)
            is_comp = bool(is_lat and comp_id and jid == comp_id)
            if is_dom:
                role = "Dominant"
            elif is_comp:
                role = "Competing"
            else:
                role = ""
            ini = safe_int(j.get("initial_demand"), 0)
            sat_pct_display = None if slots_al == 0 else (round(100.0 * slots_al / ini, 2) if ini > 0 else None)
            mrows.append(
                {
                    "jo_id": jid,
                    "type": j["type"],
                    "role": role,
                    "base_score": round(base_score(j), 4),
                    "initial_demand": ini,
                    "saturation_status": sat_status,
                    "saturation_%": sat_pct_display,
                    "is_dominant": is_dom,
                    "is_competing": is_comp,
                    "demand_fulfilled_%": round(demand_fulfilled_pct(j), 1),
                    "slots_allocated": slots_al,
                    "active_demand_remaining": safe_int(j.get("active_demand"), 0),
                }
            )
        df_m = pd.DataFrame(mrows)
        df_disp = df_m.rename(
            columns={
                "saturation_%": "🔥 saturation %",
                "demand_fulfilled_%": "📉 demand fulfilled %",
                "slots_allocated": "🎯 slots",
                "base_score": "📊 base score",
                "active_demand_remaining": "📉 demand left",
            }
        )
        for col in ("🔥 saturation %", "📉 demand fulfilled %"):
            if col in df_disp.columns:
                df_disp[col] = pd.to_numeric(df_disp[col], errors="coerce").fillna(0).round(1)
        if "initial_demand" in df_disp.columns:
            df_disp["initial_demand"] = pd.to_numeric(df_disp["initial_demand"], errors="coerce").fillna(0).round(1)
        if "📉 demand left" in df_disp.columns:
            df_disp["📉 demand left"] = (
                pd.to_numeric(df_disp["📉 demand left"], errors="coerce").fillna(0).round(1)
            )
        col_order = [
            "jo_id",
            "type",
            "role",
            "📊 base score",
            "🎯 slots",
            "📉 demand left",
            "initial_demand",
            "saturation_status",
            "🔥 saturation %",
            "📉 demand fulfilled %",
            "is_dominant",
            "is_competing",
        ]
        df_disp = df_disp[[c for c in col_order if c in df_disp.columns] + [c for c in df_disp.columns if c not in col_order]]

        try:
            st.dataframe(build_metrics_styled_table(df_disp), use_container_width=True, hide_index=True)
        except Exception:
            st.dataframe(df_m, use_container_width=True, hide_index=True)
        st.caption(
            "Base score uses live active_demand (25% weight), so it drops each time that JO receives a slot. "
            "When the dominant hits saturation, overflow tries the next lateral(s); if the top two competitors "
            f"have |Δ base| ≤ {DELTA_EQUAL}, their try-order alternates each slot."
        )

        st.subheader("Distribution fairness (lateral)")
        ff = fairness_lateral(end)
        st.metric("Fairness score (higher = more even lateral fulfillment)", f"{ff:.3f}")
        st.caption(
            "Lateral fairness = 1 − (max(ratio) − min(ratio)) with ratio = slots_allocated / initial_demand."
        )
