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
from models import SimulationResult, Slot
from scoring import base_score

st.set_page_config(page_title="Slot Allocation Simulator", layout="wide")


def default_jos() -> list[dict]:
    return [
        {
            "id": "JO-A-P0",
            "priority": 0,
            "days_remaining": 10,
            "total_duration": 60,
            "initial_demand": 15,
            "active_demand": 15,
            "project": "Alpha",
            "type": "LATERAL",
            "tech_stack": "*",
            "level": "*",
            "slots_allocated": 0,
        },
        {
            "id": "JO-B-P0",
            "priority": 0,
            "days_remaining": 15,
            "total_duration": 60,
            "initial_demand": 15,
            "active_demand": 15,
            "project": "Beta",
            "type": "LATERAL",
            "tech_stack": "*",
            "level": "*",
            "slots_allocated": 0,
        },
        {
            "id": "JO-C-P1",
            "priority": 1,
            "days_remaining": 30,
            "total_duration": 60,
            "initial_demand": 8,
            "active_demand": 8,
            "project": "Alpha",
            "type": "LATERAL",
            "tech_stack": "*",
            "level": "*",
            "slots_allocated": 0,
        },
        {
            "id": "JO-D-P1",
            "priority": 1,
            "days_remaining": 30,
            "total_duration": 60,
            "initial_demand": 8,
            "active_demand": 8,
            "project": "Gamma",
            "type": "LATERAL",
            "tech_stack": "*",
            "level": "*",
            "slots_allocated": 0,
        },
        {
            "id": "JO-E-P2",
            "priority": 2,
            "days_remaining": 60,
            "total_duration": 120,
            "initial_demand": 4,
            "active_demand": 4,
            "project": "Delta",
            "type": "LATERAL",
            "tech_stack": "*",
            "level": "*",
            "slots_allocated": 0,
        },
        {
            "id": "JO-F-P2",
            "priority": 2,
            "days_remaining": 60,
            "total_duration": 120,
            "initial_demand": 4,
            "active_demand": 4,
            "project": "Gamma",
            "type": "LATERAL",
            "tech_stack": "*",
            "level": "*",
            "slots_allocated": 0,
        },
        {
            "id": "JO-G-ELTP",
            "priority": 2,
            "days_remaining": 120,
            "total_duration": 120,
            "initial_demand": 40,
            "active_demand": 40,
            "project": "Bench",
            "type": "ELTP",
            "tech_stack": "*",
            "level": "*",
            "slots_allocated": 0,
        },
    ]


def default_slots() -> list[Slot]:
    projects = ["Alpha", "Beta", "Gamma", "Delta", "Bench"]
    out: list[Slot] = []
    for i in range(1, 51):
        proj = projects[(i - 1) % len(projects)]
        out.append(
            Slot(
                slot_id=f"S{i:02d}",
                panel_id=f"P{i:02d}",
                project=proj,
                tech_stack="*",
                level="*",
                batch_id="1",
            )
        )
    return out


def parse_batch_sizes(raw: str) -> list[int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    sizes: list[int] = []
    for part in parts:
        try:
            val = int(part)
        except ValueError:
            continue
        if val > 0:
            sizes.append(val)
    return sizes


def run_dynamic_simulation(
    jos: list[dict], slots: list[Slot], batch_sizes: list[int]
) -> tuple[SimulationResult, list[int]]:
    if not batch_sizes:
        raise ValueError("Dynamic batch sizes must include at least one positive integer.")
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

    combined = SimulationResult(
        assignments=assignments,
        jo_snapshots_end={jo["id"]: jo for jo in jos},
        batch_debug=batch_debug,
        jo_base_scores_initial=base_initial,
        score_breakdown_by_slot_jo=breakdown,
        jo_final_score_matrix=final_matrix,
        lateral_dominant_id_initial=dom_id_initial,
        lateral_competing_id_initial=comp_id_initial,
    )
    return combined, used_sizes


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
    "Per slot: try every candidate in base-score order (demand, cap, tech/level; "
    "affinity is not a gate). "
    f"Delta equal threshold = {DELTA_EQUAL}; default batch size = {BATCH_SIZE}. "
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

    use_dynamic_batches = st.checkbox("Use dynamic batch sizes", value=True)
    batch_list_default = "4, 10, 2, 15, 40, 8, 6, 5, 10"
    batch_list_raw = st.text_input("Dynamic batch sizes (comma-separated)", value=batch_list_default)
    batch_n = st.number_input("Slots per batch (static)", min_value=1, max_value=100, value=BATCH_SIZE, step=1)

    run = st.button("Run simulation", type="primary")

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if run:
    try:
        jos = jos_from_df(edited_jo)
        slots = slots_from_df(edited_sl)
        if use_dynamic_batches:
            batch_sizes = parse_batch_sizes(batch_list_raw)
            res, used_sizes = run_dynamic_simulation(jos, slots, batch_sizes)
            st.session_state.last_result = res
            st.session_state.batch_sizes_used = used_sizes
        else:
            st.session_state.last_result = run_allocation(jos, slots, batch_size=int(batch_n))
            st.session_state.batch_sizes_used = None
        st.session_state.jo_input_df = edited_jo.copy()
        st.session_state.slot_input_df = edited_sl.copy()
    except Exception as e:
        st.error(f"Simulation error: {e}")
        import traceback

        st.code(traceback.format_exc())
        st.session_state.last_result = None

res = st.session_state.last_result
batch_sizes_used = st.session_state.get("batch_sizes_used")
if res:
    total_slots = len(res.assignments)
    allocated_slots = sum(1 for a in res.assignments if a.jo_id != "-")
    pending_slots = max(0, total_slots - allocated_slots)

with tab_batch:
    st.subheader("Batch logs (mandatory transparency)")
    if not res:
        st.info("Run the simulation from the Inputs tab.")
    else:
        st.markdown(
            f"**Pending Slots:** {pending_slots}  \n"
            f"**Allocated Slots:** {allocated_slots}  \n"
            f"**Total Slots:** {total_slots}"
        )
        for bd in res.batch_debug:
            pending_after = None
            if batch_sizes_used is not None and bd.batch_index < len(batch_sizes_used):
                pending_after = max(0, total_slots - sum(batch_sizes_used[: bd.batch_index + 1]))
            with st.expander(
                f"Batch {bd.batch_index + 1} — slots: {', '.join(bd.slot_ids)}",
                expanded=bd.batch_index == 0,
            ):
                if pending_after is not None:
                    st.markdown(f"**Running Pending Slots After Batch:** {pending_after}")
                st.code("\n".join(bd.debug_log_lines), language=None)
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
