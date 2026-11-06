"""
UrbanPulse — AI-Based Urban Traffic Command & Incident Response System
Main Streamlit application entry point.

Architecture: LangGraph multi-agent pipeline
LLM: Google Gemini 1.5 Flash
UI: Streamlit

Run: streamlit run app.py
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

import time as _time
from utils.config import settings
from utils.db import init_db, get_db
from utils.audit_logger import log_action
from tools.sensor_tools import simulate_sensor_reading, get_all_zone_statuses
from agents.supervisor_agent import invoke_pipeline, resume_pipeline
from utils.guardrails import (
    run_all_input_guardrails,
    run_all_output_guardrails,
    record_token_usage,
)
from utils.performance_dashboard import render_performance_dashboard

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title=settings.APP_TITLE,
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DB initialisation (once per session) ─────────────────────────────────────

# ── Session state defaults ────────────────────────────────────────────────────
if "operator_id" not in st.session_state:
    st.session_state["operator_id"] = f"OP-{uuid.uuid4().hex[:6].upper()}"

if "active_incidents" not in st.session_state:
    st.session_state["active_incidents"] = {}   # session_id → TrafficState


def _seed_zones() -> None:
    """Populate the zones table with 10 demo zones if empty."""
    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) FROM zones").fetchone()[0]
        if count == 0:
            zones = [
                (f"Z-{i:02d}", f"Zone {i:02d}", 12.9716 + i * 0.01, 77.5946 + i * 0.01, 4, "arterial")
                for i in range(1, 11)
            ]
            conn.executemany(
                "INSERT INTO zones (zone_id, zone_name, latitude, longitude, intersections, road_type) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                zones,
            )
        # Seed some sensor readings if none exist
        reading_count = conn.execute("SELECT COUNT(*) FROM sensor_readings").fetchone()[0]
        if reading_count == 0:
            import random
            from datetime import timedelta
            now = datetime.now(timezone.utc)
            rows = []
            for i in range(1, 11):
                zid = f"Z-{i:02d}"
                for h in range(24):
                    ts = (now - timedelta(hours=h)).isoformat()
                    density = round(random.triangular(0.05, 1.0, 0.35), 2)
                    speed = round(max(2.0, random.gauss(45, 20)), 1)
                    count = int(density * 250)
                    rows.append((zid, ts, density, speed, count, "simulated"))
            conn.executemany(
                "INSERT INTO sensor_readings (zone_id, timestamp, density, avg_speed_kmh, vehicle_count, sensor_source) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                rows,
            )


if "db_initialized" not in st.session_state:
    init_db()
    _seed_zones()
    st.session_state["db_initialized"] = True



# ── Helper: severity badge ────────────────────────────────────────────────────
def _severity_badge(label: str) -> str:
    icons = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
    return f"{icons.get(label, '⚪')} {label}"


def _status_badge(density: float, speed: float) -> str:
    if density > 0.85 or speed < 10:
        return "🔴 CRITICAL"
    if density > 0.75 or speed < 20:
        return "🟠 HIGH"
    if density > 0.55 or speed < 35:
        return "🟡 MEDIUM"
    return "🟢 NORMAL"


# ── Responsible AI banner ─────────────────────────────────────────────────────
st.markdown(
    """
    <div style="background:#fff3cd;border:1px solid #ffc107;border-radius:6px;
                padding:8px 14px;margin-bottom:12px;font-size:13px;color:#1a237e;">
    ⚠️ <strong>Responsible AI Notice:</strong> UrbanPulse provides AI-assisted
    decision support only. All CRITICAL incidents require human operator approval
    before action. AI recommendations are not verified facts. This is a prototype.
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚦 UrbanPulse — Traffic Command & Incident Response")
st.caption(f"v{settings.APP_VERSION} | Operator: {st.session_state['operator_id']} | {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺️ Live Zone Monitor",
    "🚨 Incident Queue",
    "📊 Analytics",
    "📋 Audit Log",
    "⚙️ System Health",
    "📊 Performance Dashboard",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE ZONE MONITOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Live Zone Status")

    col_refresh, col_trigger, col_zone = st.columns([1, 2, 2])
    with col_refresh:
        if st.button("🔄 Refresh Readings"):
            st.rerun()
    with col_trigger:
        zone_options = [f"Z-{i:02d}" for i in range(1, 11)]
        selected_zone = st.selectbox("Select Zone", zone_options, key="trigger_zone")
    with col_zone:
        scenario_options = ["auto", "normal", "congestion", "accident"]
        selected_scenario = st.selectbox("Simulation Scenario", scenario_options, key="sim_scenario")

    if st.button("🚨 Trigger AI Investigation", type="primary"):
        # ── Input Guardrails ────────────────────────────────────────────
        zone_guard = run_all_input_guardrails(selected_zone, session_id=selected_zone)
        if zone_guard.blocked:
            st.error(f"⚠️ Input blocked: {', '.join(zone_guard.violations)}")
        else:
         with st.spinner(f"Investigating Zone {selected_zone}..."):
            simulate_sensor_reading(selected_zone, scenario=selected_scenario if selected_scenario != "auto" else None)
            session_id = str(uuid.uuid4())
            _start = _time.time()
            result = invoke_pipeline(selected_zone, session_id=session_id)
            _latency = (_time.time() - _start) * 1000
            record_token_usage(50, 300, cost_usd=0.0003)
            if "session_metrics" not in st.session_state:
                st.session_state["session_metrics"] = {}
            st.session_state["session_metrics"]["avg_latency_ms"] = _latency
            st.session_state["active_incidents"][session_id] = result

            incident = result.get("incident")
            if incident:
                severity = incident["severity_label"]
                badge = _severity_badge(severity)
                if severity == "CRITICAL":
                    st.error(f"🔴 CRITICAL INCIDENT detected in Zone {selected_zone}! Approval required.")
                elif severity == "HIGH":
                    st.warning(f"🟠 HIGH severity incident in Zone {selected_zone}.")
                else:
                    st.info(f"{badge} incident in Zone {selected_zone}.")
                st.info("→ See **🚨 Incident Queue** tab for full details and HITL approval.")
            else:
                st.success(f"✅ Zone {selected_zone} — No significant anomaly detected.")

    st.divider()
    st.subheader("Zone Status Grid")

    zone_statuses = get_all_zone_statuses()
    if not zone_statuses:
        st.info("No sensor readings available. Trigger investigations above.")
    else:
        cols = st.columns(5)
        for idx, zs in enumerate(zone_statuses):
            col = cols[idx % 5]
            badge = _status_badge(zs["density"], zs["avg_speed_kmh"])
            col.metric(
                label=zs["zone_id"],
                value=badge,
                delta=f"⚡ {zs['avg_speed_kmh']} km/h | 📦 {zs['density']:.0%}",
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — INCIDENT QUEUE + HITL
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Active Incident Queue")

    active_incidents = st.session_state.get("active_incidents", {})

    if not active_incidents:
        st.info("No active incidents this session. Trigger investigations from the Live Zone Monitor.")
    else:
        # Sort by severity descending
        sorted_incidents = sorted(
            active_incidents.items(),
            key=lambda x: x[1].get("incident", {}).get("severity_score", 0) if x[1].get("incident") else 0,
            reverse=True,
        )

        for session_id, state in sorted_incidents:
            incident = state.get("incident")
            plan = state.get("response_plan")

            if not incident:
                continue

            severity_label = incident.get("severity_label", "LOW")
            badge = _severity_badge(severity_label)

            with st.expander(
                f"{badge} | Zone {incident['zone_id']} | {incident['classification']} "
                f"| Score: {incident['severity_score']}/10 | {incident['detected_at'][:19]}",
                expanded=(severity_label in ("CRITICAL", "HIGH")),
            ):
                c1, c2 = st.columns(2)

                with c1:
                    st.markdown("**🔍 Incident Details**")
                    st.json({
                        "incident_id": incident["incident_id"][:8] + "...",
                        "zone": incident["zone_id"],
                        "classification": incident["classification"],
                        "severity": f"{incident['severity_score']}/10 ({severity_label})",
                        "detected_at": incident["detected_at"][:19],
                    })

                with c2:
                    st.markdown("**🌤️ Context**")
                    wx = incident.get("weather_context", {})
                    st.json({
                        "weather": wx.get("description", "N/A"),
                        "is_adverse": wx.get("is_adverse", False),
                        "rag_sources": incident.get("rag_sources", []),
                    })

                st.markdown("**📋 Evidence**")
                for ev in incident.get("evidence", []):
                    st.markdown(f"- {ev}")

                with st.expander("🧠 AI Reasoning (Explainability)", expanded=False):
                    st.markdown(f"> {incident.get('reasoning', 'No reasoning provided.')}")
                    st.markdown("**Agent Trace:**")
                    for step in state.get("agent_trace", []):
                        st.markdown(
                            f"- `{step.get('agent')}` → **{step.get('action')}** "
                            f"@ {str(step.get('timestamp', ''))[:19]}"
                        )

                if plan:
                    st.markdown("**🗺️ Response Plan**")
                    for route in plan.get("diversion_routes", []):
                        st.markdown(
                            f"  - 🔀 **{route['route_id']}**: {route['description']} "
                            f"(+{route['estimated_delay_min']} min)"
                        )
                    for sig in plan.get("signal_changes", []):
                        st.markdown(f"  - 🚦 **{sig['intersection_id']}**: `{sig['action']}`")

                    if plan.get("commuter_alert"):
                        st.info(f"📢 **Commuter Alert:** {plan['commuter_alert']}")
                    if plan.get("authority_alert"):
                        st.warning(f"🚨 **Authority Alert:** {plan['authority_alert']}")

                    # ── HITL Gate ─────────────────────────────────────────────
                    if state.get("awaiting_human"):
                        st.error("⛔ **CRITICAL: Human Approval Required**")
                        st.markdown("This response plan requires operator authorization before execution.")
                        col_approve, col_reject, col_notes = st.columns([1, 1, 2])
                        with col_notes:
                            notes = st.text_input(
                                "Operator Notes (optional)",
                                key=f"notes_{session_id}",
                            )
                        with col_approve:
                            if st.button("✅ APPROVE", key=f"approve_{session_id}", type="primary"):
                                hitl_decision = {
                                    "decision_id": str(uuid.uuid4()),
                                    "plan_id": plan["plan_id"],
                                    "operator_id": st.session_state["operator_id"],
                                    "decision": "APPROVED",
                                    "notes": notes,
                                    "decided_at": datetime.now(timezone.utc).isoformat(),
                                }
                                log_action(
                                    "HITL",
                                    "PLAN_APPROVED",
                                    entity_type="response_plan",
                                    entity_id=plan["plan_id"],
                                    payload=hitl_decision,
                                    human_actor=st.session_state["operator_id"],
                                    session_id=session_id,
                                )
                                final_state = resume_pipeline(session_id, hitl_decision)
                                st.session_state["active_incidents"][session_id] = final_state
                                st.success("✅ Response plan APPROVED and executed.")
                                st.rerun()

                        with col_reject:
                            if st.button("❌ REJECT", key=f"reject_{session_id}"):
                                hitl_decision = {
                                    "decision_id": str(uuid.uuid4()),
                                    "plan_id": plan["plan_id"],
                                    "operator_id": st.session_state["operator_id"],
                                    "decision": "REJECTED",
                                    "notes": notes,
                                    "decided_at": datetime.now(timezone.utc).isoformat(),
                                }
                                log_action(
                                    "HITL",
                                    "PLAN_REJECTED",
                                    entity_type="response_plan",
                                    entity_id=plan["plan_id"],
                                    payload=hitl_decision,
                                    human_actor=st.session_state["operator_id"],
                                    session_id=session_id,
                                )
                                st.warning("⛔ Response plan REJECTED. Incident escalated.")
                                st.rerun()
                    else:
                        decision = state.get("hitl_decision")
                        if decision:
                            if decision.get("decision") == "APPROVED":
                                st.success(f"✅ Approved by {decision.get('operator_id')}")
                            else:
                                st.error(f"❌ Rejected by {decision.get('operator_id')}")
                        else:
                            st.success("✅ Auto-approved (severity < CRITICAL)")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Incident Analytics")

    with get_db() as conn:
        inc_rows = conn.execute("SELECT * FROM incidents ORDER BY detected_at DESC LIMIT 100").fetchall()
        sr_rows = conn.execute(
            "SELECT zone_id, AVG(density) as avg_density, AVG(avg_speed_kmh) as avg_speed "
            "FROM sensor_readings GROUP BY zone_id ORDER BY zone_id"
        ).fetchall()

    if not inc_rows:
        st.info("No incident history yet. Trigger some investigations first.")
    else:
        import pandas as pd

        df = pd.DataFrame([dict(r) for r in inc_rows])

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Incidents", len(df))
        c2.metric("CRITICAL", len(df[df["severity_label"] == "CRITICAL"]))
        c3.metric("Avg Severity", f"{df['severity_score'].mean():.1f}/10")

        c4, c5 = st.columns(2)
        with c4:
            st.markdown("**Incidents by Classification**")
            st.bar_chart(df["classification"].value_counts())
        with c5:
            st.markdown("**Incidents by Severity Label**")
            st.bar_chart(df["severity_label"].value_counts())

        st.markdown("**Recent Incidents Table**")
        st.dataframe(
            df[["incident_id", "zone_id", "classification", "severity_score", "severity_label", "status", "detected_at"]]
            .head(20)
            .rename(columns={"incident_id": "ID", "zone_id": "Zone"}),
            use_container_width=True,
        )

    if sr_rows:
        st.divider()
        st.markdown("**Zone Traffic Overview (Averages)**")
        df_sr = pd.DataFrame([dict(r) for r in sr_rows])
        st.bar_chart(df_sr.set_index("zone_id")[["avg_density", "avg_speed"]])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — AUDIT LOG
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Audit Log")
    st.caption("Immutable record of all agent actions, tool calls, and human decisions.")

    from utils.audit_logger import get_audit_log
    import pandas as pd

    col_filter, col_limit = st.columns([3, 1])
    with col_filter:
        filter_agent = st.text_input("Filter by Agent (leave blank for all)", key="audit_agent_filter")
    with col_limit:
        limit = st.number_input("Records", min_value=10, max_value=500, value=100, step=10)

    logs = get_audit_log(limit=int(limit), agent_name=filter_agent or None)

    if not logs:
        st.info("No audit log entries yet.")
    else:
        df_log = pd.DataFrame(logs)
        st.dataframe(
            df_log[["timestamp", "agent_name", "action", "entity_type", "entity_id", "human_actor", "session_id"]],
            use_container_width=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SYSTEM HEALTH
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("System Health")

    col_llm, col_db, col_rag = st.columns(3)

    # LLM health
    with col_llm:
        st.markdown("**Google Gemini LLM**")
        if settings.GEMINI_API_KEY:
            try:
                from google import genai
                _hc = genai.Client(api_key=settings.GEMINI_API_KEY)
                r = _hc.models.generate_content(
                    model=settings.GEMINI_MODEL, contents="ping"
                )
                st.success("✅ API reachable")
            except Exception as e:
                st.error(f"❌ API error: {str(e)[:60]}")
        else:
            st.warning("⚠️ API key not configured")

    # DB health
    with col_db:
        st.markdown("**SQLite Database**")
        try:
            with get_db() as conn:
                zone_count = conn.execute("SELECT COUNT(*) FROM zones").fetchone()[0]
                reading_count = conn.execute("SELECT COUNT(*) FROM sensor_readings").fetchone()[0]
            st.success(f"✅ Connected | {zone_count} zones | {reading_count} readings")
        except Exception as e:
            st.error(f"❌ DB error: {str(e)[:60]}")

    # RAG health
    with col_rag:
        st.markdown("**RAG (ChromaDB)**")
        try:
            from tools.rag_tools import _get_collection
            coll = _get_collection()
            st.info(f"ℹ️ {coll.count()} document chunks indexed")
        except Exception as e:
            st.warning(f"⚠️ RAG: {str(e)[:60]}")

    st.divider()
    st.markdown("**Configuration**")
    st.json({
        "app_version": settings.APP_VERSION,
        "gemini_model": settings.GEMINI_MODEL,
        "density_threshold": settings.DENSITY_THRESHOLD,
        "speed_threshold_kmph": settings.SPEED_THRESHOLD_KMPH,
        "critical_severity": settings.CRITICAL_SEVERITY,
        "rag_top_k": settings.RAG_TOP_K,
        "log_level": settings.LOG_LEVEL,
    })

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — PERFORMANCE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    render_performance_dashboard(st.session_state.get("session_metrics", {}))
