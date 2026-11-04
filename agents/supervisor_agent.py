"""
Supervisor Agent — UrbanPulse
LangGraph StateGraph controller that orchestrates all specialized agents.
Implements the HITL interrupt gate for CRITICAL severity incidents.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from utils.state_models import TrafficState
from utils.audit_logger import log_action
from agents.traffic_monitoring_agent import run as run_monitoring
from agents.congestion_detection_agent import run as run_congestion
from agents.incident_investigation_agent import run as run_investigation
from agents.route_diversion_agent import run as run_diversion
from agents.signal_optimization_agent import run as run_signal
from agents.traffic_response_coordination_agent import run as run_coordination


# ── Node wrappers ─────────────────────────────────────────────────────────────

def node_monitoring(state: TrafficState) -> TrafficState:
    log_action("SupervisorAgent", "NODE_ENTERED", payload={"node": "monitoring"})
    return run_monitoring(state)


def node_congestion(state: TrafficState) -> TrafficState:
    log_action("SupervisorAgent", "NODE_ENTERED", payload={"node": "congestion_detection"})
    return run_congestion(state)


def node_investigation(state: TrafficState) -> TrafficState:
    log_action("SupervisorAgent", "NODE_ENTERED", payload={"node": "incident_investigation"})
    return run_investigation(state)


def node_diversion(state: TrafficState) -> TrafficState:
    log_action("SupervisorAgent", "NODE_ENTERED", payload={"node": "route_diversion"})
    return run_diversion(state)


def node_signal(state: TrafficState) -> TrafficState:
    log_action("SupervisorAgent", "NODE_ENTERED", payload={"node": "signal_optimization"})
    return run_signal(state)


def node_coordination(state: TrafficState) -> TrafficState:
    log_action("SupervisorAgent", "NODE_ENTERED", payload={"node": "response_coordination"})
    return run_coordination(state)


def node_hitl_gate(state: TrafficState) -> TrafficState:
    """
    HITL interrupt node.
    Sets awaiting_human=True if the incident is CRITICAL (severity >= 9).
    LangGraph's interrupt mechanism pauses here until the graph is resumed
    with a human decision injected into the state.
    """
    incident = state.get("incident")
    if incident and incident["severity_score"] >= 9:
        log_action(
            "SupervisorAgent",
            "HITL_GATE_TRIGGERED",
            entity_type="incident",
            entity_id=incident.get("incident_id"),
            payload={"severity": incident["severity_score"]},
        )
        state["awaiting_human"] = True
    return state


def node_execute(state: TrafficState) -> TrafficState:
    """
    Final execution node — marks the incident as CLOSED in the DB,
    persists the response plan, and finalises the pipeline.
    """
    import json
    from utils.db import get_db

    incident = state.get("incident")
    plan = state.get("response_plan")
    session_id = state.get("session_id", "")

    if incident and plan:
        with get_db() as conn:
            # Persist incident
            conn.execute(
                """
                INSERT OR REPLACE INTO incidents
                    (incident_id, zone_id, detected_at, classification,
                     severity_score, severity_label, weather_context,
                     rag_sources, reasoning, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    incident["incident_id"],
                    incident["zone_id"],
                    incident["detected_at"],
                    incident["classification"],
                    incident["severity_score"],
                    incident["severity_label"],
                    json.dumps(incident.get("weather_context", {})),
                    json.dumps(incident.get("rag_sources", [])),
                    incident.get("reasoning", ""),
                    "CLOSED",
                ),
            )
            # Persist response plan
            conn.execute(
                """
                INSERT OR REPLACE INTO response_plans
                    (plan_id, incident_id, diversion_routes, signal_changes,
                     commuter_alert, authority_alert, confidence_score,
                     requires_approval)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan["plan_id"],
                    plan["incident_id"],
                    json.dumps(plan.get("diversion_routes", [])),
                    json.dumps(plan.get("signal_changes", [])),
                    plan.get("commuter_alert", ""),
                    plan.get("authority_alert", ""),
                    plan.get("confidence_score", 0.0),
                    1 if plan.get("requires_approval") else 0,
                ),
            )

    state["pipeline_end"] = datetime.now(timezone.utc).isoformat()
    log_action(
        "SupervisorAgent",
        "PIPELINE_COMPLETED",
        entity_type="incident",
        entity_id=incident.get("incident_id") if incident else None,
        session_id=session_id,
    )
    return state


# ── Conditional edge routing ──────────────────────────────────────────────────

def route_after_anomaly(
    state: TrafficState,
) -> Literal["investigate", "end_normal"]:
    anomaly = state.get("anomaly")
    if anomaly and anomaly["detected"]:
        return "investigate"
    return "end_normal"


def route_after_investigation(
    state: TrafficState,
) -> Literal["plan", "end_minor"]:
    incident = state.get("incident")
    if incident and incident["severity_score"] >= 3:
        return "plan"
    return "end_minor"


def route_after_hitl(
    state: TrafficState,
) -> Literal["execute", "hitl_wait"]:
    if state.get("awaiting_human"):
        return "hitl_wait"
    return "execute"


# ── Build the graph ───────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph StateGraph.

    Graph topology:
        monitoring → congestion → [if anomaly] investigation
                                   → [if severity>=3] diversion + signal (parallel)
                                   → coordination → HITL gate
                                   → [if approved / auto] execute → END
    """
    graph = StateGraph(TrafficState)

    graph.add_node("monitoring", node_monitoring)
    graph.add_node("congestion", node_congestion)
    graph.add_node("investigation", node_investigation)
    graph.add_node("diversion", node_diversion)
    graph.add_node("signal", node_signal)
    graph.add_node("coordination", node_coordination)
    graph.add_node("hitl_gate", node_hitl_gate)
    graph.add_node("execute", node_execute)

    graph.set_entry_point("monitoring")
    graph.add_edge("monitoring", "congestion")

    graph.add_conditional_edges(
        "congestion",
        route_after_anomaly,
        {"investigate": "investigation", "end_normal": END},
    )

    graph.add_conditional_edges(
        "investigation",
        route_after_investigation,
        {"plan": "diversion", "end_minor": END},
    )

    graph.add_edge("diversion", "signal")
    graph.add_edge("signal", "coordination")
    graph.add_edge("coordination", "hitl_gate")

    graph.add_conditional_edges(
        "hitl_gate",
        route_after_hitl,
        {"execute": "execute", "hitl_wait": END},  # hitl_wait surfaces to UI
    )

    graph.add_edge("execute", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ── Public invocation API ─────────────────────────────────────────────────────

_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def invoke_pipeline(zone_id: str, session_id: str = None) -> TrafficState:
    """
    Run the full agent pipeline for a zone.

    Args:
        zone_id:    Zone to investigate.
        session_id: Optional caller-provided session ID for traceability.

    Returns:
        Final TrafficState after pipeline completion or HITL pause.
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    initial_state: TrafficState = {
        "zone_id": zone_id,
        "session_id": session_id,
        "sensor_reading": None,
        "anomaly": None,
        "incident": None,
        "diversion_routes": [],
        "signal_changes": [],
        "response_plan": None,
        "hitl_decision": None,
        "awaiting_human": False,
        "errors": [],
        "pipeline_start": datetime.now(timezone.utc).isoformat(),
        "pipeline_end": None,
        "agent_trace": [],
    }

    log_action(
        "SupervisorAgent",
        "PIPELINE_STARTED",
        entity_type="zone",
        entity_id=zone_id,
        session_id=session_id,
    )

    config = {"configurable": {"thread_id": session_id}}
    graph = get_compiled_graph()
    result = graph.invoke(initial_state, config=config)
    return result


def resume_pipeline(session_id: str, hitl_decision: dict) -> TrafficState:
    """
    Resume a paused pipeline after a human decision.

    Args:
        session_id:    The session ID of the paused pipeline.
        hitl_decision: HITLDecision dict with operator decision.

    Returns:
        Final TrafficState after pipeline completes.
    """
    config = {"configurable": {"thread_id": session_id}}
    graph = get_compiled_graph()

    # Inject the human decision and clear the awaiting flag
    update = {
        "hitl_decision": hitl_decision,
        "awaiting_human": False,
    }

    result = graph.invoke(update, config=config)
    log_action(
        "SupervisorAgent",
        "PIPELINE_RESUMED",
        session_id=session_id,
        human_actor=hitl_decision.get("operator_id"),
        payload={"decision": hitl_decision.get("decision")},
    )
    return result
