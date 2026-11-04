"""
Signal Optimization Agent — UrbanPulse
Recommends adaptive traffic signal timing changes for intersections
in and around the affected zone to improve traffic flow.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from utils.config import settings
from utils.llm_helpers import generate_content
from utils.state_models import TrafficState
from utils.audit_logger import log_action

logger = logging.getLogger("urbanpulse")

_SIGNAL_PROMPT = """\
You are a Traffic Signal Management AI Specialist.

INCIDENT CONTEXT:
- Zone: {zone_id}
- Classification: {classification}
- Severity: {severity}/10
- Diversions activated: {diversion_descriptions}

Recommend signal timing adjustments for 2–3 key intersections in and around Zone {zone_id}
to facilitate the recommended diversions and reduce congestion.

Use realistic intersection names (e.g., "Junction A-North", "Intersection 7-West").

Respond ONLY with valid JSON:
[
  {{
    "intersection_id": "INT-{zone_id}-N",
    "action": "extend_green_30s_northbound",
    "justification": "..."
  }},
  {{
    "intersection_id": "INT-{zone_id}-S",
    "action": "activate_emergency_phase",
    "justification": "..."
  }}
]
"""


def run(state: TrafficState) -> TrafficState:
    """
    Purpose:
        Generate adaptive signal timing recommendations for affected intersections.

    Inputs:  state.incident, state.diversion_routes
    Outputs: state.signal_changes (list[SignalChange])
    """
    zone_id = state["zone_id"]
    session_id = state.get("session_id", "")
    incident = state.get("incident", {})
    routes = state.get("diversion_routes", [])

    diversion_desc = "; ".join(r.get("description", "") for r in routes) or "No diversions"

    prompt = _SIGNAL_PROMPT.format(
        zone_id=zone_id,
        classification=incident.get("classification", "INCIDENT"),
        severity=incident.get("severity_score", 5),
        diversion_descriptions=diversion_desc,
    )

    signal_changes = []
    try:
        raw = generate_content(prompt).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        signal_changes = json.loads(raw)
    except Exception as exc:
        logger.error("Signal optimization LLM failed: %s", exc)
        signal_changes = [
            {
                "intersection_id": f"INT-{zone_id}-N",
                "action": "extend_green_30s_on_diversion_route",
                "justification": "Fallback: extend green time on primary diversion corridor.",
            },
            {
                "intersection_id": f"INT-{zone_id}-S",
                "action": "reduce_green_toward_incident_zone",
                "justification": "Fallback: discourage inflow into congested zone.",
            },
        ]
        log_action(
            "SignalOptimizationAgent",
            "SIGNAL_LLM_FALLBACK",
            entity_type="incident",
            entity_id=incident.get("incident_id"),
            payload={"error": str(exc)},
            session_id=session_id,
        )

    trace_entry = {
        "agent": "SignalOptimizationAgent",
        "action": "SIGNAL_PLAN_GENERATED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "intersection_count": len(signal_changes),
    }

    log_action(
        "SignalOptimizationAgent",
        "SIGNAL_PLAN_GENERATED",
        entity_type="incident",
        entity_id=incident.get("incident_id"),
        payload={"intersections": [s["intersection_id"] for s in signal_changes]},
        session_id=session_id,
    )

    state["signal_changes"] = signal_changes  # type: ignore[assignment]
    state["agent_trace"] = state.get("agent_trace", []) + [trace_entry]
    return state
