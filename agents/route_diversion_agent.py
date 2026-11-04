"""
Route Diversion Agent — UrbanPulse
Generates alternative diversion route recommendations for the affected zone
using LLM reasoning. Provides at least 2 options per incident.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from utils.config import settings
from utils.llm_helpers import generate_content
from utils.state_models import TrafficState
from utils.audit_logger import log_action

logger = logging.getLogger("urbanpulse")

_DIVERSION_PROMPT = """\
You are an Urban Traffic Route Planner AI.

INCIDENT CONTEXT:
- Zone: {zone_id}
- Classification: {classification}
- Severity: {severity}/10 ({severity_label})
- Evidence: {evidence}
- Weather: {weather}

Generate exactly 2 diversion route recommendations to redirect traffic away from Zone {zone_id}.
Routes should be realistic for an urban environment. Use generic road names (e.g., "Northern Bypass", "Ring Road via East", "Highway Alt-2").

Respond ONLY with valid JSON:
[
  {{
    "route_id": "R-1",
    "description": "...",
    "estimated_delay_min": 8,
    "confidence": 0.85
  }},
  {{
    "route_id": "R-2",
    "description": "...",
    "estimated_delay_min": 12,
    "confidence": 0.75
  }}
]
"""


def run(state: TrafficState) -> TrafficState:
    """
    Purpose:
        Generate 2+ alternative diversion routes for the identified incident.

    Inputs:  state.incident, state.zone_id
    Outputs: state.diversion_routes (list[DiversionRoute])
    """
    zone_id = state["zone_id"]
    session_id = state.get("session_id", "")
    incident = state.get("incident", {})

    prompt = _DIVERSION_PROMPT.format(
        zone_id=zone_id,
        classification=incident.get("classification", "INCIDENT"),
        severity=incident.get("severity_score", 5),
        severity_label=incident.get("severity_label", "MEDIUM"),
        evidence="; ".join(incident.get("evidence", [])),
        weather=json.dumps(incident.get("weather_context", {})),
    )

    routes = []
    try:
        raw = generate_content(prompt).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        routes = json.loads(raw)
    except Exception as exc:
        logger.error("Route diversion LLM failed: %s", exc)
        # Fallback: provide generic routes
        routes = [
            {
                "route_id": "R-FALLBACK-1",
                "description": f"Alternate route via Northern Bypass, avoiding Zone {zone_id}",
                "estimated_delay_min": 10,
                "confidence": 0.65,
            },
            {
                "route_id": "R-FALLBACK-2",
                "description": f"Diversion via Ring Road (East), bypassing Zone {zone_id}",
                "estimated_delay_min": 15,
                "confidence": 0.60,
            },
        ]
        log_action(
            "RouteDiversionAgent",
            "DIVERSION_LLM_FALLBACK",
            entity_type="incident",
            entity_id=incident.get("incident_id"),
            payload={"error": str(exc)},
            session_id=session_id,
        )

    trace_entry = {
        "agent": "RouteDiversionAgent",
        "action": "ROUTES_GENERATED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "route_count": len(routes),
    }

    log_action(
        "RouteDiversionAgent",
        "ROUTES_GENERATED",
        entity_type="incident",
        entity_id=incident.get("incident_id"),
        payload={"routes": [r["route_id"] for r in routes]},
        session_id=session_id,
    )

    state["diversion_routes"] = routes  # type: ignore[assignment]
    state["agent_trace"] = state.get("agent_trace", []) + [trace_entry]
    return state
