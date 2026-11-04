"""
Traffic Response Coordination Agent — UrbanPulse
Assembles the final coordinated response plan from all prior agent outputs.
Generates commuter and authority alerts. Assigns requires_approval flag.
This is the final planning agent before the HITL gate.
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

_COORDINATION_PROMPT = """\
You are a Traffic Response Coordination AI.

FULL INCIDENT CONTEXT:
- Zone: {zone_id}
- Classification: {classification}
- Severity: {severity}/10 ({severity_label})
- Reasoning: {reasoning}
- Diversions: {diversions}
- Signal Changes: {signal_changes}
- Weather: {weather}

TASK: Generate two short alert messages:

1. commuter_alert: A clear, helpful 1-sentence message for commuters (max 140 chars).
2. authority_alert: A detailed 2-sentence operational message for emergency services / TMC.

Also provide:
3. confidence_score: Float 0.0–1.0 for the overall response plan quality.

Respond ONLY with valid JSON:
{{
  "commuter_alert": "...",
  "authority_alert": "...",
  "confidence_score": 0.87
}}
"""


def run(state: TrafficState) -> TrafficState:
    """
    Purpose:
        Assemble the complete ResponsePlan, generate alerts, and flag
        whether human approval is required (severity >= CRITICAL_SEVERITY).

    Inputs:  state.incident, state.diversion_routes, state.signal_changes
    Outputs: state.response_plan (ResponsePlan TypedDict)
    """
    zone_id = state["zone_id"]
    session_id = state.get("session_id", "")
    incident = state.get("incident", {})
    routes = state.get("diversion_routes", [])
    signals = state.get("signal_changes", [])

    severity = incident.get("severity_score", 1)
    requires_approval = severity >= settings.CRITICAL_SEVERITY

    prompt = _COORDINATION_PROMPT.format(
        zone_id=zone_id,
        classification=incident.get("classification", "INCIDENT"),
        severity=severity,
        severity_label=incident.get("severity_label", "MEDIUM"),
        reasoning=incident.get("reasoning", ""),
        diversions=json.dumps(routes),
        signal_changes=json.dumps(signals),
        weather=json.dumps(incident.get("weather_context", {})),
    )

    commuter_alert = ""
    authority_alert = ""
    confidence_score = 0.7

    try:
        raw = generate_content(prompt).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        coord_result = json.loads(raw)
        commuter_alert = coord_result.get("commuter_alert", "")
        authority_alert = coord_result.get("authority_alert", "")
        confidence_score = float(coord_result.get("confidence_score", 0.7))

    except Exception as exc:
        logger.error("Coordination LLM failed: %s", exc)
        sev_label = incident.get("severity_label", "HIGH")
        commuter_alert = (
            f"⚠️ Traffic disruption in Zone {zone_id}. "
            f"Use alternate routes. Expect delays."
        )
        authority_alert = (
            f"INCIDENT in Zone {zone_id}: {incident.get('classification')} "
            f"(severity {severity}/10). Dispatch recommended. "
            f"Diversions and signal changes initiated."
        )
        log_action(
            "TrafficResponseCoordinationAgent",
            "COORDINATION_LLM_FALLBACK",
            entity_type="incident",
            entity_id=incident.get("incident_id"),
            payload={"error": str(exc)},
            session_id=session_id,
        )

    plan_id = str(uuid.uuid4())
    response_plan = {
        "plan_id": plan_id,
        "incident_id": incident.get("incident_id", ""),
        "diversion_routes": routes,
        "signal_changes": signals,
        "commuter_alert": commuter_alert,
        "authority_alert": authority_alert,
        "confidence_score": confidence_score,
        "requires_approval": requires_approval,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    trace_entry = {
        "agent": "TrafficResponseCoordinationAgent",
        "action": "RESPONSE_PLAN_CREATED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "plan_id": plan_id,
        "requires_approval": requires_approval,
        "confidence": confidence_score,
    }

    log_action(
        "TrafficResponseCoordinationAgent",
        "RESPONSE_PLAN_CREATED",
        entity_type="response_plan",
        entity_id=plan_id,
        payload={
            "incident_id": incident.get("incident_id"),
            "severity": severity,
            "requires_approval": requires_approval,
            "confidence": confidence_score,
        },
        session_id=session_id,
    )

    state["response_plan"] = response_plan  # type: ignore[assignment]
    state["agent_trace"] = state.get("agent_trace", []) + [trace_entry]
    return state
