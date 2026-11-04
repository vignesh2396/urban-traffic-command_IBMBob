"""
Incident Investigation Agent — UrbanPulse
Uses Gemini LLM + Weather API + RAG to classify the incident,
assign severity, and produce an explainable reasoning trace.
This is the "brain" of the investigation layer.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Literal

from utils.config import settings
from utils.llm_helpers import generate_content
from utils.state_models import TrafficState
from utils.audit_logger import log_action
from tools.weather_tools import get_weather_for_zone
from tools.rag_tools import retrieve_regulations

logger = logging.getLogger("urbanpulse")

_CLASSIFICATION_LEVELS: list[str] = ["NORMAL", "CONGESTION", "INCIDENT", "ACCIDENT"]
_SEVERITY_LABELS: dict = {
    range(1, 4): "LOW",
    range(4, 7): "MEDIUM",
    range(7, 9): "HIGH",
    range(9, 11): "CRITICAL",
}


def _severity_label(score: int) -> str:
    for r, label in _SEVERITY_LABELS.items():
        if score in r:
            return label
    return "LOW"


_INVESTIGATION_PROMPT = """\
You are an expert Urban Traffic Incident Analyst AI assistant.
Your task is to investigate a traffic anomaly and produce a structured JSON assessment.

CONTEXT:
- Zone ID: {zone_id}
- Sensor Data: density={density:.2f} (0=empty, 1=full), avg_speed={speed:.1f} km/h
- Anomaly Type Detected: {anomaly_type} (confidence={confidence:.2f})
- Weather: {weather}
- Relevant Regulations/Protocols Retrieved:
{rag_context}

TASK:
Based on all available evidence, provide:
1. classification: one of NORMAL | CONGESTION | INCIDENT | ACCIDENT
2. severity_score: integer 1–10 (1=trivial, 10=catastrophic)
3. evidence: list of 3–5 specific bullet points citing actual data
4. reasoning: 2–3 sentence explanation of your assessment
5. confidence: float 0.0–1.0

Respond ONLY with valid JSON in this exact format:
{{
  "classification": "...",
  "severity_score": 7,
  "evidence": ["bullet 1", "bullet 2", "bullet 3"],
  "reasoning": "...",
  "confidence": 0.85
}}

IMPORTANT DISCLAIMER: This is a decision-support prototype. All critical decisions
require human operator approval before execution.
"""


def run(state: TrafficState) -> TrafficState:
    """
    Purpose:
        Investigate the detected anomaly using LLM + weather + RAG.
        Produces a classified IncidentReport with severity and reasoning.

    Inputs:  state.zone_id, state.sensor_reading, state.anomaly
    Outputs: state.incident (IncidentReport TypedDict)
    """
    zone_id = state["zone_id"]
    session_id = state.get("session_id", "")
    reading = state.get("sensor_reading", {})
    anomaly = state.get("anomaly", {})

    # ── Step 1: Fetch weather ─────────────────────────────────────────────────
    weather = get_weather_for_zone(zone_id)

    # ── Step 2: RAG retrieval ─────────────────────────────────────────────────
    rag_query = (
        f"{anomaly.get('anomaly_type', 'congestion')} "
        f"traffic protocol {'adverse weather' if weather.get('is_adverse') else ''} "
        f"urban emergency response"
    )
    rag_docs = retrieve_regulations(rag_query, zone_id=zone_id)
    rag_context = (
        "\n".join(
            f"  [{i+1}] {d['source']}: {d['content'][:300]}..."
            for i, d in enumerate(rag_docs)
        )
        if rag_docs
        else "  No relevant regulations found in knowledge base."
    )
    rag_sources = [d["source"] for d in rag_docs]

    # ── Step 3: LLM classification ────────────────────────────────────────────
    prompt = _INVESTIGATION_PROMPT.format(
        zone_id=zone_id,
        density=reading.get("density", 0.0),
        speed=reading.get("avg_speed_kmh", 0.0),
        anomaly_type=anomaly.get("anomaly_type", "UNKNOWN"),
        confidence=anomaly.get("confidence", 0.0),
        weather=json.dumps(weather),
        rag_context=rag_context,
    )

    try:
        raw_text = generate_content(prompt).strip()

        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        llm_result = json.loads(raw_text)
        classification: str = llm_result.get("classification", "CONGESTION")
        severity_score: int = max(1, min(10, int(llm_result.get("severity_score", 5))))
        evidence: list[str] = llm_result.get("evidence", [])
        reasoning: str = llm_result.get("reasoning", "")
        confidence: float = float(llm_result.get("confidence", 0.7))

    except Exception as exc:
        # Graceful degradation: rule-based fallback
        logger.error("LLM investigation failed for zone %s: %s", zone_id, exc)
        classification = _fallback_classification(anomaly)
        severity_score = _fallback_severity(anomaly, weather)
        evidence = [
            f"Density at {reading.get('density', 0):.0%} capacity",
            f"Average speed {reading.get('avg_speed_kmh', 0):.1f} km/h",
            f"Anomaly type: {anomaly.get('anomaly_type', 'UNKNOWN')}",
            f"Weather: {weather.get('description', 'unknown')}",
        ]
        reasoning = (
            f"Rule-based fallback (LLM unavailable). "
            f"Anomaly: {anomaly.get('anomaly_type')}. "
            f"Adverse weather: {weather.get('is_adverse')}."
        )
        confidence = anomaly.get("confidence", 0.5)
        log_action(
            "IncidentInvestigationAgent",
            "LLM_FALLBACK_USED",
            entity_type="zone",
            entity_id=zone_id,
            payload={"error": str(exc)},
            session_id=session_id,
        )

    incident_id = str(uuid.uuid4())
    incident = {
        "incident_id": incident_id,
        "zone_id": zone_id,
        "detected_at": datetime.now(timezone.utc).isoformat(),
        "classification": classification,
        "severity_score": severity_score,
        "severity_label": _severity_label(severity_score),
        "evidence": evidence,
        "weather_context": weather,
        "rag_sources": rag_sources,
        "reasoning": reasoning,
    }

    trace_entry = {
        "agent": "IncidentInvestigationAgent",
        "action": "INCIDENT_CLASSIFIED",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "classification": classification,
        "severity_score": severity_score,
        "confidence": confidence,
    }

    log_action(
        "IncidentInvestigationAgent",
        "INCIDENT_CLASSIFIED",
        entity_type="incident",
        entity_id=incident_id,
        payload={
            "classification": classification,
            "severity_score": severity_score,
            "severity_label": _severity_label(severity_score),
            "rag_hits": len(rag_sources),
            "weather_adverse": weather.get("is_adverse"),
        },
        session_id=session_id,
    )

    state["incident"] = incident  # type: ignore[assignment]
    state["agent_trace"] = state.get("agent_trace", []) + [trace_entry]
    return state


# ── Fallback helpers (rule-based when LLM unavailable) ────────────────────────

def _fallback_classification(anomaly: dict) -> str:
    atype = anomaly.get("anomaly_type", "NONE")
    conf = anomaly.get("confidence", 0.0)
    if atype == "COMBINED" and conf > 0.8:
        return "ACCIDENT"
    if atype in ("COMBINED", "CONGESTION"):
        return "INCIDENT"
    if atype == "SPEED_DROP":
        return "CONGESTION"
    return "NORMAL"


def _fallback_severity(anomaly: dict, weather: dict) -> int:
    base = int(anomaly.get("confidence", 0.5) * 8)
    if weather.get("is_adverse"):
        base += 1
    if anomaly.get("anomaly_type") == "COMBINED":
        base += 1
    return max(1, min(10, base))
