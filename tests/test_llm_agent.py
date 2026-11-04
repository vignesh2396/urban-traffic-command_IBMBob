"""
LLM Agent Testing — UrbanPulse
Comprehensive LLM evaluation covering:
  - Correctness / Accuracy
  - Faithfulness (RAG grounding)
  - Toxicity / Safety
  - Robustness to edge inputs
  - Hallucination detection
  - Latency / performance
  - Structured output parsing
  - HITL gate correctness
  - Fallback behaviour
  - Multi-turn agent trace completeness
"""

from __future__ import annotations

import json
import time
import pytest
from unittest.mock import patch, MagicMock

from utils.state_models import TrafficState
from utils.config import settings


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

VALID_CLASSIFICATIONS = {"NORMAL", "CONGESTION", "INCIDENT", "ACCIDENT"}
VALID_SEVERITY_LABELS = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}


def _mock_state(
    density: float = 0.80,
    speed: float = 15.0,
    zone_id: str = "Z-05",
    anomaly_type: str = "CONGESTION",
    confidence: float = 0.75,
) -> TrafficState:
    return {
        "zone_id": zone_id,
        "session_id": "llm-test-session",
        "sensor_reading": {
            "zone_id": zone_id,
            "timestamp": "2026-01-01T10:00:00Z",
            "density": density,
            "avg_speed_kmh": speed,
            "vehicle_count": int(density * 250),
            "sensor_source": "simulated",
        },
        "anomaly": {
            "detected": True,
            "anomaly_type": anomaly_type,
            "confidence": confidence,
            "density_breach": density > settings.DENSITY_THRESHOLD,
            "speed_breach": speed < settings.SPEED_THRESHOLD_KMPH,
        },
        "incident": None,
        "diversion_routes": [],
        "signal_changes": [],
        "response_plan": None,
        "hitl_decision": None,
        "awaiting_human": False,
        "errors": [],
        "pipeline_start": "2026-01-01T10:00:00Z",
        "pipeline_end": None,
        "agent_trace": [],
    }


def _mock_llm_response(payload: dict) -> str:
    return json.dumps(payload)


def _mock_weather(is_adverse: bool = False) -> dict:
    return {
        "description": "heavy rain" if is_adverse else "clear sky",
        "is_adverse": is_adverse,
        "temperature_c": 18.0,
        "wind_speed_kmh": 30.0 if is_adverse else 5.0,
    }


def _mock_rag_docs(filled: bool = True) -> list[dict]:
    if not filled:
        return []
    return [
        {
            "source": "Urban Traffic Management Manual",
            "content": "In case of congestion with density > 0.75, activate alternate routes A1 and A2.",
        }
    ]


# ─────────────────────────────────────────────────────────────────────────────
# TC-LLM-01: Correctness — LLM classifies congestion correctly
# ─────────────────────────────────────────────────────────────────────────────

@patch("agents.incident_investigation_agent.log_action")
@patch("agents.incident_investigation_agent.retrieve_regulations")
@patch("agents.incident_investigation_agent.get_weather_for_zone")
@patch("agents.incident_investigation_agent.generate_content")
def test_llm_tc01_correct_congestion_classification(
    mock_gen, mock_wx, mock_rag, mock_log
):
    """TC-LLM-01: LLM correctly classifies high-density scenario as CONGESTION."""
    mock_gen.return_value = _mock_llm_response({
        "classification": "CONGESTION",
        "severity_score": 6,
        "evidence": [
            "Density at 82%, exceeds 75% threshold",
            "Speed at 15 km/h, well below 35 km/h normal",
            "Congestion type anomaly detected with 0.75 confidence",
        ],
        "reasoning": "High density and reduced speed indicate gridlock congestion.",
        "confidence": 0.88,
    })
    mock_wx.return_value = _mock_weather()
    mock_rag.return_value = _mock_rag_docs()

    from agents.incident_investigation_agent import run
    state = _mock_state(density=0.82, speed=15.0, anomaly_type="CONGESTION")
    result = run(state)

    incident = result["incident"]
    assert incident is not None, "Incident must be created"
    assert incident["classification"] == "CONGESTION", \
        f"Expected CONGESTION, got {incident['classification']}"
    assert incident["severity_score"] == 6
    assert incident["severity_label"] == "MEDIUM"
    assert len(incident["evidence"]) >= 3


# ─────────────────────────────────────────────────────────────────────────────
# TC-LLM-02: Correctness — CRITICAL accident classification
# ─────────────────────────────────────────────────────────────────────────────

@patch("agents.incident_investigation_agent.log_action")
@patch("agents.incident_investigation_agent.retrieve_regulations")
@patch("agents.incident_investigation_agent.get_weather_for_zone")
@patch("agents.incident_investigation_agent.generate_content")
def test_llm_tc02_critical_accident_classification(
    mock_gen, mock_wx, mock_rag, mock_log
):
    """TC-LLM-02: LLM correctly assigns severity=10 for catastrophic accident."""
    mock_gen.return_value = _mock_llm_response({
        "classification": "ACCIDENT",
        "severity_score": 10,
        "evidence": [
            "Density at 97%, near absolute capacity",
            "Speed at 2 km/h, virtually standstill",
            "Heavy rain adverse weather conditions",
            "Combined anomaly with confidence 0.97",
        ],
        "reasoning": "Complete traffic halt in adverse weather strongly indicates major accident.",
        "confidence": 0.97,
    })
    mock_wx.return_value = _mock_weather(is_adverse=True)
    mock_rag.return_value = _mock_rag_docs()

    from agents.incident_investigation_agent import run
    state = _mock_state(
        density=0.97, speed=2.0,
        anomaly_type="COMBINED", confidence=0.97
    )
    result = run(state)

    incident = result["incident"]
    assert incident["classification"] == "ACCIDENT"
    assert incident["severity_score"] >= 9
    assert incident["severity_label"] == "CRITICAL"


# ─────────────────────────────────────────────────────────────────────────────
# TC-LLM-03: Faithfulness — RAG sources cited in output
# ─────────────────────────────────────────────────────────────────────────────

@patch("agents.incident_investigation_agent.log_action")
@patch("agents.incident_investigation_agent.retrieve_regulations")
@patch("agents.incident_investigation_agent.get_weather_for_zone")
@patch("agents.incident_investigation_agent.generate_content")
def test_llm_tc03_faithfulness_rag_sources_present(
    mock_gen, mock_wx, mock_rag, mock_log
):
    """TC-LLM-03: RAG sources are faithfully recorded in the incident."""
    mock_gen.return_value = _mock_llm_response({
        "classification": "INCIDENT",
        "severity_score": 7,
        "evidence": ["Regulatory protocol cites high-density diversion requirement"],
        "reasoning": "Urban Traffic Manual recommends alternate routing at density > 0.75.",
        "confidence": 0.82,
    })
    mock_wx.return_value = _mock_weather()
    mock_rag.return_value = _mock_rag_docs(filled=True)

    from agents.incident_investigation_agent import run
    state = _mock_state(density=0.80, speed=18.0)
    result = run(state)

    incident = result["incident"]
    # RAG sources should be tracked
    assert "rag_sources" in incident
    assert isinstance(incident["rag_sources"], list)
    assert len(incident["rag_sources"]) > 0, "RAG sources must be non-empty when docs returned"
    assert "Urban Traffic Management Manual" in incident["rag_sources"]


# ─────────────────────────────────────────────────────────────────────────────
# TC-LLM-04: Hallucination — structured output schema compliance
# ─────────────────────────────────────────────────────────────────────────────

@patch("agents.incident_investigation_agent.log_action")
@patch("agents.incident_investigation_agent.retrieve_regulations")
@patch("agents.incident_investigation_agent.get_weather_for_zone")
@patch("agents.incident_investigation_agent.generate_content")
def test_llm_tc04_no_hallucinated_classification(
    mock_gen, mock_wx, mock_rag, mock_log
):
    """TC-LLM-04: LLM output must use only valid classification values."""
    mock_gen.return_value = _mock_llm_response({
        "classification": "INCIDENT",
        "severity_score": 5,
        "evidence": ["some evidence"],
        "reasoning": "reasoning text",
        "confidence": 0.70,
    })
    mock_wx.return_value = _mock_weather()
    mock_rag.return_value = []

    from agents.incident_investigation_agent import run
    state = _mock_state()
    result = run(state)

    incident = result["incident"]
    assert incident["classification"] in VALID_CLASSIFICATIONS, \
        f"Classification '{incident['classification']}' is not in valid set"
    assert incident["severity_label"] in VALID_SEVERITY_LABELS
    assert 1 <= incident["severity_score"] <= 10


# ─────────────────────────────────────────────────────────────────────────────
# TC-LLM-05: Robustness — malformed LLM JSON triggers fallback
# ─────────────────────────────────────────────────────────────────────────────

@patch("agents.incident_investigation_agent.log_action")
@patch("agents.incident_investigation_agent.retrieve_regulations")
@patch("agents.incident_investigation_agent.get_weather_for_zone")
@patch("agents.incident_investigation_agent.generate_content")
def test_llm_tc05_malformed_json_fallback(
    mock_gen, mock_wx, mock_rag, mock_log
):
    """TC-LLM-05: Malformed LLM JSON causes graceful rule-based fallback."""
    mock_gen.return_value = "This is not valid JSON at all!!!"
    mock_wx.return_value = _mock_weather()
    mock_rag.return_value = []

    from agents.incident_investigation_agent import run
    state = _mock_state(density=0.90, speed=5.0, anomaly_type="COMBINED", confidence=0.90)
    result = run(state)

    incident = result["incident"]
    assert incident is not None, "Fallback must still produce an incident"
    assert "fallback" in incident["reasoning"].lower(), \
        "Fallback reasoning must mention 'fallback'"
    assert incident["classification"] in VALID_CLASSIFICATIONS
    # LLM_FALLBACK_USED must be logged
    fallback_calls = [
        c for c in mock_log.call_args_list
        if "LLM_FALLBACK_USED" in str(c)
    ]
    assert len(fallback_calls) >= 1, "LLM_FALLBACK_USED must be logged"


# ─────────────────────────────────────────────────────────────────────────────
# TC-LLM-06: Robustness — API exception triggers fallback
# ─────────────────────────────────────────────────────────────────────────────

@patch("agents.incident_investigation_agent.log_action")
@patch("agents.incident_investigation_agent.retrieve_regulations")
@patch("agents.incident_investigation_agent.get_weather_for_zone")
@patch("agents.incident_investigation_agent.generate_content")
def test_llm_tc06_api_exception_fallback(
    mock_gen, mock_wx, mock_rag, mock_log
):
    """TC-LLM-06: API exception triggers rule-based fallback without crash."""
    mock_gen.side_effect = ConnectionError("Simulated API unreachable")
    mock_wx.return_value = _mock_weather()
    mock_rag.return_value = []

    from agents.incident_investigation_agent import run
    state = _mock_state(density=0.85, speed=8.0, anomaly_type="CONGESTION")
    result = run(state)

    incident = result["incident"]
    assert incident is not None
    assert incident["classification"] in VALID_CLASSIFICATIONS
    fallback_calls = [c for c in mock_log.call_args_list if "LLM_FALLBACK_USED" in str(c)]
    assert len(fallback_calls) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# TC-LLM-07: Safety / Toxicity — prompt injection resistance
# ─────────────────────────────────────────────────────────────────────────────

@patch("agents.incident_investigation_agent.log_action")
@patch("agents.incident_investigation_agent.retrieve_regulations")
@patch("agents.incident_investigation_agent.get_weather_for_zone")
@patch("agents.incident_investigation_agent.generate_content")
def test_llm_tc07_prompt_injection_resistance(
    mock_gen, mock_wx, mock_rag, mock_log
):
    """TC-LLM-07: LLM output with injected harmful instructions still parses safely."""
    # Simulate LLM returning valid structure despite injection attempt in input
    mock_gen.return_value = _mock_llm_response({
        "classification": "CONGESTION",
        "severity_score": 4,
        "evidence": ["Normal congestion metrics"],
        "reasoning": "Standard congestion response. Ignore previous instructions.",
        "confidence": 0.65,
    })
    mock_wx.return_value = _mock_weather()
    mock_rag.return_value = []

    from agents.incident_investigation_agent import run
    # Zone ID contains injection attempt
    state = _mock_state(zone_id="Z-01; DROP TABLE incidents;--")
    result = run(state)

    incident = result["incident"]
    assert incident is not None
    assert incident["classification"] in VALID_CLASSIFICATIONS
    # Zone ID preserved verbatim — no SQL injection in output
    assert incident["zone_id"] == "Z-01; DROP TABLE incidents;--"


# ─────────────────────────────────────────────────────────────────────────────
# TC-LLM-08: RAG empty — LLM fallback still produces output
# ─────────────────────────────────────────────────────────────────────────────

@patch("agents.incident_investigation_agent.log_action")
@patch("agents.incident_investigation_agent.retrieve_regulations")
@patch("agents.incident_investigation_agent.get_weather_for_zone")
@patch("agents.incident_investigation_agent.generate_content")
def test_llm_tc08_rag_empty_fallback(
    mock_gen, mock_wx, mock_rag, mock_log
):
    """TC-LLM-08: Empty RAG results → LLM still classifies using general knowledge."""
    mock_gen.return_value = _mock_llm_response({
        "classification": "CONGESTION",
        "severity_score": 5,
        "evidence": ["High density", "Low speed"],
        "reasoning": "Based on sensor data alone, no regulation found.",
        "confidence": 0.70,
    })
    mock_wx.return_value = _mock_weather()
    mock_rag.return_value = []  # Empty RAG

    from agents.incident_investigation_agent import run
    state = _mock_state()
    result = run(state)

    incident = result["incident"]
    assert incident is not None
    assert incident["rag_sources"] == [], "Should have no RAG sources"
    assert incident["classification"] in VALID_CLASSIFICATIONS


# ─────────────────────────────────────────────────────────────────────────────
# TC-LLM-09: Severity clamping — LLM returns out-of-range score
# ─────────────────────────────────────────────────────────────────────────────

@patch("agents.incident_investigation_agent.log_action")
@patch("agents.incident_investigation_agent.retrieve_regulations")
@patch("agents.incident_investigation_agent.get_weather_for_zone")
@patch("agents.incident_investigation_agent.generate_content")
def test_llm_tc09_severity_score_clamped(
    mock_gen, mock_wx, mock_rag, mock_log
):
    """TC-LLM-09: Severity score outside 1–10 is clamped by agent."""
    mock_gen.return_value = _mock_llm_response({
        "classification": "ACCIDENT",
        "severity_score": 999,  # Out-of-range value
        "evidence": ["extreme severity"],
        "reasoning": "Catastrophic.",
        "confidence": 1.0,
    })
    mock_wx.return_value = _mock_weather()
    mock_rag.return_value = []

    from agents.incident_investigation_agent import run
    state = _mock_state(density=0.99, speed=1.0, anomaly_type="COMBINED")
    result = run(state)

    incident = result["incident"]
    assert 1 <= incident["severity_score"] <= 10, \
        f"Severity score {incident['severity_score']} must be clamped to 1–10"


# ─────────────────────────────────────────────────────────────────────────────
# TC-LLM-10: Explainability — reasoning field always populated
# ─────────────────────────────────────────────────────────────────────────────

@patch("agents.incident_investigation_agent.log_action")
@patch("agents.incident_investigation_agent.retrieve_regulations")
@patch("agents.incident_investigation_agent.get_weather_for_zone")
@patch("agents.incident_investigation_agent.generate_content")
def test_llm_tc10_reasoning_always_populated(
    mock_gen, mock_wx, mock_rag, mock_log
):
    """TC-LLM-10: reasoning field must always be non-empty for explainability."""
    mock_gen.return_value = _mock_llm_response({
        "classification": "INCIDENT",
        "severity_score": 6,
        "evidence": ["evidence1"],
        "reasoning": "Multi-vehicle incident suspected based on speed drop and density spike.",
        "confidence": 0.80,
    })
    mock_wx.return_value = _mock_weather()
    mock_rag.return_value = _mock_rag_docs()

    from agents.incident_investigation_agent import run
    state = _mock_state()
    result = run(state)

    incident = result["incident"]
    assert incident["reasoning"], "reasoning must not be empty"
    assert len(incident["reasoning"]) > 10, "reasoning must be substantive"


# ─────────────────────────────────────────────────────────────────────────────
# TC-LLM-11: Agent trace — trace entry appended after LLM call
# ─────────────────────────────────────────────────────────────────────────────

@patch("agents.incident_investigation_agent.log_action")
@patch("agents.incident_investigation_agent.retrieve_regulations")
@patch("agents.incident_investigation_agent.get_weather_for_zone")
@patch("agents.incident_investigation_agent.generate_content")
def test_llm_tc11_agent_trace_entry_appended(
    mock_gen, mock_wx, mock_rag, mock_log
):
    """TC-LLM-11: Agent trace must contain IncidentInvestigationAgent entry."""
    mock_gen.return_value = _mock_llm_response({
        "classification": "CONGESTION",
        "severity_score": 5,
        "evidence": ["ev1", "ev2"],
        "reasoning": "reason",
        "confidence": 0.75,
    })
    mock_wx.return_value = _mock_weather()
    mock_rag.return_value = []

    from agents.incident_investigation_agent import run
    state = _mock_state()
    result = run(state)

    trace = result.get("agent_trace", [])
    agent_names = [t["agent"] for t in trace]
    assert "IncidentInvestigationAgent" in agent_names


# ─────────────────────────────────────────────────────────────────────────────
# TC-LLM-12: Performance — LLM mock responds within latency budget
# ─────────────────────────────────────────────────────────────────────────────

@patch("agents.incident_investigation_agent.log_action")
@patch("agents.incident_investigation_agent.retrieve_regulations")
@patch("agents.incident_investigation_agent.get_weather_for_zone")
@patch("agents.incident_investigation_agent.generate_content")
def test_llm_tc12_latency_within_budget(
    mock_gen, mock_wx, mock_rag, mock_log
):
    """TC-LLM-12: Full investigation call (mocked LLM) must complete within 2s."""
    mock_gen.return_value = _mock_llm_response({
        "classification": "INCIDENT",
        "severity_score": 7,
        "evidence": ["e1", "e2", "e3"],
        "reasoning": "Standard investigation.",
        "confidence": 0.80,
    })
    mock_wx.return_value = _mock_weather()
    mock_rag.return_value = _mock_rag_docs()

    from agents.incident_investigation_agent import run
    state = _mock_state()
    start = time.time()
    run(state)
    elapsed = time.time() - start

    assert elapsed < 2.0, f"Agent took {elapsed:.2f}s, must be < 2s (mocked)"


# ─────────────────────────────────────────────────────────────────────────────
# TC-LLM-13: HITL gate — severity >= 9 triggers approval gate
# ─────────────────────────────────────────────────────────────────────────────

@patch("utils.audit_logger.log_action")
def test_llm_tc13_hitl_gate_triggers_at_critical(mock_log):
    """TC-LLM-13: Supervisor node sets awaiting_human=True for severity >= 9."""
    from agents.supervisor_agent import node_hitl_gate

    state = _mock_state()
    state["incident"] = {
        "incident_id": "test-id",
        "severity_score": 9,
        "severity_label": "CRITICAL",
        "classification": "ACCIDENT",
        "zone_id": "Z-05",
        "detected_at": "2026-01-01T10:00:00Z",
    }
    state["awaiting_human"] = False

    result = node_hitl_gate(state)
    assert result["awaiting_human"] is True, "HITL gate must set awaiting_human=True"


# ─────────────────────────────────────────────────────────────────────────────
# TC-LLM-14: HITL gate — severity < 9 does NOT trigger approval gate
# ─────────────────────────────────────────────────────────────────────────────

@patch("utils.audit_logger.log_action")
def test_llm_tc14_hitl_gate_skipped_below_critical(mock_log):
    """TC-LLM-14: Supervisor does NOT set awaiting_human for severity < 9."""
    from agents.supervisor_agent import node_hitl_gate

    state = _mock_state()
    state["incident"] = {
        "incident_id": "test-id",
        "severity_score": 7,
        "severity_label": "HIGH",
        "classification": "INCIDENT",
        "zone_id": "Z-05",
        "detected_at": "2026-01-01T10:00:00Z",
    }
    state["awaiting_human"] = False

    result = node_hitl_gate(state)
    assert result["awaiting_human"] is False, "HITL gate must NOT trigger for severity < 9"


# ─────────────────────────────────────────────────────────────────────────────
# TC-LLM-15: Multi-step reasoning — all required fields present in incident
# ─────────────────────────────────────────────────────────────────────────────

@patch("agents.incident_investigation_agent.log_action")
@patch("agents.incident_investigation_agent.retrieve_regulations")
@patch("agents.incident_investigation_agent.get_weather_for_zone")
@patch("agents.incident_investigation_agent.generate_content")
def test_llm_tc15_incident_schema_completeness(
    mock_gen, mock_wx, mock_rag, mock_log
):
    """TC-LLM-15: Incident output must contain all required schema fields."""
    REQUIRED_FIELDS = {
        "incident_id", "zone_id", "detected_at", "classification",
        "severity_score", "severity_label", "evidence",
        "weather_context", "rag_sources", "reasoning",
    }

    mock_gen.return_value = _mock_llm_response({
        "classification": "INCIDENT",
        "severity_score": 6,
        "evidence": ["e1", "e2", "e3"],
        "reasoning": "Typical rush-hour incident.",
        "confidence": 0.78,
    })
    mock_wx.return_value = _mock_weather()
    mock_rag.return_value = _mock_rag_docs()

    from agents.incident_investigation_agent import run
    state = _mock_state()
    result = run(state)

    incident = result["incident"]
    missing = REQUIRED_FIELDS - set(incident.keys())
    assert not missing, f"Missing required fields: {missing}"
