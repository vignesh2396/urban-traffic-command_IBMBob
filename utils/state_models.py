"""
State Models — UrbanPulse LangGraph TypedDicts
Defines the shared state passed between all nodes in the StateGraph.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from typing_extensions import TypedDict


class SensorReading(TypedDict):
    zone_id: str
    timestamp: str
    density: float            # 0.0–1.0
    avg_speed_kmh: float
    vehicle_count: int
    sensor_source: str


class AnomalyResult(TypedDict):
    detected: bool
    anomaly_type: Literal["NONE", "CONGESTION", "SPEED_DROP", "COMBINED"]
    confidence: float
    threshold_density: float
    threshold_speed: float


class WeatherContext(TypedDict):
    description: str
    temperature_c: float
    humidity_pct: int
    wind_speed_ms: float
    visibility_m: int
    is_adverse: bool          # rain/fog/storm flag


class IncidentReport(TypedDict):
    incident_id: str
    zone_id: str
    detected_at: str
    classification: Literal["NORMAL", "CONGESTION", "INCIDENT", "ACCIDENT"]
    severity_score: int       # 1–10
    severity_label: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    evidence: list[str]       # bullet-point evidence strings
    weather_context: Optional[WeatherContext]
    rag_sources: list[str]
    reasoning: str            # LLM-generated explanation


class DiversionRoute(TypedDict):
    route_id: str
    description: str
    estimated_delay_min: int
    confidence: float


class SignalChange(TypedDict):
    intersection_id: str
    action: str               # e.g. "extend_green_30s"
    justification: str


class ResponsePlan(TypedDict):
    plan_id: str
    incident_id: str
    diversion_routes: list[DiversionRoute]
    signal_changes: list[SignalChange]
    commuter_alert: Optional[str]
    authority_alert: Optional[str]
    confidence_score: float
    requires_approval: bool
    created_at: str


class HITLDecision(TypedDict):
    decision_id: str
    plan_id: str
    operator_id: str
    decision: Literal["APPROVED", "REJECTED", "MODIFIED"]
    notes: Optional[str]
    decided_at: str


# ── LangGraph shared state ────────────────────────────────────────────────────

class TrafficState(TypedDict):
    """State object threaded through all LangGraph nodes."""

    # Input
    zone_id: str
    session_id: str

    # Populated by Traffic Monitoring Agent
    sensor_reading: Optional[SensorReading]

    # Populated by Congestion Detection Agent
    anomaly: Optional[AnomalyResult]

    # Populated by Incident Investigation Agent
    incident: Optional[IncidentReport]

    # Populated by Route Diversion Agent
    diversion_routes: list[DiversionRoute]

    # Populated by Signal Optimization Agent
    signal_changes: list[SignalChange]

    # Populated by Traffic Response Coordination Agent
    response_plan: Optional[ResponsePlan]

    # HITL gate
    hitl_decision: Optional[HITLDecision]
    awaiting_human: bool

    # Errors / diagnostics
    errors: list[str]
    pipeline_start: str
    pipeline_end: Optional[str]
    agent_trace: list[dict[str, Any]]   # [{agent, action, timestamp}]
