"""
Congestion Detection Agent — UrbanPulse
Applies threshold-based rules to determine whether a sensor reading
constitutes a traffic anomaly. Uses configurable thresholds from settings.
"""

from __future__ import annotations

from datetime import datetime, timezone

from utils.state_models import TrafficState
from utils.audit_logger import log_action
from utils.config import settings


def run(state: TrafficState) -> TrafficState:
    """
    Purpose:
        Analyse the sensor reading and determine whether an anomaly exists.

    Inputs:  state.sensor_reading
    Outputs: state.anomaly (AnomalyResult TypedDict)

    Detection logic:
        • density > DENSITY_THRESHOLD              → CONGESTION
        • avg_speed_kmh < SPEED_THRESHOLD_KMPH     → SPEED_DROP
        • both conditions true                     → COMBINED (highest risk)
        • neither                                  → NONE
    """
    reading = state.get("sensor_reading")
    session_id = state.get("session_id", "")
    zone_id = state.get("zone_id", "UNKNOWN")

    # Guard: no reading available
    if reading is None:
        anomaly = {
            "detected": False,
            "anomaly_type": "NONE",
            "confidence": 0.0,
            "threshold_density": settings.DENSITY_THRESHOLD,
            "threshold_speed": settings.SPEED_THRESHOLD_KMPH,
        }
        state["anomaly"] = anomaly  # type: ignore[assignment]
        log_action(
            "CongestionDetectionAgent",
            "ANOMALY_SKIPPED_NO_DATA",
            entity_type="zone",
            entity_id=zone_id,
            session_id=session_id,
        )
        return state

    density = reading["density"]
    speed = reading["avg_speed_kmh"]
    density_breach = density > settings.DENSITY_THRESHOLD
    speed_breach = speed < settings.SPEED_THRESHOLD_KMPH

    if density_breach and speed_breach:
        anomaly_type = "COMBINED"
        # Confidence scales with how far beyond thresholds values are
        density_excess = (density - settings.DENSITY_THRESHOLD) / (1.0 - settings.DENSITY_THRESHOLD)
        speed_deficit = (settings.SPEED_THRESHOLD_KMPH - speed) / settings.SPEED_THRESHOLD_KMPH
        confidence = round(min(1.0, (density_excess + speed_deficit) / 2 + 0.5), 3)
        detected = True
    elif density_breach:
        anomaly_type = "CONGESTION"
        confidence = round(
            min(1.0, (density - settings.DENSITY_THRESHOLD) / (1.0 - settings.DENSITY_THRESHOLD) + 0.4),
            3,
        )
        detected = True
    elif speed_breach:
        anomaly_type = "SPEED_DROP"
        confidence = round(
            min(1.0, (settings.SPEED_THRESHOLD_KMPH - speed) / settings.SPEED_THRESHOLD_KMPH + 0.4),
            3,
        )
        detected = True
    else:
        anomaly_type = "NONE"
        confidence = 0.0
        detected = False

    anomaly = {
        "detected": detected,
        "anomaly_type": anomaly_type,
        "confidence": confidence,
        "threshold_density": settings.DENSITY_THRESHOLD,
        "threshold_speed": settings.SPEED_THRESHOLD_KMPH,
    }

    trace_entry = {
        "agent": "CongestionDetectionAgent",
        "action": f"ANOMALY_{anomaly_type}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "zone_id": zone_id,
        "confidence": confidence,
        "density": density,
        "speed": speed,
    }

    log_action(
        "CongestionDetectionAgent",
        f"ANOMALY_{anomaly_type}",
        entity_type="zone",
        entity_id=zone_id,
        payload={
            "detected": detected,
            "confidence": confidence,
            "density": density,
            "speed": speed,
        },
        session_id=session_id,
    )

    state["anomaly"] = anomaly  # type: ignore[assignment]
    state["agent_trace"] = state.get("agent_trace", []) + [trace_entry]
    return state
