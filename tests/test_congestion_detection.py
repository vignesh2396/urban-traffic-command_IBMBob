"""
Unit Tests — Congestion Detection Agent
Tests the threshold-based anomaly detection logic.
"""

import pytest
from unittest.mock import patch

from utils.state_models import TrafficState
from utils.config import settings


def _make_state(density: float, speed: float, zone_id: str = "Z-01") -> TrafficState:
    return {
        "zone_id": zone_id,
        "session_id": "test-session",
        "sensor_reading": {
            "zone_id": zone_id,
            "timestamp": "2026-01-01T10:00:00Z",
            "density": density,
            "avg_speed_kmh": speed,
            "vehicle_count": int(density * 250),
            "sensor_source": "test",
        },
        "anomaly": None,
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


@patch("utils.audit_logger.log_action")
def test_normal_traffic_no_anomaly(mock_log):
    from agents.congestion_detection_agent import run
    state = _make_state(density=0.30, speed=55.0)
    result = run(state)
    anomaly = result["anomaly"]
    assert anomaly["detected"] is False
    assert anomaly["anomaly_type"] == "NONE"
    assert anomaly["confidence"] == 0.0


@patch("utils.audit_logger.log_action")
def test_high_density_congestion_detected(mock_log):
    from agents.congestion_detection_agent import run
    state = _make_state(density=0.85, speed=40.0)  # only density breach
    result = run(state)
    anomaly = result["anomaly"]
    assert anomaly["detected"] is True
    assert anomaly["anomaly_type"] == "CONGESTION"
    assert anomaly["confidence"] > 0.4


@patch("utils.audit_logger.log_action")
def test_low_speed_speed_drop_detected(mock_log):
    from agents.congestion_detection_agent import run
    state = _make_state(density=0.40, speed=10.0)  # only speed breach
    result = run(state)
    anomaly = result["anomaly"]
    assert anomaly["detected"] is True
    assert anomaly["anomaly_type"] == "SPEED_DROP"


@patch("utils.audit_logger.log_action")
def test_combined_anomaly_highest_risk(mock_log):
    from agents.congestion_detection_agent import run
    state = _make_state(density=0.95, speed=3.0)  # both breached
    result = run(state)
    anomaly = result["anomaly"]
    assert anomaly["detected"] is True
    assert anomaly["anomaly_type"] == "COMBINED"
    assert anomaly["confidence"] >= 0.7


@patch("utils.audit_logger.log_action")
def test_missing_sensor_reading_handled(mock_log):
    from agents.congestion_detection_agent import run
    state = _make_state(density=0.5, speed=30.0)
    state["sensor_reading"] = None
    result = run(state)
    anomaly = result["anomaly"]
    assert anomaly["detected"] is False
    assert anomaly["anomaly_type"] == "NONE"


@patch("utils.audit_logger.log_action")
def test_agent_trace_populated(mock_log):
    from agents.congestion_detection_agent import run
    state = _make_state(density=0.90, speed=5.0)
    result = run(state)
    assert len(result["agent_trace"]) >= 1
    assert result["agent_trace"][-1]["agent"] == "CongestionDetectionAgent"
