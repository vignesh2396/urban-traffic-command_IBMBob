"""
Traffic Monitoring Agent — UrbanPulse
Reads the latest sensor data for a zone and populates the shared state.
This is the first agent in the pipeline.
"""

from __future__ import annotations

from datetime import datetime, timezone

from utils.state_models import TrafficState
from utils.audit_logger import log_action
from tools.sensor_tools import get_latest_sensor_reading, simulate_sensor_reading


def run(state: TrafficState) -> TrafficState:
    """
    Purpose:
        Ingest the most recent traffic sensor reading for the target zone.
        If no live reading exists, generate a simulated one.

    Inputs:  state.zone_id
    Outputs: state.sensor_reading (SensorReading TypedDict)

    Responsible for:
        • Fetching or simulating sensor data
        • Recording a trace entry in agent_trace
        • Logging to audit trail
    """
    zone_id = state["zone_id"]
    session_id = state.get("session_id", "")

    trace_entry = {
        "agent": "TrafficMonitoringAgent",
        "action": "SENSOR_READ_ATTEMPT",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "zone_id": zone_id,
    }

    # Attempt to read from DB; fall back to simulation
    reading = get_latest_sensor_reading(zone_id)

    if reading is None:
        reading = simulate_sensor_reading(zone_id)
        trace_entry["action"] = "SENSOR_READ_SIMULATED"
    else:
        trace_entry["action"] = "SENSOR_READ_SUCCESS"

    trace_entry["density"] = reading["density"]
    trace_entry["avg_speed_kmh"] = reading["avg_speed_kmh"]

    log_action(
        agent_name="TrafficMonitoringAgent",
        action=trace_entry["action"],
        entity_type="zone",
        entity_id=zone_id,
        payload={
            "density": reading["density"],
            "avg_speed_kmh": reading["avg_speed_kmh"],
            "vehicle_count": reading.get("vehicle_count"),
        },
        session_id=session_id,
    )

    state["sensor_reading"] = reading  # type: ignore[assignment]
    state["agent_trace"] = state.get("agent_trace", []) + [trace_entry]
    return state
