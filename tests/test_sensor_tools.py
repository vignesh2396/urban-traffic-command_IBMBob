"""
Unit Tests — Sensor Tools
Tests DB reads, simulation, and graceful handling of missing zones.
"""

import pytest
from unittest.mock import patch, MagicMock


@patch("utils.audit_logger.log_action")
@patch("tools.sensor_tools.get_db")
def test_simulate_creates_reading(mock_get_db, mock_log):
    from tools.sensor_tools import simulate_sensor_reading

    mock_conn = MagicMock()
    mock_conn.__enter__ = lambda s: mock_conn
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_get_db.return_value = mock_conn

    reading = simulate_sensor_reading("Z-01", scenario="normal")

    assert reading["zone_id"] == "Z-01"
    assert 0.0 <= reading["density"] <= 1.0
    assert reading["avg_speed_kmh"] >= 1.0
    assert reading["sensor_source"] == "simulated"
    assert "timestamp" in reading


@patch("utils.audit_logger.log_action")
@patch("tools.sensor_tools.get_db")
def test_simulate_accident_scenario_high_density(mock_get_db, mock_log):
    from tools.sensor_tools import simulate_sensor_reading

    mock_conn = MagicMock()
    mock_conn.__enter__ = lambda s: mock_conn
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_get_db.return_value = mock_conn

    reading = simulate_sensor_reading("Z-07", scenario="accident")
    assert reading["density"] >= 0.88
    assert reading["avg_speed_kmh"] <= 8.0


@patch("tools.sensor_tools.log_action")
@patch("tools.sensor_tools.get_db")
def test_get_latest_returns_none_for_missing_zone(mock_get_db, mock_log):
    from tools.sensor_tools import get_latest_sensor_reading

    mock_conn = MagicMock()
    mock_conn.__enter__ = lambda s: mock_conn
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_conn.execute.return_value.fetchone.return_value = None
    mock_get_db.return_value = mock_conn

    result = get_latest_sensor_reading("Z-99")
    assert result is None
    # Verify audit log called with SENSOR_DATA_MISSING
    mock_log.assert_called_once()
    call_kwargs = mock_log.call_args[1] if mock_log.call_args[1] else {}
    assert call_kwargs.get("action") == "SENSOR_DATA_MISSING"
