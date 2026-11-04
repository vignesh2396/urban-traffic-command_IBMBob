"""
Sensor Tools — UrbanPulse
Reads traffic sensor readings from the SQLite database.
Includes a simulation helper that generates synthetic readings for zones.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from typing import Optional

from utils.db import get_db
from utils.config import settings
from utils.audit_logger import log_action


SENSOR_SOURCE_SIMULATED = "simulated"


def get_latest_sensor_reading(zone_id: str) -> Optional[dict]:
    """
    Retrieve the most recent sensor reading for a zone.

    Returns:
        Dict with keys: zone_id, timestamp, density, avg_speed_kmh,
        vehicle_count, sensor_source — or None if zone has no readings.
    """
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT * FROM sensor_readings
            WHERE zone_id = ?
            ORDER BY timestamp DESC
            LIMIT 1
            """,
            (zone_id,),
        ).fetchone()

    if row is None:
        log_action(
            agent_name="SensorTool",
            action="SENSOR_DATA_MISSING",
            entity_type="zone",
            entity_id=zone_id,
            payload={"reason": "No sensor records found for zone"},
        )
        return None

    return dict(row)


def get_all_zone_statuses() -> list[dict]:
    """
    Return the latest sensor reading for every zone in the system.
    Used by the dashboard to show a zone-wide overview.
    """
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT sr.*
            FROM sensor_readings sr
            INNER JOIN (
                SELECT zone_id, MAX(timestamp) AS max_ts
                FROM sensor_readings
                GROUP BY zone_id
            ) latest ON sr.zone_id = latest.zone_id
                     AND sr.timestamp = latest.max_ts
            ORDER BY sr.zone_id
            """,
        ).fetchall()
    return [dict(r) for r in rows]


def simulate_sensor_reading(
    zone_id: str,
    scenario: Optional[str] = None,
) -> dict:
    """
    Generate a synthetic sensor reading and persist it.

    Args:
        zone_id:  Target zone identifier.
        scenario: Optional override — 'normal' | 'congestion' | 'accident' | 'missing'

    Returns:
        The generated sensor reading dict.
    """
    if scenario == "accident":
        density = round(random.uniform(0.88, 0.99), 2)
        speed = round(random.uniform(1.0, 8.0), 1)
        count = random.randint(180, 250)
    elif scenario == "congestion":
        density = round(random.uniform(0.76, 0.88), 2)
        speed = round(random.uniform(8.0, 20.0), 1)
        count = random.randint(120, 180)
    elif scenario == "normal":
        density = round(random.uniform(0.1, 0.65), 2)
        speed = round(random.uniform(30.0, 70.0), 1)
        count = random.randint(20, 100)
    else:
        # Random realistic reading weighted toward normal
        density = round(random.triangular(0.05, 1.0, 0.35), 2)
        speed = round(max(1.0, random.gauss(45, 20)), 1)
        count = int(density * 250)

    timestamp = datetime.now(timezone.utc).isoformat()

    reading = {
        "zone_id": zone_id,
        "timestamp": timestamp,
        "density": density,
        "avg_speed_kmh": speed,
        "vehicle_count": count,
        "sensor_source": SENSOR_SOURCE_SIMULATED,
    }

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO sensor_readings
                (zone_id, timestamp, density, avg_speed_kmh, vehicle_count, sensor_source)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                zone_id,
                timestamp,
                density,
                speed,
                count,
                SENSOR_SOURCE_SIMULATED,
            ),
        )

    log_action(
        agent_name="SensorTool",
        action="SENSOR_READING_GENERATED",
        entity_type="zone",
        entity_id=zone_id,
        payload={"density": density, "speed": speed, "scenario": scenario},
    )

    return reading
