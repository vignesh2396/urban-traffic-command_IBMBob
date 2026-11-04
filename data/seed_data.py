"""
Seed Data Script — UrbanPulse
Creates the SQLite database and populates it with realistic synthetic data
for demonstration purposes. Safe to run multiple times (idempotent).

Usage:
    python -m data.seed_data
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from utils.db import init_db, get_db


ZONE_DEFINITIONS = [
    ("Z-01", "Central Business District",   12.9716, 77.5946, 8, "arterial"),
    ("Z-02", "South Extension",             12.9352, 77.6245, 6, "arterial"),
    ("Z-03", "North Corridor",              13.0067, 77.5570, 5, "local"),
    ("Z-04", "East Industrial Zone",        12.9082, 77.6476, 4, "arterial"),
    ("Z-05", "Airport Access Road",         12.9698, 77.7500, 3, "highway"),
    ("Z-06", "Ring Road North",             13.0297, 77.6028, 6, "highway"),
    ("Z-07", "Old Town Market Area",        12.8456, 77.6603, 7, "local"),
    ("Z-08", "Tech Park West",              12.9121, 77.5390, 5, "arterial"),
    ("Z-09", "University Campus Junction",  13.0712, 77.5783, 4, "local"),
    ("Z-10", "Stadium Road",               12.9762, 77.6033, 6, "arterial"),
]


def seed() -> None:
    """Populate all tables with synthetic demonstration data."""
    init_db()

    with get_db() as conn:
        # ── Zones ─────────────────────────────────────────────────────────────
        existing = conn.execute("SELECT COUNT(*) FROM zones").fetchone()[0]
        if existing == 0:
            conn.executemany(
                """
                INSERT INTO zones
                    (zone_id, zone_name, latitude, longitude, intersections, road_type)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                ZONE_DEFINITIONS,
            )
            print(f"  ✅ Seeded {len(ZONE_DEFINITIONS)} zones")
        else:
            print(f"  ℹ️  Zones already seeded ({existing} records)")

        # ── Sensor Readings — 48 hours of simulated history ───────────────────
        existing_readings = conn.execute("SELECT COUNT(*) FROM sensor_readings").fetchone()[0]
        if existing_readings == 0:
            now = datetime.now(timezone.utc)
            rows = []
            for zone_id, *_ in ZONE_DEFINITIONS:
                for h in range(48):
                    for m in (0, 15, 30, 45):
                        ts = now - timedelta(hours=h, minutes=m)
                        hour_of_day = ts.hour
                        # Simulate rush hour peaks at 8–10am and 5–7pm
                        if hour_of_day in range(8, 10) or hour_of_day in range(17, 20):
                            density = round(random.uniform(0.65, 0.95), 2)
                            speed = round(random.uniform(5, 25), 1)
                        elif hour_of_day in range(0, 5):
                            density = round(random.uniform(0.02, 0.15), 2)
                            speed = round(random.uniform(50, 80), 1)
                        else:
                            density = round(random.uniform(0.2, 0.60), 2)
                            speed = round(random.uniform(25, 60), 1)
                        rows.append(
                            (
                                zone_id,
                                ts.isoformat(),
                                density,
                                speed,
                                int(density * 250),
                                "seeded",
                            )
                        )

            conn.executemany(
                """
                INSERT INTO sensor_readings
                    (zone_id, timestamp, density, avg_speed_kmh, vehicle_count, sensor_source)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            print(f"  ✅ Seeded {len(rows)} sensor readings (48-hour history)")
        else:
            print(f"  ℹ️  Sensor readings already seeded ({existing_readings} records)")

    print("✅ Seed data complete.")


if __name__ == "__main__":
    seed()
