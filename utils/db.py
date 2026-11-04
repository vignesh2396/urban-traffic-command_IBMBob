"""
Database Utility — UrbanPulse
Thread-safe SQLite connection factory + schema initialisation.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from utils.config import settings

_lock = threading.local()


def _get_connection() -> sqlite3.Connection:
    """Return a thread-local SQLite connection (creates if not present)."""
    if not hasattr(_lock, "conn"):
        db_path = Path(settings.DB_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _lock.conn = conn
    return _lock.conn


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Context manager for database access — auto-commits or rolls back."""
    conn = _get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db() -> None:
    """Create all tables if they do not exist. Safe to call on every startup."""
    ddl = """
    CREATE TABLE IF NOT EXISTS zones (
        zone_id       TEXT PRIMARY KEY,
        zone_name     TEXT NOT NULL,
        latitude      REAL,
        longitude     REAL,
        intersections INTEGER DEFAULT 0,
        road_type     TEXT,
        created_at    TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS sensor_readings (
        reading_id    INTEGER PRIMARY KEY AUTOINCREMENT,
        zone_id       TEXT NOT NULL REFERENCES zones(zone_id),
        timestamp     TEXT NOT NULL,
        density       REAL NOT NULL,
        avg_speed_kmh REAL NOT NULL,
        vehicle_count INTEGER,
        sensor_source TEXT DEFAULT 'simulated'
    );

    CREATE INDEX IF NOT EXISTS idx_sr_zone_time
        ON sensor_readings(zone_id, timestamp);

    CREATE TABLE IF NOT EXISTS incidents (
        incident_id    TEXT PRIMARY KEY,
        zone_id        TEXT NOT NULL REFERENCES zones(zone_id),
        detected_at    TEXT NOT NULL,
        classification TEXT NOT NULL,
        severity_score INTEGER NOT NULL,
        severity_label TEXT NOT NULL,
        weather_context TEXT,
        rag_sources    TEXT,
        reasoning      TEXT,
        status         TEXT DEFAULT 'OPEN',
        resolved_at    TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_inc_zone   ON incidents(zone_id);
    CREATE INDEX IF NOT EXISTS idx_inc_status ON incidents(status);

    CREATE TABLE IF NOT EXISTS response_plans (
        plan_id          TEXT PRIMARY KEY,
        incident_id      TEXT NOT NULL REFERENCES incidents(incident_id),
        diversion_routes TEXT NOT NULL,
        signal_changes   TEXT NOT NULL,
        commuter_alert   TEXT,
        authority_alert  TEXT,
        confidence_score REAL,
        requires_approval INTEGER DEFAULT 0,
        created_at       TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS hitl_decisions (
        decision_id TEXT PRIMARY KEY,
        plan_id     TEXT NOT NULL REFERENCES response_plans(plan_id),
        operator_id TEXT NOT NULL,
        decision    TEXT NOT NULL,
        notes       TEXT,
        decided_at  TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS audit_log (
        log_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT NOT NULL,
        agent_name  TEXT NOT NULL,
        action      TEXT NOT NULL,
        entity_type TEXT,
        entity_id   TEXT,
        payload     TEXT,
        human_actor TEXT,
        session_id  TEXT
    );
    """
    with get_db() as conn:
        conn.executescript(ddl)
