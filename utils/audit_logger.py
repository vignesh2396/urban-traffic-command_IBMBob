"""
Audit Logger — UrbanPulse
Writes a structured, immutable audit trail for every agent action,
tool call, and human decision.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from utils.config import settings
from utils.db import get_db


# ── File logger setup ─────────────────────────────────────────────────────────

def _setup_file_logger() -> logging.Logger:
    logger = logging.getLogger("urbanpulse")
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        # Always attach a console handler so logging works even if the file fails.
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # Attempt to attach a rotating file handler; skip gracefully on any
        # permission error (common when the project lives inside OneDrive).
        try:
            log_path = Path(settings.LOG_FILE)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=5_000_000,
                backupCount=3,
                encoding="utf-8",
                delay=True,   # defer actual open until first write
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except (PermissionError, OSError) as exc:
            logger.warning(
                "Could not create log file at %s (%s). "
                "Logging to console only.",
                settings.LOG_FILE,
                exc,
            )

    logger.setLevel(getattr(logging, settings.LOG_LEVEL, logging.INFO))
    return logger


_logger = _setup_file_logger()


# ── Audit log writer ──────────────────────────────────────────────────────────

def log_action(
    agent_name: str,
    action: str,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    payload: Optional[dict[str, Any]] = None,
    human_actor: Optional[str] = None,
    session_id: Optional[str] = None,
) -> None:
    """
    Persist an audit entry to the database AND emit a structured log line.

    Args:
        agent_name:  Name of the agent or system component performing the action.
        action:      Short verb describing what happened (e.g. "INCIDENT_DETECTED").
        entity_type: Optional entity class (e.g. "incident", "response_plan").
        entity_id:   Optional ID of the entity affected.
        payload:     Optional JSON-serialisable dict with details.
        human_actor: Operator ID if a human initiated or overrode the action.
        session_id:  LangGraph session / conversation ID for traceability.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    payload_json = json.dumps(payload, default=str) if payload else None

    # Write to database
    try:
        with get_db() as conn:
            conn.execute(
                """
                INSERT INTO audit_log
                    (timestamp, agent_name, action, entity_type,
                     entity_id, payload, human_actor, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    agent_name,
                    action,
                    entity_type,
                    entity_id,
                    payload_json,
                    human_actor,
                    session_id,
                ),
            )
    except Exception as exc:
        _logger.error("audit_log DB write failed: %s", exc)

    # Emit log line
    _logger.info(
        "AUDIT | agent=%s | action=%s | entity=%s/%s | actor=%s | session=%s",
        agent_name,
        action,
        entity_type,
        entity_id,
        human_actor,
        session_id,
    )


def get_audit_log(
    limit: int = 200,
    agent_name: Optional[str] = None,
    session_id: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Retrieve recent audit log entries for UI display."""
    query = "SELECT * FROM audit_log WHERE 1=1"
    params: list[Any] = []
    if agent_name:
        query += " AND agent_name = ?"
        params.append(agent_name)
    if session_id:
        query += " AND session_id = ?"
        params.append(session_id)
    query += " ORDER BY log_id DESC LIMIT ?"
    params.append(limit)

    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]
