"""
Configuration Management — UrbanPulse
Loads all settings from environment variables (via .env).
Fails fast with clear messages if required variables are missing.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root (works in both local and Docker)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)


def _require(key: str) -> str:
    """Return env var value or raise ValueError with actionable message."""
    value = os.getenv(key)
    if not value:
        raise ValueError(
            f"Required environment variable '{key}' is not set. "
            f"Copy .env.example to .env and provide the value."
        )
    return value


class Settings:
    """Centralised application settings. Import the singleton `settings`."""

    # --- LLM ---
    GEMINI_API_KEY: str = _require("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL") or "models/gemini-2.5-flash"

    # --- Weather ---
    OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
    OPENWEATHER_BASE_URL: str = "https://api.openweathermap.org/data/2.5/weather"

    # --- Database ---
    # Default to %LOCALAPPDATA% so SQLite/ChromaDB files are never inside
    # OneDrive (which applies Deny ACLs that block file creation).
    # Use `or` so that empty-string values in .env also fall back to the default.
    _data_dir: Path = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "urbanpulse"
    DB_PATH: str = os.getenv("DB_PATH") or str(_data_dir / "traffic_data.db")

    # --- ChromaDB ---
    CHROMA_PATH: str = os.getenv("CHROMA_PATH") or str(_data_dir / "chroma_store")

    # --- Thresholds ---
    DENSITY_THRESHOLD: float = float(os.getenv("DENSITY_THRESHOLD", "0.75"))
    SPEED_THRESHOLD_KMPH: float = float(os.getenv("SPEED_THRESHOLD_KMPH", "20.0"))
    CRITICAL_SEVERITY: int = int(os.getenv("CRITICAL_SEVERITY", "9"))
    HIGH_SEVERITY: int = int(os.getenv("HIGH_SEVERITY", "7"))

    # --- Application ---
    APP_TITLE: str = "UrbanPulse — Traffic Command & Incident Response"
    APP_VERSION: str = "1.0.0"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL") or "INFO"
    LOG_FILE: str = os.getenv("LOG_FILE") or str(_data_dir / "logs" / "app.log")

    # --- RAG ---
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "3"))
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "models/embedding-001"
    )  # Gemini embedding


settings = Settings()
