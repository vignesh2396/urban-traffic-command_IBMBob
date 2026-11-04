"""
Weather Tools — UrbanPulse
Fetches current weather conditions for a zone's coordinates.
Falls back to a safe default if the API is unavailable (graceful degradation).
"""

from __future__ import annotations

import logging
from typing import Optional

import requests

from utils.config import settings
from utils.audit_logger import log_action

logger = logging.getLogger("urbanpulse")

# Zones → approximate coordinates (city-agnostic placeholder)
# In production, these would be stored in the `zones` DB table.
ZONE_COORDINATES: dict[str, tuple[float, float]] = {
    "Z-01": (12.9716, 77.5946),   # Bengaluru centre
    "Z-02": (12.9352, 77.6245),
    "Z-03": (13.0067, 77.5570),
    "Z-04": (12.9082, 77.6476),
    "Z-05": (12.9698, 77.7500),
    "Z-06": (13.0297, 77.6028),
    "Z-07": (12.8456, 77.6603),
    "Z-08": (12.9121, 77.5390),
    "Z-09": (13.0712, 77.5783),
    "Z-10": (12.9762, 77.6033),
}

_ADVERSE_KEYWORDS = {
    "rain", "drizzle", "thunderstorm", "snow", "sleet", "mist",
    "fog", "haze", "smoke", "dust", "sand", "tornado",
}


def get_weather_for_zone(zone_id: str) -> dict:
    """
    Fetch current weather context for a zone.

    Returns a WeatherContext-compatible dict. On API failure, returns a
    safe default so the pipeline continues without crashing.
    """
    coords = ZONE_COORDINATES.get(zone_id, (12.9716, 77.5946))
    lat, lon = coords

    if not settings.OPENWEATHER_API_KEY:
        logger.warning("OPENWEATHER_API_KEY not set — using default weather context")
        return _default_weather_context()

    try:
        response = requests.get(
            settings.OPENWEATHER_BASE_URL,
            params={
                "lat": lat,
                "lon": lon,
                "appid": settings.OPENWEATHER_API_KEY,
                "units": "metric",
            },
            timeout=5,
        )
        response.raise_for_status()
        data = response.json()

        description = data["weather"][0]["description"].lower()
        is_adverse = any(kw in description for kw in _ADVERSE_KEYWORDS)

        result = {
            "description": description,
            "temperature_c": data["main"]["temp"],
            "humidity_pct": data["main"]["humidity"],
            "wind_speed_ms": data["wind"]["speed"],
            "visibility_m": data.get("visibility", 10000),
            "is_adverse": is_adverse,
        }

        log_action(
            agent_name="WeatherTool",
            action="WEATHER_FETCHED",
            entity_type="zone",
            entity_id=zone_id,
            payload={"description": description, "is_adverse": is_adverse},
        )
        return result

    except requests.exceptions.Timeout:
        logger.warning("Weather API timeout for zone %s — using default", zone_id)
        log_action(
            agent_name="WeatherTool",
            action="WEATHER_API_TIMEOUT",
            entity_type="zone",
            entity_id=zone_id,
        )
        return _default_weather_context()

    except requests.exceptions.RequestException as exc:
        logger.error("Weather API error for zone %s: %s", zone_id, exc)
        log_action(
            agent_name="WeatherTool",
            action="WEATHER_API_ERROR",
            entity_type="zone",
            entity_id=zone_id,
            payload={"error": str(exc)},
        )
        return _default_weather_context()


def _default_weather_context() -> dict:
    """Return a safe, non-adverse weather context when API is unavailable."""
    return {
        "description": "data unavailable",
        "temperature_c": 25.0,
        "humidity_pct": 60,
        "wind_speed_ms": 2.0,
        "visibility_m": 10000,
        "is_adverse": False,
    }
