"""
Open-Meteo weather API client: historical (archive) data for training, and
forecast data for future dates.

Fetches hourly shortwave radiation (GHI), temperature, and cloud cover for a
given location and date range, for use as input features to the GES
production model in ges_uretim_tahmini.py (ghi_forecast, temp_c, cloud_cover).

No API key required:
    https://archive-api.open-meteo.com/v1/archive  (past dates)
    https://api.open-meteo.com/v1/forecast          (today + up to ~16 days ahead)
"""

from datetime import date
from typing import Any

import pandas as pd

from shared import ApiError, request_with_retries

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

REQUEST_TIMEOUT = 15

HOURLY_VARIABLES = "shortwave_radiation,temperature_2m,cloudcover"


def _fetch_hourly(url: str, lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": HOURLY_VARIABLES,
        # "auto" resolves to the plant's local timezone (e.g. Europe/Istanbul),
        # matching the local timestamps EPİAŞ generation data uses.
        "timezone": "auto",
    }

    response = request_with_retries(
        "GET",
        url,
        timeout=REQUEST_TIMEOUT,
        error_context=f"Could not fetch weather data for ({lat}, {lon})",
        params=params,
    )

    if not response.ok:
        raise ApiError(
            f"Open-Meteo API returned an error (HTTP {response.status_code}) "
            f"for ({lat}, {lon})."
        )

    body = response.json()
    hourly = body.get("hourly", {})
    cloud_cover_pct = hourly.get("cloudcover", [])

    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(hourly.get("time", [])),
            "ghi_forecast": hourly.get("shortwave_radiation", []),
            "temp_c": hourly.get("temperature_2m", []),
            "cloud_cover": [
                v / 100.0 if v is not None else None for v in cloud_cover_pct
            ],
        }
    )


def fetch_weather_range(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    """Fetch hourly HISTORICAL weather data for [start, end] at (lat, lon).

    Returns a DataFrame with columns:
        timestamp    : datetime
        ghi_forecast : shortwave (global horizontal) radiation, W/m^2
        temp_c       : ambient temperature, °C
        cloud_cover  : cloud cover fraction, 0-1 (Open-Meteo returns %, normalized here)
    """
    return _fetch_hourly(ARCHIVE_URL, lat, lon, start, end)


def fetch_weather_forecast(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    """Fetch hourly FORECAST weather data for [start, end] at (lat, lon).

    Same columns as fetch_weather_range. Open-Meteo's forecast endpoint only
    covers today through roughly the next 16 days - for dates further out or
    in the past, use fetch_weather_range instead.
    """
    return _fetch_hourly(FORECAST_URL, lat, lon, start, end)
