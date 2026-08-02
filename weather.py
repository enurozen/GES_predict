"""
Open-Meteo weather API client: historical (archive) data for training, and
forecast data for future dates.

Fetches hourly shortwave radiation (GHI), direct normal irradiance (DNI),
diffuse radiation (DHI), temperature, and cloud cover for a given location
and date range, for use as input features to the GES production model in
ges_uretim_tahmini.py. DNI/DHI let the model compute plane-of-array (POA)
irradiance for tilted/tracked panels instead of assuming a flat horizontal
surface (see ges_uretim_tahmini.poa_irradiance).

No API key required:
    https://archive-api.open-meteo.com/v1/archive             (past dates, actual/reanalysis)
    https://api.open-meteo.com/v1/forecast                     (today + up to ~16 days ahead)
    https://historical-forecast-api.open-meteo.com/v1/forecast (archived past FORECASTS, for honest backtesting)
"""

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from shared import ApiError, request_with_retries

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# Open-Meteo'nun desteklediği model kimlikleri zaman içinde değişebilir -
# güncel liste için open-meteo.com/en/docs'taki "Weather models" bölümüne
# bakın. Burada üç büyük operasyonel NWP merkezi (ECMWF, NOAA, DWD) seçildi -
# tek modele göre sistematik (model-specific) bias'ı azaltmak için.
DEFAULT_ENSEMBLE_MODELS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless"]

REQUEST_TIMEOUT = 15

HOURLY_VARIABLES = (
    "shortwave_radiation,direct_normal_irradiance,diffuse_radiation,"
    "temperature_2m,cloudcover"
)


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
            "dni_forecast": hourly.get("direct_normal_irradiance", []),
            "dhi_forecast": hourly.get("diffuse_radiation", []),
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
        ghi_forecast : global horizontal irradiance, W/m^2
        dni_forecast : direct normal irradiance, W/m^2
        dhi_forecast : diffuse horizontal irradiance, W/m^2
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


def fetch_historical_forecast(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    """Fetch what the weather FORECAST actually said, for a past date range.

    Same columns as fetch_weather_range, but the values come from Open-Meteo's
    historical-forecast-api - archived model output as it was actually
    produced at the time, not the after-the-fact reanalysis that
    fetch_weather_range returns. This is the only honest way to backtest
    predict.py: fetch_weather_range/archive-api tells you what the weather
    WAS, not what a same-day forecast WOULD HAVE SAID, so evaluating against
    it silently assumes a perfect weather forecast and overstates accuracy.
    """
    return _fetch_hourly(HISTORICAL_FORECAST_URL, lat, lon, start, end)


def fetch_weather_forecast_ensemble(lat: float, lon: float, start: date, end: date,
                                     models: list[str] | None = None) -> pd.DataFrame:
    """
    Aynı anda birden fazla NWP modelinden (varsayılan: ECMWF, GFS, ICON) gün-
    öncesi hava tahmini çeker; ensemble ORTALAMASINI ve model-arası standart
    sapmasını (belirsizlik göstergesi) döner. Open-Meteo bunu ek ücret/kimlik
    bilgisi gerektirmeden, tek istekte `models` parametresiyle sağlıyor - tek
    modele bağımlı kalmaktan kaynaklanan sistematik (model-specific) hatayı
    azaltmanın en ucuz yolu bu.

    Returns aynı sütunlar (ghi_forecast, dni_forecast, dhi_forecast, temp_c,
    cloud_cover) + her biri için *_std belirsizlik sütunu (ör. ghi_forecast_std).
    Bir modelin verisi eksikse (ör. model kimliği değişmişse) o model sessizce
    ortalamadan çıkarılır; hiçbiri bulunamazsa suffix'siz tek-model anahtarına
    (fetch_weather_forecast'ın kullandığı) düşülür.

    NOT: Sadece api.open-meteo.com/v1/forecast (gelecek tarihler) çoklu model
    destekliyor - archive/historical-forecast endpoint'leri tek model/
    reanalysis döner, bu fonksiyon onlarla kullanılmaz.
    """
    if models is None:
        models = DEFAULT_ENSEMBLE_MODELS

    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": HOURLY_VARIABLES,
        "models": ",".join(models),
        "timezone": "auto",
    }

    response = request_with_retries(
        "GET",
        FORECAST_URL,
        timeout=REQUEST_TIMEOUT,
        error_context=f"Could not fetch ensemble weather data for ({lat}, {lon})",
        params=params,
    )

    if not response.ok:
        raise ApiError(
            f"Open-Meteo API returned an error (HTTP {response.status_code}) "
            f"for ({lat}, {lon})."
        )

    body = response.json()
    hourly = body.get("hourly", {})
    timestamps = pd.to_datetime(hourly.get("time", []))
    n = len(timestamps)

    result: dict[str, Any] = {"timestamp": timestamps}
    for var_name, out_col, pct_to_fraction in [
        ("shortwave_radiation", "ghi_forecast", False),
        ("direct_normal_irradiance", "dni_forecast", False),
        ("diffuse_radiation", "dhi_forecast", False),
        ("temperature_2m", "temp_c", False),
        ("cloudcover", "cloud_cover", True),
    ]:
        per_model = [
            hourly[key] for model in models
            if (key := f"{var_name}_{model}") in hourly
        ]
        if not per_model and var_name in hourly:
            # Model kimlikleri tanınmadı (ör. Open-Meteo'nun model listesi
            # değişmiş) - suffix'siz tek-model anahtarına düş.
            per_model = [hourly[var_name]]

        if per_model:
            arr = np.array(per_model, dtype=float)
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
            if pct_to_fraction:
                mean, std = mean / 100.0, std / 100.0
        else:
            mean = np.full(n, np.nan)
            std = np.full(n, np.nan)

        result[out_col] = mean
        result[f"{out_col}_std"] = std

    return pd.DataFrame(result)
