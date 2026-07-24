import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import weather
from shared import ApiError


def _mock_response(status_code, json_data=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.ok = status_code < 400
    if json_data is not None:
        resp.json.return_value = json_data
    return resp


def _sample_body():
    return {
        "hourly": {
            "time": ["2026-07-01T00:00", "2026-07-01T01:00"],
            "shortwave_radiation": [0.0, 120.5],
            "temperature_2m": [18.2, 18.9],
            "cloudcover": [80, 40],
        }
    }


# --------------------------------------------------------------------------
# fetch_weather_range
# --------------------------------------------------------------------------

def test_fetch_weather_range_success():
    resp = _mock_response(200, json_data=_sample_body())
    with patch("shared.requests.request", return_value=resp) as mock_request:
        df = weather.fetch_weather_range(39.9, 32.8, date(2026, 7, 1), date(2026, 7, 1))

    assert list(df.columns) == ["timestamp", "ghi_forecast", "temp_c", "cloud_cover"]
    assert list(df["ghi_forecast"]) == [0.0, 120.5]
    assert list(df["temp_c"]) == [18.2, 18.9]
    assert list(df["cloud_cover"]) == [0.8, 0.4]

    called_kwargs = mock_request.call_args.kwargs
    assert called_kwargs["params"]["latitude"] == 39.9
    assert called_kwargs["params"]["longitude"] == 32.8
    assert called_kwargs["params"]["start_date"] == "2026-07-01"
    assert called_kwargs["params"]["hourly"] == weather.HOURLY_VARIABLES


def test_fetch_weather_range_http_client_error():
    resp = _mock_response(400)
    with patch("shared.requests.request", return_value=resp):
        with pytest.raises(ApiError, match="Open-Meteo"):
            weather.fetch_weather_range(39.9, 32.8, date(2026, 7, 1), date(2026, 7, 1))


def test_fetch_weather_range_retries_on_5xx_then_succeeds():
    responses = [
        _mock_response(500),
        _mock_response(500),
        _mock_response(200, json_data=_sample_body()),
    ]
    with patch("shared.requests.request", side_effect=responses), patch("shared.time.sleep"):
        df = weather.fetch_weather_range(39.9, 32.8, date(2026, 7, 1), date(2026, 7, 1))

    assert len(df) == 2


def test_fetch_weather_range_connection_error_does_not_retry_forever():
    with patch(
        "shared.requests.request",
        side_effect=requests.exceptions.ConnectionError("no route"),
    ), patch("shared.time.sleep"):
        with pytest.raises(ApiError, match="Could not fetch weather data"):
            weather.fetch_weather_range(39.9, 32.8, date(2026, 7, 1), date(2026, 7, 1))


# --------------------------------------------------------------------------
# fetch_weather_forecast
# --------------------------------------------------------------------------

def test_fetch_weather_forecast_hits_forecast_url_not_archive():
    resp = _mock_response(200, json_data=_sample_body())
    with patch("shared.requests.request", return_value=resp) as mock_request:
        df = weather.fetch_weather_forecast(39.9, 32.8, date(2026, 7, 22), date(2026, 7, 24))

    assert len(df) == 2
    called_args = mock_request.call_args.args
    assert called_args[1] == weather.FORECAST_URL
    assert weather.ARCHIVE_URL not in called_args


def test_fetch_weather_forecast_http_error():
    resp = _mock_response(400)
    with patch("shared.requests.request", return_value=resp):
        with pytest.raises(ApiError, match="Open-Meteo"):
            weather.fetch_weather_forecast(39.9, 32.8, date(2026, 7, 22), date(2026, 7, 24))
