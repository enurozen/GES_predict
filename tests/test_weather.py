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
            "direct_normal_irradiance": [0.0, 300.2],
            "diffuse_radiation": [0.0, 45.1],
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

    assert list(df.columns) == [
        "timestamp", "ghi_forecast", "dni_forecast", "dhi_forecast", "temp_c", "cloud_cover",
    ]
    assert list(df["ghi_forecast"]) == [0.0, 120.5]
    assert list(df["dni_forecast"]) == [0.0, 300.2]
    assert list(df["dhi_forecast"]) == [0.0, 45.1]
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


# --------------------------------------------------------------------------
# fetch_weather_forecast_ensemble
# --------------------------------------------------------------------------

def _ensemble_body(models):
    # Open-Meteo suffixes each hourly variable with the model id when
    # multiple models are requested (e.g. shortwave_radiation_gfs_seamless).
    hourly = {"time": ["2026-07-01T00:00", "2026-07-01T01:00"]}
    ghi_by_model = {models[0]: [100.0, 200.0], models[1]: [120.0, 180.0], models[2]: [80.0, 220.0]}
    temp_by_model = {models[0]: [20.0, 21.0], models[1]: [22.0, 23.0], models[2]: [18.0, 19.0]}
    for m in models:
        hourly[f"shortwave_radiation_{m}"] = ghi_by_model[m]
        hourly[f"direct_normal_irradiance_{m}"] = [v * 1.5 for v in ghi_by_model[m]]
        hourly[f"diffuse_radiation_{m}"] = [v * 0.2 for v in ghi_by_model[m]]
        hourly[f"temperature_2m_{m}"] = temp_by_model[m]
        hourly[f"cloudcover_{m}"] = [50.0, 60.0]
    return {"hourly": hourly}


def test_fetch_weather_forecast_ensemble_averages_across_models():
    models = weather.DEFAULT_ENSEMBLE_MODELS
    resp = _mock_response(200, json_data=_ensemble_body(models))

    with patch("shared.requests.request", return_value=resp) as mock_request:
        df = weather.fetch_weather_forecast_ensemble(39.9, 32.8, date(2026, 7, 22), date(2026, 7, 22))

    assert list(df["ghi_forecast"]) == pytest.approx([100.0, 200.0])  # mean of [100,120,80] and [200,180,220]
    assert df["ghi_forecast_std"].iloc[0] > 0  # models disagree -> nonzero spread
    assert df["cloud_cover"].iloc[0] == pytest.approx(0.5)  # 50% -> 0.5 fraction

    called_kwargs = mock_request.call_args.kwargs
    assert called_kwargs["params"]["models"] == ",".join(models)


def test_fetch_weather_forecast_ensemble_falls_back_to_single_model_key():
    # Simulate an unrecognized/changed model id: no suffixed keys present,
    # only the plain (single-model-style) key.
    resp = _mock_response(200, json_data={
        "hourly": {
            "time": ["2026-07-01T00:00"],
            "shortwave_radiation": [150.0],
            "direct_normal_irradiance": [300.0],
            "diffuse_radiation": [30.0],
            "temperature_2m": [21.0],
            "cloudcover": [40.0],
        }
    })

    with patch("shared.requests.request", return_value=resp):
        df = weather.fetch_weather_forecast_ensemble(39.9, 32.8, date(2026, 7, 22), date(2026, 7, 22),
                                                       models=["some_unknown_model"])

    assert df["ghi_forecast"].iloc[0] == pytest.approx(150.0)
    assert df["ghi_forecast_std"].iloc[0] == pytest.approx(0.0)


def test_fetch_weather_forecast_ensemble_http_error():
    resp = _mock_response(400)
    with patch("shared.requests.request", return_value=resp):
        with pytest.raises(ApiError, match="Open-Meteo"):
            weather.fetch_weather_forecast_ensemble(39.9, 32.8, date(2026, 7, 22), date(2026, 7, 22))
