import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import predict


def _fake_bundle():
    # A trivially-fit model is enough - predict.py just needs the bundle shape.
    model = RandomForestRegressor(n_estimators=2, random_state=0)
    X = pd.DataFrame({
        "hour": [10], "day_of_year": [200], "month": [7], "temp_c": [20.0],
        "ghi_forecast": [500.0], "cloud_cover": [0.2], "hour_sin": [0.0], "hour_cos": [0.0],
        "doy_sin": [0.0], "doy_cos": [0.0], "is_afternoon": [0], "ghi_x_afternoon": [0.0],
    })
    model.fit(X, [1.0])
    return {
        "model": model,
        "calibration": {"efficiency_scale": 0.3, "temp_coeff": -0.008, "ac_capacity_mw": 200.0},
        "plant": {"name": "Test GES", "lat": 37.79, "lon": 33.58, "capacity_mw": 1000.0},
    }


def _weather_df():
    return pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-07-22 10:00", "2026-07-22 11:00"]),
        "ghi_forecast": [400.0, 450.0],
        "temp_c": [25.0, 26.0],
        "cloud_cover": [0.1, 0.2],
    })


def test_main_writes_predictions_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "models" / "2579"
    model_dir.mkdir(parents=True)
    joblib.dump(_fake_bundle(), model_dir / "model.joblib")

    with patch("predict.fetch_weather_forecast", return_value=_weather_df()) as m_wx:
        rc = predict.main([
            "--plant-id", "2579", "--start", "2026-07-22", "--end", "2026-07-22",
            "--output", "predictions/out.csv",
        ])

    assert rc == 0
    m_wx.assert_called_once_with(37.79, 33.58, date(2026, 7, 22), date(2026, 7, 22))
    out = pd.read_csv(tmp_path / "predictions" / "out.csv")
    assert list(out.columns) == ["timestamp", "predicted_mwh"]
    assert len(out) == 2


def test_main_missing_model_fails_cleanly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    rc = predict.main([
        "--plant-id", "2579", "--start", "2026-07-22", "--end", "2026-07-22",
        "--output", "predictions/out.csv",
    ])

    assert rc == 1
    assert not (tmp_path / "predictions").exists()


def test_main_start_after_end_fails(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    rc = predict.main([
        "--plant-id", "2579", "--start", "2026-07-24", "--end", "2026-07-22",
        "--output", "predictions/out.csv",
    ])

    assert rc == 1


def test_main_empty_forecast_fails_cleanly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model_dir = tmp_path / "models" / "2579"
    model_dir.mkdir(parents=True)
    joblib.dump(_fake_bundle(), model_dir / "model.joblib")

    with patch("predict.fetch_weather_forecast", return_value=pd.DataFrame(columns=["timestamp", "ghi_forecast", "temp_c", "cloud_cover"])):
        rc = predict.main([
            "--plant-id", "2579", "--start", "2026-07-22", "--end", "2026-07-22",
            "--output", "predictions/out.csv",
        ])

    assert rc == 1
