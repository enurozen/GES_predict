import sys
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import api

client = TestClient(api.app)


# --------------------------------------------------------------------------
# health / plants (no network, no training - fast)
# --------------------------------------------------------------------------

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_get_plants_includes_registered_plant():
    resp = client.get("/plants")
    assert resp.status_code == 200
    plant_ids = [p["plant_id"] for p in resp.json()]
    assert 2579 in plant_ids


def test_get_plant_success():
    resp = client.get("/plants/2579")
    assert resp.status_code == 200
    body = resp.json()
    assert body["plant_id"] == 2579
    assert body["name"]
    assert body["capacity_mw"] > 0


def test_get_plant_not_found():
    resp = client.get("/plants/999999")
    assert resp.status_code == 404


# --------------------------------------------------------------------------
# train / evaluate: real integration against the repo's actual data/2579
# CSVs (no network - these endpoints never touch EPİAŞ/Open-Meteo).
# --------------------------------------------------------------------------

def test_train_plant_writes_model_and_returns_calibration(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parent.parent)  # need real data/2579
    monkeypatch.setattr(api, "MODELS_DIR", tmp_path / "models")

    resp = client.post("/plants/2579/train")

    assert resp.status_code == 200
    body = resp.json()
    assert body["plant_id"] == 2579
    assert body["rows"] > 0
    assert 0.0 < body["efficiency_scale"] < 2.0
    model_path = Path(body["model_path"])
    assert model_path.exists()
    bundle = joblib.load(model_path)
    assert hasattr(bundle["model"], "predict")


def test_train_plant_not_found(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).resolve().parent.parent)
    monkeypatch.setattr(api, "MODELS_DIR", tmp_path / "models")

    resp = client.post("/plants/999999/train")

    assert resp.status_code == 404


def test_evaluate_plant_returns_metrics():
    resp = client.get("/plants/2579/evaluate", params={"test_days": 7})

    assert resp.status_code == 200
    body = resp.json()
    assert body["plant_id"] == 2579
    assert body["test_days"] == 7
    assert body["hybrid_mae_mwh"] >= 0
    assert 0 <= body["accuracy_pct"] <= 100
    # The hybrid model should generally beat (or at least not be wildly
    # worse than) the physical-only baseline on this plant's real data.
    assert body["hybrid_mae_mwh"] <= body["physical_mae_mwh"]


def test_evaluate_plant_not_found():
    resp = client.get("/plants/999999/evaluate")
    assert resp.status_code == 404


# --------------------------------------------------------------------------
# predict: mocks the weather fetch (network), needs a trained model on disk
# --------------------------------------------------------------------------

def _fake_bundle():
    model = RandomForestRegressor(n_estimators=2, random_state=0)
    X = pd.DataFrame({
        "hour": [10], "day_of_year": [200], "month": [7], "temp_c": [20.0],
        "ghi_forecast": [500.0], "cloud_cover": [0.2], "hour_sin": [0.0], "hour_cos": [0.0],
        "doy_sin": [0.0], "doy_cos": [0.0], "is_afternoon": [0], "ghi_x_afternoon": [0.0],
        "solar_elevation_deg": [45.0], "clear_sky_index": [0.8], "ghi_ramp_1h": [0.0],
    })
    model.fit(X, [1.0])
    return {
        "model": model,
        "calibration": {"efficiency_scale": 0.3, "temp_coeff": -0.008, "ac_capacity_mw": 200.0},
        "plant": {"name": "Test GES", "lat": 37.79, "lon": 33.58, "capacity_mw": 1000.0,
                   "tilt_deg": None, "azimuth_deg": None, "tracker_type": None},
    }


def _fake_weather_df():
    return pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-08-10 10:00", "2026-08-10 11:00"]),
        "ghi_forecast": [400.0, 450.0],
        "dni_forecast": [500.0, 520.0],
        "dhi_forecast": [60.0, 70.0],
        "temp_c": [25.0, 26.0],
        "cloud_cover": [0.1, 0.2],
    })


def test_predict_plant_success(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODELS_DIR", tmp_path / "models")
    model_dir = tmp_path / "models" / "2579"
    model_dir.mkdir(parents=True)
    joblib.dump(_fake_bundle(), model_dir / "model.joblib")

    with patch("api.fetch_weather_forecast", return_value=_fake_weather_df()):
        resp = client.post("/plants/2579/predict", params={"start": "2026-08-10", "end": "2026-08-10"})

    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    assert "predicted_mwh" in body[0]


def test_predict_plant_no_model(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODELS_DIR", tmp_path / "models")

    resp = client.post("/plants/2579/predict", params={"start": "2026-08-10", "end": "2026-08-10"})

    assert resp.status_code == 404


def test_predict_plant_start_after_end(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODELS_DIR", tmp_path / "models")

    resp = client.post("/plants/2579/predict", params={"start": "2026-08-12", "end": "2026-08-10"})

    assert resp.status_code == 400


def test_predict_plant_empty_forecast(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "MODELS_DIR", tmp_path / "models")
    model_dir = tmp_path / "models" / "2579"
    model_dir.mkdir(parents=True)
    joblib.dump(_fake_bundle(), model_dir / "model.joblib")

    empty_df = pd.DataFrame(columns=["timestamp", "ghi_forecast", "dni_forecast", "dhi_forecast", "temp_c", "cloud_cover"])
    with patch("api.fetch_weather_forecast", return_value=empty_df):
        resp = client.post("/plants/2579/predict", params={"start": "2026-08-10", "end": "2026-08-10"})

    assert resp.status_code == 422
