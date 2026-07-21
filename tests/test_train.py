import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train


def _write_day(data_dir: Path, day: str, hours: int = 24):
    rows = []
    for h in range(hours):
        rows.append({
            "timestamp": f"{day} {h:02d}:00:00",
            "ghi_forecast": 500.0 if 8 <= h <= 16 else 0.0,
            "temp_c": 20.0,
            "cloud_cover": 0.2,
            "production_mwh": 5.0 if 8 <= h <= 16 else 0.0,
        })
    pd.DataFrame(rows).to_csv(data_dir / f"{day}.csv", index=False)


def test_load_training_data_concatenates_all_days(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data" / "2579"
    data_dir.mkdir(parents=True)
    _write_day(data_dir, "2026-01-01")
    _write_day(data_dir, "2026-01-02")

    df = train.load_training_data(2579)

    assert len(df) == 48
    assert df["timestamp"].is_monotonic_increasing


def test_load_training_data_missing_dir_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(FileNotFoundError):
        train.load_training_data(2579)


def test_main_trains_and_saves_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data" / "2579"
    data_dir.mkdir(parents=True)
    for day in ["2026-01-01", "2026-01-02", "2026-01-03"]:
        _write_day(data_dir, day)

    rc = train.main(["--plant-id", "2579", "--output", "models/2579/model.joblib"])

    assert rc == 0
    model_path = tmp_path / "models" / "2579" / "model.joblib"
    assert model_path.exists()
    bundle = joblib.load(model_path)
    assert hasattr(bundle["model"], "predict")
    assert "efficiency_scale" in bundle["calibration"]
    assert bundle["plant"]["name"]


def test_main_unregistered_plant_fails_before_loading_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    rc = train.main(["--plant-id", "9999", "--output", "models/9999/model.joblib"])

    assert rc == 1
    assert not (tmp_path / "models").exists()
