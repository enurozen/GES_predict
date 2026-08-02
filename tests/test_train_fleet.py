import sys
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train_fleet
from plants import PlantNotFoundError


def _write_day(data_dir: Path, day: str, hours: int = 24):
    rows = []
    for h in range(hours):
        rows.append({
            "timestamp": f"{day} {h:02d}:00:00",
            "ghi_forecast": 500.0 if 8 <= h <= 16 else 0.0,
            "dni_forecast": 550.0 if 8 <= h <= 16 else 0.0,
            "dhi_forecast": 60.0 if 8 <= h <= 16 else 0.0,
            "temp_c": 20.0,
            "cloud_cover": 0.2,
            "production_mwh": 5.0 if 8 <= h <= 16 else 0.0,
        })
    pd.DataFrame(rows).to_csv(data_dir / f"{day}.csv", index=False)


def _fake_plant(plant_id, capacity_mw):
    return {
        "name": f"Test GES {plant_id}",
        "lat": 37.7908, "lon": 33.5847,
        "capacity_mw": capacity_mw,
        "tilt_deg": None, "azimuth_deg": None, "tracker_type": None,
    }


def _fake_load_plant(plant_id):
    capacities = {9001: 10.0, 9002: 50.0}
    if plant_id not in capacities:
        raise PlantNotFoundError(f"Plant ID {plant_id} not found")
    return _fake_plant(plant_id, capacities[plant_id])


def test_build_pooled_dataset_single_plant(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data" / "9001"
    data_dir.mkdir(parents=True)
    for day in ["2026-01-01", "2026-01-02", "2026-01-03"]:
        _write_day(data_dir, day)

    with patch("train_fleet.load_plant", side_effect=_fake_load_plant):
        X, y, calibrations, plants = train_fleet.build_pooled_dataset([9001])

    assert len(X) == 72
    assert len(y) == 72
    assert 9001 in calibrations
    assert 9001 in plants
    # Residual is normalized by capacity (fraction of MWp), not raw MWh.
    assert y.abs().max() < 10.0


def test_build_pooled_dataset_pools_multiple_plants(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for plant_id in [9001, 9002]:
        data_dir = tmp_path / "data" / str(plant_id)
        data_dir.mkdir(parents=True)
        for day in ["2026-01-01", "2026-01-02"]:
            _write_day(data_dir, day)

    with patch("train_fleet.load_plant", side_effect=_fake_load_plant):
        X, y, calibrations, plants = train_fleet.build_pooled_dataset([9001, 9002])

    assert len(X) == 96  # 2 plants x 2 days x 24h
    assert set(calibrations.keys()) == {9001, 9002}
    assert set(plants.keys()) == {9001, 9002}


def test_main_trains_and_saves_fleet_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for plant_id in [9001, 9002]:
        data_dir = tmp_path / "data" / str(plant_id)
        data_dir.mkdir(parents=True)
        for day in ["2026-01-01", "2026-01-02", "2026-01-03"]:
            _write_day(data_dir, day)

    with patch("train_fleet.load_plant", side_effect=_fake_load_plant):
        rc = train_fleet.main(["--plant-ids", "9001,9002", "--output", "models/fleet/model.joblib"])

    assert rc == 0
    bundle = joblib.load(tmp_path / "models" / "fleet" / "model.joblib")
    assert hasattr(bundle["model"], "predict")
    assert bundle["plant_ids"] == [9001, 9002]
    assert bundle["residual_normalized_by_capacity"] is True
    assert set(bundle["calibrations"].keys()) == {9001, 9002}


def test_main_unknown_plant_fails_cleanly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with patch("train_fleet.load_plant", side_effect=_fake_load_plant):
        rc = train_fleet.main(["--plant-ids", "9999", "--output", "models/fleet/model.joblib"])

    assert rc == 1
    assert not (tmp_path / "models").exists()


def test_main_invalid_plant_ids_fails_cleanly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    rc = train_fleet.main(["--plant-ids", "not-a-number", "--output", "models/fleet/model.joblib"])

    assert rc == 1
