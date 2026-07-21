import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import evaluate


def _write_day(data_dir: Path, day: str):
    rows = []
    for h in range(24):
        ghi = 500.0 if 8 <= h <= 16 else 0.0
        rows.append({
            "timestamp": f"{day} {h:02d}:00:00",
            "ghi_forecast": ghi,
            "temp_c": 20.0,
            "cloud_cover": 0.2,
            "production_mwh": ghi / 100.0,
        })
    pd.DataFrame(rows).to_csv(data_dir / f"{day}.csv", index=False)


def test_main_evaluates_and_returns_zero(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data" / "2579"
    data_dir.mkdir(parents=True)
    days = [f"2026-01-{d:02d}" for d in range(1, 11)]
    for day in days:
        _write_day(data_dir, day)

    rc = evaluate.main(["--plant-id", "2579", "--test-days", "2"])

    assert rc == 0


def test_main_unregistered_plant_fails_before_loading_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    rc = evaluate.main(["--plant-id", "9999"])

    assert rc == 1


def test_main_missing_data_fails_cleanly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    rc = evaluate.main(["--plant-id", "2579"])

    assert rc == 1


def test_normalized_mae_scales_by_capacity():
    actual = np.array([100.0, 200.0])
    predicted = np.array([110.0, 180.0])  # abs errors: 10, 20 -> MAE = 15

    assert evaluate.normalized_mae(actual, predicted, capacity_mw=1000.0) == 1.5


def test_hit_rate_counts_hours_within_tolerance():
    actual = np.array([100.0, 100.0, 100.0, 100.0])
    predicted = np.array([102.0, 200.0, 99.0, 50.0])  # abs errors: 2, 100, 1, 50
    # capacity 1000, tolerance 3% -> 30 MW band: only errors 2 and 1 qualify -> 2/4 = 50%

    assert evaluate.hit_rate(actual, predicted, capacity_mw=1000.0, tolerance_pct=3.0) == 50.0


def test_hit_rate_zero_tolerance_requires_exact_match():
    actual = np.array([10.0, 10.0])
    predicted = np.array([10.0, 11.0])

    assert evaluate.hit_rate(actual, predicted, capacity_mw=1000.0, tolerance_pct=0.0) == 50.0
