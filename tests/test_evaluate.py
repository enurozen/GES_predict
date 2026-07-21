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

    rc = evaluate.main(["--plant-id", "2579"])

    assert rc == 0


def test_main_unregistered_plant_fails_before_loading_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    rc = evaluate.main(["--plant-id", "9999"])

    assert rc == 1


def test_main_missing_data_fails_cleanly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    rc = evaluate.main(["--plant-id", "2579"])

    assert rc == 1
