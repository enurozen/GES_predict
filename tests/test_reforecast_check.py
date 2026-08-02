import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import reforecast_check


def _preds(values):
    return pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-07-22 10:00", "2026-07-22 11:00", "2026-07-22 12:00"]),
        "predicted_mwh": values,
    })


def test_compare_forecasts_flags_hours_over_tolerance():
    day_ahead = _preds([100.0, 200.0, 300.0])
    refresh = _preds([100.0, 250.0, 280.0])  # hour 2: +50 MWh, hour 3: -20 MWh

    result = reforecast_check.compare_forecasts(day_ahead, refresh, capacity_mw=1000.0, tolerance_pct=3.0)

    assert list(result["deviation_mwh"]) == pytest.approx([0.0, 50.0, -20.0])
    assert list(result["needs_review"]) == [False, True, False]  # 5% > 3%, 2% < 3%


def test_compare_forecasts_no_deviation_flags_nothing():
    day_ahead = _preds([100.0, 200.0, 300.0])
    result = reforecast_check.compare_forecasts(day_ahead, day_ahead.copy(), capacity_mw=1000.0, tolerance_pct=3.0)

    assert not result["needs_review"].any()


def test_main_writes_comparison_csv(tmp_path):
    day_ahead_path = tmp_path / "day_ahead.csv"
    refresh_path = tmp_path / "refresh.csv"
    _preds([100.0, 200.0, 300.0]).to_csv(day_ahead_path, index=False)
    _preds([100.0, 250.0, 300.0]).to_csv(refresh_path, index=False)

    output_path = tmp_path / "comparison.csv"
    rc = reforecast_check.main([
        "--day-ahead", str(day_ahead_path), "--refresh", str(refresh_path),
        "--capacity-mw", "1000", "--output", str(output_path),
    ])

    assert rc == 0
    assert output_path.exists()
    out = pd.read_csv(output_path)
    assert len(out) == 3


def test_main_missing_file_fails_cleanly(tmp_path):
    rc = reforecast_check.main([
        "--day-ahead", str(tmp_path / "missing.csv"), "--refresh", str(tmp_path / "also_missing.csv"),
        "--capacity-mw", "1000",
    ])

    assert rc == 1
