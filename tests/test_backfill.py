import os
import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backfill


def _generation_df(days: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": days,
            "Hour": ["00:00"] * len(days),
            "Generation (MWh)": [1.0] * len(days),
        }
    )


def _weather_df(days: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime([f"{d} 00:00" for d in days]),
            "ghi_forecast": [0.0] * len(days),
            "temp_c": [18.0] * len(days),
            "cloud_cover": [0.5] * len(days),
        }
    )


def _set_creds(monkeypatch):
    monkeypatch.setenv("EPIAS_USERNAME", "user")
    monkeypatch.setenv("EPIAS_PASSWORD", "pass")


def test_backfill_writes_one_csv_per_day(tmp_path, monkeypatch):
    _set_creds(monkeypatch)
    monkeypatch.chdir(tmp_path)

    days = ["2026-06-01", "2026-06-02", "2026-06-03"]
    with patch("backfill.get_tgt", return_value="TGT"), \
         patch("backfill.fetch_generation_range", return_value=_generation_df(days)), \
         patch("backfill.fetch_weather_range", return_value=_weather_df(days)):
        rc = backfill.main([
            "--plant-id", "2579",
            "--start", "2026-06-01", "--end", "2026-06-03",
        ])

    assert rc == 0
    out_dir = tmp_path / "data" / "2579"
    written = sorted(p.name for p in out_dir.glob("*.csv"))
    assert written == ["2026-06-01.csv", "2026-06-02.csv", "2026-06-03.csv"]


def test_backfill_skips_days_that_already_have_a_csv(tmp_path, monkeypatch):
    _set_creds(monkeypatch)
    monkeypatch.chdir(tmp_path)

    out_dir = tmp_path / "data" / "2579"
    out_dir.mkdir(parents=True)
    (out_dir / "2026-06-01.csv").write_text("already,here\n1,2\n")

    days = ["2026-06-02", "2026-06-03"]
    with patch("backfill.get_tgt", return_value="TGT") as m_tgt, \
         patch("backfill.fetch_generation_range", return_value=_generation_df(days)) as m_gen, \
         patch("backfill.fetch_weather_range", return_value=_weather_df(days)):
        rc = backfill.main([
            "--plant-id", "2579",
            "--start", "2026-06-01", "--end", "2026-06-03",
        ])

    assert rc == 0
    # Only the missing days should have been fetched (start/end narrowed to 06-02..06-03).
    fetched_start, fetched_end = m_gen.call_args[0][2], m_gen.call_args[0][3]
    assert (fetched_start, fetched_end) == (date(2026, 6, 2), date(2026, 6, 3))
    assert m_tgt.call_count == 1
    # The pre-existing file must be untouched.
    assert (out_dir / "2026-06-01.csv").read_text() == "already,here\n1,2\n"


def test_backfill_all_days_already_present_skips_network_entirely(tmp_path, monkeypatch):
    _set_creds(monkeypatch)
    monkeypatch.chdir(tmp_path)

    out_dir = tmp_path / "data" / "2579"
    out_dir.mkdir(parents=True)
    (out_dir / "2026-06-01.csv").write_text("x\n1\n")

    with patch("backfill.get_tgt") as m_tgt:
        rc = backfill.main([
            "--plant-id", "2579",
            "--start", "2026-06-01", "--end", "2026-06-01",
        ])

    assert rc == 0
    m_tgt.assert_not_called()


def test_backfill_overwrite_refetches_existing_days(tmp_path, monkeypatch):
    _set_creds(monkeypatch)
    monkeypatch.chdir(tmp_path)

    out_dir = tmp_path / "data" / "2579"
    out_dir.mkdir(parents=True)
    (out_dir / "2026-06-01.csv").write_text("stale,data\n0,0\n")

    days = ["2026-06-01"]
    with patch("backfill.get_tgt", return_value="TGT"), \
         patch("backfill.fetch_generation_range", return_value=_generation_df(days)), \
         patch("backfill.fetch_weather_range", return_value=_weather_df(days)):
        rc = backfill.main([
            "--plant-id", "2579",
            "--start", "2026-06-01", "--end", "2026-06-01",
            "--overwrite",
        ])

    assert rc == 0
    content = (out_dir / "2026-06-01.csv").read_text()
    assert "stale" not in content
    assert "production_mwh" in content


def test_backfill_unregistered_plant_fails_before_any_network_call(tmp_path, monkeypatch):
    _set_creds(monkeypatch)
    monkeypatch.chdir(tmp_path)

    with patch("backfill.get_tgt") as m_tgt:
        rc = backfill.main([
            "--plant-id", "9999",
            "--start", "2026-06-01", "--end", "2026-06-01",
        ])

    assert rc == 1
    m_tgt.assert_not_called()


def test_backfill_rejects_end_date_of_today_or_later(tmp_path, monkeypatch):
    _set_creds(monkeypatch)
    monkeypatch.chdir(tmp_path)

    today = date.today().isoformat()
    with patch("backfill.get_tgt") as m_tgt:
        rc = backfill.main([
            "--plant-id", "2579",
            "--start", "2026-06-01", "--end", today,
        ])

    assert rc == 1
    m_tgt.assert_not_called()
