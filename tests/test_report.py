import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import report


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


def test_main_writes_html_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data" / "2579"
    data_dir.mkdir(parents=True)
    for d in range(1, 11):
        _write_day(data_dir, f"2026-01-{d:02d}")

    rc = report.main(["--plant-id", "2579", "--output", "reports/out.html", "--test-days", "2"])

    assert rc == 0
    out_path = tmp_path / "reports" / "out.html"
    assert out_path.exists()
    html = out_path.read_text(encoding="utf-8")
    assert "__DATA__" not in html
    assert "__PLANT_NAME__" not in html
    assert "Kalyon Karapınar YEKA-1 GES" in html
    assert "\"actual\"" in html


def test_main_unregistered_plant_fails_before_loading_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    rc = report.main(["--plant-id", "9999", "--output", "reports/out.html"])

    assert rc == 1
    assert not (tmp_path / "reports").exists()
