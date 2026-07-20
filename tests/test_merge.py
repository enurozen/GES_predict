import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from merge import build_training_dataset


def _generation_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": ["2026-07-01", "2026-07-01", "2026-07-01"],
            "Hour": ["00:00", "01:00", "02:00"],
            "Generation (MWh)": [0.0, 1.5, 3.2],
        }
    )


def _weather_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-07-01 00:00", "2026-07-01 01:00", "2026-07-01 03:00"]
            ),
            "ghi_forecast": [0.0, 50.0, 200.0],
            "temp_c": [18.0, 18.5, 20.0],
            "cloud_cover": [0.9, 0.7, 0.1],
        }
    )


def test_build_training_dataset_inner_joins_on_timestamp():
    df = build_training_dataset(_generation_df(), _weather_df())

    assert list(df.columns) == [
        "timestamp",
        "ghi_forecast",
        "temp_c",
        "cloud_cover",
        "production_mwh",
    ]
    # 02:00 has generation but no weather -> dropped. 03:00 has weather but
    # no generation -> dropped too, since it's an inner join.
    assert len(df) == 2
    assert list(df["production_mwh"]) == [0.0, 1.5]


def test_build_training_dataset_logs_warning_for_missing_weather_hour(caplog):
    with caplog.at_level("WARNING"):
        build_training_dataset(_generation_df(), _weather_df())

    assert any("no matching weather data" in message for message in caplog.messages)


def test_build_training_dataset_drops_and_logs_unparsable_hour(caplog):
    gen = _generation_df()
    gen.loc[0, "Hour"] = "not-a-time"

    with caplog.at_level("WARNING"):
        df = build_training_dataset(gen, _weather_df())

    assert any("unparsable Date/Hour" in message for message in caplog.messages)
    # Only 01:00 survives: 00:00 was unparsable, 02:00 has no weather match.
    assert len(df) == 1
    assert df.iloc[0]["production_mwh"] == 1.5


def test_build_training_dataset_no_overlap_returns_empty():
    weather = _weather_df()
    weather["timestamp"] = pd.to_datetime(
        ["2026-08-01 00:00", "2026-08-01 01:00", "2026-08-01 03:00"]
    )

    df = build_training_dataset(_generation_df(), weather)

    assert df.empty
    assert list(df.columns) == [
        "timestamp",
        "ghi_forecast",
        "temp_c",
        "cloud_cover",
        "production_mwh",
    ]
