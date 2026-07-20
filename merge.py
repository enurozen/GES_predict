"""
Merge EPİAŞ generation data with Open-Meteo weather data into a single
hourly training dataset for the GES production model.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = ["timestamp", "ghi_forecast", "temp_c", "cloud_cover", "production_mwh"]


def build_training_dataset(generation_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join hourly generation and weather data on timestamp.

    generation_df is expected to have "Date", "Hour", and "Generation (MWh)"
    columns (the shape returned by epias.fetch_generation_range). weather_df
    is expected to have "timestamp", "ghi_forecast", "temp_c", and
    "cloud_cover" columns (the shape returned by weather.fetch_weather_range).

    Rows with an unparsable Date/Hour, or an hour present on one side but
    missing on the other, are dropped and logged as a warning.
    """
    gen = generation_df.copy()
    gen["timestamp"] = pd.to_datetime(
        gen["Date"].astype(str) + " " + gen["Hour"].astype(str),
        format="%Y-%m-%d %H:%M",
        errors="coerce",
    )
    gen = gen.rename(columns={"Generation (MWh)": "production_mwh"})

    bad_rows = gen["timestamp"].isna()
    if bad_rows.any():
        logger.warning(
            "%d generation row(s) have an unparsable Date/Hour and were dropped.",
            int(bad_rows.sum()),
        )
        gen = gen[~bad_rows]

    merged = gen[["timestamp", "production_mwh"]].merge(
        weather_df[["timestamp", "ghi_forecast", "temp_c", "cloud_cover"]],
        on="timestamp",
        how="inner",
    )

    missing_hours = set(gen["timestamp"]) - set(merged["timestamp"])
    if missing_hours:
        sample = ", ".join(str(ts) for ts in sorted(missing_hours)[:5])
        logger.warning(
            "%d hour(s) had no matching weather data and were dropped (e.g. %s).",
            len(missing_hours),
            sample,
        )

    merged = merged.sort_values("timestamp").reset_index(drop=True)
    return merged[OUTPUT_COLUMNS]
