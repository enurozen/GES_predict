"""
Train the hybrid (physical + ML residual) production model on real
collected data: reads every data/<plant_id>/*.csv, looks up capacity_mw/
lat/lon from plants.yaml, calibrates site-specific physical parameters
(efficiency, temp coefficient, real AC/inverter ceiling) from the same
data, trains the residual model via ges_uretim_tahmini, and saves the
fitted model bundled with its calibration.

Usage:
    python train.py --plant-id 2579 --output models/2579/model.joblib
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

from ges_uretim_tahmini import build_physical_baseline, calibrate_site_parameters, train_residual_model
from plants import PlantNotFoundError, load_plant

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_training_data(plant_id: int) -> pd.DataFrame:
    """Concatenate every daily CSV under data/<plant_id>/ into one DataFrame."""
    data_dir = Path("data") / str(plant_id)
    csv_paths = sorted(data_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under {data_dir}")

    df = pd.concat((pd.read_csv(p, parse_dates=["timestamp"]) for p in csv_paths), ignore_index=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


def split_train_test(
    df: pd.DataFrame, test_days: int = 30, test_fraction: float | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologically split into train/test by whole days (never splits a day in half).

    Default holds out the most recent `test_days` days - as more data
    accumulates, a fixed recent window stays representative of "now" and
    lets training use everything older, instead of a %-based split that
    would hold out a growing chunk of the most recent (most relevant) data.
    Pass test_fraction to use a %-based split instead (legacy behavior).
    """
    dates = sorted(df["timestamp"].dt.date.unique())
    if test_fraction is not None:
        split_i = int(len(dates) * (1 - test_fraction))
    else:
        split_i = max(0, len(dates) - test_days)

    test_dates = set(dates[split_i:])
    date_col = df["timestamp"].dt.date
    train_df = df[~date_col.isin(test_dates)].reset_index(drop=True)
    test_df = df[date_col.isin(test_dates)].reset_index(drop=True)
    return train_df, test_df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the GES hybrid production model on collected data.")
    parser.add_argument(
        "--plant-id", type=int, required=True,
        help="Plant ID; reads data/<plant_id>/*.csv and looks it up in plants.yaml",
    )
    parser.add_argument("--output", required=True, help="Output path for the trained model (joblib)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        plant = load_plant(args.plant_id)
    except PlantNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    try:
        df = load_training_data(args.plant_id)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Loaded %d rows for %s (%s to %s)", len(df), plant["name"], df["timestamp"].min(), df["timestamp"].max())

    calibration = calibrate_site_parameters(df, plant["lat"], plant["lon"], plant["capacity_mw"])
    logger.info(
        "Kalibrasyon: efficiency_scale=%.4f, temp_coeff=%.5f, ac_capacity_mw=%.1f (%d gündüz örneği)",
        calibration["efficiency_scale"], calibration["temp_coeff"],
        calibration["ac_capacity_mw"], calibration["n_daylight_samples"],
    )

    baseline = build_physical_baseline(
        df, plant["lat"], plant["lon"], plant["capacity_mw"],
        temp_coeff=calibration["temp_coeff"], efficiency_scale=calibration["efficiency_scale"],
    )
    model = train_residual_model(df, baseline)

    bundle = {"model": model, "calibration": calibration, "plant": plant}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    logger.info("Wrote trained model to %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
