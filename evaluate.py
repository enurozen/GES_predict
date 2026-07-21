"""
Evaluate the hybrid model's real-world accuracy on held-out data: splits
data/<plant_id>/*.csv chronologically (80/20), trains only on the first
80%, then compares the physical-only baseline vs. the hybrid (physical+ML)
model on the unseen last 20%.

Usage:
    python evaluate.py --plant-id 2579
"""

import argparse
import logging
import sys

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ges_uretim_tahmini import build_physical_baseline, predict_production, train_residual_model
from plants import PlantNotFoundError, load_plant
from train import load_training_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the GES hybrid model on held-out real data.")
    parser.add_argument("--plant-id", type=int, required=True)
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Fraction of days held out for testing")
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

    split_idx = int(len(df) * (1 - args.test_fraction))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    logger.info(
        "Train: %d rows (%s to %s) | Test: %d rows (%s to %s)",
        len(train_df), train_df["timestamp"].min(), train_df["timestamp"].max(),
        len(test_df), test_df["timestamp"].min(), test_df["timestamp"].max(),
    )

    lat, lon, capacity_mw = plant["lat"], plant["lon"], plant["capacity_mw"]

    baseline_train = build_physical_baseline(train_df, lat, lon, capacity_mw)
    model = train_residual_model(train_df, baseline_train)

    baseline_test = build_physical_baseline(test_df, lat, lon, capacity_mw)
    hybrid_test = predict_production(test_df, lat, lon, capacity_mw, model)

    actual = test_df["production_mwh"]
    physical_mae = mean_absolute_error(actual, baseline_test)
    physical_rmse = np.sqrt(mean_squared_error(actual, baseline_test))
    hybrid_mae = mean_absolute_error(actual, hybrid_test)
    hybrid_rmse = np.sqrt(mean_squared_error(actual, hybrid_test))
    improvement = (1 - hybrid_mae / physical_mae) * 100 if physical_mae else 0.0

    logger.info("=== Test seti sonuçları (%s) ===", plant["name"])
    logger.info("Sadece fiziksel model : MAE=%.3f MWh, RMSE=%.3f MWh", physical_mae, physical_rmse)
    logger.info("Hibrit (fiziksel+ML)  : MAE=%.3f MWh, RMSE=%.3f MWh", hybrid_mae, hybrid_rmse)
    logger.info("İyileşme: %%%.1f", improvement)
    return 0


if __name__ == "__main__":
    sys.exit(main())
