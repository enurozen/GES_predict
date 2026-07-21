"""
Evaluate the hybrid model's real-world accuracy on held-out data: holds
out the most recent N days of data/<plant_id>/*.csv, trains only on
everything older, then compares the physical-only baseline vs. the
hybrid (physical+ML) model on the unseen recent days.

Usage:
    python evaluate.py --plant-id 2579
    python evaluate.py --plant-id 2579 --test-days 14
"""

import argparse
import logging
import sys

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ges_uretim_tahmini import build_physical_baseline, calibrate_site_parameters, predict_production, train_residual_model
from plants import PlantNotFoundError, load_plant
from train import load_training_data, split_train_test

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def normalized_mae(actual: np.ndarray, predicted: np.ndarray, capacity_mw: float) -> float:
    """MAE as a % of installed capacity (nMAE) - the standard solar-forecasting accuracy metric.

    Plain MAPE blows up near sunrise/sunset, where actual production is close
    to zero and any small absolute error becomes a huge percentage error.
    Normalizing by capacity instead avoids that instability.
    """
    mae = mean_absolute_error(actual, predicted)
    return float(mae / capacity_mw * 100)


def hit_rate(actual: np.ndarray, predicted: np.ndarray, capacity_mw: float, tolerance_pct: float) -> float:
    """% of hours where |predicted - actual| stays within tolerance_pct of capacity.

    This is closer to what matters for imbalance-penalty exposure than an
    average error: a few large misses can hide inside a good mean MAE, but
    each one is what actually gets penalized. Tolerance is capacity-relative
    (not %-of-actual) for the same reason nMAE is: actual is often near zero.
    """
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    tolerance_mw = capacity_mw * (tolerance_pct / 100)
    return float(np.mean(np.abs(actual - predicted) <= tolerance_mw) * 100)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the GES hybrid model on held-out real data.")
    parser.add_argument("--plant-id", type=int, required=True)
    parser.add_argument("--test-days", type=int, default=30, help="Most recent N days held out for testing")
    parser.add_argument(
        "--test-fraction", type=float, default=None,
        help="Use a %%-based split instead of --test-days (e.g. 0.2 for the most recent 20%% of days)",
    )
    parser.add_argument("--tolerance-pct", type=float, default=3.0, help="Hit-rate tolerance, as %% of capacity")
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

    train_df, test_df = split_train_test(df, test_days=args.test_days, test_fraction=args.test_fraction)
    logger.info(
        "Train: %d rows (%s to %s) | Test: %d rows (%s to %s)",
        len(train_df), train_df["timestamp"].min(), train_df["timestamp"].max(),
        len(test_df), test_df["timestamp"].min(), test_df["timestamp"].max(),
    )

    lat, lon, capacity_mw = plant["lat"], plant["lon"], plant["capacity_mw"]

    # Kalibrasyon sadece train_df'ten - test setine sızmasın, dürüst bir
    # held-out değerlendirme olsun.
    calibration = calibrate_site_parameters(train_df, lat, lon, capacity_mw)
    temp_coeff, efficiency_scale = calibration["temp_coeff"], calibration["efficiency_scale"]
    ac_capacity_mw = calibration["ac_capacity_mw"]
    logger.info(
        "Kalibrasyon (train'den): efficiency_scale=%.4f, temp_coeff=%.5f, ac_capacity_mw=%.1f",
        efficiency_scale, temp_coeff, ac_capacity_mw,
    )

    baseline_train = build_physical_baseline(train_df, lat, lon, capacity_mw, temp_coeff, efficiency_scale)
    model = train_residual_model(train_df, baseline_train)

    baseline_test = build_physical_baseline(test_df, lat, lon, capacity_mw, temp_coeff, efficiency_scale)
    hybrid_test = predict_production(
        test_df, lat, lon, capacity_mw, model,
        temp_coeff=temp_coeff, efficiency_scale=efficiency_scale, ac_capacity_mw=ac_capacity_mw,
    )

    actual = test_df["production_mwh"]
    physical_mae = mean_absolute_error(actual, baseline_test)
    physical_rmse = np.sqrt(mean_squared_error(actual, baseline_test))
    hybrid_mae = mean_absolute_error(actual, hybrid_test)
    hybrid_rmse = np.sqrt(mean_squared_error(actual, hybrid_test))
    improvement = (1 - hybrid_mae / physical_mae) * 100 if physical_mae else 0.0

    physical_nmae = normalized_mae(actual, baseline_test, capacity_mw)
    hybrid_nmae = normalized_mae(actual, hybrid_test, capacity_mw)

    physical_hit = hit_rate(actual, baseline_test, capacity_mw, args.tolerance_pct)
    hybrid_hit = hit_rate(actual, hybrid_test, capacity_mw, args.tolerance_pct)

    logger.info("=== Test seti sonuçları (%s) ===", plant["name"])
    logger.info("Sadece fiziksel model : MAE=%.3f MWh, RMSE=%.3f MWh, nMAE=%%%.1f", physical_mae, physical_rmse, physical_nmae)
    logger.info("Hibrit (fiziksel+ML)  : MAE=%.3f MWh, RMSE=%.3f MWh, nMAE=%%%.1f", hybrid_mae, hybrid_rmse, hybrid_nmae)
    logger.info("İyileşme (MAE): %%%.1f", improvement)
    logger.info("Doğruluk (kapasiteye göre, 100-nMAE): %%%.1f", 100 - hybrid_nmae)
    logger.info(
        "Tutturma oranı (±%%%.0f kapasite içinde kalan saat): Fiziksel=%%%.1f, Hibrit=%%%.1f",
        args.tolerance_pct, physical_hit, hybrid_hit,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
