"""
Honest backtest of real-world day-ahead forecast accuracy: trains on
everything before the test window (same as evaluate.py), but scores the
test window against archived FORECAST weather (historical-forecast-api)
instead of archived ACTUAL weather (archive-api).

evaluate.py answers "how good is the model if the weather forecast were
perfect?" - it feeds the model archive-api's after-the-fact reanalysis for
the test days, which predict.py never has access to in real time. This
script answers the question that actually matters for a live 1-day-ahead
forecast: how accurate is predict.py once real forecast error is included?

Usage:
    python backtest_forecast.py --plant-id 2579 --test-days 14
"""

import argparse
import logging
import sys

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from ges_uretim_tahmini import build_physical_baseline, calibrate_site_parameters, predict_production, train_residual_model
from evaluate import hit_rate, normalized_mae
from plants import PlantNotFoundError, geometry_kwargs, load_plant
from shared import ApiError
from train import load_training_data, split_train_test
from weather import fetch_historical_forecast

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest day-ahead forecast accuracy using real archived forecast weather."
    )
    parser.add_argument("--plant-id", type=int, required=True)
    parser.add_argument("--test-days", type=int, default=14, help="Most recent N days held out for testing")
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

    train_df, test_df = split_train_test(df, test_days=args.test_days)
    test_start, test_end = test_df["timestamp"].min().date(), test_df["timestamp"].max().date()
    logger.info(
        "Train: %d rows (%s to %s) | Test: %d rows (%s to %s)",
        len(train_df), train_df["timestamp"].min(), train_df["timestamp"].max(),
        len(test_df), test_start, test_end,
    )

    lat, lon, capacity_mw = plant["lat"], plant["lon"], plant["capacity_mw"]
    geometry = geometry_kwargs(plant)

    # Kalibrasyon ve model sadece train_df'ten - test setine sızmasın.
    calibration = calibrate_site_parameters(train_df, lat, lon, capacity_mw, **geometry)
    temp_coeff, efficiency_scale = calibration["temp_coeff"], calibration["efficiency_scale"]
    ac_capacity_mw = calibration["ac_capacity_mw"]

    baseline_train = build_physical_baseline(train_df, lat, lon, capacity_mw, temp_coeff, efficiency_scale, **geometry)
    model = train_residual_model(train_df, baseline_train, lat, lon)

    # Test penceresi icin GERCEK arsivlenmis FORECAST hava verisi (archive-api
    # DEGIL - o gercekten ne oldugunu soyler, biz o gun gercekte ne tahmin
    # edilmis oldugunu istiyoruz).
    logger.info("Fetching archived forecast weather for %s to %s...", test_start, test_end)
    try:
        forecast_weather = fetch_historical_forecast(lat, lon, test_start, test_end)
    except ApiError as exc:
        logger.error("Could not fetch historical forecast weather: %s", exc)
        return 1

    if forecast_weather.empty:
        logger.error("No archived forecast weather returned for %s to %s.", test_start, test_end)
        return 1

    # Gercek uretimle (test_df) forecast havayi timestamp uzerinden birlestir.
    aligned = test_df[["timestamp", "production_mwh"]].merge(
        forecast_weather, on="timestamp", how="inner",
    )
    dropped = len(test_df) - len(aligned)
    if dropped:
        logger.warning("%d test hour(s) had no matching forecast weather and were dropped.", dropped)
    if aligned.empty:
        logger.error("No overlapping hours between test data and forecast weather.")
        return 1

    actual = aligned["production_mwh"]

    baseline_forecast = build_physical_baseline(aligned, lat, lon, capacity_mw, temp_coeff, efficiency_scale, **geometry)
    hybrid_forecast = predict_production(
        aligned, lat, lon, capacity_mw, model,
        temp_coeff=temp_coeff, efficiency_scale=efficiency_scale, ac_capacity_mw=ac_capacity_mw,
        **geometry,
    )

    # Karsilastirma icin: ayni test penceresi, ama "mukemmel hava bilgisi"
    # varsayimiyla (evaluate.py'nin yaptigi gibi archive-api / gercek hava).
    baseline_hindsight = build_physical_baseline(test_df, lat, lon, capacity_mw, temp_coeff, efficiency_scale, **geometry)
    hybrid_hindsight = predict_production(
        test_df, lat, lon, capacity_mw, model,
        temp_coeff=temp_coeff, efficiency_scale=efficiency_scale, ac_capacity_mw=ac_capacity_mw,
        **geometry,
    )
    actual_hindsight = test_df["production_mwh"]

    def report(label: str, actual_s, pred_s):
        mae = mean_absolute_error(actual_s, pred_s)
        rmse = np.sqrt(mean_squared_error(actual_s, pred_s))
        nmae = normalized_mae(actual_s, pred_s, capacity_mw)
        hit = hit_rate(actual_s, pred_s, capacity_mw, args.tolerance_pct)
        logger.info(
            "%-32s MAE=%.3f MWh  RMSE=%.3f MWh  nMAE=%%%.2f  doğruluk=%%%.2f  tutturma(±%%%.0f)=%%%.1f",
            label, mae, rmse, nmae, 100 - nmae, args.tolerance_pct, hit,
        )

    logger.info("=== Gerçek gün-öncesi (forecast weather) vs 'mükemmel hava bilgisi' (hindsight) - %s ===", plant["name"])
    logger.info("--- Fiziksel model ---")
    report("Hindsight (archive-api)", actual_hindsight, baseline_hindsight)
    report("Gerçek forecast (day-ahead)", actual, baseline_forecast)
    logger.info("--- Hibrit model (fiziksel+ML) ---")
    report("Hindsight (archive-api)", actual_hindsight, hybrid_hindsight)
    report("Gerçek forecast (day-ahead)", actual, hybrid_forecast)
    return 0


if __name__ == "__main__":
    sys.exit(main())
