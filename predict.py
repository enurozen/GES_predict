"""
Forecast future production using an already-trained model: no EPİAŞ
credentials needed (there's no real generation to fetch for future dates),
only Open-Meteo's forecast weather data for the requested range.

Usage:
    python predict.py --plant-id 2579 --start 2026-07-22 --end 2026-07-24 \\
        --model models/2579/model.joblib --output predictions/2579.csv
"""

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import joblib

from ges_uretim_tahmini import predict_production
from shared import ApiError
from weather import fetch_weather_forecast

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forecast future GES production from a trained model.")
    parser.add_argument("--plant-id", type=int, required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=date.fromisoformat, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--model", default=None,
        help="Path to trained model (joblib). Defaults to models/<plant_id>/model.joblib",
    )
    parser.add_argument("--output", required=True, help="Output CSV path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.start > args.end:
        logger.error("Start date cannot be after end date.")
        return 1

    model_path = Path(args.model) if args.model else Path("models") / str(args.plant_id) / "model.joblib"
    if not model_path.exists():
        logger.error(
            "No trained model at %s. Run train.py --plant-id %s first.", model_path, args.plant_id,
        )
        return 1

    bundle = joblib.load(model_path)
    model, calibration, plant = bundle["model"], bundle["calibration"], bundle["plant"]
    lat, lon, capacity_mw = plant["lat"], plant["lon"], plant["capacity_mw"]

    try:
        logger.info(
            "Fetching weather forecast for %s (%s to %s)...", plant["name"], args.start, args.end,
        )
        weather_df = fetch_weather_forecast(lat, lon, args.start, args.end)
        if weather_df.empty:
            logger.error(
                "No forecast data returned - Open-Meteo's forecast API only covers "
                "today through ~16 days ahead; the requested range may be out of that window."
            )
            return 1

        prediction = predict_production(
            weather_df, lat, lon, capacity_mw, model,
            temp_coeff=calibration["temp_coeff"], efficiency_scale=calibration["efficiency_scale"],
            ac_capacity_mw=calibration["ac_capacity_mw"],
        )
    except ApiError as exc:
        logger.error("%s", exc)
        return 1

    out_df = weather_df[["timestamp"]].copy()
    out_df["predicted_mwh"] = prediction.round(2)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    logger.info(
        "Wrote %d hourly predictions (toplam %.1f MWh) to %s",
        len(out_df), out_df["predicted_mwh"].sum(), output_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
