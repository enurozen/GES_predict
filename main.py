"""
CLI to build a training dataset for the GES production model.

Pipeline: authenticate with EPİAŞ -> fetch hourly generation -> fetch hourly
weather -> merge on timestamp -> write CSV.

Credentials are read from the EPIAS_USERNAME / EPIAS_PASSWORD environment
variables; they are never hardcoded or logged.

Plant identity (lat/lon/capacity_mw) is looked up from plants.yaml by
--plant-id rather than typed in by hand, so it can't drift between runs.

Usage:
    python main.py --plant-id 2579 \\
        --start 2025-06-01 --end 2025-06-30 --output data/training_set.csv
"""

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

from epias import fetch_generation_range, get_tgt
from merge import build_training_dataset
from plants import PlantNotFoundError, load_plant
from shared import ApiError
from weather import fetch_weather_range

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a GES training dataset from EPİAŞ generation and Open-Meteo weather data."
    )
    parser.add_argument(
        "--plant-id", type=int, required=True,
        help="EPİAŞ santral (power plant) ID; must have an entry in plants.yaml",
    )
    parser.add_argument("--start", type=date.fromisoformat, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=date.fromisoformat, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.start > args.end:
        logger.error("Start date cannot be after end date.")
        return 1

    today = date.today()
    if args.end >= today:
        logger.error(
            "End date (%s) must be before today (%s): EPİAŞ generation data "
            "isn't settled/available until the day is fully over.",
            args.end, today,
        )
        return 1

    try:
        plant = load_plant(args.plant_id)
    except PlantNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    username = os.environ.get("EPIAS_USERNAME")
    password = os.environ.get("EPIAS_PASSWORD")
    if not username or not password:
        logger.error("EPIAS_USERNAME and EPIAS_PASSWORD environment variables must be set.")
        return 1

    try:
        logger.info("Authenticating with EPİAŞ...")
        tgt = get_tgt(username, password)

        logger.info(
            "Fetching generation data for plant %s (%s to %s)...",
            args.plant_id, args.start, args.end,
        )
        generation_df = fetch_generation_range(tgt, args.plant_id, args.start, args.end)
        if generation_df.empty:
            logger.error("No generation data returned for the given plant/date range.")
            return 1

        logger.info("Fetching weather data for %s (%s, %s)...", plant["name"], plant["lat"], plant["lon"])
        weather_df = fetch_weather_range(plant["lat"], plant["lon"], args.start, args.end)

        logger.info("Merging generation and weather data...")
        training_df = build_training_dataset(generation_df, weather_df)
        if training_df.empty:
            logger.error("No overlapping hours between generation and weather data.")
            return 1
    except ApiError as exc:
        logger.error("%s", exc)
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(output_path, index=False)
    logger.info("Wrote %d rows to %s", len(training_df), output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
