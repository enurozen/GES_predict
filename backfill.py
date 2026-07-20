"""
Backfill historical generation+weather data for a date range, one CSV per
day, matching the data/<plant_id>/<date>.csv structure daily_datapull.yml
maintains going forward.

EPİAŞ and Open-Meteo are both historical/archive APIs - there's no need to
wait for the daily automation to accumulate days one at a time; this fills
in a whole past range in a single run.

Days that already have a CSV under data/<plant_id>/ are skipped by default,
so an interrupted or repeated backfill is safe to re-run. Pass --overwrite
to refetch and replace them anyway.

Usage:
    python backfill.py --plant-id 2579 --start 2025-01-01 --end 2026-07-19
"""

import argparse
import logging
import os
import sys
from datetime import date, timedelta
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
        description="Backfill historical GES training data, one CSV per day."
    )
    parser.add_argument(
        "--plant-id", type=int, required=True,
        help="EPİAŞ santral (power plant) ID; must have an entry in plants.yaml",
    )
    parser.add_argument("--start", type=date.fromisoformat, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=date.fromisoformat, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Refetch and overwrite days that already have a CSV (default: skip them).",
    )
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

    out_dir = Path("data") / str(args.plant_id)
    days = [args.start + timedelta(days=i) for i in range((args.end - args.start).days + 1)]

    if not args.overwrite:
        pending = [d for d in days if not (out_dir / f"{d.isoformat()}.csv").exists()]
        skipped = len(days) - len(pending)
        if skipped:
            logger.info("Skipping %d day(s) that already have a CSV.", skipped)
        days = pending

    if not days:
        logger.info("Nothing to backfill, every day in range already has a CSV.")
        return 0

    fetch_start, fetch_end = days[0], days[-1]

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
            args.plant_id, fetch_start, fetch_end,
        )
        generation_df = fetch_generation_range(tgt, args.plant_id, fetch_start, fetch_end)
        if generation_df.empty:
            logger.error("No generation data returned for the given plant/date range.")
            return 1

        logger.info("Fetching weather data for %s (%s, %s)...", plant["name"], plant["lat"], plant["lon"])
        weather_df = fetch_weather_range(plant["lat"], plant["lon"], fetch_start, fetch_end)

        logger.info("Merging generation and weather data...")
        training_df = build_training_dataset(generation_df, weather_df)
        if training_df.empty:
            logger.error("No overlapping hours between generation and weather data for the requested range.")
            return 1
    except ApiError as exc:
        logger.error("%s", exc)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for day, day_df in training_df.groupby(training_df["timestamp"].dt.date):
        day_df.to_csv(out_dir / f"{day.isoformat()}.csv", index=False)
        written += 1

    logger.info("Wrote %d daily CSV file(s) to %s", written, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
