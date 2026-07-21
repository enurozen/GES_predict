"""
Generate a local, self-contained HTML report comparing actual production
against the physical-only and hybrid model predictions on held-out data -
a daily overview chart plus a per-day hourly drill-down (chart or table).

No credentials needed - reads local data/<plant_id>/*.csv only. Open the
output file directly in a browser (double-click it), no server required.

Usage:
    python report.py --plant-id 2579 --output reports/2579.html
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from evaluate import hit_rate, normalized_mae
from ges_uretim_tahmini import build_physical_baseline, calibrate_site_parameters, predict_production, train_residual_model
from plants import PlantNotFoundError, load_plant
from train import load_training_data, split_train_test

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "report_template.html"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a local HTML actual-vs-predicted report.")
    parser.add_argument("--plant-id", type=int, required=True)
    parser.add_argument("--test-days", type=int, default=30, help="Most recent N days held out for the report")
    parser.add_argument(
        "--test-fraction", type=float, default=None,
        help="Use a %%-based split instead of --test-days (e.g. 0.2 for the most recent 20%% of days)",
    )
    parser.add_argument("--tolerance-pct", type=float, default=3.0, help="Hit-rate tolerance, as %% of capacity")
    parser.add_argument("--output", default="reports/report.html", help="Output HTML path")
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

    lat, lon, capacity_mw = plant["lat"], plant["lon"], plant["capacity_mw"]
    calibration = calibrate_site_parameters(train_df, lat, lon, capacity_mw)
    temp_coeff, efficiency_scale = calibration["temp_coeff"], calibration["efficiency_scale"]
    ac_capacity_mw = calibration["ac_capacity_mw"]

    baseline_train = build_physical_baseline(train_df, lat, lon, capacity_mw, temp_coeff, efficiency_scale)
    model = train_residual_model(train_df, baseline_train)

    baseline_test = build_physical_baseline(test_df, lat, lon, capacity_mw, temp_coeff, efficiency_scale)
    hybrid_test = predict_production(
        test_df, lat, lon, capacity_mw, model,
        temp_coeff=temp_coeff, efficiency_scale=efficiency_scale, ac_capacity_mw=ac_capacity_mw,
    )

    actual = test_df["production_mwh"]
    physical_mae = mean_absolute_error(actual, baseline_test)
    hybrid_mae = mean_absolute_error(actual, hybrid_test)
    improvement = (1 - hybrid_mae / physical_mae) * 100 if physical_mae else 0.0
    hybrid_nmae = normalized_mae(actual, hybrid_test, capacity_mw)
    hybrid_hit = hit_rate(actual, hybrid_test, capacity_mw, args.tolerance_pct)

    rows = [
        {
            "t": test_df["timestamp"].iloc[i].isoformat(),
            "actual": round(float(actual.iloc[i]), 2),
            "physical": round(float(baseline_test.iloc[i]), 2),
            "hybrid": round(float(hybrid_test.iloc[i]), 2),
        }
        for i in range(len(test_df))
    ]

    html = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = html.replace("__PLANT_NAME__", plant["name"])
    html = html.replace("__DATA__", json.dumps(rows, ensure_ascii=False))
    html = html.replace("__RANGE__", f"{test_df['timestamp'].min().date()} – {test_df['timestamp'].max().date()}")
    test_label = f"%{round(args.test_fraction * 100)}" if args.test_fraction is not None else f"{args.test_days} gün"
    html = html.replace("__TEST_FRACTION_PCT__", test_label)
    html = html.replace("__PHYS_MAE__", f"{physical_mae:.1f}")
    html = html.replace("__HYB_MAE__", f"{hybrid_mae:.1f}")
    html = html.replace("__IMPROVE__", f"{improvement:.1f}")
    html = html.replace("__ACCURACY__", f"{100 - hybrid_nmae:.1f}")
    html = html.replace("__TOLERANCE__", f"{args.tolerance_pct:.0f}")
    html = html.replace("__HIT_RATE__", f"{hybrid_hit:.1f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("Wrote report to %s - open it directly in a browser", output_path.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
