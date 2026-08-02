"""
Train ONE residual model pooled across multiple plants ("fleet" training).

Rationale: build_features already normalizes by capacity and geography
(clear_sky_index, solar_elevation_deg instead of raw GHI/MW) - pooling
several plants' data lets the residual layer learn a shared "cloud
response"/ramp-rate pattern more robustly than fitting one small model per
plant, which matters most for a newly-commissioned plant with little
history of its own. The residual target is also capacity-normalized
(fraction of nameplate capacity) so plants of very different sizes can be
pooled without one dominating the loss.

Works with ANY number of plants.yaml entries - today that's just Karapınar,
so this trains a "fleet of one" (architecturally identical to train.py's
result, just a different bundle format). The moment a second plant is
registered, pass both IDs and pooling kicks in with no code changes.

Usage:
    python train_fleet.py --plant-ids 2579 --output models/fleet/model.joblib
    python train_fleet.py --plant-ids 2579,1234 --output models/fleet/model.joblib

NOT wired into predict.py yet (different bundle format - see
ges_uretim_tahmini.predict_production_from_fleet for how to consume it).
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from ges_uretim_tahmini import build_features, build_physical_baseline, calibrate_site_parameters
from plants import PlantNotFoundError, geometry_kwargs, load_plant
from train import load_training_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one residual model pooled across multiple plants.")
    parser.add_argument("--plant-ids", required=True, help="Comma-separated plant IDs (must all exist in plants.yaml)")
    parser.add_argument("--output", required=True, help="Output path for the trained fleet model (joblib)")
    return parser.parse_args(argv)


def build_pooled_dataset(plant_ids: list[int]) -> tuple[pd.DataFrame, pd.Series, dict, dict]:
    """
    Her santral için ayrı ayrı kalibrasyon + fiziksel baz model hesaplar,
    sonra kapasiteye göre normalize edilmiş residual'ları (MW/MWp) ve
    özellikleri tek bir havuzda birleştirir. calibrate_site_parameters her
    santral için AYRI çalışır - havuzlanan sadece ML katmanı, fiziksel
    kalibrasyon santrale özgü kalır (efficiency_scale/temp_coeff santraller
    arası paylaşılmaz, doğru olmaz).
    """
    calibrations, plants = {}, {}
    pooled_features, pooled_residual = [], []

    for plant_id in plant_ids:
        plant = load_plant(plant_id)
        df = load_training_data(plant_id)

        geometry = geometry_kwargs(plant)
        calibration = calibrate_site_parameters(
            df, plant["lat"], plant["lon"], plant["capacity_mw"], **geometry,
        )
        baseline = build_physical_baseline(
            df, plant["lat"], plant["lon"], plant["capacity_mw"],
            temp_coeff=calibration["temp_coeff"], efficiency_scale=calibration["efficiency_scale"],
            **geometry,
        )
        feats = build_features(df, plant["lat"], plant["lon"])
        residual_normalized = (df["production_mwh"] - baseline) / plant["capacity_mw"]

        calibrations[plant_id] = calibration
        plants[plant_id] = plant
        pooled_features.append(feats)
        pooled_residual.append(residual_normalized)

        logger.info(
            "Plant %s (%s): %d satır, efficiency_scale=%.4f, temp_coeff=%.5f",
            plant_id, plant["name"], len(df), calibration["efficiency_scale"], calibration["temp_coeff"],
        )

    X = pd.concat(pooled_features, ignore_index=True)
    y = pd.concat(pooled_residual, ignore_index=True).reset_index(drop=True)
    return X, y, calibrations, plants


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        plant_ids = [int(p.strip()) for p in args.plant_ids.split(",") if p.strip()]
    except ValueError:
        logger.error("--plant-ids must be a comma-separated list of integers, e.g. 2579,1234")
        return 1
    if not plant_ids:
        logger.error("--plant-ids must contain at least one plant ID.")
        return 1

    try:
        X, y, calibrations, plants = build_pooled_dataset(plant_ids)
    except (PlantNotFoundError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 1

    logger.info("Havuzlanan eğitim seti: %d satır, %d santral", len(X), len(plant_ids))

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    model = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42)
    model.fit(X_train, y_train)

    val_mae = mean_absolute_error(y_val, model.predict(X_val))
    logger.info("[Fleet residual model doğrulama] MAE=%.5f (kapasiteye göre normalize, MW/MWp)", val_mae)

    bundle = {
        "model": model,
        "calibrations": calibrations,
        "plants": plants,
        "plant_ids": plant_ids,
        "residual_normalized_by_capacity": True,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)
    logger.info("Wrote pooled fleet model to %s (%d plant(s))", output_path, len(plant_ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
