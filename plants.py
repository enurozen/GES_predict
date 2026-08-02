"""
Plant registry: reads per-santral identity (lat, lon, capacity_mw) from
plants.yaml, so it's defined once instead of being re-typed on every
main.py call or in the workflow file.
"""

from pathlib import Path
from typing import Any

import yaml

from ges_uretim_tahmini import DEFAULT_GCR

REGISTRY_PATH = Path(__file__).resolve().parent / "plants.yaml"


class PlantNotFoundError(Exception):
    """Raised when a plant ID has no entry in the registry."""


def load_plant(plant_id: int, registry_path: Path = REGISTRY_PATH) -> dict[str, Any]:
    """Look up a plant's lat/lon/capacity_mw/name from the registry file."""
    with open(registry_path, encoding="utf-8") as f:
        registry = yaml.safe_load(f) or {}

    plant = registry.get(plant_id)
    if plant is None:
        raise PlantNotFoundError(
            f"Plant ID {plant_id} not found in {registry_path}. "
            "Add an entry with lat/lon/capacity_mw before pulling its data."
        )
    return plant


def geometry_kwargs(plant: dict[str, Any]) -> dict[str, Any]:
    """
    Extract the optional panel-geometry fields (tilt_deg, panel_azimuth_deg,
    tracker_type, module_noct_c, gcr) from a plant registry entry, as kwargs
    ready to pass to calibrate_site_parameters/build_physical_baseline/
    predict_production/train_residual_model.

    Missing fields default to None (unknown geometry -> flat-GHI fallback in
    ges_uretim_tahmini.py), except module_noct_c/gcr which default to
    industry-typical values (45°C NOCT, 0.4 ground coverage ratio) when no
    datasheet value is registered.
    """
    return {
        "tilt_deg": plant.get("tilt_deg"),
        "panel_azimuth_deg": plant.get("azimuth_deg"),
        "tracker_type": plant.get("tracker_type"),
        "module_noct_c": plant.get("module_noct_c", 45.0),
        "gcr": plant.get("gcr", DEFAULT_GCR),
    }
