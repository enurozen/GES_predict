"""
Plant registry: reads per-santral identity (lat, lon, capacity_mw) from
plants.yaml, so it's defined once instead of being re-typed on every
main.py call or in the workflow file.
"""

from pathlib import Path
from typing import Any

import yaml

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
