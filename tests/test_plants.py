import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from plants import PlantNotFoundError, load_plant


def _write_registry(tmp_path: Path, contents: dict) -> Path:
    path = tmp_path / "plants.yaml"
    path.write_text(yaml.safe_dump(contents), encoding="utf-8")
    return path


def test_load_plant_returns_registered_entry(tmp_path):
    registry_path = _write_registry(
        tmp_path,
        {2579: {"name": "Karapınar GES", "lat": 39.9, "lon": 32.8, "capacity_mw": 10.0}},
    )

    plant = load_plant(2579, registry_path=registry_path)

    assert plant == {"name": "Karapınar GES", "lat": 39.9, "lon": 32.8, "capacity_mw": 10.0}


def test_load_plant_unregistered_id_raises(tmp_path):
    registry_path = _write_registry(
        tmp_path,
        {2579: {"name": "Karapınar GES", "lat": 39.9, "lon": 32.8, "capacity_mw": 10.0}},
    )

    with pytest.raises(PlantNotFoundError, match="9999"):
        load_plant(9999, registry_path=registry_path)


def test_load_plant_uses_default_registry_path():
    plant = load_plant(2579)

    assert plant["name"] == "Kalyon Karapınar YEKA-1 GES"
    assert plant["lat"] == pytest.approx(37.7908)
    assert plant["lon"] == pytest.approx(33.5847)
