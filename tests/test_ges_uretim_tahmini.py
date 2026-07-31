import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ges_uretim_tahmini as g

LAT, LON = 37.7908, 33.5847  # Karapınar


# --------------------------------------------------------------------------
# solar_azimuth_deg
# --------------------------------------------------------------------------

def test_solar_azimuth_is_zero_at_solar_noon():
    # Solar noon (local hour 12, no lon correction in this simplified model)
    # on an equinox-ish date: sun should be (near) due South -> azimuth ~ 0
    # in the South=0 convention used throughout this module.
    ts = pd.Timestamp("2026-03-20 12:00")
    zenith, cos_z = g.solar_position(ts, LAT, LON)
    az = g.solar_azimuth_deg(ts, LAT, LON, zenith, cos_z)
    assert az == pytest.approx(0.0, abs=1.0)


def test_solar_azimuth_sign_morning_vs_afternoon():
    # East (morning, before solar noon) should be negative azimuth,
    # West (afternoon) positive, in the South=0/West=+90 convention.
    morning = pd.Timestamp("2026-06-21 08:00")
    afternoon = pd.Timestamp("2026-06-21 16:00")

    az_morning = g.solar_azimuth_deg(morning, LAT, LON)
    az_afternoon = g.solar_azimuth_deg(afternoon, LAT, LON)

    assert az_morning < 0
    assert az_afternoon > 0


# --------------------------------------------------------------------------
# erbs_decomposition
# --------------------------------------------------------------------------

def test_erbs_decomposition_zero_ghi_returns_zero():
    dni, dhi = g.erbs_decomposition(0.0, 0.8)
    assert (dni, dhi) == (0.0, 0.0)


def test_erbs_decomposition_night_returns_zero():
    dni, dhi = g.erbs_decomposition(100.0, 0.0)
    assert (dni, dhi) == (0.0, 0.0)


def test_erbs_decomposition_high_clearness_is_mostly_direct():
    # kt close to 1 (GHI close to extraterrestrial) -> low diffuse fraction.
    cos_zenith = 0.9
    ghi_et = g.SOLAR_CONSTANT_W_M2 * cos_zenith
    dni, dhi = g.erbs_decomposition(ghi_et * 0.85, cos_zenith)

    assert dni > dhi  # mostly direct under very clear conditions
    assert dni >= 0 and dhi >= 0


def test_erbs_decomposition_low_clearness_is_mostly_diffuse():
    cos_zenith = 0.9
    ghi_et = g.SOLAR_CONSTANT_W_M2 * cos_zenith
    dni, dhi = g.erbs_decomposition(ghi_et * 0.1, cos_zenith)  # heavily overcast

    assert dhi > dni


# --------------------------------------------------------------------------
# poa_irradiance / AOI
# --------------------------------------------------------------------------

def test_flat_panel_cos_aoi_matches_cos_zenith_regardless_of_azimuth():
    # tilt=0 -> the azimuth term drops out entirely (sin(0)=0), so POA from
    # DNI alone should equal DNI * cos(zenith) for ANY panel azimuth.
    zenith_deg = 30.0
    for panel_az in [-90.0, 0.0, 90.0, 180.0]:
        cos_aoi = g._cos_aoi_fixed_tilt(zenith_deg, sun_azimuth_deg := 45.0, tilt_deg=0.0,
                                         panel_azimuth_deg=panel_az)
        assert cos_aoi == pytest.approx(np.cos(np.radians(zenith_deg)), abs=1e-9)


def test_fixed_tilt_facing_sun_maximizes_cos_aoi():
    # Panel tilted exactly at the zenith angle and facing the sun's azimuth
    # should be perpendicular to the sun -> cos(AOI) ~ 1.
    zenith_deg, sun_az = 40.0, 25.0
    cos_aoi = g._cos_aoi_fixed_tilt(zenith_deg, sun_az, tilt_deg=zenith_deg, panel_azimuth_deg=sun_az)
    assert cos_aoi == pytest.approx(1.0, abs=1e-9)


def test_tracker_rotation_near_zero_at_solar_noon():
    # Sun due South (azimuth=0 in this convention) -> a horizontal N-S axis
    # tracker should be flat (no reason to tilt East or West).
    rotation_deg, cos_aoi = g._tracker_rotation_and_cos_aoi(zenith_deg=30.0, azimuth_deg_sun=0.0)
    assert rotation_deg == pytest.approx(0.0, abs=1e-6)
    assert cos_aoi == pytest.approx(np.cos(np.radians(30.0)), abs=1e-9)


def test_tracker_rotation_clips_to_max():
    # Near-horizon sun with a large azimuth offset would need >60 deg
    # rotation - must clip to the mechanical limit.
    rotation_deg, _ = g._tracker_rotation_and_cos_aoi(zenith_deg=85.0, azimuth_deg_sun=89.0,
                                                        max_rotation_deg=60.0)
    assert abs(rotation_deg) <= 60.0 + 1e-9


def test_poa_irradiance_zero_below_horizon():
    assert g.poa_irradiance(500, 100, 400, zenith_deg=95, azimuth_deg_sun=0,
                             tilt_deg=20, panel_azimuth_deg=0, tracker_type=None) == 0.0


def test_poa_irradiance_flat_dhi_only_matches_isotropic_formula():
    # tilt=0 -> sky-view factor (1+cos(0))/2 = 1, ground term (1-cos(0))/2 = 0
    # -> POA should equal DNI*cos(zenith) + DHI exactly (pure horizontal GHI decomposition).
    zenith_deg = 20.0
    poa = g.poa_irradiance(dni=600, dhi=80, ghi=600 * np.cos(np.radians(zenith_deg)) + 80,
                            zenith_deg=zenith_deg, azimuth_deg_sun=10.0,
                            tilt_deg=0.0, panel_azimuth_deg=0.0, tracker_type=None)
    expected = 600 * np.cos(np.radians(zenith_deg)) + 80
    assert poa == pytest.approx(expected, rel=1e-6)


# --------------------------------------------------------------------------
# cell_temperature
# --------------------------------------------------------------------------

def test_cell_temperature_equals_ambient_at_zero_irradiance():
    assert g.cell_temperature(0.0, temp_c=22.0, noct_c=45.0) == pytest.approx(22.0)


def test_cell_temperature_rises_with_irradiance():
    cold = g.cell_temperature(200.0, temp_c=20.0, noct_c=45.0)
    hot = g.cell_temperature(900.0, temp_c=20.0, noct_c=45.0)
    assert hot > cold


# --------------------------------------------------------------------------
# build_physical_baseline: flat-GHI fallback vs POA path
# --------------------------------------------------------------------------

def _sample_df():
    return pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-06-21 00:00", "2026-06-21 12:00"]),
        "ghi_forecast": [0.0, 700.0],
        "dni_forecast": [0.0, 750.0],
        "dhi_forecast": [0.0, 90.0],
        "temp_c": [18.0, 32.0],
    })


def test_build_physical_baseline_unknown_geometry_matches_legacy_formula():
    df = _sample_df()
    baseline = g.build_physical_baseline(df, LAT, LON, capacity_mw=100.0)

    assert baseline.iloc[0] == pytest.approx(0.0)  # midnight -> no production
    assert baseline.iloc[1] > 0  # noon -> some production

    # Cross-check against the original flat-GHI formula directly.
    zenith, cos_z = g.solar_position(df["timestamp"].iloc[1], LAT, LON)
    ghi_cs = g.clearsky_ghi_estimate(cos_z)
    expected = g.physical_pv_power(df["ghi_forecast"].iloc[1], df["temp_c"].iloc[1], 100.0, ghi_cs)
    assert baseline.iloc[1] == pytest.approx(expected)


def test_build_physical_baseline_known_tilt_uses_poa_path_and_differs_from_flat():
    df = _sample_df()
    flat = g.build_physical_baseline(df, LAT, LON, capacity_mw=100.0)
    tilted = g.build_physical_baseline(df, LAT, LON, capacity_mw=100.0, tilt_deg=20.0, panel_azimuth_deg=0.0)

    assert tilted.iloc[0] == pytest.approx(0.0)
    # Different physical model (POA vs flat GHI) should generally produce a
    # different noon value - this just confirms the POA branch is actually
    # taken, not silently falling back to the legacy path.
    assert tilted.iloc[1] != pytest.approx(flat.iloc[1])


def test_build_physical_baseline_missing_dni_dhi_falls_back_to_erbs():
    df = _sample_df().drop(columns=["dni_forecast", "dhi_forecast"])
    # Should not raise even though dni/dhi columns are entirely absent
    # (e.g. older data/ CSVs collected before those columns existed).
    baseline = g.build_physical_baseline(df, LAT, LON, capacity_mw=100.0, tilt_deg=20.0, panel_azimuth_deg=0.0)
    assert baseline.iloc[1] > 0


# --------------------------------------------------------------------------
# build_features: new columns
# --------------------------------------------------------------------------

def test_build_features_includes_new_generalizable_columns():
    df = _sample_df()
    feats = g.build_features(df, LAT, LON)

    assert "solar_elevation_deg" in feats.columns
    assert "clear_sky_index" in feats.columns
    assert "ghi_ramp_1h" in feats.columns

    # Midnight: sun below horizon -> elevation negative, clear-sky index 0.
    assert feats["solar_elevation_deg"].iloc[0] < 0
    assert feats["clear_sky_index"].iloc[0] == pytest.approx(0.0)
    # Noon: sun up -> positive elevation, clear-sky index in a sane range.
    assert feats["solar_elevation_deg"].iloc[1] > 0
    assert 0.0 <= feats["clear_sky_index"].iloc[1] <= 1.5
    # Ramp is the diff from the previous row; first row has no predecessor.
    assert feats["ghi_ramp_1h"].iloc[0] == pytest.approx(0.0)
    assert feats["ghi_ramp_1h"].iloc[1] == pytest.approx(700.0)
