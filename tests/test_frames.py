"""Tests for src/frames.py.

Two kinds of test live here, and the distinction matters:

**Pure tests** check properties that hold by definition, with values written
into the test itself. They need no data on disk, so they run anywhere, including
on a machine that has never downloaded a bag.

**Data tests** check the module against the real Urban04 tables. They are more
convincing, because they exercise the same path the project actually uses, but
they are skipped automatically when the Parquet files are absent. A test that
fails because data is missing tells you nothing useful.

Run with::

    pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.frames import (  # noqa: E402
    Origin,
    add_enu,
    azimuth_to_yaw,
    enu_to_geodetic,
    geodetic_to_enu,
    origin_from_reference,
    wrap_angle,
    yaw_to_azimuth,
)

PROCESSED = REPO / "data" / "processed" / "Urban04"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reference() -> pd.DataFrame:
    """The Urban04 ground-truth table, or skip if it has not been built."""
    path = PROCESSED / "reference.parquet"
    if not path.exists():
        pytest.skip(f"{path} missing; run src/data_loader.py --sequence Urban04")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def gnss() -> pd.DataFrame:
    path = PROCESSED / "gnss.parquet"
    if not path.exists():
        pytest.skip(f"{path} missing; run src/data_loader.py --sequence Urban04")
    return pd.read_parquet(path)


@pytest.fixture(scope="module")
def origin(reference: pd.DataFrame) -> Origin:
    return origin_from_reference(reference)


# A plausible origin for tests that need no data at all.
FIXED_ORIGIN = Origin(lat=44.2385078473, lon=-76.5004042101, alt=66.2978210449)


# --------------------------------------------------------------------------
# wrap_angle
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "degrees_in, degrees_out",
    [
        (0.0, 0.0),
        (90.0, 90.0),
        (179.0, 179.0),
        (181.0, -179.0),      # just past the boundary, must come back negative
        (-181.0, 179.0),
        (360.0, 0.0),         # a full turn is no turn
        (720.0, 0.0),
        (-360.0, 0.0),
    ],
)
def test_wrap_angle_known_values(degrees_in, degrees_out):
    got = np.degrees(wrap_angle(np.radians(degrees_in)))
    assert got == pytest.approx(degrees_out, abs=1e-9)


def test_wrap_angle_fixes_the_boundary_difference():
    """Truth at +179 degrees against an estimate at -179 degrees is a 2 degree
    error, not 358. Unwrapped, that value gets squared and divided by a small
    variance, and the NEES statistic becomes meaningless.
    """
    error = wrap_angle(np.radians(179.0) - np.radians(-179.0))
    assert abs(np.degrees(error)) == pytest.approx(2.0, abs=1e-9)


def test_wrap_angle_output_always_in_range():
    angles = np.radians(np.arange(-1000.0, 1000.0, 7.0))
    wrapped = wrap_angle(angles)
    assert np.all(wrapped > -np.pi - 1e-12)
    assert np.all(wrapped <= np.pi + 1e-12)


def test_wrap_angle_is_idempotent():
    """Wrapping something already wrapped must change nothing."""
    angles = np.radians(np.arange(-1000.0, 1000.0, 7.0))
    once = wrap_angle(angles)
    assert np.allclose(wrap_angle(once), once, atol=1e-12)


# --------------------------------------------------------------------------
# Angle conventions
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "azimuth_deg, yaw_deg",
    [
        (0.0, 90.0),      # north: 90 degrees counterclockwise from east
        (90.0, 0.0),      # east is the yaw origin
        (180.0, -90.0),   # south
        (45.0, 45.0),     # northeast happens to coincide
        (359.0, 91.0),    # needs wrapping to stay in range
    ],
)
def test_azimuth_to_yaw_known_directions(azimuth_deg, yaw_deg):
    got = np.degrees(azimuth_to_yaw(azimuth_deg))
    assert got == pytest.approx(yaw_deg, abs=1e-9)


def test_azimuth_to_yaw_stays_in_range():
    """Azimuth runs to 360, and 90 - 359 is well outside (-pi, pi] unwrapped."""
    yaw = azimuth_to_yaw(np.arange(0.0, 360.0, 1.0))
    assert np.all(yaw >= -np.pi - 1e-12)
    assert np.all(yaw <= np.pi + 1e-12)


@pytest.mark.parametrize("azimuth_deg", list(range(0, 360, 15)))
def test_azimuth_yaw_round_trip(azimuth_deg):
    """Regression test.

    An earlier version used ``pi - yaw`` instead of ``90 - degrees(yaw)``,
    mixing radians with degrees. It was wrong by a varying amount and nothing
    else caught it.
    """
    back = float(np.asarray(yaw_to_azimuth(azimuth_to_yaw(float(azimuth_deg)))))
    # compare modulo 360, so 0 and 360 count as equal
    assert (back - azimuth_deg + 180.0) % 360.0 - 180.0 == pytest.approx(0.0, abs=1e-9)


def test_yaw_to_azimuth_range():
    """The docstring promises [0, 360), which wrap_angle would not give."""
    azimuth = yaw_to_azimuth(np.linspace(-np.pi, np.pi, 361))
    assert np.all(azimuth >= 0.0)
    assert np.all(azimuth < 360.0)


# --------------------------------------------------------------------------
# geodetic <-> ENU, no data needed
# --------------------------------------------------------------------------

def test_origin_maps_to_zero():
    east, north, up = geodetic_to_enu(
        FIXED_ORIGIN.lat, FIXED_ORIGIN.lon, FIXED_ORIGIN.alt, FIXED_ORIGIN
    )
    assert float(east) == pytest.approx(0.0, abs=1e-9)
    assert float(north) == pytest.approx(0.0, abs=1e-9)
    assert float(up) == pytest.approx(0.0, abs=1e-9)


def test_zero_enu_maps_to_origin():
    lat, lon, alt = enu_to_geodetic(0.0, 0.0, 0.0, FIXED_ORIGIN)
    assert float(lat) == pytest.approx(FIXED_ORIGIN.lat, abs=1e-12)
    assert float(lon) == pytest.approx(FIXED_ORIGIN.lon, abs=1e-12)
    assert float(alt) == pytest.approx(FIXED_ORIGIN.alt, abs=1e-6)


def test_axes_point_the_right_way():
    """North of the origin must be +north; east of it must be +east.

    A sign error here would be invisible in a trajectory plot (the shape would
    just be mirrored) but would corrupt every heading comparison.
    """
    north_of, _, _ = geodetic_to_enu(
        FIXED_ORIGIN.lat + 0.001, FIXED_ORIGIN.lon, FIXED_ORIGIN.alt, FIXED_ORIGIN
    )
    _, n_component, _ = geodetic_to_enu(
        FIXED_ORIGIN.lat + 0.001, FIXED_ORIGIN.lon, FIXED_ORIGIN.alt, FIXED_ORIGIN
    )
    e_component, _, _ = geodetic_to_enu(
        FIXED_ORIGIN.lat, FIXED_ORIGIN.lon + 0.001, FIXED_ORIGIN.alt, FIXED_ORIGIN
    )
    up_component = geodetic_to_enu(
        FIXED_ORIGIN.lat, FIXED_ORIGIN.lon, FIXED_ORIGIN.alt + 10.0, FIXED_ORIGIN
    )[2]

    assert float(n_component) > 0.0
    assert float(e_component) > 0.0
    assert float(up_component) == pytest.approx(10.0, abs=1e-6)


def test_one_degree_of_latitude_is_about_111_km():
    """A scale check, so a unit error could not pass unnoticed."""
    _, north, _ = geodetic_to_enu(
        FIXED_ORIGIN.lat + 1.0, FIXED_ORIGIN.lon, FIXED_ORIGIN.alt, FIXED_ORIGIN
    )
    assert 110_000 < float(north) < 112_000


def test_round_trip_synthetic_grid():
    """Convert a grid of ENU points to geodetic and back."""
    east = np.array([-1000.0, -10.0, 0.0, 10.0, 1000.0])
    north = np.array([500.0, -500.0, 0.0, 250.0, -250.0])
    up = np.zeros_like(east)

    lat, lon, alt = enu_to_geodetic(east, north, up, FIXED_ORIGIN)
    e2, n2, u2 = geodetic_to_enu(lat, lon, alt, FIXED_ORIGIN)

    assert np.allclose(np.asarray(e2), east, atol=1e-6)
    assert np.allclose(np.asarray(n2), north, atol=1e-6)
    assert np.allclose(np.asarray(u2), up, atol=1e-6)


# --------------------------------------------------------------------------
# Against the real Urban04 tables
# --------------------------------------------------------------------------

def test_origin_from_reference_takes_the_first_sample(reference, origin):
    assert origin.lat == reference["lat"].iloc[0]
    assert origin.lon == reference["lon"].iloc[0]
    assert origin.alt == reference["height"].iloc[0]


def test_reference_track_starts_at_zero(reference, origin):
    out = add_enu(reference, origin)
    assert out["east"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert out["north"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_reference_track_extent_matches_the_published_route(reference, origin):
    """Urban04 is 4.87 km driven inside a bounding box of roughly 1.4 by 0.9 km."""
    out = add_enu(reference, origin)
    extent_e = out["east"].max() - out["east"].min()
    extent_n = out["north"].max() - out["north"].min()
    assert 1300 < extent_e < 1500
    assert 800 < extent_n < 1000


def test_round_trip_over_the_whole_reference_track(reference, origin):
    out = add_enu(reference, origin)
    lat, lon, alt = enu_to_geodetic(out["east"], out["north"], out["up"], origin)

    # back in degrees, converted to metres for a threshold that means something
    d_north = (np.asarray(lat) - reference["lat"].to_numpy()) * 111_320
    d_east = (
        (np.asarray(lon) - reference["lon"].to_numpy())
        * 111_320
        * np.cos(np.radians(origin.lat))
    )
    assert np.hypot(d_east, d_north).max() < 1e-3   # under a millimetre


def test_add_enu_does_not_modify_its_input(reference, origin):
    before = list(reference.columns)
    add_enu(reference, origin)
    assert list(reference.columns) == before


def test_add_enu_needs_the_right_altitude_column(gnss, origin):
    """The gnss table uses 'alt'; the reference uses 'height'.

    The asymmetry comes from the source messages: INSPVA publishes height,
    NavSatFix publishes altitude. Forgetting it should fail loudly.
    """
    with pytest.raises(KeyError):
        add_enu(gnss, origin)

    out = add_enu(gnss, origin, alt_col="alt")
    assert {"east", "north", "up"} <= set(out.columns)
    assert len(out) == len(gnss)


def test_reference_azimuth_converts_to_valid_yaw(reference):
    yaw = azimuth_to_yaw(reference["azimuth"].to_numpy())
    assert np.all(np.isfinite(yaw))
    assert np.all(np.abs(yaw) <= np.pi + 1e-9)


def test_gnss_agrees_with_reference_to_within_a_few_metres(reference, gnss, origin):
    """A consumer GNSS fix should sit metres from the truth, not hundreds.

    This is a sanity bound, not an accuracy claim. It would catch a wrong
    origin, a swapped axis or a degrees-for-radians mistake, all of which
    produce errors far outside this range.
    """
    ref_enu = add_enu(reference, origin)
    gnss_enu = add_enu(gnss, origin, alt_col="alt")

    ref_e = np.interp(gnss_enu["t"], ref_enu["t"], ref_enu["east"])
    ref_n = np.interp(gnss_enu["t"], ref_enu["t"], ref_enu["north"])
    error = np.hypot(gnss_enu["east"] - ref_e, gnss_enu["north"] - ref_n)

    assert np.median(error) < 5.0
    assert error.max() < 50.0
