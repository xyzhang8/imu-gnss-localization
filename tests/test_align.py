"""Tests for src/align.py.

Same split as test_frames.py. Pure tests build tiny synthetic streams in the
test itself, so the exact expected answer can be written down by hand and the
suite runs without any data on disk. Data tests check against the real Urban04
tables and skip when the Parquet files are absent.

The property worth protecting above all others: **no measurement is invented,
lost, or used twice.** A filter cannot tell a fabricated measurement from a
real one, so a bug here would show up only as a filter that is mysteriously
overconfident.

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

from src.align import (  # noqa: E402
    build_timeline,
    measurement_events,
    reference_at,
)
from src.frames import (  # noqa: E402
    add_enu,
    azimuth_to_yaw,
    origin_from_reference,
    wrap_angle,
)

PROCESSED = REPO / "data" / "processed" / "Urban04"


# --------------------------------------------------------------------------
# Synthetic fixtures, so the expected answer can be written by hand
# --------------------------------------------------------------------------

def make_imu(times: list[float]) -> pd.DataFrame:
    t = np.asarray(times, dtype=float)
    return pd.DataFrame({"t": t, "t_ns": (t * 1e9).astype(np.int64), "wz": np.zeros_like(t)})


def make_stream(times: list[float], values: list[float] | None = None) -> pd.DataFrame:
    t = np.asarray(times, dtype=float)
    v = np.arange(len(t), dtype=float) if values is None else np.asarray(values, float)
    return pd.DataFrame({"t": t, "value": v})


# --------------------------------------------------------------------------
# measurement_events: the counting logic
# --------------------------------------------------------------------------

def test_one_measurement_per_interval_is_marked_once():
    imu = make_imu([0.0, 0.1, 0.2, 0.3])
    meas = make_stream([0.15])          # falls inside (0.1, 0.2]
    out = measurement_events(imu, {"m": meas})

    assert out["m_new"].tolist() == [False, False, True, False]
    assert out.loc[2, "m_row"] == 0
    assert (out.loc[out["m_new"] == False, "m_row"] == -1).all()


def test_measurement_exactly_on_an_imu_sample_belongs_to_that_sample():
    """The interval is half-open: (t[k-1], t[k]]. A reading at exactly t[k]
    is new at k, not at k+1."""
    imu = make_imu([0.0, 0.1, 0.2])
    out = measurement_events(imu, {"m": make_stream([0.1])})
    assert out["m_new"].tolist() == [False, True, False]


def test_first_sample_claims_everything_before_it():
    """A reading slightly preceding the IMU stream must not be silently lost."""
    imu = make_imu([1.0, 1.1, 1.2])
    out = measurement_events(imu, {"m": make_stream([0.5])})
    assert out["m_new"].tolist() == [True, False, False]
    assert out.loc[0, "m_row"] == 0


def test_two_readings_in_one_interval_keeps_the_most_recent():
    """Documented behaviour, not an accident. Urban05's GNSS is irregular
    enough that this can happen."""
    imu = make_imu([0.0, 1.0])
    out = measurement_events(imu, {"m": make_stream([0.2, 0.8])})
    assert out["m_new"].tolist() == [False, True]
    assert out.loc[1, "m_row"] == 1          # the later of the two


def test_no_measurements_at_all():
    imu = make_imu([0.0, 0.1, 0.2])
    out = measurement_events(imu, {"m": make_stream([5.0, 6.0])})
    assert not out["m_new"].any()
    assert (out["m_row"] == -1).all()


def test_streams_are_independent():
    """Two streams can both fire on the same row."""
    imu = make_imu([0.0, 0.1, 0.2])
    out = measurement_events(
        imu, {"a": make_stream([0.15]), "b": make_stream([0.15])}
    )
    assert out.loc[2, "a_new"] and out.loc[2, "b_new"]


def test_dt_comes_from_t_ns():
    imu = make_imu([0.0, 0.01, 0.02, 0.03])
    out = measurement_events(imu, {})
    assert np.allclose(out["dt"].to_numpy()[1:], 0.01)
    # the first row has no predecessor and borrows the median
    assert out["dt"].iloc[0] == pytest.approx(0.01)


def test_unsorted_input_raises_rather_than_being_sorted():
    """Sorting internally would make the returned row indices refer to an
    order the caller cannot see."""
    imu = make_imu([0.0, 0.1, 0.2])
    with pytest.raises(ValueError):
        measurement_events(imu.iloc[::-1], {"m": make_stream([0.15])})
    with pytest.raises(ValueError):
        measurement_events(imu, {"m": make_stream([0.15, 0.05])})


def test_missing_columns_raise_keyerror():
    imu = make_imu([0.0, 0.1])
    with pytest.raises(KeyError):
        measurement_events(imu.drop(columns="t_ns"), {"m": make_stream([0.05])})
    with pytest.raises(KeyError):
        measurement_events(imu, {"m": pd.DataFrame({"time": [0.05]})})


# --------------------------------------------------------------------------
# reference_at: interpolation
# --------------------------------------------------------------------------

def make_reference(times: list[float], east: list[float], yaw: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"t": times, "east": east, "north": east, "yaw": yaw})


def test_exact_times_return_exact_values():
    ref = make_reference([0.0, 1.0, 2.0], [10.0, 20.0, 40.0], [0.0, 0.1, 0.2])
    out = reference_at(np.array([0.0, 1.0, 2.0]), ref, columns=("east",))
    assert out["east"].tolist() == [10.0, 20.0, 40.0]


def test_midpoint_is_linear():
    ref = make_reference([0.0, 1.0], [10.0, 20.0], [0.0, 0.0])
    out = reference_at(np.array([0.25, 0.5]), ref, columns=("east",))
    assert out["east"].tolist() == [12.5, 15.0]


def test_angle_interpolation_crosses_the_boundary_correctly():
    """The whole reason angle columns are handled separately.

    Halfway between +179 and -179 degrees is 180, not 0. Urban04 crosses this
    twice, at t = 185.98 s and t = 767.40 s.
    """
    ref = make_reference([0.0, 1.0], [0.0, 0.0], list(np.radians([179.0, -179.0])))
    out = reference_at(np.array([0.5]), ref, columns=(), angle_columns=("yaw",))
    got = abs(np.degrees(out["yaw"].iloc[0]))
    assert got == pytest.approx(180.0, abs=1e-9)


def test_angle_output_is_wrapped():
    ref = make_reference([0.0, 1.0], [0.0, 0.0], list(np.radians([170.0, 190.0])))
    out = reference_at(np.linspace(0, 1, 11), ref, columns=(), angle_columns=("yaw",))
    assert np.all(np.abs(out["yaw"]) <= np.pi + 1e-12)


def test_times_outside_the_reference_span_raise():
    """np.interp clamps silently, which would freeze the reference and look
    like a filter problem rather than a data problem."""
    ref = make_reference([0.0, 1.0], [10.0, 20.0], [0.0, 0.0])
    with pytest.raises(ValueError):
        reference_at(np.array([1.5]), ref, columns=("east",))
    with pytest.raises(ValueError):
        reference_at(np.array([-0.5]), ref, columns=("east",))


def test_degrees_passed_as_radians_raise():
    """A bounds check, not a conversion. Without it the error is a silent
    factor of 57."""
    ref = make_reference([0.0, 1.0], [0.0, 0.0], [0.0, 179.0])
    with pytest.raises(ValueError):
        reference_at(np.array([0.5]), ref, columns=(), angle_columns=("yaw",))


def test_missing_reference_column_raises_keyerror():
    ref = make_reference([0.0, 1.0], [10.0, 20.0], [0.0, 0.0])
    with pytest.raises(KeyError):
        reference_at(np.array([0.5]), ref, columns=("nonexistent",))


# --------------------------------------------------------------------------
# Against the real Urban04 tables
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tables() -> dict[str, pd.DataFrame]:
    needed = ["imu", "gnss", "odometry", "reference"]
    if not all((PROCESSED / f"{n}.parquet").exists() for n in needed):
        pytest.skip("Urban04 Parquet missing; run src/data_loader.py --sequence Urban04")

    out = {n: pd.read_parquet(PROCESSED / f"{n}.parquet") for n in needed}
    origin = origin_from_reference(out["reference"])
    out["reference"] = add_enu(out["reference"], origin)
    out["reference"]["yaw"] = azimuth_to_yaw(out["reference"]["azimuth"])
    out["gnss"] = add_enu(out["gnss"], origin, alt_col="alt")
    return out


@pytest.fixture(scope="module")
def timeline(tables) -> pd.DataFrame:
    return build_timeline(
        tables["imu"], tables["gnss"], tables["odometry"], tables["reference"]
    )


def test_timeline_has_one_row_per_imu_sample(timeline, tables):
    assert len(timeline) == len(tables["imu"])


def test_every_measurement_is_used_exactly_once(timeline, tables):
    """The property that matters most. A lost fix is missing information; a
    duplicated one is fabricated evidence, and the filter cannot tell."""
    for name in ("gnss", "odometry"):
        used = sorted(timeline.loc[timeline[f"{name}_new"], f"{name}_row"])
        assert used == list(range(len(tables[name]))), f"{name} rows not used exactly once"


def test_no_measurement_rows_are_marked_without_an_index(timeline):
    for name in ("gnss", "odometry"):
        assert (timeline.loc[timeline[f"{name}_new"], f"{name}_row"] >= 0).all()
        assert (timeline.loc[~timeline[f"{name}_new"], f"{name}_row"] == -1).all()


def test_most_rows_have_no_gnss(timeline):
    """GNSS at 4 Hz against an IMU at 100 Hz: about 4% of rows, no more.

    If this ever rises toward 100% someone has resampled the measurements,
    which would silently make the filter overconfident.
    """
    fraction = timeline["gnss_new"].mean()
    assert 0.03 < fraction < 0.05


def test_dt_is_the_imu_interval(timeline):
    assert np.allclose(timeline["dt"].to_numpy(), 0.01, atol=1e-6)


def test_reference_columns_are_prefixed_and_complete(timeline):
    for col in ("ref_east", "ref_north", "ref_yaw"):
        assert col in timeline.columns
        assert timeline[col].notna().all()


def test_reference_yaw_is_wrapped(timeline):
    assert np.all(np.abs(timeline["ref_yaw"].to_numpy()) <= np.pi + 1e-9)


def test_no_truth_columns_without_a_reference(tables):
    """Ground truth must be impossible to reach by accident from a filter."""
    bare = build_timeline(tables["imu"], tables["gnss"], tables["odometry"])
    assert not [c for c in bare.columns if c.startswith("ref_")]


def test_interpolated_reference_matches_at_real_sample_times(tables):
    """Where an IMU time coincides with a reference time, interpolation must
    return the reference value rather than a blend."""
    ref = tables["reference"]
    probe = ref["t"].to_numpy()[1000:1010]
    got = reference_at(probe, ref)
    assert np.allclose(got["east"], ref["east"].to_numpy()[1000:1010])
    assert np.allclose(
        wrap_angle(got["yaw"].to_numpy() - ref["yaw"].to_numpy()[1000:1010]), 0, atol=1e-9
    )


def test_reference_is_smooth_across_the_known_wrap(tables):
    """Urban04's azimuth wraps at t = 185.98 s. Interpolated yaw must step by
    a fraction of a degree there, not by 180."""
    out = reference_at(np.linspace(185.9, 186.1, 201), tables["reference"])
    steps = np.abs(np.degrees(wrap_angle(np.diff(out["yaw"].to_numpy()))))
    assert steps.max() < 1.0
