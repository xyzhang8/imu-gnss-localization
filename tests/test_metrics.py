"""Tests for src/metrics.py.

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
PROCESSED = REPO / "data" / "processed" / "Urban04"

from src.align import build_timeline, reference_at  # noqa: E402
from src.dead_reckoning import (  # noqa: E402
    dead_reckon,
    estimate_gyro_bias,
    find_stationary_window,
)
from src.frames import (  # noqa: E402
    add_enu,
    azimuth_to_yaw,
    origin_from_reference,
)
from src.metrics import (  # noqa: E402
    heading_error,
    horizontal_error,
    summarise,
    to_csv,
)

METRIC_NAMES = [
    "horizontal_rmse",
    "horizontal_mean",
    "horizontal_max",
    "horizontal_final",
    "heading_rmse",
    "heading_max",
    "heading_final",
    "duration",
    "samples",
]


def make_pair(east, north, yaw=0.0, ref_east=0.0, ref_north=0.0, ref_yaw=0.0, dt=0.01):
    """Build a matching (estimate, timeline) pair for a handful of samples.

    Any argument may be a scalar, which is broadcast to every row, or a
    sequence. The length is taken from ``east``. Both tables get the same ``t``
    column, so a window can be expressed in seconds.

    Parameters
    ----------
    east, north, yaw : float or sequence
        The estimate. ``yaw`` is in radians.
    ref_east, ref_north, ref_yaw : float or sequence
        The reference columns of the timeline, same units.
    dt : float
        Spacing of the shared time column, seconds.

    Returns
    -------
    estimate, timeline : pd.DataFrame
        Of equal length, with the columns the metrics functions read.
    """
    east = np.atleast_1d(np.asarray(east, dtype=float))
    n = len(east)

    def col(value):
        return np.full(n, float(value)) if np.isscalar(value) else np.asarray(value, dtype=float)

    t = np.arange(n) * dt
    estimate = pd.DataFrame({
        "t": t,
        "east": east,
        "north": col(north),
        "yaw": col(yaw),
        "speed": np.zeros(n),
    })
    timeline = pd.DataFrame({
        "t": t,
        "ref_east": col(ref_east),
        "ref_north": col(ref_north),
        "ref_yaw": col(ref_yaw),
    })
    return estimate, timeline


def metric_value(table, name):
    """Pull one value out of a long-format table from :func:`summarise`."""
    return float(table.loc[table["metric"] == name, "value"].iloc[0])


@pytest.fixture(scope="module")
def urban04():
    """Both dead-reckoning runs on Urban04, with the timeline they share.

    Skips when the Parquet is absent, so the suite still passes on a fresh
    clone. Module-scoped: the loading and the two integrations happen once for
    every test that asks for it.

    Returns
    -------
    dict
        ``timeline``, ``odometer``, ``inertial`` and ``t_move``, the time the
        vehicle starts moving, which is where the 5-minute window begins.
    """
    needed = ["imu", "gnss", "odometry", "reference"]
    if not all((PROCESSED / f"{name}.parquet").exists() for name in needed):
        pytest.skip("Urban04 Parquet missing; run src/data_loader.py --sequence Urban04")

    tables = {name: pd.read_parquet(PROCESSED / f"{name}.parquet") for name in needed}
    origin = origin_from_reference(tables["reference"])
    reference = add_enu(tables["reference"], origin)
    reference["yaw"] = azimuth_to_yaw(reference["azimuth"])
    gnss = add_enu(tables["gnss"], origin, alt_col="alt")

    timeline = build_timeline(tables["imu"], gnss, tables["odometry"], reference=reference)
    _, t_move = find_stationary_window(tables["odometry"])
    bias = estimate_gyro_bias(tables["imu"], 0.0, t_move)

    start = reference_at(np.array([timeline["t"].iloc[0]]), reference)
    pose = dict(
        east0=float(start["east"][0]),
        north0=float(start["north"][0]),
        yaw0=float(start["yaw"][0]),
        speed0=0.0,
        bias=bias,
    )
    return {
        "timeline": timeline,
        "odometer": dead_reckon(timeline, tables["odometry"], speed_source="odometer", **pose),
        "inertial": dead_reckon(timeline, tables["odometry"], speed_source="inertial", **pose),
        "t_move": t_move,
    }


# --------------------------------------------------------------------------
# horizontal_error
# --------------------------------------------------------------------------

def test_known_triangle_gives_hypotenuse():
    """3 m east and 4 m north off the reference is exactly 5 m of error.

    The only test of the formula itself; everything else below tests
    bookkeeping around it.
    """
    estimate, timeline = make_pair(east=3, north=4)
    assert horizontal_error(estimate, timeline) == pytest.approx(5.0)


def test_zero_when_estimate_matches_reference():
    """Identical positions give exactly zero, with no floating-point residue."""
    estimate, timeline = make_pair(east=10, north=-5, ref_east=10, ref_north=-5)
    assert (horizontal_error(estimate, timeline) == 0.0).all()


def test_error_is_never_negative():
    """It is a distance, so no input can produce a negative value."""
    estimate, timeline = make_pair(
        east=[-4.0, 0.0, 7.5, -1.0],
        north=[3.0, -2.0, 0.0, -9.0],
        ref_east=2.0,
        ref_north=-1.0,
    )
    assert (horizontal_error(estimate, timeline) >= 0.0).all()


def test_direction_does_not_change_magnitude():
    """3 m east and 3 m west are both 3 m of error.

    Confirms the sign really is discarded, which is what makes this metric
    independent of how the local frame happens to be oriented.
    """
    estimate, timeline = make_pair(
        east=[13.0, 7.0, 10.0, 10.0],
        north=[-5.0, -5.0, -2.0, -8.0],
        ref_east=10.0,
        ref_north=-5.0,
    )

    error = horizontal_error(estimate, timeline)

    assert error == pytest.approx([3.0, 3.0, 3.0, 3.0])


def test_output_length_matches_input():
    """One value per sample, not a single number.

    Regression test: an earlier version reduced both columns with ``.all()``
    and collapsed 106,500 positions into one boolean.
    """
    estimate, timeline = make_pair(east=np.arange(7.0), north=0.0)

    error = horizontal_error(estimate, timeline)

    assert isinstance(error, np.ndarray)
    assert error.shape == (7,)


def test_horizontal_length_mismatch_raises():
    """Different lengths mean something upstream resampled, so the comparison
    would be silently misaligned. Must raise rather than broadcast."""
    estimate, timeline = make_pair(east=np.zeros(5), north=0.0)

    with pytest.raises(ValueError):
        horizontal_error(estimate, timeline.iloc[:3])


def test_missing_reference_columns_raises():
    """A timeline built without a reference has no ``ref_east``, and asking for
    error against truth that was never loaded must fail loudly."""
    estimate, timeline = make_pair(east=np.zeros(5), north=0.0)
    without_truth = timeline[["t"]]

    with pytest.raises(KeyError):
        horizontal_error(estimate, without_truth)


# --------------------------------------------------------------------------
# heading_error
# --------------------------------------------------------------------------

def test_heading_zero_when_headings_match():
    """Baseline, and it also catches a stray offset in the wrapping."""
    estimate, timeline = make_pair(east=0.0, north=0.0, yaw=0.7, ref_yaw=0.7)
    assert (heading_error(estimate, timeline) == 0.0).all()


def test_wraps_across_the_boundary():
    """Truth at +179 degrees against an estimate at -179 degrees is 2 degrees.

    The test this function exists for. An unwrapped difference gives 358, which
    survives squaring and destroys any RMSE or NEES it enters. Urban04 crosses
    the boundary at t = 185.98 s and t = 767.40 s, so it is not hypothetical.
    """
    estimate, timeline = make_pair(
        east=0.0,
        north=0.0,
        yaw=np.radians(-179.0),
        ref_yaw=np.radians(179.0),
    )

    error = np.degrees(heading_error(estimate, timeline))

    assert error == pytest.approx(2.0)


def test_sign_says_which_way():
    """An estimate rotated counterclockwise of truth gives a positive error.

    Yaw is counterclockwise from east. The sign is kept because a gyro bias
    produces a monotonic error of one sign, which is how it is recognised.
    """
    ahead, _ = make_pair(east=0.0, north=0.0, yaw=0.4)
    behind, timeline = make_pair(east=0.0, north=0.0, yaw=0.2)
    timeline["ref_yaw"] = 0.3

    assert heading_error(ahead, timeline) == pytest.approx(0.1)
    assert heading_error(behind, timeline) == pytest.approx(-0.1)


def test_result_stays_within_pi():
    """Over inputs spanning several turns, every output lies in (-pi, pi]."""
    estimate, timeline = make_pair(
        east=np.zeros(101),
        north=0.0,
        yaw=np.linspace(-10.0, 10.0, 101),
        ref_yaw=0.3,
    )

    error = heading_error(estimate, timeline)

    assert (error > -np.pi - 1e-12).all()
    assert (error <= np.pi + 1e-12).all()


def test_returns_radians_not_degrees():
    """A 90 degree difference comes back as about 1.5708."""
    estimate, timeline = make_pair(east=0.0, north=0.0, yaw=np.pi / 2, ref_yaw=0.0)
    assert heading_error(estimate, timeline) == pytest.approx(1.5707963, abs=1e-6)


def test_heading_length_mismatch_raises():
    """Same guard as horizontal_error."""
    estimate, timeline = make_pair(east=np.zeros(5), north=0.0)

    with pytest.raises(ValueError):
        heading_error(estimate, timeline.iloc[:3])


# --------------------------------------------------------------------------
# summarise
# --------------------------------------------------------------------------

def test_returns_long_format_columns():
    """Exactly run_id, metric, value, unit.

    This is the contract the DuckDB ``metrics`` table is built against on
    Day 8, so a change here is a schema change.
    """
    estimate, timeline = make_pair(east=np.arange(5.0), north=0.0)

    table = summarise(estimate, timeline, run_id="run_a")

    assert list(table.columns) == ["run_id", "metric", "value", "unit"]


def test_one_row_per_metric():
    """Nine rows, carrying the nine metric names the docstring promises."""
    estimate, timeline = make_pair(east=np.arange(5.0), north=0.0)

    table = summarise(estimate, timeline, run_id="run_a")

    assert len(table) == len(METRIC_NAMES)
    assert list(table["metric"]) == METRIC_NAMES


def test_run_id_is_carried_through():
    """Every row carries it; it becomes the foreign key into ``runs``."""
    estimate, timeline = make_pair(east=np.arange(5.0), north=0.0)

    table = summarise(estimate, timeline, run_id="dr_odometer")

    assert (table["run_id"] == "dr_odometer").all()


def test_rmse_is_root_not_mean_square():
    """Build errors where the two differ clearly and check the root is taken.

    Errors of 0, 0, 0, 10 give a mean square of 25 and an RMSE of 5. Without
    ``np.sqrt`` the value is a mean square labelled metres: it stays plausible,
    shrinking errors below 1 m and growing those above, so nothing looks
    obviously wrong while every comparison is distorted.
    """
    estimate, timeline = make_pair(east=[0.0, 0.0, 0.0, 10.0], north=0.0)

    table = summarise(estimate, timeline, run_id="run_a")

    assert metric_value(table, "horizontal_rmse") == pytest.approx(5.0)


def test_rmse_is_at_least_mean():
    """True for any non-negative data, so it holds whatever input you choose."""
    estimate, timeline = make_pair(east=[1.0, 4.0, 9.0, 2.0, 7.0], north=3.0)

    table = summarise(estimate, timeline, run_id="run_a")

    assert metric_value(table, "horizontal_rmse") >= metric_value(table, "horizontal_mean")


def test_heading_max_is_positive_for_negative_errors():
    """An all-negative heading error must report a positive maximum.

    Regression test: a plain ``max`` over signed values returns the sample
    closest to zero, reporting the best moment of the run as the worst.
    """
    estimate, timeline = make_pair(
        east=np.zeros(3),
        north=0.0,
        yaw=[-0.1, -0.2, -0.3],
        ref_yaw=0.0,
    )

    table = summarise(estimate, timeline, run_id="run_a")

    assert metric_value(table, "heading_max") == pytest.approx(np.degrees(0.3))


def test_heading_rows_are_in_degrees():
    """A known error in radians appears converted in the table."""
    estimate, timeline = make_pair(east=0.0, north=0.0, yaw=0.5, ref_yaw=0.0)

    table = summarise(estimate, timeline, run_id="run_a")

    assert metric_value(table, "heading_rmse") == pytest.approx(np.degrees(0.5))
    assert metric_value(table, "heading_final") == pytest.approx(np.degrees(0.5))
    assert (table.loc[table["metric"].str.startswith("heading"), "unit"] == "deg").all()


def test_window_restricts_sample_count():
    """``samples`` counts the rows inside the window, not the whole run."""
    estimate, timeline = make_pair(east=np.zeros(100), north=0.0)   # dt = 0.01 s

    table = summarise(estimate, timeline, run_id="run_a", window=(0.0, 0.495))

    assert metric_value(table, "samples") == 50


def test_window_changes_final():
    """``horizontal_final`` is the last sample in the window.

    The error is 9 m at row 2 and 1 m everywhere else, so a version that slices
    the metrics but not the final value gives 1 instead of 9. The subtlest test
    here.
    """
    estimate, timeline = make_pair(east=[1.0, 1.0, 9.0, 1.0, 1.0, 1.0], north=0.0)

    whole = summarise(estimate, timeline, run_id="run_a")
    windowed = summarise(estimate, timeline, run_id="run_a", window=(0.0, 0.025))

    assert metric_value(whole, "horizontal_final") == pytest.approx(1.0)
    assert metric_value(windowed, "horizontal_final") == pytest.approx(9.0)


def test_no_window_uses_whole_run():
    """The default is unchanged by the windowing code."""
    estimate, timeline = make_pair(east=np.arange(10.0), north=0.0)

    default = summarise(estimate, timeline, run_id="run_a")
    everything = summarise(estimate, timeline, run_id="run_a", window=(-1.0, 1.0))

    assert default["value"].to_numpy() == pytest.approx(everything["value"].to_numpy())


def test_empty_window_raises():
    """A window selecting no samples must raise, not return a table of nan."""
    estimate, timeline = make_pair(east=np.zeros(10), north=0.0)

    with pytest.raises(ValueError):
        summarise(estimate, timeline, run_id="run_a", window=(100.0, 200.0))


def test_duration_matches_window_span():
    """Catches a ``t`` that was not sliced alongside the error arrays."""
    estimate, timeline = make_pair(east=np.zeros(100), north=0.0)   # dt = 0.01 s

    table = summarise(estimate, timeline, run_id="run_a", window=(0.0, 0.495))

    assert metric_value(table, "duration") == pytest.approx(0.49)


# --------------------------------------------------------------------------
# to_csv
# --------------------------------------------------------------------------

def test_creates_file_with_header(tmp_path):
    """First write gives a header plus one row per metric.

    ``tmp_path`` is a pytest fixture: ask for it by name and you get a fresh
    empty directory, removed afterwards. Nothing is written into the repository.
    """
    estimate, timeline = make_pair(east=np.arange(5.0), north=0.0)
    path = tmp_path / "metrics.csv"

    to_csv(summarise(estimate, timeline, run_id="run_a"), str(path))

    lines = path.read_text().splitlines()
    assert lines[0] == "run_id,metric,value,unit"
    assert len(lines) == 1 + len(METRIC_NAMES)


def test_append_does_not_repeat_header(tmp_path):
    """Two writes give one header and eighteen rows, not two headers.

    Without ``header=not path.exists()`` the column names land in the middle of
    the data and are read back as a row.
    """
    estimate, timeline = make_pair(east=np.arange(5.0), north=0.0)
    path = tmp_path / "metrics.csv"

    to_csv(summarise(estimate, timeline, run_id="run_a"), str(path))
    to_csv(summarise(estimate, timeline, run_id="run_b"), str(path))

    text = path.read_text()
    assert text.count("run_id,metric,value,unit") == 1
    assert len(text.splitlines()) == 1 + 2 * len(METRIC_NAMES)


def test_no_index_column(tmp_path):
    """Reading back gives four columns. The row numbers are an artifact of how
    the frame was built and would become a spurious column in DuckDB."""
    estimate, timeline = make_pair(east=np.arange(5.0), north=0.0)
    path = tmp_path / "metrics.csv"

    to_csv(summarise(estimate, timeline, run_id="run_a"), str(path))

    assert list(pd.read_csv(path).columns) == ["run_id", "metric", "value", "unit"]


def test_creates_missing_parent_directory(tmp_path):
    """``results/`` does not exist on a fresh clone and to_csv will not make it."""
    estimate, timeline = make_pair(east=np.arange(5.0), north=0.0)
    path = tmp_path / "results" / "metrics.csv"

    to_csv(summarise(estimate, timeline, run_id="run_a"), str(path))

    assert path.exists()


def test_round_trip_preserves_values(tmp_path):
    """Write, read back, and the values match the frame that went in."""
    estimate, timeline = make_pair(east=np.arange(5.0), north=2.0)
    path = tmp_path / "metrics.csv"
    table = summarise(estimate, timeline, run_id="run_a")

    to_csv(table, str(path))
    back = pd.read_csv(path)

    assert list(back["metric"]) == list(table["metric"])
    assert back["value"].to_numpy() == pytest.approx(table["value"].to_numpy())


# --------------------------------------------------------------------------
# Data tests: skipped when the Urban04 Parquet is absent
# --------------------------------------------------------------------------

def test_urban04_odometer_rmse_is_about_five_point_seven(urban04):
    """The measured E1 number, over 5 minutes from the moment the car moves.

    End to end through the loader, frames, align, dead reckoning and metrics,
    so it fails if any of the five breaks.
    """
    window = (urban04["t_move"], urban04["t_move"] + 300.0)
    table = summarise(urban04["odometer"], urban04["timeline"], "dr_odometer", window)

    assert metric_value(table, "horizontal_rmse") == pytest.approx(5.67, abs=0.05)


def test_heading_metrics_identical_between_modes(urban04):
    """Heading depends only on wz and the bias, so both modes must agree.

    This is why the inertial trajectory has the right shape at the wrong scale:
    every difference between the two is a speed difference. Breaks loudly if
    speed ever leaks into the heading integration.
    """
    window = (urban04["t_move"], urban04["t_move"] + 300.0)
    odo = summarise(urban04["odometer"], urban04["timeline"], "dr_odometer", window)
    ins = summarise(urban04["inertial"], urban04["timeline"], "dr_inertial", window)

    for name in ("heading_rmse", "heading_max", "heading_final"):
        assert metric_value(odo, name) == metric_value(ins, name)

    assert metric_value(odo, "horizontal_rmse") < metric_value(ins, "horizontal_rmse") / 100
