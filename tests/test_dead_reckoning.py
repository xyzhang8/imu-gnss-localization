"""Tests for src/dead_reckoning.py.


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

from src.dead_reckoning import dead_reckon, estimate_gyro_bias, find_stationary_window

def make_timeline(n, dt=0.01, wz=0.0, ax=0.0):
    """A synthetic timeline: n rows, constant dt, constant wz and ax.

    One odometry reading at row 0 and none after, so the zero-order hold
    carries that speed through the whole run.
    """
    return pd.DataFrame({
        "t": np.arange(n) * dt,
        "dt": np.full(n, dt),
        "wz": np.full(n, wz),
        "ax": np.full(n, ax),
        "odometry_new": np.r_[True, np.zeros(n - 1, bool)],
        "odometry_row": np.r_[0, np.full(n - 1, -1)],
    })
    
def test_straight_line_at_constant_speed():
    timeline = make_timeline(n=2000)
    odometry = pd.DataFrame({"speed": [10.0]})

    out = dead_reckon(timeline, odometry, east0=0.0, north0=0.0,
                      yaw0=0.0, speed0=0.0, bias=0.0, speed_source="odometer")

    assert out["east"].iloc[-1] == pytest.approx(200.0, abs=0.1)
    assert out["north"].iloc[-1] == pytest.approx(0.0, abs=1e-9)
    
def test_inital_heading_sets_direction():
    timeline = make_timeline(n=2000)
    odometry = pd.DataFrame({"speed": [10.0]})
    
    out = dead_reckon(timeline, odometry, east0=0.0, north0=0.0,
                      yaw0=(np.pi)/2, speed0=0.0, bias=0.0, speed_source="odometer")
    
    assert out["east"].iloc[-1] == pytest.approx(0.0, abs=1e-9)
    assert out["north"].iloc[-1] == pytest.approx(200.0, abs=1e-9)
    
def test_zero_speed_stays_put():
    timeline = make_timeline(n=2000)
    odometry = pd.DataFrame({"speed": [0.0]})
    
    out = dead_reckon(timeline, odometry, east0=0.0, north0=0.0,
                      yaw0=(np.pi)/2, speed0=0.0, bias=0.0, speed_source="odometer")
    
    assert out["east"].iloc[-1] == pytest.approx(0.0, abs=1e-9)
    assert out["north"].iloc[-1] == pytest.approx(0.0, abs=1e-9)
    
def test_constant_yaw_rate_traces_a_circle():
    """Constant speed and constant yaw rate must trace a circle of radius v/w.

    Geometry. Starting at the origin heading east (yaw = 0) with a positive
    yaw rate, the vehicle turns left, so the centre of the circle is 90
    degrees to the left of the initial heading: at (0, R).
    """
    v, w, dt = 10.0, 0.1, 0.01
    radius = v / w                       # 100 m
    period = 2 * np.pi / w               # 62.83 s for one revolution
    n = int(round(period / dt))

    timeline = make_timeline(n + 1, dt=dt, wz=w)
    odometry = pd.DataFrame({"speed": [v]})

    out = dead_reckon(timeline, odometry, east0=0.0, north0=0.0,
                      yaw0=0.0, speed0=0.0, bias=0.0, speed_source="odometer")

    # every point sits on the circle centred at (0, radius)
    r = np.hypot(out["east"] - 0.0, out["north"] - radius)
    assert r.min() == pytest.approx(radius, rel=1e-3)
    assert r.max() == pytest.approx(radius, rel=1e-3)

    # after one full revolution it comes back to where it started
    gap = np.hypot(out["east"].iloc[-1], out["north"].iloc[-1])
    assert gap < 0.2            # against a 628 m circumference

    # and is pointing the way it started
    assert np.degrees(out["yaw"].iloc[-1]) == pytest.approx(0.0, abs=0.1)
    
def test_bias_is_subtracted():
    wz = bias = 10
    timeline = make_timeline(n=2000, wz= wz)
    odometry = pd.DataFrame({"speed": [10.0]})
    
    yaw0 = (np.pi)/2
    out = dead_reckon(timeline, odometry, east0=0.0, north0=0.0,
                      yaw0=yaw0, speed0=0.0, bias=bias, speed_source="odometer")
    
    assert out["yaw"].iloc[-1] == pytest.approx(yaw0)
    
def test_uncorrected_bias_drifts():
    """The complement to the test above: there was something to correct.

    Without this, test_bias_is_subtracted could pass for the wrong reason, for
    instance if the heading never changed under any circumstances.
    """
    wz = np.radians(0.0977)                 # the measured Urban04 bias
    timeline = make_timeline(n=2000, wz=wz)
    odometry = pd.DataFrame({"speed": [0.0]})

    yaw0 = np.pi / 2
    out = dead_reckon(timeline, odometry, east0=0.0, north0=0.0,
                      yaw0=yaw0, speed0=0.0, bias=0.0, speed_source="odometer")

    assert out["yaw"].iloc[-1] != pytest.approx(yaw0)


def test_yaw_stays_wrapped():
    """Spin for long enough to pass +/-pi many times; yaw must stay in range.

    Without wrapping, yaw would grow without bound and every comparison
    against the reference would break.
    """
    timeline = make_timeline(n=20000, wz=1.0)      # 200 s at 1 rad/s, ~32 turns
    odometry = pd.DataFrame({"speed": [0.0]})

    out = dead_reckon(timeline, odometry, east0=0.0, north0=0.0,
                      yaw0=0.0, speed0=0.0, bias=0.0, speed_source="odometer")

    assert out["yaw"].min() >= -np.pi - 1e-12
    assert out["yaw"].max() <= np.pi + 1e-12


def test_output_length_matches_input():
    """One row out per row in, so the result lines up with ref_* columns
    without any further alignment."""
    timeline = make_timeline(n=1234)
    odometry = pd.DataFrame({"speed": [10.0]})

    out = dead_reckon(timeline, odometry, east0=0.0, north0=0.0,
                      yaw0=0.0, speed0=0.0, bias=0.0, speed_source="odometer")

    assert len(out) == len(timeline)
    assert list(out.columns) == ["t", "east", "north", "yaw", "speed"]
    assert np.allclose(out["t"], timeline["t"])


def test_odometer_holds_speed_between_readings():
    """One reading at row 0 and none after: the speed must persist.

    Odometry arrives at 16.7 Hz against the IMU's 100 Hz, so roughly six steps
    pass between readings and the last value has to carry through them.
    """
    timeline = make_timeline(n=2000)               # only row 0 has a reading
    odometry = pd.DataFrame({"speed": [10.0]})

    out = dead_reckon(timeline, odometry, east0=0.0, north0=0.0,
                      yaw0=0.0, speed0=0.0, bias=0.0, speed_source="odometer")

    assert (out["speed"] == 10.0).all()


def test_modes_produce_identical_yaw():
    """Heading depends only on wz and the bias, so speed_source cannot touch it.

    This is why the IMU-only trajectory has the right shape at the wrong scale:
    every difference between the two modes is a speed difference.
    """
    timeline = make_timeline(n=2000, wz=0.05, ax=0.3)
    odometry = pd.DataFrame({"speed": [10.0]})

    kwargs = dict(east0=0.0, north0=0.0, yaw0=0.3, speed0=0.0, bias=0.001)
    odo = dead_reckon(timeline, odometry, speed_source="odometer", **kwargs)
    ins = dead_reckon(timeline, odometry, speed_source="inertial", **kwargs)

    assert np.array_equal(odo["yaw"].to_numpy(), ins["yaw"].to_numpy())
    assert not np.allclose(odo["speed"], ins["speed"])     # but the speeds differ


def test_inertial_integrates_acceleration():
    """In inertial mode the odometer is ignored entirely: constant acceleration
    from rest gives v = a*t and distance = a*t^2/2."""
    a, dt, n = 2.0, 0.01, 1000                     # 2 m/s^2 for 10 s
    timeline = make_timeline(n, dt=dt, ax=a)
    odometry = pd.DataFrame({"speed": [999.0]})    # deliberately absurd, must be unused

    out = dead_reckon(timeline, odometry, east0=0.0, north0=0.0,
                      yaw0=0.0, speed0=0.0, bias=0.0, speed_source="inertial")

    t_end = n * dt
    assert out["speed"].iloc[-1] == pytest.approx(a * t_end, rel=1e-6)
    assert out["east"].iloc[-1] == pytest.approx(0.5 * a * t_end**2, rel=1e-2)
    
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

# --------------------------------------------------------------------------
# find_stationary_window
# --------------------------------------------------------------------------

def make_odometry(speeds, dt=0.06):
    t = np.arange(len(speeds)) * dt
    return pd.DataFrame({"t": t, "speed": np.asarray(speeds, dtype=float)})


def test_stationary_window_ends_when_the_vehicle_moves():
    odometry = make_odometry([0.0, 0.0, 0.0, 5.0, 5.0])
    t_start, t_end = find_stationary_window(odometry)
    assert t_start == 0.0
    assert t_end == pytest.approx(odometry["t"].iloc[3])


def test_stationary_window_when_the_vehicle_never_moves():
    """Must return something usable rather than raising, so a caller gets a
    window rather than an UnboundLocalError."""
    odometry = make_odometry([0.0] * 10)
    t_start, t_end = find_stationary_window(odometry)
    assert t_start == 0.0
    assert t_end == pytest.approx(odometry["t"].iloc[-1])


def test_stationary_window_when_the_vehicle_moves_immediately():
    odometry = make_odometry([5.0, 5.0, 5.0])
    t_start, t_end = find_stationary_window(odometry)
    assert t_end == pytest.approx(odometry["t"].iloc[0])


def test_quantisation_step_does_not_count_as_moving():
    """The odometer is quantised to 1 km/h = 0.278 m/s. The default threshold
    of 0.1 must sit below that step, so a genuine first reading registers."""
    odometry = make_odometry([0.0, 0.0, 0.2778])
    _, t_end = find_stationary_window(odometry)
    assert t_end == pytest.approx(odometry["t"].iloc[2])


# --------------------------------------------------------------------------
# estimate_gyro_bias
# --------------------------------------------------------------------------

def make_imu(wz_values, dt=0.01):
    wz = np.asarray(wz_values, dtype=float)
    return pd.DataFrame({"t": np.arange(len(wz)) * dt, "wz": wz,
                         "wx": np.zeros(len(wz)), "wy": np.full(len(wz), 7.0)})


def test_bias_of_a_constant_signal_is_that_constant():
    imu = make_imu([0.004] * 500)
    assert estimate_gyro_bias(imu, 0.0, 5.0) == pytest.approx(0.004)


def test_noise_averages_away():
    """The point of using a long window. Bias and noise are the same size in
    Urban04, so one sample says nothing and 9,860 give a standard error near
    0.001 deg/s."""
    rng = np.random.default_rng(0)
    true_bias = 0.0017
    imu = make_imu(true_bias + rng.normal(0.0, 0.0018, 10000))
    assert estimate_gyro_bias(imu, 0.0, 100.0) == pytest.approx(true_bias, abs=1e-4)


def test_only_the_window_is_averaged():
    """Samples outside the stationary window must not contribute."""
    imu = make_imu([0.004] * 500 + [10.0] * 500)
    assert estimate_gyro_bias(imu, 0.0, 4.99) == pytest.approx(0.004)


def test_axis_parameter_selects_the_column():
    imu = make_imu([0.004] * 500)
    assert estimate_gyro_bias(imu, 0.0, 5.0, axis="wy") == pytest.approx(7.0)


# --------------------------------------------------------------------------
# Against the real Urban04 data
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def timeline(tables) -> pd.DataFrame:
    return build_timeline(tables["imu"], tables["gnss"],
                          tables["odometry"], tables["reference"])


@pytest.fixture(scope="module")
def bias(tables) -> float:
    t_start, t_end = find_stationary_window(tables["odometry"])
    return estimate_gyro_bias(tables["imu"], t_start, t_end)


def test_urban04_stationary_window(tables):
    t_start, t_end = find_stationary_window(tables["odometry"])
    assert t_start == 0.0
    assert t_end == pytest.approx(98.6, abs=0.5)


def test_urban04_bias_matches_the_measurement(bias):
    assert np.degrees(bias) == pytest.approx(0.0977, abs=0.001)


def _error(estimate, timeline):
    return np.hypot(estimate["east"] - timeline["ref_east"],
                    estimate["north"] - timeline["ref_north"])


def test_odometer_mode_stays_under_ten_metres(timeline, tables, bias):
    """A regression guard on E1's headline number, currently 5.67 m."""
    out = dead_reckon(timeline, tables["odometry"],
                      east0=timeline["ref_east"].iloc[0],
                      north0=timeline["ref_north"].iloc[0],
                      yaw0=timeline["ref_yaw"].iloc[0],
                      speed0=0.0, bias=bias, speed_source="odometer")
    window = (timeline["t"] >= 98.6) & (timeline["t"] <= 398.6)
    rmse = np.sqrt((_error(out, timeline)[window] ** 2).mean())
    assert rmse < 10.0


def test_inertial_mode_is_orders_of_magnitude_worse(timeline, tables, bias):
    """Asserts the E1 *claim*, not just that the code runs.

    Gravity leaking through 2.27 degrees of parked roll is 3,627 times the
    accelerometer's own bias, so integrating it is catastrophic. If someone
    later "fixes" inertial mode by quietly using the odometer, this fails.
    """
    kwargs = dict(east0=timeline["ref_east"].iloc[0],
                  north0=timeline["ref_north"].iloc[0],
                  yaw0=timeline["ref_yaw"].iloc[0], speed0=0.0, bias=bias)
    odo = dead_reckon(timeline, tables["odometry"], speed_source="odometer", **kwargs)
    ins = dead_reckon(timeline, tables["odometry"], speed_source="inertial", **kwargs)

    window = (timeline["t"] >= 98.6) & (timeline["t"] <= 398.6)
    rmse_odo = np.sqrt((_error(odo, timeline)[window] ** 2).mean())
    rmse_ins = np.sqrt((_error(ins, timeline)[window] ** 2).mean())
    assert rmse_ins > 100 * rmse_odo


def test_inertial_mode_invents_motion_while_parked(timeline, tables, bias):
    """At t = 98.6 s the vehicle has not moved. The odometer knows; the
    accelerometer reports 1.77 m/s and 63.7 m of travel."""
    kwargs = dict(east0=timeline["ref_east"].iloc[0],
                  north0=timeline["ref_north"].iloc[0],
                  yaw0=timeline["ref_yaw"].iloc[0], speed0=0.0, bias=bias)
    odo = dead_reckon(timeline, tables["odometry"], speed_source="odometer", **kwargs)
    ins = dead_reckon(timeline, tables["odometry"], speed_source="inertial", **kwargs)

    k = int(np.searchsorted(timeline["t"].to_numpy(), 98.6))
    assert _error(odo, timeline).iloc[k] < 1.0
    assert _error(ins, timeline).iloc[k] > 50.0


def test_real_data_modes_agree_on_heading(timeline, tables, bias):
    kwargs = dict(east0=timeline["ref_east"].iloc[0],
                  north0=timeline["ref_north"].iloc[0],
                  yaw0=timeline["ref_yaw"].iloc[0], speed0=0.0, bias=bias)
    odo = dead_reckon(timeline, tables["odometry"], speed_source="odometer", **kwargs)
    ins = dead_reckon(timeline, tables["odometry"], speed_source="inertial", **kwargs)
    assert np.array_equal(odo["yaw"].to_numpy(), ins["yaw"].to_numpy())
