"""Open-loop integration of the IMU. The baseline every filter must beat.

Dead reckoning is the Kalman filter's **prediction step run alone**, with no
measurement updates and therefore no feedback. The same model reappears in
``ekf.py``, so writing it here first means debugging the motion model without
covariances in the way. If it is wrong, every later estimator is wrong in the
same way.

The model
---------
State ``[east, north, yaw, speed]``, input ``[wz, ax]`` from the IMU::

    yaw   <- wrap(yaw + (wz - bias) * dt)
    speed <- speed + ax * dt            (inertial mode only)
    east  <- east  + speed * cos(yaw) * dt
    north <- north + speed * sin(yaw) * dt

Kinematic, not dynamic: no masses, forces or tyre models. The IMU measures the
output of the vehicle's dynamics directly, so there is nothing left to predict.
The price is that the model inherits every sensor error, with nothing to
correct it.

Four states, not the filter's five. Gyro bias is estimated once from the
stationary period and held fixed, because with no measurements nothing could
observe it changing. It becomes a state in the EKF, where GNSS can observe it.

Two modes, and why both exist
-----------------------------
``"inertial"`` integrates ``ax`` for speed, and is the honest answer to E1's
"how bad is the problem". It is very bad: the accelerometer is dominated by
gravity leaking through vehicle tilt, 3,627 times its own in-run bias. On
Urban04 it reports 1.77 m/s for a car that has not moved, and 63.7 m of travel,
before the vehicle sets off at all.

``"odometer"`` reads speed from the wheel sensor and integrates only the gyro.
Measured on Urban04: **5.7 m RMSE over five minutes against 2,250 m**, a factor
of 400. The odometer figure is comparable to the published NavINST baseline of
13.72 m ATE, with the caveat that the metrics differ.

Heading is identical in both modes, since ``yaw`` depends only on ``wz`` and the
bias. Every difference between the two trajectories is therefore a speed
difference, which is why the inertial track has the right shape at ten times the
scale.

Conventions
-----------
``yaw`` is radians counterclockwise from east, matching ``frames.azimuth_to_yaw``.
``ax`` is the forward axis, established by correlating it against a smoothed
``d(speed)/dt`` at +0.784 against -0.181 for ``ay``. The IMU is mounted x
forward, y lateral, z up.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from src.frames import wrap_angle

SpeedSource = Literal["inertial", "odometer"]


def find_stationary_window(
    odometry: pd.DataFrame,
    speed_threshold: float = 0.1,
) -> tuple[float, float]:
    """Find the stationary period at the start of a run.

    Every NavINST trajectory begins and ends with roughly two minutes parked.
    Detected from the odometer rather than the IMU: wheel speed reads exactly
    zero when stationary, whereas the IMU always shows noise.

    Only the leading window is returned. A bias estimated from the trailing one
    would be contaminated by whatever drifted during the run, confounding
    turn-on bias with bias instability.

    Parameters
    ----------
    odometry : pd.DataFrame
        Odometry table with ``t`` and ``speed`` in m/s.
    speed_threshold : float
        Speed above which the vehicle counts as moving. The odometer is
        quantised to 1 km/h = 0.278 m/s, so any value strictly inside that step
        works.

    Returns
    -------
    t_start, t_end : float
        Bounds of the initial stationary window, seconds. ``t_start`` is 0.
        If the vehicle never moves, ``t_end`` is the end of the recording.

    Notes
    -----
    Urban04 gives 0 to 98.6 s, and is stationary again for the last 125.3 s.
    """
    t_start = 0.0
    t_end = odometry["t"].iloc[-1]          # fallback: the vehicle never moves
    for i in range(len(odometry)):
        if odometry["speed"].iloc[i] > speed_threshold:
            t_end = odometry["t"].iloc[i]
            break
    return t_start, t_end


def estimate_gyro_bias(
    imu: pd.DataFrame,
    t_start: float,
    t_end: float,
    axis: str = "wz",
) -> float:
    """Average the gyro over a stationary window to get its turn-on bias.

    While genuinely still the true angular rate is zero, so whatever the gyro
    reports is bias plus noise. Averaging kills the noise and leaves the bias:

    .. math:: \\hat{b} = \\frac{1}{M}\\sum_{k=1}^{M} \\tilde{\\omega}_k

    Parameters
    ----------
    imu : pd.DataFrame
        IMU table with ``t`` and the named axis.
    t_start, t_end : float
        Stationary window in seconds, from :func:`find_stationary_window`.
    axis : str
        Gyro column to average. Only ``wz`` matters for a planar model.

    Returns
    -------
    float
        Bias in rad/s, to subtract from every subsequent sample.

    Notes
    -----
    Urban04 gives +0.0977 deg/s on the yaw axis from 9,860 samples. The Xsens
    datasheet quotes no turn-on bias, so this measured value is the one to use.

    The bias and the noise are the same size here: sample standard deviation is
    0.1012 deg/s against a 0.0977 deg/s mean. One sample says nothing, while
    the average of 9,860 has a standard error near 0.001 deg/s. That is why the
    two-minute window is generous rather than wasteful.
    """
    still = imu[(imu.t >= t_start) & (imu.t <= t_end)]
    values = still[axis].to_numpy()
    M = len(values)

    total = 0.0
    for k in range(M):
        total += values[k]

    return (1 / M) * total


def dead_reckon(
    timeline: pd.DataFrame,
    odometry: pd.DataFrame,
    east0: float,
    north0: float,
    yaw0: float,
    speed0: float,
    bias: float,
    speed_source: SpeedSource = "odometer",
) -> pd.DataFrame:
    """Integrate the IMU forward, open loop, with no corrections.

    One prediction step per IMU sample. Nothing updates the state, which is the
    point: the growth of the error against the reference is E1's result.

    Parameters
    ----------
    timeline : pd.DataFrame
        From ``align.build_timeline``: one row per IMU sample with ``t``,
        ``dt``, ``wz``, ``ax``, ``odometry_new`` and ``odometry_row``.
    odometry : pd.DataFrame
        Odometry table with ``speed`` in m/s, indexed as ``odometry_row``
        refers to it.
    east0, north0 : float
        Starting position in the local ENU frame, metres.
    yaw0 : float
        Starting heading, radians counterclockwise from east.
    speed0 : float
        Starting forward speed, m/s. Zero if the run begins stationary.
    bias : float
        Yaw-axis turn-on bias in rad/s, from :func:`estimate_gyro_bias`.
    speed_source : {"inertial", "odometer"}
        ``"inertial"`` integrates ``ax``. ``"odometer"`` takes speed from the
        wheel sensor whenever a reading arrives and holds it between readings.

    Returns
    -------
    pd.DataFrame
        One row per input row, with ``t``, ``east``, ``north``, ``yaw`` and
        ``speed``. Same length as ``timeline``, so it compares against ``ref_*``
        columns without further alignment.

    Notes
    -----
    Euler integration, deliberately. Over a 10 ms step the vehicle moves at most
    12.5 cm and turns at most 0.27 degrees, so a higher-order scheme would buy
    far less than the sensor errors cost. Using the same first-order step the
    EKF will use also keeps the two comparable.

    Heading is updated before position, so the step advances along the new
    heading. ``yaw`` is wrapped every step, both to stop it growing without
    bound and so that errors against the reference are computed on the circle.

    In ``"odometer"`` mode the speed is a zero-order hold: the most recent
    reading persists until the next arrives, up to 60 ms later. A real
    approximation, small against everything else here, and one for the
    limitations section.

    The initial pose is expected to come from the reference, which is the one
    place an estimator reads truth. Heading cannot come from GNSS, since a
    parked vehicle has no course and differencing two fixes 1.2 m apart over
    0.25 s gives nonsense, and the magnetometer is excluded. Starting from a
    known pose also means E1 measures drift rather than initialisation error.
    State this in the README.
    """
    speeds: list[float] = []
    easts: list[float] = []
    norths: list[float] = []
    yaws: list[float] = []

    speed, yaw, east, north = speed0, yaw0, east0, north0

    wz = timeline["wz"].to_numpy()
    ax = timeline["ax"].to_numpy()
    dt = timeline["dt"].to_numpy()
    odo_new = timeline["odometry_new"].to_numpy()
    odo_row = timeline["odometry_row"].to_numpy()
    odo_speed = odometry["speed"].to_numpy()

    for k in range(len(timeline)):
        # heading: integrate the bias-corrected yaw rate, then wrap onto the circle
        yaw = float(wrap_angle(yaw + (wz[k] - bias) * dt[k]))

        # speed: integrate the accelerometer, or read the wheel sensor
        if speed_source == "inertial":
            speed = speed + ax[k] * dt[k]
        elif odo_new[k]:
            speed = odo_speed[odo_row[k]]
        # no else: with no new reading, the previous speed is held

        # position: advance along the new heading
        east = east + speed * np.cos(yaw) * dt[k]
        north = north + speed * np.sin(yaw) * dt[k]

        speeds.append(speed)
        yaws.append(yaw)
        easts.append(east)
        norths.append(north)

    return pd.DataFrame({
        "t": timeline["t"].to_numpy(),
        "east": easts,
        "north": norths,
        "yaw": yaws,
        "speed": speeds,
    })
