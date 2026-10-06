"""Put multi-rate sensor streams onto one timeline for filtering and evaluation.

No two streams share a timestamp: across 106,500 IMU samples and 53,251
reference samples in Urban04 there is not one exact match. The rates are 100 Hz
(IMU), 50 Hz (reference), 16.7 Hz (odometry) and 4 Hz (GNSS).

The IMU is the clock
--------------------
The filter predicts on every IMU sample and updates only when a measurement
arrived, so the IMU sample index is the loop variable. ``dt`` is computed from
``t_ns`` rather than assumed, so a sequence with jitter fails loudly.

Two operations, which must never be confused
--------------------------------------------
**Measurements are located, never interpolated.** For each IMU step, the only
question is whether a reading arrived since the previous step. Usually nothing
did, and those empty steps are the physical situation the filter exists to
handle, not missing data.

Resampling GNSS from 4 Hz to 100 Hz would fabricate 96% of the rows. The filter
cannot tell an invented measurement from a real one, so it would treat 25 copies
of one fix as 25 independent observations and shrink $P$ about five times faster
than the evidence justifies.

**The reference is interpolated, never located.** It never enters the filter; it
is the answer key, read at whatever instant an estimate exists. The nearest
reference sample is a median of 5.0 ms away, 1.6 cm at the local speed, against
drift errors of metres.

Heading must be unwrapped before interpolating and wrapped after. Urban04's
azimuth crosses the boundary at t = 185.98 s and t = 767.40 s, where the number
moves 360 degrees while the vehicle turns by a tenth of one, and interpolating
straight across points the vehicle backwards.

Conventions are the caller's job
--------------------------------
This module knows about **time**, not about frames, units or angle conventions.
Callers convert first::

    ref = frames.add_enu(reference, origin)
    ref["yaw"] = frames.azimuth_to_yaw(ref["azimuth"])

Keeping those two lines at the call site means only one module has an opinion
about what a number means, so the two cannot quietly disagree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.frames import wrap_angle


def reference_at(
    times: np.ndarray,
    reference: pd.DataFrame,
    columns: tuple[str, ...] = ("east", "north"),
    angle_columns: tuple[str, ...] = ("yaw",),
) -> pd.DataFrame:
    """Interpolate the ground-truth table onto arbitrary times.

    Linear for ordinary columns. Columns in ``angle_columns`` are unwrapped
    first and wrapped afterwards.

    Parameters
    ----------
    times : np.ndarray
        Times to evaluate the reference at, in seconds on the sequence's
        shared origin, i.e. the ``t`` column produced by ``data_loader``.
        Usually the IMU's ``t``.
    reference : pd.DataFrame
        Ground truth, sorted by time, with a ``t`` column plus every column
        named in ``columns`` and ``angle_columns``, **already in the units the
        caller wants back**. Nothing is converted here.
    columns : tuple of str
        Columns to interpolate linearly.
    angle_columns : tuple of str
        Columns in radians, to unwrap before interpolating and wrap after.

    Returns
    -------
    pd.DataFrame
        One row per entry in ``times``, with a ``t`` column and one column per
        requested name. Index 0 to len(times) - 1.

    Raises
    ------
    KeyError
        If a requested column is absent from ``reference``.
    ValueError
        If ``times`` extends beyond the reference's span. ``np.interp`` clamps
        silently, which would freeze the reference rather than fail.
    ValueError
        If an angle column exceeds 2*pi in magnitude, which means degrees were
        passed where radians were meant. A bounds check, not a conversion.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim == 0:
        times = np.atleast_1d(times)

    if "t" not in reference.columns:
        raise KeyError("reference is missing required 't' column")

    ref = reference.sort_values("t").copy()
    t_ref = ref["t"].to_numpy(dtype=float)

    if len(t_ref) == 0:
        raise ValueError("reference is empty")

    requested = tuple(columns) + tuple(angle_columns)
    missing = [name for name in requested if name not in ref.columns]
    if missing:
        raise KeyError(f"reference missing required columns: {missing}")

    # Fail fast if the requested times are outside the reference span.
    # np.interp silently clamps, which would freeze the reference instead of erroring.
    if np.any(times < t_ref[0]) or np.any(times > t_ref[-1]):
        raise ValueError(
            "requested interpolation times are outside the reference time span"
        )

    # Angle columns are expected to be in radians. This catches the common
    # 'passed degrees, not radians' bug before interpolation.
    for name in angle_columns:
        vals = ref[name].to_numpy(dtype=float)
        if np.any(np.abs(vals) > 2 * np.pi):
            raise ValueError(
                f"Angle column '{name}' exceeds 2*pi in magnitude; "
                "expected radians, not degrees."
            )

    out = pd.DataFrame({"t": times})

    # Ordinary columns: simple linear interpolation.
    for name in columns:
        out[name] = np.interp(times, t_ref, ref[name].to_numpy(dtype=float))

    # Angle columns: unwrap before interpolation, wrap after.
    for name in angle_columns:
        arr = ref[name].to_numpy(dtype=float)
        unwrapped = np.unwrap(arr)
        interpolated = np.interp(times, t_ref, unwrapped)
        wrapped = wrap_angle(interpolated)
        out[name] = wrapped

    return out


def measurement_events(
    imu: pd.DataFrame,
    measurements: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Mark which measurements arrived in each IMU interval.

    For every IMU sample ``k``, records whether each stream produced a reading
    in the half-open interval ``(t[k-1], t[k]]``. Nothing is invented. The
    first sample owns everything at or before it, so a reading slightly
    preceding the IMU stream is not lost. If two readings land in one interval
    the most recent wins; that cannot happen at Urban04's rates but Urban05's
    GNSS is irregular.

    Inputs must already be sorted by time, which ``data_loader`` guarantees.
    Sorting here would make the returned row indices refer to a reordering the
    caller cannot see.

    Parameters
    ----------
    imu : pd.DataFrame
        The IMU table, sorted, with ``t`` and ``t_ns``.
    measurements : dict of str to pd.DataFrame
        Measurement streams keyed by name, each sorted with a ``t`` column,
        for example ``{"gnss": gnss_df, "odometry": odo_df}``.

    Returns
    -------
    pd.DataFrame
        One row per IMU sample, with ``t``, ``t_ns``, ``dt`` (seconds since
        the previous sample; the first row borrows the median), and per stream
        ``<name>_new`` (bool) and ``<name>_row`` (index into that stream, or
        -1). Indices rather than values, so this function needs to know
        nothing about each stream's contents. The ``_new`` flags are
        independent, so several can be True on one row.

    Raises
    ------
    KeyError
        If ``imu`` lacks ``t`` or ``t_ns``, or a stream lacks ``t``.
    ValueError
        If any input is not sorted by time.
    """
    for col in ("t", "t_ns"):
        if col not in imu.columns:
            raise KeyError(f"imu is missing required '{col}' column")

    imu_t = imu["t"].to_numpy(float)
    t_ns = imu["t_ns"].to_numpy(np.int64)
    if np.any(np.diff(t_ns) < 0):
        raise ValueError("imu is not sorted by time")

    # dt from t_ns, not assumed. The first sample has no predecessor, so it
    # borrows the median rather than inventing a zero-length step.
    dt = np.diff(t_ns, prepend=t_ns[0]) / 1e9
    dt[0] = float(np.median(dt[1:])) if len(dt) > 1 else 0.0

    out = pd.DataFrame({"t": imu_t, "t_ns": t_ns, "dt": dt})

    for name, stream in measurements.items():
        if "t" not in stream.columns:
            raise KeyError(f"measurement stream '{name}' is missing required 't' column")

        meas_t = stream["t"].to_numpy(float)
        if np.any(np.diff(meas_t) < 0):
            raise ValueError(f"measurement stream '{name}' is not sorted by time")

        # Running count of readings at or before each IMU sample. Where that
        # count goes up, a reading landed inside the interval since the previous
        # sample. prepend=0 gives the first sample everything at or before it.
        count = np.searchsorted(meas_t, imu_t, side="right")
        arrived = np.diff(count, prepend=0) > 0

        out[f"{name}_new"] = arrived
        out[f"{name}_row"] = np.where(arrived, count - 1, -1)

    return out


def build_timeline(
    imu: pd.DataFrame,
    gnss: pd.DataFrame,
    odometry: pd.DataFrame,
    reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble the single table that the estimators and metrics both consume.

    Wraps :func:`measurement_events` together with the IMU's own columns and,
    optionally, :func:`reference_at`. One row per IMU sample.

    Parameters
    ----------
    imu : pd.DataFrame
        IMU table with ``t``, ``t_ns`` and the rate and acceleration columns.
    gnss : pd.DataFrame
        GNSS table, already through ``frames.add_enu`` with ``alt_col="alt"``.
    odometry : pd.DataFrame
        Odometry table with ``speed`` in m/s.
    reference : pd.DataFrame or None
        Ground truth, **already converted by the caller** to ENU with a ``yaw``
        column. When given, truth columns are interpolated onto every row and
        prefixed ``ref_``. When None the timeline carries inputs only, so that
        nothing downstream can reach truth by accident.

    Returns
    -------
    pd.DataFrame
        One row per IMU sample: the IMU columns, ``dt``, the ``<name>_new`` and
        ``<name>_row`` columns, and ``ref_*`` when a reference was supplied.

    Notes
    -----
    One wide table rather than a bespoke object: inspectable in a notebook,
    writes to Parquet unchanged, and maps onto the DuckDB ASOF join planned for
    Day 8, which should be verifiable against this implementation row for row.
    """
    imu = imu.reset_index(drop=True)
    events = measurement_events(imu, {"gnss": gnss, "odometry": odometry})

    out = pd.concat([imu, events.drop(columns=["t", "t_ns"])], axis=1)

    if reference is not None:
        truth = reference_at(out["t"].to_numpy(float), reference)
        out = out.join(truth.drop(columns="t").add_prefix("ref_"))

    return out
