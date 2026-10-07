"""Error metrics for comparing an estimate against the reference.

Unaligned error
---------------
Position error is reported **without fitting a transform first**. Standard ATE
begins by finding the rigid transform that best aligns the estimate to the
truth, which absorbs global drift and can understate absolute error
substantially. These estimators output globally referenced positions, so they
should be judged that way. Aligned figures may appear alongside, labelled.

Long format
-----------
:func:`summarise` returns one row per ``(run_id, metric, value)`` rather than a
wide table or a dict. Three reasons:

1. It is the shape the planned DuckDB ``metrics`` table wants, so the Day 8
   retrofit is an ingest rather than a rewrite.
2. Comparing runs becomes a ``GROUP BY`` instead of a pile of files.
3. A new metric adds rows, not columns, so nothing downstream needs changing.

Angles are not positions
------------------------
Heading error must be wrapped before it is squared. Truth at +179 degrees
against an estimate at -179 degrees is a 2 degree error, not 358, and the
unwrapped value is what destroys E6's NEES. Every angular difference here goes
through ``frames.wrap_angle``.

What is deliberately not here yet
---------------------------------
NEES and NIS need a covariance, which dead reckoning does not produce. They
arrive with the EKF on Day 4 and are E6's subject. The drift-rate and
per-outage metrics arrive with ``outages.py`` on Day 9.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

from src.frames import wrap_angle


def horizontal_error(
    estimate: pd.DataFrame,
    timeline: pd.DataFrame,
) -> np.ndarray:
    """Euclidean distance between estimated and true position, per sample.

    No alignment, no transform fitting: the straight distance between where the
    estimator thinks it is and where it was.

    Parameters
    ----------
    estimate : pd.DataFrame
        Estimator output with ``east`` and ``north`` in the local ENU frame.
    timeline : pd.DataFrame
        From ``align.build_timeline`` with a reference, so it carries
        ``ref_east`` and ``ref_north``. Must be the same length as
        ``estimate``, which holds by construction since both have one row per
        IMU sample.

    Returns
    -------
    np.ndarray
        Horizontal error in metres, one value per sample.

    Raises
    ------
    KeyError
        If the reference columns are absent, which means ``build_timeline`` was
        called without a reference.
    ValueError
        If the two tables differ in length, since that means something upstream
        resampled and the comparison would be silently misaligned.
    """
    if len(estimate) != len(timeline):
        raise ValueError(
            f"estimate has {len(estimate)} rows, timeline has {len(timeline)}; "
            "one row per IMU sample is expected in both"
        )
    e1 = estimate["east"].to_numpy() - timeline["ref_east"].to_numpy()
    e2 = estimate["north"].to_numpy() - timeline["ref_north"].to_numpy()
    return np.sqrt(e1 ** 2 + e2 ** 2)


def heading_error(
    estimate: pd.DataFrame,
    timeline: pd.DataFrame,
) -> np.ndarray:
    """Signed heading error, wrapped onto (-pi, pi].

    Parameters
    ----------
    estimate : pd.DataFrame
        Estimator output with ``yaw`` in radians.
    timeline : pd.DataFrame
        Carrying ``ref_yaw`` in radians.

    Returns
    -------
    np.ndarray
        Heading error in **radians**, one value per sample. Signed, so the sign
        says which way the estimate is rotated. Convert for display; keeping
        radians here avoids a unit guess at every call site.

    Notes
    -----
    Wrapped, for the reason in the module docstring. On Urban04 the reference
    azimuth crosses the boundary at t = 185.98 s and t = 767.40 s, so an
    unwrapped difference produces two spurious 360 degree spikes.
    """
    if len(estimate) != len(timeline):
        raise ValueError(
            f"estimate has {len(estimate)} rows, timeline has {len(timeline)}; "
            "one row per IMU sample is expected in both"
        )
    yaw_est = estimate["yaw"].to_numpy()
    yaw_gt = timeline["ref_yaw"].to_numpy()
    return wrap_angle(yaw_est - yaw_gt)


def summarise(
    estimate: pd.DataFrame,
    timeline: pd.DataFrame,
    run_id: str,
    window: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Reduce one run to a long-format table of scalar metrics.

    Parameters
    ----------
    estimate : pd.DataFrame
        Estimator output.
    timeline : pd.DataFrame
        Timeline with reference columns.
    run_id : str
        Identifies this run, for example ``"dr_odometer"`` or ``"ekf_outage60"``.
        Becomes the ``run_id`` column, and later the foreign key into the
        DuckDB ``runs`` table.
    window : tuple of float, or None
        ``(t_from, t_to)`` in seconds to restrict the metrics to. Needed for the
        comparison against the published NavINST baseline, which is quoted over
        **5-minute scenarios** rather than whole trajectories. None uses the
        whole run.

    Returns
    -------
    pd.DataFrame
        Columns ``run_id``, ``metric``, ``value``, ``unit``. One row per metric:

        - ``horizontal_rmse``, ``horizontal_mean``, ``horizontal_max``,
          ``horizontal_final`` in metres
        - ``heading_rmse``, ``heading_max``, ``heading_final`` in degrees
        - ``duration`` in seconds and ``samples`` as a count, so a row can be
          interpreted without going back to the source

    Notes
    -----
    RMSE rather than mean, as the headline, because squaring penalises the
    large excursions that matter for a localisation system: being 50 m out once
    is worse than being 5 m out ten times.

    ``horizontal_final`` is worth reporting separately because for dead
    reckoning the error is a random walk with structure, not a monotonic climb.
    On Urban04 it falls from 5.0 m to 0.3 m between t = 200 and 280 s, since a
    heading error that pushes left on one leg pushes right on the return leg of
    a loop. A single final number would therefore mislead on its own.
    """
    e = horizontal_error(estimate, timeline)
    e_psi = heading_error(estimate, timeline)
    t = timeline["t"].to_numpy()

    if window is not None:
        t_from, t_to = window
        keep = (t >= t_from) & (t <= t_to)
        e, e_psi, t = e[keep], e_psi[keep], t[keep]
    
    if len(e) == 0:
        raise ValueError(f"window {window} selects no samples")
    
    horizontal_rmse = np.sqrt(((1 / len(e)) * np.sum(e**2)))
    horizontal_mean = np.mean(e)
    horizontal_max = np.max(e)
    horizontal_final = e[-1]
    
    heading_rmse = np.sqrt(((1 / len(e)) * np.sum(e_psi**2)))
    heading_max = np.max(np.abs(e_psi)) #e_psi is signed so absolute value is needed
    heading_final = e_psi[-1]
    rows = [
        ("horizontal_rmse", horizontal_rmse, "m"),
        ("horizontal_mean", horizontal_mean, "m"),
        ("horizontal_max", horizontal_max, "m"),
        ("horizontal_final", horizontal_final, "m"),
        ("heading_rmse", np.degrees(heading_rmse), "deg"),
        ("heading_max", np.degrees(heading_max), "deg"),
        ("heading_final", np.degrees(heading_final), "deg"),
        ("duration", t[-1] - t[0], "s"),
        ("samples", len(e), "count"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "unit"]).assign(
        run_id=run_id
    )[["run_id", "metric", "value", "unit"]]


def to_csv(metrics: pd.DataFrame, path: str) -> None:
    """Append long-format metrics to a CSV, creating it with a header if absent.

    A stopgap until the DuckDB ``metrics`` table exists on Day 8. Appending
    rather than overwriting means several runs accumulate in one file, which is
    the same shape the database will hold.

    Parameters
    ----------
    metrics : pd.DataFrame
        From :func:`summarise`, or several concatenated.
    path : str
        Destination, conventionally under ``results/``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(path, mode="a", header=not path.exists(), index=False)
    
