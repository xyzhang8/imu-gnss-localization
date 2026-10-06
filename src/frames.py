"""Coordinate frames: geodetic to local Cartesian, and angle conventions.

Local ENU frame
---------------
GNSS and reference positions arrive as WGS84 latitude and longitude in degrees.
Everything downstream needs metres: the error metrics, the covariances, and the
filter state ``[east, north, heading, speed, gyro_bias]``, whose motion model
adds ``v * dt * cos(heading)`` to east and so only makes sense on a plane.

So both are converted once into a local ENU frame: a plane tangent to the
ellipsoid at a fixed origin, x east, y north, z up, in metres. UTM is the other
option but has zone discontinuities and a varying scale factor; a tangent plane
has neither.

The conversion is exact, not an approximation. A round trip over all 53,251
Urban04 reference samples stays within a few billionths of a millimetre. What a
tangent plane discards is curvature, and that lands entirely in ``up``: the
Earth falls away from the plane by 3.8 cm at 700 m, 15 cm at 1.4 km and 7.8 m
at 10 km. The model is 2D and ignores ``up``, so it costs nothing here, but it
is why the approach would not survive a 3D state spanning tens of kilometres.

**The origin must be identical across every stream and every run that will be
compared.** Changing it shifts all coordinates and silently invalidates any
comparison against earlier results.

Angle conventions
-----------------
Two incompatible conventions meet in this project:

- **Azimuth**, from ``INSPVA.azimuth``: degrees, clockwise from north.
  North is 0, east is 90.
- **Yaw**, the filter state: radians, counterclockwise from east, so that
  ``cos(yaw)`` is the east component of a unit heading vector.

They differ by a rotation and a sign, which is converted here once rather than
at each call site.

Deferred: the lever arm
-----------------------
The Xsens GNSS antenna and the NovAtel reference antenna sit at different points
on the roof, about 1.2 m apart as measured on Urban04. The offset is fixed in
the vehicle body frame, so in ENU it rotates with heading and is not a constant
shift. See ``apply_lever_arm``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pymap3d import geodetic2enu, enu2geodetic

ArrayLike = np.ndarray | pd.Series | float


@dataclass(frozen=True)
class Origin:
    """The point where the local ENU plane touches the ellipsoid.

    Frozen: an origin that changed mid-run would silently shift every
    coordinate computed after it.

    Attributes
    ----------
    lat, lon : float
        Geodetic latitude and longitude of the origin, degrees, WGS84.
    alt : float
        Height of the origin above the ellipsoid, metres.
    """

    lat: float
    lon: float
    alt: float


def origin_from_reference(reference: pd.DataFrame) -> Origin:
    """Take the ENU origin from the first sample of a ground-truth table.

    The reference rather than the GNSS stream, because it is the most accurate
    position available and the frame every error is measured in.

    Parameters
    ----------
    reference : pd.DataFrame
        The ``reference`` table from ``data_loader``, sorted by time, with
        columns ``lat``, ``lon`` and ``height``.

    Returns
    -------
    Origin
        The position of the first row.
    """
    return Origin(
        lat = reference["lat"].iloc[0],
        lon = reference["lon"].iloc[0],
        alt = reference["height"].iloc[0],
    )


def geodetic_to_enu(
    lat: ArrayLike,
    lon: ArrayLike,
    alt: ArrayLike,
    origin: Origin,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert WGS84 geodetic coordinates to local ENU metres.

    Wraps ``pymap3d.geodetic2enu`` so the origin travels as one object and
    ``lat0`` and ``lon0`` cannot be transposed at a call site.

    Parameters
    ----------
    lat, lon : array_like
        Geodetic latitude and longitude in **degrees**, WGS84. Scalars, or
        arrays of equal length.
    alt : array_like
        Height above the ellipsoid, metres.
    origin : Origin
        Tangent point of the local plane.

    Returns
    -------
    east, north, up : np.ndarray
        Position in metres relative to ``origin``. ``up`` is returned for
        sanity checks; the 2D model ignores it.
    """
    east, north, up = geodetic2enu(lat, lon, alt, origin.lat, origin.lon, origin.alt)
    return np.asarray(east), np.asarray(north), np.asarray(up)


def enu_to_geodetic(
    east: ArrayLike,
    north: ArrayLike,
    up: ArrayLike,
    origin: Origin,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert local ENU metres back to WGS84 geodetic coordinates.

    Inverse of :func:`geodetic_to_enu`, for putting a trajectory on a map or
    exporting it for another tool.

    Parameters
    ----------
    east, north, up : array_like
        Position in metres relative to ``origin``.
    origin : Origin
        **The same origin used for the forward conversion.** A different one
        produces a silently shifted trajectory.

    Returns
    -------
    lat, lon, alt : np.ndarray
        Latitude and longitude in degrees, height in metres.
    """
    lat, lon, alt = enu2geodetic(east, north, up, origin.lat, origin.lon, origin.alt)
    return np.asarray(lat), np.asarray(lon), np.asarray(alt)


def add_enu(
    df: pd.DataFrame,
    origin: Origin,
    lat_col: str = "lat",
    lon_col: str = "lon",
    alt_col: str = "height",
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``east``, ``north`` and ``up`` added.

    Parameters
    ----------
    df : pd.DataFrame
        A table with geodetic position columns.
    origin : Origin
        Shared origin for the sequence.
    lat_col, lon_col, alt_col : str
        Column names. The defaults match the ``reference`` table; the ``gnss``
        table names its height column ``alt``, so pass ``alt_col="alt"``.

    Returns
    -------
    pd.DataFrame
        A **copy** with three columns appended, so that a mistaken origin
        cannot corrupt a table already in memory.
    """
    east, north, up = geodetic_to_enu(
        df[lat_col], df[lon_col], df[alt_col], origin
    )
    out = df.copy()
    out["east"] = np.asarray(east)
    out["north"] = np.asarray(north)
    out["up"] = np.asarray(up)
    return out


def azimuth_to_yaw(azimuth_deg: ArrayLike) -> np.ndarray:
    """Convert NovAtel azimuth to the filter's yaw convention.

    ``yaw = wrap(radians(90 - azimuth))``. North (azimuth 0) gives +pi/2, east
    (azimuth 90) gives 0. The wrap matters because azimuth runs to 360.

    Parameters
    ----------
    azimuth_deg : array_like
        Azimuth in degrees clockwise from north, as published in
        ``INSPVA.azimuth``.

    Returns
    -------
    np.ndarray
        Yaw in radians counterclockwise from east, wrapped to (-pi, pi].
    """
    yaw_deg = 90 - azimuth_deg
    yaw_rad = np.radians(yaw_deg)
    wrapped_yaw = wrap_angle(yaw_rad)
    return wrapped_yaw


def yaw_to_azimuth(yaw_rad: ArrayLike) -> np.ndarray:
    """Convert the filter's yaw back to azimuth degrees.

    Inverse of :func:`azimuth_to_yaw`. Uses ``% 360`` rather than
    :func:`wrap_angle`, because a compass bearing is unsigned whereas
    ``wrap_angle`` returns a signed difference.

    Parameters
    ----------
    yaw_rad : array_like
        Yaw in radians counterclockwise from east.

    Returns
    -------
    np.ndarray
        Azimuth in degrees clockwise from north, in [0, 360).
    """
    azimuth_deg = 90 - np.degrees(yaw_rad)
    wrapped_azimuth  = azimuth_deg % 360
    return wrapped_azimuth


def wrap_angle(angle_rad: ArrayLike) -> np.ndarray:
    """Wrap an angle, or an angular difference, into (-pi, pi].

    **Every angular subtraction in this project must pass through here.**
    Heading lives on a circle, so the naive difference between +179 and -179
    degrees is 358 degrees rather than 2. Unwrapped, that value is squared and
    divided by a small variance in the NEES statistic, which destroys it.

    Also needed after the prediction step, so heading does not grow without
    bound, and in the UKF sigma-point mean, where points straddling +/-pi
    cannot be averaged with a plain weighted sum.

    Parameters
    ----------
    angle_rad : array_like
        An angle, or an angular difference, in radians.

    Returns
    -------
    np.ndarray
        The equivalent angle in (-pi, pi].
    """
    y = np.sin(angle_rad)
    x = np.cos(angle_rad)
    wrapped_angle = np.arctan2(y,x)
    return wrapped_angle


def apply_lever_arm(
    east: ArrayLike,
    north: ArrayLike,
    yaw_rad: ArrayLike,
    offset_body: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Shift an ENU position by a body-frame lever arm. **Not implemented.**

    Because the offset is fixed in the body frame it rotates with ``yaw_rad``,
    so subtracting a constant ENU offset would be wrong: over a drive with
    varied headings it partly averages itself away and leaves a
    heading-dependent residual.

    Parameters
    ----------
    east, north : array_like
        Position in the local ENU frame, metres.
    yaw_rad : array_like
        Vehicle heading at each sample, radians counterclockwise from east.
    offset_body : tuple of float
        ``(forward, left)`` offset in metres, in the vehicle body frame.

    Returns
    -------
    east, north : np.ndarray
        Position shifted to the other reference point.

    Notes
    -----
    Deferred because it needs a heading estimate, so it cannot run before dead
    reckoning exists.

    The dataset's published calibration does **not** give the offset needed
    here. ``setup_transformations_1.txt`` lists ``KVH -> XSENS`` (the IMU unit)
    and ``KVH -> PWRPAK7_ANTENNA``, which difference to 0.38 m, but the
    MTi-670G uses a separate external GNSS antenna with no entry in the file.

    Fitted from Urban04 instead, by rotating the GNSS error into the body
    frame over the samples above 2 m/s: **1.34 m backward, 0.20 m left**.
    Checked against speed to rule out a pure timestamp lag, which would grow
    with speed where a lever arm does not; the fit is ``-0.053 * speed - 1.34``,
    so mostly geometry plus a minor 53 ms timing component.
    """
    raise NotImplementedError
