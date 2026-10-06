"""Read NavINST ROS1 bags into one Parquet table per sensor stream.

Raw values only. Resampling, time alignment and filtering belong downstream.

Two exclusions, which are the integrity claim of the whole project:

1. NovAtel and KVH hardware is ground truth, never a filter input.
2. The Xsens MTi-670G runs its own internal GNSS/INS estimator. Its fused
   outputs (``/xsens/filter/*`` and the ``orientation`` field inside
   ``/xsens/imu``) are not loaded, since using them would mean the sensor
   fusion had already been done elsewhere.

Usage:
    python src/data_loader.py --sequence Urban04
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
PROCESSED = REPO / "data" / "processed"

# Short name -> the timestamped run directory inside data/raw.
SEQUENCES = {
    "Urban03": Path("Urban03") / "2023-11-03-09-42-41-Day",
    "Urban04": Path("Urban04") / "2023-11-01-12-16-16-Day",
    "Urban05": Path("Urban05") / "2024-06-04-21-14-50-Day",
}


def _stamp_ns(msg) -> int:
    """Sensor timestamp in integer nanoseconds.

    Uses ``header.stamp``, not the bag record time, which lags by a variable
    driver latency. Kept as int64: float64 cannot hold nanosecond resolution
    at epoch magnitudes.

    Parameters
    ----------
    msg : Any
        Any decoded ROS message carrying a ``std_msgs/Header``.

    Returns
    -------
    int
        Nanoseconds since the Unix epoch.
    """
    return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec


def _imu(msg) -> dict:
    """Raw gyroscope and accelerometer from a ``sensor_msgs/Imu``.

    ``msg.orientation`` is populated but is the Xsens's own fused attitude, so
    it is dropped.

    Parameters
    ----------
    msg : sensor_msgs/Imu
        A decoded message from ``/xsens/imu``.

    Returns
    -------
    dict
        ``t_ns``, angular rates ``wx, wy, wz`` in rad/s, and specific force
        ``ax, ay, az`` in m/s^2.
    """
    w, a = msg.angular_velocity, msg.linear_acceleration
    return {
        "t_ns": _stamp_ns(msg),
        "wx": w.x, "wy": w.y, "wz": w.z,          # rad/s
        "ax": a.x, "ay": a.y, "az": a.z,          # m/s^2
    }


def _gnss(msg) -> dict:
    """Raw fix plus the quality fields that let R vary per measurement.

    Parameters
    ----------
    msg : sensor_msgs/NavSatFix
        A decoded message from ``/xsens/gnss``.

    Returns
    -------
    dict
        ``t_ns``, geodetic position, the ENU diagonal of the reported
        covariance in m^2, and the fix status fields.
    """
    cov = np.asarray(msg.position_covariance)     # row-major 3x3, m^2, ENU
    return {
        "t_ns": _stamp_ns(msg),
        "lat": msg.latitude,                       # deg, WGS84
        "lon": msg.longitude,                      # deg, WGS84
        "alt": msg.altitude,                       # m
        "var_e": cov[0], "var_n": cov[4], "var_u": cov[8],   # m^2
        "cov_type": int(msg.position_covariance_type),       # 2 = diagonal known
        "status": int(msg.status.status),          # -1 no fix, 0 fix, 1 SBAS, 2 GBAS
        "service": int(msg.status.service),
    }


def _odometry(msg) -> dict:
    """Forward speed in m/s.

    The message reports km/h and declares it in ``unit``, which is checked
    rather than assumed.

    Parameters
    ----------
    msg : navinst_obd_driver/ObdMeasurmentStamped
        A decoded message from ``/obd2/speed``.

    Returns
    -------
    dict
        ``t_ns`` and ``speed`` in m/s.

    Raises
    ------
    ValueError
        If the message declares any unit other than ``kph``.
    """
    if msg.unit != "kph":
        raise ValueError(f"expected OBD speed in kph, got {msg.unit!r}")
    return {"t_ns": _stamp_ns(msg), "speed": msg.data / 3.6}


def _reference(msg) -> dict:
    """Ground truth pose. Never a filter input.

    Parameters
    ----------
    msg : navinst_ros/INSPVA
        A decoded message from the post-processed NovAtel reference.

    Returns
    -------
    dict
        ``t_ns``, geodetic position, ENU velocity in m/s, attitude in degrees
        with ``azimuth`` clockwise from north, and the solution status.
    """
    return {
        "t_ns": _stamp_ns(msg),
        "lat": msg.latitude,
        "lon": msg.longitude,
        "height": msg.height,
        "v_e": msg.east_velocity,                  # m/s, fixed ENU frame
        "v_n": msg.north_velocity,
        "v_u": msg.up_velocity,
        "roll": msg.roll,                          # deg
        "pitch": msg.pitch,                        # deg
        "azimuth": msg.azimuth,                    # deg, clockwise from north
        "status": int(msg.status.status),          # 3 = INS_SOLUTION_GOOD
    }


def _reference_stdev(msg) -> dict:
    """Per-epoch uncertainty of the ground truth itself, at 1 Hz.

    About 1 cm in position, which is what justifies treating the reference as
    exact when filter errors are metres.

    Parameters
    ----------
    msg : navinst_ros/INSPVAX
        A decoded message from the extended reference topic.

    Returns
    -------
    dict
        ``t_ns``, per-axis standard deviations, and the position and INS
        status codes.
    """
    return {
        "t_ns": _stamp_ns(msg),
        "lat_stdev": msg.latitude_stdev,
        "lon_stdev": msg.longitude_stdev,
        "height_stdev": msg.height_stdev,
        "azimuth_stdev": msg.azimuth_stdev,
        "roll_stdev": msg.roll_stdev,
        "pitch_stdev": msg.pitch_stdev,
        "pos_type": int(msg.pos_type.type),
        "ins_status": int(msg.ins_status.status),
    }


# stream name -> (bag file, topic, extractor)
STREAMS = {
    "imu":       ("imus.bag",      "/xsens/imu",                                _imu),
    "gnss":      ("gnss.bag",      "/xsens/gnss",                               _gnss),
    "odometry":  ("imus.bag",      "/obd2/speed",                               _odometry),
    "reference": ("reference.bag", "/novatel/reference/postprocessed/inspva",   _reference),
    "reference_stdev": (
        "reference.bag", "/novatel/reference/postprocessed/inspvax", _reference_stdev,
    ),
}


def read_stream(run_dir: Path, bag: str, topic: str, extract) -> pd.DataFrame:
    """Decode every message on one topic into a DataFrame.

    Custom NavINST types need no registration: ROS1 bags embed their own
    message definitions.

    Parameters
    ----------
    run_dir : Path
        The timestamped run directory inside ``data/raw``.
    bag : str
        Bag file name, for example ``"imus.bag"``.
    topic : str
        Topic to read, for example ``"/xsens/imu"``.
    extract : callable
        One of the ``_imu``-style extractors: takes a decoded message, returns
        a dict of one row.

    Returns
    -------
    pd.DataFrame
        One row per message, sorted by ``t_ns``, index reset.

    Raises
    ------
    FileNotFoundError
        If the bag does not exist.
    KeyError
        If the topic is absent from the bag.
    """
    path = run_dir / bag
    if not path.exists():
        raise FileNotFoundError(path)

    rows = []
    with AnyReader([path]) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        if not conns:
            raise KeyError(f"{topic} not found in {path.name}")
        for conn, _, raw in reader.messages(connections=conns):
            rows.append(extract(reader.deserialize(raw, conn.msgtype)))

    df = pd.DataFrame(rows).sort_values("t_ns").reset_index(drop=True)
    df["t_ns"] = df["t_ns"].astype("int64")
    return df


def load_sequence(name: str) -> dict[str, pd.DataFrame]:
    """Load every stream of one sequence, on a shared time origin.

    Parameters
    ----------
    name : str
        A key of ``SEQUENCES``, for example ``"Urban04"``.

    Returns
    -------
    dict of str to pd.DataFrame
        One table per entry in ``STREAMS``, each with ``t_ns`` and ``t``.
    """
    run_dir = RAW / SEQUENCES[name]
    tables = {
        stream: read_stream(run_dir, bag, topic, extract)
        for stream, (bag, topic, extract) in STREAMS.items()
    }

    # Shared origin so t is comparable between streams. Kept alongside t_ns,
    # which stays exact for joining.
    t0 = min(df["t_ns"].iloc[0] for df in tables.values())
    for df in tables.values():
        df.insert(1, "t", (df["t_ns"] - t0) / 1e9)

    return tables


def write_parquet(name: str, tables: dict[str, pd.DataFrame]) -> Path:
    """Write each table to ``data/processed/<name>/<stream>.parquet``.

    Parameters
    ----------
    name : str
        Sequence name, used as the output directory.
    tables : dict of str to pd.DataFrame
        As returned by :func:`load_sequence`.

    Returns
    -------
    Path
        The directory written to.
    """
    out_dir = PROCESSED / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for stream, df in tables.items():
        df.to_parquet(out_dir / f"{stream}.parquet", index=False)
    return out_dir


def main() -> None:
    """Load one sequence, write its Parquet tables, and print a summary."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sequence", default="Urban04", choices=sorted(SEQUENCES))
    args = parser.parse_args()

    tables = load_sequence(args.sequence)
    out_dir = write_parquet(args.sequence, tables)

    print(f"{args.sequence} -> {out_dir.relative_to(REPO)}")
    for stream, df in tables.items():
        span = df["t"].iloc[-1] - df["t"].iloc[0]
        rate = (len(df) - 1) / span if span > 0 else float("nan")
        print(f"  {stream:16s} {len(df):7d} rows  {span:7.1f} s  {rate:6.1f} Hz")


if __name__ == "__main__":
    main()
