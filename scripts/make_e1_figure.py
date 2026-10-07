"""E1: how far does dead reckoning drift, and what does the odometer buy?

Regenerates ``results/e1_dead_reckoning.png`` and ``results/metrics.csv`` from
the processed Parquet. Run from the repository root::

    python scripts/make_e1_figure.py
    python scripts/make_e1_figure.py --sequence Urban03

The work is split so that the figures can be reworked without touching the
estimation, and so a notebook can import every piece:

- :func:`load_e1` runs the pipeline and returns plain DataFrames.
- :func:`plot_e1` takes those and returns the static three-panel figure.
- :func:`animate_run` takes those and writes a GIF or MP4 of one run.

Everything about how they *look* lives in :data:`STYLE`.

Nothing is defined in the notebooks, deliberately: every published figure has to
be reproducible from a clone by running this file, with no notebook state and no
cells run in the right order.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.align import build_timeline, reference_at  # noqa: E402
from src.dead_reckoning import (  # noqa: E402
    SpeedSource,
    dead_reckon,
    estimate_gyro_bias,
    find_stationary_window,
)
from src.frames import add_enu, azimuth_to_yaw, origin_from_reference  # noqa: E402
from src.metrics import horizontal_error, summarise, to_csv  # noqa: E402

# Okabe-Ito, which stays distinguishable in greyscale and for colour blindness.
STYLE = {
    "reference": {"color": "#000000", "ls": "--", "lw": 1.2, "label": "reference (truth)"},
    "odometer": {"color": "#0072B2", "lw": 1.2, "label": "IMU and odometer"},
    "inertial": {"color": "#E69F00", "lw": 1.2, "label": "IMU only"},
    "start": {"color": "#CC79A7", "ms": 11, "mew": 2.0},
    "window": {"color": "#CC79A7", "alpha": 0.10},
    "basemap_alpha": 0.7,
    "figsize": (11.0, 9.0),
    "anim_figsize": (6.5, 6.5),
    "dpi": 200,
    # The animation is a raster, so its pixel size is figsize * anim_dpi.
    # 130 gives 845 px, which still looks sharp at a README's width; the
    # default 100 gives 650 px and reads soft once the browser upscales it.
    "anim_dpi": 130,
    # Five minutes is the horizon the project cares about: the longest synthetic
    # outage in E4 is 60 s, so short-horizon drift is what matters. The published
    # NavINST baseline happens to use the same span, but its metric differs, so
    # that comparison belongs in the README with its caveats, not on the figure.
    "window_s": 300.0,
    "error_floor_m": 0.01,   # bottom of the log axis in panel (c)
}


def load_e1(sequence: str = "Urban04") -> dict:
    """Run both dead-reckoning modes and return everything the figure needs.

    Parameters
    ----------
    sequence : str
        Folder name under ``data/processed``.

    Returns
    -------
    dict
        ``timeline``, ``odometer``, ``inertial`` (DataFrames), ``bias`` (rad/s),
        ``t_move`` (when the vehicle first moves) and ``window``.
    """
    processed = REPO / "data" / "processed" / sequence
    tables = {
        name: pd.read_parquet(processed / f"{name}.parquet")
        for name in ("imu", "gnss", "odometry", "reference")
    }

    origin = origin_from_reference(tables["reference"])
    reference = add_enu(tables["reference"], origin)
    reference["yaw"] = azimuth_to_yaw(reference["azimuth"])
    gnss = add_enu(tables["gnss"], origin, alt_col="alt")

    timeline = build_timeline(tables["imu"], gnss, tables["odometry"], reference=reference)

    _, t_move = find_stationary_window(tables["odometry"])
    bias = estimate_gyro_bias(tables["imu"], 0.0, t_move)

    # The one place an estimator reads truth: the initial pose.
    start = reference_at(np.array([timeline["t"].iloc[0]]), reference)
    pose = dict(
        east0=float(start["east"][0]),
        north0=float(start["north"][0]),
        yaw0=float(start["yaw"][0]),
        speed0=0.0,
        bias=bias,
    )

    return {
        "sequence": sequence,
        "timeline": timeline,
        "odometer": dead_reckon(timeline, tables["odometry"], speed_source="odometer", **pose),
        "inertial": dead_reckon(timeline, tables["odometry"], speed_source="inertial", **pose),
        "bias": bias,
        "t_move": t_move,
        "window": (t_move, t_move + STYLE["window_s"]),
    }


def read_basemap(sequence: str, results: Path | None = None):
    """Load the cached street map and its extent in the local ENU frame.

    Written once by ``scripts/fetch_basemap.py`` and committed, so reading it
    needs neither a network connection nor ``contextily``.

    Parameters
    ----------
    sequence : str
        Sequence the map was fetched for.
    results : Path or None
        Where to look. Defaults to ``results/``.

    Returns
    -------
    image, extent : np.ndarray and list of float, or (None, None)
        ``(None, None)`` when the cache is absent, so callers can carry on
        without it rather than failing.
    attribution : str or None
        The credit the tile provider requires, to be drawn on the figure.
    """
    results = Path(results) if results is not None else REPO / "results"
    image_path = results / f"basemap_{sequence}.png"
    meta_path = results / f"basemap_{sequence}.json"
    if not (image_path.exists() and meta_path.exists()):
        return None, None, None

    meta = json.loads(meta_path.read_text())
    return mpimg.imread(image_path), meta["extent_enu"], meta["attribution"]


def plot_e1(
    data: dict,
    style: dict | None = None,
    basemap: bool = True,
    results: Path | None = None,
) -> plt.Figure:
    """Three panels: the trajectories at both scales, and the error over time.

    Parameters
    ----------
    data : dict
        From :func:`load_e1`.
    style : dict or None
        Overrides :data:`STYLE`. Only the keys you pass are replaced.
    basemap : bool
        Draw the cached street map under panel (b). Skipped silently if the
        cache is missing, so the figure still builds on a bare clone.
    results : Path or None
        Where to look for the cached basemap. Defaults to ``results/``.

    Returns
    -------
    plt.Figure

    Notes
    -----
    Only panel (b) gets the map. Panel (a) spans 17 km, where the cached tiles
    would cover a small patch in the middle and the streets would be unreadable
    at that zoom anyway.
    """
    s = {**STYLE, **(style or {})}
    tl, odo, ins = data["timeline"], data["odometer"], data["inertial"]
    t_from, t_to = data["window"]

    fig, ax = plt.subplot_mosaic(
        [["full", "zoom"], ["error", "error"]],
        figsize=s["figsize"],
        layout="constrained",
    )

    # (a) full scale: the IMU-only track leaves the map entirely
    for name, df in (("reference", tl.rename(columns={"ref_east": "east", "ref_north": "north"})),
                     ("odometer", odo),
                     ("inertial", ins)):
        ax["full"].plot(df["east"], df["north"], **s[name])
    ax["full"].set_title("(a) full scale")

    # (b) zoomed to the route, where the two tracks separate visibly
    if basemap:
        image, extent, credit = read_basemap(data["sequence"], results)
        if image is not None:
            ax["zoom"].imshow(image, extent=extent, alpha=s["basemap_alpha"], zorder=0)
            ax["zoom"].text(0.99, 0.01, credit, transform=ax["zoom"].transAxes,
                            ha="right", va="bottom", fontsize=5, color="0.35")
    for name, df in (("reference", tl.rename(columns={"ref_east": "east", "ref_north": "north"})),
                     ("odometer", odo)):
        ax["zoom"].plot(df["east"], df["north"], zorder=3, **s[name])
    # imshow would stretch the panel to the whole cached tile area, which is
    # larger than the route, so the limits come from the tracks instead.
    east = np.r_[tl["ref_east"], odo["east"]]
    north = np.r_[tl["ref_north"], odo["north"]]
    pad = 0.04 * max(np.ptp(east), np.ptp(north))
    ax["zoom"].set_xlim(east.min() - pad, east.max() + pad)
    ax["zoom"].set_ylim(north.min() - pad, north.max() + pad)
    ax["zoom"].set_title("(b) route, zoomed")

    # The route closes to within 0.31 m, so one marker is both start and end.
    for key in ("full", "zoom"):
        ax[key].plot(tl["ref_east"].iloc[0], tl["ref_north"].iloc[0], "o",
                     mfc="none", label="true start / end", **s["start"])
        ax[key].set_aspect("equal")
        ax[key].set_xlabel("East [m]")
        ax[key].set_ylabel("North [m]")
        ax[key].legend(loc="best", fontsize=8)
        ax[key].grid(True, alpha=0.2)

    # (c) horizontal error against time, log scale so 5 m and 2 km both read
    for name, df in (("odometer", odo), ("inertial", ins)):
        ax["error"].semilogy(tl["t"], horizontal_error(df, tl), **s[name])
    ax["error"].axvspan(t_from, t_to, label="5-minute window", **s["window"])
    ax["error"].set_xlabel("Time [s]")
    ax["error"].set_ylabel("Horizontal position error [m]")
    ax["error"].set_title("(c) dead-reckoning error against the reference")
    # Both runs start from the reference pose, so the error begins near zero and
    # a log axis would otherwise give six decades of uninteresting floor.
    ax["error"].set_ylim(bottom=s["error_floor_m"])
    ax["error"].legend(loc="lower right", fontsize=8)
    ax["error"].grid(True, which="both", alpha=0.2)

    return fig


def shrink_gif(path: Path, fps: int, colors: int = 128) -> None:
    """Re-encode a GIF with one shared palette, which makes it delta-compress.

    Matplotlib writes every frame in full with its own palette, so nothing
    compresses between frames and a 650 px animation over a basemap comes out
    at 15 MB. Giving every frame the same palette means unchanged pixels keep
    identical indices and GIF's own inter-frame compression can do its job: the
    same animation drops to about 0.36 MB with no visible difference.

    Parameters
    ----------
    path : Path
        GIF to rewrite in place.
    fps : int
        Frame rate, needed because the frame durations are rewritten too.
    colors : int
        Palette size. 128 covers the antialiasing tones of line art over a
        grey basemap with room to spare; 64 is visually identical here but
        leaves no margin at higher resolutions.
    """
    from PIL import Image

    source = Image.open(path)
    frames = []
    try:
        index = 0
        while True:
            source.seek(index)
            frames.append(source.convert("RGB"))
            index += 1
    except EOFError:
        pass
    source.close()

    # The last frame carries the most ink, so its palette covers every colour
    # that appears in any earlier frame.
    palette = frames[-1].quantize(colors=colors, method=Image.MEDIANCUT)
    indexed = [f.quantize(palette=palette, dither=Image.NONE) for f in frames]
    indexed[0].save(path, save_all=True, append_images=indexed[1:], loop=0,
                    duration=int(1000 / fps), optimize=True)


def animate_run(
    data: dict,
    mode: SpeedSource = "odometer",
    step: int = 500,
    fps: int = 20,
    fmt: str = "gif",
    basemap: bool = False,
    basemap_alpha: float = 0.7,
    trail_only: bool = False,
    title: str | None = None,
    results: Path | None = None,
    style: dict | None = None,
) -> Path:
    """Animate one dead-reckoning run drawing itself over the reference.

    Parameters
    ----------
    data : dict
        From :func:`load_e1`.
    mode : {"odometer", "inertial"}
        Which run to draw.
    step : int
        Keep every ``step``-th sample. 500 gives 213 frames from 100 Hz data;
        106,500 frames is not an animation.
    fps : int
        Frames per second in the output.
    fmt : {"gif", "mp4"}
        ``mp4`` is smaller but needs ffmpeg on the PATH, and does not embed in
        a GitHub README.
    basemap : bool
        Draw the cached street map underneath, from
        ``results/basemap_<sequence>.png``. Skipped with a warning when that
        file is absent, so a fresh clone still produces the animation.
    basemap_alpha : float
        Transparency of the map. The map fades, the tracks do not.
    trail_only : bool
        Hide the reference, leaving only the estimate.
    title : str or None
        Line above the running clock. Defaults to the run's label.
    results : Path or None
        Output directory. Defaults to ``results/``.
    style : dict or None
        Overrides :data:`STYLE`, as in :func:`plot_e1`.

    Returns
    -------
    Path
        Where the animation was written.

    Notes
    -----
    The basemap is used here but deliberately not in :func:`plot_e1`. The static
    figure must reproduce on a bare clone with no network and no cached tiles;
    this one is for the README, where the street grid is worth the dependency.
    That the route lands on real streets is also an independent check on the
    coordinate conversion, which no unit test provides.
    """
    s = {**STYLE, **(style or {})}
    results = Path(results) if results is not None else REPO / "results"
    run = data[mode].iloc[::step]
    ref = data["timeline"].iloc[::step]

    fig, ax = plt.subplots(figsize=s["anim_figsize"], dpi=s["anim_dpi"],
                           layout="constrained")

    credit = None
    if basemap:
        image, extent, credit = read_basemap(data["sequence"], results)
        if image is not None:
            ax.imshow(image, extent=extent, alpha=basemap_alpha, zorder=0)
        else:
            print(f"no cached basemap for {data['sequence']}; "
                  "run scripts/fetch_basemap.py. Continuing without it.")

    if not trail_only:
        ax.plot(ref["ref_east"], ref["ref_north"], zorder=2, **s["reference"])

    (trail,) = ax.plot([], [], lw=1.8, zorder=3,
                       color=s[mode]["color"], label=s[mode]["label"])
    (head,) = ax.plot([], [], "o", ms=7, zorder=4, color=s[mode]["color"])

    # Blitting does not autoscale, so the limits are set once, from both tracks.
    east = np.r_[ref["ref_east"], run["east"]]
    north = np.r_[ref["ref_north"], run["north"]]
    pad = 0.05 * max(np.ptp(east), np.ptp(north))
    ax.set_xlim(east.min() - pad, east.max() + pad)
    ax.set_ylim(north.min() - pad, north.max() + pad)

    ax.set_aspect("equal")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.25)
    if credit:
        ax.text(0.99, 0.01, credit, transform=ax.transAxes, ha="right",
                va="bottom", fontsize=6, color="0.35")

    # A GIF travels without its caption, so the title has to name the method.
    # "Dead reckoning" is common to both runs, so it belongs here rather than in
    # the legend, which exists to tell the two curves apart.
    heading = f"Dead reckoning: {s[mode]['label']}" if title is None else title
    clock = ax.set_title("")

    def frame(k):
        trail.set_data(run["east"].iloc[:k], run["north"].iloc[:k])
        head.set_data(run["east"].iloc[k - 1:k], run["north"].iloc[k - 1:k])
        clock.set_text(f"{heading}\nt = {run['t'].iloc[k - 1]:6.1f} s")
        return trail, head, clock

    anim = FuncAnimation(fig, frame, frames=range(1, len(run)),
                         interval=1000 // fps, blit=True)

    results.mkdir(parents=True, exist_ok=True)
    path = results / f"e1_{mode}{'_map' if basemap and credit else ''}.{fmt}"
    writer = PillowWriter(fps=fps) if fmt == "gif" else FFMpegWriter(fps=fps, bitrate=1800)
    anim.save(path, writer=writer)
    plt.close(fig)

    if fmt == "gif":
        shrink_gif(path, fps)
    print(f"{len(run)} frames -> {path.name} ({path.stat().st_size / 1e6:.2f} MB)")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="Urban04")
    parser.add_argument("--results", default=str(REPO / "results"))
    parser.add_argument("--no-animations", action="store_true",
                        help="skip the GIFs, which take about a minute")
    args = parser.parse_args()

    results = Path(args.results)
    data = load_e1(args.sequence)

    # to_csv appends, so a stale file would double every run.
    metrics_path = results / "metrics.csv"
    metrics_path.unlink(missing_ok=True)
    for name in ("odometer", "inertial"):
        table = summarise(data[name], data["timeline"], f"dr_{name}", data["window"])
        to_csv(table, str(metrics_path))
        rmse = table.loc[table["metric"] == "horizontal_rmse", "value"].iloc[0]
        print(f"{name:>9}: 5-min horizontal RMSE {rmse:10.2f} m")

    # SVG, not PNG: the trajectories are long but Matplotlib's path
    # simplification collapses the sub-pixel steps, so the whole three-panel
    # figure is about 140 kB of vector and stays sharp at any zoom.
    figure_path = results / "e1_dead_reckoning.svg"
    plot_e1(data).savefig(figure_path, dpi=STYLE["dpi"])
    print(f"wrote {figure_path.name} and {metrics_path.name}")

    if not args.no_animations:
        animate_run(data, "odometer", basemap=True, results=results)
        animate_run(data, "inertial", results=results)


if __name__ == "__main__":
    # Only when run as a script, never on import: this file writes files and has
    # no use for a window, and the interactive backend bounces a Python icon in
    # the Dock for every figure. A notebook importing these functions keeps its
    # own inline backend, which is why this is not at module level.
    import matplotlib

    matplotlib.use("Agg")
    main()
