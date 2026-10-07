"""Fetch an OpenStreetMap basemap for a sequence's route and cache it locally.

Run once, with a network connection::

    python scripts/fetch_basemap.py --sequence Urban04

It writes ``results/basemap_<sequence>.png`` and ``results/basemap_<sequence>.json``
carrying the image's extent in the local ENU frame. Everything downstream reads
those two files, so the figures reproduce offline, with no tile server, no API
key and no dependency on this script having worked today. Both files are
committed for that reason.

The attribution the provider requires is stored in the JSON and must appear on
any figure that uses the image.

Why the extent is stored rather than recomputed
-----------------------------------------------
Tiles arrive in Web Mercator, where one unit is not one ground metre: the scale
factor is ``1 / cos(latitude)``, which is 1.40 at Kingston's 44.24 degrees.
Drawing an ENU trajectory straight onto a Mercator extent would stretch the map
by 40% against the track, and both are nominally "metres" so it would look
plausible. The corners are therefore converted back through
``frames.enu_to_geodetic``'s inverse and the extent is stored in ENU metres.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.frames import (  # noqa: E402
    add_enu,
    enu_to_geodetic,
    geodetic_to_enu,
    origin_from_reference,
)


def fetch(
    sequence: str = "Urban04",
    margin: float = 0.12,
    zoom: int = 16,
    provider: str = "Esri.WorldGrayCanvas",
) -> None:
    """Download tiles covering the route and cache the image plus its ENU extent.

    Parameters
    ----------
    sequence : str
        Folder name under ``data/processed``.
    margin : float
        Fraction of the route's span to add on each side.
    zoom : int
        Tile zoom level. 16 gives readable street names over a few square
        kilometres; 15 is a quarter of the tiles and correspondingly coarser.
    provider : str
        Dotted name in ``contextily.providers``. The default is a light grey
        canvas, which sits under coloured trajectories without competing with
        them. ``OpenStreetMap.Mapnik`` works too but is busy and saturated.
        ``CartoDB.Positron`` requires an API key and is not usable here.
    """
    import contextily as ctx       # imported here so the project runs without it

    # OpenStreetMap's servers reject the default client, and their usage policy
    # requires an identifying agent. This is a one-off fetch of a few dozen
    # tiles, which is within the policy; bulk scraping would not be.
    ctx.tile.USER_AGENT = "imu-gnss-localization/0.1 (research figure; github.com/xyzhang8)"

    source = ctx.providers
    for part in provider.split("."):
        source = source[part]

    processed = REPO / "data" / "processed" / sequence
    reference = pd.read_parquet(processed / "reference.parquet")
    origin = origin_from_reference(reference)
    enu = add_enu(reference, origin)

    pad_e = margin * float(np.ptp(enu["east"]))
    pad_n = margin * float(np.ptp(enu["north"]))
    east0, east1 = enu["east"].min() - pad_e, enu["east"].max() + pad_e
    north0, north1 = enu["north"].min() - pad_n, enu["north"].max() + pad_n

    # ENU corners -> geodetic -> the lat/lon box contextily wants.
    lat0, lon0 = to_geodetic(origin, east0, north0)
    lat1, lon1 = to_geodetic(origin, east1, north1)

    image, bbox = ctx.bounds2img(lon0, lat0, lon1, lat1, zoom=zoom, ll=True, source=source)

    # bounds2img snaps outwards to whole tiles, so it returns more than was
    # asked for. Convert what it actually returned back into ENU metres.
    lon_min, lon_max, lat_min, lat_max = mercator_bbox_to_ll(bbox)
    e_min, n_min = to_enu(origin, lat_min, lon_min)
    e_max, n_max = to_enu(origin, lat_max, lon_max)

    results = REPO / "results"
    results.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    plt.imsave(results / f"basemap_{sequence}.png", image)
    (results / f"basemap_{sequence}.json").write_text(json.dumps({
        "sequence": sequence,
        "extent_enu": [e_min, e_max, n_min, n_max],
        "zoom": zoom,
        "origin": {"lat": origin.lat, "lon": origin.lon, "alt": origin.alt},
        "provider": provider,
        "attribution": source.get("attribution", provider),
    }, indent=2) + "\n")

    print(f"wrote basemap_{sequence}.png  {image.shape[1]}x{image.shape[0]} px")
    print(f"extent ENU: east {e_min:.0f} to {e_max:.0f}, north {n_min:.0f} to {n_max:.0f} m")


def to_geodetic(origin, east, north):
    """ENU metres to (lat, lon) degrees."""
    lat, lon, _ = enu_to_geodetic(
        np.array([east]), np.array([north]), np.array([0.0]), origin
    )
    return float(lat[0]), float(lon[0])


def to_enu(origin, lat, lon):
    """(lat, lon) degrees to ENU east, north in metres."""
    east, north, _ = geodetic_to_enu(
        np.array([lat]), np.array([lon]), np.array([origin.alt]), origin
    )
    return float(east[0]), float(north[0])


def mercator_bbox_to_ll(bbox):
    """contextily's (left, right, bottom, top) in Web Mercator to lat/lon."""
    import mercantile

    left, right, bottom, top = bbox
    lon_min, lat_min = mercantile.lnglat(left, bottom)
    lon_max, lat_max = mercantile.lnglat(right, top)
    return lon_min, lon_max, lat_min, lat_max


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", default="Urban04")
    parser.add_argument("--zoom", type=int, default=16)
    parser.add_argument("--provider", default="Esri.WorldGrayCanvas")
    args = parser.parse_args()
    fetch(args.sequence, zoom=args.zoom, provider=args.provider)


if __name__ == "__main__":
    main()
