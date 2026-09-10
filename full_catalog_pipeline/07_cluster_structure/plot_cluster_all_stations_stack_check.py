#!/usr/bin/env python3
"""All-stations version of plot_cluster_deej_stack_check.py: one page, one row per station
that has usable P+S picks for the cluster, stations ordered top-to-bottom by increasing
distance from the cluster's centroid (haversine, station.dat coords vs. mean event lat/lon --
not hypoDD's own .src file, which only exists for cluster3's dedicated SVD rerun, not for
clusters in general). Built to extend the DEEJ/TJTJ/LILA firn-reverberation cross-station
check ([[t1-sp-vs-depth-vpvs-check-result]] Result 4/5) to any cluster, in one figure.

Usage:
    python full_catalog_pipeline/plot_cluster_all_stations_stack_check.py [--ids-file ... --tag ...]
"""

# --- pipeline path bootstrap (added by the 2026-09 reorganization) ---------------
# Scripts live in stage subdirectories but import shared modules (catalog_paths, config,
# vels1d_model, lib.*) and occasionally each other. Python only puts a script's OWN
# directory on sys.path, so add the pipeline root and every stage directory. Invoke from
# the REPOSITORY root -- data paths inside these scripts are relative to it.
import os as _os, sys as _sys
_PIPE = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in [_PIPE] + [_os.path.join(_PIPE, _s) for _s in sorted(_os.listdir(_PIPE))
                     if _os.path.isdir(_os.path.join(_PIPE, _s))
                     and not _s.startswith(('.', 'artifacts', '__'))]:
    if _d not in _sys.path:
        _sys.path.insert(0, _d)
# --- end bootstrap ---------------------------------------------------------------

import argparse
from math import radians, sin, cos, sqrt, atan2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from lib.day_volume_index import load_day_file_index_csv
from lib.windowing import RollingDayCache, extract_window
from lib.deej_waveform_common import (
    DEFAULTS, RELOC_COLS, load_ids, load_merged, clean, horizontal_envelope,
)

MIN_EVENTS = 10  # below this, a station's stack is too noisy/meaningless to include
POST_S_MARGIN = 0.65  # seconds of view kept AFTER the median S pick, to catch any secondary burst
STATION_DAT_DEFAULT = f"{DEFAULTS['out_dir']}/input_files/station.dat"


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * r * atan2(sqrt(a), sqrt(1 - a))


def load_station_dat(path):
    """station name (no network prefix) -> (lat, lon)."""
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            out[parts[0].split(".")[-1]] = (float(parts[1]), float(parts[2]))
    return out


def cluster_centroid_latlon(ids, reloc_file):
    with open(reloc_file) as f:
        n_fields = len(f.readline().split())
    cols = RELOC_COLS if n_fields == 18 else RELOC_COLS[:17] + [
        "nccp", "nccs", "nctp", "ncts", "rcc", "rct", "cid"]
    reloc = pd.read_csv(reloc_file, sep=r"\s+", header=None, names=cols)
    g = reloc[reloc["id"].isin(ids)]
    return g["lat"].mean(), g["lon"].mean()


def stack_for_station(station, ids, args, day_index):
    df = load_merged(ids, args.phase_dat, args.reloc, station, network=args.network)
    if len(df) < MIN_EVENTS:
        return None

    # window_post is adaptive per station: distant stations have a later S pick, so a fixed
    # window (sized for DEEJ, the closest station) would cut off before or right at their S
    # pick and hide any post-S secondary burst -- see [[t1-sp-vs-depth-vpvs-check-result]].
    median_sp_preview = (df["s_offset"] - df["p_offset"]).median()
    window_post = max(args.window_post, median_sp_preview + POST_S_MARGIN)

    cache = RollingDayCache(day_index, station)
    n_samp = int(round((args.window_pre + window_post) * config.SAMPLE_RATE_HZ))

    z_stack = np.zeros(n_samp)
    h_stack = np.zeros(n_samp)
    n_ok = 0
    for row in df.itertuples():
        p_abs = row.origin + row.p_offset
        w0, w1 = p_abs - args.window_pre, p_abs + window_post
        r = {"primary_date": p_abs.date.isoformat()}
        data, status = extract_window(cache, r, w0, w1)
        if data is None or len(data["Z"]) < n_samp:
            continue
        z = clean(data["Z"][:n_samp])
        h = horizontal_envelope(data["N"][:n_samp], data["E"][:n_samp])
        zpk = np.max(np.abs(z)) or 1.0
        z_stack += np.abs(z) / zpk
        h_stack += h
        n_ok += 1
    if n_ok < MIN_EVENTS:
        return None
    z_stack /= n_ok
    h_stack /= n_ok

    median_sp = (df["s_offset"] - df["p_offset"]).median()
    depth_min, depth_max = df["depth"].min(), df["depth"].max()
    return dict(station=station, n_ok=n_ok, n_total=len(df), z_stack=z_stack, h_stack=h_stack,
                median_sp=median_sp, depth_min=depth_min, depth_max=depth_max)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ids-file", default=DEFAULTS["ids_file"])
    p.add_argument("--phase-dat", default=DEFAULTS["phase_dat"])
    p.add_argument("--reloc", default=DEFAULTS["reloc"])
    p.add_argument("--station-dat", default=STATION_DAT_DEFAULT,
                   help="Station coordinate file (name lat lon [elev]), for centroid distance.")
    p.add_argument("--out-dir", default=DEFAULTS["out_dir"])
    p.add_argument("--tag", default="cluster3_all_stations")
    p.add_argument("--network", default=DEFAULTS["network"])
    p.add_argument("--stations", nargs="+", default=None,
                   help="Station codes to consider (default: all stations in --station-dat).")
    p.add_argument("--window-pre", type=float, default=DEFAULTS["window_pre"])
    p.add_argument("--window-post", type=float, default=DEFAULTS["window_post"])
    args = p.parse_args()

    ids = load_ids(args.ids_file)
    day_index = load_day_file_index_csv(config.DAY_FILE_INDEX_CSV)

    station_coords = load_station_dat(args.station_dat)
    stations = args.stations if args.stations is not None else sorted(station_coords)

    centroid_lat, centroid_lon = cluster_centroid_latlon(ids, args.reloc)
    station_dist_km = {
        sta: haversine_km(centroid_lat, centroid_lon, *station_coords[sta])
        for sta in stations if sta in station_coords
    }

    results = []
    for station in stations:
        print(f"=== {station} ===")
        try:
            res = stack_for_station(station, ids, args, day_index)
        except (KeyError, ValueError):
            res = None
        if res is None:
            print(f"  skipped (fewer than {MIN_EVENTS} usable events)")
            continue
        res["dist_km"] = station_dist_km.get(station, float("nan"))
        print(f"  n_ok={res['n_ok']}/{res['n_total']}, dist={res['dist_km']:.2f} km, "
              f"median S-P={res['median_sp']*1000:.0f} ms")
        results.append(res)

    results.sort(key=lambda r: r["dist_km"])

    n_rows = len(results)
    dt = 1.0 / config.SAMPLE_RATE_HZ
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 2.6 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for i, res in enumerate(results):
        n_samp = len(res["z_stack"])
        xs = np.arange(n_samp) * dt - args.window_pre
        ax_z, ax_h = axes[i, 0], axes[i, 1]

        row_info = (f"{res['station']}: n={res['n_ok']}, dist={res['dist_km']:.1f} km, "
                    f"depth {res['depth_min']:.2f}-{res['depth_max']:.2f} km")

        ax_z.plot(xs, res["z_stack"], color="#0b0b0b", linewidth=1.1)
        ax_z.axvline(0, color="#2a78d6", linestyle="--", linewidth=1)
        ax_z.axvline(res["median_sp"], color="crimson", linestyle="--", linewidth=1)
        ax_z.set_ylabel("mean |Z|\n(stacked)", fontsize=9)
        ax_z.set_title(("Z (vertical) -- " if i == 0 else "") + row_info, fontsize=9, loc="left")

        ax_h.plot(xs, res["h_stack"], color="#0b0b0b", linewidth=1.1)
        ax_h.axvline(0, color="#2a78d6", linestyle="--", linewidth=1,
                     label="P" if i == 0 else None)
        ax_h.axvline(res["median_sp"], color="crimson", linestyle="--", linewidth=1,
                     label="median actual S pick" if i == 0 else None)
        ax_h.set_ylabel("mean horiz.\nenvelope", fontsize=9)
        ax_h.set_title(("Horizontal envelope -- " if i == 0 else "") + row_info, fontsize=9, loc="left")
        if i == 0:
            ax_h.legend(fontsize=8, loc="upper right")

        ax_z.set_xlim(xs[0], xs[-1])
        ax_h.set_xlim(xs[0], xs[-1])
        ax_z.set_xlabel("Time relative to P pick (s)")
        ax_h.set_xlabel("Time relative to P pick (s)")

    fig.suptitle(f"{args.tag}: stacked P-aligned envelopes, stations ordered by distance from cluster",
                 fontsize=12, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_png = f"{args.out_dir}/{args.tag}_stack_check.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
