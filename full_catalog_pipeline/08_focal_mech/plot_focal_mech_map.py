#!/usr/bin/env python3
"""Per user request: map all 7U stations plus every T1 cluster's composite focal mechanism
(fit via skhash_composite_fit.py) as a beachball at that cluster's centroid location.

Reads each cluster's SKHASH composite output (<label>_skhash_composite_human/hash2/OUT/out.csv
by default -- the human-polarity-derived composite fit; pass --suffix to point at a different
composite run, e.g. "" for the amplitude-stack version or "_rpnet" for an RPNet-on-stack one)
and each cluster's centroid (mean lat/lon of its events, same convention as
skhash_composite_fit.py's cluster_centroid()).

Plots in a LOCAL equirectangular km projection (not raw lon/lat degrees) so obspy's beach()
circular patches render as true circles -- lon/lat degrees aren't isotropic at this latitude
(1 deg lon ~= cos(lat) * 1 deg lat in km), and any aspect-ratio correction applied to make the
overall map look right would otherwise squash the beachball patches into ellipses.

Some cluster centroids sit within ~1km of each other (much closer than the ~5-16km station
spacing) -- close enough that beachballs at a visible size would overlap. Those are
automatically decluttered: spread around their group's mean position with a thin leader line
back to the true centroid (marked with a small dot).

Usage:
    python full_catalog_pipeline/plot_focal_mech_map.py
    python full_catalog_pipeline/plot_focal_mech_map.py --suffix ""   # amplitude-stack version
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
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from obspy.imaging.beachball import beach

import catalog_paths

HYPODD_DIR = catalog_paths.work_dir("T1")
STATION_SEL = f"{HYPODD_DIR}/input_files/station.sel"
CLUSTERS = ["cluster0", "cluster1", "cluster2", "cluster3", "cluster4", "cluster5", "cluster8", "cluster9"]
RELOC_COLS = ["id", "lat", "lon", "depth", "x", "y", "z", "ex", "ey", "ez",
              "yr", "mo", "dy", "hr", "mi", "sc", "mag",
              "nccp", "nccs", "nctp", "ncts", "rcc", "rct", "cid"]
KM_PER_DEG_LAT = 111.32


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suffix", default="_human",
                    help="Selects <label>_skhash_composite<suffix>/hash2/OUT/out.csv as the mechanism source.")
    ap.add_argument("--reloc-file", default=f"{HYPODD_DIR}/output_files/hypoDD.reloc")
    ap.add_argument("--out-png", default=None,
                    help=f"Defaults to {HYPODD_DIR}/t1_focal_mech_map<suffix>.png")
    ap.add_argument("--beach-size-km", type=float, default=1.1, help="Beachball diameter in km.")
    ap.add_argument("--declutter-radius-km", type=float, default=2.2,
                    help="Cluster centroids closer than this get spread apart around their group mean.")
    args = ap.parse_args()
    if args.out_png is None:
        args.out_png = f"{HYPODD_DIR}/t1_focal_mech_map{args.suffix}.png"
    return args


def load_stations(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["net_sta", "lat", "lon", "elv"])
    df["sta"] = df["net_sta"].str.split(".").str[-1]
    return df


def cluster_centroid(cluster, reloc_file):
    ids_file = f"{HYPODD_DIR}/{cluster}_event_ids.txt"
    ids = set(int(x) for x in open(ids_file))
    reloc = pd.read_csv(reloc_file, sep=r"\s+", header=None, names=RELOC_COLS)
    reloc = reloc[reloc["id"].isin(ids)]
    return dict(lat=reloc["lat"].mean(), lon=reloc["lon"].mean(), n_events=len(reloc))


def load_mechanism(cluster, suffix):
    path = f"{HYPODD_DIR}/{cluster}_skhash_composite{suffix}/hash2/OUT/out.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    row = df.iloc[0]
    return dict(strike=row["strike"], dip=row["dip"], rake=row["rake"], quality=row["quality"],
                fault_unc=row["fault_plane_uncertainty"], aux_unc=row["aux_plane_uncertainty"])


def declutter(clusters, radius_km):
    """Groups clusters whose true (x_km, y_km) centroids are within radius_km of ANY other
    cluster in the group (single-link), then spreads each group's members evenly on a small
    circle around the group's mean position. Returns plot (x,y) per cluster, equal to the true
    centroid for un-grouped clusters."""
    n = len(clusters)
    pts = np.array([[c["x_km"], c["y_km"]] for c in clusters])
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(*(pts[i] - pts[j])) < radius_km:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    plot_xy = [None] * n
    for members in groups.values():
        if len(members) == 1:
            i = members[0]
            plot_xy[i] = tuple(pts[i])
            continue
        mean = pts[members].mean(axis=0)
        spread_r = radius_km * 0.6
        for k, i in enumerate(members):
            angle = 2 * np.pi * k / len(members)
            plot_xy[i] = (mean[0] + spread_r * np.cos(angle), mean[1] + spread_r * np.sin(angle))
    return plot_xy


def main():
    args = parse_args()
    stations = load_stations(STATION_SEL)

    clusters = []
    for c in CLUSTERS:
        mech = load_mechanism(c, args.suffix)
        centroid = cluster_centroid(c, args.reloc_file)
        clusters.append(dict(label=c, mech=mech, **centroid))

    lat0 = np.mean(list(stations["lat"]) + [c["lat"] for c in clusters])
    lon0 = np.mean(list(stations["lon"]) + [c["lon"] for c in clusters])
    km_per_deg_lon = KM_PER_DEG_LAT * np.cos(np.radians(lat0))

    def to_km(lat, lon):
        return (lon - lon0) * km_per_deg_lon, (lat - lat0) * KM_PER_DEG_LAT

    stations["x_km"], stations["y_km"] = zip(*[to_km(r.lat, r.lon) for r in stations.itertuples()])
    for c in clusters:
        c["x_km"], c["y_km"] = to_km(c["lat"], c["lon"])
        status = (f"strike={c['mech']['strike']:.0f} dip={c['mech']['dip']:.0f} "
                  f"rake={c['mech']['rake']:.0f} Q={c['mech']['quality']}") if c["mech"] else "no mechanism"
        print(f"{c['label']}: n={c['n_events']}, xy_km=({c['x_km']:.2f},{c['y_km']:.2f}), {status}")

    plot_xy = declutter(clusters, args.declutter_radius_km)

    fig, ax = plt.subplots(figsize=(10, 9))
    all_x = list(stations["x_km"]) + [p[0] for p in plot_xy]
    all_y = list(stations["y_km"]) + [p[1] for p in plot_xy]
    pad = 0.22 * max(max(all_x) - min(all_x), max(all_y) - min(all_y)) + 1.0
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
    ax.set_aspect("equal")

    ax.scatter(stations["x_km"], stations["y_km"], marker="^", s=240, facecolor="#2b6f8f",
               edgecolor="white", linewidth=1.2, zorder=5, label="7U station")
    for r in stations.itertuples():
        ax.annotate(r.sta, (r.x_km, r.y_km), textcoords="offset points", xytext=(0, 13),
                    ha="center", fontsize=10.5, fontweight="bold", color="#1a4a61", zorder=6)

    quality_colors = {"A": "#1f7a4a", "B": "#3f8f5f", "C": "#b58a1f", "D": "#7a3f3f"}
    for c, (px, py) in zip(clusters, plot_xy):
        n_label = f"{c['label'].replace('cluster', 'C')} (n={c['n_events']})"
        moved = np.hypot(px - c["x_km"], py - c["y_km"]) > 1e-6
        if moved:
            ax.plot([c["x_km"], px], [c["y_km"], py], color="0.6", lw=0.8, ls="-", zorder=3)
            ax.scatter([c["x_km"]], [c["y_km"]], marker="o", s=14, color="0.4", zorder=4)
        if c["mech"] is None:
            ax.scatter([px], [py], marker="x", s=90, color="0.5", zorder=7)
            ax.annotate(n_label + "\nno mechanism", (px, py), textcoords="offset points",
                        xytext=(0, -20), ha="center", fontsize=8.5, color="0.4", zorder=9)
            continue
        m = c["mech"]
        color = quality_colors.get(m["quality"], "#555555")
        b = beach([m["strike"], m["dip"], m["rake"]], xy=(px, py), width=args.beach_size_km,
                   facecolor=color, edgecolor="black", linewidth=0.9, zorder=8)
        ax.add_collection(b)
        ax.annotate(n_label, (px, py), textcoords="offset points", xytext=(0, -args.beach_size_km * 55 - 6),
                    ha="center", fontsize=8.5, color="#333333", zorder=9)

    legend_elems = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#2b6f8f", markeredgecolor="white",
               markersize=14, label="7U station"),
    ]
    for q, col in quality_colors.items():
        legend_elems.append(Patch(facecolor=col, edgecolor="black", label=f"composite mechanism, quality {q}"))
    legend_elems.append(Line2D([0], [0], marker="o", color="0.6", markerfacecolor="0.4", markersize=5,
                                label="true centroid (decluttered)"))
    ax.legend(handles=legend_elems, loc="upper left", fontsize=9, framealpha=0.9)

    ax.set_xlabel(f"East-west distance from array center (km)")
    ax.set_ylabel(f"North-south distance from array center (km)")
    ax.set_title(f"T1 array: composite focal mechanisms by cluster "
                 f"(polarity source: {args.suffix.strip('_') or 'amplitude-stack'})", fontsize=12)
    ax.grid(True, alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=170, bbox_inches="tight")
    print(f"\nwrote {args.out_png}")


if __name__ == "__main__":
    main()
