#!/usr/bin/env python3
"""Per user request: map all 7U stations plus, for every T1 cluster, the FULL RANGE of
acceptable composite focal mechanisms (SKHASH's bootstrap ensemble of nodal-plane solutions
that satisfy the polarity misfit tolerance -- <label>_skhash_composite_human/hash2/OUT/out2.csv,
500 solutions per cluster), not just the single best-fit mechanism plotted by
plot_focal_mech_map.py.

Every cluster here graded worst quality "D" ([[t1_focal_mech_allclusters_human_polarity_result]]),
meaning the single best-fit strike/dip/rake in out.csv is not actually well constrained -- this
plot makes that uncertainty visible instead of hiding it behind one clean-looking beachball.

Method: draw a semi-transparent "cloud" of N sampled ensemble mechanisms per cluster (colored by
rake class -- thrust/normal/strike-slip -- so a genuinely consistent style shows up as a solid
color, and a genuinely unresolved mechanism shows up as a muddy blend), with the single best-fit
mechanism outlined on top for reference. Reuses plot_focal_mech_map.py's local equirectangular
projection and centroid-decluttering so beachballs render as true circles and don't overlap.

Usage:
    python full_catalog_pipeline/plot_focal_mech_map_range.py
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
from matplotlib.patches import Circle
from obspy.imaging.beachball import beach

from plot_focal_mech_map import (
    HYPODD_DIR, STATION_SEL, CLUSTERS, RELOC_COLS, KM_PER_DEG_LAT,
    load_stations, cluster_centroid, load_mechanism, declutter,
)

RAKE_CLASS_COLORS = {"thrust": "#c0392b", "normal": "#2b6f8f", "strike-slip": "#3f8f3f"}


def rake_class(rake):
    r = ((rake + 180) % 360) - 180
    if abs(r) <= 45 or abs(r) >= 135:
        return "strike-slip"
    elif 45 < r < 135:
        return "thrust"
    else:
        return "normal"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suffix", default="_human")
    ap.add_argument("--reloc-file", default=f"{HYPODD_DIR}/output_files/hypoDD.reloc")
    ap.add_argument("--out-png", default=f"{HYPODD_DIR}/t1_focal_mech_map_range.png")
    ap.add_argument("--beach-size-km", type=float, default=1.3)
    ap.add_argument("--declutter-radius-km", type=float, default=2.6)
    ap.add_argument("--n-sample", type=int, default=50,
                     help="How many of the 500 ensemble solutions to draw per cluster.")
    ap.add_argument("--cloud-alpha", type=float, default=0.05)
    return ap.parse_args()


def load_ensemble(cluster, suffix, n_sample, seed):
    path = f"{HYPODD_DIR}/{cluster}_skhash_composite{suffix}/hash2/OUT/out2.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if len(df) == 0:
        return None
    frac = df["rake"].apply(rake_class).value_counts(normalize=True).to_dict()
    rng = np.random.RandomState(seed)
    sample = df.sample(n=min(n_sample, len(df)), random_state=rng)
    return dict(sample=sample, frac=frac, n_total=len(df))


def main():
    args = parse_args()
    stations = load_stations(STATION_SEL)

    clusters = []
    for i, c in enumerate(CLUSTERS):
        mech = load_mechanism(c, args.suffix)
        ens = load_ensemble(c, args.suffix, args.n_sample, seed=i)
        centroid = cluster_centroid(c, args.reloc_file)
        clusters.append(dict(label=c, mech=mech, ens=ens, **centroid))

    lat0 = np.mean(list(stations["lat"]) + [c["lat"] for c in clusters])
    lon0 = np.mean(list(stations["lon"]) + [c["lon"] for c in clusters])
    km_per_deg_lon = KM_PER_DEG_LAT * np.cos(np.radians(lat0))

    def to_km(lat, lon):
        return (lon - lon0) * km_per_deg_lon, (lat - lat0) * KM_PER_DEG_LAT

    stations["x_km"], stations["y_km"] = zip(*[to_km(r.lat, r.lon) for r in stations.itertuples()])
    for c in clusters:
        c["x_km"], c["y_km"] = to_km(c["lat"], c["lon"])
        if c["ens"]:
            print(f"{c['label']}: n_events={c['n_events']}, ensemble_frac={c['ens']['frac']}")

    plot_xy = declutter(clusters, args.declutter_radius_km)

    fig, ax = plt.subplots(figsize=(11, 10))
    all_x = list(stations["x_km"]) + [p[0] for p in plot_xy]
    all_y = list(stations["y_km"]) + [p[1] for p in plot_xy]
    pad = 0.24 * max(max(all_x) - min(all_x), max(all_y) - min(all_y)) + 1.2
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
    ax.set_aspect("equal")

    ax.scatter(stations["x_km"], stations["y_km"], marker="^", s=220, facecolor="#2b6f8f",
               edgecolor="white", linewidth=1.2, zorder=5, label="7U station")
    for r in stations.itertuples():
        ax.annotate(r.sta, (r.x_km, r.y_km), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=10, fontweight="bold", color="#1a4a61", zorder=6)

    for c, (px, py) in zip(clusters, plot_xy):
        n_label = f"{c['label'].replace('cluster', 'C')} (n={c['n_events']})"
        moved = np.hypot(px - c["x_km"], py - c["y_km"]) > 1e-6
        if moved:
            ax.plot([c["x_km"], px], [c["y_km"], py], color="0.6", lw=0.8, ls="-", zorder=3)
            ax.scatter([c["x_km"]], [c["y_km"]], marker="o", s=14, color="0.4", zorder=4)
        if c["ens"] is None:
            ax.scatter([px], [py], marker="x", s=90, color="0.5", zorder=7)
            continue

        ax.add_patch(Circle((px, py), radius=args.beach_size_km / 2, facecolor="none",
                             edgecolor="0.75", linewidth=0.8, zorder=6))

        for _, row in c["ens"]["sample"].iterrows():
            col = RAKE_CLASS_COLORS[rake_class(row["rake"])]
            b = beach([row["strike"], row["dip"], row["rake"]], xy=(px, py),
                      width=args.beach_size_km, facecolor=col, bgcolor="none", edgecolor="none",
                      alpha=args.cloud_alpha, zorder=7)
            ax.add_collection(b)

        if c["mech"] is not None:
            m = c["mech"]
            b = beach([m["strike"], m["dip"], m["rake"]], xy=(px, py), width=args.beach_size_km,
                      facecolor="none", bgcolor="none", edgecolor="black", linewidth=1.4, zorder=8)
            ax.add_collection(b)

        frac = c["ens"]["frac"]
        frac_txt = " / ".join(f"{k[:1].upper()}{frac.get(k,0)*100:.0f}%"
                               for k in ("thrust", "normal", "strike-slip"))
        ax.annotate(f"{n_label}\n{frac_txt}", (px, py), textcoords="offset points",
                    xytext=(0, -args.beach_size_km * 50 - 8), ha="center", fontsize=8, color="#333333",
                    zorder=9)

    legend_elems = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#2b6f8f", markeredgecolor="white",
               markersize=13, label="7U station"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=RAKE_CLASS_COLORS["thrust"],
               alpha=0.6, markersize=11, label="ensemble solution: thrust-like"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=RAKE_CLASS_COLORS["normal"],
               alpha=0.6, markersize=11, label="ensemble solution: normal-like"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=RAKE_CLASS_COLORS["strike-slip"],
               alpha=0.6, markersize=11, label="ensemble solution: strike-slip-like"),
        Line2D([0], [0], marker="o", color="none", markeredgecolor="black", markerfacecolor="none",
               markersize=11, label="single \"best-fit\" mechanism (out.csv)"),
        Line2D([0], [0], marker="o", color="0.6", markerfacecolor="0.4", markersize=5,
               label="true centroid (decluttered)"),
    ]
    ax.legend(handles=legend_elems, loc="upper left", fontsize=8.5, framealpha=0.9)

    ax.set_xlabel("East-west distance from array center (km)")
    ax.set_ylabel("North-south distance from array center (km)")
    ax.set_title(f"T1 array: RANGE of acceptable composite focal mechanisms per cluster\n"
                 f"({args.n_sample} of 500 SKHASH bootstrap solutions shown per cluster; "
                 f"polarity source: human-verified stack)", fontsize=11.5)
    ax.grid(True, alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=170, bbox_inches="tight")
    print(f"\nwrote {args.out_png}")


if __name__ == "__main__":
    main()
