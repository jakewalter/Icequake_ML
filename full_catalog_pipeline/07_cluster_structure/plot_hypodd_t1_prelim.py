#!/usr/bin/env python3
"""Preliminary look at the T1 pyocto ccscale_0.33 hypoDD relocation (the half of
the parallel T1 relocation that has finished; the QuakeMigrate-catalog run is
still in progress). Compares the pre-hypoDD pyocto v5 locations against the
post-hypoDD relocated ones for the same array, focused on the basal-ice-event
question motivating the whole exercise: does hypoDD's continuous-optimization
relocation confirm or remove the depth-banding seen near the ice-bed interface
in pyocto's own grid-search locations (see pyocto_full_catalog_rebuild memory)?

Note: this is an aggregate (catalog-level) comparison, not a per-event
before/after match — pyocto_events.csv is indexed by pyocto's own idx while
hypoDD.reloc uses its own numeric event IDs, and only 2164/2400 events survived
relocation. Good enough for a distributional look while we wait on the
QuakeMigrate-catalog run to finish.

Usage:
    python full_catalog_pipeline/plot_hypodd_t1_prelim.py
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy.geodetics import gps2dist_azimuth

import catalog_paths

PRE_CSV = "full_catalog_pipeline/artifacts/full_run/T1_v5/pyocto_events.csv"
RELOC = catalog_paths.reloc("T1")
STATION_SEL = catalog_paths.station_sel("T1")
OUT_DIR = catalog_paths.work_dir("T1")

ICE_BED_DEPTH_KM = 3.24  # BedMachine/Bedmap2 average, T1 (see pyocto_full_catalog_rebuild memory)

COLOR_PRE = "#8c96a6"    # matches plot_full_archive_locations.py's COLOR_QM (neutral gray)
COLOR_POST = "#15616d"   # matches COLOR_MATCHED
COLOR_STATION = "#2b2b2b"

RELOC_COLS = [
    "id", "lat", "lon", "depth", "x", "y", "z", "ex", "ey", "ez",
    "yr", "mo", "dy", "hr", "mi", "sc", "mag",
    "nccp", "nccs", "nctp", "ncts", "rcc", "rct", "cid",
]


def load_stations():
    stations = pd.read_csv(STATION_SEL, sep=r"\s+", header=None,
                            names=["id", "lat", "lon", "elev"])
    stations["code"] = stations["id"].str.split(".").str[1]
    return stations


def load_pre():
    df = pd.read_csv(PRE_CSV)
    return df.rename(columns={"latitude": "lat", "longitude": "lon"})


def load_post():
    return pd.read_csv(RELOC, sep=r"\s+", header=None, names=RELOC_COLS)


def dist_from_centroid(lat, lon, centroid_lat, centroid_lon):
    d_m, _, _ = gps2dist_azimuth(centroid_lat, centroid_lon, lat, lon)
    return d_m / 1000.0


def plot_map(pre, post, stations):
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(pre["lon"], pre["lat"], s=8, color=COLOR_PRE, alpha=0.25,
               label=f"pyocto pre-hypoDD (n={len(pre)})", zorder=1)
    ax.scatter(post["lon"], post["lat"], s=10, color=COLOR_POST, alpha=0.5,
               label=f"hypoDD relocated (n={len(post)})", zorder=2)
    ax.scatter(stations["lon"], stations["lat"], s=160, color=COLOR_STATION, marker="^",
               edgecolor="white", linewidth=1, label="T1 stations", zorder=3)
    for _, s in stations.iterrows():
        ax.annotate(s["code"], (s["lon"], s["lat"]), fontsize=9,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("T1: pyocto v5 locations before vs after hypoDD relocation\n(ccscale_0.33, QuakeMigrate-catalog run still in progress)", fontsize=13)
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    out = f"{OUT_DIR}/hypodd_reloc_map_prelim.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_depth_section(pre, post, stations):
    centroid_lat, centroid_lon = stations["lat"].mean(), stations["lon"].mean()
    pre_dist = [dist_from_centroid(la, lo, centroid_lat, centroid_lon) for la, lo in zip(pre["lat"], pre["lon"])]
    post_dist = [dist_from_centroid(la, lo, centroid_lat, centroid_lon) for la, lo in zip(post["lat"], post["lon"])]
    station_dist = [dist_from_centroid(la, lo, centroid_lat, centroid_lon) for la, lo in zip(stations["lat"], stations["lon"])]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, dist, depth, color, label, n in (
        (ax1, pre_dist, pre["depth"], COLOR_PRE, "pyocto pre-hypoDD", len(pre)),
        (ax2, post_dist, post["depth"], COLOR_POST, "hypoDD relocated", len(post)),
    ):
        ax.scatter(dist, depth, s=10, color=color, alpha=0.4, zorder=2)
        ax.scatter(station_dist, [0] * len(station_dist), s=140, color=COLOR_STATION, marker="^",
                   edgecolor="white", linewidth=1, zorder=3)
        ax.axhline(ICE_BED_DEPTH_KM, color="firebrick", linestyle="--", linewidth=1.5,
                   label=f"ice-bed interface ({ICE_BED_DEPTH_KM} km)", zorder=1)
        ax.set_xlabel("Distance from array centroid (km)")
        ax.set_title(f"{label} (n={n})", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=9, loc="lower right")
    ax1.invert_yaxis()
    ax1.set_ylabel("Depth (km)")
    fig.suptitle("T1 depth cross-section: pre- vs post-hypoDD (ccscale_0.33)", fontsize=14)
    fig.tight_layout()
    out = f"{OUT_DIR}/hypodd_reloc_depth_section_prelim.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_depth_histogram(pre, post):
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 6, 121)
    ax.hist(pre["depth"], bins=bins, color=COLOR_PRE, alpha=0.5, label=f"pyocto pre-hypoDD (n={len(pre)})", density=True)
    ax.hist(post["depth"], bins=bins, color=COLOR_POST, alpha=0.5, label=f"hypoDD relocated (n={len(post)})", density=True)
    ax.axvline(ICE_BED_DEPTH_KM, color="firebrick", linestyle="--", linewidth=1.5,
               label=f"ice-bed interface ({ICE_BED_DEPTH_KM} km)")
    ax.set_xlabel("Depth (km)")
    ax.set_ylabel("Density")
    ax.set_title("T1 depth distribution: does hypoDD confirm or remove the\npyocto grid-search banding near the ice-bed interface?", fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = f"{OUT_DIR}/hypodd_reloc_depth_histogram_prelim.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    stations = load_stations()
    pre = load_pre()
    post = load_post()
    print(f"pre-hypoDD (pyocto v5): {len(pre)} events")
    print(f"post-hypoDD (ccscale_0.33 relocated): {len(post)} events")

    outputs = [
        plot_map(pre, post, stations),
        plot_depth_section(pre, post, stations),
        plot_depth_histogram(pre, post),
    ]
    for o in outputs:
        print(f"wrote {o}")


if __name__ == "__main__":
    main()
