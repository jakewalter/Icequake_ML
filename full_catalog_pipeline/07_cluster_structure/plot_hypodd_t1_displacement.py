#!/usr/bin/env python3
"""Per-event before/after analysis of the T1 pyocto-v5 -> hypoDD (ccscale_0.33)
relocation: how far did individual events actually move, and why.

Unlike plot_hypodd_t1_prelim.py (aggregate distributions only, pre and post
compared as unmatched populations), this matches each relocated event back to
its specific pyocto input event and computes a real per-event displacement.

The match key: hypoDDpy builds event.dat/event.sel by iterating pyocto_events.csv
in row order and assigning sequential IDs starting at 1, so hypoDD's numeric
event ID == pyocto CSV `idx` + 1. Verified directly (see session notes) by
checking the first 10 relocated events' pre-hypoDD lat/lon/depth against
pyocto_events.csv at idx = id - 1: values match to the precision written in
event.sel. All 2165 relocated IDs fall inside the source 0..2399 idx range, so
the join is exhaustive, not a lucky subset.

Usage:
    python full_catalog_pipeline/plot_hypodd_t1_displacement.py
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

ICE_BED_DEPTH_KM = 3.24
# pyocto v5's two known residual grid-search artifact depths (see
# pyocto_full_catalog_rebuild memory): a shared degenerate flat-loss-surface
# node, and a travel-time degeneracy at the (widened but not eliminated)
# ice-bed velocity step.
ARTIFACT_FLAT_NODE_KM = 2.507
ARTIFACT_VELOCITY_STEP_KM = 3.204

COLOR_PRE = "#8c96a6"       # matches this repo's existing COLOR_QM convention
COLOR_POST = "#15616d"      # matches this repo's existing COLOR_MATCHED
COLOR_STATION = "#2b2b2b"
COLOR_CONNECTOR = "#c3c2b7"
CMAP_DISPLACEMENT = "Blues"  # sequential, single hue, light->dark = near->far

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


def load_matched():
    pre = pd.read_csv(PRE_CSV).rename(columns={"latitude": "lat", "longitude": "lon"})
    post = pd.read_csv(RELOC, sep=r"\s+", header=None, names=RELOC_COLS)
    post["pyocto_idx"] = post["id"] - 1
    m = post.merge(pre.set_index("idx"), left_on="pyocto_idx", right_index=True,
                    suffixes=("_post", "_pre"))
    assert len(m) == len(post), (
        f"expected every relocated event to match its pyocto source "
        f"({len(post)} post events, {len(m)} matched)")

    horiz_km = np.array([
        gps2dist_azimuth(a, b, c, d)[0] / 1000.0
        for a, b, c, d in zip(m["lat_pre"], m["lon_pre"], m["lat_post"], m["lon_post"])
    ])
    m["horiz_km"] = horiz_km
    m["dz_km"] = m["depth_post"] - m["depth_pre"]
    m["dist3d_km"] = np.sqrt(m["horiz_km"] ** 2 + m["dz_km"] ** 2)
    return m


def dist_from_centroid(lat, lon, centroid_lat, centroid_lon):
    d_m, _, _ = gps2dist_azimuth(centroid_lat, centroid_lon, lat, lon)
    return d_m / 1000.0


def plot_displacement_map(m, stations):
    fig, ax = plt.subplots(figsize=(9.5, 8))
    norm = plt.Normalize(vmin=0, vmax=np.percentile(m["dist3d_km"], 95))
    for _, r in m.iterrows():
        ax.plot([r["lon_pre"], r["lon_post"]], [r["lat_pre"], r["lat_post"]],
                color=COLOR_CONNECTOR, linewidth=0.5, alpha=0.5, zorder=1)
    sc = ax.scatter(m["lon_post"], m["lat_post"], s=14, c=m["dist3d_km"], cmap=CMAP_DISPLACEMENT,
                     norm=norm, alpha=0.85, zorder=2, edgecolor="none")
    ax.scatter(stations["lon"], stations["lat"], s=160, color=COLOR_STATION, marker="^",
               edgecolor="white", linewidth=1, label="T1 stations", zorder=3)
    for _, s in stations.iterrows():
        ax.annotate(s["code"], (s["lon"], s["lat"]), fontsize=9,
                    xytext=(4, 4), textcoords="offset points")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("3D displacement, pyocto -> hypoDD (km)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"T1: per-event relocation displacement (pyocto v5 -> hypoDD ccscale_0.33)\n"
        f"n={len(m)}, median 3D displacement {m['dist3d_km'].median():.2f} km "
        f"(gray lines connect each event's pre- and post-hypoDD position)",
        fontsize=12, pad=12)
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = f"{OUT_DIR}/hypodd_reloc_displacement_map.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_depth_dumbbell(m, stations):
    centroid_lat, centroid_lon = stations["lat"].mean(), stations["lon"].mean()
    pre_dist = np.array([dist_from_centroid(la, lo, centroid_lat, centroid_lon)
                          for la, lo in zip(m["lat_pre"], m["lon_pre"])])
    post_dist = np.array([dist_from_centroid(la, lo, centroid_lat, centroid_lon)
                           for la, lo in zip(m["lat_post"], m["lon_post"])])
    station_dist = [dist_from_centroid(la, lo, centroid_lat, centroid_lon)
                     for la, lo in zip(stations["lat"], stations["lon"])]

    fig, ax = plt.subplots(figsize=(13, 7))
    for pd_, dpre, ppd, dpost in zip(pre_dist, m["depth_pre"], post_dist, m["depth_post"]):
        ax.plot([pd_, ppd], [dpre, dpost], color=COLOR_CONNECTOR, linewidth=0.4,
                 alpha=0.35, zorder=1)
    ax.scatter(pre_dist, m["depth_pre"], s=8, color=COLOR_PRE, alpha=0.35,
               label=f"pyocto pre-hypoDD (n={len(m)})", zorder=2)
    ax.scatter(post_dist, m["depth_post"], s=10, color=COLOR_POST, alpha=0.6,
               label=f"hypoDD relocated (n={len(m)})", zorder=3)
    ax.scatter(station_dist, [0] * len(station_dist), s=140, color=COLOR_STATION, marker="^",
               edgecolor="white", linewidth=1, zorder=4)
    ax.axhline(ICE_BED_DEPTH_KM, color="firebrick", linestyle="--", linewidth=1.5,
               label=f"ice-bed interface ({ICE_BED_DEPTH_KM} km)", zorder=1)
    ax.axhline(ARTIFACT_FLAT_NODE_KM, color="#eb6834", linestyle=":", linewidth=1.2,
               label=f"pyocto flat-node artifact ({ARTIFACT_FLAT_NODE_KM} km)", zorder=1)
    ax.axhline(ARTIFACT_VELOCITY_STEP_KM, color="#eda100", linestyle=":", linewidth=1.2,
               label=f"pyocto velocity-step artifact ({ARTIFACT_VELOCITY_STEP_KM} km)", zorder=1)
    ax.invert_yaxis()
    ax.set_xlabel("Distance from array centroid (km)")
    ax.set_ylabel("Depth (km)")
    ax.set_title("T1 depth cross-section: every event's pyocto -> hypoDD move (ccscale_0.33)",
                 fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    out = f"{OUT_DIR}/hypodd_reloc_depth_dumbbell.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_displacement_vs_predepth(m):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(m["depth_pre"], m["dist3d_km"], s=8, color=COLOR_POST, alpha=0.35, zorder=2)
    ax.axvline(ARTIFACT_FLAT_NODE_KM, color="#eb6834", linestyle=":", linewidth=1.5,
               label=f"flat-node artifact ({ARTIFACT_FLAT_NODE_KM} km)", zorder=1)
    ax.axvline(ARTIFACT_VELOCITY_STEP_KM, color="#eda100", linestyle=":", linewidth=1.5,
               label=f"velocity-step artifact ({ARTIFACT_VELOCITY_STEP_KM} km)", zorder=1)
    ax.axvline(ICE_BED_DEPTH_KM, color="firebrick", linestyle="--", linewidth=1.2,
               label=f"ice-bed interface ({ICE_BED_DEPTH_KM} km)", zorder=1)
    ax.set_xlabel("Pre-hypoDD (pyocto) depth (km)")
    ax.set_ylabel("3D displacement to hypoDD location (km)")
    ax.set_title(
        "Why events moved: displacement spikes at pyocto's own known\n"
        "grid-search artifact depths, not a uniform shift", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    out = f"{OUT_DIR}/hypodd_reloc_displacement_vs_predepth.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_displacement_histogram(m):
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, np.percentile(m["dist3d_km"], 99), 60)
    for col, label, color in (
        ("horiz_km", "horizontal", "#2a78d6"),
        ("dz_km", "vertical (|depth change|)", "#eb6834"),
        ("dist3d_km", "3D", "#15616d"),
    ):
        vals = m[col].abs() if col == "dz_km" else m[col]
        ax.hist(vals, bins=bins, histtype="step", linewidth=2, color=color,
                label=f"{label} (median {vals.median():.2f} km)")
    ax.set_xlabel("Displacement (km)")
    ax.set_ylabel("Event count")
    ax.set_title(f"T1: relocation displacement magnitude, pyocto v5 -> hypoDD (n={len(m)})",
                 fontsize=13)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = f"{OUT_DIR}/hypodd_reloc_displacement_histogram.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    stations = load_stations()
    m = load_matched()
    print(f"matched {len(m)} relocated events back to their pyocto source")
    print(f"median 3D displacement: {m['dist3d_km'].median():.3f} km "
          f"(mean {m['dist3d_km'].mean():.3f}, p90 {m['dist3d_km'].quantile(0.9):.3f}, "
          f"max {m['dist3d_km'].max():.3f})")

    outputs = [
        plot_displacement_map(m, stations),
        plot_depth_dumbbell(m, stations),
        plot_displacement_vs_predepth(m),
        plot_displacement_histogram(m),
    ]
    for o in outputs:
        print(f"wrote {o}")


if __name__ == "__main__":
    main()
