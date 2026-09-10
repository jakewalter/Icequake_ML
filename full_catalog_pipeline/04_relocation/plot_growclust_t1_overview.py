#!/usr/bin/env python3
"""T1 GrowClust overview plots: map view, depth cross-section, and cluster-size
distribution for both catalogs, making the relocated-vs-singleton split
explicit. GrowClust uses ONLY cross-correlation (dt.cc) differential times --
unlike hypoDD, it never touches catalog-pick (dt.ct) differential times at
all -- so any event without enough CC linkage to another event stays a
"singleton" (nbranch=1, unmoved from its original location). This is expected
behavior, not a bug: pyocto's CC data is dense (114,364 observations / 2,396
events), so most events (81.8%) get relocated; QuakeMigrate's CC data is much
sparser (11,653 / 9,814, largely because dt.ct dominates that catalog's
original hypoDD linkage and was never used here), so only 20.4% do.

Usage:
    python full_catalog_pipeline/plot_growclust_t1_overview.py
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
import pyproj

from plot_hypodd_t1_basal_3d import PYOCTO_STA, ICE_BED_DEPTH_KM, COLOR_STATION

CAT_COLS = [
    "yr", "mo", "dy", "hr", "mi", "sec", "id", "lat", "lon", "depth", "mag",
    "iq", "cluster", "nbranch", "npair", "ndiffP", "ndiffS", "rmsP", "rmsS",
    "madh", "madz", "madt", "lat_orig", "lon_orig", "depth_orig",
]

SOURCES = {
    "pyocto": "full_catalog_pipeline/artifacts/full_run/T1_v5/growclust_pyocto/OUT/out.growclust_cat",
    "qm": "full_catalog_pipeline/artifacts/full_run/T1_v5/growclust_qm/OUT/out.growclust_cat",
}
LABELS = {"pyocto": "pyocto catalog", "qm": "QuakeMigrate catalog"}
OUT_DIR = "full_catalog_pipeline/artifacts/full_run/T1_v5"

COLOR_RELOC = "#15616d"
COLOR_SINGLE = "#c6c6c6"


def load_cat(path, to_ps, cx, cy):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=CAT_COLS)
    x, y = to_ps.transform(df["lon"].values, df["lat"].values)
    df["ex"] = (x - cx) / 1000.0
    df["nx"] = (y - cy) / 1000.0
    return df


def plot_map(df, tag, stations_xy, out_path):
    reloc = df[df["nbranch"] >= 2]
    single = df[df["nbranch"] < 2]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(single["ex"], single["nx"], s=6, color=COLOR_SINGLE, alpha=0.4,
               label=f"unrelocated singletons (n={len(single)}, {100*len(single)/len(df):.0f}%)",
               zorder=1)
    ax.scatter(reloc["ex"], reloc["nx"], s=8, color=COLOR_RELOC, alpha=0.55,
               label=f"relocated, nbranch>=2 (n={len(reloc)}, {100*len(reloc)/len(df):.0f}%)",
               zorder=2)
    sx, sy = stations_xy
    ax.scatter(sx, sy, s=160, color=COLOR_STATION, marker="^", edgecolor="white",
               linewidth=1, label="T1 stations", zorder=3)
    ax.set_xlabel("East of centroid (km)")
    ax.set_ylabel("North of centroid (km)")
    ax.set_aspect("equal")
    ax.set_title(f"T1 {LABELS[tag]}: GrowClust map view\n"
                 f"(CC-only relocation -- events without enough cross-correlation "
                 f"linkage stay unmoved)")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_depth_section(df, tag, stations_xy, out_path):
    reloc = df[df["nbranch"] >= 2]
    single = df[df["nbranch"] < 2]
    sx, sy = stations_xy
    station_dist = np.hypot(sx, sy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True, sharex=True)
    for ax, sub, color, label in (
        (ax1, single, COLOR_SINGLE, "unrelocated singletons"),
        (ax2, reloc, COLOR_RELOC, "relocated (nbranch>=2)"),
    ):
        r = np.hypot(sub["ex"], sub["nx"])
        ax.scatter(r, sub["depth"], s=8, color=color, alpha=0.4)
        ax.scatter(station_dist, np.zeros_like(station_dist), s=100, color=COLOR_STATION,
                   marker="^", zorder=3)
        ax.axhline(ICE_BED_DEPTH_KM, color="#888888", linestyle="--", linewidth=1,
                   label=f"ice-bed interface ({ICE_BED_DEPTH_KM} km)")
        ax.set_xlabel("Distance from array centroid (km)")
        ax.set_title(f"{label} (n={len(sub)})")
        ax.legend(fontsize=8)
    ax1.invert_yaxis()
    ax1.set_ylabel("Depth (km)")
    fig.suptitle(f"T1 {LABELS[tag]}: GrowClust depth section")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_nbranch_histogram(df, tag, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    max_nb = df["nbranch"].max()
    bins = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, max(max_nb + 1, 501)]
    counts, edges = np.histogram(df["nbranch"], bins=bins)
    labels = [f"{edges[i]:.0f}-{edges[i+1]-1:.0f}" if edges[i + 1] - edges[i] > 1
              else f"{edges[i]:.0f}" for i in range(len(edges) - 1)]
    ax.bar(range(len(counts)), counts, color=COLOR_RELOC)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.set_xlabel("Events per GrowClust cluster (nbranch)")
    ax.set_ylabel("Number of events (log scale)")
    n_single = (df["nbranch"] == 1).sum()
    n_reloc = (df["nbranch"] >= 2).sum()
    ax.set_title(f"T1 {LABELS[tag]}: cluster-size distribution\n"
                 f"{n_single} singletons ({100*n_single/len(df):.0f}%), "
                 f"{n_reloc} relocated ({100*n_reloc/len(df):.0f}%), n_total={len(df)}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_relocated_fraction_summary(cats, out_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    tags = list(cats.keys())
    fracs = [100 * (cats[t]["nbranch"] >= 2).mean() for t in tags]
    totals = [len(cats[t]) for t in tags]
    bars = ax.bar([LABELS[t] for t in tags], fracs, color=[COLOR_RELOC, "#eb6834"])
    for bar, frac, total in zip(bars, fracs, totals):
        n_reloc = int(round(frac / 100 * total))
        ax.annotate(f"{n_reloc}/{total}\n({frac:.1f}%)", (bar.get_x() + bar.get_width() / 2, frac),
                    ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Relocated fraction (%)")
    ax.set_ylim(0, 100)
    ax.set_title("T1: fraction of events GrowClust actually relocated\n"
                 "(CC-only method -- limited by cross-correlation linkage density, not a failure)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    to_ps = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    sta = pd.read_csv(PYOCTO_STA, sep=r"\s+", header=None, names=["id", "lat", "lon", "elev"])
    st_x, st_y = to_ps.transform(sta["lon"].values, sta["lat"].values)
    cx, cy = st_x.mean(), st_y.mean()
    stations_xy = ((st_x - cx) / 1000.0, (st_y - cy) / 1000.0)

    cats = {}
    for tag, path in SOURCES.items():
        df = load_cat(path, to_ps, cx, cy)
        cats[tag] = df
        n_single = (df["nbranch"] == 1).sum()
        n_reloc = (df["nbranch"] >= 2).sum()
        print(f"{tag}: {len(df)} total, {n_reloc} relocated ({100*n_reloc/len(df):.1f}%), "
              f"{n_single} singletons ({100*n_single/len(df):.1f}%)")

        out_map = f"{OUT_DIR}/growclust_{tag}/growclust_{tag}_overview_map.png"
        plot_map(df, tag, stations_xy, out_map)
        print(f"wrote {out_map}")

        out_section = f"{OUT_DIR}/growclust_{tag}/growclust_{tag}_overview_depth_section.png"
        plot_depth_section(df, tag, stations_xy, out_section)
        print(f"wrote {out_section}")

        out_hist = f"{OUT_DIR}/growclust_{tag}/growclust_{tag}_nbranch_histogram.png"
        plot_nbranch_histogram(df, tag, out_hist)
        print(f"wrote {out_hist}")

    out_summary = f"{OUT_DIR}/growclust_relocated_fraction_summary.png"
    plot_relocated_fraction_summary(cats, out_summary)
    print(f"wrote {out_summary}")


if __name__ == "__main__":
    main()
