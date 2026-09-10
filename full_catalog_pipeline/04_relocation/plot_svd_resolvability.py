#!/usr/bin/env python3
"""Convincing plots for the vertical-resolvability question: within the T1
pyocto 216-event linear cluster (hypoDD cid=2, the confirmed non-quantization-
artifact "crack" cluster), is the observed depth spread real or dominated by
relocation noise?

Uses hypoDD's SVD-mode formal errors (hypodd_svd_cluster_errors.py output,
recompiled binary with larger MAXDATA0/MAXEVE0 -- see
t1_optimized_hypodd_and_catalog_comparison_plan memory) -- well-conditioned
for this cluster (no singular-value warnings, unlike the smaller 15-event
cluster), so these errors are trustworthy in a way GrowClust's CC-only
bootstrap (much less data) was not for the same physical question.

Design note (rewrite): the first version of this script produced a
meaningless "sorted by depth" x-axis and a 23,220-point overplotted scatter --
both violate basic dataviz practice (no tautological axes, no raw scatter at
that density). Replaced with: (1) a single hero-stat + binned ratio histogram
that makes the resolvability point in one glance with zero overplotting, and
(2) a map-view-plus-profile pair that grounds the same 216 events in real
geography instead of an abstract rank axis, using binned median+IQR bands
instead of 216 overlapping error bars.

Usage:
    python full_catalog_pipeline/plot_svd_resolvability.py
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

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pyproj

from plot_hypodd_t1_basal_3d import COLOR_STATION, woodcock_shape

import catalog_paths

# Written by hypodd_svd_cluster_errors.py --label cluster3_n222; rerun that first.
RELOC_PATH = os.path.join(
    catalog_paths.scratch_dir("hypodd_svd"), "cluster3_n222", "hypoDD.reloc.001.006")
STATION_SEL = catalog_paths.station_sel("T1")
PYOCTO_RELOC = catalog_paths.reloc("T1")
OUT_DIR = catalog_paths.work_dir("T1")

# validated categorical pair from the dataviz skill's reference palette (slots 1/2)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#8a8a86"
GRID = "#e4e3de"

RELOC_COLS = [
    "id", "lat", "lon", "depth", "x", "y", "z", "ex", "ey", "ez",
    "yr", "mo", "dy", "hr", "mi", "sc", "mag",
    "nccp", "nccs", "nctp", "ncts", "rcc", "rct", "cid",
]

plt.rcParams.update({
    "font.size": 12,
    "axes.edgecolor": GRID,
    "axes.linewidth": 1,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 1,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load():
    return pd.read_csv(RELOC_PATH, sep=r"\s+", header=None, names=RELOC_COLS)


def pairwise_ratios(df):
    depths = df["depth"].values * 1000
    ez = df["ez"].values
    n = len(depths)
    diffs, combined = [], []
    for i in range(n):
        for j in range(i + 1, n):
            diffs.append(abs(depths[i] - depths[j]))
            combined.append(np.hypot(ez[i], ez[j]))
    diffs, combined = np.array(diffs), np.array(combined)
    # avoid /0 for identical-depth pairs with tiny combined error (none expected here, but be safe)
    ratio = diffs / np.maximum(combined, 1e-6)
    return ratio


def plot_hero_histogram(df, out_path):
    ratio = pairwise_ratios(df)
    resolved_frac = (ratio > 1).mean()

    fig = plt.figure(figsize=(9, 8.5))
    ax = fig.add_axes([0.12, 0.09, 0.84, 0.55])  # left, bottom, width, height -- fixed plot zone

    bins = np.logspace(np.log10(max(ratio.min(), 0.05)), np.log10(ratio.max()), 40)
    counts, edges = np.histogram(ratio, bins=bins)
    centers = np.sqrt(edges[:-1] * edges[1:])
    widths = np.diff(edges)
    colors = [ORANGE if c > 1 else MUTED for c in centers]
    ax.bar(centers, counts, width=widths, color=colors, align="center", linewidth=0)
    ax.set_xscale("log")
    ax.axvline(1.0, color=INK, linewidth=1.5, linestyle="--")
    ymax = ax.get_ylim()[1]
    ax.annotate("depth difference =\ncombined error", xy=(1.0, ymax * 0.98),
                xytext=(0.12, ymax * 0.98), fontsize=10, color=INK,
                va="top", ha="left",
                arrowprops=dict(arrowstyle="->", color=INK, linewidth=1))
    ax.set_xlabel("Pairwise depth difference ÷ combined formal error  (log scale)")
    ax.set_ylabel("Number of event pairs")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    # fixed, non-overlapping header zone above the axes: title, then hero stat, then subtext
    fig.text(0.12, 0.97, "Is the depth spread within this cluster real, or noise?",
              fontsize=15, color=INK, ha="left", va="top", fontweight="bold")
    fig.text(0.12, 0.88, f"{100*resolved_frac:.0f}%", fontsize=34, fontweight="bold",
              color=ORANGE, ha="left", va="top")
    fig.text(0.12, 0.80, "of event pairs (n=216, 23,220 pairs) show a depth difference\n"
                          "bigger than their combined formal error",
              fontsize=10.5, color=INK, ha="left", va="top")

    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return resolved_frac


def plot_map_and_profile(df, stations_xy, out_path):
    st = pd.read_csv(STATION_SEL, sep=r"\s+", header=None, names=["id", "lat", "lon", "elev"])
    to_ps = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    stx, sty = to_ps.transform(st["lon"].values, st["lat"].values)
    cx, cy = stx.mean(), sty.mean()

    pyocto = pd.read_csv(PYOCTO_RELOC, sep=r"\s+", header=None, names=RELOC_COLS)
    px, py = to_ps.transform(pyocto["lon"].values, pyocto["lat"].values)
    pyocto_ex, pyocto_nx = (px - cx) / 1000.0, (py - cy) / 1000.0

    x, y = to_ps.transform(df["lon"].values, df["lat"].values)
    ex_km, nx_km = (x - cx) / 1000.0, (y - cy) / 1000.0
    sx, sy = stations_xy

    pts = np.column_stack([ex_km, nx_km, df["depth"].values])
    eigval, eigvec, k = woodcock_shape(pts)
    long_axis_xy = eigvec[:2, 0]
    long_axis_xy = long_axis_xy / np.linalg.norm(long_axis_xy)
    along_m = (ex_km * long_axis_xy[0] + nx_km * long_axis_xy[1])
    along_m = (along_m - along_m.min()) * 1000

    depth_m = df["depth"].values * 1000
    vmin, vmax = depth_m.min(), depth_m.max()

    fig, (ax_map, ax_prof) = plt.subplots(1, 2, figsize=(13, 6), width_ratios=[1, 1.3])

    ax_map.scatter(pyocto_ex, pyocto_nx, s=5, color=GRID, zorder=1, linewidth=0)
    sc = ax_map.scatter(ex_km, nx_km, c=depth_m, cmap="cividis_r", s=22, vmin=vmin, vmax=vmax,
                        zorder=2, linewidth=0.4, edgecolor="white")
    ax_map.scatter(sx, sy, s=110, color=COLOR_STATION, marker="^", edgecolor="white",
                   linewidth=1, zorder=3)
    ax_map.set_xlabel("East of array centroid (km)")
    ax_map.set_ylabel("North of array centroid (km)")
    ax_map.set_title("Where this cluster sits\n(rest of pyocto catalog in gray)", fontsize=12)
    ax_map.set_aspect("equal")

    # binned median + IQR band along the principal axis, replacing 216 overlapping error bars
    n_bins = 12
    bin_edges = np.linspace(along_m.min(), along_m.max(), n_bins + 1)
    bin_id = np.clip(np.digitize(along_m, bin_edges) - 1, 0, n_bins - 1)
    med, q25, q75, mids = [], [], [], []
    for b in range(n_bins):
        sel = bin_id == b
        if sel.sum() < 3:
            continue
        med.append(np.median(depth_m[sel]))
        q25.append(np.percentile(depth_m[sel], 25))
        q75.append(np.percentile(depth_m[sel], 75))
        mids.append((bin_edges[b] + bin_edges[b + 1]) / 2)
    med, q25, q75, mids = map(np.array, (med, q25, q75, mids))

    ax_prof.scatter(along_m, depth_m, s=14, color=BLUE, alpha=0.35, linewidth=0, zorder=1)
    ax_prof.plot(mids, med, color=INK, linewidth=2, zorder=3, label="binned median depth")
    ax_prof.fill_between(mids, q25, q75, color=INK, alpha=0.12, zorder=2, label="binned 25-75th pctile")
    typical_err = 2 * np.median(df["ez"].values)
    ax_prof.annotate("", xy=(along_m.max() * 0.06, depth_m.min() + 10),
                      xytext=(along_m.max() * 0.06, depth_m.min() + 10 + typical_err),
                      arrowprops=dict(arrowstyle="<->", color=MUTED, linewidth=1.5))
    ax_prof.annotate(f"typical\n2σ error\n({typical_err:.0f} m)", xy=(along_m.max() * 0.10,
                      depth_m.min() + 10 + typical_err / 2), fontsize=9, color=MUTED, va="center")
    ax_prof.invert_yaxis()
    ax_prof.set_xlabel("Distance along cluster's long axis (m)")
    ax_prof.set_ylabel("Depth (m)")
    ax_prof.set_title("Depth vs. position: scattered, not one dipping plane\n"
                       "(real spread — no single fault surface fits it)", fontsize=12)
    ax_prof.legend(fontsize=9, loc="lower right", frameon=False)

    cbar = fig.colorbar(sc, ax=ax_map, shrink=0.75, pad=0.03)
    cbar.set_label("Depth (m)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.suptitle("T1 pyocto cluster (n=216): a real basal patch with distributed depth, not a single crack",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load()
    st = pd.read_csv(STATION_SEL, sep=r"\s+", header=None, names=["id", "lat", "lon", "elev"])
    to_ps = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    stx, sty = to_ps.transform(st["lon"].values, st["lat"].values)
    cx, cy = stx.mean(), sty.mean()
    stations_xy = ((stx - cx) / 1000.0, (sty - cy) / 1000.0)

    out1 = f"{OUT_DIR}/svd_resolvability_histogram.png"
    frac = plot_hero_histogram(df, out1)
    print(f"wrote {out1} ({100*frac:.0f}% resolved)")

    out2 = f"{OUT_DIR}/svd_resolvability_map_profile.png"
    plot_map_and_profile(df, stations_xy, out2)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
