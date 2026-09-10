#!/usr/bin/env python3
"""Comprehensive T1-vs-T2 depth comparison across BOTH association methods
(pyocto grid-search association, QuakeMigrate) and both refinement stages
(pre-hypoDD raw locations, post-hypoDD ccscale_0.33 relocation).

Extends plot_t1_t2_thickness_comparison.py (pyocto-only, T2 pre-hypoDD-only)
now that both sites have a QuakeMigrate catalog and both have a completed
hypoDD relocation using the same validated ccscale_0.33 reweighting scheme
(see hypodd_cc_heldout_cv_validation / hypodd_relocate.py's REWEIGHTING_SCHEME
fix 2026-08-11), with each site's velocity model referenced to its own
BedMachine/Bedmap2 ice-bed depth (T1=3.24km, T2=2.02km).

Eight series, laid out as a 2 (site) x 4 (pyocto-pre, pyocto-post, QM-pre,
QM-post) grid, plus a summary panel overlaying just the four
best-available-per-site-and-source post-hypoDD/raw-fallback curves.

Color follows SITE (identity), not source or stage: T1 = blue, T2 = orange,
consistent with the rest of this pipeline's figures (plot_svd_resolvability.py,
plot_t1_t2_thickness_comparison.py). Pre-hypoDD panels are drawn with a hatch
and lower alpha; post-hypoDD panels are solid. Source (pyocto vs QuakeMigrate)
and stage (pre vs post) are encoded by panel position/label, not color, so no
new hues are needed for an 8-series comparison.

Usage:
    python full_catalog_pipeline/plot_t1_t2_qm_depth_comparison.py
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

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

import catalog_paths

T1_PYOCTO_PRE_CSV = "full_catalog_pipeline/artifacts/full_run/T1_v5/pyocto_events.csv"
T1_PYOCTO_POST_RELOC = catalog_paths.reloc("T1")
T1_QM_PRE_JSON = "/scratch2/qm/t1/output/working_files/events.json"
T1_QM_POST_RELOC = "/scratch2/qm/t1/output/hypodd_optimized/input_files/hypoDD.reloc"

T2_PYOCTO_PRE_CSV = "full_catalog_pipeline/artifacts/full_run/T2_v5/pyocto_events.csv"
T2_PYOCTO_POST_RELOC = catalog_paths.reloc("T2")
T2_QM_PRE_JSON = "/scratch2/qm/t2/quakeml/output/working_files/events.json"
T2_QM_POST_RELOC = "/scratch2/qm/t2/quakeml/output/hypodd_optimized/input_files/hypoDD.reloc"

OUT_DIR = "full_catalog_pipeline/artifacts/full_run"
OUT_PNG = f"{OUT_DIR}/t1_t2_qm_depth_comparison.png"
OUT_SUMMARY_PNG = f"{OUT_DIR}/t1_t2_qm_depth_comparison_summary.png"
OUT_CSV = f"{OUT_DIR}/t1_t2_qm_depth_comparison_stats.csv"

T1_BED_KM = 3.24
T2_BED_KM = 2.02

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

# (0.05, 5.5) km drops events piled at pyocto's grid-search box edges (see
# plot_t1_t2_thickness_comparison.py) and, for QuakeMigrate, the analogous
# implausible near-zero/negative and very-deep locate-grid-edge events --
# hypoDD's own IAQ=1 already excludes the near-surface ones post-relocation,
# so this filter only ever changes the *_pre series.
EDGE_MIN_KM, EDGE_MAX_KM = 0.05, 5.5

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


def _drop_edges(depth_km, label):
    n0 = len(depth_km)
    kept = depth_km[(depth_km > EDGE_MIN_KM) & (depth_km < EDGE_MAX_KM)]
    print(f"  {label}: dropped box/grid-edge events {n0} -> {len(kept)}")
    return kept


def load_pyocto_csv(path, bed_km, label, is_pre):
    depth_km = pd.read_csv(path)["depth"].values
    if is_pre:
        depth_km = _drop_edges(depth_km, label)
    return (bed_km - depth_km) * 1000.0


def load_reloc(path, bed_km, label):
    reloc = pd.read_csv(path, sep=r"\s+", header=None, names=RELOC_COLS)
    depth_km = reloc["depth"].values
    return (bed_km - depth_km) * 1000.0


def load_qm_json(path, bed_km, label):
    with open(path) as f:
        events = json.load(f)
    depth_km = np.array([e["origin_depth"] for e in events], dtype=float) / 1000.0
    depth_km = _drop_edges(depth_km, label)
    return (bed_km - depth_km) * 1000.0


def load_all():
    print("loading T1...")
    t1 = {
        "pyocto_pre": load_pyocto_csv(T1_PYOCTO_PRE_CSV, T1_BED_KM, "T1 pyocto pre", is_pre=True),
        "pyocto_post": load_reloc(T1_PYOCTO_POST_RELOC, T1_BED_KM, "T1 pyocto post"),
        "qm_pre": load_qm_json(T1_QM_PRE_JSON, T1_BED_KM, "T1 QM pre"),
        "qm_post": load_reloc(T1_QM_POST_RELOC, T1_BED_KM, "T1 QM post"),
    }
    print("loading T2...")
    t2 = {
        "pyocto_pre": load_pyocto_csv(T2_PYOCTO_PRE_CSV, T2_BED_KM, "T2 pyocto pre", is_pre=True),
        "pyocto_post": load_reloc(T2_PYOCTO_POST_RELOC, T2_BED_KM, "T2 pyocto post"),
        "qm_pre": load_qm_json(T2_QM_PRE_JSON, T2_BED_KM, "T2 QM pre"),
        "qm_post": load_reloc(T2_QM_POST_RELOC, T2_BED_KM, "T2 QM post"),
    }
    return {"T1": t1, "T2": t2}


def summarize(data):
    rows = []
    for site, series in data.items():
        for key, above_bed_m in series.items():
            rows.append({
                "site": site,
                "catalog": key,
                "n": len(above_bed_m),
                "median_m": np.median(above_bed_m),
                "iqr_m": np.percentile(above_bed_m, 75) - np.percentile(above_bed_m, 25),
                "p05_m": np.percentile(above_bed_m, 5),
                "p95_m": np.percentile(above_bed_m, 95),
                "p90_span_m": np.percentile(above_bed_m, 95) - np.percentile(above_bed_m, 5),
                "frac_above_200m": (above_bed_m > 200).mean(),
                "frac_above_500m": (above_bed_m > 500).mean(),
            })
    return pd.DataFrame(rows)


def plot_grid(data, stats, out_path):
    cols = ["pyocto_pre", "pyocto_post", "qm_pre", "qm_post"]
    col_titles = ["pyocto\n(pre-hypoDD)", "pyocto\n(post-hypoDD, ccscale_0.33)",
                  "QuakeMigrate\n(pre-hypoDD)", "QuakeMigrate\n(post-hypoDD, ccscale_0.33)"]
    sites = ["T1", "T2"]
    site_color = {"T1": BLUE, "T2": ORANGE}
    xlim = (-800, 2200)
    bins = np.linspace(xlim[0], xlim[1], 55)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey="row", sharex=True)
    for r, site in enumerate(sites):
        for c, key in enumerate(cols):
            ax = axes[r, c]
            above_bed_m = data[site][key]
            row = stats[(stats.site == site) & (stats.catalog == key)].iloc[0]
            is_pre = key.endswith("_pre")
            ax.hist(above_bed_m, bins=bins, color=site_color[site],
                     alpha=0.45 if is_pre else 0.85,
                     hatch="///" if is_pre else None,
                     edgecolor=site_color[site] if is_pre else "none", linewidth=0.8)
            ax.axvline(0, color=INK, linewidth=1.3, linestyle="--")
            ax.axvline(row["median_m"], color=INK, linewidth=1.1, linestyle="-")
            ax.set_xlim(xlim)
            ax.set_title(f"n={row['n']:.0f}\nmedian={row['median_m']:.0f}m  IQR={row['iqr_m']:.0f}m",
                          fontsize=9.5)
            if r == 1:
                ax.set_xlabel("Height above ice-bed interface (m)")
            if c == 0:
                ax.set_ylabel(f"{site}\nNumber of events", fontsize=11, fontweight="bold")

    for c, title in enumerate(col_titles):
        axes[0, c].annotate(title, xy=(0.5, 1.22), xycoords="axes fraction",
                             ha="center", fontsize=11.5, fontweight="bold")

    legend_handles = [
        Patch(facecolor=BLUE, alpha=0.85, label="T1"),
        Patch(facecolor=ORANGE, alpha=0.85, label="T2"),
        Patch(facecolor=MUTED, alpha=0.45, hatch="///", edgecolor=MUTED, label="pre-hypoDD (raw)"),
        Patch(facecolor=MUTED, alpha=0.85, label="post-hypoDD (refined)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=10.5)

    fig.suptitle("Vertical extent of icequake-occupied ice above bed: T1 vs T2, "
                  "pyocto vs QuakeMigrate, pre vs post hypoDD",
                  fontsize=15, fontweight="bold", y=1.0)
    fig.text(0.5, 0.965,
              "Dashed line = ice-bed interface (0 m); solid line = median. Both sites' post-hypoDD "
              "runs use the identical validated ccscale_0.33 reweighting scheme.",
              ha="center", fontsize=10, color=MUTED)
    fig.tight_layout(rect=[0, 0.03, 1, 0.88])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary(data, stats, out_path):
    """Overlay the four post-hypoDD (best-available) curves as smooth step
    density curves, for direct cross-site/cross-source comparison."""
    fig, ax = plt.subplots(figsize=(10, 6.5))
    xlim = (-800, 2200)
    bins = np.linspace(xlim[0], xlim[1], 70)
    specs = [
        ("T1", "pyocto_post", BLUE, "-", "T1 pyocto"),
        ("T1", "qm_post", BLUE, "--", "T1 QuakeMigrate"),
        ("T2", "pyocto_post", ORANGE, "-", "T2 pyocto"),
        ("T2", "qm_post", ORANGE, "--", "T2 QuakeMigrate"),
    ]
    for site, key, color, ls, label in specs:
        above_bed_m = data[site][key]
        row = stats[(stats.site == site) & (stats.catalog == key)].iloc[0]
        counts, edges = np.histogram(above_bed_m, bins=bins, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, counts, color=color, linestyle=ls, linewidth=2.2,
                 label=f"{label} (n={row['n']:.0f}, median={row['median_m']:.0f}m)")
    ax.axvline(0, color=INK, linewidth=1.3, linestyle=":")
    ax.set_xlim(xlim)
    ax.set_xlabel("Height above ice-bed interface (m)")
    ax.set_ylabel("Probability density")
    ax.legend(frameon=False, fontsize=10.5)
    ax.set_title("Post-hypoDD depth distributions, both sites and both association methods",
                  fontsize=13.5, fontweight="bold")
    fig.text(0.5, -0.02,
              "Dotted line = ice-bed interface (0 m). All four curves use the same ccscale_0.33 "
              "hypoDD reweighting scheme, referenced to each site's own BedMachine/Bedmap2 bed depth.",
              ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    data = load_all()
    stats = summarize(data)
    pd.set_option("display.width", 200)
    print(stats.to_string(index=False))
    stats.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")
    plot_grid(data, stats, OUT_PNG)
    print(f"wrote {OUT_PNG}")
    plot_summary(data, stats, OUT_SUMMARY_PNG)
    print(f"wrote {OUT_SUMMARY_PNG}")


if __name__ == "__main__":
    main()
