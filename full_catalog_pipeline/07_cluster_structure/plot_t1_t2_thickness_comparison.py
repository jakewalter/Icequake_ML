#!/usr/bin/env python3
"""Compare the vertical thickness of the icequake-occupied zone at T1 vs T2:
does one array see seismicity confined to a thin layer at the ice-bed
interface while the other sees it extend further up into the ice column, or
are the two sites similar?

Three catalogs, not two, because T1 and T2 are at different refinement
stages (T2's hypoDD relocation is still cross-correlating as of 2026-07-28 --
see hypodd_relocation_setup memory -- so no post-hypoDD T2 catalog exists
yet):
  - T1 pyocto v5 (pre-hypoDD, raw grid-search locations)
  - T1 hypoDD ccscale_0.33 (post-hypoDD, continuous-optimization refined)
  - T2 pyocto v5 (pre-hypoDD, raw grid-search locations -- best available)

Including T1's own pre-vs-post pair alongside T2's pre-only catalog lets the
reader calibrate how much a raw pyocto depth distribution typically changes
after hypoDD refinement, before trusting any T1-vs-T2 comparison that
necessarily leans on T2's still-preliminary depths. Known pyocto-stage
caveat (pyocto_full_catalog_rebuild memory): raw grid-search locations show
flat depth-banding artifacts near the ice-bed interface -- any "thickness"
read from a *_pyocto-raw series alone should be treated as an upper bound on
real structure, not a clean measurement.

Depths are referenced to each array's own established ice-bed interface
(T1=3.24 km, T2=2.02 km, both BedMachine/Bedmap2-derived, see
pyocto_full_catalog_rebuild / hypodd_relocation_setup memories) and flipped
so positive = up into the ice column, 0 = the interface.

Usage:
    python full_catalog_pipeline/plot_t1_t2_thickness_comparison.py
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

import catalog_paths

T1_PRE_CSV = "full_catalog_pipeline/artifacts/full_run/T1_v5/pyocto_events.csv"
T1_RELOC = catalog_paths.reloc("T1")
T2_PRE_CSV = "full_catalog_pipeline/artifacts/full_run/T2_v5/pyocto_events.csv"
OUT_DIR = "full_catalog_pipeline/artifacts/full_run"
OUT_PNG = f"{OUT_DIR}/t1_t2_depth_above_bed_comparison.png"

T1_BED_KM = 3.24
T2_BED_KM = 2.02

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#3f9b5c"
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


def load_series():
    t1_pre = pd.read_csv(T1_PRE_CSV)
    t1_post = pd.read_csv(T1_RELOC, sep=r"\s+", header=None, names=RELOC_COLS)
    t2_pre = pd.read_csv(T2_PRE_CSV)

    # drop events piled at the pyocto search-box depth edges (near-surface "air
    # quakes" at depth~0, and the rare deep-box-edge events near depth~6km) --
    # these are grid-search boundary artifacts, not real shallow/deep icequakes.
    # hypoDD's own IAQ=1 setting already removes the near-surface ones from the
    # post-hypoDD catalog, so this filter only changes the two *_pre series.
    n_t1_pre_before, n_t2_pre_before = len(t1_pre), len(t2_pre)
    t1_pre = t1_pre[(t1_pre["depth"] > 0.05) & (t1_pre["depth"] < 5.5)]
    t2_pre = t2_pre[(t2_pre["depth"] > 0.05) & (t2_pre["depth"] < 5.5)]
    print(f"dropped box-edge events: T1 pre {n_t1_pre_before}->{len(t1_pre)}, "
          f"T2 pre {n_t2_pre_before}->{len(t2_pre)}")

    series = {
        "T1 pyocto (pre-hypoDD)": (T1_BED_KM - t1_pre["depth"].values) * 1000,
        "T1 hypoDD ccscale_0.33 (post-hypoDD)": (T1_BED_KM - t1_post["depth"].values) * 1000,
        "T2 pyocto (pre-hypoDD, no hypoDD yet)": (T2_BED_KM - t2_pre["depth"].values) * 1000,
    }
    return series


def summarize(series):
    rows = []
    for name, above_bed_m in series.items():
        rows.append({
            "catalog": name,
            "n": len(above_bed_m),
            "median_m": np.median(above_bed_m),
            "p05_m": np.percentile(above_bed_m, 5),
            "p95_m": np.percentile(above_bed_m, 95),
            "iqr_m": np.percentile(above_bed_m, 75) - np.percentile(above_bed_m, 25),
            "p90_span_m": np.percentile(above_bed_m, 95) - np.percentile(above_bed_m, 5),
            "frac_above_200m": (above_bed_m > 200).mean(),
            "frac_above_500m": (above_bed_m > 500).mean(),
            "max_m": above_bed_m.max(),
        })
    return pd.DataFrame(rows)


def plot(series, stats, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    colors = [MUTED, ORANGE, BLUE]
    xlim = (-800, 2200)
    bins = np.linspace(xlim[0], xlim[1], 60)

    for ax, (name, above_bed_m), color, (_, row) in zip(axes, series.items(), colors, stats.iterrows()):
        ax.hist(above_bed_m, bins=bins, color=color, alpha=0.85, linewidth=0)
        ax.axvline(0, color=INK, linewidth=1.5, linestyle="--")
        ax.axvline(row["median_m"], color=INK, linewidth=1.2, linestyle="-")
        ax.set_xlim(xlim)
        ax.set_xlabel("Height above ice-bed interface (m)")
        ax.set_title(f"{name}\nn={row['n']:.0f}, median={row['median_m']:.0f} m, "
                      f"5-95th pctile span={row['p90_span_m']:.0f} m", fontsize=10.5)

    axes[0].set_ylabel("Number of events")
    fig.text(0.5, 1.02, "Vertical extent of icequake-occupied ice: T1 (pre & post hypoDD) vs. T2 (pre only)",
              ha="center", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.965, "Dashed line = interface (0 m); solid line = median. Negative = below interface "
                          "(bedrock side, likely grid-search/association-box artifact, not real).",
              ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    series = load_series()
    stats = summarize(series)
    pd.set_option("display.width", 160)
    print(stats.to_string(index=False))
    plot(series, stats, OUT_PNG)
    print(f"wrote {OUT_PNG}")
    stats.to_csv(f"{OUT_DIR}/t1_t2_depth_above_bed_stats.csv", index=False)
    print(f"wrote {OUT_DIR}/t1_t2_depth_above_bed_stats.csv")


if __name__ == "__main__":
    main()
