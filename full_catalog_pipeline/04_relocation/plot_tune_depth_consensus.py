#!/usr/bin/env python3
"""Do the surviving hypoDD configs actually agree about depth?

The per-config small multiples make the depth picture look wildly config-dependent. Two
separate questions hide behind that impression, and they have different answers:

  1. Do configs place INDIVIDUAL EVENTS at the same depth?  Largely yes -- the median
     per-event depth difference between robust configs is a few tens of metres.
  2. Do they agree on the DEPTH DISTRIBUTION relative to the ice-bed interface?  No. T2's
     events pile up within ~200 m of the interface, so a systematic offset far smaller than
     the pile's width swings the fraction placed below the bed from ~8% to ~46%.

So relative structure is robust and the "how many events are subglacial" statistic is not.
This figure shows both at once: depth histograms (left, 25 m bins, depth on the
vertical axis) with the interface marked, and the per-event agreement matrix (right).

Usage:
    python full_catalog_pipeline/plot_tune_depth_consensus.py --array T2
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

from hypodd_tune import RELOC_COLS, tune_root
from vels1d_model import bed_markers

# Categorical slots 1-5 (references/palette.md). These are step lines, so the adjacent-pair
# validation applies; a legend is present and every series is also directly labelled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, SECONDARY, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8a86", "#e4e3de"

DEFAULT_CONFIGS = ["clean_flat30", "clean_0.2_0.5", "dtmax_0.5", "base", "damp_low"]
MATRIX_CONFIGS = ["clean_flat30", "clean_0.2_0.5", "dtmax_0.5", "damp_flat30",
                  "wrct_tight", "damp_high", "base", "damp_low"]

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.linewidth": 1,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "xtick.color": SECONDARY, "ytick.color": SECONDARY,
    "axes.labelcolor": SECONDARY, "text.color": INK,
})


def load(array, name):
    p = os.path.join(tune_root(array), name, "prod", "hypoDD.reloc")
    if not (os.path.exists(p) and os.path.getsize(p) > 0):
        return None
    return pd.read_csv(p, sep=r"\s+", header=None, names=RELOC_COLS)[["id", "depth"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--array", default="T2", choices=["T1", "T2"])
    args = ap.parse_args()
    array = args.array
    bed = bed_markers(array)          # measured by the reflection profile, not BedMachine
    ice_base = bed["ice_base"]

    curves = [(n, d) for n in DEFAULT_CONFIGS if (d := load(array, n)) is not None]
    mats = [(n, d) for n in MATRIX_CONFIGS if (d := load(array, n)) is not None]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.4),
                                  gridspec_kw={"width_ratios": [1.05, 1]})

    # ---- left: cumulative depth distribution ----
    # Depth on the vertical axis, counts horizontal -- the conventional way to read a
    # hypocentre depth distribution, and it puts the ice-bed interface where the eye expects
    # a horizon. Fine bins (25 m) because the whole question is a sub-100 m offset.
    bins = np.arange(0.8, 3.5 + 0.025, 0.025)
    centers = 0.5 * (bins[:-1] + bins[1:])
    for (name, d), color in zip(curves, SERIES):
        counts, _ = np.histogram(d["depth"].values, bins=bins)
        below = 100 * (d["depth"] > ice_base).mean()
        peak = centers[counts.argmax()]
        ax.step(counts, centers, where="mid", color=color, lw=1.8,
                label=f"{name}  ·  peak {peak:.2f} km  ·  {below:.0f}% below ice base")
    # The profile resolves the bed as a ~510 m ramp, so it is drawn as a band: ice above
    # ice_base, bedrock below bedrock_top, and a transition between where the question
    # "ice or bed?" has no sharp answer.
    ax.axhspan(ice_base, bed["bedrock_top"], color=MUTED, alpha=0.16, zorder=0)
    for key, style in [("ice_base", "--"), ("bedrock_top", ":")]:
        ax.axhline(bed[key], color=INK, ls=style, lw=1.1)
    ax.annotate(f"ice base {ice_base:.2f} km", xy=(ax.get_xlim()[1], ice_base),
                xytext=(-4, -4), textcoords="offset points", fontsize=8.5, color=INK,
                ha="right", va="top")
    ax.annotate(f"bedrock {bed['bedrock_top']:.2f} km", xy=(ax.get_xlim()[1], bed["bedrock_top"]),
                xytext=(-4, 4), textcoords="offset points", fontsize=8.5, color=INK,
                ha="right", va="bottom")
    ax.set_ylim(3.2, 0.8)
    ax.set_xlabel("events per 25 m depth bin")
    ax.set_ylabel("depth (km)")
    ax.set_title("Two peaks in every config. The englacial peak (~1.35 km) is identical;\n"
                 "the basal peak spans 1.84-2.04 km across the measured ice base",
                 fontsize=11, color=INK)
    ax.legend(fontsize=8.5, loc="lower right", frameon=False)

    # ---- right: per-event agreement matrix ----
    n = len(mats)
    M = np.full((n, n), np.nan)
    for i, (_, a) in enumerate(mats):
        for j, (_, b) in enumerate(mats):
            if i == j:
                continue
            m = a.merge(b, on="id", suffixes=("_a", "_b"))
            M[i, j] = np.median(np.abs(m["depth_a"] - m["depth_b"])) * 1000
    im = ax2.imshow(M, cmap="Blues", vmin=0, vmax=np.nanmax(M))
    ax2.set_xticks(range(n)); ax2.set_yticks(range(n))
    ax2.set_xticklabels([m[0] for m in mats], rotation=45, ha="right", fontsize=8.5)
    ax2.set_yticklabels([m[0] for m in mats], fontsize=8.5)
    ax2.grid(False)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            v = M[i, j]
            ax2.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                     color="#ffffff" if v > 0.6 * np.nanmax(M) else INK)
    ax2.set_title("Median per-event depth difference between configs (m)\n"
                  "— relative structure agrees to a few tens of metres",
                  fontsize=11, color=INK)
    cb = fig.colorbar(im, ax=ax2, shrink=0.7, pad=0.02)
    cb.set_label("metres", color=SECONDARY)

    fig.suptitle(f"{array} — how much do the robust configs actually disagree about depth?",
                 fontsize=13, color=INK, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = os.path.join(tune_root(array), f"{array.lower()}_depth_consensus.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
