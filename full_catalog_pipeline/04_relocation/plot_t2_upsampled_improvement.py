#!/usr/bin/env python3
"""Did regenerating dt.cc with the sub-sample-fit fix improve T2's relocation?

Compares the previous preferred solution (config clean_0.2_0.5 on the original dt.cc, the
current authoritative catalog) against clean_ccstrong on the regenerated, upsampled dt.cc.

The claim this figure supports, and the one it does NOT:

  SUPPORTED  -- the INVERSION is better conditioned. hypoDD's LSQR returned no error estimate
                (ez = 0, i.e. it stopped without converging) on 41.5% of the old catalog's
                events and on 0.1% of the new one. That is a property of the solve, measured
                by hypoDD itself, and it is not open to interpretation.

  NOT SUPPORTED -- that the new DEPTHS are more correct. T2 depth is not resolved
                (see [[t2-depth-resolvability-tests]]): it trades off ~1:1 with velocity
                scale, and the waveform tests that could arbitrate are swamped by ~142 ms of
                station-specific S-P offsets that a 1D model cannot produce. The sections are
                shown so the depth change is VISIBLE, not as evidence that it is right.

Usage:
    ICEQUAKE_RELOC=hypodd_vels1d_ccstrong \
        python full_catalog_pipeline/plot_t2_upsampled_improvement.py
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
import numpy as np
import pandas as pd
import pyproj

import catalog_paths

# The BASELINE is the authoritative relocation, named explicitly because it is a fixed
# comparison point rather than "whatever is selected".
BASE = "full_catalog_pipeline/artifacts/full_run/T2_v5/hypodd_vels1d"
OLD = f"{BASE}/output_files/hypoDD.reloc"
# The run under evaluation, and where the figure goes, both follow ICEQUAKE_RELOC -- this
# figure is DERIVED from that relocation, so per catalog_paths it belongs in its work_dir.
NEW = catalog_paths.reloc("T2")
STA = catalog_paths.station_sel("T2")
OUT = os.path.join(catalog_paths.work_dir("T2"), "t2_upsampled_improvement.png")
ICE_BED_KM = 2.02

RELOC_COLS = [
    "id", "lat", "lon", "depth", "x", "y", "z", "ex", "ey", "ez",
    "yr", "mo", "dy", "hr", "mi", "sc", "mag",
    "nccp", "nccs", "nctp", "ncts", "rcc", "rct", "cid",
]

# categorical slots 1 and 2 of the validated palette (all-pairs CVD dE 24.7, normal 33.6)
C_OLD, C_NEW = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#dedddA", "#fcfcfb"


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9)


def main():
    rd = lambda p: pd.read_csv(p, sep=r"\s+", header=None, names=RELOC_COLS)
    old, new = rd(OLD), rd(NEW)
    sta = pd.read_csv(STA, sep=r"\s+", header=None, names=["id", "lat", "lon", "elev"])
    to = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    sx, sy = to.transform(sta["lon"].values, sta["lat"].values)
    cx, cy = sx.mean(), sy.mean()
    for d in (old, new):
        x, y = to.transform(d["lon"].values, d["lat"].values)
        d["ex"], d["nx"] = (x - cx) / 1000.0, (y - cy) / 1000.0

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.34, wspace=0.24)

    # ---- cross-sections, side by side so structure is readable -------------------
    lim = (max(old["ex"].min(), -8), min(old["ex"].max(), 8))
    for col, (d, c, name, sub) in enumerate([
            (old, C_OLD, "previous preferred", "clean_0.2_0.5 · original dt.cc · 2.7% P"),
            (new, C_NEW, "upsampled dt.cc", "clean_ccstrong · regenerated dt.cc · 62.2% P")]):
        ax = fig.add_subplot(gs[0, col])
        ax.scatter(d["ex"], d["depth"], s=5, color=c, alpha=0.35, linewidths=0, zorder=3)
        ax.axhline(ICE_BED_KM, color=INK, linestyle="--", linewidth=1.2, zorder=5,
                   label=f"ice-bed interface ({ICE_BED_KM} km)")
        ax.set_xlim(*lim)
        ax.set_ylim(3.1, -0.1)
        ax.set_xlabel("easting from array centroid (km)", fontsize=9.5, color=MUTED)
        ax.set_ylabel("depth (km)", fontsize=9.5, color=MUTED)
        ax.set_title(f"{name}   (n={len(d)})\n{sub}", fontsize=11.5, color=INK, loc="left")
        style(ax)
        ax.legend(fontsize=8, frameon=False, loc="lower left")

    # ---- the actual measured improvement -----------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    zo = (old["ez"] == 0).mean() * 100
    zn = (new["ez"] == 0).mean() * 100
    bars = ax.bar([0, 1], [zo, zn], color=[C_OLD, C_NEW], width=0.55, zorder=3)
    for b, v in zip(bars, [zo, zn]):
        ax.annotate(f"{v:.1f}%", (b.get_x() + b.get_width() / 2, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=13, color=INK,
                    fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["original\ndt.cc", "upsampled\ndt.cc"], fontsize=9.5)
    ax.set_ylabel("events where LSQR did not converge\n(hypoDD returned ez = 0)",
                  fontsize=9.5, color=MUTED)
    ax.set_ylim(0, 50)
    ax.set_title("the measured improvement\nhypoDD's own solver diagnostic",
                 fontsize=11.5, color=INK, loc="left")
    style(ax)

    # ---- depth histograms ---------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    bins = np.arange(0, 3.2, 0.05)
    ax.hist(old["depth"], bins=bins, color=C_OLD, alpha=0.55, label="previous preferred")
    ax.hist(new["depth"], bins=bins, color=C_NEW, alpha=0.55, label="upsampled dt.cc")
    ax.axvline(ICE_BED_KM, color=INK, linestyle="--", linewidth=1.2, label="ice-bed")
    ax.set_xlabel("depth (km)", fontsize=9.5, color=MUTED)
    ax.set_ylabel("events", fontsize=9.5, color=MUTED)
    ax.set_title("depth distribution\ndifferent — but NOT shown to be more correct",
                 fontsize=11.5, color=INK, loc="left")
    style(ax)
    ax.legend(fontsize=8, frameon=False)

    # ---- what changed in the data ------------------------------------------------
    # Deliberately NOT a bar chart: observations (millions), P fraction (%) and event count
    # are incommensurate, and putting them on one axis would invite a false visual comparison.
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    ax.set_title("what changed in the input data\nthe sub-sample fit recovered 111x more P",
                 fontsize=11.5, color=INK, loc="left")
    rows = [("CC observations", "592,548", "2,847,206", "4.8x"),
            ("P observations", "15,947", "1,769,987", "111x"),
            ("P fraction", "2.7%", "62.2%", ""),
            ("events relocated", "3,293", "3,546", "+253"),
            ("LSQR non-converged", "41.5%", "0.1%", "")]
    y = 0.86
    ax.text(0.02, y + 0.10, "", fontsize=9)
    ax.text(0.50, y + 0.09, "previous", fontsize=9, color=C_OLD, ha="center",
            fontweight="bold")
    ax.text(0.78, y + 0.09, "upsampled", fontsize=9, color=C_NEW, ha="center",
            fontweight="bold")
    ax.text(0.97, y + 0.09, "change", fontsize=9, color=MUTED, ha="center")
    for label, a, b, ch in rows:
        ax.text(0.02, y, label, fontsize=9, color=INK, va="center")
        ax.text(0.50, y, a, fontsize=9, color=MUTED, ha="center", va="center")
        ax.text(0.78, y, b, fontsize=9, color=INK, ha="center", va="center")
        ax.text(0.97, y, ch, fontsize=9, color=MUTED, ha="center", va="center")
        y -= 0.155

    # ---- formal depth errors ------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    oe, ne = old.loc[old["ez"] > 0, "ez"], new.loc[new["ez"] > 0, "ez"]
    b = np.arange(0, 6.1, 0.25)
    ax.hist(oe, bins=b, color=C_OLD, alpha=0.55,
            label=f"previous ({len(oe)} converged)")
    ax.hist(ne, bins=b, color=C_NEW, alpha=0.55,
            label=f"upsampled ({len(ne)} converged)")
    ax.set_xlabel("hypoDD formal depth error ez (m), converged events only",
                  fontsize=9.5, color=MUTED)
    ax.set_ylabel("events", fontsize=9.5, color=MUTED)
    ax.set_title("formal depth error\nLSQR underestimates these — SVD check pending",
                 fontsize=11.5, color=INK, loc="left")
    style(ax)
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle(
        "T2: regenerating dt.cc with the sub-sample fit — what it did and did not change",
        fontsize=13.5, color=INK, y=0.975)
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=SURF)
    print(f"wrote {OUT}")
    print(f"  LSQR non-convergence {zo:.1f}% -> {zn:.1f}%")
    print(f"  events {len(old)} -> {len(new)}; median ez {oe.median():.2f} -> {ne.median():.2f} m")


if __name__ == "__main__":
    main()
