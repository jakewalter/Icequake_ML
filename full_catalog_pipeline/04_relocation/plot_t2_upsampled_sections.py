#!/usr/bin/env python3
"""Cross-sections of T2's hypoDD relocation on the upsampled dt.cc, against the
authoritative catalog it would replace.

The upsampled cross-correlation file ([[cc-p-deficit-root-cause]] fix) takes dt.cc from
2.2% P to 53% P, which for the first time lets the CC data form S-P. These sections show
what that did to the locations. The catalog-wide medians are small, but they HIDE a
bimodal response: hypoDD cluster 2 (~680 events, 21%) moves as a body while everything
else stays put, and how far it moves is set by the CC weighting -- -13 m under
clean_flat30, -501 m under clean_ccstrong. The last panel is that test.

Usage:
    python full_catalog_pipeline/plot_t2_upsampled_sections.py [--config clean_ccstrong]
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
import pyproj

from plot_hypodd_t2_basal_3d import (
    PYOCTO_STA, OUT_DIR, ICE_BED_DEPTH_KM, RELOC_COLS,
    load_stations, load_reloc, project_km,
)

import catalog_paths

# Fixed comparison baseline (the authoritative relocation), and the tune tree the candidate
# configs live in. Figures are written to the work_dir of the relocation ICEQUAKE_RELOC
# selects, since they are derived from it -- see catalog_paths.
ROOT = "full_catalog_pipeline/artifacts/full_run/T2_v5/hypodd_vels1d"
AUTH = f"{ROOT}/output_files/hypoDD.reloc"
# This figure is derived from ONE TUNE CONFIG (compared against the authoritative baseline),
# and a tune config has no work_dir of its own -- so it belongs beside that config's outputs
# rather than in the work_dir of whatever relocation happens to be selected.

# Categorical slots 1 and 2 of the validated reference palette. Two series, all-pairs
# validated (CVD dE 24.7, normal-vision 33.6, both clear of the floors).
C_OLD = "#2a78d6"   # blue   - authoritative (old dt.cc, 2.2% P)
C_NEW = "#eb6834"   # orange - upsampled dt.cc (53% P)
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#dedddA"


def sections(ax, df, color, label, xcol, s=5, alpha=0.45):
    ax.scatter(df[xcol], df["depth"], s=s, color=color, alpha=alpha,
               linewidths=0, label=label, zorder=3)


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="clean_ccstrong")
    args = ap.parse_args()

    new_path = f"{ROOT}/tune_upsampled/{args.config}/prod/hypoDD.reloc"
    for p in (AUTH, new_path):
        if not os.path.exists(p):
            raise SystemExit(f"missing: {p}")

    to_ps = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    sta = load_stations(PYOCTO_STA)
    sx, sy = to_ps.transform(sta["lon"].values, sta["lat"].values)
    cx, cy = sx.mean(), sy.mean()

    old = load_reloc(AUTH)
    new = load_reloc(new_path)
    for d in (old, new):
        d["ex"], d["nx"] = project_km(d, to_ps, cx, cy)

    # per-event change, on the events the two catalogs share
    j = old.set_index("id").join(new.set_index("id"), lsuffix="_a", rsuffix="_n",
                                 how="inner")
    dz = (j["depth_n"] - j["depth_a"]) * 1000.0
    dh = np.hypot(j["ex_n"] - j["ex_a"], j["nx_n"] - j["nx_a"]) * 1000.0

    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=0.30, wspace=0.26)

    # ---- map view -------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(old["ex"], old["nx"], s=4, color=C_OLD, alpha=0.40, linewidths=0,
               label=f"authoritative  (n={len(old)})", zorder=3)
    ax.scatter(new["ex"], new["nx"], s=4, color=C_NEW, alpha=0.40, linewidths=0,
               label=f"upsampled dt.cc  (n={len(new)})", zorder=4)
    ax.scatter((sx - cx) / 1000, (sy - cy) / 1000, marker="v", s=70, color=INK,
               zorder=6, label="stations")
    for _, r in sta.iterrows():
        x, y = to_ps.transform(r["lon"], r["lat"])
        ax.annotate(r["code"], ((x - cx) / 1000, (y - cy) / 1000),
                    textcoords="offset points", xytext=(0, 7), ha="center",
                    fontsize=7.5, color=INK, zorder=7)
    ax.set_xlabel("easting from array centroid (km)", fontsize=9.5, color=MUTED)
    ax.set_ylabel("northing (km)", fontsize=9.5, color=MUTED)
    ax.set_title("map view", fontsize=11, color=INK, loc="left")
    ax.set_aspect("equal")
    style(ax)
    ax.legend(fontsize=8, frameon=False, loc="lower left")

    # ---- E-W section ----------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    sections(ax, old, C_OLD, "authoritative", "ex")
    sections(ax, new, C_NEW, "upsampled dt.cc", "ex")
    ax.axhline(ICE_BED_DEPTH_KM, color=INK, linestyle="--", linewidth=1.2, zorder=5,
               label=f"ice-bed interface ({ICE_BED_DEPTH_KM} km)")
    ax.invert_yaxis()
    ax.set_xlabel("easting from array centroid (km)", fontsize=9.5, color=MUTED)
    ax.set_ylabel("depth (km)", fontsize=9.5, color=MUTED)
    ax.set_title("W–E cross-section", fontsize=11, color=INK, loc="left")
    style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    # ---- N-S section ----------------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    sections(ax, old, C_OLD, "authoritative", "nx")
    sections(ax, new, C_NEW, "upsampled dt.cc", "nx")
    ax.axhline(ICE_BED_DEPTH_KM, color=INK, linestyle="--", linewidth=1.2, zorder=5,
               label=f"ice-bed interface ({ICE_BED_DEPTH_KM} km)")
    ax.invert_yaxis()
    ax.set_xlabel("northing from array centroid (km)", fontsize=9.5, color=MUTED)
    ax.set_ylabel("depth (km)", fontsize=9.5, color=MUTED)
    ax.set_title("S–N cross-section", fontsize=11, color=INK, loc="left")
    style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    # ---- depth histogram ------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    bins = np.arange(0.0, 3.2, 0.05)
    ax.hist(old["depth"], bins=bins, color=C_OLD, alpha=0.55, label="authoritative")
    ax.hist(new["depth"], bins=bins, color=C_NEW, alpha=0.55, label="upsampled dt.cc")
    ax.axvline(ICE_BED_DEPTH_KM, color=INK, linestyle="--", linewidth=1.2,
               label="ice-bed interface")
    ax.set_xlabel("depth (km)", fontsize=9.5, color=MUTED)
    ax.set_ylabel("events", fontsize=9.5, color=MUTED)
    ax.set_title("depth distribution", fontsize=11, color=INK, loc="left")
    style(ax)
    ax.legend(fontsize=8, frameon=False)

    # ---- per-event depth change ----------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(dz, bins=np.arange(-600, 620, 20), color=C_NEW, alpha=0.75)
    ax.axvline(0, color=INK, linewidth=1.0)
    ax.axvline(np.median(dz), color=INK, linestyle="--", linewidth=1.2,
               label=f"median {np.median(dz):+.0f} m")
    ax.set_xlabel("depth change, new − authoritative (m)", fontsize=9.5, color=MUTED)
    ax.set_ylabel(f"events (n={len(dz)} shared)", fontsize=9.5, color=MUTED)
    n_up = int((dz < -300).sum())
    med_up = np.median(dz[dz < -300]) if n_up else float("nan")
    ax.annotate(f"{n_up} events ({n_up/len(dz):.0%})\nmove up, median {med_up:+.0f} m",
                (-430, ax.get_ylim()[1] * 0.45), fontsize=8.5, color=INK, ha="center",
                bbox=dict(boxstyle="round,pad=0.4", fc="#fcfcfb", ec=MUTED, lw=0.7))
    ax.set_title("per-event depth change", fontsize=11, color=INK, loc="left")
    style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    # ---- the moving population, across every config --------------------------
    # The per-event depth change is BIMODAL: hypoDD cluster 2 moves as a body while
    # everything else stays put. Whether that move is data-driven or a tuning artifact
    # is the question the median hides, so test the same population in every config.
    ax = fig.add_subplot(gs[1, 2])
    movers = j.index[dz < -300]
    configs = ["clean_flat30", "clean_ccweak", "clean_0.2_0.5", "ccmin_0.5",
               "base", "clean_ccstrong"]
    meds, labels = [], []
    for c in configs:
        p = f"{ROOT}/tune_upsampled/{c}/prod/hypoDD.reloc"
        if not os.path.exists(p):
            continue
        nn = load_reloc(p).set_index("id")
        jj = old.set_index("id").join(nn, lsuffix="_a", rsuffix="_n", how="inner")
        d = (jj["depth_n"] - jj["depth_a"]) * 1000.0
        meds.append(np.median(d[jj.index.isin(movers)]))
        labels.append(c)
    bars = ax.barh(range(len(meds)), meds, color=C_OLD, height=0.62, zorder=3)
    bars[labels.index(args.config)].set_color(C_NEW)
    for i, m in enumerate(meds):
        ax.annotate(f"{m:+.0f} m", (m, i), xytext=(-6 if m < -160 else 6, 0),
                    textcoords="offset points", ha="right" if m < -160 else "left",
                    va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.axvline(0, color=MUTED, linewidth=0.8, zorder=2)
    ax.set_xlabel(f"median depth change of the same {len(movers)} events (m)",
                  fontsize=9.5, color=MUTED)
    ax.set_title(f"is the moving cluster's shift real?  (n={len(movers)})",
                 fontsize=11, color=INK, loc="left")
    ax.set_xlim(min(meds) * 1.35, max(0, max(meds)) + 120)
    style(ax)

    fig.suptitle(
        f"T2 hypoDD on upsampled dt.cc (config {args.config}, 53% P) vs authoritative "
        f"(2.2% P)\n"
        f"depth median change {np.median(dz):+.0f} m (p90 |Δ| {np.percentile(np.abs(dz), 90):.0f} m) · "
        f"horizontal median {np.median(dh):.0f} m",
        fontsize=12.5, color=INK, y=0.975, ha="center")

    out = os.path.join(ROOT, "tune_upsampled", args.config,
                       f"t2_upsampled_sections_{args.config}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"wrote {out}")
    print(f"  shared events {len(j)}; depth median {np.median(dz):+.1f} m, "
          f"p90|Δ| {np.percentile(np.abs(dz), 90):.0f} m")
    print(f"  horizontal median |Δh| {np.median(dh):.0f} m")
    print(f"  moving population n={len(movers)}; median dz per config: "
          + ", ".join(f"{c}={m:+.0f}m" for c, m in zip(labels, meds)))


if __name__ == "__main__":
    main()
