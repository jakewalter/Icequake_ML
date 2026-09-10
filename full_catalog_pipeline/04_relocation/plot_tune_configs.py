#!/usr/bin/env python3
"""Map views, depth sections and depth histograms for every hypoDD tuning config, as small
multiples, so the sweep's candidate relocations can be compared by eye rather than only
through the summary table.

Three figures per array:

  *_mapviews.png    plan view, events coloured by depth (sequential blue: light = shallow,
                    dark = deep), stations marked. Answers "does the epicentral pattern hold
                    together, or has a config smeared/split it?"
  *_sections.png    distance from the array centroid vs depth, with the ice-bed interface
                    drawn in. Answers "where in the column does this config put the events?"
  *_stability.png   depth histogram of the production run (100% of the CC data) against the
                    train run (80%), per config. This is the one that shows the failure
                    directly: an unstable config's two histograms sit at different depths.

Panels are ordered by cluster-1 depth shift, most stable first, and each panel's title
carries that number, so the visual and the metric are read together.

Usage:
    python full_catalog_pipeline/plot_tune_configs.py --array T2
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
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from matplotlib.colors import LinearSegmentedColormap, Normalize

from hypodd_tune import RELOC_COLS, tune_root
from vels1d_model import bed_markers

# Sequential blue ramp, steps 100->700 (references/palette.md). Depth is a magnitude, so it
# gets one hue light->dark; deeper reads darker.
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
DEPTH_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
# Categorical slots 1 and 2 -- the documented all-pairs-passing pair.
C_PROD, C_TRAIN = "#2a78d6", "#eb6834"

INK, SECONDARY, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8a86", "#e4e3de"
MAX_DEPTH_KM = 6.0
MAP_RADIUS_KM = 6.0

plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "xtick.color": SECONDARY, "ytick.color": SECONDARY,
    "axes.labelcolor": SECONDARY, "text.color": INK,
})


def load_stations(path):
    st = pd.read_csv(path, sep=r"\s+", header=None, names=["sta", "lat", "lon", "elev"])
    return st


def collect(array):
    """Every config that produced a production relocation, with its stability metric."""
    root = tune_root(array)
    out = []
    for name in sorted(os.listdir(root)):
        cdir = os.path.join(root, name)
        prod = os.path.join(cdir, "prod", "hypoDD.reloc")
        if not (os.path.isdir(cdir) and os.path.exists(prod) and os.path.getsize(prod) > 0):
            continue
        rj = os.path.join(cdir, "result.json")
        meta = json.load(open(rj)) if os.path.exists(rj) else {}
        out.append({
            "name": name,
            "prod": prod,
            "train": os.path.join(cdir, "train", "hypoDD.reloc"),
            "shift": meta.get("cluster1_depth_shift_m"),
            "medabs": meta.get("heldout_medabs_ms"),
        })
    # Most stable first; configs with no stability number sort last.
    out.sort(key=lambda d: (d["shift"] is None, d["shift"] if d["shift"] is not None else 0))
    return out


def grid(n):
    ncol = 4 if n > 6 else 3
    return int(np.ceil(n / ncol)), ncol


def panel_title(ax, cfg):
    bits = [cfg["name"]]
    if cfg["shift"] is not None:
        bits.append(f"c1 shift {cfg['shift']:.0f} m")
    if cfg["medabs"] is not None:
        bits.append(f"{cfg['medabs']:.2f} ms")
    ax.set_title("\n".join([bits[0], "  ·  ".join(bits[1:])]) if len(bits) > 1 else bits[0],
                 fontsize=8, color=INK, pad=4)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--array", default="T2", choices=["T1", "T2"])
    args = ap.parse_args()
    array = args.array
    marks = bed_markers(array)   # reflection-profile bed, not BedMachine
    bed = marks["ice_base"]

    configs = collect(array)
    if not configs:
        print(f"no production relocations found for {array}")
        return
    print(f"{array}: {len(configs)} configs with a production relocation")

    root = tune_root(array)
    sta = load_stations(os.path.join(os.path.dirname(root), "input_files", "station.sel"))
    to_ps = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    sx, sy = to_ps.transform(sta["lon"].values, sta["lat"].values)
    cx, cy = sx.mean(), sy.mean()
    sx, sy = (sx - cx) / 1000.0, (sy - cy) / 1000.0

    for cfg in configs:
        d = pd.read_csv(cfg["prod"], sep=r"\s+", header=None, names=RELOC_COLS)
        x, y = to_ps.transform(d["lon"].values, d["lat"].values)
        d["ex"], d["ny"] = (x - cx) / 1000.0, (y - cy) / 1000.0
        d["r"] = np.hypot(d["ex"], d["ny"])
        cfg["df"] = d

    nrow, ncol = grid(len(configs))
    norm = Normalize(vmin=0, vmax=MAX_DEPTH_KM)

    # ---------------- map views ----------------
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.2 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, cfg in zip(axes, configs):
        d = cfg["df"]
        keep = d["r"] <= MAP_RADIUS_KM
        ax.scatter(d.loc[keep, "ex"], d.loc[keep, "ny"], c=d.loc[keep, "depth"],
                   cmap=DEPTH_CMAP, norm=norm, s=3, linewidths=0, alpha=0.85)
        ax.scatter(sx, sy, marker="^", s=26, facecolor="none", edgecolor=INK, linewidths=0.9)
        ax.set_aspect("equal")
        ax.set_xlim(-MAP_RADIUS_KM, MAP_RADIUS_KM)
        ax.set_ylim(-MAP_RADIUS_KM, MAP_RADIUS_KM)
        panel_title(ax, cfg)
    for ax in axes[len(configs):]:
        ax.axis("off")
    for ax in axes[:len(configs)]:
        if ax in axes.reshape(nrow, ncol)[-1]:
            ax.set_xlabel("east (km)")
    for ax in axes.reshape(nrow, ncol)[:, 0]:
        ax.set_ylabel("north (km)")
    sm = plt.cm.ScalarMappable(cmap=DEPTH_CMAP, norm=norm)
    cb = fig.colorbar(sm, ax=axes.tolist(), shrink=0.35, pad=0.015, aspect=30)
    cb.set_label("depth (km)", color=SECONDARY)
    cb.ax.invert_yaxis()
    fig.suptitle(f"{array} — map view per hypoDD tuning config (triangles = stations; "
                 f"most stable first)", fontsize=11, color=INK, y=1.0, va="bottom")
    out = os.path.join(root, f"{array.lower()}_tune_mapviews.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # ---------------- depth sections ----------------
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.0 * nrow),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, cfg in zip(axes, configs):
        d = cfg["df"]
        keep = d["r"] <= MAP_RADIUS_KM
        ax.scatter(d.loc[keep, "r"], d.loc[keep, "depth"], c=d.loc[keep, "depth"],
                   cmap=DEPTH_CMAP, norm=norm, s=3, linewidths=0, alpha=0.8)
        ax.axhspan(bed, marks["bedrock_top"], color=MUTED, alpha=0.16, zorder=0)
        ax.axhline(bed, color=INK, linestyle="--", linewidth=0.9)
        ax.set_ylim(MAX_DEPTH_KM, 0)
        ax.set_xlim(0, MAP_RADIUS_KM)
        panel_title(ax, cfg)
    for ax in axes[len(configs):]:
        ax.axis("off")
    for ax in axes.reshape(nrow, ncol)[-1]:
        ax.set_xlabel("distance from centroid (km)")
    for ax in axes.reshape(nrow, ncol)[:, 0]:
        ax.set_ylabel("depth (km)")
    fig.tight_layout(rect=(0, 0, 1, 0.985), h_pad=1.6)
    fig.suptitle(f"{array} — depth section per config (dashed = {bed:.2f} km measured ice base; "
                 f"shaded = ice-bed transition to {marks['bedrock_top']:.2f} km)",
                 fontsize=11, color=INK, y=0.995)
    out = os.path.join(root, f"{array.lower()}_tune_sections.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # ---------------- stability: prod vs train depth ----------------
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.7 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    bins = np.linspace(0, MAX_DEPTH_KM, 70)
    for ax, cfg in zip(axes, configs):
        ax.hist(cfg["df"]["depth"].clip(0, MAX_DEPTH_KM), bins=bins, color=C_PROD,
                alpha=0.75, label="100% CC")
        if os.path.exists(cfg["train"]) and os.path.getsize(cfg["train"]) > 0:
            t = pd.read_csv(cfg["train"], sep=r"\s+", header=None, names=RELOC_COLS)
            ax.hist(t["depth"].clip(0, MAX_DEPTH_KM), bins=bins, histtype="step",
                    linewidth=1.4, color=C_TRAIN, label="80% CC")
        ax.axvspan(bed, marks["bedrock_top"], color=MUTED, alpha=0.16, zorder=0)
        ax.axvline(bed, color=INK, linestyle="--", linewidth=0.9)
        panel_title(ax, cfg)
    for ax in axes[len(configs):]:
        ax.axis("off")
    for ax in axes.reshape(nrow, ncol)[-1]:
        ax.set_xlabel("depth (km)")
    for ax in axes.reshape(nrow, ncol)[:, 0]:
        ax.set_ylabel("events")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=1.8)
    fig.suptitle(f"{array} — depth stability: full data vs 20% of the cross-correlation "
                 f"data withheld\n(two curves apart = the config's depths are not resolved)",
                 fontsize=11, color=INK, y=0.995)
    out = os.path.join(root, f"{array.lower()}_tune_stability.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
