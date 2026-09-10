#!/usr/bin/env python3
"""Plot T1 and T2's 1D layered velocity models (Vp and Vs vs depth) side by side, pulled
directly from hypodd_relocate.py's ARRAY_CONFIG so this always reflects whatever model is
actually in use for the official relocations (not a re-typed copy that could drift out of
sync). T2's model is T1's own layer structure/Vp progression, depth-shifted so the ice-bed
transition sits at T2's own ~2.02km BedMachine/Bedmap2 thickness instead of T1's ~3.24km
(see ARRAY_CONFIG's comments) -- plotting them together makes that shift, and the identical
shape/Vp/Vs-ratio otherwise, directly visible.

Usage:
    python full_catalog_pipeline/plot_velocity_models.py
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

from hypodd_relocate import ARRAY_CONFIG

OUT_PNG = "full_catalog_pipeline/artifacts/full_run/t1_t2_velocity_models.png"

BED_KM = {"T1": 3.24, "T2": 2.02}
COLOR = {"T1": "#2a78d6", "T2": "#eb6834"}
MAX_DEPTH_KM = 10.0  # crop the plot to the shallow/crustal layers that matter for these arrays

INK = "#0b0b0b"
MUTED = "#8a8a86"
GRID = "#e4e3de"

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


def step_profile(layers, max_depth):
    """Layers are (depth_top_km, vp, vs) tuples defining a piecewise-constant model.
    Returns (depths, vps, vss) arrays suitable for a steps-post plot with depth on the
    y-axis (each layer's velocity extends down to the next layer's top, or max_depth for
    the last one)."""
    depths, vps, vss = [], [], []
    for i, (d0, vp, vs) in enumerate(layers):
        d1 = layers[i + 1][0] if i + 1 < len(layers) else max_depth
        if d0 > max_depth:
            break
        depths += [d0, min(d1, max_depth)]
        vps += [vp, vp]
        vss += [vs, vs]
    return depths, vps, vss


def main():
    fig, ax = plt.subplots(figsize=(8, 9))

    for site in ["T1", "T2"]:
        layers = ARRAY_CONFIG[site]["velocity_layers"]
        depths, vps, vss = step_profile(layers, MAX_DEPTH_KM)
        color = COLOR[site]
        ax.plot(vps, depths, color=color, linestyle="-", linewidth=2.2, label=f"{site} Vp")
        ax.plot(vss, depths, color=color, linestyle="--", linewidth=2.2, label=f"{site} Vs")
        ax.axhline(BED_KM[site], color=color, linestyle=":", linewidth=1.3, alpha=0.8)
        ax.annotate(f"{site} ice-bed ({BED_KM[site]} km)", xy=(0.15, BED_KM[site]),
                    xytext=(3, -3), textcoords="offset points", fontsize=9, color=color,
                    va="top")

    ax.set_ylim(MAX_DEPTH_KM, 0)
    ax.set_xlim(0, 9)
    ax.set_xlabel("Velocity (km/s)")
    ax.set_ylabel("Depth (km)")
    ax.set_title("T1 vs T2 1D velocity models\n(solid = Vp, dashed = Vs; dotted = each site's own ice-bed depth)",
                 fontsize=13)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
