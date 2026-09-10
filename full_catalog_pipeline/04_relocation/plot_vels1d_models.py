#!/usr/bin/env python3
"""Plot the reflection-seismology 1D profile (vels1d/) against the hypoDD layer models it is
discretized into, for both arrays, and against the hand-assembled models the v5 relocations
used.

Three things this is meant to make visible at a glance:
  * how closely the ~30-layer hypoDD discretization tracks the continuous 5 m profile
    (harmonic-mean velocities, so vertical traveltime is preserved layer by layer);
  * the T1 shift -- identical firn, 1.22 km of extra deep ice, and the same ice-bed
    transition shape and bedrock velocities riding down with it; and
  * how far the old models were from the measured profile, in particular the flat 2.50 km/s
    slab over the top 100 m and the Vs the old configs actually ran with (which is NOT the
    Vs they listed -- see vels1d_model.py; the dotted "v5 effective Vs" curve is Vp divided
    by the number hypoDD really read as the ratio).

Usage:
    python full_catalog_pipeline/plot_vels1d_models.py
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
from vels1d_model import ICE_THICKNESS_KM, build_layers, load_profile, shifted_profile

OUT_PNG = "full_catalog_pipeline/artifacts/full_run/vels1d_velocity_models.png"
MAX_DEPTH_KM = 5.0

INK = "#0b0b0b"
GRID = "#e4e3de"
C_PROFILE = "#8a8a86"
C_VP = "#2a78d6"
C_VS = "#eb6834"
C_OLD = "#b0aca4"

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": GRID,
    "axes.linewidth": 1,
    "axes.grid": True,
    "grid.color": GRID,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def step_xy(layers, max_depth, which):
    """Piecewise-constant (depth, velocity) pairs for a steps plot, depth on the y-axis."""
    xs, ys = [], []
    for i, layer in enumerate(layers):
        top = layer[0]
        bottom = layers[i + 1][0] if i + 1 < len(layers) else max_depth
        if top > max_depth:
            break
        v = layer[1] if which == "vp" else layer[2]
        ys += [top, min(bottom, max_depth)]
        xs += [v, v]
    return xs, ys


def old_layers(site):
    """The v5 model as hypoDD ACTUALLY ran it: the third line was read as the Vp/Vs ratio,
    so effective Vs = Vp / (the number that line lists)."""
    return [(d, vp, vp / listed) for d, vp, listed in ARRAY_CONFIG[site]["velocity_layers"]]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 8.5), sharey=True)

    for ax, site in zip(axes, ["T2", "T1"]):
        depth_m, vp, vs = shifted_profile(site)
        depth_km = depth_m / 1000.0
        ax.plot(vp / 1000.0, depth_km, color=C_PROFILE, lw=3.5, alpha=0.45,
                label="reflection profile (5 m)")
        ax.plot(vs / 1000.0, depth_km, color=C_PROFILE, lw=3.5, alpha=0.45)

        layers = build_layers(site)
        for which, color, name in [("vp", C_VP, "Vp"), ("vs", C_VS, "Vs")]:
            xs, ys = step_xy(layers, MAX_DEPTH_KM, which)
            ax.plot(xs, ys, color=color, lw=1.8, label=f"hypoDD layers, {name}")

        old = old_layers(site)
        for which, ls, name in [("vp", "-", "Vp"), ("vs", ":", "effective Vs")]:
            xs, ys = step_xy(old, MAX_DEPTH_KM, which)
            ax.plot(xs, ys, color=C_OLD, lw=1.5, linestyle=ls, label=f"v5 model, {name}")

        ax.axhline(ICE_THICKNESS_KM[site], color=INK, lw=1, linestyle="--", alpha=0.6)
        ax.annotate(f"BedMachine ice thickness {ICE_THICKNESS_KM[site]} km",
                    xy=(0.9, ICE_THICKNESS_KM[site]), xytext=(0, -5),
                    textcoords="offset points", fontsize=9, color=INK, va="top")
        n_ice = "profile as measured" if site == "T2" else "profile + 1.22 km deep ice"
        ax.set_title(f"{site}  ({len(layers)} layers, {n_ice})", fontsize=12)
        ax.set_xlabel("Velocity (km/s)")
        ax.set_xlim(0.8, 6.2)

    axes[0].set_ylim(MAX_DEPTH_KM, 0)
    axes[0].set_ylabel("Depth below ice surface (km)")
    axes[0].legend(fontsize=9, loc="lower left", framealpha=0.95)
    fig.suptitle("Reflection-seismology 1D model (shot over T2) as used for hypoDD relocation",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
