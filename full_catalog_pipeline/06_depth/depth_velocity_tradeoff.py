#!/usr/bin/env python3
"""Synthesis for PLAN_depth_resolvability.md: how much of T2's "depth" is really an assumed
Vp/Vs?

Test 5 (composite_sp_depth_pin.py) profiles out a multiplicative velocity scale k at every
trial depth, so that a mis-calibrated model cannot fake a minimum. That protects the SHAPE of
the RMS(z) curve, but it hides something the plan's +-3% caveat did not anticipate: the fitted
k is different for every cluster -- 0.84 to 1.02 across T2's five. The clusters sit under the
same ice column, so they cannot each have their own Vp/Vs. Only one of two things can be true:
either the depths are absorbing a velocity error, or the velocity scale is absorbing a depth
error. This script measures which, in two ways.

1. k sweep. Re-minimise each cluster's S-P misfit with k held at a sequence of FIXED, stated
   values instead of fitted per cluster, and tabulate where the depth minimum lands. The width
   of that column is the honest depth uncertainty due to velocity alone -- a quantity Test 5
   cannot report, because profiling k out makes it invisible.

2. Joint fit. Impose what physics requires: one common k shared by every cluster, each cluster
   keeping its own depth. Report the depths, the shift from Test 5's per-cluster-k answer, and
   the depth range spanned by k's own 1-sigma interval.

The second is the one that matters for interpretation, because the between-cluster depth
ORDER -- which cluster is shallower than which, i.e. the whole "englacial versus basal" claim
-- is not invariant under it.

Usage:
    python full_catalog_pipeline/depth_velocity_tradeoff.py --array T2
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
from pyrocko import cake

import catalog_paths
import vels1d_model
from composite_sp_depth_pin import SE_FLOOR_MS, Z_MIN, Z_MAX, Z_STEP, combine_components
from test_sp_absolute_depth import build_cake_model, first_arrival_grid, interp

K_TABLE = (1.00, 0.98, 0.95, 0.90, 0.85)
K_MIN, K_MAX, K_STEP = 0.80, 1.10, 0.002

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#8a5cd6", "#c4a01a"]
INK, SECONDARY, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3de", "#8a8a86"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.linewidth": 1,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "xtick.color": SECONDARY, "ytick.color": SECONDARY,
    "axes.labelcolor": SECONDARY, "text.color": INK,
})


def build(array):
    """Per cluster: observed S-P, weights, and the predicted S-P at every trial depth."""
    work = catalog_paths.work_dir(array)
    comp = combine_components(pd.read_csv(os.path.join(
        work, f"{array.lower()}_cluster_composite_arrivals.csv")))
    model = build_cake_model(array)
    zs, rs, gp = first_arrival_grid(model, [cake.PhaseDef(d) for d in ("p", "P")])
    _, _, gs = first_arrival_grid(model, [cake.PhaseDef(d) for d in ("s", "S")])
    tz = np.arange(Z_MIN, Z_MAX + 1e-9, Z_STEP)
    data = {}
    for cl, g in comp.groupby("cluster"):
        r = g.dist_km.values
        pred = np.array([(interp(zs, rs, gs, np.full_like(r, z), r)
                          - interp(zs, rs, gp, np.full_like(r, z), r)) * 1000.0 for z in tz])
        ok = np.all(np.isfinite(pred) & (pred > 0), axis=1)
        data[int(cl)] = dict(obs=g.sp_ms.values,
                             w=1.0 / np.maximum(g.se_ms.values, SE_FLOOR_MS) ** 2,
                             pred=pred, ok=np.flatnonzero(ok), n=len(g),
                             depth_hypodd=float(g.depth_hypodd.median()))
    return work, tz, data


def chi2_curve(d, k=None):
    """chi-square against depth; k fixed if given, else profiled at each depth."""
    out = np.full(len(d["pred"]), np.inf)
    for i in d["ok"]:
        p = d["pred"][i]
        kk = float(np.sum(d["w"] * d["obs"] * p) / np.sum(d["w"] * p * p)) if k is None else k
        out[i] = float(np.sum(d["w"] * (d["obs"] - kk * p) ** 2))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--array", default="T2", choices=["T1", "T2"])
    args = ap.parse_args()

    work, tz, data = build(args.array)
    ice_base = vels1d_model.bed_markers(args.array)["ice_base"]
    clusters = sorted(data)

    # --- 1. k sweep -------------------------------------------------------------------
    print(f"{args.array}: depth of the S-P minimum against a FIXED velocity scale k\n")
    hdr = f"{'cl':>3} {'k fitted':>9} {'z (k free)':>11} " + " ".join(
        f"{'z(k='+format(k,'.2f')+')':>11}" for k in K_TABLE) + f" {'swing':>7}"
    print(hdr)
    rows = []
    for cl in clusters:
        d = data[cl]
        c = chi2_curve(d)
        i = int(np.argmin(c))
        p = d["pred"][i]
        k_free = float(np.sum(d["w"] * d["obs"] * p) / np.sum(d["w"] * p * p))
        zs_fixed = [tz[int(np.argmin(chi2_curve(d, k)))] for k in K_TABLE]
        swing = max(zs_fixed) - min(zs_fixed)
        print(f"{cl:>3} {k_free:9.3f} {tz[i]:11.3f} "
              + " ".join(f"{z:11.3f}" for z in zs_fixed) + f" {swing:7.3f}")
        rows.append(dict(cluster=cl, k_fitted=k_free, z_k_free_km=float(tz[i]),
                         **{f"z_k_{k:.2f}_km": float(z) for k, z in zip(K_TABLE, zs_fixed)},
                         z_swing_over_k_km=float(swing)))
    print("\n'swing' is how far the depth moves across a 15% change in velocity scale — "
          "the depth uncertainty\ndue to velocity alone, which profiling k out makes invisible.\n")

    # --- 2. joint fit, one shared k ----------------------------------------------------
    ks = np.arange(K_MIN, K_MAX + 1e-9, K_STEP)
    total = np.array([sum(chi2_curve(data[cl], k).min() for cl in clusters) for k in ks])
    jb = int(np.argmin(total))
    k_best = float(ks[jb])
    ndat = sum(data[cl]["n"] for cl in clusters)
    dof = max(ndat - (len(clusters) + 1), 1)
    scale = max(total[jb] / dof, 1.0)
    k_in = ks[total <= total[jb] + scale]

    print(f"Joint fit with ONE shared velocity scale (what the physics requires):")
    print(f"  best common k = {k_best:.3f}, 1-sigma [{k_in.min():.3f}, {k_in.max():.3f}]")
    print(f"  chi2 = {total[jb]:.0f} on {ndat} observations, {len(clusters)+1} parameters "
          f"-> chi2/dof = {total[jb]/dof:.0f}  (the model does NOT fit)\n")
    print(f"{'cl':>3} {'z @ shared k':>13} {'z @ own k':>10} {'shift':>8} "
          f"{'z over k 1-sigma':>22} {'hypoDD':>8}")
    joint = []
    for cl in clusters:
        d = data[cl]
        z_shared = float(tz[int(np.argmin(chi2_curve(d, k_best)))])
        z_free = float(tz[int(np.argmin(chi2_curve(d)))])
        lo = float(tz[int(np.argmin(chi2_curve(d, float(k_in.max()))))])
        hi = float(tz[int(np.argmin(chi2_curve(d, float(k_in.min()))))])
        lo, hi = min(lo, hi), max(lo, hi)
        print(f"{cl:>3} {z_shared:13.3f} {z_free:10.3f} {z_shared-z_free:+8.3f} "
              f"{lo:10.3f} - {hi:<9.3f} {d['depth_hypodd']:8.3f}")
        joint.append(dict(cluster=cl, z_shared_k_km=z_shared, z_own_k_km=z_free,
                          shift_km=z_shared - z_free, z_lo_km=lo, z_hi_km=hi,
                          depth_hypodd_km=d["depth_hypodd"]))

    order_free = [c for _, c in sorted((r["z_own_k_km"], r["cluster"]) for r in joint)]
    order_shared = [c for _, c in sorted((r["z_shared_k_km"], r["cluster"]) for r in joint)]
    print(f"\n  shallow->deep order with per-cluster k: {order_free}")
    print(f"  shallow->deep order with one shared k : {order_shared}")
    print("  -> the between-cluster depth ORDER is not invariant to the velocity scale."
          if order_free != order_shared else
          "  -> the order survives the shared-k constraint.")

    # --- figure -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for i, cl in enumerate(clusters):
        d = data[cl]
        zs_k = [tz[int(np.argmin(chi2_curve(d, k)))] for k in ks]
        axes[0].plot(ks, zs_k, color=SERIES[i % len(SERIES)], lw=1.5, label=f"cluster {cl}")
    axes[0].axvline(k_best, color=INK, lw=1.2)
    axes[0].axvspan(k_in.min(), k_in.max(), color=MUTED, alpha=0.18, lw=0)
    axes[0].axhline(ice_base, color=SECONDARY, lw=1, ls="--")
    axes[0].set_xlabel("velocity scale k applied to the reflection model")
    axes[0].set_ylabel("depth of the S-P misfit minimum (km)")
    axes[0].set_title("depth is a restatement of the assumed velocity", fontsize=10)
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].invert_yaxis()

    axes[1].plot(ks, total / total.min(), color=SERIES[0], lw=1.6)
    axes[1].axvline(k_best, color=INK, lw=1.2)
    axes[1].axvspan(k_in.min(), k_in.max(), color=MUTED, alpha=0.18, lw=0)
    axes[1].set_xlabel("shared velocity scale k")
    axes[1].set_ylabel("joint χ² / min")
    axes[1].set_ylim(0.9, 3.0)
    axes[1].set_title(f"joint fit over all clusters (best k = {k_best:.3f})", fontsize=10)
    fig.suptitle(f"{args.array} — depth/velocity trade-off in the composite S-P depth estimate; "
                 f"dashed line is the ice base", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(work, f"{args.array.lower()}_depth_velocity_tradeoff.png")
    fig.savefig(out, dpi=145, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")
    pd.DataFrame(rows).to_csv(os.path.join(
        work, f"{args.array.lower()}_depth_velocity_ksweep.csv"), index=False)
    pd.DataFrame(joint).to_csv(os.path.join(
        work, f"{args.array.lower()}_depth_velocity_joint.csv"), index=False)
    print(f"wrote {args.array.lower()}_depth_velocity_ksweep.csv and _joint.csv")


if __name__ == "__main__":
    main()
