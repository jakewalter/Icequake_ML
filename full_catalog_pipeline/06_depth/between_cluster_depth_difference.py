#!/usr/bin/env python3
"""Test 4 of PLAN_depth_resolvability.md: is the depth DIFFERENCE between two clusters real?

The plan separates three questions. Within-cluster spread was answered (not supported by the
waveforms) and absolute depth was addressed by Tests 3 and 5. This is the second question, and
it is the one that "the englacial clusters sit above the basal population" actually asserts:
not where any cluster is in absolute terms, but whether two of them are at different depths.

Why a differential test earns its place. Anything that biases S-P at a station the same way for
every cluster -- a site delay in the firn, an instrument response, a systematic error in the
velocity model at that station's distance -- cancels exactly when two clusters' composite S-P
are differenced at that SAME station. What does not cancel is a genuine depth difference,
because the two clusters sit at different distances from that station, so the difference
varies from station to station in a pattern fixed by the geometry.

That gives the test its discriminant. The comparison is not "does a depth difference fit?" --
with three free parameters and four to six stations something always fits. It is:

  M0  the station-to-station differences are a single CONSTANT offset (1 parameter)
  M1  they follow the geometry of two clusters at depths z_A and z_B (z_A, z_B, and a shared
      velocity scale k, profiled out exactly as in Test 5 so a mis-calibrated Vp/Vs cannot
      manufacture a difference)

M0 is M1 restricted to z_A = z_B: the two clusters at a COMMON depth, each still at its own
epicentral distance. It has to be that and not a constant time offset -- the clusters have
different epicentres, so their distances to a given station differ by kilometres and their S-P
differs by hundreds of ms with no depth difference whatever. A constant-offset null is a straw
man that any geometry beats; scoring against it would have declared nearly every pair resolved.
If M1 cannot beat the common-depth fit by more than its extra parameter buys, the depth
difference is not resolved -- regardless of how far apart hypoDD puts the two clusters.

The headline number is the confidence interval on dz = z_A - z_B, from the profiled chi-square
(delta-chi2 = 1 after minimising over z_B and k at each dz). An interval containing zero means
the data do not distinguish the two depths. That interval, not the best-fit dz, is the result.

Usage:
    python full_catalog_pipeline/between_cluster_depth_difference.py --array T2
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
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyrocko import cake

import catalog_paths
from composite_sp_depth_pin import (
    ICE_BASE_KM, Z_MIN, Z_MAX, Z_STEP, combine_components,
)
from test_sp_absolute_depth import build_cake_model, first_arrival_grid, interp

# M1 spends three parameters (z_A, z_B, k). With only three stations it fits ANY data exactly
# -- the first run of this test duly returned 0.2 ms residuals and an absurdly tight interval
# for every three-station pair. Four is the minimum that leaves anything to test.
MIN_COMMON_STATIONS = 4

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, SECONDARY, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3de", "#8a8a86"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.linewidth": 1,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "xtick.color": SECONDARY, "ytick.color": SECONDARY,
    "axes.labelcolor": SECONDARY, "text.color": INK,
})


def sp_model(zs, rs, gp, gs, z, r_km):
    """Predicted S-P in ms at depth z (km) and the given distances."""
    zz = np.full_like(r_km, float(z))
    return (interp(zs, rs, gs, zz, r_km) - interp(zs, rs, gp, zz, r_km)) * 1000.0


def chi2_grid(d_obs, d_se, r_a, r_b, trial_z, zs, rs, gp, gs):
    """chi-square of model M1 over the (z_A, z_B) grid, with the velocity scale k profiled out
    in closed form at every node."""
    w = 1.0 / d_se ** 2
    n = len(trial_z)
    out = np.full((n, n), np.nan)
    pred_a = np.array([sp_model(zs, rs, gp, gs, z, r_a) for z in trial_z])
    pred_b = np.array([sp_model(zs, rs, gp, gs, z, r_b) for z in trial_z])
    for i in range(n):
        pa = pred_a[i]
        if not np.all(np.isfinite(pa) & (pa > 0)):
            continue
        for j in range(n):
            pb = pred_b[j]
            if not np.all(np.isfinite(pb) & (pb > 0)):
                continue
            p = pa - pb
            denom = float(np.sum(w * p * p))
            if denom <= 0:
                continue
            k = float(np.sum(w * d_obs * p) / denom)
            resid = d_obs - k * p
            out[i, j] = float(np.sum(w * resid ** 2))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--array", default="T2", choices=["T1", "T2"])
    ap.add_argument("--clusters", nargs="+", type=int, default=None)
    args = ap.parse_args()

    work = catalog_paths.work_dir(args.array)
    df = pd.read_csv(os.path.join(
        work, f"{args.array.lower()}_cluster_composite_arrivals.csv"))
    comp = combine_components(df)
    if args.clusters:
        comp = comp[comp.cluster.isin(args.clusters)]

    print("building traveltime grids from the reflection model ...")
    model = build_cake_model(args.array)
    zs, rs, gp = first_arrival_grid(model, [cake.PhaseDef(d) for d in ("p", "P")])
    _, _, gs = first_arrival_grid(model, [cake.PhaseDef(d) for d in ("s", "S")])
    trial_z = np.arange(Z_MIN, Z_MAX + 1e-9, Z_STEP)

    clusters = sorted(comp.cluster.unique())
    rows = []
    pairs = []
    for ca, cb in itertools.combinations(clusters, 2):
        ga = comp[comp.cluster == ca].set_index("station")
        gb = comp[comp.cluster == cb].set_index("station")
        common = sorted(set(ga.index) & set(gb.index))
        if len(common) < MIN_COMMON_STATIONS:
            print(f"clusters {ca}-{cb}: only {len(common)} common stations, skipped")
            continue
        a, b = ga.loc[common], gb.loc[common]
        d_obs = (a.sp_ms - b.sp_ms).values
        d_se = np.sqrt(a.se_ms.values ** 2 + b.se_ms.values ** 2)
        w = 1.0 / d_se ** 2

        # M1: two depths plus a shared velocity scale (M0 is read off its diagonal below)
        g = chi2_grid(d_obs, d_se, a.dist_km.values, b.dist_km.values,
                      trial_z, zs, rs, gp, gs)
        if not np.any(np.isfinite(g)):
            print(f"clusters {ca}-{cb}: no traveltime coverage, skipped")
            continue
        i, j = np.unravel_index(np.nanargmin(g), g.shape)
        chi2_1 = float(g[i, j])
        dof1 = max(len(common) - 3, 1)
        za, zb = trial_z[i], trial_z[j]

        # Profile chi-square onto dz = z_A - z_B: at each dz keep the best (z_B, k).
        dz_all = (trial_z[:, None] - trial_z[None, :]).ravel()
        flat = g.ravel()
        ok = np.isfinite(flat)
        order = np.argsort(dz_all[ok])
        dz_sorted, chi_sorted = dz_all[ok][order], flat[ok][order]
        # bin to the grid step so each dz has a single profiled minimum
        bins = np.round(dz_sorted / Z_STEP).astype(int)
        prof_dz, prof_chi = [], []
        for bkey in np.unique(bins):
            m = bins == bkey
            prof_dz.append(bkey * Z_STEP)
            prof_chi.append(chi_sorted[m].min())
        prof_dz, prof_chi = np.array(prof_dz), np.array(prof_chi)
        chi_min = prof_chi.min()
        # Error rescaling, one-directional. The S-P uncertainties are formal standard errors on
        # a stack of hundreds of events (a few ms) and badly understate the real model+site
        # budget, so where the fit is poor the interval must widen: scale by the reduced
        # chi-square. Where the fit is better than the stated errors, they are NOT deflated --
        # shrinking the interval below what the quoted uncertainties support is how the first
        # run produced its spurious sub-100 m "resolved" differences.
        scale = max(chi_min / dof1, 1.0)
        within = prof_dz[prof_chi <= chi_min + scale]
        dz_lo, dz_hi = float(within.min()), float(within.max())
        dz_best = float(prof_dz[int(np.argmin(prof_chi))])
        includes_zero = bool(dz_lo <= 0.0 <= dz_hi)

        # M0 = the common-depth fit, i.e. M1 on its z_A = z_B diagonal.
        diag = np.array([g[m, m] for m in range(len(trial_z))])
        chi2_0 = float(np.nanmin(diag)) if np.any(np.isfinite(diag)) else np.nan
        z_common = float(trial_z[int(np.nanargmin(diag))]) if np.isfinite(chi2_0) else np.nan
        # One extra parameter separates M0 from M1, so delta-chi2 (rescaled) is the evidence.
        dchi2_vs_common = (chi2_0 - chi2_1) / scale if np.isfinite(chi2_0) else np.nan
        dz_hypodd = float(a.depth_hypodd.iloc[0] - b.depth_hypodd.iloc[0])

        rows.append(dict(
            cluster_a=ca, cluster_b=cb, n_common=len(common),
            stations=",".join(common),
            dz_hypodd_km=dz_hypodd, dz_best_km=dz_best,
            dz_lo_km=dz_lo, dz_hi_km=dz_hi, includes_zero=includes_zero,
            z_a_best_km=float(za), z_b_best_km=float(zb),
            z_common_km=z_common, dof_M1=dof1, chi2_red_M1=float(chi2_1 / dof1),
            rms_common_depth_ms=float(np.sqrt(chi2_0 / np.sum(w))),
            rms_M1_ms=float(np.sqrt(chi2_1 / np.sum(w))),
            dchi2_vs_common_depth=float(dchi2_vs_common),
        ))
        pairs.append((ca, cb, prof_dz, prof_chi, chi_min, scale, dz_hypodd))
        print(f"clusters {ca}-{cb} ({len(common)} common stations: {','.join(common)}):")
        print(f"  hypoDD says dz = {dz_hypodd:+.3f} km")
        print(f"  M1 best dz = {dz_best:+.3f} km, delta-chi2=1 range "
              f"[{dz_lo:+.3f}, {dz_hi:+.3f}] km -> "
              f"{'CONSISTENT WITH ZERO' if includes_zero else 'excludes zero'}")
        print(f"  RMS: common-depth M0 (z={z_common:.2f} km) = "
              f"{np.sqrt(chi2_0/np.sum(w)):.1f} ms, "
              f"two-depth M1 = {np.sqrt(chi2_1/np.sum(w)):.1f} ms; "
              f"rescaled Δχ² = {dchi2_vs_common:.2f} on 1 extra parameter "
              f"(dof={dof1}, χ²_red={chi2_1/dof1:.1f})\n")

    if not rows:
        print("no cluster pair had enough common stations")
        return

    n = len(pairs)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.4 * nrow), squeeze=False)
    for ax, (ca, cb, pdz, pchi, cmin, scale, dz_dd) in zip(axes.ravel(), pairs):
        ax.plot(pdz, (pchi - cmin) / scale, color=SERIES[0], lw=1.5)
        ax.axhline(1.0, color=SERIES[1], lw=1, ls="--")
        ax.axvline(0.0, color=INK, lw=1)
        ax.axvline(dz_dd, color=MUTED, lw=1.2, ls=":")
        ax.set_ylim(0, 12)
        ax.set_title(f"clusters {ca} − {cb}", fontsize=10)
        ax.set_xlabel("depth difference z$_A$ − z$_B$ (km)")
        ax.set_ylabel("Δχ² (scaled)")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"{args.array} — between-cluster depth difference from composite S-P; "
                 f"dashed Δχ²=1, dotted = hypoDD's difference", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(work, f"{args.array.lower()}_between_cluster_depth_difference.png")
    fig.savefig(out, dpi=145, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    csv = os.path.join(work, f"{args.array.lower()}_between_cluster_depth_difference.csv")
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
