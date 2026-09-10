#!/usr/bin/env python3
"""Test 5 of PLAN_depth_resolvability.md: station-differential composite S-P vs trial depth.

Each cluster's composite S-P per station (from build_cluster_stacks.py, Test 1 -- CC-measured
on the raw waveforms, not the catalog picks) depends only on hypocentral distance and depth,
through the reflection-model traveltimes. Origin time cancels exactly, so this needs no
assumption about a shared composite origin across events -- unlike a full 4-parameter (x,y,z,t0)
relocation (Test 2), it uses only the S-P vector and the epicentre hypoDD already gives (which,
per the plan's caveats, is the one thing that IS robust across every config tried).

Method. Combine each cluster/station's composite S-P across components (Z, N, E) by inverse-
variance weighting on the measured scatter (a small floor is added since several combinations
converge to ~0 scatter at the 200 Hz sample grid -- real, not noise, but not literally zero
uncertainty either). Grid trial depth z. At each z, predicted S-P(z, r) = ts(z,r) - tp(z,r) from
the same pyrocko.cake reflection-model grids test_sp_absolute_depth.py already built for this
purpose. A single multiplicative velocity-scale nuisance k is fit in closed form at every z
(profiled out) before computing the weighted RMS, so a shared Vp/Vs mis-calibration cannot fake
a spurious depth minimum -- see the plan's caveat that the reflection model over-predicts S-P by
~2.3% (k=0.977).

Reads: a sharp RMS(z) minimum means depth is resolved by this data; a flat curve means it isn't
-- report RMS at z=2.00 km (measured ice base) vs the cluster's hypoDD median depth.

Usage:
    python full_catalog_pipeline/composite_sp_depth_pin.py --array T2
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
from test_sp_absolute_depth import build_cake_model, first_arrival_grid, interp

ICE_BASE_KM = 2.00
Z_MIN, Z_MAX, Z_STEP = 0.20, 3.00, 0.025
SE_FLOOR_MS = 2.5  # half a sample at 200 Hz -- several station/comp combos land on ~0 scatter
K_SENSITIVITY = 0.03  # +/-3% velocity-scale uncertainty carried into the reported minima

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#c0392b", "#8a8a86"]
INK, SECONDARY, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3de", "#8a8a86"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.linewidth": 1,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "xtick.color": SECONDARY, "ytick.color": SECONDARY,
    "axes.labelcolor": SECONDARY, "text.color": INK,
})


def combine_components(df):
    """Inverse-variance-weighted composite S-P per (cluster, station), across components."""
    df = df.copy()
    df["w"] = 1.0 / np.maximum(df["sp_se_ms"], SE_FLOOR_MS) ** 2
    rows = []
    for (cl, sta), g in df.groupby(["cluster", "station"]):
        w = g["w"].values
        sp = np.average(g["sp_composite"].values * 1000.0, weights=w)  # ms
        se = 1.0 / np.sqrt(w.sum())
        rows.append(dict(cluster=cl, station=sta, dist_km=float(g["dist_km"].iloc[0]),
                         depth_hypodd=float(g["depth_hypodd"].iloc[0]),
                         sp_ms=sp, se_ms=se, n_comp=len(g)))
    return pd.DataFrame(rows)


def rms_curve(obs_ms, se_ms, r_km, trial_z, grid_zs, grid_rs, gp, gs):
    """RMS(z) at each of `trial_z`, profiling out a single velocity-scale k at every depth.
    `grid_zs`/`grid_rs` are the native traveltime-grid axes used for bilinear interpolation."""
    w = 1.0 / se_ms ** 2
    out = np.full(len(trial_z), np.nan)
    ks = np.full(len(trial_z), np.nan)
    for i, z in enumerate(trial_z):
        pred_s = interp(grid_zs, grid_rs, gs, np.full_like(r_km, z), r_km)
        pred_p = interp(grid_zs, grid_rs, gp, np.full_like(r_km, z), r_km)
        pred = (pred_s - pred_p) * 1000.0
        good = np.isfinite(pred) & (pred > 0)
        if good.sum() < 3:
            continue
        o, p, ww = obs_ms[good], pred[good], w[good]
        k = float(np.sum(ww * o * p) / np.sum(ww * p * p))
        resid = o - k * p
        out[i] = float(np.sqrt(np.sum(ww * resid ** 2) / np.sum(ww)))
        ks[i] = k
    return out, ks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--array", default="T2", choices=["T1", "T2"])
    args = ap.parse_args()

    work = catalog_paths.work_dir(args.array)
    csv = os.path.join(work, f"{args.array.lower()}_cluster_composite_arrivals.csv")
    df = pd.read_csv(csv)
    comp = combine_components(df)

    print("building traveltime grids from the reflection model ...")
    model = build_cake_model(args.array)
    zs, rs, gp = first_arrival_grid(model, [cake.PhaseDef(d) for d in ("p", "P")])
    _, _, gs = first_arrival_grid(model, [cake.PhaseDef(d) for d in ("s", "S")])
    print(f"  grid {gp.shape}, P coverage {100*np.isfinite(gp).mean():.0f}%, "
          f"S coverage {100*np.isfinite(gs).mean():.0f}%\n")

    trial_z = np.arange(Z_MIN, Z_MAX + 1e-9, Z_STEP)
    summary = []
    fig, axes = plt.subplots(1, len(comp["cluster"].unique()),
                             figsize=(4.2 * comp["cluster"].unique().size, 5.2), sharey=True)
    axes = np.atleast_1d(axes)

    for ax, (cl, g) in zip(axes, sorted(comp.groupby("cluster"))):
        g = g.sort_values("dist_km")
        print(f"cluster {cl}: {len(g)} stations -- "
              + ", ".join(f"{r.station}={r.sp_ms:.0f}±{r.se_ms:.1f}ms@{r.dist_km:.1f}km"
                          for r in g.itertuples()))
        rms, ks = rms_curve(g["sp_ms"].values, g["se_ms"].values, g["dist_km"].values,
                            trial_z, zs, rs, gp, gs)
        valid = np.isfinite(rms)
        if valid.sum() < 5:
            print(f"  insufficient traveltime coverage, skipped\n")
            continue
        zv, rv, kv = trial_z[valid], rms[valid], ks[valid]
        i_min = int(np.argmin(rv))
        z_best, rms_best = zv[i_min], rv[i_min]
        # curvature-based 1-sigma: where RMS-weighted chi2 rises by 1 (using the number of
        # stations as the scale for what "delta-chi2=1" means in RMS units).
        n_sta = len(g)
        target = rms_best * np.sqrt(1.0 + 1.0 / max(n_sta - 1, 1))
        within = zv[rv <= target]
        z_lo, z_hi = (within.min(), within.max()) if len(within) else (np.nan, np.nan)
        i_base = int(np.argmin(np.abs(zv - ICE_BASE_KM)))
        i_hd = int(np.argmin(np.abs(zv - g["depth_hypodd"].median())))
        print(f"  RMS(z) min at z={z_best:.3f} km (RMS={rms_best:.1f} ms, k={kv[i_min]:.3f}), "
              f"68% range [{z_lo:.3f}, {z_hi:.3f}] km" if np.isfinite(z_lo) else
              f"  RMS(z) min at z={z_best:.3f} km (RMS={rms_best:.1f} ms, k={kv[i_min]:.3f})")
        print(f"  RMS at ice base (2.00 km) = {rv[i_base]:.1f} ms; "
              f"RMS at hypoDD median ({g['depth_hypodd'].median():.3f} km) = {rv[i_hd]:.1f} ms; "
              f"flat-curve span (RMS < 1.2*min) = "
              f"[{zv[rv < 1.2*rms_best].min():.2f}, {zv[rv < 1.2*rms_best].max():.2f}] km\n")
        summary.append(dict(cluster=cl, n_stations=n_sta, z_best_km=z_best, rms_best_ms=rms_best,
                            z_lo_km=z_lo, z_hi_km=z_hi, k_at_min=kv[i_min],
                            rms_at_ice_base_ms=rv[i_base],
                            depth_hypodd_median_km=g["depth_hypodd"].median(),
                            rms_at_hypodd_ms=rv[i_hd],
                            flat_span_lo_km=zv[rv < 1.2*rms_best].min(),
                            flat_span_hi_km=zv[rv < 1.2*rms_best].max()))

        ax.plot(zv, rv, color=SERIES[0], lw=1.6)
        ax.axvline(z_best, color=SERIES[1], lw=1.2, ls="--", label=f"min z={z_best:.2f} km")
        ax.axvline(ICE_BASE_KM, color=INK, lw=1, ls=":", label="ice base 2.00 km")
        ax.axvline(g["depth_hypodd"].median(), color=SERIES[2], lw=1.2, ls="-.",
                  label=f"hypoDD median {g['depth_hypodd'].median():.2f} km")
        if np.isfinite(z_lo):
            ax.axvspan(z_lo, z_hi, color=SERIES[0], alpha=0.12)
        ax.set_xlabel("trial depth z (km)")
        ax.set_title(f"cluster {cl}  (n={n_sta} stations)", fontsize=10)
        ax.legend(fontsize=7, frameon=False, loc="upper right")
    axes[0].set_ylabel("weighted RMS residual (ms)")
    fig.suptitle(f"{args.array} — Test 5: composite S-P vs trial depth, per cluster\n"
                f"(velocity-scale k profiled out at every z; sharp minimum = depth resolved)",
                fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png = os.path.join(work, f"{args.array.lower()}_composite_sp_depth_pin.png")
    fig.savefig(out_png, dpi=145, bbox_inches="tight")
    plt.close(fig)

    out_csv = os.path.join(work, f"{args.array.lower()}_composite_sp_depth_pin_summary.csv")
    pd.DataFrame(summary).to_csv(out_csv, index=False)
    print(f"wrote {out_png}\nwrote {out_csv}")


if __name__ == "__main__":
    main()
