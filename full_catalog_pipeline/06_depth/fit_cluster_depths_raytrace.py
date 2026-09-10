#!/usr/bin/env python3
"""Per-cluster depth from CC-refined S-P, using PROPER 1D ray tracing.

Supersedes the straight-ray joint fit. That model assumed one average velocity along a
straight source-receiver line, which is defensible at 0.2 km offset and wrong at 5-6 km, and
it failed visibly: per-station depths of 1.94 km (JULA, r=0.21) versus 0.00 km (WICH/EPJZ,
r~4.9) for the SAME events, and a joint fit that placed most clusters below the 2.02 km bed
while demanding Vp/Vs 1.80 against the model's 2.01.

Here predicted S-P comes from lib/raytrace1d, which integrates the vels1d profile over
direct, turning and head-wave branches and takes the first arrival. Validated to <2 ms
against an analytic homogeneous half-space and to ~1 ms against a two-layer head-wave case.

Reported per cluster:
  z_free     depth with the velocity model fixed as measured
  z_scaled   depth when a single global velocity scale is also free (the degeneracy that
             [[t2-depth-resolvability-tests]] found dominates every absolute-depth estimate)
  ci_lo/hi   depth range within +1 ms rms of that cluster's own best fit

Usage:
    ICEQUAKE_RELOC=hypodd_vels1d_ccstrong python full_catalog_pipeline/fit_cluster_depths_raytrace.py
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
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import catalog_paths
from lib.raytrace1d import t2_sp_table

RELOC_COLS = [
    "id", "lat", "lon", "depth", "x", "y", "z", "ex", "ey", "ez",
    "yr", "mo", "dy", "hr", "mi", "sc", "mag",
    "nccp", "nccs", "nctp", "ncts", "rcc", "rct", "cid",
]


def collect(work, reloc, min_n=40, min_cc=0.5):
    rel = pd.read_csv(reloc, sep=r"\s+", header=None, names=RELOC_COLS).set_index("id")
    rows = []
    for f in sorted(glob.glob(os.path.join(work, "t2sp_c*_cc_refine_*.csv"))):
        b = os.path.basename(f).split("_")
        cluster, station, comp = int(b[1][1:]), b[2], b[3]
        t = pd.read_csv(f).set_index("id")
        t = t[np.isfinite(t["sp_refined"]) & (t["cc"] >= min_cc)]
        if len(t) < min_n:
            continue
        ids = [i for i in t.index if i in rel.index]
        rows.append(dict(cluster=cluster, station=station, component=comp,
                         r=float(t["dist_km"].median()),
                         sp=float(t["sp_refined"].median()),
                         n=len(ids), cc=float(t["cc"].mean()),
                         z_hypodd=float(rel.loc[ids, "depth"].median())))
    d = pd.DataFrame(rows)
    # average Z and N -- they are the same measurement on two components, not independent data
    return (d.groupby(["cluster", "station"])
             .agg(r=("r", "mean"), sp=("sp", "mean"), n=("n", "max"),
                  cc=("cc", "mean"), z_hypodd=("z_hypodd", "first"))
             .reset_index())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-stations", type=int, default=2)
    args = ap.parse_args()

    work, reloc = catalog_paths.work_dir("T2"), catalog_paths.reloc("T2")
    d = collect(work, reloc)
    keep = d.groupby("cluster")["station"].transform("size") >= args.min_stations
    d = d[keep].reset_index(drop=True)
    print(f"relocation: {reloc}")
    print(f"{len(d)} cluster-station observations, {d.cluster.nunique()} clusters\n")

    T = t2_sp_table()
    cl = sorted(d.cluster.unique())
    idx = {c: i for i, c in enumerate(cl)}
    ci = np.array([idx[c] for c in d.cluster])
    obs = d.sp.values * 1000.0
    r = d.r.values

    def pred(z_by_cluster, scale=1.0):
        return T(z_by_cluster[ci], r) * 1000.0 / scale

    def res_free(p):
        return np.nan_to_num(pred(p) - obs, nan=1e3)

    def res_scaled(p):
        return np.nan_to_num(pred(p[:-1], p[-1]) - obs, nan=1e3)

    z0 = np.clip(d.groupby("cluster").z_hypodd.first().values, 0.2, 3.2)
    lo, hi = 0.10, 3.40
    f1 = least_squares(res_free, z0, bounds=([lo] * len(cl), [hi] * len(cl)))
    f2 = least_squares(res_scaled, np.r_[z0, 1.0],
                       bounds=([lo] * len(cl) + [0.85], [hi] * len(cl) + [1.15]))
    rms1 = np.sqrt(np.mean(f1.fun ** 2))
    rms2 = np.sqrt(np.mean(f2.fun ** 2))

    # per-cluster depth scan: how well is each cluster's depth actually pinned?
    zs = np.arange(lo, hi, 0.02)
    out = []
    for c in cl:
        m = d.cluster.values == c
        rr, oo = r[m], obs[m]
        mis = np.array([np.sqrt(np.nanmean((T(np.full(m.sum(), z), rr) * 1000 - oo) ** 2))
                        for z in zs])
        best = np.nanargmin(mis)
        ok = zs[mis <= mis[best] + 1.0]
        out.append(dict(cluster=c, n_sta=int(m.sum()),
                        z_hypodd=d[m].z_hypodd.iloc[0],
                        z_free=f1.x[idx[c]], z_scaled=f2.x[idx[c]],
                        z_alone=zs[best], rms_alone=mis[best],
                        ci_lo=ok.min() if ok.size else np.nan,
                        ci_hi=ok.max() if ok.size else np.nan,
                        r_min=rr.min(), r_max=rr.max()))
    res = pd.DataFrame(out)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\njoint fit, velocity FIXED as measured : rms {rms1:.1f} ms")
    print(f"joint fit, one global velocity scale  : rms {rms2:.1f} ms, "
          f"scale {f2.x[-1]:.3f} (Vp/Vs {2.008 / f2.x[-1]:.3f} effective)")
    print(f"ice-bed interface: 2.02 km")

    p = os.path.join(work, "t2_cluster_depths_raytrace.csv")
    res.to_csv(p, index=False)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
