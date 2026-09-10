#!/usr/bin/env python3
"""Where does hypoDD's DEPTH information actually come from at T2?

Motivating puzzle: cross-correlation of the raw waveforms shows no S-P variation with hypoDD's
within-cluster depths (dS-P/dz ~ 0 against a required +131 ms/km, and MCCC is validated by an
injection test), yet hypoDD produces a depth spread from cross-correlation data. Two candidate
explanations: hypoDD's CC data contain a depth signal the S-P test cannot see, or hypoDD's
depths are not being set by the CC data at all.

This script measures the information budget rather than arguing about it. For every differential
-time observation hypoDD was given it computes the traveltime partial derivative with respect to
source depth in the reflection model, and sums the weighted square -- the diagonal Fisher
information for depth. The same for the horizontal partial. Reported per station, per phase, and
per file (dt.cc versus dt.ct), because that ratio is what decides which observations control the
solution.

Two structural facts this exposes, neither visible from the relocation output:

1. dt.cc is 97.8% S. The catalog picks are balanced (16.4k P / 15.5k S) and so is dt.ct
   (1.79M P / 1.78M S), so this is lost in the cross-correlation stage, not in picking. With
   almost no P, the CC data cannot form S-P at all -- so no CC-measured S-P signal exists for
   the waveform test to have contradicted. The two results were never in conflict.

2. With S-only data, depth is degenerate with origin time. A pair's origin-time difference is
   common to every station, so depth is resolvable only through VARIATION in dT_S/dz across the
   contributing stations. The script therefore reports the weighted spread of dT_S/dz, not just
   its size: a station set whose depth partials are all alike carries no depth information
   however many observations it holds. That is the quantitative form of "is the inversion
   dominated by distant stations?".

Usage:
    python full_catalog_pipeline/hypodd_cc_depth_information.py --array T2 --clusters 0 1 2
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

import numpy as np
import pandas as pd
import pyproj
from pyrocko import cake

import catalog_paths
from lib.deej_waveform_common import RELOC_COLS
from test_sp_absolute_depth import build_cake_model, first_arrival_grid, interp

H_KM = 0.05  # half-step for the numerical partials


def read_dt(path, is_cc):
    """dt.cc / dt.ct -> DataFrame(ev1, ev2, sta, dt, w, ph). dt.ct has an extra time column."""
    rows, ev1, ev2 = [], None, None
    with open(path) as fh:
        for line in fh:
            if line[0] == "#":
                parts = line.split()
                ev1, ev2 = int(parts[1]), int(parts[2])
            else:
                p = line.split()
                if is_cc and len(p) >= 4:
                    rows.append((ev1, ev2, p[0].split(".")[-1], float(p[1]), float(p[2]), p[3]))
                elif not is_cc and len(p) >= 5:
                    # sta t1 t2 weight phase -> differential time is t1 - t2
                    rows.append((ev1, ev2, p[0].split(".")[-1],
                                 float(p[1]) - float(p[2]), float(p[3]), p[4]))
    return pd.DataFrame(rows, columns=["ev1", "ev2", "sta", "dt", "w", "ph"])


def partials(zs, rs, grid, z, r):
    """(dT/dz, dT/dr) in ms/km by central differences on the traveltime grid."""
    z = np.atleast_1d(float(z)); r = np.atleast_1d(np.asarray(r, float))
    zz = np.full_like(r, z[0])
    dz = (interp(zs, rs, grid, zz + H_KM, r) - interp(zs, rs, grid, zz - H_KM, r)) / (2 * H_KM)
    dr = (interp(zs, rs, grid, zz, r + H_KM) - interp(zs, rs, grid, zz, r - H_KM)) / (2 * H_KM)
    return dz * 1000.0, dr * 1000.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--array", default="T2", choices=["T1", "T2"])
    ap.add_argument("--clusters", nargs="+", type=int, required=True)
    ap.add_argument("--dtcc", default="dt.cc",
                    help="which CC file to score. The authoritative T2 relocation ran on the "
                         "FILTERED copy (dt.cc.authoritative, |dt|<=0.2s and coeff>=0.5 per "
                         "PROVENANCE.md); plain dt.cc is the unfiltered raw product kept for "
                         "other analyses. Check PROVENANCE.md before assuming.")
    args = ap.parse_args()

    work = catalog_paths.work_dir(args.array)
    reloc = catalog_paths.reloc(args.array)
    n_fields = len(open(reloc).readline().split())
    cols = RELOC_COLS if n_fields == 18 else RELOC_COLS[:17] + [
        "nccp", "nccs", "nctp", "ncts", "rcc", "rct", "cid"]
    rel = pd.read_csv(reloc, sep=r"\s+", header=None, names=cols)[["id", "lat", "lon", "depth"]]

    sta = pd.read_csv(catalog_paths.station_sel(args.array), sep=r"\s+", header=None,
                      names=["sta", "lat", "lon", "elev"])
    sta["code"] = sta["sta"].str.split(".").str[-1]
    to_ps = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    sx, sy = to_ps.transform(sta["lon"].values, sta["lat"].values)
    sxy = {c: (x / 1000.0, y / 1000.0) for c, x, y in zip(sta["code"], sx, sy)}
    ex, ey = to_ps.transform(rel["lon"].values, rel["lat"].values)
    rel["x"], rel["y"] = ex / 1000.0, ey / 1000.0

    print("building traveltime grids ...")
    model = build_cake_model(args.array)
    zs, rs, gp = first_arrival_grid(model, [cake.PhaseDef(d) for d in ("p", "P")])
    _, _, gs = first_arrival_grid(model, [cake.PhaseDef(d) for d in ("s", "S")])
    grids = {"P": gp, "S": gs}

    print("reading differential-time files ...")
    dts = {args.dtcc: read_dt(os.path.join(work, "input_files", args.dtcc), True),
           "dt.ct": read_dt(os.path.join(work, "input_files", "dt.ct"), False)}
    for name, d in dts.items():
        vc = d.ph.value_counts()
        print(f"  {name}: {len(d):,} observations — "
              + ", ".join(f"{k} {v:,} ({100*v/len(d):.1f}%)" for k, v in vc.items()))

    out = []
    for cl in args.clusters:
        ids_file = os.path.join(work, f"{args.array.lower()}_cluster{cl}_event_ids.txt")
        if not os.path.exists(ids_file):
            continue
        with open(ids_file) as f:
            ids = set(int(l.split()[0]) for l in f if l.strip())
        g = rel[rel["id"].isin(ids)]
        if g.empty:
            continue
        z0 = float(g["depth"].median())
        cx, cy = float(g["x"].median()), float(g["y"].median())
        dist = {c: float(np.hypot(sxy[c][0] - cx, sxy[c][1] - cy)) for c in sxy}

        print(f"\n{'='*104}\ncluster {cl}: n={len(g)} events, hypoDD median depth {z0:.3f} km")
        print(f"{'file':7} {'sta':5} {'ph':2} {'r_km':>6} {'n_obs':>9} {'w_mean':>7} "
              f"{'dT/dz':>8} {'dT/dr':>8} {'I_z share':>10} {'I_r share':>10}")
        per_cluster = []
        for name, d in dts.items():
            sub = d[d.ev1.isin(ids) & d.ev2.isin(ids)]
            if sub.empty:
                continue
            recs = []
            for (code, ph), h in sub.groupby(["sta", "ph"]):
                if code not in dist:
                    continue
                r = np.array([dist[code]])
                pz, pr = partials(zs, rs, grids[ph], z0, r)
                if not (np.isfinite(pz[0]) and np.isfinite(pr[0])):
                    continue
                wsum = float(h["w"].sum())
                recs.append(dict(file=name, sta=code, ph=ph, r_km=dist[code], n=len(h),
                                 w_mean=float(h["w"].mean()),
                                 dTdz=float(pz[0]), dTdr=float(pr[0]),
                                 I_z=wsum * pz[0] ** 2, I_r=wsum * pr[0] ** 2, w_sum=wsum))
            if not recs:
                continue
            rdf = pd.DataFrame(recs)
            rdf["I_z_share"] = rdf.I_z / rdf.I_z.sum()
            rdf["I_r_share"] = rdf.I_r / rdf.I_r.sum()
            for r_ in rdf.sort_values("I_z", ascending=False).itertuples():
                print(f"{r_.file:7} {r_.sta:5} {r_.ph:2} {r_.r_km:6.2f} {r_.n:9,} "
                      f"{r_.w_mean:7.3f} {r_.dTdz:8.1f} {r_.dTdr:8.1f} "
                      f"{r_.I_z_share:9.1%} {r_.I_r_share:9.1%}")
            # Depth is only separable from origin time through VARIATION in dT/dz across the
            # contributing stations, so report the weighted spread, not just the magnitudes.
            for ph in sorted(rdf.ph.unique()):
                q = rdf[rdf.ph == ph]
                w = q.w_sum.values
                mu = float(np.average(q.dTdz, weights=w))
                sd = float(np.sqrt(np.average((q.dTdz - mu) ** 2, weights=w)))
                print(f"  -> {name} {ph}: weighted mean dT/dz = {mu:7.1f} ms/km, "
                      f"weighted spread across stations = {sd:6.1f} ms/km "
                      f"({100*sd/abs(mu) if mu else float('nan'):.1f}% of the mean)")
            rdf["cluster"] = cl
            per_cluster.append(rdf)
            out.extend(rdf.to_dict("records"))
        # ratio of total depth information between the two files
        tot = {n: float(p.I_z.sum()) for n, p in
               ((pc.file.iloc[0], pc) for pc in per_cluster)}
        if len(tot) == 2:
            a, b = args.dtcc, "dt.ct"
            if a in tot and b in tot:
                print(f"  -> total depth information: dt.cc {tot[a]:.3g} vs dt.ct {tot[b]:.3g} "
                      f"(dt.ct carries {tot[b]/tot[a]:.1f}x the CC data)")

    if out:
        csv = os.path.join(work, f"{args.array.lower()}_cc_depth_information.csv")
        pd.DataFrame(out).to_csv(csv, index=False)
        print(f"\nwrote {csv}")


if __name__ == "__main__":
    main()
