#!/usr/bin/env python3
"""Discriminate between hypoDD configs that are equally stable but disagree about ABSOLUTE
depth, using observed S-P times.

The tuning sweep left two configs that both hold still under data perturbation but place T2's
basal peak 170 m apart (1.84 vs 2.01 km, either side of the 2.02 km ice-bed). Stability cannot
settle that: each is internally consistent. Neither can hypoDD's own residuals -- RMSCT/RMSCC
are differential (a common depth error cancels) and RMSST is 11-13 ms for every candidate.

S-P does settle it. The S-P interval depends only on hypocentral DISTANCE and the velocity
model -- the origin time cancels exactly -- so it is an absolute constraint on depth that the
double-difference inversion never used. For an event 4 km out at 2 km depth, a 170 m depth
change moves S-P by ~20 ms, which is resolvable across thousands of observations even though
it is at the level of a single pick's precision.

Predicted traveltimes come from pyrocko.cake on the same reflection-seismology layered model
the relocation used (vels1d_model.build_layers), precomputed on a distance-depth grid and
bilinearly interpolated.

Usage:
    python full_catalog_pipeline/test_sp_absolute_depth.py --array T2 \
        --configs clean_flat30 clean_0.2_0.5 dtmax_0.5 base
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

from hypodd_tune import RELOC_COLS, tune_root
from vels1d_model import build_layers

KM = 1000.0
DIST_MAX_KM, DIST_STEP_KM = 9.0, 0.1
DEPTH_MAX_KM, DEPTH_STEP_KM = 5.0, 0.05


def build_cake_model(array):
    """pyrocko.cake model from the same layers hypoDD was given."""
    layers = build_layers(array)
    lines = []
    for i, (top, vp, vs) in enumerate(layers):
        bot = layers[i + 1][0] if i + 1 < len(layers) else DEPTH_MAX_KM + 6.0
        if top >= DEPTH_MAX_KM + 5.0:
            break
        # cake's nd format: depth[km] vp[km/s] vs[km/s] rho -- constant within each layer, so
        # the top and bottom of every layer carry the same velocities (a stack of homogeneous
        # layers, exactly what hypoDD's ttime assumes).
        lines.append(f"{top:.4f} {vp:.4f} {vs:.4f} 0.917")
        lines.append(f"{bot:.4f} {vp:.4f} {vs:.4f} 0.917")
    return cake.LayeredModel.from_scanlines(_scanlines("\n".join(lines)))


def _scanlines(text):
    """cake.LayeredModel.from_scanlines wants (depth_m, Material, name) triples."""
    for line in text.strip().splitlines():
        z, vp, vs, rho = (float(x) for x in line.split())
        yield z * KM, cake.Material(vp=vp * KM, vs=vs * KM, rho=rho * KM), None


def first_arrival_grid(model, phase_defs):
    """First-arrival traveltime on a (depth, distance) grid, seconds. NaN where no ray."""
    depths = np.arange(0.0, DEPTH_MAX_KM + 1e-9, DEPTH_STEP_KM)
    dists = np.arange(0.0, DIST_MAX_KM + 1e-9, DIST_STEP_KM)
    grid = np.full((len(depths), len(dists)), np.nan)
    dist_deg = dists * KM * cake.m2d
    for i, z in enumerate(depths):
        for j, dd in enumerate(dist_deg):
            best = np.inf
            for ray in model.arrivals([dd], phases=phase_defs, zstart=z * KM):
                best = min(best, ray.t)
            if np.isfinite(best):
                grid[i, j] = best
    return depths, dists, grid


def interp(depths, dists, grid, z, r):
    """Bilinear lookup, vectorised."""
    zi = np.clip((z - depths[0]) / (depths[1] - depths[0]), 0, len(depths) - 1.001)
    ri = np.clip((r - dists[0]) / (dists[1] - dists[0]), 0, len(dists) - 1.001)
    z0, r0 = zi.astype(int), ri.astype(int)
    fz, fr = zi - z0, ri - r0
    g = grid
    return ((1 - fz) * (1 - fr) * g[z0, r0] + fz * (1 - fr) * g[z0 + 1, r0]
            + (1 - fz) * fr * g[z0, r0 + 1] + fz * fr * g[z0 + 1, r0 + 1])


def load_picks(phase_dat):
    """phase.dat -> DataFrame(id, sta, phase, tt). Times are already relative to the origin."""
    rows, eid = [], None
    with open(phase_dat) as f:
        for line in f:
            if line.startswith("#"):
                eid = int(line.split()[-1])
            else:
                p = line.split()
                if len(p) >= 4:
                    rows.append((eid, p[0], p[3], float(p[1])))
    return pd.DataFrame(rows, columns=["id", "sta", "phase", "tt"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--array", default="T2", choices=["T1", "T2"])
    ap.add_argument("--configs", nargs="+", required=True)
    args = ap.parse_args()

    root = tune_root(args.array)
    inp = os.path.join(os.path.dirname(root), "input_files")

    print("building traveltime grids from the reflection model ...")
    model = build_cake_model(args.array)
    zs, rs, gp = first_arrival_grid(model, [cake.PhaseDef(d) for d in ("p", "P")])
    _, _, gs = first_arrival_grid(model, [cake.PhaseDef(d) for d in ("s", "S")])
    print(f"  grid {gp.shape}, P coverage {100*np.isfinite(gp).mean():.0f}%, "
          f"S coverage {100*np.isfinite(gs).mean():.0f}%")

    sta = pd.read_csv(os.path.join(inp, "station.sel"), sep=r"\s+", header=None,
                      names=["sta", "lat", "lon", "elev"])
    to_ps = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    sx, sy = to_ps.transform(sta["lon"].values, sta["lat"].values)
    sta["x"], sta["y"] = sx, sy

    picks = load_picks(os.path.join(inp, "phase.dat"))
    p = picks[picks["phase"] == "P"].rename(columns={"tt": "tp"})[["id", "sta", "tp"]]
    s = picks[picks["phase"] == "S"].rename(columns={"tt": "ts"})[["id", "sta", "ts"]]
    sp = p.merge(s, on=["id", "sta"]).merge(sta[["sta", "x", "y"]], on="sta")
    sp["sp_obs"] = sp["ts"] - sp["tp"]
    sp = sp[(sp["sp_obs"] > 0) & (sp["sp_obs"] < 5)]
    print(f"{len(sp)} station-event pairs with both P and S\n")

    # A raw S-P misfit cannot tell a depth error from a Vp/Vs error: every config here
    # over-predicts S-P, so the shallowest one wins on raw misfit whether or not its depths
    # are right. S-P scales as r*(1/Vs - 1/Vp), so a ratio error is a pure MULTIPLE of the
    # prediction while a depth error is not (depth enters via r = hypot(epi, z), so it moves
    # near-station observations far more than distant ones). Fitting and removing a single
    # robust scale factor k per config absorbs the ratio error; what survives is the
    # depth-sensitive part, and that is what discriminates.
    print(f"{'config':16s} {'n':>7s} {'raw|r|':>8s} {'bias':>8s} | {'k':>6s} "
          f"{'scaled|r|':>10s} {'MAD':>7s} | {'near|r|':>8s} {'far|r|':>8s}")
    print(f"{'':16s} {'':>7s} {'ms':>8s} {'ms':>8s} | {'':>6s} {'ms':>10s} {'ms':>7s} | "
          f"{'<2.5km':>8s} {'>4km':>8s}")
    for name in args.configs:
        reloc = os.path.join(root, name, "prod", "hypoDD.reloc")
        if not (os.path.exists(reloc) and os.path.getsize(reloc) > 0):
            print(f"{name:16s} (no relocation)")
            continue
        ev = pd.read_csv(reloc, sep=r"\s+", header=None, names=RELOC_COLS)[
            ["id", "lat", "lon", "depth"]]
        ex, ey = to_ps.transform(ev["lon"].values, ev["lat"].values)
        ev["x"], ev["y"] = ex, ey
        m = sp.merge(ev, on="id", suffixes=("_s", "_e"))
        r_km = np.hypot(m["x_e"] - m["x_s"], m["y_e"] - m["y_s"]) / KM
        z_km = m["depth"].values
        keep = (r_km <= DIST_MAX_KM - 0.2) & (z_km >= 0) & (z_km <= DEPTH_MAX_KM - 0.1)
        r_km, z_km, obs = r_km[keep], z_km[keep], m["sp_obs"].values[keep]
        pred = interp(zs, rs, gs, z_km, r_km) - interp(zs, rs, gp, z_km, r_km)
        good = np.isfinite(pred)
        o, pr, rr = obs[good], pred[good], r_km[good]
        resid = (o - pr) * 1000.0
        k = float(np.median(o / pr))                    # robust ratio correction
        scaled = (o - k * pr) * 1000.0
        near, far = rr < 2.5, rr > 4.0
        print(f"{name:16s} {good.sum():7d} {np.median(np.abs(resid)):8.1f} "
              f"{np.median(resid):8.1f} | {k:6.4f} {np.median(np.abs(scaled)):10.1f} "
              f"{np.median(np.abs(scaled - np.median(scaled))):7.1f} | "
              f"{np.median(np.abs(scaled[near])):8.1f} {np.median(np.abs(scaled[far])):8.1f}")


if __name__ == "__main__":
    main()
