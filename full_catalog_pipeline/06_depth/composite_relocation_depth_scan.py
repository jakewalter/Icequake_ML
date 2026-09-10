#!/usr/bin/env python3
"""Test 2 of PLAN_depth_resolvability.md: depth-pinned composite relocation, epicentre free.

Test 5 (composite_sp_depth_pin.py) already scanned depth using the S-P vector alone, holding
the epicentre at hypoDD's. That is the origin-time-free formulation and it produced sharp RMS
minima. This is the complementary one the plan asks for: treat each cluster as a single
composite event and, at every trial depth, re-solve for the epicentre AND origin time before
scoring the fit. The two tests share the observations but not the assumptions, and the plan is
explicit that agreement between them is worth more than either alone.

What freeing the epicentre buys. A depth minimum found with x and y nailed down can be an
artifact of exactly that: if the assumed epicentre is wrong, depth is the only parameter left
to absorb the error, and the curve develops a minimum that is really about horizontal position.
Letting the epicentre move at every trial depth removes that escape route -- whatever depth
structure survives is not a mis-located epicentre in disguise.

Observations. Composite CC-refined P and S arrival times per station (Test 1), each still
measured from its event's hypoDD origin. Those origins are a single unknown constant per
cluster, absorbed exactly by the free origin-time parameter, so no absolute timing is assumed;
what the data carry is the P moveout across the array plus S-P at each station. As in Test 5 a
single multiplicative velocity scale k is profiled out at every depth, so a mis-calibrated
Vp/Vs cannot manufacture a minimum.

Reads. A sharp RMS(z) minimum that survives the free epicentre means depth is resolved by the
traveltimes. A curve that flattens once x and y are free means Test 5's minimum was borrowing
resolution from the fixed epicentre. Reported against the ice base and hypoDD's own median.

Usage:
    python full_catalog_pipeline/composite_relocation_depth_scan.py --array T2
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
from pyrocko import cake
from scipy.optimize import minimize

import catalog_paths
import vels1d_model
from composite_sp_depth_pin import SE_FLOOR_MS, Z_MIN, Z_MAX, Z_STEP
from lib.deej_waveform_common import RELOC_COLS
from test_sp_absolute_depth import build_cake_model, first_arrival_grid, interp

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


def station_xy(station_sel):
    """Station code -> (x, y) km in Antarctic Polar Stereographic, the same projection
    load_dist_geodetic uses so distances stay consistent with Tests 1 and 5."""
    sta = pd.read_csv(station_sel, sep=r"\s+", header=None,
                      names=["sta", "lat", "lon", "elev"])
    sta["code"] = sta["sta"].str.split(".").str[-1]
    to_ps = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    x, y = to_ps.transform(sta["lon"].values, sta["lat"].values)
    return {c: (xx / 1000.0, yy / 1000.0) for c, xx, yy in zip(sta["code"], x, y)}


def cluster_centroid_xy(reloc_file, ids):
    n_fields = len(open(reloc_file).readline().split())
    cols = RELOC_COLS if n_fields == 18 else RELOC_COLS[:17] + [
        "nccp", "nccs", "nctp", "ncts", "rcc", "rct", "cid"]
    rel = pd.read_csv(reloc_file, sep=r"\s+", header=None, names=cols)
    rel = rel[rel["id"].isin(ids)]
    to_ps = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    x, y = to_ps.transform(rel["lon"].values, rel["lat"].values)
    return float(np.median(x)) / 1000.0, float(np.median(y)) / 1000.0


def combine(df, cl):
    """Inverse-variance combine P and S composite offsets across components, per station."""
    g = df[df.cluster == cl]
    rows = []
    for sta, h in g.groupby("station"):
        wp = 1.0 / np.maximum(h["p_se_ms"].values, SE_FLOOR_MS) ** 2
        ws = 1.0 / np.maximum(h["s_se_ms"].values, SE_FLOOR_MS) ** 2
        rows.append(dict(
            station=sta,
            p_ms=float(np.average(h["p_composite"].values * 1000.0, weights=wp)),
            s_ms=float(np.average(h["s_composite"].values * 1000.0, weights=ws)),
            p_se_ms=float(1.0 / np.sqrt(wp.sum())),
            s_se_ms=float(1.0 / np.sqrt(ws.sum())),
            depth_hypodd=float(h["depth_hypodd"].iloc[0]),
        ))
    return pd.DataFrame(rows)


def misfit_at_depth(z, obs, sxy, x0y0, zs, rs, gp, gs):
    """Weighted RMS residual (ms) with the epicentre, origin time and velocity scale all free
    at this fixed depth. Returns (rms, x, y, k)."""
    o = np.concatenate([obs["p_ms"].values, obs["s_ms"].values])
    se = np.concatenate([np.maximum(obs["p_se_ms"].values, SE_FLOOR_MS),
                         np.maximum(obs["s_se_ms"].values, SE_FLOOR_MS)])
    w = 1.0 / se ** 2
    sx = np.array([sxy[c][0] for c in obs["station"]])
    sy = np.array([sxy[c][1] for c in obs["station"]])

    def resid_rms(xy):
        r = np.hypot(sx - xy[0], sy - xy[1])
        zz = np.full_like(r, float(z))
        tp = interp(zs, rs, gp, zz, r) * 1000.0
        ts = interp(zs, rs, gs, zz, r) * 1000.0
        pred = np.concatenate([tp, ts])
        if not np.all(np.isfinite(pred)) or np.any(pred <= 0):
            return np.inf, np.nan, np.nan
        # Profile k (velocity scale) and t0 (origin) out together in closed form: with
        # pred_i scaled by k and shifted by t0, the weighted least-squares solution for
        # (k, t0) is a 2x2 linear system.
        s11 = np.sum(w * pred * pred)
        s12 = np.sum(w * pred)
        s22 = np.sum(w)
        b1 = np.sum(w * o * pred)
        b2 = np.sum(w * o)
        det = s11 * s22 - s12 * s12
        if abs(det) < 1e-12:
            return np.inf, np.nan, np.nan
        k = (b1 * s22 - b2 * s12) / det
        t0 = (s11 * b2 - s12 * b1) / det
        res = o - (k * pred + t0)
        return float(np.sqrt(np.sum(w * res ** 2) / np.sum(w))), k, t0

    best = minimize(lambda xy: resid_rms(xy)[0], x0=np.asarray(x0y0, float),
                    method="Nelder-Mead",
                    options=dict(xatol=1e-3, fatol=1e-4, maxiter=2000))
    rms, k, _ = resid_rms(best.x)
    return rms, float(best.x[0]), float(best.x[1]), k


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--array", default="T2", choices=["T1", "T2"])
    ap.add_argument("--clusters", nargs="+", type=int, default=None)
    args = ap.parse_args()

    work = catalog_paths.work_dir(args.array)
    df = pd.read_csv(os.path.join(
        work, f"{args.array.lower()}_cluster_composite_arrivals.csv"))
    if "p_composite" not in df.columns:
        raise SystemExit("arrivals CSV predates the composite P/S columns -- "
                         "re-run build_cluster_stacks.py first")
    ice_base = vels1d_model.bed_markers(args.array)["ice_base"]
    sxy = station_xy(catalog_paths.station_sel(args.array))
    reloc_path = catalog_paths.reloc(args.array)

    print("building traveltime grids from the reflection model ...")
    model = build_cake_model(args.array)
    zs, rs, gp = first_arrival_grid(model, [cake.PhaseDef(d) for d in ("p", "P")])
    _, _, gs = first_arrival_grid(model, [cake.PhaseDef(d) for d in ("s", "S")])
    trial_z = np.arange(Z_MIN, Z_MAX + 1e-9, Z_STEP)

    clusters = args.clusters or sorted(df.cluster.unique())
    summary, curves = [], []
    for cl in clusters:
        obs = combine(df, cl)
        if len(obs) < 4:
            print(f"cluster {cl}: {len(obs)} stations, need 4 for a free epicentre, skipped")
            continue
        ids_file = os.path.join(work, f"{args.array.lower()}_cluster{cl}_event_ids.txt")
        with open(ids_file) as f:
            ids = [int(l.split()[0]) for l in f if l.strip()]
        x0, y0 = cluster_centroid_xy(reloc_path, ids)
        z_dd = float(obs["depth_hypodd"].median())

        rms = np.full(len(trial_z), np.nan)
        moved = np.full(len(trial_z), np.nan)
        for i, z in enumerate(trial_z):
            r, x, y, k = misfit_at_depth(z, obs, sxy, (x0, y0), zs, rs, gp, gs)
            if np.isfinite(r):
                rms[i] = r
                moved[i] = np.hypot(x - x0, y - y0)
        if not np.any(np.isfinite(rms)):
            print(f"cluster {cl}: no traveltime coverage, skipped")
            continue
        j = int(np.nanargmin(rms))
        z_best, rms_best = trial_z[j], rms[j]
        flat = trial_z[np.isfinite(rms) & (rms < 1.2 * rms_best)]
        i_base = int(np.argmin(np.abs(trial_z - ice_base)))
        i_dd = int(np.argmin(np.abs(trial_z - z_dd)))
        summary.append(dict(
            cluster=cl, n_stations=len(obs), z_best_km=float(z_best),
            rms_best_ms=float(rms_best),
            epicentre_moved_km=float(moved[j]),
            flat_span_lo_km=float(flat.min()), flat_span_hi_km=float(flat.max()),
            rms_at_ice_base_ms=float(rms[i_base]),
            depth_hypodd_median_km=z_dd, rms_at_hypodd_ms=float(rms[i_dd]),
        ))
        curves.append((cl, trial_z, rms, z_dd))
        print(f"cluster {cl} ({len(obs)} stations): RMS(z) min at z={z_best:.3f} km "
              f"(RMS={rms_best:.1f} ms, epicentre moved {moved[j]*1000:.0f} m)")
        print(f"  RMS at ice base ({ice_base:.2f} km) = {rms[i_base]:.1f} ms; "
              f"at hypoDD median ({z_dd:.3f} km) = {rms[i_dd]:.1f} ms; "
              f"flat span (RMS < 1.2x min) = [{flat.min():.2f}, {flat.max():.2f}] km")

    if not summary:
        print("nothing to plot")
        return

    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.6), sharey=False, squeeze=False)
    for ax, (cl, tz, rr, z_dd) in zip(axes[0], curves):
        ax.plot(tz, rr, color=SERIES[0], lw=1.6)
        ax.axvline(ice_base, color=INK, lw=1.2)
        ax.axvline(z_dd, color=MUTED, lw=1.2, ls=":")
        ax.set_title(f"cluster {cl}", fontsize=10)
        ax.set_xlabel("trial depth (km)")
    axes[0][0].set_ylabel("weighted RMS residual (ms), epicentre free")
    fig.suptitle(f"{args.array} — composite relocation depth scan with the epicentre re-solved "
                 f"at every depth (solid: ice base, dotted: hypoDD median)",
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(work, f"{args.array.lower()}_composite_relocation_depth_scan.png")
    fig.savefig(out, dpi=145, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")
    csv = os.path.join(work, f"{args.array.lower()}_composite_relocation_depth_scan.csv")
    pd.DataFrame(summary).to_csv(csv, index=False)
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
