#!/usr/bin/env python3
"""Depth sections rotated into the shear margin's own frame (T1 or T2).

The array-centred E-W / N-S sections used so far are arbitrary with respect to the ice
dynamics. This rotates the catalogue into the margin frame instead, so "along the margin"
and "across the margin" become the section axes.

Defining the margin direction. The margin IS the line of highest surface-velocity gradient,
so the gradient vector of ITS_LIVE speed points ACROSS it and the margin trend is
perpendicular to that. Averaging gradient vectors directly is wrong -- a gradient and its
negative describe the same margin orientation, so opposite-facing pixels cancel. Instead the
dominant orientation comes from the structure tensor

    J = [[<gx^2>, <gx gy>], [<gx gy>, <gy^2>]]

whose principal eigenvector is the mean gradient AXIS (sign-free). That eigenvector is the
across-margin direction; its perpendicular is along-margin. Coherence (l1-l2)/(l1+l2) says
how well-defined the orientation is: ~1 means a clean linear margin, ~0 means no preferred
direction and the rotation is meaningless.

Everything is computed on the ITS_LIVE speed field smoothed at SMOOTH_KM, the same
smoothing plot_hypodd_t2_shear_margin_map.py uses, so the two figures are consistent.

Usage:
    python full_catalog_pipeline/plot_margin_oriented_sections.py --array T1
    ICEQUAKE_RELOC=hypodd_vels1d_ccstrong \\
        python full_catalog_pipeline/plot_margin_oriented_sections.py --array T2
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
import rasterio
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

import catalog_paths

import vels1d_model


def paths(array):
    """Per-array inputs. The ice-bed depth is vels1d_model's, not a second hard-coded copy.

    The ITS_LIVE window may have been cached under a SUPERSEDED run directory (T1's lives
    under hypodd/, not hypodd_vels1d/), so fall back across sibling run dirs rather than
    re-fetching or silently failing.
    """
    work = catalog_paths.work_dir(array)
    a = array.lower()
    vx = vy = None
    for cand in [work] + [os.path.join(os.path.dirname(work), d)
                          for d in ("hypodd", "hypodd_vels1d")]:
        p1 = os.path.join(cand, f"its_live_vx_{a}_window.tif")
        p2 = os.path.join(cand, f"its_live_vy_{a}_window.tif")
        if os.path.exists(p1) and os.path.exists(p2):
            vx, vy = p1, p2
            break
    return dict(reloc=catalog_paths.reloc(array),
                station_sel=catalog_paths.station_sel(array),
                out_dir=work, vx=vx, vy=vy,
                ice_bed_km=vels1d_model.ICE_THICKNESS_KM[array],
                out_png=os.path.join(work, f"{a}_margin_oriented_sections.png"))


SMOOTH_KM = 3.0
PIX_M = 120.0
SMOOTH_SIGMA_PX = SMOOTH_KM * 1000 / PIX_M
# Radius around the array centroid over which the margin orientation is averaged. Large
# enough to see the margin, small enough that it is the LOCAL orientation at T2.
ORIENT_RADIUS_KM = 8.0
SECTION_HALF_KM = 8.0

INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#dedddA", "#fcfcfb"
C_ACCENT = "#15616d"
# diverging: two hues with a NEUTRAL GREY midpoint (never a hue at the midpoint)
DIVERGING = LinearSegmentedColormap.from_list(
    "across", ["#2a78d6", "#b9b8b2", "#eb6834"])

RELOC_COLS = [
    "id", "lat", "lon", "depth", "x", "y", "z", "ex", "ey", "ez",
    "yr", "mo", "dy", "hr", "mi", "sc", "mag",
    "nccp", "nccs", "nctp", "ncts", "rcc", "rct", "cid",
]


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9)


def margin_axes(speed, bounds, cx, cy):
    """Across- and along-margin unit vectors from the speed-gradient structure tensor."""
    # gradient in map coordinates: rows increase downward (southward) in a north-up raster
    d_drow, d_dcol = np.gradient(speed, PIX_M)
    gx, gy = d_dcol, -d_drow

    nrow, ncol = speed.shape
    left, bottom, right, top = bounds
    xs = left + (np.arange(ncol) + 0.5) * PIX_M
    ys = top - (np.arange(nrow) + 0.5) * PIX_M
    X, Y = np.meshgrid(xs, ys)
    sel = np.hypot(X - cx, Y - cy) <= ORIENT_RADIUS_KM * 1000

    gxs, gys = gx[sel], gy[sel]
    ok = np.isfinite(gxs) & np.isfinite(gys)
    gxs, gys = gxs[ok], gys[ok]
    J = np.array([[np.mean(gxs * gxs), np.mean(gxs * gys)],
                  [np.mean(gxs * gys), np.mean(gys * gys)]])
    w, v = np.linalg.eigh(J)
    across = v[:, -1] / np.linalg.norm(v[:, -1])        # dominant gradient axis
    along = np.array([-across[1], across[0]])           # perpendicular to it
    coherence = float((w[-1] - w[0]) / (w[-1] + w[0]))
    return across, along, coherence, np.hypot(gx, gy), (X, Y)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--array", default="T2", choices=["T1", "T2"])
    args = ap.parse_args()
    P = paths(args.array)
    ICE_BED_KM = P["ice_bed_km"]
    VX_CACHE, VY_CACHE = P["vx"], P["vy"]
    OUT_PNG = P["out_png"]
    if not VX_CACHE:
        raise SystemExit(f"no ITS_LIVE cache for {args.array} -- run "
                         f"plot_hypodd_{args.array.lower()}_shear_margin_map.py to fetch it")
    print(f"{args.array}: reloc {P['reloc']}")
    print(f"{args.array}: ITS_LIVE {VX_CACHE}")
    print(f"{args.array}: ice-bed {ICE_BED_KM} km")

    ev = pd.read_csv(P["reloc"], sep=r"\s+", header=None, names=RELOC_COLS)
    sta = pd.read_csv(P["station_sel"], sep=r"\s+", header=None,
                      names=["id", "lat", "lon", "elev"])
    sta["code"] = sta["id"].str.split(".").str[1]

    to = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    ex, ey = to.transform(ev["lon"].values, ev["lat"].values)
    sx, sy = to.transform(sta["lon"].values, sta["lat"].values)
    cx, cy = sx.mean(), sy.mean()

    with rasterio.open(VX_CACHE) as ds:
        vx = ds.read(1).astype(float)
        bounds = tuple(ds.bounds)
    with rasterio.open(VY_CACHE) as ds:
        vy = ds.read(1).astype(float)
    vx[vx == -32767] = np.nan
    vy[vy == -32767] = np.nan
    speed = gaussian_filter(np.nan_to_num(np.hypot(vx, vy)), sigma=SMOOTH_SIGMA_PX)

    across, along, coh, gmag, (X, Y) = margin_axes(speed, bounds, cx, cy)
    az_across = np.degrees(np.arctan2(across[0], across[1])) % 180
    az_along = np.degrees(np.arctan2(along[0], along[1])) % 180
    print(f"margin orientation from velocity-gradient structure tensor "
          f"(r <= {ORIENT_RADIUS_KM:g} km):")
    print(f"   across-margin axis: {az_across:5.1f} deg   along-margin axis: {az_along:5.1f} deg")
    print(f"   orientation coherence: {coh:.3f}  (1 = one consistent gradient direction)")
    print(f"   CAUTION: high coherence means a consistent gradient DIRECTION, not a sharp")
    print(f"   margin -- a smooth regional ramp also scores ~1. Check the gradient range:")
    gsel = gmag[np.hypot(X - cx, Y - cy) <= ORIENT_RADIUS_KM * 1000] * 1000.0
    print(f"   |grad speed| over the fit region: median {np.median(gsel):.2f}, "
          f"p99 {np.percentile(gsel, 99):.2f} (m/yr)/km  -- ratio {np.percentile(gsel, 99)/max(np.median(gsel),1e-9):.1f}x")

    # plot to just below the bed or the deepest event, whichever is deeper -- T1's bed is at
    # 3.24 km and a hard-coded 3.1 km limit (fine for T2) silently clipped both it and the
    # events beneath it.
    zmax_plot = max(ICE_BED_KM * 1.08, float(np.percentile(ev["depth"], 99.5)))

    d = np.c_[ex - cx, ey - cy] / 1000.0
    s_across = d @ across
    s_along = d @ along
    st = np.c_[sx - cx, sy - cy] / 1000.0
    st_across, st_along = st @ across, st @ along

    fig = plt.figure(figsize=(15.5, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.30, wspace=0.22)

    # ---- map: speed field, margin axes, events -----------------------------------
    ax = fig.add_subplot(gs[:, 0])
    ext = [(bounds[0] - cx) / 1000, (bounds[2] - cx) / 1000,
           (bounds[1] - cy) / 1000, (bounds[3] - cy) / 1000]
    im = ax.imshow(gmag * 1000.0, extent=ext, origin="upper", cmap="Blues", zorder=1)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label(f"|grad speed| ((m/yr)/km), {SMOOTH_KM:g} km smoothed\n"
                 f"this is what defines the margin", fontsize=9, color=MUTED)
    ax.scatter(d[:, 0], d[:, 1], s=4, color="#eb6834", alpha=0.5, linewidths=0, zorder=4,
               label=f"relocated events (n={len(ev)})")
    ax.scatter(st[:, 0], st[:, 1], marker="v", s=70, color=INK, zorder=6, label="stations")
    L = SECTION_HALF_KM
    ax.plot([-along[0] * L, along[0] * L], [-along[1] * L, along[1] * L],
            color=INK, lw=2.2, zorder=5, label=f"along margin ({az_along:.0f}°)")
    ax.plot([-across[0] * L, across[0] * L], [-across[1] * L, across[1] * L],
            color=INK, lw=2.2, ls="--", zorder=5, label=f"across margin ({az_across:.0f}°)")
    ax.set_xlim(-14, 14)
    ax.set_ylim(-14, 14)
    ax.set_aspect("equal")
    ax.set_xlabel("east of array centroid (km)", fontsize=9.5, color=MUTED)
    ax.set_ylabel("north of array centroid (km)", fontsize=9.5, color=MUTED)
    ax.set_title(f"margin frame from the velocity gradient\n"
                 f"orientation coherence {coh:.2f}", fontsize=11.5, color=INK, loc="left")
    style(ax)
    ax.legend(fontsize=8, frameon=True, facecolor=SURF, edgecolor=MUTED, loc="lower left")

    lim = max(abs(np.percentile(s_across, [1, 99]))) * 1.05

    # ---- section ALONG the margin -------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    sc = ax.scatter(s_along, ev["depth"], c=s_across, cmap=DIVERGING,
                    vmin=-lim, vmax=lim, s=6, alpha=0.6, linewidths=0, zorder=3)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.015)
    cb.set_label("across-margin position (km)", fontsize=8.5, color=MUTED)
    ax.axhline(ICE_BED_KM, color=INK, ls="--", lw=1.2, zorder=5,
               label=f"ice-bed ({ICE_BED_KM} km)")
    for i, (a, c) in enumerate(zip(st_along, sta["code"])):
        ax.annotate(c, (a, 0.02 + 0.13 * (i % 3)), fontsize=7, color=MUTED,
                    ha="center", va="top")
    ax.set_ylim(zmax_plot, -0.05 * zmax_plot)
    ax.set_xlim(-SECTION_HALF_KM, SECTION_HALF_KM)
    ax.set_xlabel(f"distance ALONG margin (km, axis {az_along:.0f}°)",
                  fontsize=9.5, color=MUTED)
    ax.set_ylabel("depth (km)", fontsize=9.5, color=MUTED)
    ax.set_title("section PARALLEL to the margin", fontsize=11.5, color=INK, loc="left")
    style(ax)
    ax.legend(fontsize=8, frameon=False, loc="lower left")

    # ---- section ACROSS the margin ------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(s_across, ev["depth"], s=6, color=C_ACCENT, alpha=0.5, linewidths=0, zorder=3)
    ax.axhline(ICE_BED_KM, color=INK, ls="--", lw=1.2, zorder=5,
               label=f"ice-bed ({ICE_BED_KM} km)")
    ax.axvline(0, color=MUTED, lw=0.9, zorder=2)
    for i, (a, c) in enumerate(zip(st_across, sta["code"])):
        ax.annotate(c, (a, 0.02 + 0.13 * (i % 3)), fontsize=7, color=MUTED,
                    ha="center", va="top")
    ax.set_ylim(zmax_plot, -0.05 * zmax_plot)
    ax.set_xlim(-SECTION_HALF_KM, SECTION_HALF_KM)
    ax.set_xlabel(f"distance ACROSS margin (km, axis {az_across:.0f}°)\n"
                  f"increasing toward higher surface speed", fontsize=9.5, color=MUTED)
    ax.set_ylabel("depth (km)", fontsize=9.5, color=MUTED)
    ax.set_title("section PERPENDICULAR to the margin", fontsize=11.5, color=INK, loc="left")
    style(ax)
    ax.legend(fontsize=8, frameon=False, loc="lower left")

    run_name = os.path.basename(os.path.dirname(os.path.dirname(P["reloc"])))
    fig.suptitle(f"{args.array} seismicity in the shear-margin frame — {run_name}",
                 fontsize=13.5, color=INK, y=0.965)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=SURF)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
