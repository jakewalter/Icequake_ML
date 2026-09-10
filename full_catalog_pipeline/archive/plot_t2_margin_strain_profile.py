#!/usr/bin/env python3
"""Seismicity and surface strain rate as a function of across-margin distance, T2.

Companion to plot_t2_margin_oriented_sections.py, which establishes the margin frame from
the structure tensor of the ITS_LIVE speed gradient. This projects both the catalogue and
the strain-rate field onto the across-margin axis so the two profiles can be read against
each other -- does seismicity track strain?

The two quantities share an x axis and are drawn as STACKED PANELS, never as one plot with
two y scales. Events-per-km and 1/yr are incommensurate; overlaying them on twin axes lets
the reader's eye "see" a correlation that is an artifact of arbitrary axis scaling.

The strain profile averages over an along-margin swath (+-SWATH_KM) so it is a genuine
1D profile of the field rather than a single transect through one pixel row.

Usage:
    ICEQUAKE_RELOC=hypodd_vels1d_ccstrong \\
        python full_catalog_pipeline/plot_t2_margin_strain_profile.py
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

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr

import catalog_paths
from plot_t2_margin_oriented_sections import (
    RELOC_COLS, ICE_BED_KM, SMOOTH_KM, PIX_M, SMOOTH_SIGMA_PX,
    ORIENT_RADIUS_KM, margin_axes, style, INK, MUTED, GRID, SURF,
)

RELOC = catalog_paths.reloc("T2")
STATION_SEL = catalog_paths.station_sel("T2")
OUT_DIR = catalog_paths.work_dir("T2")
OUT_PNG = f"{OUT_DIR}/t2_margin_strain_seismicity_profile.png"
VX_CACHE = f"{OUT_DIR}/its_live_vx_t2_window.tif"
VY_CACHE = f"{OUT_DIR}/its_live_vy_t2_window.tif"

SWATH_KM = 6.0      # along-margin half-width the strain profile averages over
BIN_KM = 0.25
HALF_KM = 8.0

C_SEIS = "#eb6834"
C_STRAIN = "#2a78d6"


def effective_strain_rate(vx, vy):
    """Second invariant of the horizontal strain-rate tensor (1/yr), as in the margin map."""
    vxs = gaussian_filter(np.nan_to_num(vx), sigma=SMOOTH_SIGMA_PX)
    vys = gaussian_filter(np.nan_to_num(vy), sigma=SMOOTH_SIGMA_PX)
    d_vy_drow, d_vy_dcol = np.gradient(vys, PIX_M)
    d_vx_drow, d_vx_dcol = np.gradient(vxs, PIX_M)
    exx = d_vx_dcol
    eyy = -d_vy_drow
    exy = 0.5 * (-d_vx_drow + d_vy_dcol)
    return np.sqrt(exx ** 2 + eyy ** 2 + exx * eyy + exy ** 2)


def main():
    ev = pd.read_csv(RELOC, sep=r"\s+", header=None, names=RELOC_COLS)
    sta = pd.read_csv(STATION_SEL, sep=r"\s+", header=None,
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
    strain = effective_strain_rate(vx, vy)

    across, along, coh, gmag, (X, Y) = margin_axes(speed, bounds, cx, cy)
    az_across = np.degrees(np.arctan2(across[0], across[1])) % 180

    # project the raster onto the margin frame
    px = (X - cx) / 1000.0
    py = (Y - cy) / 1000.0
    r_across = px * across[0] + py * across[1]
    r_along = px * along[0] + py * along[1]
    swath = np.abs(r_along) <= SWATH_KM

    # project events
    d = np.c_[ex - cx, ey - cy] / 1000.0
    s_across = d @ across
    s_along = d @ along
    in_swath = np.abs(s_along) <= SWATH_KM

    edges = np.arange(-HALF_KM, HALF_KM + BIN_KM, BIN_KM)
    centres = 0.5 * (edges[:-1] + edges[1:])
    counts, _ = np.histogram(s_across[in_swath], bins=edges)

    prof_med, prof_lo, prof_hi, prof_speed = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = swath & (r_across >= a) & (r_across < b)
        v = strain[m]
        v = v[np.isfinite(v)]
        s = speed[m]
        if v.size:
            prof_med.append(np.median(v))
            prof_lo.append(np.percentile(v, 25))
            prof_hi.append(np.percentile(v, 75))
            prof_speed.append(np.median(s[np.isfinite(s)]) if s.size else np.nan)
        else:
            prof_med.append(np.nan); prof_lo.append(np.nan)
            prof_hi.append(np.nan); prof_speed.append(np.nan)
    prof_med = np.array(prof_med); prof_lo = np.array(prof_lo)
    prof_hi = np.array(prof_hi); prof_speed = np.array(prof_speed)

    ok = np.isfinite(prof_med) & (counts > 0)
    rp, pp = pearsonr(prof_med[ok], counts[ok]) if ok.sum() > 3 else (np.nan, np.nan)
    rs, ps = spearmanr(prof_med[ok], counts[ok]) if ok.sum() > 3 else (np.nan, np.nan)

    print(f"across-margin axis {az_across:.1f} deg, swath +-{SWATH_KM:g} km, bin {BIN_KM} km")
    print(f"events in swath: {int(in_swath.sum())} of {len(ev)}")
    print(f"strain rate over profile: {np.nanmin(prof_med):.2e} to {np.nanmax(prof_med):.2e} 1/yr "
          f"({np.nanmax(prof_med)/max(np.nanmin(prof_med),1e-12):.1f}x range)")
    print(f"seismicity vs strain, per bin (n={int(ok.sum())} bins):")
    print(f"   Pearson  r={rp:+.3f} (p={pp:.3g})")
    print(f"   Spearman r={rs:+.3f} (p={ps:.3g})")

    fig, axes = plt.subplots(3, 1, figsize=(11, 10.5), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.0, 0.85, 1.15], hspace=0.12))

    # ---- seismicity histogram -----------------------------------------------------
    ax = axes[0]
    ax.bar(centres, counts, width=BIN_KM * 0.92, color=C_SEIS, zorder=3)
    ax.set_ylabel(f"events per {BIN_KM} km bin", fontsize=9.5, color=MUTED)
    ax.set_title(f"seismicity across the margin   (n={int(in_swath.sum())} events "
                 f"within ±{SWATH_KM:g} km along-margin swath)",
                 fontsize=11.5, color=INK, loc="left")
    style(ax)

    # ---- strain-rate profile ------------------------------------------------------
    ax = axes[1]
    ax.fill_between(centres, prof_lo, prof_hi, color=C_STRAIN, alpha=0.22, zorder=2,
                    label="inter-quartile range across the swath")
    ax.plot(centres, prof_med, color=C_STRAIN, lw=2.0, zorder=3, label="median strain rate")
    ax.set_ylabel("effective strain rate (1/yr)", fontsize=9.5, color=MUTED)
    ax.set_title(f"surface strain rate across the margin   "
                 f"(ITS_LIVE, {SMOOTH_KM:g} km smoothed)",
                 fontsize=11.5, color=INK, loc="left")
    style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    # ---- depth section for reference ----------------------------------------------
    ax = axes[2]
    ax.scatter(s_across[in_swath], ev["depth"].values[in_swath], s=5, color="#15616d",
               alpha=0.45, linewidths=0, zorder=3)
    ax.axhline(ICE_BED_KM, color=INK, ls="--", lw=1.2, zorder=5,
               label=f"ice-bed ({ICE_BED_KM} km)")
    ax.set_ylim(3.1, -0.15)
    ax.set_ylabel("depth (km)", fontsize=9.5, color=MUTED)
    ax.set_xlabel(f"distance ACROSS margin (km, axis {az_across:.0f}°)  —  "
                  f"increasing toward higher surface speed", fontsize=10, color=MUTED)
    ax.set_title("depth section, same swath", fontsize=11.5, color=INK, loc="left")
    style(ax)
    ax.legend(fontsize=8, frameon=False, loc="lower left")

    axes[0].set_xlim(-HALF_KM, HALF_KM)
    fig.suptitle("T2: does seismicity track surface strain across the margin?",
                 fontsize=13.5, color=INK, y=0.945)
    fig.text(0.5, 0.915,
             f"per-bin correlation  Pearson r = {rp:+.2f} (p = {pp:.2g})   ·   "
             f"Spearman r = {rs:+.2f} (p = {ps:.2g})",
             ha="center", fontsize=10, color=MUTED)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor=SURF)
    print(f"wrote {OUT_PNG}")

    out = pd.DataFrame(dict(across_km=centres, n_events=counts,
                            strain_median=prof_med, strain_q25=prof_lo,
                            strain_q75=prof_hi, speed_median=prof_speed))
    csv = f"{OUT_DIR}/t2_margin_strain_seismicity_profile.csv"
    out.to_csv(csv, index=False)
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
