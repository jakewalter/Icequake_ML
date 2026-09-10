#!/usr/bin/env python3
"""T2 port of plot_hypodd_t1_shear_margin_map.py: relocation map with ice-flow
context (ITS_LIVE effective strain rate), to check whether T2 -- like T1 --
sits on an ice-stream shear margin, and what flow speed it implies for basal
event nature (fast shear-margin ice favors basal stick-slip; slow, more
uniform flow is less obviously consistent with that).

Unlike the T1 script, there's no pre-existing T2 displacement-map crop box to
borrow xlim/ylim from, so the plotted extent is derived directly from the
event+station data's own footprint (with padding), not a manually-captured
box from another figure.

Usage:
    python full_catalog_pipeline/plot_hypodd_t2_shear_margin_map.py
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

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio
from matplotlib.colors import LogNorm
from rasterio.windows import from_bounds
from scipy.ndimage import gaussian_filter

import catalog_paths

PRE_CSV = "full_catalog_pipeline/artifacts/full_run/T2_v5/pyocto_events.csv"
# Paths come from catalog_paths so the authoritative relocation is named in ONE place
# (see that module). Set ICEQUAKE_RELOC=hypodd to reproduce a figure against the
# superseded v5 relocation without editing anything.
RELOC = catalog_paths.reloc("T2")
STATION_SEL = catalog_paths.station_sel("T2")
OUT_DIR = catalog_paths.work_dir("T2")
OUT_PNG = f"{OUT_DIR}/hypodd_t2_reloc_shear_margin_map.png"

ITS_LIVE_BASE = "https://its-live-data.s3.amazonaws.com/velocity_mosaic/v2/static/cog"
VX_URL = f"/vsicurl/{ITS_LIVE_BASE}/ITS_LIVE_velocity_120m_RGI19A_0000_v02_vx.tif"
VY_URL = f"/vsicurl/{ITS_LIVE_BASE}/ITS_LIVE_velocity_120m_RGI19A_0000_v02_vy.tif"
VX_CACHE = f"{OUT_DIR}/its_live_vx_t2_window.tif"
VY_CACHE = f"{OUT_DIR}/its_live_vy_t2_window.tif"
HALF_WIDTH_M = 50_000  # 100 km fetch box
SMOOTH_KM = 3.0
SMOOTH_SIGMA_PX = SMOOTH_KM * 1000 / 120
PAD_KM = 3.0  # padding around the event+station footprint for the plotted crop

COLOR_PRE = "#e1e0d9"
COLOR_POST = "#eb6834"
COLOR_STATION = "#0b0b0b"

RELOC_COLS = [
    "id", "lat", "lon", "depth", "x", "y", "z", "ex", "ey", "ez",
    "yr", "mo", "dy", "hr", "mi", "sc", "mag",
    "nccp", "nccs", "nctp", "ncts", "rcc", "rct", "cid",
]


def load_stations():
    stations = pd.read_csv(STATION_SEL, sep=r"\s+", header=None,
                            names=["id", "lat", "lon", "elev"])
    stations["code"] = stations["id"].str.split(".").str[1]
    return stations


def load_events():
    pre = pd.read_csv(PRE_CSV).rename(columns={"latitude": "lat", "longitude": "lon"})
    post = pd.read_csv(RELOC, sep=r"\s+", header=None, names=RELOC_COLS)
    return pre, post


def fetch_component_window(url, cache_path, cx, cy):
    if os.path.exists(cache_path):
        with rasterio.open(cache_path) as ds:
            data = ds.read(1)
            bounds = tuple(ds.bounds)
        return data, bounds

    with rasterio.open(url) as ds:
        window = from_bounds(cx - HALF_WIDTH_M, cy - HALF_WIDTH_M,
                              cx + HALF_WIDTH_M, cy + HALF_WIDTH_M, transform=ds.transform)
        data = ds.read(1, window=window)
        win_transform = ds.window_transform(window)
        profile = ds.profile.copy()
        profile.update(height=data.shape[0], width=data.shape[1], transform=win_transform)
        with rasterio.open(cache_path, "w", **profile) as dst:
            dst.write(data, 1)
        bounds = rasterio.windows.bounds(window, ds.transform)
    return data, bounds


def compute_strain_rate(vx, vy, dx):
    vx = np.where(vx == -32767, np.nan, vx)
    vy = np.where(vy == -32767, np.nan, vy)
    vxs = gaussian_filter(np.nan_to_num(vx), sigma=SMOOTH_SIGMA_PX)
    vys = gaussian_filter(np.nan_to_num(vy), sigma=SMOOTH_SIGMA_PX)
    d_vy_drow, d_vy_dcol = np.gradient(vys, dx)
    d_vx_drow, d_vx_dcol = np.gradient(vxs, dx)
    exx = d_vx_dcol
    eyy = -d_vy_drow
    exy = 0.5 * (-d_vx_drow + d_vy_dcol)
    return np.sqrt(exx ** 2 + eyy ** 2 + exx * eyy + exy ** 2)


def main():
    stations = load_stations()
    pre, post = load_events()
    to_ps = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)

    st_x, st_y = to_ps.transform(stations["lon"].values, stations["lat"].values)
    pre_x, pre_y = to_ps.transform(pre["lon"].values, pre["lat"].values)
    post_x, post_y = to_ps.transform(post["lon"].values, post["lat"].values)
    cx, cy = st_x.mean(), st_y.mean()

    vx, bounds = fetch_component_window(VX_URL, VX_CACHE, cx, cy)
    vy, _ = fetch_component_window(VY_URL, VY_CACHE, cx, cy)
    strain = compute_strain_rate(vx, vy, dx=120.0)

    left, bottom, right, top = bounds
    extent_km = [(left - cx) / 1000, (right - cx) / 1000,
                 (bottom - cy) / 1000, (top - cy) / 1000]

    # crop to the event+station footprint (+ padding) since there's no
    # pre-existing displacement-map box to borrow, unlike T1
    all_x_km = np.concatenate([(post_x - cx) / 1000, (st_x - cx) / 1000])
    all_y_km = np.concatenate([(post_y - cy) / 1000, (st_y - cy) / 1000])
    xlim_km = (all_x_km.min() - PAD_KM, all_x_km.max() + PAD_KM)
    ylim_km = (all_y_km.min() - PAD_KM, all_y_km.max() + PAD_KM)

    row0 = int((extent_km[3] - ylim_km[1]) / (extent_km[3] - extent_km[2]) * strain.shape[0])
    row1 = int((extent_km[3] - ylim_km[0]) / (extent_km[3] - extent_km[2]) * strain.shape[0])
    col0 = int((xlim_km[0] - extent_km[0]) / (extent_km[1] - extent_km[0]) * strain.shape[1])
    col1 = int((xlim_km[1] - extent_km[0]) / (extent_km[1] - extent_km[0]) * strain.shape[1])
    row0, row1 = max(row0, 0), min(row1, strain.shape[0])
    col0, col1 = max(col0, 0), min(col1, strain.shape[1])
    visible = strain[row0:row1, col0:col1]
    visible_positive = visible[np.isfinite(visible) & (visible > 0)]
    vmin = max(np.percentile(visible_positive, 5), 1e-5)
    vmax = np.nanmax(visible_positive)
    print(f"strain rate over displayed crop: {visible.shape}, vmin={vmin:.2e} "
          f"vmax(true peak)={vmax:.2e} 1/yr")

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(np.clip(strain, 1e-6, None), extent=extent_km, origin="upper", cmap="inferno",
                   norm=LogNorm(vmin=vmin, vmax=vmax), zorder=0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(f"Effective strain rate (1/yr, log scale) -- ITS_LIVE RGI19A v02, "
                   f"{SMOOTH_KM:g} km smoothed")

    x_coords = np.linspace(extent_km[0], extent_km[1], strain.shape[1])
    y_coords = np.linspace(extent_km[3], extent_km[2], strain.shape[0])
    contour_levels = np.geomspace(vmin, vmax, 6)
    cs = ax.contour(x_coords, y_coords, strain, levels=contour_levels, colors="white",
                     linewidths=0.8, alpha=0.7, zorder=1)
    ax.clabel(cs, fmt=lambda v: f"{v:.1e}", fontsize=7, colors="white")

    ax.scatter((pre_x - cx) / 1000, (pre_y - cy) / 1000, s=8, color=COLOR_PRE, alpha=0.4,
               label=f"pyocto pre-hypoDD (n={len(pre)})", zorder=1)
    ax.scatter((post_x - cx) / 1000, (post_y - cy) / 1000, s=10, color=COLOR_POST, alpha=0.75,
               label=f"hypoDD relocated (n={len(post)})", zorder=2)
    ax.scatter((st_x - cx) / 1000, (st_y - cy) / 1000, s=170, color=COLOR_STATION, marker="^",
               edgecolor="white", linewidth=1.2, label="T2 stations", zorder=3)
    for code, x, y in zip(stations["code"], (st_x - cx) / 1000, (st_y - cy) / 1000):
        ax.annotate(code, (x, y), fontsize=9, color="white",
                    xytext=(5, 5), textcoords="offset points",
                    path_effects=[patheffects.withStroke(linewidth=2, foreground="black")])

    ax.set_xlim(xlim_km)
    ax.set_ylim(ylim_km)
    ax.set_xlabel("Distance east of T2 array centroid (km)")
    ax.set_ylabel("Distance north of T2 array centroid (km)")
    ax.set_title("T2 relocations vs. ITS_LIVE strain rate\n"
                 "(pyocto v5 -> hypoDD ccscale_0.33)", fontsize=13)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95, facecolor="white",
              edgecolor="#c3c2b7")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
