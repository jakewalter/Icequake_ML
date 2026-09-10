#!/usr/bin/env python3
"""T1 relocation map with ice-flow context, to show the array sitting on an
ice-stream shear margin.

Supersedes plot_hypodd_t1_prelim.py's plot_map() -- writes the same
hypodd_reloc_map_prelim.png.

Backdrop is *effective strain rate*, not raw speed, computed from the ITS_LIVE
Antarctica velocity mosaic's vx/vy (RGI19A, 120 m, v02 static composite;
https://its-live-data.s3.amazonaws.com, public/no-auth AWS Open Data) -- a
direct comparison (see session notes) showed raw log-speed only shows a
diffuse gradient here, while strain rate = |grad(velocity)| resolves the
margin as a sharp, narrow ridge (it's literally where the ice is shearing
fastest, not just where it's fast). Differentiating the composite velocity
field directly is dominated by feature-tracking noise (checked: an unsmoothed
version is salt-and-pepper with no visible margin); ~3 km Gaussian smoothing
before differentiating (standard glaciological practice, e.g. Alley/King-style
strain-rate smoothing scales) cleans this up without washing out the margin.

Plotted in the mosaic's native Antarctic Polar Stereographic CRS (distance
east/north of the T1 array centroid, km) -- the orientation of the original
wide-view figure -- but cropped to the same close-in extent as
plot_hypodd_t1_displacement.py's displacement map (that script's autoscaled
lon/lat bounds, reprojected here to km, define XLIM_KM/YLIM_KM below). No
raster reprojection needed for this: only the four bounding-box corners and
the event/station points go through pyproj's point transform, so this avoids
the rasterio/PROJ .db conflict a full raster reproject ran into earlier.

Usage:
    python full_catalog_pipeline/plot_hypodd_t1_shear_margin_map.py
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

PRE_CSV = "full_catalog_pipeline/artifacts/full_run/T1_v5/pyocto_events.csv"
RELOC = catalog_paths.reloc("T1")
STATION_SEL = catalog_paths.station_sel("T1")
OUT_DIR = catalog_paths.work_dir("T1")
OUT_PNG = f"{OUT_DIR}/hypodd_reloc_map_prelim.png"

ITS_LIVE_BASE = "https://its-live-data.s3.amazonaws.com/velocity_mosaic/v2/static/cog"
VX_URL = f"/vsicurl/{ITS_LIVE_BASE}/ITS_LIVE_velocity_120m_RGI19A_0000_v02_vx.tif"
VY_URL = f"/vsicurl/{ITS_LIVE_BASE}/ITS_LIVE_velocity_120m_RGI19A_0000_v02_vy.tif"
VX_CACHE = f"{OUT_DIR}/its_live_vx_t1_window.tif"
VY_CACHE = f"{OUT_DIR}/its_live_vy_t1_window.tif"
HALF_WIDTH_M = 50_000  # 100 km fetch box -- comfortably covers the close-in crop below
SMOOTH_KM = 3.0
SMOOTH_SIGMA_PX = SMOOTH_KM * 1000 / 120  # suppresses feature-tracking noise pre-gradient

# plot_hypodd_t1_displacement.py's plot_map() autoscales to these lon/lat limits (captured
# by rendering that figure and reading ax.get_xlim()/get_ylim()) -- reprojected to km-from-
# centroid below so both figures show the same physical extent.
DISPLACEMENT_MAP_XLIM = (-100.78949814809079, -100.16578083310542)
DISPLACEMENT_MAP_YLIM = (-77.38212133781971, -77.23244278743576)

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


def stations_centroid_ps(stations, to_ps):
    x, y = to_ps.transform(stations["lon"].values, stations["lat"].values)
    return x.mean(), y.mean()


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
    exx = d_vx_dcol           # dvx/dx (col increases eastward, same sense as x)
    eyy = -d_vy_drow          # dvy/dy (row increases southward, so flip sign)
    exy = 0.5 * (-d_vx_drow + d_vy_dcol)
    return np.sqrt(exx ** 2 + eyy ** 2 + exx * eyy + exy ** 2)


def displacement_map_bounds_km(to_ps, cx, cy):
    lons = [DISPLACEMENT_MAP_XLIM[0], DISPLACEMENT_MAP_XLIM[0],
            DISPLACEMENT_MAP_XLIM[1], DISPLACEMENT_MAP_XLIM[1]]
    lats = [DISPLACEMENT_MAP_YLIM[0], DISPLACEMENT_MAP_YLIM[1],
            DISPLACEMENT_MAP_YLIM[0], DISPLACEMENT_MAP_YLIM[1]]
    x, y = to_ps.transform(lons, lats)
    return ((min(x) - cx) / 1000, (max(x) - cx) / 1000,
            (min(y) - cy) / 1000, (max(y) - cy) / 1000)


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
    xlim_km0, xlim_km1, ylim_km0, ylim_km1 = displacement_map_bounds_km(to_ps, cx, cy)
    xlim_km, ylim_km = (xlim_km0, xlim_km1), (ylim_km0, ylim_km1)

    # scale color on the true peak within the displayed crop, not a percentile that
    # would wash it out -- inferno's near-white top end makes that peak pop distinctly
    # from the broader elevated band around it, unlike viridis's muted yellow-green top.
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

    # log-spaced contours of the same field, so the margin's shape reads as lines
    # (and their spacing shows the gradient steepness) rather than only color.
    x_coords = np.linspace(extent_km[0], extent_km[1], strain.shape[1])
    y_coords = np.linspace(extent_km[3], extent_km[2], strain.shape[0])  # origin="upper"
    contour_levels = np.geomspace(vmin, vmax, 6)
    cs = ax.contour(x_coords, y_coords, strain, levels=contour_levels, colors="white",
                     linewidths=0.8, alpha=0.7, zorder=1)
    ax.clabel(cs, fmt=lambda v: f"{v:.1e}", fontsize=7, colors="white")

    ax.scatter((pre_x - cx) / 1000, (pre_y - cy) / 1000, s=8, color=COLOR_PRE, alpha=0.4,
               label=f"pyocto pre-hypoDD (n={len(pre)})", zorder=1)
    ax.scatter((post_x - cx) / 1000, (post_y - cy) / 1000, s=10, color=COLOR_POST, alpha=0.75,
               label=f"hypoDD relocated (n={len(post)})", zorder=2)
    ax.scatter((st_x - cx) / 1000, (st_y - cy) / 1000, s=170, color=COLOR_STATION, marker="^",
               edgecolor="white", linewidth=1.2, label="T1 stations", zorder=3)
    for code, x, y in zip(stations["code"], (st_x - cx) / 1000, (st_y - cy) / 1000):
        ax.annotate(code, (x, y), fontsize=9, color="white",
                    xytext=(5, 5), textcoords="offset points",
                    path_effects=[patheffects.withStroke(linewidth=2, foreground="black")])

    ax.set_xlim(xlim_km)
    ax.set_ylim(ylim_km)
    ax.set_xlabel("Distance east of T1 array centroid (km)")
    ax.set_ylabel("Distance north of T1 array centroid (km)")
    ax.set_title("T1 sits on an ice-stream shear margin: relocations vs. ITS_LIVE strain rate\n"
                 "(pyocto v5 -> hypoDD ccscale_0.33)", fontsize=13)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95, facecolor="white",
              edgecolor="#c3c2b7")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
