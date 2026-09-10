#!/usr/bin/env python3
"""T1 relocation map with ice speed (not strain rate) as context, same view as
plot_hypodd_t1_shear_margin_map.py -- for side-by-side comparison of the two
quantities over the identical crop/orientation/smoothing.

Speed is smoothed the same way and by the same amount (3 km Gaussian on vx/vy)
as the strain-rate figure, so the two are directly comparable: this shows
raw log-scaled speed only shows a diffuse gradient here (no sharp margin),
which is what motivated switching to strain rate in the first place (see
plot_hypodd_t1_shear_margin_map.py's docstring).

Usage:
    python full_catalog_pipeline/plot_hypodd_t1_speed_map.py
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

import numpy as np
import pyproj
from matplotlib.colors import LogNorm

from plot_hypodd_t1_shear_margin_map import (
    COLOR_PRE, COLOR_POST, COLOR_STATION, OUT_DIR, SMOOTH_KM, SMOOTH_SIGMA_PX,
    VX_CACHE, VX_URL, VY_CACHE, VY_URL,
    displacement_map_bounds_km, fetch_component_window, load_events, load_stations,
    stations_centroid_ps,
)

OUT_PNG = f"{OUT_DIR}/hypodd_reloc_map_speed.png"


def compute_smoothed_speed(vx, vy):
    from scipy.ndimage import gaussian_filter
    vx = np.where(vx == -32767, np.nan, vx)
    vy = np.where(vy == -32767, np.nan, vy)
    vxs = gaussian_filter(np.nan_to_num(vx), sigma=SMOOTH_SIGMA_PX)
    vys = gaussian_filter(np.nan_to_num(vy), sigma=SMOOTH_SIGMA_PX)
    return np.sqrt(vxs ** 2 + vys ** 2)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patheffects as patheffects
    import matplotlib.pyplot as plt

    stations = load_stations()
    pre, post = load_events()
    to_ps = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)

    st_x, st_y = to_ps.transform(stations["lon"].values, stations["lat"].values)
    pre_x, pre_y = to_ps.transform(pre["lon"].values, pre["lat"].values)
    post_x, post_y = to_ps.transform(post["lon"].values, post["lat"].values)
    cx, cy = st_x.mean(), st_y.mean()

    vx, bounds = fetch_component_window(VX_URL, VX_CACHE, cx, cy)
    vy, _ = fetch_component_window(VY_URL, VY_CACHE, cx, cy)
    speed = compute_smoothed_speed(vx, vy)

    left, bottom, right, top = bounds
    extent_km = [(left - cx) / 1000, (right - cx) / 1000,
                 (bottom - cy) / 1000, (top - cy) / 1000]
    xlim_km0, xlim_km1, ylim_km0, ylim_km1 = displacement_map_bounds_km(to_ps, cx, cy)
    xlim_km, ylim_km = (xlim_km0, xlim_km1), (ylim_km0, ylim_km1)

    # true peak within the displayed crop, same convention as the strain-rate figure
    row0 = int((extent_km[3] - ylim_km[1]) / (extent_km[3] - extent_km[2]) * speed.shape[0])
    row1 = int((extent_km[3] - ylim_km[0]) / (extent_km[3] - extent_km[2]) * speed.shape[0])
    col0 = int((xlim_km[0] - extent_km[0]) / (extent_km[1] - extent_km[0]) * speed.shape[1])
    col1 = int((xlim_km[1] - extent_km[0]) / (extent_km[1] - extent_km[0]) * speed.shape[1])
    row0, row1 = max(row0, 0), min(row1, speed.shape[0])
    col0, col1 = max(col0, 0), min(col1, speed.shape[1])
    visible = speed[row0:row1, col0:col1]
    visible_positive = visible[np.isfinite(visible) & (visible > 0)]
    vmin = max(np.percentile(visible_positive, 5), 1e-3)
    vmax = np.nanmax(visible_positive)
    print(f"speed over displayed crop: {visible.shape}, vmin={vmin:.2f} "
          f"vmax(true peak)={vmax:.2f} m/yr")

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(np.clip(speed, 1e-3, None), extent=extent_km, origin="upper", cmap="viridis",
                   norm=LogNorm(vmin=vmin, vmax=vmax), zorder=0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(f"Ice surface speed (m/yr, log scale) -- ITS_LIVE RGI19A v02, "
                   f"{SMOOTH_KM:g} km smoothed")

    x_coords = np.linspace(extent_km[0], extent_km[1], speed.shape[1])
    y_coords = np.linspace(extent_km[3], extent_km[2], speed.shape[0])  # origin="upper"
    contour_levels = np.geomspace(vmin, vmax, 6)
    cs = ax.contour(x_coords, y_coords, speed, levels=contour_levels, colors="white",
                     linewidths=0.8, alpha=0.7, zorder=1)
    ax.clabel(cs, fmt=lambda v: f"{v:.0f}", fontsize=7, colors="white")

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
    ax.set_title("T1 relocations vs. ITS_LIVE ice speed (same view/smoothing as the\n"
                 "strain-rate figure) -- (pyocto v5 -> hypoDD ccscale_0.33)", fontsize=13)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.95, facecolor="white",
              edgecolor="#c3c2b7")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
