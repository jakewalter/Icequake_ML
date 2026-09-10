#!/usr/bin/env python3
"""T2 basal-ice-event structure, pyocto catalog only (T2 has no QuakeMigrate-
catalog hypoDD run analogous to T1's hypodd_optimized comparison at the time
this was written), viewed from multiple 3D angles to test whether near-bed
seismicity sits on a single planar surface (the ice-bed interface itself,
~2.02 km per BedMachine/Bedmap2, see hypodd_relocate.py's ARRAY_CONFIG["T2"])
or resolves into discrete linear/planar structures consistent with specific
basal cracks/crevasses.

Direct T2 port of plot_hypodd_t1_basal_3d.py -- same two tools (fixed 3D
viewpoints + Woodcock/Flinn PCA shape test), single catalog instead of a
pyocto-vs-QuakeMigrate comparison.

Usage:
    python full_catalog_pipeline/plot_hypodd_t2_basal_3d.py
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

import catalog_paths

# Paths come from catalog_paths so the authoritative relocation is named in ONE place
# (see that module). Set ICEQUAKE_RELOC=hypodd to reproduce a figure against the
# superseded v5 relocation without editing anything.
PYOCTO_RELOC = catalog_paths.reloc("T2")
PYOCTO_STA = catalog_paths.station_sel("T2")
OUT_DIR = catalog_paths.work_dir("T2")

ICE_BED_DEPTH_KM = 2.02  # BedMachine/Bedmap2 average, T2 (hypodd_relocate.py ARRAY_CONFIG)
# Near-bed band, fit to T2's OWN depth histogram (NOT a copy of T1's -440m/+760m offsets --
# that earlier choice cut through the middle of T2's actual dense population, see
# t2_basal_event_nature_multi_analysis memory's correction). T2's relocated catalog is densely
# populated from ~1.2-2.0km (up to ~800m above the 2.02km bed) with a sharp drop right at the
# interface and a sparse tail below -- this band captures the whole dense core plus the
# immediate below-bed tail.
#
# The band is CATALOG-SPECIFIC and these defaults were fit to the authoritative (hypodd_vels1d)
# catalog. A different relocation can move a whole population across the floor: under the
# upsampled-dt.cc `clean_ccstrong` candidate, 739 events (20.8%, vs 195 / 5.9% authoritative)
# sit ABOVE 1.1 km, forming their own mode peaked at ~0.85 km with a clean trough at 1.00-1.15.
# The default floor cuts through that trough, so every "basal" figure silently drops the very
# population that distinguishes that run. Override per-run rather than editing this line:
#   T2_BASAL_MIN_KM=0.6 T2_BASAL_MAX_KM=2.3 ICEQUAKE_RELOC=... python ...
BASAL_MIN_KM = float(os.environ.get("T2_BASAL_MIN_KM", 1.1))
BASAL_MAX_KM = float(os.environ.get("T2_BASAL_MAX_KM", 2.3))
RADIUS_KM = 15.0  # kept for parity with the T1 script's far-field guard (unused single-catalog, pyocto's own association box is already local)

COLOR_PYOCTO = "#15616d"
COLOR_STATION = "#0b0b0b"
CMAP = "cividis_r"  # sequential, single hue, perceptually uniform; _r so shallow=light, deep=dark

RELOC_COLS = [
    "id", "lat", "lon", "depth", "x", "y", "z", "ex", "ey", "ez",
    "yr", "mo", "dy", "hr", "mi", "sc", "mag",
    "nccp", "nccs", "nctp", "ncts", "rcc", "rct", "cid",
]


def load_stations(path):
    st = pd.read_csv(path, sep=r"\s+", header=None, names=["id", "lat", "lon", "elev"])
    st["code"] = st["id"].str.split(".").str[1]
    return st


def load_reloc(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=RELOC_COLS)


def project_km(df, to_ps, cx, cy):
    x, y = to_ps.transform(df["lon"].values, df["lat"].values)
    return (x - cx) / 1000.0, (y - cy) / 1000.0


def woodcock_shape(points):
    """Eigen-decomposition of the 3x3 point covariance + Woodcock/Flinn K.

    Standard structural-geology fabric-analysis technique for classifying a
    3D point/orientation cloud as cluster-like (line, K>>1), girdle-like
    (plane, K<<1), or isotropic (K~1). Eigenvalues normalized to sum to 1
    (Woodcock 1977 convention) so ln-ratios are well-defined.
    """
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigval, eigvec = np.linalg.eigh(cov)  # ascending order
    order = np.argsort(eigval)[::-1]  # descending: l1 >= l2 >= l3
    eigval, eigvec = eigval[order], eigvec[:, order]
    l1, l2, l3 = eigval / eigval.sum()
    eps = 1e-9
    k = np.log((l1 + eps) / (l2 + eps)) / np.log((l2 + eps) / (l3 + eps))
    return eigval, eigvec, k


def dir_to_elev_azim(v):
    v = v / np.linalg.norm(v)
    elev = np.degrees(np.arcsin(np.clip(v[2], -1, 1)))
    azim = np.degrees(np.arctan2(v[1], v[0]))
    return elev, azim


def make_3d_panel(fig, pos, x, y, z, depth, elev, azim, title, stations_xy=None):
    ax = fig.add_subplot(pos[0], pos[1], pos[2], projection="3d")
    sc = ax.scatter(x, y, z, c=depth, cmap=CMAP, s=6, alpha=0.6, linewidth=0)
    if stations_xy is not None:
        sx, sy = stations_xy
        ax.scatter(sx, sy, np.zeros_like(sx), c=COLOR_STATION, marker="^", s=60,
                   depthshade=False)
    ax.set_xlabel("East of centroid (km)", fontsize=8)
    ax.set_ylabel("North of centroid (km)", fontsize=8)
    ax.set_zlabel("Depth (km)", fontsize=8)
    ax.invert_zaxis()  # deeper = visually lower
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    return sc


def make_2d_map_panel(fig, pos, x, y, depth, title, stations_xy=None):
    """True top-down projection as a plain 2D axes -- avoids matplotlib's 3D
    axes rendering a degenerate, cluttered z-axis at elev=90."""
    ax = fig.add_subplot(pos[0], pos[1], pos[2])
    sc = ax.scatter(x, y, c=depth, cmap=CMAP, s=6, alpha=0.6, linewidth=0)
    if stations_xy is not None:
        sx, sy = stations_xy
        ax.scatter(sx, sy, c=COLOR_STATION, marker="^", s=60)
    ax.set_xlabel("East of centroid (km)", fontsize=8)
    ax.set_ylabel("North of centroid (km)", fontsize=8)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    return sc


def plot_multi_viewpoint(name, x, y, z, depth, eigvec, k, stations_xy, out_path):
    long_axis = eigvec[:, 0]
    normal_axis = eigvec[:, 2]
    elev_long, azim_long = dir_to_elev_azim(long_axis)
    elev_norm, azim_norm = dir_to_elev_azim(normal_axis)

    views_3d = [
        (0, -90, "Looking along local-North axis (view from South)"),
        (0, 0, "Looking along local-East axis (view from West)"),
        (20, -45, "Oblique 3/4 view"),
        (elev_long, azim_long, "Looking down inferred long axis\n(collapses to a blob if line-like)"),
        (elev_norm, azim_norm, "Looking down inferred normal axis\n(collapses to a sliver if plane-like)"),
    ]

    fig = plt.figure(figsize=(16, 10))
    sc = make_2d_map_panel(fig, (2, 3, 1), x, y, depth,
                           "Map view (true top-down)", stations_xy)
    for i, (elev, azim, title) in enumerate(views_3d, start=2):
        sc = make_3d_panel(fig, (2, 3, i), x, y, z, depth, elev, azim, title, stations_xy)

    shape_word = "cluster/linear (crack-like)" if k > 1 else "girdle/planar (interface-like)"
    fig.suptitle(
        f"T2 {name}: basal events ({BASAL_MIN_KM}-{BASAL_MAX_KM} km depth, n={len(x)})\n"
        f"Woodcock K={k:.2f} -> {shape_word}",
        fontsize=13,
    )
    cbar = fig.colorbar(sc, ax=fig.axes, shrink=0.6, pad=0.02, label="Depth (km)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_map(basal_p, stations_xy, out_path):
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(basal_p["ex"], basal_p["nx"], s=10, color=COLOR_PYOCTO, alpha=0.5,
               label=f"pyocto catalog (n={len(basal_p)})", zorder=2)
    sx, sy = stations_xy
    ax.scatter(sx, sy, s=160, color=COLOR_STATION, marker="^", edgecolor="white",
               linewidth=1, label="T2 stations", zorder=3)
    ax.set_xlabel("East of centroid (km)")
    ax.set_ylabel("North of centroid (km)")
    ax.set_aspect("equal")
    ax.set_title(f"T2 basal events ({BASAL_MIN_KM}-{BASAL_MAX_KM} km depth): pyocto catalog\n"
                 f"(hypoDD-relocated, ccscale_0.33 config)", fontsize=12)
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_depth_section(basal_p, stations_xy, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    r = np.hypot(basal_p["ex"], basal_p["nx"])
    ax.scatter(r, basal_p["depth"], s=10, color=COLOR_PYOCTO, alpha=0.4)
    ax.axhline(ICE_BED_DEPTH_KM, color="#888888", linestyle="--", linewidth=1,
               label=f"ice-bed interface ({ICE_BED_DEPTH_KM} km)")
    ax.set_xlabel("Distance from array centroid (km)")
    ax.set_ylabel("Depth (km)")
    ax.set_title(f"T2 basal-event depth section: distance from centroid vs depth (n={len(basal_p)})")
    ax.invert_yaxis()
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    to_ps = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)

    sta_p = load_stations(PYOCTO_STA)
    st_x, st_y = to_ps.transform(sta_p["lon"].values, sta_p["lat"].values)
    cx, cy = st_x.mean(), st_y.mean()
    stations_xy = ((st_x - cx) / 1000.0, (st_y - cy) / 1000.0)

    pyocto = load_reloc(PYOCTO_RELOC)
    pyocto["ex"], pyocto["nx"] = project_km(pyocto, to_ps, cx, cy)

    r_p = np.hypot(pyocto["ex"], pyocto["nx"])
    dropped_p = (r_p > RADIUS_KM).sum()
    pyocto = pyocto[r_p <= RADIUS_KM]
    if dropped_p:
        print(f"pyocto: dropped {dropped_p} far-field events (> {RADIUS_KM} km from centroid)")

    basal_p = pyocto[(pyocto["depth"] >= BASAL_MIN_KM) & (pyocto["depth"] <= BASAL_MAX_KM)]
    print(f"pyocto basal events: {len(basal_p)} / {len(pyocto)}")

    out_map = f"{OUT_DIR}/hypodd_t2_basal_map.png"
    out_section = f"{OUT_DIR}/hypodd_t2_basal_depth_section.png"
    plot_map(basal_p, stations_xy, out_map)
    plot_depth_section(basal_p, stations_xy, out_section)
    print(f"wrote {out_map}")
    print(f"wrote {out_section}")

    points = basal_p[["ex", "nx", "depth"]].values
    eigval, eigvec, k = woodcock_shape(points)
    print(f"pyocto catalog: eigenvalues (normalized) = {eigval / eigval.sum()}, Woodcock K = {k:.3f}")
    out_3d = f"{OUT_DIR}/hypodd_t2_basal_3d_pyocto.png"
    plot_multi_viewpoint(
        "pyocto catalog", basal_p["ex"].values, basal_p["nx"].values,
        basal_p["depth"].values, basal_p["depth"].values,
        eigvec, k, stations_xy, out_3d,
    )
    print(f"wrote {out_3d}")


if __name__ == "__main__":
    main()
