#!/usr/bin/env python3
"""T1 basal-ice-event structure: pyocto-catalog vs QuakeMigrate-catalog hypoDD
relocations (both under the finalized ccscale_0.33 config, see
t1_optimized_hypodd_and_catalog_comparison_plan memory), viewed from multiple
3D angles to test whether near-bed seismicity sits on a single planar surface
(the ice-bed interface itself, ~3.24 km per BedMachine/Bedmap2, see
pyocto_full_catalog_rebuild memory) or resolves into discrete linear/planar
structures consistent with specific basal cracks/crevasses.

Two complementary tools:
1. Fixed 3D viewpoints (map/top-down, along local-East, along local-North,
   oblique) -- the straightforward "look at it from different angles" pass.
2. PCA/eigen-decomposition of the basal-event point cloud + a Woodcock/Flinn
   shape parameter (standard structural-geology fabric-analysis technique for
   exactly this cluster-vs-girdle question) -- gives quantitative viewpoints
   too: looking down the inferred long axis (should collapse to a blob if it's
   really a line/crack) and down the inferred normal (should collapse to a
   thin sliver if it's really a plane).

QuakeMigrate's catalog spans the whole ~70 km region (not confined to
pyocto's ~10 km association box), so it's restricted to within RADIUS_KM of
the array centroid before comparison -- otherwise far-field events swamp the
local structure this script is about.

Usage:
    python full_catalog_pipeline/plot_hypodd_t1_basal_3d.py
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

import catalog_paths

PYOCTO_RELOC = catalog_paths.reloc("T1")
PYOCTO_STA = catalog_paths.station_sel("T1")
QM_RELOC = "/scratch2/qm/t1/output/hypodd_optimized/input_files/hypoDD.reloc"
QM_STA = "/scratch2/qm/t1/output/hypodd_optimized/input_files/station.sel"
OUT_DIR = catalog_paths.work_dir("T1")

ICE_BED_DEPTH_KM = 3.24  # BedMachine/Bedmap2 average, T1 (pyocto_full_catalog_rebuild memory)
BASAL_MIN_KM, BASAL_MAX_KM = 2.8, 4.0  # near-bed band bracketing the interface
RADIUS_KM = 15.0  # QM restriction so far-field events don't swamp local structure

COLOR_PYOCTO = "#15616d"
COLOR_QM = "#eb6834"
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
        f"T1 {name}: basal events ({BASAL_MIN_KM}-{BASAL_MAX_KM} km depth, n={len(x)})\n"
        f"Woodcock K={k:.2f} -> {shape_word}",
        fontsize=13,
    )
    cbar = fig.colorbar(sc, ax=fig.axes, shrink=0.6, pad=0.02, label="Depth (km)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_comparison_map(basal_p, basal_q, stations_xy, out_path):
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(basal_q["ex"], basal_q["nx"], s=8, color=COLOR_QM, alpha=0.35,
               label=f"QuakeMigrate catalog (n={len(basal_q)})", zorder=1)
    ax.scatter(basal_p["ex"], basal_p["nx"], s=10, color=COLOR_PYOCTO, alpha=0.5,
               label=f"pyocto catalog (n={len(basal_p)})", zorder=2)
    sx, sy = stations_xy
    ax.scatter(sx, sy, s=160, color=COLOR_STATION, marker="^", edgecolor="white",
               linewidth=1, label="T1 stations", zorder=3)
    ax.set_xlabel("East of centroid (km)")
    ax.set_ylabel("North of centroid (km)")
    ax.set_aspect("equal")
    ax.set_title(f"T1 basal events ({BASAL_MIN_KM}-{BASAL_MAX_KM} km depth): pyocto vs QuakeMigrate catalog\n"
                 f"(both hypoDD-relocated, ccscale_0.33 config)", fontsize=12)
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_comparison_depth_section(basal_p, basal_q, stations_xy, out_path):
    sx, sy = stations_xy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, df, color, label in (
        (ax1, basal_p, COLOR_PYOCTO, "pyocto"),
        (ax2, basal_q, COLOR_QM, "QuakeMigrate"),
    ):
        r = np.hypot(df["ex"], df["nx"])
        ax.scatter(r, df["depth"], s=10, color=color, alpha=0.4)
        ax.axhline(ICE_BED_DEPTH_KM, color="#888888", linestyle="--", linewidth=1,
                   label=f"ice-bed interface ({ICE_BED_DEPTH_KM} km)")
        ax.set_xlabel("Distance from array centroid (km)")
        ax.set_title(f"{label} (n={len(df)})")
        ax.legend(fontsize=8)
    ax1.invert_yaxis()  # shared y-axis: invert once only, or the second call cancels the first
    ax1.set_ylabel("Depth (km)")
    fig.suptitle("T1 basal-event depth section: distance from centroid vs depth")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    to_ps = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)

    sta_p = load_stations(PYOCTO_STA)
    sta_q = load_stations(QM_STA)
    st_x, st_y = to_ps.transform(sta_p["lon"].values, sta_p["lat"].values)
    cx, cy = st_x.mean(), st_y.mean()
    stations_xy = ((st_x - cx) / 1000.0, (st_y - cy) / 1000.0)

    pyocto = load_reloc(PYOCTO_RELOC)
    qm = load_reloc(QM_RELOC)

    pyocto["ex"], pyocto["nx"] = project_km(pyocto, to_ps, cx, cy)
    qm["ex"], qm["nx"] = project_km(qm, to_ps, cx, cy)

    r_p = np.hypot(pyocto["ex"], pyocto["nx"])
    r_q = np.hypot(qm["ex"], qm["nx"])
    pyocto = pyocto[r_p <= RADIUS_KM]
    dropped_q = (r_q > RADIUS_KM).sum()
    qm = qm[r_q <= RADIUS_KM]
    print(f"QuakeMigrate: dropped {dropped_q} far-field events (> {RADIUS_KM} km from centroid)")

    basal_p = pyocto[(pyocto["depth"] >= BASAL_MIN_KM) & (pyocto["depth"] <= BASAL_MAX_KM)]
    basal_q = qm[(qm["depth"] >= BASAL_MIN_KM) & (qm["depth"] <= BASAL_MAX_KM)]
    print(f"pyocto basal events: {len(basal_p)} / {len(pyocto)}")
    print(f"QM basal events: {len(basal_q)} / {len(qm)}")

    out_map = f"{OUT_DIR}/hypodd_t1_basal_comparison_map.png"
    out_section = f"{OUT_DIR}/hypodd_t1_basal_comparison_depth_section.png"
    plot_comparison_map(basal_p, basal_q, stations_xy, out_map)
    plot_comparison_depth_section(basal_p, basal_q, stations_xy, out_section)
    print(f"wrote {out_map}")
    print(f"wrote {out_section}")

    for name, df, out_3d in (
        ("pyocto catalog", basal_p, f"{OUT_DIR}/hypodd_t1_basal_3d_pyocto.png"),
        ("QuakeMigrate catalog", basal_q, f"{OUT_DIR}/hypodd_t1_basal_3d_qm.png"),
    ):
        points = df[["ex", "nx", "depth"]].values
        eigval, eigvec, k = woodcock_shape(points)
        print(f"{name}: eigenvalues (normalized) = {eigval / eigval.sum()}, Woodcock K = {k:.3f}")
        plot_multi_viewpoint(
            name, df["ex"].values, df["nx"].values, df["depth"].values, df["depth"].values,
            eigvec, k, stations_xy, out_3d,
        )
        print(f"wrote {out_3d}")


if __name__ == "__main__":
    main()
