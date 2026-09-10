#!/usr/bin/env python3
"""T2 basal-event structure, per-cluster: spatially cluster the basal events
(DBSCAN, eps on the order of the hypothesized crack length), then run the
Woodcock/Flinn linearity test on EACH cluster independently -- same method
as plot_hypodd_t1_basal_clusters.py, T2 pyocto-only (no QuakeMigrate-catalog
hypoDD comparison available for T2 at the time this was written).

Unlike the T1 script (which only printed per-cluster stats), this ALSO
writes <OUT_DIR>/t2_clusterN_event_ids.txt for every cluster large enough
for PCA, so results feed directly into hypodd_svd_cluster_errors.py and the
cluster-based focal-mechanism scripts without a manual extraction step.

Usage:
    python full_catalog_pipeline/plot_hypodd_t2_basal_clusters.py
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN

from plot_hypodd_t2_basal_3d import (
    PYOCTO_RELOC, PYOCTO_STA, OUT_DIR,
    BASAL_MIN_KM, BASAL_MAX_KM, RADIUS_KM, COLOR_STATION,
    load_stations, load_reloc, project_km, woodcock_shape, dir_to_elev_azim,
    make_3d_panel,
)

DBSCAN_EPS_KM = 0.15   # same as T1: smaller than the hypothesized ~500 m crack length
DBSCAN_MIN_SAMPLES = 12
MIN_CLUSTER_FOR_PCA = 15


def cluster_basal(df):
    pts = df[["ex", "nx", "depth"]].values
    labels = DBSCAN(eps=DBSCAN_EPS_KM, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(pts)
    df = df.copy()
    df["cluster"] = labels
    return df


QUANTIZATION_SUSPECT_THRESHOLD = 0.5


def cluster_stats(df):
    rows = []
    for cid, g in df[df["cluster"] >= 0].groupby("cluster"):
        if len(g) < MIN_CLUSTER_FOR_PCA:
            continue
        pts = g[["ex", "nx", "depth"]].values
        eigval, eigvec, k = woodcock_shape(pts)
        extent_km = pdist(pts).max()
        depth_uniq = g["depth"].nunique() / len(g)
        rows.append({
            "cluster": cid, "n": len(g),
            "centroid_e": g["ex"].mean(), "centroid_n": g["nx"].mean(),
            "centroid_depth": g["depth"].mean(),
            "extent_km": extent_km, "woodcock_k": k,
            "depth_uniqueness": depth_uniq,
            "quantization_suspect": depth_uniq < QUANTIZATION_SUSPECT_THRESHOLD,
            "eigval": eigval, "eigvec": eigvec,
        })
    return pd.DataFrame(rows).sort_values("woodcock_k", ascending=False).reset_index(drop=True)


def plot_cluster_map(df, stats, stations_xy, name, out_path):
    fig, ax = plt.subplots(figsize=(10, 9))
    noise = df[df["cluster"] == -1]
    ax.scatter(noise["ex"], noise["nx"], s=6, color="#cccccc", alpha=0.4,
               label=f"unclustered noise (n={len(noise)})", zorder=1)
    cmap = plt.get_cmap("tab20")
    kept_ids = set(stats["cluster"]) if len(stats) else set()
    small = df[(df["cluster"] >= 0) & (~df["cluster"].isin(kept_ids))]
    ax.scatter(small["ex"], small["nx"], s=8, color="#999999", alpha=0.5,
               label=f"clusters too small for PCA (n<{MIN_CLUSTER_FOR_PCA}, n={len(small)})", zorder=1)
    for i, row in stats.iterrows():
        g = df[df["cluster"] == row["cluster"]]
        color = cmap(i % 20)
        ax.scatter(g["ex"], g["nx"], s=14, color=color, zorder=2)
        suspect_tag = " [SUSPECT]" if row["quantization_suspect"] else ""
        ax.annotate(f"#{int(row['cluster'])} K={row['woodcock_k']:.2f}{suspect_tag}",
                    (row["centroid_e"], row["centroid_n"]), fontsize=7,
                    color="#cc0000" if row["quantization_suspect"] else "black",
                    xytext=(3, 3), textcoords="offset points")
    sx, sy = stations_xy
    ax.scatter(sx, sy, s=160, color=COLOR_STATION, marker="^", edgecolor="white",
               linewidth=1, label="T2 stations", zorder=3)
    ax.set_xlabel("East of centroid (km)")
    ax.set_ylabel("North of centroid (km)")
    ax.set_aspect("equal")
    ax.set_title(f"T2 {name}: DBSCAN clusters of basal events (eps={DBSCAN_EPS_KM} km)\n"
                 f"labels show Woodcock K per cluster (K>1 linear/crack-like, K<1 planar)")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_k_vs_extent(stats_p, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    if len(stats_p):
        ax.scatter(stats_p["extent_km"], stats_p["woodcock_k"], s=stats_p["n"] * 2,
                   color="#15616d", alpha=0.6, label=f"pyocto (n={len(stats_p)} clusters)")
    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1)
    ax.axvline(0.5, color="#888888", linestyle=":", linewidth=1, label="500 m hypothesis")
    ax.set_xlabel("Cluster spatial extent, max pairwise distance (km)")
    ax.set_ylabel("Woodcock K (>1 linear/crack-like, <1 planar)")
    ax.set_yscale("log")
    ax.set_title("T2: per-cluster shape vs size (marker size = event count)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_example_clusters(df, stats, name, out_path, n_examples=4):
    linear = stats[stats["woodcock_k"] > 1].sort_values(
        ["quantization_suspect", "woodcock_k"], ascending=[True, False])
    top = linear.head(n_examples)
    if len(top) == 0:
        return
    fig = plt.figure(figsize=(16, 4 * len(top)))
    for row_i, (_, row) in enumerate(top.iterrows()):
        g = df[df["cluster"] == row["cluster"]]
        eigvec = row["eigvec"]
        long_axis, normal_axis = eigvec[:, 0], eigvec[:, 2]
        elev_l, azim_l = dir_to_elev_azim(long_axis)
        elev_n, azim_n = dir_to_elev_azim(normal_axis)
        panels = [
            (20, -45, "Oblique view"),
            (elev_l, azim_l, "Down inferred long axis\n(blob if line-like)"),
            (elev_n, azim_n, "Down inferred normal axis\n(sliver if plane-like)"),
        ]
        suspect_tag = " [QUANTIZATION SUSPECT]" if row["quantization_suspect"] else ""
        for col, (elev, azim, title) in enumerate(panels, start=1):
            pos = (len(top), 3, row_i * 3 + col)
            make_3d_panel(fig, pos, g["ex"].values, g["nx"].values, g["depth"].values,
                          g["depth"].values, elev, azim,
                          f"Cluster #{int(row['cluster'])} n={row['n']} "
                          f"extent={row['extent_km']*1000:.0f}m K={row['woodcock_k']:.2f}"
                          f"{suspect_tag}\n{title}")
    fig.suptitle(f"T2 {name}: 3D views of the {len(top)} most line-like clusters (highest Woodcock K)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_event_id_lists(df, stats, out_dir):
    """Write <out_dir>/t2_clusterN_event_ids.txt for every PCA-eligible
    cluster, so hypodd_svd_cluster_errors.py / the focal-mech pipeline can
    consume them directly without a manual extraction step."""
    paths = {}
    for _, row in stats.iterrows():
        cid = int(row["cluster"])
        ids = df[df["cluster"] == cid]["id"].astype(int).tolist()
        path = f"{out_dir}/t2_cluster{cid}_event_ids.txt"
        with open(path, "w") as f:
            f.write("\n".join(str(i) for i in ids) + "\n")
        paths[cid] = path
    return paths


def main():
    to_ps = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    sta_p = load_stations(PYOCTO_STA)
    st_x, st_y = to_ps.transform(sta_p["lon"].values, sta_p["lat"].values)
    cx, cy = st_x.mean(), st_y.mean()
    stations_xy = ((st_x - cx) / 1000.0, (st_y - cy) / 1000.0)

    pyocto = load_reloc(PYOCTO_RELOC)
    pyocto["ex"], pyocto["nx"] = project_km(pyocto, to_ps, cx, cy)
    pyocto = pyocto[np.hypot(pyocto["ex"], pyocto["nx"]) <= RADIUS_KM]

    basal_p = pyocto[(pyocto["depth"] >= BASAL_MIN_KM) & (pyocto["depth"] <= BASAL_MAX_KM)]

    name, df, tag = "pyocto catalog", basal_p, "pyocto"
    clustered = cluster_basal(df)
    stats = cluster_stats(clustered)
    n_linear = (stats["woodcock_k"] > 1).sum()
    n_planar = (stats["woodcock_k"] <= 1).sum()
    print(f"\n=== {name} ===")
    print(f"DBSCAN eps={DBSCAN_EPS_KM}km, min_samples={DBSCAN_MIN_SAMPLES}: "
          f"{clustered['cluster'].max() + 1} raw clusters, "
          f"{len(stats)} with >= {MIN_CLUSTER_FOR_PCA} events for PCA")
    print(f"  linear/crack-like (K>1): {n_linear}, planar (K<=1): {n_planar}")
    if len(stats):
        n_suspect_linear = (stats["quantization_suspect"] & (stats["woodcock_k"] > 1)).sum()
        print(f"  median extent: {stats['extent_km'].median()*1000:.0f} m, "
              f"median K: {stats['woodcock_k'].median():.2f}")
        print(f"  of the {n_linear} linear clusters, {n_suspect_linear} are quantization-suspect")
        print(stats[["cluster", "n", "extent_km", "woodcock_k", "depth_uniqueness",
                     "quantization_suspect"]].to_string(index=False))

    out_map = f"{OUT_DIR}/hypodd_t2_basal_clusters_map_{tag}.png"
    plot_cluster_map(clustered, stats, stations_xy, name, out_map)
    print(f"wrote {out_map}")

    out_examples = f"{OUT_DIR}/hypodd_t2_basal_clusters_examples_{tag}.png"
    plot_example_clusters(clustered, stats, name, out_examples)
    print(f"wrote {out_examples}")

    out_scatter = f"{OUT_DIR}/hypodd_t2_basal_clusters_k_vs_extent.png"
    plot_k_vs_extent(stats, out_scatter)
    print(f"wrote {out_scatter}")

    id_paths = write_event_id_lists(clustered, stats, OUT_DIR)
    print(f"\nwrote {len(id_paths)} event-id list(s):")
    for cid, path in id_paths.items():
        print(f"  cluster {cid}: {path}")


if __name__ == "__main__":
    main()
