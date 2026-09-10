#!/usr/bin/env python3
"""Analyze GrowClust's T1 relocations (pyocto and QuakeMigrate catalogs) with
the same basal-crack methodology used for hypoDD (plot_hypodd_t1_basal_3d.py /
plot_hypodd_t1_basal_clusters.py), as an independent cross-check: GrowClust's
hierarchical-clustering algorithm never assembles one big linear system, so
it isn't vulnerable to hypoDD's LSQR instability, and it uses ONLY cross-
correlation data (no catalog-pick dt.ct at all) -- a genuinely different
method on an overlapping but not identical data subset.

Three things tested, mirroring the hypoDD analysis:
1. Whole-catalog (relocated events only, nbranch>=2) Woodcock K -- compare
   against hypoDD's whole-catalog planar finding (K=0.10 pyocto, K=0.04 QM).
2. Per-GrowClust-cluster Woodcock K, using GrowClust's OWN native clusters
   (its "tree" grouping, column 13 of out.growclust_cat) rather than
   re-clustering spatially with DBSCAN. GrowClust links events by waveform-
   similarity/rfactor, not spatial proximity, so its native clusters turn out
   much larger (km-scale) than the ~500 m the user hypothesized -- informative,
   but not a like-for-like test of that specific hypothesis.
3. The actual apples-to-apples test: DBSCAN (same eps=0.15 km, min_samples=12
   as plot_hypodd_t1_basal_clusters.py) on GrowClust's own relocated
   positions -- does the same ~500 m-scale linear structure independently
   reappear using a completely different relocation algorithm and dataset
   (CC-only, not CC+catalog)?

Usage:
    python full_catalog_pipeline/analyze_growclust_t1.py
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
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN

from plot_hypodd_t1_basal_3d import (
    PYOCTO_STA, BASAL_MIN_KM, BASAL_MAX_KM, COLOR_STATION, woodcock_shape,
)
from plot_hypodd_t1_basal_clusters import (
    MIN_CLUSTER_FOR_PCA, QUANTIZATION_SUSPECT_THRESHOLD, DBSCAN_EPS_KM, DBSCAN_MIN_SAMPLES,
)


def dbscan_cluster_stats(df):
    """Same spatial-proximity clustering + Woodcock K test as
    plot_hypodd_t1_basal_clusters.py's cluster_stats(), applied here to
    GrowClust's relocated positions instead of hypoDD's."""
    pts = df[["ex", "nx", "depth"]].values
    labels = DBSCAN(eps=DBSCAN_EPS_KM, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(pts)
    df = df.copy()
    df["dbcluster"] = labels
    rows = []
    for cid, g in df[df["dbcluster"] >= 0].groupby("dbcluster"):
        if len(g) < MIN_CLUSTER_FOR_PCA:
            continue
        gpts = g[["ex", "nx", "depth"]].values
        eigval, eigvec, k = woodcock_shape(gpts)
        extent_km = pdist(gpts).max()
        depth_uniq = g["depth"].nunique() / len(g)
        rows.append({
            "dbcluster": cid, "n": len(g), "extent_km": extent_km, "woodcock_k": k,
            "depth_uniqueness": depth_uniq,
            "quantization_suspect": depth_uniq < QUANTIZATION_SUSPECT_THRESHOLD,
        })
    return df, pd.DataFrame(rows).sort_values("woodcock_k", ascending=False).reset_index(drop=True)

CAT_COLS = [
    "yr", "mo", "dy", "hr", "mi", "sec", "id", "lat", "lon", "depth", "mag",
    "iq", "cluster", "nbranch", "npair", "ndiffP", "ndiffS", "rmsP", "rmsS",
    "madh", "madz", "madt", "lat_orig", "lon_orig", "depth_orig",
]

SOURCES = {
    "pyocto": "full_catalog_pipeline/artifacts/full_run/T1_v5/growclust_pyocto/OUT/out.growclust_cat",
    "qm": "full_catalog_pipeline/artifacts/full_run/T1_v5/growclust_qm/OUT/out.growclust_cat",
}
OUT_DIR = "full_catalog_pipeline/artifacts/full_run/T1_v5"


def load_cat(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=CAT_COLS)


def project_km(lat, lon, to_ps, cx, cy):
    x, y = to_ps.transform(lon.values, lat.values)
    return (x - cx) / 1000.0, (y - cy) / 1000.0


def cluster_stats_growclust(df):
    rows = []
    for cid, g in df.groupby("cluster"):
        if len(g) < MIN_CLUSTER_FOR_PCA:
            continue
        pts = g[["ex", "nx", "depth"]].values
        eigval, eigvec, k = woodcock_shape(pts)
        extent_km = pdist(pts).max()
        depth_uniq = g["depth"].nunique() / len(g)
        rows.append({
            "cluster": cid, "n": len(g), "extent_km": extent_km, "woodcock_k": k,
            "depth_uniqueness": depth_uniq,
            "quantization_suspect": depth_uniq < QUANTIZATION_SUSPECT_THRESHOLD,
        })
    return pd.DataFrame(rows).sort_values("woodcock_k", ascending=False).reset_index(drop=True)


def plot_map(df, stats, stations_xy, name, out_path):
    fig, ax = plt.subplots(figsize=(10, 9))
    singles = df[df["nbranch"] < 2]
    ax.scatter(singles["ex"], singles["nx"], s=5, color="#cccccc", alpha=0.3,
               label=f"unrelocated singletons (n={len(singles)})", zorder=1)
    small = df[(df["nbranch"] >= 2) & (~df["cluster"].isin(set(stats["cluster"])))]
    ax.scatter(small["ex"], small["nx"], s=8, color="#999999", alpha=0.5,
               label=f"relocated, cluster too small for PCA (n={len(small)})", zorder=1)
    cmap = plt.get_cmap("tab20")
    for i, row in stats.iterrows():
        g = df[df["cluster"] == row["cluster"]]
        color = cmap(i % 20)
        ax.scatter(g["ex"], g["nx"], s=14, color=color, zorder=2)
        tag = " [SUSPECT]" if row["quantization_suspect"] else ""
        ax.annotate(f"#{int(row['cluster'])} K={row['woodcock_k']:.2f}{tag}",
                    (g["ex"].mean(), g["nx"].mean()), fontsize=7,
                    color="#cc0000" if row["quantization_suspect"] else "black",
                    xytext=(3, 3), textcoords="offset points")
    sx, sy = stations_xy
    ax.scatter(sx, sy, s=160, color=COLOR_STATION, marker="^", edgecolor="white",
               linewidth=1, label="T1 stations", zorder=3)
    ax.set_xlabel("East of centroid (km)")
    ax.set_ylabel("North of centroid (km)")
    ax.set_aspect("equal")
    ax.set_title(f"T1 {name}: GrowClust native clusters, basal events "
                 f"({BASAL_MIN_KM}-{BASAL_MAX_KM} km)\nlabels: Woodcock K per cluster")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    to_ps = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    sta = pd.read_csv(PYOCTO_STA, sep=r"\s+", header=None, names=["id", "lat", "lon", "elev"])
    st_x, st_y = to_ps.transform(sta["lon"].values, sta["lat"].values)
    cx, cy = st_x.mean(), st_y.mean()
    stations_xy = ((st_x - cx) / 1000.0, (st_y - cy) / 1000.0)

    for tag, path in SOURCES.items():
        cat = load_cat(path)
        cat["ex"], cat["nx"] = project_km(cat["lat"], cat["lon"], to_ps, cx, cy)
        n_total = len(cat)
        n_relocated = (cat["nbranch"] >= 2).sum()

        reloc = cat[cat["nbranch"] >= 2]
        basal = reloc[(reloc["depth"] >= BASAL_MIN_KM) & (reloc["depth"] <= BASAL_MAX_KM)]

        print(f"\n=== {tag} (GrowClust) ===")
        print(f"total events: {n_total}, relocated (nbranch>=2): {n_relocated} "
              f"({100*n_relocated/n_total:.1f}%)")
        print(f"basal band relocated events: {len(basal)}")

        if len(basal) >= MIN_CLUSTER_FOR_PCA:
            pts = basal[["ex", "nx", "depth"]].values
            eigval, eigvec, k = woodcock_shape(pts)
            depth_uniq = basal["depth"].nunique() / len(basal)
            print(f"WHOLE-CATALOG basal Woodcock K = {k:.3f} "
                  f"(eigenvalues normalized: {eigval/eigval.sum()}), "
                  f"depth-uniqueness = {depth_uniq:.3f}")

        stats = cluster_stats_growclust(basal)
        n_linear = (stats["woodcock_k"] > 1).sum()
        n_suspect_linear = (stats["quantization_suspect"] & (stats["woodcock_k"] > 1)).sum()
        print(f"GrowClust native clusters with >= {MIN_CLUSTER_FOR_PCA} basal events: {len(stats)}")
        print(f"  linear (K>1): {n_linear}, of which quantization-suspect: {n_suspect_linear}")
        if len(stats):
            print(stats[["cluster", "n", "extent_km", "woodcock_k", "depth_uniqueness",
                         "quantization_suspect"]].to_string(index=False))

        out_map = f"{OUT_DIR}/growclust_{tag}/growclust_{tag}_basal_clusters_map.png"
        plot_map(basal, stats, stations_xy, tag, out_map)
        print(f"wrote {out_map}")

        # apples-to-apples test: same DBSCAN spatial scale as the hypoDD analysis,
        # applied to GrowClust's own (independently-derived) relocated positions
        _, db_stats = dbscan_cluster_stats(basal)
        n_db_linear = (db_stats["woodcock_k"] > 1).sum()
        n_db_suspect_linear = (db_stats["quantization_suspect"] & (db_stats["woodcock_k"] > 1)).sum()
        print(f"\nDBSCAN (eps={DBSCAN_EPS_KM}km) spatial clusters, same scale as hypoDD analysis: "
              f"{len(db_stats)} with >= {MIN_CLUSTER_FOR_PCA} events")
        print(f"  linear (K>1): {n_db_linear}, of which quantization-suspect: {n_db_suspect_linear}")
        if len(db_stats):
            print(db_stats[["dbcluster", "n", "extent_km", "woodcock_k", "depth_uniqueness",
                            "quantization_suspect"]].to_string(index=False))


if __name__ == "__main__":
    main()
