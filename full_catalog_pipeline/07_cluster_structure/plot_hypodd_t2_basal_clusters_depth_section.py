#!/usr/bin/env python3
"""Depth cross-section (distance from array centroid vs. depth) for T2's basal population,
colored by the SAME DBSCAN cluster assignment and color scheme as
plot_hypodd_t2_basal_clusters.py's map view (hypodd_t2_basal_clusters_map_pyocto.png) --
so a reader can directly connect which colored cluster in the map corresponds to which
population in the depth section (e.g. cluster1's near-horizontal plunge is easy to miss in
map view but obvious in cross-section).

Usage:
    python full_catalog_pipeline/plot_hypodd_t2_basal_clusters_depth_section.py
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
import pyproj

from plot_hypodd_t2_basal_3d import (
    PYOCTO_RELOC, PYOCTO_STA, OUT_DIR, ICE_BED_DEPTH_KM,
    BASAL_MIN_KM, BASAL_MAX_KM, RADIUS_KM,
    load_stations, load_reloc, project_km,
)
from plot_hypodd_t2_basal_clusters import (
    DBSCAN_EPS_KM, MIN_CLUSTER_FOR_PCA, cluster_basal, cluster_stats,
)

OUT_PNG = f"{OUT_DIR}/hypodd_t2_basal_clusters_depth_section.png"

MUTED = "#8a8a86"
NOISE = "#cccccc"
INK = "#0b0b0b"


def main():
    to_ps = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    sta_p = load_stations(PYOCTO_STA)
    st_x, st_y = to_ps.transform(sta_p["lon"].values, sta_p["lat"].values)
    cx, cy = st_x.mean(), st_y.mean()

    pyocto = load_reloc(PYOCTO_RELOC)
    pyocto["ex"], pyocto["nx"] = project_km(pyocto, to_ps, cx, cy)
    pyocto = pyocto[np.hypot(pyocto["ex"], pyocto["nx"]) <= RADIUS_KM]
    basal_p = pyocto[(pyocto["depth"] >= BASAL_MIN_KM) & (pyocto["depth"] <= BASAL_MAX_KM)]

    clustered = cluster_basal(basal_p)
    stats = cluster_stats(clustered)
    clustered["r"] = np.hypot(clustered["ex"], clustered["nx"])

    fig, ax = plt.subplots(figsize=(11, 8))

    noise = clustered[clustered["cluster"] == -1]
    ax.scatter(noise["r"], noise["depth"], s=6, color=NOISE, alpha=0.4,
               label=f"unclustered noise (n={len(noise)})", zorder=1)

    kept_ids = set(stats["cluster"]) if len(stats) else set()
    small = clustered[(clustered["cluster"] >= 0) & (~clustered["cluster"].isin(kept_ids))]
    ax.scatter(small["r"], small["depth"], s=8, color="#999999", alpha=0.5,
               label=f"clusters too small for PCA (n<{MIN_CLUSTER_FOR_PCA}, n={len(small)})",
               zorder=1)

    cmap = plt.get_cmap("tab20")
    for i, row in stats.iterrows():
        g = clustered[clustered["cluster"] == row["cluster"]]
        color = cmap(i % 20)
        suspect_tag = " [SUSPECT]" if row["quantization_suspect"] else ""
        ax.scatter(g["r"], g["depth"], s=16, color=color, zorder=2,
                   label=f"#{int(row['cluster'])} n={row['n']} K={row['woodcock_k']:.2f}{suspect_tag}")

    ax.axhline(ICE_BED_DEPTH_KM, color=INK, linestyle="--", linewidth=1.3,
               label=f"ice-bed interface ({ICE_BED_DEPTH_KM} km)")
    ax.invert_yaxis()
    ax.set_xlabel("Distance from array centroid (km)")
    ax.set_ylabel("Depth (km)")
    ax.set_title(f"T2 basal-event depth section colored by DBSCAN cluster "
                 f"(eps={DBSCAN_EPS_KM}km, n={len(clustered)})\n"
                 f"same cluster IDs/colors as hypodd_t2_basal_clusters_map_pyocto.png")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
