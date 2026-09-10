#!/usr/bin/env python3
"""Same format as plot_cluster1_cc_confirmation.py, for cluster0 (JULA, the nearest station)
-- depth vs. S-P interval, raw catalog picks vs. from-scratch single-component (Z)
cross-correlation refinement. Cluster0 is the contrast case: unlike cluster1, refinement does
NOT reveal/strengthen a hidden depth signal here -- both panels stay flat/scattered.

Usage:
    python full_catalog_pipeline/plot_cluster0_cc_confirmation.py
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
import pandas as pd

from plot_cluster1_cc_confirmation import scatter_panel, GRID, MUTED  # noqa: F401 (rcParams side effect below)

import catalog_paths

OUT_DIR = catalog_paths.work_dir("T2")
JULA_CSV = f"{OUT_DIR}/t2_cluster0_cc_refine_Z.csv"
OUT_PNG = f"{OUT_DIR}/t2_cluster0_cc_confirmation.png"

COLOR = "#c1440e"  # distinct from cluster1's WICH/DRSC colors, echoes "artifact/negative" framing

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": GRID,
    "axes.linewidth": 1,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 1,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def main():
    df = pd.read_csv(JULA_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    scatter_panel(axes[0], df, "depth", "sp_original", COLOR, "JULA: raw catalog picks")
    scatter_panel(axes[1], df, "depth", "sp_refined", COLOR,
                  "JULA: cross-correlation refined (Z component)")
    for ax in axes:
        ax.set_xlabel("hypoDD depth (m)")
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("S-P interval (ms)")
    axes[0].annotate("cluster0\n(n=1667, dominant)", xy=(-0.32, 0.5), xycoords="axes fraction",
                      fontsize=13, fontweight="bold", color=COLOR, ha="center", va="center",
                      rotation=90)

    fig.suptitle("Cluster0: depth vs. S-P interval, raw picks vs. cross-correlation-refined\n"
                 "(from-scratch single-component (Z) iterative refinement, not hypoDD's dt.cc)",
                 fontsize=14, y=1.12)
    fig.text(0.5, 1.0,
              "Dashed line = linear fit on CC>=0.6 events. Contrast with cluster1: refinement "
              "does NOT reveal a hidden trend here -- stays flat/scattered before and after.",
              ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=[0.05, 0.03, 1, 0.80])
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
