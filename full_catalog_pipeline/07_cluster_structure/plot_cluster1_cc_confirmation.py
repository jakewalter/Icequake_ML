#!/usr/bin/env python3
"""Demonstrates where/how cluster1's depth signal is confirmed: depth vs. S-P interval at
WICH and DRSC, before (raw catalog picks) and after (from-scratch single-component Z
cross-correlation refinement, cc_refine_component.py) -- shows the correlation directly,
including how much CC refinement STRENGTHENS the DRSC signal specifically.

Usage:
    python full_catalog_pipeline/plot_cluster1_cc_confirmation.py
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
import scipy.stats as st

import catalog_paths

OUT_DIR = catalog_paths.work_dir("T2")
WICH_CSV = f"{OUT_DIR}/t2_cluster1_wich_cc_refine_Z.csv"
DRSC_CSV = f"{OUT_DIR}/t2_cluster1_drsc_cc_refine_Z.csv"
OUT_PNG = f"{OUT_DIR}/t2_cluster1_cc_confirmation.png"

COLOR = {"WICH": "#15616d", "DRSC": "#7a4fa3"}
MIN_CC = 0.6
GRID = "#e4e3de"
MUTED = "#8a8a86"
INK = "#0b0b0b"

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


def scatter_panel(ax, df, x_col, y_col, color, title):
    ok = df["cc"] >= MIN_CC
    ax.scatter(df.loc[~ok, x_col] * 1000, df.loc[~ok, y_col] * 1000, s=10, color=MUTED,
               alpha=0.25, linewidth=0, label=f"CC<{MIN_CC} (n={(~ok).sum()})")
    ax.scatter(df.loc[ok, x_col] * 1000, df.loc[ok, y_col] * 1000, s=14, color=color,
               alpha=0.65, linewidth=0, label=f"CC>={MIN_CC} (n={ok.sum()})")

    sub = df[ok]
    r, p = st.pearsonr(sub[x_col], sub[y_col])
    slope, intercept, *_ = st.linregress(sub[x_col], sub[y_col])
    xs = np.array([df[x_col].min(), df[x_col].max()])
    ax.plot(xs * 1000, (intercept + slope * xs) * 1000, color=INK, linewidth=1.8, linestyle="--",
            zorder=5)
    ax.set_title(f"{title}\nr={r:+.3f} (p={p:.1g}), n={ok.sum()}", fontsize=11)
    return r, p


def main():
    wich = pd.read_csv(WICH_CSV)
    drsc = pd.read_csv(DRSC_CSV)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex="col")

    for row, (name, df) in enumerate([("WICH", wich), ("DRSC", drsc)]):
        color = COLOR[name]
        scatter_panel(axes[row, 0], df, "depth", "sp_original", color,
                      f"{name}: raw catalog picks")
        scatter_panel(axes[row, 1], df, "depth", "sp_refined", color,
                      f"{name}: cross-correlation refined (Z component)")
        for ax in axes[row]:
            ax.set_ylabel("S-P interval (ms)")
            ax.legend(fontsize=8, loc="best")
        axes[row, 0].annotate(name, xy=(-0.28, 0.5), xycoords="axes fraction", fontsize=15,
                               fontweight="bold", color=color, ha="center", va="center",
                               rotation=90)

    for ax in axes[1]:
        ax.set_xlabel("hypoDD depth (m)")

    fig.suptitle("Cluster1: depth vs. S-P interval, raw picks vs. cross-correlation-refined\n"
                 "(from-scratch single-component (Z) iterative refinement, not hypoDD's dt.cc)",
                 fontsize=14, y=1.0)
    fig.text(0.5, 0.955,
              "Dashed line = linear fit on CC>=0.6 events. Note DRSC's correlation gets "
              "STRONGER after refinement (picking noise was masking it), while WICH's stays "
              "strong throughout.",
              ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=[0.03, 0.03, 1, 0.85])
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
