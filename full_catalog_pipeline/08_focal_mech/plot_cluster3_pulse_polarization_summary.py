#!/usr/bin/env python3
"""Summary figure for cluster3_sp_pulse_polarization.py's per-event CSVs: dip
(1st vs 2nd pulse) and azimuth-shift-mod-180 (2nd relative to 1st), per station.
See [[t1-sp-vs-depth-vpvs-check-result]] Result 7."""

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

import catalog_paths

_HYPODD = catalog_paths.work_dir("T1")
STATIONS = ["DEEJ", "LILA", "TJTJ"]

fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey="row")
for col, station in enumerate(STATIONS):
    df = pd.read_csv(f"{_HYPODD}/cluster3_sp_pulse_polarization_{station.lower()}.csv")

    ax = axes[0, col]
    bins = np.linspace(0, 90, 19)
    ax.hist(df["p1_dip"], bins=bins, alpha=0.5, label="1st pulse (S)", color="tab:blue")
    ax.hist(df["p2_dip"], bins=bins, alpha=0.5, label="2nd pulse (site burst)", color="tab:red")
    ax.set_title(f"{station}: polarization dip off horizontal (n={len(df)})")
    ax.set_xlabel("dip (deg, 0=horizontal, 90=vertical)")
    if col == 0:
        ax.set_ylabel("count")
        ax.legend(fontsize=8)

    ax = axes[1, col]
    shift = (df["p2_horiz_azimuth"] - df["p1_horiz_azimuth"]) % 180
    ax.hist(shift, bins=np.linspace(0, 180, 19), color="tab:purple", alpha=0.7)
    ax.axvline(90, color="k", ls="--", lw=1, label="90 deg (orthogonal)")
    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.set_title(f"{station}: horiz. azimuth shift, 2nd rel. 1st (mod 180)")
    ax.set_xlabel("azimuth shift (deg)")
    if col == 0:
        ax.set_ylabel("count")
        ax.legend(fontsize=8)

fig.suptitle("Cluster3 S-window double-pulse: does the 2nd/site-effect pulse share the\n"
             "1st/direct-S pulse's polarization (body-wave multiple) or differ (guided/converted mode)?",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out_png = f"{_HYPODD}/cluster3_sp_pulse_polarization_summary.png"
fig.savefig(out_png, dpi=140)
print(f"wrote {out_png}")
