#!/usr/bin/env python3
"""Compare each array's hypoDD relocation under the reflection-seismology velocity model
(hypodd_vels1d/) against the v5 run it replaces (hypodd/).

Both runs relocate the SAME events from the SAME dt.ct/dt.cc, so every difference here is
attributable to the velocity model alone and events can be compared one-to-one by id.

Reports, per array:
  * how many events each run relocated, and how many are common to both;
  * the depth distribution before/after, plus where the ice-bed interface sits;
  * per-event horizontal and vertical shifts; and
  * the weighted RMS residuals hypoDD itself reports (hypoDD.log), which is the closest
    thing to an objective "did the model fit the data better" number available -- lower
    residual for the same data and the same weighting scheme means the new model explains
    the differential times better.

Usage:
    python full_catalog_pipeline/compare_vels1d_relocations.py --array T2
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

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj

from hypodd_relocate import ARRAY_CONFIG
from vels1d_model import ICE_THICKNESS_KM

RELOC_COLS = [
    "id", "lat", "lon", "depth", "x", "y", "z", "ex", "ey", "ez",
    "year", "month", "day", "hour", "minute", "second", "mag",
    "nccp", "nccs", "nctp", "nlts", "rcc", "rct", "cid",
]

C_OLD = "#8a8a86"
C_NEW = "#2a78d6"
INK = "#0b0b0b"
GRID = "#e4e3de"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": GRID, "axes.linewidth": 1,
    "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_reloc(path):
    return pd.read_csv(path, sep=r"\s+", header=None, names=RELOC_COLS)


def final_iteration(log_path, cluster=1):
    """Final-iteration summary for one cluster: retained-data percentages and RMS residuals.

    Returns dict(ev_pct, ct_pct, cc_pct, rmsct_ms, rmscc_ms, rmsst_ms), or None.

    The retained percentages matter as much as the residuals: hypoDD's later iteration sets
    discard outlier differential times, so a run can post a lower RMS simply by having thrown
    more data away. An RMS comparison is only meaningful alongside how much data produced it.

    hypoDD reprints a two-line header before every iteration's summary row:
        IT EV CT CC  RMSCT      RMSCC   RMSST  DX DY DZ DT OS AQ CND
            %  %  %  ms     %   ms     %   ms   m  m  m ms  m
    so the data row is always the second line after the header -- which is how we find it,
    because the leading IT/EV fields sometimes print blank and the row is then 16 fields
    instead of 17. The trailing 12 fields are stable either way, so RMSCT/RMSCC are read
    from the end.

    Cluster 1 is hypoDD's largest cluster; the log runs clusters back to back, so the last
    row in the file belongs to whichever 2-event cluster finished last, not to the catalog.
    """
    if not os.path.exists(log_path):
        return None
    row, in_cluster, countdown = None, False, 0
    with open(log_path, errors="replace") as f:
        for line in f:
            if line.startswith("RELOCATION OF CLUSTER:"):
                # Fortran writes "**" when the cluster number overflows its format width;
                # those are high-numbered (tiny) clusters, never the one we want.
                field = line.split(":")[1].split()[0]
                in_cluster = field.isdigit() and int(field) == cluster
                continue
            if not in_cluster:
                continue
            if line.lstrip().startswith("IT   EV"):
                countdown = 2
                continue
            if countdown:
                countdown -= 1
                if countdown == 0:
                    parts = line.split()
                    if len(parts) >= 15:
                        try:
                            row = {
                                "ev_pct": float(parts[-15]),
                                "ct_pct": float(parts[-14]),
                                "cc_pct": float(parts[-13]),
                                "rmsct_ms": float(parts[-12]),
                                "rmscc_ms": float(parts[-10]),
                                "rmsst_ms": float(parts[-8]),
                            }
                        except ValueError:
                            pass
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--array", required=True, choices=sorted(ARRAY_CONFIG))
    args = parser.parse_args()
    array = args.array

    old_dir = ARRAY_CONFIG[array]["working_dir"]
    new_dir = os.path.join(os.path.dirname(old_dir), "hypodd_vels1d")
    old = load_reloc(os.path.join(old_dir, "output_files", "hypoDD.reloc"))
    new = load_reloc(os.path.join(new_dir, "output_files", "hypoDD.reloc"))

    merged = old.merge(new, on="id", suffixes=("_old", "_new"))
    to_ps = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    x_old, y_old = to_ps.transform(merged["lon_old"].values, merged["lat_old"].values)
    x_new, y_new = to_ps.transform(merged["lon_new"].values, merged["lat_new"].values)
    dh = np.hypot(x_new - x_old, y_new - y_old) / 1000.0
    dz = merged["depth_new"].values - merged["depth_old"].values

    bed = ICE_THICKNESS_KM[array]
    print(f"\n=== {array}: v5 model vs reflection (vels1d) model ===")
    print(f"  events relocated: v5 {len(old)}, vels1d {len(new)}, common {len(merged)}")
    for name, d in [("v5", old["depth"]), ("vels1d", new["depth"])]:
        print(f"  depth {name:>7}: median {d.median():.3f} km, "
              f"IQR {d.quantile(.25):.3f}-{d.quantile(.75):.3f}, "
              f"{(d > bed).mean() * 100:.1f}% below the {bed} km interface")
    print(f"  per-event shift: horizontal median {np.median(dh) * 1000:.0f} m "
          f"(90th pct {np.percentile(dh, 90) * 1000:.0f} m), "
          f"vertical median {np.median(dz) * 1000:+.0f} m "
          f"(10-90th pct {np.percentile(dz, 10) * 1000:+.0f} to {np.percentile(dz, 90) * 1000:+.0f} m)")
    print("  final iteration of hypoDD's largest cluster (RMS is only comparable alongside "
          "how much data survived the outlier trimming):")
    for name, d in [("v5", old_dir), ("vels1d", new_dir)]:
        it = final_iteration(os.path.join(d, "hypoDD_log.txt"))
        if it:
            print(f"    {name:>7}: RMS catalog {it['rmsct_ms']:.0f} ms, cross-corr "
                  f"{it['rmscc_ms']:.0f} ms, station {it['rmsst_ms']:.0f} ms  |  data kept: "
                  f"events {it['ev_pct']:.0f}%, catalog dt {it['ct_pct']:.0f}%, "
                  f"cross-corr dt {it['cc_pct']:.0f}%")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # A handful of numerically loose events land many km below the array; they would
    # otherwise stretch every axis and squash the population that matters.
    depth_hi = float(np.percentile(np.concatenate([old["depth"], new["depth"]]), 99.5))
    dz_lim = float(np.percentile(np.abs(dz) * 1000.0, 99.0))

    ax = axes[0]
    bins = np.linspace(0, depth_hi, 80)
    ax.hist(old["depth"], bins=bins, orientation="horizontal", color=C_OLD, alpha=0.65,
            label=f"v5 model (n={len(old)})")
    ax.hist(new["depth"], bins=bins, orientation="horizontal", histtype="step", lw=1.8,
            color=C_NEW, label=f"vels1d model (n={len(new)})")
    ax.axhline(bed, color=INK, ls="--", lw=1, label=f"ice-bed ({bed} km)")
    ax.set_ylim(depth_hi, 0)
    ax.set_xlabel("events")
    ax.set_ylabel("depth (km)")
    ax.set_title("Relocated depth distribution")
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.hist(np.clip(dz * 1000.0, -dz_lim, dz_lim), bins=80, color=C_NEW, alpha=0.8)
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlim(-dz_lim, dz_lim)
    ax.set_xlabel("depth change, vels1d - v5 (m)  [clipped at the 99th pct]")
    ax.set_ylabel("events")
    ax.set_title(f"Per-event vertical shift (n={len(merged)})")

    ax = axes[2]
    ax.scatter(merged["depth_old"], dz * 1000.0, s=4, alpha=0.25, color=C_NEW)
    ax.axhline(0, color=INK, lw=1)
    ax.axvline(bed, color=INK, ls="--", lw=1)
    ax.set_xlim(0, depth_hi)
    ax.set_ylim(-dz_lim, dz_lim)
    ax.set_xlabel("v5 depth (km)")
    ax.set_ylabel("depth change (m)")
    ax.set_title("Vertical shift vs. original depth")

    fig.suptitle(f"{array}: hypoDD relocation under the reflection 1D model vs. the v5 model "
                 f"(same events, same dt.ct/dt.cc)", fontsize=12)
    fig.tight_layout()
    out_png = os.path.join(new_dir, f"vels1d_vs_v5_{array}.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_png}\n")


if __name__ == "__main__":
    main()
