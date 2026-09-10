#!/usr/bin/env python3
"""Why is T1 cluster3's 2nd S-window pulse (the loud, fixed-delay Love-wave-like
guided mode at DEEJ/LILA, near-zero-lag genuine S at LILA/TJTJ per
cluster3_direct_s_identification.py) SOMETIMES much larger than the 1st
(early, catalog-pick-adjacent) pulse and sometimes only modestly larger?

Two competing explanations:
  (a) path/radiation-pattern control: the ratio tracks station azimuth or
      S take-off angle (a population-level, shared-geometry effect), OR
  (b) per-event source control: cluster3 events do NOT share one focal
      mechanism (see t1_composite_focal_mech_result -- every station shows a
      mix of both P polarities across events), so the 1st/direct-S pulse's
      amplitude reflects each event's OWN (different) radiation pattern
      relative to that station -- which azimuth alone cannot predict without
      knowing each event's individual mechanism -- while the 2nd/guided pulse
      is roughly amplitude-stable because it is path/site-controlled, not
      source-controlled. Under (b), the ratio should show ~zero correlation
      with station azimuth/take-off angle in a population sense, even though
      it varies a lot event-to-event.

Reuses cluster3_direct_s_identification.py's per-event early_amp (1st pulse)
and global_amp (2nd/loudest pulse) envelope amplitudes, merged with each
event x station's azimuth/take-off angle from hypoDD's own partials.f source
file (cluster3_svd_hypoDD.src, the same geometry focal_mech_cluster3_fit.py
uses) and hypoDD depth.

Usage:
    python full_catalog_pipeline/cluster3_pulse_ratio_vs_geometry.py
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

import catalog_paths

_HYPODD = catalog_paths.work_dir("T1")
DIRECT_CSV = f"{_HYPODD}/cluster3_direct_s_identification.csv"
SRC_FILE = f"{_HYPODD}/cluster3_svd_hypoDD.src"
OUT_CSV = f"{_HYPODD}/cluster3_pulse_ratio_vs_geometry.csv"
OUT_PNG = f"{_HYPODD}/cluster3_pulse_ratio_vs_geometry.png"

SRC_COLS = ["evid", "lat", "lon", "sta", "elv", "dist", "az", "ainp", "ains",
            "ttp", "tts", "xp", "yp", "zp", "xs", "ys", "zs"]

STATIONS = ["DEEJ", "LILA", "TJTJ"]


def load():
    direct = pd.read_csv(DIRECT_CSV)
    direct = direct[direct["has_early_peak"]].copy()
    direct["ratio_2_over_1"] = direct["global_amp"] / direct["early_amp"]

    src = pd.read_csv(SRC_FILE, sep=r"\s+", header=None, names=SRC_COLS)
    src["sta"] = src["sta"].str.split(".").str[-1]
    src = src[["evid", "sta", "az", "ains"]].rename(columns={"evid": "id"})

    df = direct.merge(src, left_on=["id", "station"], right_on=["id", "sta"], how="inner")
    return df.drop_duplicates(subset=["id", "station"])


def main():
    df = load()
    print(f"merged n={len(df)}")

    rows = []
    for sta, g in df.groupby("station"):
        log_ratio = np.log(g["ratio_2_over_1"])
        az = np.radians(g["az"])
        feats = {
            "cos(az)": np.cos(az), "sin(az)": np.sin(az),
            "cos(2az)": np.cos(2 * az), "sin(2az)": np.sin(2 * az),
            "take-off(ains)": g["ains"], "depth": g["depth"],
        }
        corrs = {name: np.corrcoef(log_ratio, feat)[0, 1] for name, feat in feats.items()}
        row = {"station": sta, "n": len(g),
               "ratio_median": g["ratio_2_over_1"].median(),
               "ratio_iqr_lo": g["ratio_2_over_1"].quantile(.25),
               "ratio_iqr_hi": g["ratio_2_over_1"].quantile(.75)}
        row.update({f"corr_logratio_{k}": v for k, v in corrs.items()})
        rows.append(row)
        print(f"  {sta}: n={len(g)}, ratio median={row['ratio_median']:.2f} "
              f"IQR=[{row['ratio_iqr_lo']:.2f},{row['ratio_iqr_hi']:.2f}]")
        for name, c in corrs.items():
            print(f"    corr(log ratio, {name}) = {c:+.2f}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")

    fig, axes = plt.subplots(1, len(STATIONS), figsize=(4.5 * len(STATIONS), 4), sharey=True)
    for ax, sta in zip(axes, STATIONS):
        g = df[df["station"] == sta]
        ax.scatter(g["az"], g["ratio_2_over_1"], s=12, alpha=0.5)
        ax.set_yscale("log")
        ax.set_xlabel("station azimuth (deg)")
        ax.set_title(sta)
    axes[0].set_ylabel("2nd-pulse / 1st-pulse envelope amplitude")
    fig.suptitle("Cluster3: S-window pulse-2/pulse-1 amplitude ratio vs. azimuth\n"
                  "(flat/scattered => per-event source control, not shared path/radiation control)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
