#!/usr/bin/env python3
"""Investigation 1, QC step (see t1_composite_focal_mech_and_vpvs_consistency_plan memory):
visual review of a stratified sample of automated P first-motion polarity picks (up/down x
low/high SNR, across stations) -- the plan explicitly flags that automated polarity needs
checking against a hand-reviewed subset before trusting it at scale.

Re-fetches only the sampled event-station waveform snippets (cheap, ~1 read/sec each) rather
than re-running the full cluster sweep.

Usage:
    python full_catalog_pipeline/focal_mech_cluster3_qc_panel.py
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

import sys
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import UTCDateTime

sys.path.insert(0, ".")
import config
from focal_mech_cluster3_polarities import (
    IDS_FILE, PHASE_DAT, STATIONS, NETWORK, NOISE_WINDOW, SIGNAL_LOOKAHEAD,
    load_events, first_motion_polarity,
)
from lib.day_volume_index import load_day_file_index_csv
from lib.windowing import RollingDayCache

import catalog_paths

POLARITIES_CSV = catalog_paths.work_dir("T1") + "/cluster3_polarities.csv"
OUT_PNG = catalog_paths.work_dir("T1") + "/cluster3_polarity_qc_panel.png"
PLOT_WINDOW = (-0.15, 0.20)  # seconds relative to pick, for the QC plot only
N_PER_STRATUM = 6  # up/high-snr, up/low-snr, down/high-snr, down/low-snr


def sample_stratified(df, rng):
    """Sample across BOTH accepted polarities (up/down, split by borderline vs. comfortably
    above the MIN_SNR=3 threshold) AND rejected/indeterminate picks -- checking only the
    accepted ones would miss whether the threshold itself is well-calibrated (e.g. rejecting
    genuinely clear first motions, or accepting noisy ones just above the cut)."""
    df = df.copy()
    strata = [
        ("up, high-SNR", (df["polarity"] == 1) & (df["snr"] >= 6)),
        ("up, borderline-SNR", (df["polarity"] == 1) & (df["snr"] < 6)),
        ("down, high-SNR", (df["polarity"] == -1) & (df["snr"] >= 6)),
        ("down, borderline-SNR", (df["polarity"] == -1) & (df["snr"] < 6)),
        ("indeterminate (rejected)", df["polarity"] == 0),
    ]
    samples = []
    for label, mask in strata:
        sub = df[mask]
        n = min(N_PER_STRATUM, len(sub))
        if n == 0:
            print(f"  stratum '{label}': 0 rows, skipping")
            continue
        samples.append(sub.sample(n=n, random_state=rng))
    return pd.concat(samples, ignore_index=True)


def main():
    pol_df = pd.read_csv(POLARITIES_CSV)
    print(f"polarity rows to sample from: {len(pol_df)}")

    sample = sample_stratified(pol_df, rng=0)
    print(f"QC sample size: {len(sample)}")
    print(sample.groupby(["polarity", "reliable"]).size())

    ids = set(sample["id"].unique().tolist())
    events = load_events(ids, PHASE_DAT, STATIONS, network=NETWORK)
    day_index = load_day_file_index_csv(config.DAY_FILE_INDEX_CSV)
    caches = {sta: RollingDayCache(day_index, sta) for sta in STATIONS}

    n = len(sample)
    ncols = 6
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, row) in zip(axes, sample.iterrows()):
        eid, sta = int(row["id"]), row["station"]
        ev = events[eid]
        p_offset = ev["picks"][sta]
        pick_time = ev["origin"] + p_offset
        d = date(pick_time.year, pick_time.month, pick_time.day)
        traces = caches[sta].get(d.isoformat())
        tr = traces["HHZ"]
        sr = tr.stats.sampling_rate
        pick_idx = int(round((pick_time - tr.stats.starttime) * sr))
        i0 = pick_idx + int(round(PLOT_WINDOW[0] * sr))
        i1 = pick_idx + int(round(PLOT_WINDOW[1] * sr))
        z = tr.data.astype(np.float64)[max(0, i0):i1]
        t = (np.arange(len(z)) + max(0, i0) - pick_idx) / sr

        ax.plot(t, z, color="black", linewidth=0.8)
        ax.axvline(0, color="red", linewidth=1, linestyle="--")
        ax.axhline(z[:max(1, int(round(-PLOT_WINDOW[0] * sr)))].mean() if i0 >= 0 else 0,
                   color="grey", linewidth=0.5)
        tag = {1: "UP", -1: "DOWN", 0: "REJECTED"}[row["polarity"]]
        snr_txt = f"snr={row['snr']:.1f}" if np.isfinite(row["snr"]) else "snr=nan"
        ax.set_title(f"evt{eid} {sta}\n{tag} {snr_txt}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        caches[sta].evict_before(d.isoformat())

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Cluster3 P first-motion polarity QC sample\n"
                 "(red dashed = catalog P pick; automated polarity = sign of first "
                 f"{SIGNAL_LOOKAHEAD*1000:.0f}ms excursion after pick)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
