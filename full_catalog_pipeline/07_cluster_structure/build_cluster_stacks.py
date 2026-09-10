#!/usr/bin/env python3
"""Test 1 of PLAN_depth_resolvability.md: per-cluster, per-station cross-correlation stacks.

Not a test on its own -- the input to the depth tests (2-5). For each cluster at each station
it MCCC-aligns the events on P, stacks the aligned single components (Z, N and E kept
separate; no envelopes, which discard polarity and smear the onset), and writes:

  * the stacked waveform per station and component
  * a COMPOSITE P and S arrival time per station, with the alignment scatter as its
    uncertainty -- these are the observations the depth-pinned relocation (Test 2) inverts
  * a per-cluster figure: every station's stack on a common time axis relative to P

Why the composite is worth more than the individual picks: stacking N coherent traces lifts
SNR by ~sqrt(N), so the composite onset is far better determined than any single pick, and
the alignment is measured from the waveforms rather than read from the catalog -- which is the
whole point, since the catalog picks are what hypoDD already fit.

P and S are aligned SEPARATELY (each against its own stack, in its own window), so the
composite S-P is a waveform measurement end to end.

Usage:
    python full_catalog_pipeline/build_cluster_stacks.py --array T2 --clusters 0 1 2 6 7
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import catalog_paths
import config
from cc_refine_component import normxcorr_lag
from lib.day_volume_index import load_day_file_index_csv
from lib.deej_waveform_common import (
    load_ids, load_events, load_reloc, load_dist_geodetic, clean,
)
from lib.windowing import RollingDayCache, extract_window

SEARCH_PRE, SEARCH_POST = 0.15, 0.45
MAX_LAG = 0.15
ITERATIONS = 3
MIN_EVENTS = 15
MIN_CC = 0.5

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, SECONDARY, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3de", "#8a8a86"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.linewidth": 1,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "xtick.color": SECONDARY, "ytick.color": SECONDARY,
    "axes.labelcolor": SECONDARY, "text.color": INK,
})


def align_and_stack(df, cache, pick_col, component, sr):
    """MCCC-align every event's window around `pick_col` and return
    (stack, lags, ccs, n_core, n_pad). Lags are seconds relative to each event's own pick."""
    n_core = int(round((SEARCH_PRE + SEARCH_POST) * sr))
    n_pad = int(round(MAX_LAG * sr))
    n_total = n_core + 2 * n_pad
    padded = {}
    for row in df.itertuples():
        t_abs = row.origin + getattr(row, pick_col)
        w0 = t_abs - SEARCH_PRE - MAX_LAG
        w1 = t_abs + SEARCH_POST + MAX_LAG
        r = {"primary_date": w0.date.isoformat()}
        data, status = extract_window(cache, r, w0, w1)
        cache.evict_before(r["primary_date"])
        if data is None or len(data[component]) < n_total:
            continue
        tr = clean(data[component][:n_total])
        peak = np.max(np.abs(tr))
        if peak > 0:
            padded[row.id] = tr / peak
    ids_ok = [e for e in df["id"] if e in padded]
    if len(ids_ok) < MIN_EVENTS:
        return None, {}, {}, n_core, n_pad
    lags = {e: 0.0 for e in ids_ok}
    ccs = {e: 0.0 for e in ids_ok}
    stack = None
    for _ in range(ITERATIONS):
        stack = np.zeros(n_core)
        for e in ids_ok:
            start = n_pad + int(round(lags[e] * sr))
            stack += padded[e][start:start + n_core]
        stack /= len(ids_ok)
        for e in ids_ok:
            lags[e], ccs[e] = normxcorr_lag(padded[e], stack, sr, MAX_LAG)
    # final stack from the converged lags
    stack = np.zeros(n_core)
    for e in ids_ok:
        start = n_pad + int(round(lags[e] * sr))
        stack += padded[e][start:start + n_core]
    stack /= len(ids_ok)
    return stack, lags, ccs, n_core, n_pad


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--array", default="T2", choices=["T1", "T2"])
    ap.add_argument("--clusters", nargs="+", type=int, required=True)
    ap.add_argument("--components", nargs="+", default=["Z", "N", "E"])
    args = ap.parse_args()

    sr = config.SAMPLE_RATE_HZ
    work = catalog_paths.work_dir(args.array)
    out_dir = os.path.join(work, "stacks")
    os.makedirs(out_dir, exist_ok=True)
    reloc_path = catalog_paths.reloc(args.array)
    sta = pd.read_csv(catalog_paths.station_sel(args.array), sep=r"\s+", header=None,
                      names=["sta", "lat", "lon", "elev"])
    sta["code"] = sta["sta"].str.split(".").str[-1]
    day_index = load_day_file_index_csv(config.DAY_FILE_INDEX_CSV)

    rows = []
    for cl in args.clusters:
        ids_file = os.path.join(work, f"{args.array.lower()}_cluster{cl}_event_ids.txt")
        if not os.path.exists(ids_file):
            print(f"cluster {cl}: no id list")
            continue
        ids = load_ids(ids_file)
        stacks = {}
        for code in sta["code"]:
            df = load_events(ids, catalog_paths.phase_dat(args.array), code, network="7U")
            if len(df) < MIN_EVENTS:
                continue
            df = df.merge(load_reloc(reloc_path), on="id")
            df = df.merge(load_dist_geodetic(reloc_path, catalog_paths.station_sel(args.array),
                                             code, network="7U"), on="id")
            cache = RollingDayCache(day_index, code)
            for comp in args.components:
                p_stack, p_lags, p_ccs, n_core, _ = align_and_stack(df, cache, "p_offset", comp, sr)
                s_stack, s_lags, s_ccs, _, _ = align_and_stack(df, cache, "s_offset", comp, sr)
                if p_stack is None or s_stack is None:
                    continue
                common = [e for e in df["id"] if e in p_lags and e in s_lags]
                good = [e for e in common if p_ccs[e] >= MIN_CC and s_ccs[e] >= MIN_CC]
                use = good if len(good) >= MIN_EVENTS else common
                # Composite S-P: each event's own CC-refined S minus its CC-refined P, then
                # the median across the cluster. Its uncertainty is the scatter of that
                # quantity, not the scatter of the raw picks.
                idx = df.set_index("id")
                sp = np.array([(idx.loc[e, "s_offset"] + s_lags[e])
                               - (idx.loc[e, "p_offset"] + p_lags[e]) for e in use])
                rows.append(dict(
                    cluster=cl, station=code, component=comp, n=len(use),
                    dist_km=float(df["dist_km"].median()),
                    depth_hypodd=float(df["depth"].median()),
                    sp_composite=float(np.median(sp)),
                    # CC-refined composite P and S offsets, still measured from each event's
                    # hypoDD origin. That origin is a shared unknown per cluster, so it cancels
                    # in any inter-station difference -- which is what lets the depth-pinned
                    # relocation (Test 2) free the epicentre instead of taking hypoDD's.
                    p_composite=float(np.median(
                        [idx.loc[e, "p_offset"] + p_lags[e] for e in use])),
                    s_composite=float(np.median(
                        [idx.loc[e, "s_offset"] + s_lags[e] for e in use])),
                    p_se_ms=float(np.std([idx.loc[e, "p_offset"] + p_lags[e] for e in use],
                                         ddof=1) / np.sqrt(len(use)) * 1000),
                    s_se_ms=float(np.std([idx.loc[e, "s_offset"] + s_lags[e] for e in use],
                                         ddof=1) / np.sqrt(len(use)) * 1000),
                    sp_mad_ms=float(np.median(np.abs(sp - np.median(sp))) * 1000),
                    sp_se_ms=float(np.std(sp, ddof=1) / np.sqrt(len(sp)) * 1000),
                    mean_cc_p=float(np.mean([p_ccs[e] for e in use])),
                    mean_cc_s=float(np.mean([s_ccs[e] for e in use])),
                ))
                np.savez(os.path.join(out_dir, f"{args.array.lower()}_c{cl}_{code}_{comp}.npz"),
                         p_stack=p_stack, s_stack=s_stack, sr=sr,
                         search_pre=SEARCH_PRE, sp_composite=float(np.median(sp)))
                if comp == "Z":
                    stacks[code] = (p_stack, s_stack, float(np.median(sp)), len(use))
                print(f"  cluster {cl} {code} {comp}: n={len(use)} "
                      f"S-P={np.median(sp)*1000:.1f}±{np.std(sp, ddof=1)/np.sqrt(len(sp))*1000:.1f} ms "
                      f"CC(P/S)={np.mean([p_ccs[e] for e in use]):.2f}/{np.mean([s_ccs[e] for e in use]):.2f}")

        if stacks:
            fig, axes = plt.subplots(len(stacks), 1, figsize=(11, 1.5 * len(stacks) + 1.5),
                                     sharex=True)
            axes = np.atleast_1d(axes)
            t = np.arange(len(next(iter(stacks.values()))[0])) / sr - SEARCH_PRE
            for ax, (code, (ps, ss, spc, n)) in zip(axes, sorted(stacks.items())):
                ax.plot(t, ps / max(np.abs(ps).max(), 1e-12), color=SERIES[0], lw=1.1,
                        label="P-aligned stack")
                ax.plot(t + spc, ss / max(np.abs(ss).max(), 1e-12), color=SERIES[1], lw=1.1,
                        label="S-aligned stack (placed at composite S-P)")
                ax.axvline(0, color=INK, lw=1)
                ax.set_ylabel(f"{code}\nn={n}", fontsize=8)
                ax.set_yticks([])
            axes[0].legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
            axes[-1].set_xlabel("time relative to the composite P arrival (s)")
            fig.suptitle(f"{args.array} cluster {cl} — CC stacks, vertical component "
                         f"(each station's S stack drawn at its measured S-P)",
                         fontsize=11, color=INK)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            out = os.path.join(work, f"{args.array.lower()}_cluster{cl}_stacks.png")
            fig.savefig(out, dpi=145, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {out}")

    if rows:
        df_out = pd.DataFrame(rows)
        csv = os.path.join(work, f"{args.array.lower()}_cluster_composite_arrivals.csv")
        df_out.to_csv(csv, index=False)
        print(f"\nwrote {csv}  ({len(df_out)} cluster/station/component rows)")


if __name__ == "__main__":
    main()
