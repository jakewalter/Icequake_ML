#!/usr/bin/env python3
"""Test 3 of PLAN_depth_resolvability.md: search the CC stacks for a depth phase.

A source at depth z radiates rays in every direction, not just the up-going one that becomes
the direct P. The down-going ray reflects off the ice base and returns to the same surface
receiver a fixed time later; a ray leaving upward reflects off the free surface and can arrive
later still. Either delay is set by the source's height in the ice column and the velocity --
NOT by the origin time, and (to first order at these steep takeoff angles) NOT by the
epicentre. That is what makes it the one observable here that measures ABSOLUTE depth, which
neither the S-P slope tests nor hypoDD can do.

Numbers that matter for reading the output: at Vp = 3.85 km/s and near-vertical incidence,
a bed bounce arrives 2(H-z)/Vp after the direct P -- 0.35 s for hypoDD's 1.32 km cluster-1
depth under a 2.00 km ice base, and ~0 s for a source actually at the bed. A free-surface
ghost arrives 2z/Vp later -- 0.69 s at 1.32 km, 1.04 s at the bed. The two hypotheses
therefore move in OPPOSITE directions with depth, so which one a coherent arrival matches
is itself diagnostic.

This script does the measurement half only: build long P-aligned stacks and locate coherent
late energy. It deliberately does not fit a depth. Per the plan, a coherent ghost at a
consistent delay would settle absolute depth, but its ABSENCE is uninformative (the phase may
simply be too weak or the free surface too rough at 200 Hz), so there is no point building
traveltime machinery before knowing there is a signal to fit.

Coherence, not amplitude, is the read. A linear stack of N traces suppresses incoherent noise
by ~sqrt(N) but the S coda is large and only partly coherent, so amplitude alone would flag it.
Semblance -- the energy of the sum over the summed energy, sample-normalised to [0, 1] -- is
near 1 only where the traces agree waveform-for-waveform across the whole cluster, which is
what a genuine repeatable phase does and what scattered coda does not.

Windows are extracted ONCE per event, long, and the P alignment runs on a short sub-slice of
that same array, so adding 2 s of record costs no extra disk reads over Test 1.

Usage:
    python full_catalog_pipeline/cluster_depth_phase_stack.py --array T2 --clusters 0 1 2 6 7
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

# P-alignment sub-window: identical to build_cluster_stacks.py so the alignment is the same one
# Test 1 measured; only the stacking window is longer.
SEARCH_PRE, SEARCH_POST = 0.15, 0.45
MAX_LAG = 0.15
ITERATIONS = 3
MIN_EVENTS = 15
MIN_CC = 0.5

LONG_PRE, LONG_POST = 0.30, 2.50   # stacked record length either side of the direct P
SEMB_WIN_S = 0.05                  # semblance window, ~4 cycles at the dominant frequency

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


def extract_long(df, cache, component, sr):
    """One long window per event, P-pick centred, padded by MAX_LAG on both sides.
    Returns {id: array} plus the sample index of the (unshifted) P pick within it."""
    n_pad = int(round(MAX_LAG * sr))
    i_p = n_pad + int(round(LONG_PRE * sr))
    n_total = i_p + int(round(LONG_POST * sr)) + n_pad
    out = {}
    for row in df.itertuples():
        t_abs = row.origin + row.p_offset
        w0 = t_abs - LONG_PRE - MAX_LAG
        w1 = t_abs + LONG_POST + MAX_LAG
        r = {"primary_date": w0.date.isoformat()}
        data, status = extract_window(cache, r, w0, w1)
        cache.evict_before(r["primary_date"])
        if data is None or len(data[component]) < n_total:
            continue
        tr = clean(data[component][:n_total])
        # Normalise on the P window only, so late arrivals keep their size RELATIVE to P
        # rather than every trace being scaled by whatever its largest coda swing happened
        # to be.
        peak = np.max(np.abs(tr[i_p - int(round(SEARCH_PRE * sr)):i_p + int(round(SEARCH_POST * sr))]))
        if peak > 0:
            out[row.id] = tr / peak
    return out, i_p


def align_p(traces, i_p, sr):
    """MCCC on the short P sub-window of the long traces. Returns {id: lag_s}, {id: cc}."""
    n_pre, n_post = int(round(SEARCH_PRE * sr)), int(round(SEARCH_POST * sr))
    n_core = n_pre + n_post
    ids = list(traces)
    lags = {e: 0.0 for e in ids}
    ccs = {e: 0.0 for e in ids}
    n_lag = int(round(MAX_LAG * sr))
    for _ in range(ITERATIONS):
        stack = np.zeros(n_core)
        for e in ids:
            s = i_p - n_pre + int(round(lags[e] * sr))
            stack += traces[e][s:s + n_core]
        stack /= len(ids)
        for e in ids:
            # normxcorr_lag wants the search trace padded by max_lag on both sides
            s = i_p - n_pre - n_lag
            lags[e], ccs[e] = normxcorr_lag(traces[e][s:s + n_core + 2 * n_lag], stack, sr, MAX_LAG)
    return lags, ccs


def stack_and_semblance(traces, ids, i_p, lags, sr):
    """Linear stack and sliding-window semblance over the long window, both on the
    P-aligned traces. Time axis is seconds relative to the direct P."""
    n_pre, n_post = int(round(LONG_PRE * sr)), int(round(LONG_POST * sr))
    n = n_pre + n_post
    mat = np.zeros((len(ids), n))
    for k, e in enumerate(ids):
        s = i_p - n_pre + int(round(lags[e] * sr))
        mat[k] = traces[e][s:s + n]
    stack = mat.mean(axis=0)
    num = np.cumsum(np.insert((mat.sum(axis=0)) ** 2, 0, 0.0))
    den = np.cumsum(np.insert((mat ** 2).sum(axis=0), 0, 0.0))
    w = int(round(SEMB_WIN_S * sr))
    semb = np.full(n, np.nan)
    half = w // 2
    lo = np.clip(np.arange(n) - half, 0, n - w)
    hi = lo + w
    d = den[hi] - den[lo]
    ok = d > 0
    semb[ok] = (num[hi] - num[lo])[ok] / (len(ids) * d[ok])
    t = np.arange(n) / sr - LONG_PRE
    return t, stack, semb, mat


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--array", default="T2", choices=["T1", "T2"])
    ap.add_argument("--clusters", nargs="+", type=int, required=True)
    ap.add_argument("--components", nargs="+", default=["Z"])
    ap.add_argument("--network", default="7U")
    args = ap.parse_args()

    sr = config.SAMPLE_RATE_HZ
    work = catalog_paths.work_dir(args.array)
    out_dir = os.path.join(work, "depth_phase")
    os.makedirs(out_dir, exist_ok=True)
    reloc_path = catalog_paths.reloc(args.array)
    sta_sel = catalog_paths.station_sel(args.array)
    sta = pd.read_csv(sta_sel, sep=r"\s+", header=None, names=["sta", "lat", "lon", "elev"])
    sta["code"] = sta["sta"].str.split(".").str[-1]
    day_index = load_day_file_index_csv(config.DAY_FILE_INDEX_CSV)

    rows = []
    for cl in args.clusters:
        ids_file = os.path.join(work, f"{args.array.lower()}_cluster{cl}_event_ids.txt")
        if not os.path.exists(ids_file):
            print(f"cluster {cl}: no id list")
            continue
        ids_all = load_ids(ids_file)
        panels = {}
        for code in sta["code"]:
            df = load_events(ids_all, catalog_paths.phase_dat(args.array), code,
                             network=args.network)
            if len(df) < MIN_EVENTS:
                continue
            df = df.merge(load_reloc(reloc_path), on="id")
            df = df.merge(load_dist_geodetic(reloc_path, sta_sel, code,
                                             network=args.network), on="id")
            cache = RollingDayCache(day_index, code)
            for comp in args.components:
                traces, i_p = extract_long(df, cache, comp, sr)
                if len(traces) < MIN_EVENTS:
                    continue
                lags, ccs = align_p(traces, i_p, sr)
                good = [e for e in traces if ccs[e] >= MIN_CC]
                use = good if len(good) >= MIN_EVENTS else list(traces)
                t, stack, semb, mat = stack_and_semblance(traces, use, i_p, lags, sr)
                np.savez(os.path.join(out_dir,
                                      f"{args.array.lower()}_c{cl}_{code}_{comp}_long.npz"),
                         t=t, stack=stack, semb=semb, n=len(use), sr=sr,
                         dist_km=float(df["dist_km"].median()),
                         depth_hypodd=float(df["depth"].median()))
                # Report the strongest coherent arrivals AFTER the direct P and after the S
                # coda has had time to decay -- the delay range where either candidate depth
                # phase could sit for a source anywhere in this ice column.
                for lo, hi, tag in ((0.15, 0.60, "bed_bounce_range"),
                                    (0.60, 1.40, "surface_ghost_range")):
                    m = (t >= lo) & (t < hi)
                    j = int(np.nanargmax(semb[m]))
                    tt = t[m][j]
                    rows.append(dict(
                        cluster=cl, station=code, component=comp, n=len(use), window=tag,
                        dist_km=float(df["dist_km"].median()),
                        depth_hypodd=float(df["depth"].median()),
                        t_peak_s=float(tt), semblance=float(semb[m][j]),
                        amp_rel_p=float(np.abs(stack[m][j]) / np.abs(stack).max()),
                        semb_median_window=float(np.nanmedian(semb[m])),
                        semb_at_P=float(np.nanmax(semb[np.abs(t) < 0.05])),
                    ))
                if comp == "Z":
                    panels[code] = (t, stack, semb, len(use),
                                    float(df["depth"].median()))
                print(f"  cluster {cl} {code} {comp}: n={len(use)} "
                      f"semb(P)={np.nanmax(semb[np.abs(t) < 0.05]):.2f} "
                      f"best 0.15-0.6 s={t[(t>=0.15)&(t<0.6)][int(np.nanargmax(semb[(t>=0.15)&(t<0.6)]))]:.3f} s "
                      f"best 0.6-1.4 s={t[(t>=0.6)&(t<1.4)][int(np.nanargmax(semb[(t>=0.6)&(t<1.4)]))]:.3f} s")

        if panels:
            fig, axes = plt.subplots(len(panels), 1, figsize=(12, 1.9 * len(panels) + 1.4),
                                     sharex=True)
            axes = np.atleast_1d(axes)
            for ax, (code, (t, stack, semb, n, zdd)) in zip(axes, sorted(panels.items())):
                ax.plot(t, stack / max(np.abs(stack).max(), 1e-12), color=SERIES[0], lw=0.9,
                        label="P-aligned stack")
                ax.set_ylim(-1.15, 1.15)
                ax.set_ylabel(f"{code}\nn={n}", fontsize=8)
                ax.set_yticks([])
                ax2 = ax.twinx()
                ax2.plot(t, semb, color=SERIES[1], lw=1.0, label="semblance")
                ax2.set_ylim(0, 1)
                ax2.grid(False)
                ax2.tick_params(labelsize=7)
                ax.axvline(0, color=INK, lw=1)
                # candidate depth-phase delays for this cluster's hypoDD depth, drawn only as
                # reading aids -- nothing here is fit to them
                ax.axvline(2 * (2.00 - zdd) / 3.85, color=MUTED, lw=1, ls="--")
                ax.axvline(2 * zdd / 3.85, color=MUTED, lw=1, ls=":")
            axes[0].legend(fontsize=8, frameon=False, loc="upper right")
            axes[-1].set_xlabel("time relative to the direct P arrival (s)")
            fig.suptitle(f"{args.array} cluster {cl} — long P-aligned stacks and semblance "
                         f"(dashed: bed bounce, dotted: surface ghost, at hypoDD's depth)",
                         fontsize=11, color=INK)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            out = os.path.join(work, f"{args.array.lower()}_cluster{cl}_depth_phase.png")
            fig.savefig(out, dpi=145, bbox_inches="tight")
            plt.close(fig)
            print(f"wrote {out}")

    if rows:
        df_out = pd.DataFrame(rows)
        csv = os.path.join(work, f"{args.array.lower()}_depth_phase_semblance.csv")
        df_out.to_csv(csv, index=False)
        print(f"\nwrote {csv}  ({len(df_out)} rows)")


if __name__ == "__main__":
    main()
