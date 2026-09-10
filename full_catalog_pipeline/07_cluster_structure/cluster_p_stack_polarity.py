#!/usr/bin/env python3
"""Per user request: does stacking many events' raw (un-enveloped, sign-preserved) P
waveforms at one station -- after cross-correlation-based time alignment -- give a
cleaner, higher-SNR polarity determination than any single noisy event, and does the
resulting stack come out COHERENT (a clean, unambiguous first motion) or WASHED OUT
(no consistent polarity) at each station?

This is a genuinely independent, visual/waveform-domain test of the same question
[[rpnet_polarity_integration_result]] and [[skhash_cluster_fit_integration]] already
addressed via discrete per-event polarity picks: if a cluster's events truly shared one
focal mechanism, stacking should reinforce a clean, high-amplitude first break at every
station (same sign every event => constructive stack); if events have different/mixed
mechanisms, the stack should wash out toward the noise floor right at the P onset even
though the rest of the trace (background noise, later coda) still gains the expected
sqrt(N) SNR improvement.

Critically -- and this is why the user called out "no enveloping" -- alignment MUST be
done on the signed waveform without allowing a sign flip: cross-correlating for the
lag of maximum ABSOLUTE correlation would let two oppositely-polarized events "align" by
silently flipping one's sign, which would mechanically force a coherent-looking stack
regardless of whether the events actually agree in polarity -- defeating the entire
point of the test. This script aligns on the lag of maximum SIGNED correlation only
(never allowing a sign flip), against an iteratively-refined running stack (starting
from the catalog P-pick alignment, no CC).

Usage:
    python full_catalog_pipeline/cluster_p_stack_polarity.py --label cluster3
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
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import UTCDateTime

import config
from lib.day_volume_index import load_day_file_index_csv
from lib.deej_waveform_common import clean, cc_lag_signed, shift
from lib.windowing import RollingDayCache, extract_window

import catalog_paths

HYPODD_DIR = catalog_paths.work_dir("T1")
PHASE_DAT = f"{HYPODD_DIR}/input_files/phase.dat"
ALL_STATIONS = ["DEEJ", "ELZA", "LILA", "TJTJ", "OTIS", "LOUS", "SQIG"]
NETWORK = "7U"

WINDOW_PRE = 0.20     # seconds before the P pick fetched (also serves as the noise window)
WINDOW_POST = 0.30    # seconds after the P pick fetched
NOISE_WINDOW = (-0.18, -0.03)   # relative to P pick, for per-trace amplitude normalization
MAX_LAG_S = 0.03       # +/- CC search window for residual timing jitter
N_CC_ITERS = 3         # iterative re-alignment-to-running-stack passes
SIGNAL_LOOKAHEAD = 0.08  # seconds after the (CC-refined) pick to look for the first swing --
                         # validated 2026-07-30 against a 41-station/cluster human visual review
                         # (t1_stack_polarity_human_review.csv, tune_stack_first_motion.py): a
                         # 0.045s window (the old default) agreed with the human read only
                         # 42.3% of the time and left 13/39 cases indeterminate, because it often
                         # sliced through the leading edge of the true dominant P pulse, catching
                         # a small/early sample immediately dwarfed by a much bigger swing just
                         # past the cutoff (verified directly on DEEJ: -4.28 at 0.045s vs +5.72 at
                         # 0.050s, +9.91 at 0.055s). Sweeping window width against the human
                         # labels showed a real inflection at 0.08s (72% agreement, 0
                         # indeterminate) -- wider windows (0.10-0.30s) drop back to 61.5%,
                         # apparently picking up later, unrelated coda energy. 0.08s is a
                         # genuine sweet spot, not "wider is always better."
MIN_SNR = 3.0


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="cluster3")
    ap.add_argument("--ids-file", default=None, help=f"Defaults to {HYPODD_DIR}/<label>_event_ids.txt")
    ap.add_argument("--min-picks", type=int, default=10,
                    help="Minimum P picks within this cluster for a station to be included.")
    ap.add_argument("--out-dir", default=None, help=f"Defaults to {HYPODD_DIR}/<label>_stack_polarity")
    args = ap.parse_args()
    if args.ids_file is None:
        args.ids_file = f"{HYPODD_DIR}/{args.label}_event_ids.txt"
    if args.out_dir is None:
        args.out_dir = f"{HYPODD_DIR}/{args.label}_stack_polarity"
    return args


def load_events(ids, phase_dat, stations, network="7U"):
    sta_tags = {f"{network}.{s}": s for s in stations}
    events = {}
    cur = None
    with open(phase_dat) as f:
        for line in f:
            if line.startswith("#"):
                p = line.split()
                eid = int(p[-1])
                cur = eid if eid in ids else None
                if cur is not None:
                    yr, mo, dy, hr, mi, sc = p[1:7]
                    origin = UTCDateTime(int(yr), int(mo), int(dy), int(hr), int(mi), float(sc))
                    events[cur] = {"id": cur, "origin": origin, "picks": {}}
            elif cur is not None:
                parts = line.split()
                if len(parts) != 4:
                    continue
                sta, tt, wt, ph = parts
                if sta in sta_tags and ph == "P":
                    events[cur]["picks"][sta_tags[sta]] = float(tt)
    return events


def fetch_trace(cache, pick_time, sr):
    d = date(pick_time.year, pick_time.month, pick_time.day)
    row = {"primary_date": d.isoformat()}
    window_start = pick_time - WINDOW_PRE
    window_end = pick_time + WINDOW_POST
    data, status = extract_window(cache, row, window_start, window_end)
    cache.evict_before(row["primary_date"])
    if data is None or not data["complete"]:
        return None
    z = clean(data["Z"])
    t = np.arange(len(z)) / sr - WINDOW_PRE  # seconds relative to the catalog P pick
    noise_mask = (t >= NOISE_WINDOW[0]) & (t <= NOISE_WINDOW[1])
    noise_rms = np.sqrt(np.mean(z[noise_mask] ** 2)) if noise_mask.any() else 0.0
    if noise_rms == 0:
        return None
    return z / noise_rms, t  # noise-RMS-normalized (SNR-equal-weighted), sign preserved


def first_motion(stack, t, sr):
    noise_mask = (t >= NOISE_WINDOW[0]) & (t <= NOISE_WINDOW[1])
    noise_amp = np.std(stack[noise_mask]) if noise_mask.any() else 0.0
    if noise_amp == 0:
        return 0, np.nan
    pick_idx = int(round(-t[0] * sr))  # index of t==0 (the catalog P pick)
    sig_i1 = pick_idx + max(1, int(round(SIGNAL_LOOKAHEAD * sr)))
    window = stack[pick_idx:sig_i1]
    if len(window) == 0:
        return 0, np.nan
    peak_i = np.argmax(np.abs(window))
    peak_val = window[peak_i]
    snr = abs(peak_val) / noise_amp
    if snr < MIN_SNR:
        return 0, snr
    return (1 if peak_val > 0 else -1), snr


def stack_station(station, events, sr):
    day_index = load_day_file_index_csv(config.DAY_FILE_INDEX_CSV)
    cache = RollingDayCache(day_index, station)

    traces, common_t = [], None
    for eid, ev in sorted(events.items(), key=lambda kv: kv[1]["origin"]):
        if station not in ev["picks"]:
            continue
        pick_time = ev["origin"] + ev["picks"][station]
        result = fetch_trace(cache, pick_time, sr)
        if result is None:
            continue
        z, t = result
        if common_t is None:
            common_t = t
        elif len(z) != len(common_t):
            continue
        traces.append(z)

    if len(traces) < 3:
        return None

    mat = np.vstack(traces)
    # pass 0: catalog-pick-aligned stack, no CC yet
    stack = mat.mean(axis=0)
    for _ in range(N_CC_ITERS):
        lags = [cc_lag_signed(tr, stack, sr, MAX_LAG_S) for tr in mat]
        aligned = np.vstack([shift(tr, -lag) for tr, lag in zip(mat, lags)])
        stack = aligned.mean(axis=0)
        mat = aligned

    polarity, snr = first_motion(stack, common_t, sr)
    return dict(station=station, n=len(traces), t=common_t, mat=mat, stack=stack,
                polarity=polarity, snr=snr)


def plot_station(result, out_path, label):
    t, mat, stack = result["t"], result["mat"], result["stack"]
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})
    for tr in mat:
        axes[0].plot(t, tr, color="0.75", lw=0.5, alpha=0.6)
    axes[0].plot(t, stack, color="black", lw=1.8, label=f"stack (n={result['n']})")
    axes[0].axvline(0, color="steelblue", ls="--", lw=1)
    axes[0].set_ylabel("noise-RMS-normalized amplitude")
    axes[0].legend(loc="upper right", fontsize=9)
    pol_label = {1: "up", -1: "down", 0: "indeterminate"}[result["polarity"]]
    axes[0].set_title(f"{label} / {result['station']}: n={result['n']}, "
                       f"polarity={pol_label}, stack SNR={result['snr']:.1f}")
    axes[1].plot(t, stack, color="black", lw=1.5)
    axes[1].axvline(0, color="steelblue", ls="--", lw=1)
    axes[1].axhline(0, color="0.6", lw=0.8)
    axes[1].set_xlabel("seconds relative to catalog P pick")
    axes[1].set_ylabel("stack only")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ids = set(int(x) for x in open(args.ids_file))
    all_events = load_events(ids, PHASE_DAT, ALL_STATIONS, network=NETWORK)
    print(f"[{args.label}] events with any P pick: {len(all_events)}")

    pick_counts = {sta: 0 for sta in ALL_STATIONS}
    for ev in all_events.values():
        for sta in ev["picks"]:
            pick_counts[sta] += 1
    stations = [sta for sta in ALL_STATIONS if pick_counts[sta] >= args.min_picks]
    print(f"[{args.label}] stations used (>={args.min_picks} picks): {stations}")

    sr = config.SAMPLE_RATE_HZ
    rows = []
    for sta in stations:
        result = stack_station(sta, all_events, sr)
        if result is None:
            print(f"  {sta}: too few usable traces, skipped")
            continue
        pol_label = {1: "up", -1: "down", 0: "indeterminate"}[result["polarity"]]
        print(f"  {sta}: n={result['n']}, stack polarity={pol_label}, SNR={result['snr']:.1f}")
        out_png = f"{args.out_dir}/{args.label}_{sta.lower()}_stack.png"
        plot_station(result, out_png, args.label)
        rows.append(dict(station=sta, n=result["n"], polarity=result["polarity"], snr=result["snr"]))

    out_csv = f"{args.out_dir}/{args.label}_stack_polarities.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
