#!/usr/bin/env python3
"""Uses the human manual review (full_catalog_pipeline/artifacts/full_run/T1_v5/hypodd/
human_review/t1_stack_polarity_human_review.csv) as ground truth to find out WHY
cluster_p_stack_polarity.py's automated first-motion window (single max-|abs| sample within
0-45ms post-pick) disagrees with a human reader so often (51.6% overall, 0/5 in cluster0) --
and to find a better window/algorithm.

Recomputes each (cluster, station)'s CC-aligned stack once (identical fetch/align to
cluster_p_stack_polarity.py), saves it to .npz for reuse, then sweeps candidate first-motion
window widths (and a same-sign-neighbor-corroborated variant) against the human labels to see
which choice actually reproduces what a human calls "the first motion" in these plots.

Usage:
    python full_catalog_pipeline/tune_stack_first_motion.py
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

import os
from datetime import date

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
HUMAN_CSV = f"{HYPODD_DIR}/human_review/t1_stack_polarity_human_review.csv"
STACK_CACHE_DIR = f"{HYPODD_DIR}/human_review/stack_cache"
NETWORK = "7U"

WINDOW_PRE = 0.20
WINDOW_POST = 0.30
NOISE_WINDOW = (-0.18, -0.03)
MAX_LAG_S = 0.03
N_CC_ITERS = 3
MIN_SNR = 3.0

CANDIDATE_WINDOWS = [0.045, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
NEIGHBOR_FRAC_CANDIDATES = [None, 0.3, 0.5]  # None = no robustness check


def load_events_for_station(sta, phase_dat, ids, network="7U"):
    tag = f"{network}.{sta}"
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
                    events[cur] = {"origin": origin, "picks": {}}
            elif cur is not None:
                parts = line.split()
                if len(parts) != 4:
                    continue
                s, tt, wt, ph = parts
                if s == tag and ph == "P":
                    events[cur]["picks"][sta] = float(tt)
    return {k: v for k, v in events.items() if sta in v["picks"]}


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
    t = np.arange(len(z)) / sr - WINDOW_PRE
    noise_mask = (t >= NOISE_WINDOW[0]) & (t <= NOISE_WINDOW[1])
    noise_rms = np.sqrt(np.mean(z[noise_mask] ** 2)) if noise_mask.any() else 0.0
    if noise_rms == 0:
        return None
    return z / noise_rms, t


def build_stack(cluster, sta, ids_file, sr):
    ids = set(int(x) for x in open(ids_file))
    events = load_events_for_station(sta, PHASE_DAT, ids, NETWORK)
    day_index = load_day_file_index_csv(config.DAY_FILE_INDEX_CSV)
    cache = RollingDayCache(day_index, sta)

    traces, common_t = [], None
    for eid, ev in sorted(events.items(), key=lambda kv: kv[1]["origin"]):
        pick_time = ev["origin"] + ev["picks"][sta]
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
    stack = mat.mean(axis=0)
    for _ in range(N_CC_ITERS):
        lags = [cc_lag_signed(tr, stack, sr, MAX_LAG_S) for tr in mat]
        mat = np.vstack([shift(tr, -lag) for tr, lag in zip(mat, lags)])
        stack = mat.mean(axis=0)
    return dict(t=common_t, stack=stack, n=len(traces))


def first_motion_windowed(stack, t, sr, window_end_s, neighbor_frac=None):
    noise_mask = (t >= NOISE_WINDOW[0]) & (t <= NOISE_WINDOW[1])
    noise_amp = np.std(stack[noise_mask]) if noise_mask.any() else 0.0
    if noise_amp == 0:
        return 0, np.nan
    pick_idx = int(round(-t[0] * sr))
    end_idx = pick_idx + max(1, int(round(window_end_s * sr)))
    window = stack[pick_idx:end_idx]
    if len(window) == 0:
        return 0, np.nan
    peak_i = int(np.argmax(np.abs(window)))
    peak_val = window[peak_i]
    peak_idx = pick_idx + peak_i
    snr = abs(peak_val) / noise_amp
    if snr < MIN_SNR:
        return 0, snr
    if neighbor_frac is not None:
        neighbors = [i for i in (peak_idx - 1, peak_idx + 1) if 0 <= i < len(stack)]
        robust = any(np.sign(stack[i]) == np.sign(peak_val) and abs(stack[i]) >= neighbor_frac * abs(peak_val)
                     for i in neighbors)
        if not robust:
            return 0, snr
    return (1 if peak_val > 0 else -1), snr


def main():
    os.makedirs(STACK_CACHE_DIR, exist_ok=True)
    human = pd.read_csv(HUMAN_CSV)
    human["human_short"] = human["human_polarity"].map({"up": 1, "down": -1, "indeterminate": 0})
    sr = config.SAMPLE_RATE_HZ

    stacks = {}
    for _, row in human.iterrows():
        cluster, sta = row["cluster"], row["station"]
        cache_path = f"{STACK_CACHE_DIR}/{cluster}_{sta}.npz"
        if os.path.exists(cache_path):
            d = np.load(cache_path)
            stacks[(cluster, sta)] = dict(t=d["t"], stack=d["stack"], n=int(d["n"]))
            continue
        ids_file = f"{HYPODD_DIR}/{cluster}_event_ids.txt"
        result = build_stack(cluster, sta, ids_file, sr)
        if result is None:
            print(f"  {cluster}/{sta}: too few traces, skipped")
            continue
        stacks[(cluster, sta)] = result
        np.savez(cache_path, t=result["t"], stack=result["stack"], n=result["n"])
        print(f"  {cluster}/{sta}: n={result['n']}, cached -> {cache_path}")

    print(f"\n{len(stacks)}/{len(human)} stacks available\n")

    definite = human[human["human_short"] != 0]
    print(f"Human definite calls: {len(definite)}/{len(human)}\n")

    print(f"{'window_s':>9} {'neighbor':>9} {'accuracy':>10} {'n_scored':>9} {'n_indet':>8}")
    best = None
    for w in CANDIDATE_WINDOWS:
        for nf in NEIGHBOR_FRAC_CANDIDATES:
            correct, scored, indet = 0, 0, 0
            for _, row in definite.iterrows():
                key = (row["cluster"], row["station"])
                if key not in stacks:
                    continue
                s = stacks[key]
                pol, snr = first_motion_windowed(s["stack"], s["t"], sr, w, nf)
                if pol == 0:
                    indet += 1
                    continue
                scored += 1
                if pol == row["human_short"]:
                    correct += 1
            acc = correct / scored if scored else float("nan")
            print(f"{w:>9.3f} {str(nf):>9} {acc:>10.3f} {scored:>9} {indet:>8}")
            if best is None or (scored >= 20 and acc > best[2]):
                best = (w, nf, acc, scored, indet)

    print(f"\nbest (requiring >=20 scored): window={best[0]}, neighbor_frac={best[1]}, "
          f"accuracy={best[2]:.3f}, scored={best[3]}, indet={best[4]}")

    print("\nper-cluster breakdown at best setting:")
    w, nf = best[0], best[1]
    for cluster, g in definite.groupby("cluster"):
        correct, scored = 0, 0
        for _, row in g.iterrows():
            key = (row["cluster"], row["station"])
            if key not in stacks:
                continue
            s = stacks[key]
            pol, snr = first_motion_windowed(s["stack"], s["t"], sr, w, nf)
            if pol == 0:
                continue
            scored += 1
            if pol == row["human_short"]:
                correct += 1
        print(f"  {cluster}: {correct}/{scored}")


if __name__ == "__main__":
    main()
