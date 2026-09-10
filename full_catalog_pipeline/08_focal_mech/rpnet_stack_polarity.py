#!/usr/bin/env python3
"""Per user request: run RPNet (the deep-learning P-polarity classifier already used
per-event in rpnet_cluster3_polarities.py) on the STACKED (composite/aggregate) waveform
for each cluster+station, instead of on individual noisy events -- then compare that
reading to what we already have (the amplitude-threshold stack polarity from
cluster_p_stack_polarity.py) before feeding it into the composite SKHASH fit
(skhash_composite_fit.py).

Method: builds ONE composite Z-trace per station by the same proven CC-alignment as
cluster_p_stack_polarity.py (short onset window, never allows a sign flip, iterative
running-stack alignment -- see [[cluster_p_stack_polarity_result]]), but applies the
resulting per-event lag to a much WIDER raw (unfiltered) fetch window (+/-3.5s, matching
rpnet_cluster3_polarities.py's own HALF_WIN_PAD) so there's enough context for RPNet's own
preprocessing. The averaged wide raw traces become ONE composite trace per station, which
is then run through RPNet's EXACT preprocessing pipeline (interpolate to 100Hz, highpass
1Hz, trim to +/-2.5s, normalize, first 500 samples -- same as rpnet_window() in
rpnet_cluster3_polarities.py) and RPNet's own pred_rpnet(), so the composite waveform is
treated identically to how RPNet sees a single real event -- just built from the whole
cluster's stacked signal instead of one noisy trace.

Saves the composite waveform itself (both the short alignment-window version and the wide
RPNet-input version, plus every individual event's own aligned wide trace) to an .npz per
station, since nothing upstream currently persists these raw stacked waveforms.

Must run in the `rpnet` conda env:
    conda activate rpnet
    python full_catalog_pipeline/rpnet_stack_polarity.py --label cluster3
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

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import Stream, Trace, UTCDateTime

import config
from lib.day_volume_index import load_day_file_index_csv
from lib.deej_waveform_common import clean, cc_lag_signed, shift
from lib.windowing import RollingDayCache, extract_window

from rpnet.predict import pred_rpnet

import catalog_paths

HYPODD_DIR = catalog_paths.work_dir("T1")
PHASE_DAT = f"{HYPODD_DIR}/input_files/phase.dat"
MODEL_PATH = "full_catalog_pipeline/rpnet_model/RPNet_v1.h5"
ALL_STATIONS = ["DEEJ", "ELZA", "LILA", "TJTJ", "OTIS", "LOUS", "SQIG"]
NETWORK = "7U"

SR = 200.0
# alignment (short window): identical convention to cluster_p_stack_polarity.py
ALIGN_WINDOW_PRE = 0.20
ALIGN_WINDOW_POST = 0.30
NOISE_WINDOW = (-0.18, -0.03)
MAX_LAG_S = 0.03
N_CC_ITERS = 3
# wide fetch for RPNet input: identical convention to rpnet_cluster3_polarities.py
FETCH_HALF_WINDOW = 3.5   # HALF_WIN_PAD
TARGET_SR = 100.0
HALF_WIN_FINAL = 2.5
N_SAMPLES = 500
MID_POINT = 250
STD_THRESHOLD = 0.2
ITERATIONS = 100


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="cluster3")
    ap.add_argument("--ids-file", default=None, help=f"Defaults to {HYPODD_DIR}/<label>_event_ids.txt")
    ap.add_argument("--min-picks", type=int, default=10,
                    help="Minimum P picks within this cluster for a station to be included.")
    ap.add_argument("--out-dir", default=None, help=f"Defaults to {HYPODD_DIR}/<label>_stack_rpnet")
    ap.add_argument("--model-path", default=MODEL_PATH)
    ap.add_argument("--std-threshold", type=float, default=STD_THRESHOLD)
    ap.add_argument("--iterations", type=int, default=ITERATIONS)
    args = ap.parse_args()
    if args.ids_file is None:
        args.ids_file = f"{HYPODD_DIR}/{args.label}_event_ids.txt"
    if args.out_dir is None:
        args.out_dir = f"{HYPODD_DIR}/{args.label}_stack_rpnet"
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


def fetch_wide_raw(cache, pick_time, sr):
    """Raw (unfiltered, un-normalized) Z trace over +/-FETCH_HALF_WINDOW around the pick."""
    d = date(pick_time.year, pick_time.month, pick_time.day)
    row = {"primary_date": d.isoformat()}
    window_start = pick_time - FETCH_HALF_WINDOW
    window_end = pick_time + FETCH_HALF_WINDOW
    data, status = extract_window(cache, row, window_start, window_end)
    cache.evict_before(row["primary_date"])
    if data is None or not data["complete"]:
        return None
    return data["Z"].astype(np.float64)


def short_align_trace(wide_raw, sr):
    """Extracts the proven short onset sub-window from the wide raw fetch, cleaned and
    noise-normalized -- used ONLY to solve the per-event alignment lag."""
    center_idx = int(round(FETCH_HALF_WINDOW * sr))
    i0 = center_idx - int(round(ALIGN_WINDOW_PRE * sr))
    i1 = center_idx + int(round(ALIGN_WINDOW_POST * sr))
    seg = clean(wide_raw[i0:i1])
    t = np.arange(len(seg)) / sr - ALIGN_WINDOW_PRE
    noise_mask = (t >= NOISE_WINDOW[0]) & (t <= NOISE_WINDOW[1])
    noise_rms = np.sqrt(np.mean(seg[noise_mask] ** 2)) if noise_mask.any() else 0.0
    if noise_rms == 0:
        return None
    return seg / noise_rms


def build_composite_trace(station, events, sr):
    day_index = load_day_file_index_csv(config.DAY_FILE_INDEX_CSV)
    cache = RollingDayCache(day_index, station)

    eids, wide_traces, short_traces = [], [], []
    for eid, ev in sorted(events.items(), key=lambda kv: kv[1]["origin"]):
        if station not in ev["picks"]:
            continue
        pick_time = ev["origin"] + ev["picks"][station]
        wide = fetch_wide_raw(cache, pick_time, sr)
        if wide is None:
            continue
        short = short_align_trace(wide, sr)
        if short is None:
            continue
        if wide_traces and len(wide) != len(wide_traces[0]):
            continue
        eids.append(eid)
        wide_traces.append(wide)
        short_traces.append(short)

    if len(wide_traces) < 3:
        return None

    wide_mat = np.vstack(wide_traces)
    short_mat = np.vstack(short_traces)
    short_stack = short_mat.mean(axis=0)
    for _ in range(N_CC_ITERS):
        lags = [cc_lag_signed(tr, short_stack, sr, MAX_LAG_S) for tr in short_mat]
        short_mat = np.vstack([shift(tr, -lag) for tr, lag in zip(short_mat, lags)])
        short_stack = short_mat.mean(axis=0)
        wide_mat = np.vstack([shift(tr, -lag) for tr, lag in zip(wide_mat, lags)])

    composite_wide = wide_mat.mean(axis=0)
    return dict(station=station, eids=eids, wide_mat=wide_mat, composite_wide=composite_wide,
                short_stack=short_stack, sr=sr)


def rpnet_input_from_composite(composite_wide, sr):
    """Runs the composite (aggregate) raw trace through RPNet's own exact preprocessing
    pipeline (rpnet_cluster3_polarities.py's rpnet_window()), by wrapping it in an obspy
    Trace with a synthetic starttime placing the alignment reference at the trace's own
    center -- i.e. treating the composite exactly like one real event's window."""
    synthetic_start = UTCDateTime(0)
    pick_time = synthetic_start + FETCH_HALF_WINDOW
    tr = Trace(data=composite_wide, header={"sampling_rate": sr, "starttime": synthetic_start})
    st = Stream([tr])
    if tr.stats.sampling_rate != TARGET_SR:
        st.interpolate(TARGET_SR)
    st.filter("highpass", freq=1.0)
    st.trim(pick_time - HALF_WIN_FINAL, pick_time + HALF_WIN_FINAL)
    if len(st) == 0:
        return None
    st.normalize()
    arr = st[0].data
    if len(arr) < N_SAMPLES:
        return None
    return arr[:N_SAMPLES]


def plot_station(result, rpnet_row, amp_row, out_path, label):
    sr = result["sr"]
    t_wide = np.arange(len(result["composite_wide"])) / sr - FETCH_HALF_WINDOW
    t_short = np.arange(len(result["short_stack"])) / sr - ALIGN_WINDOW_PRE

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(t_wide, result["composite_wide"], color="black", lw=1)
    ax.axvline(0, color="steelblue", ls="--", lw=1)
    ax.set_xlabel("seconds relative to catalog P pick")
    ax.set_ylabel("composite raw amplitude (stacked, unfiltered)")
    ax.set_title(f"{label} / {result['station']}: composite wide trace (n={len(result['eids'])})\n"
                 f"fed to RPNet exactly as a single event's raw window", fontsize=9)

    ax = axes[1]
    ax.plot(t_short, result["short_stack"], color="black", lw=1.5)
    ax.axvline(0, color="steelblue", ls="--", lw=1)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlim(-0.05, 0.15)
    ax.set_xlabel("seconds relative to catalog P pick")
    ax.set_ylabel("noise-RMS-normalized amplitude")
    rp = f"RPNet-on-stack: {rpnet_row['predict']} (prob={rpnet_row['prob']:.3f}, std={rpnet_row['std']:.3f})"
    ap = f"amplitude-stack: {amp_row}"
    ax.set_title(f"onset region\n{rp}\n{ap}", fontsize=9)
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

    # existing amplitude-threshold stack polarity, for comparison
    amp_csv = f"{HYPODD_DIR}/{args.label}_stack_polarity/{args.label}_stack_polarities.csv"
    amp_pol = pd.read_csv(amp_csv).set_index("station") if os.path.exists(amp_csv) else None

    sr = SR
    in_mats, meta_rows, results_by_sta = [], [], {}
    for sta in stations:
        result = build_composite_trace(sta, all_events, sr)
        if result is None:
            print(f"  {sta}: too few usable traces, skipped")
            continue
        arr = rpnet_input_from_composite(result["composite_wide"], sr)
        if arr is None:
            print(f"  {sta}: RPNet preprocessing failed (short trace after trim), skipped")
            continue
        in_mats.append(arr[np.newaxis, :])
        meta_rows.append({"id": f"composite_{args.label}", "station": sta})
        results_by_sta[sta] = result
        npz_path = f"{args.out_dir}/{args.label}_{sta.lower()}_composite_waveform.npz"
        np.savez(npz_path, eids=np.array(result["eids"]), wide_mat=result["wide_mat"],
                 composite_wide=result["composite_wide"], short_stack=result["short_stack"], sr=sr)

    if not in_mats:
        print(f"[{args.label}] no usable composite waveforms, aborting")
        return

    in_mat = np.vstack(in_mats)
    meta_df = pd.DataFrame(meta_rows)
    r_df = pred_rpnet(args.model_path, in_mat, meta_df, batch_size=2 ** 13,
                       iteration=args.iterations, gpu_num=-1, time_shift=0.0, mid_point=MID_POINT)
    r_df["polarity_raw"] = r_df["predict"].map({"U": 1, "D": -1, "K": 0})
    r_df.loc[r_df["std"] > args.std_threshold, "predict"] = "K"
    r_df["polarity"] = r_df["predict"].map({"U": 1, "D": -1, "K": 0})

    print(f"\n[{args.label}] RPNet-on-stack vs amplitude-stack comparison:")
    rows = []
    for _, row in r_df.iterrows():
        sta = row["station"]
        amp_str = "n/a"
        if amp_pol is not None and sta in amp_pol.index:
            a = amp_pol.loc[sta]
            amp_pol_label = {1: "up", -1: "down", 0: "indet"}[int(a["polarity"])]
            amp_str = f"{amp_pol_label} (SNR={a['snr']:.1f})"
        pol_label = {1: "up", -1: "down", 0: "indet"}[row["polarity"]]
        agree = (amp_pol is not None and sta in amp_pol.index and
                 row["polarity"] != 0 and int(amp_pol.loc[sta]["polarity"]) == row["polarity"])
        print(f"  {sta}: RPNet-on-stack={pol_label} (prob={row['prob']:.3f}, std={row['std']:.3f}) "
              f"| amplitude-stack={amp_str} | agree={agree}")
        rows.append(dict(station=sta, polarity=row["polarity"], prob=row["prob"], std=row["std"],
                          amplitude_stack_agree=agree))
        if sta in results_by_sta:
            out_png = f"{args.out_dir}/{args.label}_{sta.lower()}_stack_rpnet.png"
            plot_station(results_by_sta[sta], row, amp_str, out_png, args.label)

    out_csv = f"{args.out_dir}/{args.label}_stack_rpnet_polarities.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
