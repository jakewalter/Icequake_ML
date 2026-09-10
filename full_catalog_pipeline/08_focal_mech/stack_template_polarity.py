#!/usr/bin/env python3
"""Per user request: derive PER-EVENT P-polarity picks by reading each event's own amplitude
at the already-established first-motion sample of its station's population-level stack --
rather than trusting a single noisy event in isolation the way RPNet
(rpnet_cluster3_polarities.py) or the older heuristic picker do.

[[cluster_p_stack_polarity_result]] already proved the stack itself is a clean, high-SNR
(up to 300+), unambiguous first-motion template per station, built via cluster_p_stack_polarity.py's
iterative CC alignment (never allows a sign flip) -- reused here verbatim (same fetch window,
same alignment, same first_motion() peak-finding convention) so this script's per-station
polarity/SNR numbers reproduce that already-validated result exactly. That script only ever
reports ONE polarity per station -- it never asks how well any individual event agrees with
the population answer. That per-event agreement is exactly the reliable polarity signal
[[skhash_cluster_fit_integration]] needs: SKHASH fits ONE mechanism per event from that
event's own per-station polarities, so a noisy per-event picker directly degrades the fit,
independent of the station-count limit already established as the main bottleneck.

Method: once the stack settles, locate its dominant first-motion sample the same way
first_motion() does (single max-abs sample within a short post-pick lookahead). Then, for
each individual event, read that SAME event's own (CC-aligned) amplitude at that exact
sample (a tiny +/- local window around it, to absorb residual sub-sample jitter) --
polarity = sign of that reading if its magnitude clears an SNR-like threshold (each trace is
already noise-RMS-normalized at fetch time, so "1.0" = 1x that event's own noise floor),
else indeterminate. This is the same reading a human/automated picker would take off a
single trace, but pointed at the exact time index the population already validated as
carrying real signal -- removing single-event pick-timing uncertainty as a source of noise.

Output feeds directly into skhash_cluster_fit.py via --polarities-csv (same id/station/
polarity schema as *_polarities_rpnet.csv).

Usage:
    python full_catalog_pipeline/stack_template_polarity.py --label cluster3
    python full_catalog_pipeline/skhash_cluster_fit.py --label cluster3 \\
        --polarities-csv full_catalog_pipeline/artifacts/full_run/T1_v5/hypodd/cluster3_stack_template_polarity/cluster3_stack_template_polarities.csv \\
        --out-dir full_catalog_pipeline/artifacts/full_run/T1_v5/hypodd/cluster3_skhash_stacktemplate
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
WINDOW_POST = 0.30    # seconds after the P pick fetched -- matches cluster_p_stack_polarity.py
NOISE_WINDOW = (-0.18, -0.03)   # relative to P pick, for per-trace amplitude normalization
MAX_LAG_S = 0.03       # +/- CC search window for residual timing jitter
N_CC_ITERS = 3         # iterative re-alignment-to-running-stack passes -- identical scheme to
                       # cluster_p_stack_polarity.py: lag solved over the WHOLE fetched trace,
                       # not a sub-window, so this script's stack/SNR/first-motion numbers
                       # reproduce that already-validated per-station result exactly.
SIGNAL_LOOKAHEAD = 0.05  # seconds after the (CC-refined) pick to look for the stack's first swing --
                          # matches cluster_p_stack_polarity.py's first_motion() exactly.
PEAK_LOCAL_HALFWIDTH_S = 0.01  # per-event reading is the max-|amplitude| sample within +/- this
                                # many seconds of the stack's own identified peak sample (absorbs
                                # a sample or two of residual per-event timing jitter without
                                # searching so far that a later, unrelated swing gets picked up).
MIN_TEMPLATE_SNR = 3.0     # skip a station entirely if even its stack's first motion is indeterminate
                           # (matches cluster_p_stack_polarity.py's own MIN_SNR gate).
MIN_EVENT_SNR = 1.5        # per-event reading at the peak must be at least this many multiples of
                            # that event's own noise floor to be called reliable (else indeterminate,
                            # polarity=0) -- individual traces are noisier than the stack, so this is
                            # deliberately looser than the station-level MIN_TEMPLATE_SNR gate.


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="cluster3")
    ap.add_argument("--ids-file", default=None, help=f"Defaults to {HYPODD_DIR}/<label>_event_ids.txt")
    ap.add_argument("--min-picks", type=int, default=10,
                    help="Minimum P picks within this cluster for a station to be included.")
    ap.add_argument("--out-dir", default=None, help=f"Defaults to {HYPODD_DIR}/<label>_stack_template_polarity")
    ap.add_argument("--min-template-snr", type=float, default=MIN_TEMPLATE_SNR,
                    help="Skip a station's per-event scoring if the station's own stack SNR is below this.")
    ap.add_argument("--min-event-snr", type=float, default=MIN_EVENT_SNR,
                    help="Minimum per-event |amplitude| (in units of that event's own noise floor) at the "
                         "peak sample to call a reliable polarity, else indeterminate.")
    args = ap.parse_args()
    if args.ids_file is None:
        args.ids_file = f"{HYPODD_DIR}/{args.label}_event_ids.txt"
    if args.out_dir is None:
        args.out_dir = f"{HYPODD_DIR}/{args.label}_stack_template_polarity"
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
    t = np.arange(len(z)) / sr - WINDOW_PRE
    noise_mask = (t >= NOISE_WINDOW[0]) & (t <= NOISE_WINDOW[1])
    noise_rms = np.sqrt(np.mean(z[noise_mask] ** 2)) if noise_mask.any() else 0.0
    if noise_rms == 0:
        return None
    return z / noise_rms, t  # noise-RMS-normalized (SNR-equal-weighted), sign preserved


def first_motion(stack, t, sr):
    """Identical convention to cluster_p_stack_polarity.py's first_motion(): single max-|abs|
    sample within [pick, pick+SIGNAL_LOOKAHEAD]."""
    noise_mask = (t >= NOISE_WINDOW[0]) & (t <= NOISE_WINDOW[1])
    noise_amp = np.std(stack[noise_mask]) if noise_mask.any() else 0.0
    if noise_amp == 0:
        return 0, np.nan, None
    pick_idx = int(round(-t[0] * sr))
    sig_i1 = pick_idx + max(1, int(round(SIGNAL_LOOKAHEAD * sr)))
    window = stack[pick_idx:sig_i1]
    if len(window) == 0:
        return 0, np.nan, None
    peak_i = np.argmax(np.abs(window))
    peak_val = window[peak_i]
    snr = abs(peak_val) / noise_amp
    peak_idx = pick_idx + peak_i
    if snr < 3.0:  # matches cluster_p_stack_polarity.py's MIN_SNR
        return 0, snr, peak_idx
    return (1 if peak_val > 0 else -1), snr, peak_idx


def build_stack(station, events, sr):
    """Reproduces cluster_p_stack_polarity.py's fetch+iterative-CC-align loop exactly, but
    also returns the event id parallel to each row of the final aligned matrix, so per-event
    readings can be written back out against a real event id."""
    day_index = load_day_file_index_csv(config.DAY_FILE_INDEX_CSV)
    cache = RollingDayCache(day_index, station)

    eids, traces, common_t = [], [], None
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
        eids.append(eid)
        traces.append(z)

    if len(traces) < 3:
        return None

    mat = np.vstack(traces)
    stack = mat.mean(axis=0)  # pass 0: catalog-pick-aligned, no CC yet
    for _ in range(N_CC_ITERS):
        lags = [cc_lag_signed(tr, stack, sr, MAX_LAG_S) for tr in mat]
        mat = np.vstack([shift(tr, -lag) for tr, lag in zip(mat, lags)])
        stack = mat.mean(axis=0)

    polarity, snr, peak_idx = first_motion(stack, common_t, sr)
    return dict(station=station, eids=eids, t=common_t, mat=mat, stack=stack,
                polarity=polarity, snr=snr, peak_idx=peak_idx)


def score_events(result, sr, min_event_snr, local_halfwidth_s=PEAK_LOCAL_HALFWIDTH_S):
    """For each event, read its own amplitude in a small window around the stack's already-
    established peak sample (max |abs| within that local window -- absorbs a sample or two of
    residual jitter), and classify polarity by that reading's sign, gated on its own SNR (each
    trace is noise-RMS-normalized at fetch, so amplitude is directly in noise-floor units)."""
    peak_idx = result["peak_idx"]
    halfwidth = max(1, int(round(local_halfwidth_s * sr)))
    lo, hi = max(0, peak_idx - halfwidth), min(len(result["stack"]), peak_idx + halfwidth + 1)
    rows = []
    for eid, tr in zip(result["eids"], result["mat"]):
        window = tr[lo:hi]
        i = int(np.argmax(np.abs(window)))
        val = window[i]
        polarity = (1 if val > 0 else -1) if abs(val) >= min_event_snr else 0
        rows.append(dict(id=eid, station=result["station"], polarity=polarity, amplitude=val))
    df = pd.DataFrame(rows)
    meta = dict(peak_idx=peak_idx, peak_t=result["t"][peak_idx], window_lo_t=result["t"][lo],
                window_hi_t=result["t"][hi - 1])
    return df, meta


def plot_station(result, df, meta, out_path, label):
    t, mat, stack = result["t"], result["mat"], result["stack"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    for tr in mat:
        ax.plot(t, tr, color="0.75", lw=0.4, alpha=0.5)
    ax.plot(t, stack, color="black", lw=1.8, label=f"template stack (n={len(mat)})")
    ax.axvspan(meta["window_lo_t"], meta["window_hi_t"], color="firebrick", alpha=0.25,
               label="per-event reading window")
    ax.axvline(0, color="steelblue", ls="--", lw=1)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlim(-0.05, 0.15)
    ax.set_xlabel("seconds relative to catalog P pick")
    ax.set_ylabel("noise-RMS-normalized amplitude")
    ax.legend(loc="upper right", fontsize=8)
    pol_label = {1: "up", -1: "down", 0: "indeterminate"}[result["polarity"]]
    ax.set_title(f"{label} / {result['station']}: stack polarity={pol_label}, SNR={result['snr']:.1f}", fontsize=10)

    ax = axes[1]
    ax.hist(df["amplitude"], bins=40, color="0.4")
    ax.axvline(0, color="0.6", lw=0.8)
    n_pos = (df["polarity"] == 1).sum()
    n_neg = (df["polarity"] == -1).sum()
    n_indet = (df["polarity"] == 0).sum()
    ax.set_title(f"per-event reading at peak: up={n_pos} down={n_neg} indet={n_indet}", fontsize=10)
    ax.set_xlabel("per-event amplitude at peak (noise-floor units)")
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
    all_rows = []
    summary = []
    for sta in stations:
        result = build_stack(sta, all_events, sr)
        if result is None:
            print(f"  {sta}: too few usable traces, skipped")
            continue
        if not np.isfinite(result["snr"]) or result["snr"] < args.min_template_snr or result["polarity"] == 0:
            print(f"  {sta}: template SNR={result['snr']:.1f} < {args.min_template_snr} (or indeterminate), skipped")
            continue
        df, meta = score_events(result, sr, args.min_event_snr)
        all_rows.append(df)
        n_pos = (df["polarity"] == 1).sum()
        n_neg = (df["polarity"] == -1).sum()
        n_indet = (df["polarity"] == 0).sum()
        pol_label = {1: "up", -1: "down"}[result["polarity"]]
        print(f"  {sta}: n={len(df)}, stack polarity={pol_label} (SNR={result['snr']:.1f}), peak t={meta['peak_t']:.3f}, "
              f"per-event up={n_pos} down={n_neg} indet={n_indet} "
              f"(reliable frac={1 - n_indet / len(df):.2f}, agree-with-stack frac="
              f"{( (df['polarity'] == result['polarity']).sum() / (df['polarity'] != 0).sum() if (df['polarity'] != 0).sum() else float('nan')):.2f})")
        out_png = f"{args.out_dir}/{args.label}_{sta.lower()}_stack_template_polarity.png"
        plot_station(result, df, meta, out_png, args.label)
        summary.append(dict(station=sta, n=len(df), stack_polarity=result["polarity"], stack_snr=result["snr"],
                             peak_t=meta["peak_t"], n_up=n_pos, n_down=n_neg, n_indet=n_indet))

    if not all_rows:
        print(f"[{args.label}] no stations produced usable per-event scores.")
        return

    out_df = pd.concat(all_rows, ignore_index=True)
    out_csv = f"{args.out_dir}/{args.label}_stack_template_polarities.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv} ({len(out_df)} rows)")

    picks_per_event = out_df[out_df["polarity"] != 0].groupby("id").size()
    print(f"[{args.label}] reliable picks/event: median={picks_per_event.median():.0f}, "
          f"max={picks_per_event.max()}, events with >=5 picks: {(picks_per_event >= 5).sum()}, "
          f"events with >=4 picks: {(picks_per_event >= 4).sum()}")

    summary_csv = f"{args.out_dir}/{args.label}_stack_template_polarity_summary.csv"
    pd.DataFrame(summary).to_csv(summary_csv, index=False)
    print(f"wrote {summary_csv}")


if __name__ == "__main__":
    main()
