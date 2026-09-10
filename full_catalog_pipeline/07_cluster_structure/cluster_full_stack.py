#!/usr/bin/env python3
"""Per user suggestion: generalizes [[cluster_p_stack_polarity_result]]'s proven CC-aligned,
sign-preserved stacking -- proven there for a short P/Z window -- to a full P+S+coda window
across all three components (Z, N, E), producing a genuine stacked AGGREGATE WAVEFORM per
cluster and per station. Every downstream S-wave/firn diagnostic in this pipeline
(plot_cluster_deej_stack_check.py, plot_cluster_all_stations_stack_check.py,
plot_cluster_sp_cascade.py, plot_cluster_deej_cc_alignment.py) stacks a rectified
horizontal_envelope(N, E) instead -- always positive, so timing jitter blurs peaks but can't
literally cancel them, at the cost of throwing away sign/phase entirely. This script keeps
the sign, so alignment must do the real work of preventing destructive interference.

Alignment is ONE lag per event, not one per channel: Z/N/E share a single digitizer clock, so
solving independent per-channel lags would be physically wrong. The lag is solved on the Z
channel only (cleanest, sharpest onset) via the same iterative running-stack signed-only
cross-correlation as cluster_p_stack_polarity.py (never allows a sign flip -- see that
script's docstring for why that matters), then applied identically to that event's N and E.

Known, expected caveat (not a bug): [[t1_composite_focal_mech_result]] and
[[skhash_cluster_fit_integration]] already established that cluster3's events do not share
one focal mechanism. A perfectly time-aligned horizontal (N/E) stack can still wash out if
different events' radiation patterns point particle motion in different directions at the
same station -- a rotation/polarization mismatch that timing-only CC alignment cannot fix.
Either outcome (coherent vs. washed out) is itself informative here, not a failure of this
tool.

Usage:
    python full_catalog_pipeline/cluster_full_stack.py --label cluster3
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
COMPONENTS = ["Z", "N", "E"]

WINDOW_PRE = 0.20       # seconds before the P pick fetched (also serves as the noise window)
SP_CODA_BUFFER = 0.4    # seconds of coda kept past the cluster's own max observed S-P
MIN_WINDOW_POST = 1.40  # floor, in case a cluster has no S picks to size the window from
NOISE_WINDOW = (-0.18, -0.03)  # relative to P pick, for per-trace/per-channel normalization
MAX_LAG_S = 0.03        # +/- CC search window for residual timing jitter (solved on Z only)
N_CC_ITERS = 3          # iterative re-alignment-to-running-Z-stack passes


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="cluster3")
    ap.add_argument("--ids-file", default=None, help=f"Defaults to {HYPODD_DIR}/<label>_event_ids.txt")
    ap.add_argument("--min-picks", type=int, default=10,
                    help="Minimum P picks within this cluster for a station to be included.")
    ap.add_argument("--out-dir", default=None, help=f"Defaults to {HYPODD_DIR}/<label>_full_stack")
    ap.add_argument("--window-pre", type=float, default=WINDOW_PRE)
    ap.add_argument("--window-post", type=float, default=None,
                    help="Seconds after the P pick to fetch. Default: auto-sized per cluster "
                         f"from its own max observed S-P interval + {SP_CODA_BUFFER}s coda "
                         f"(floor {MIN_WINDOW_POST}s) -- S-P varies a lot station-to-station "
                         "within one cluster, so a fixed default risks silently truncating S "
                         "at the more distant stations.")
    args = ap.parse_args()
    if args.ids_file is None:
        args.ids_file = f"{HYPODD_DIR}/{args.label}_event_ids.txt"
    if args.out_dir is None:
        args.out_dir = f"{HYPODD_DIR}/{args.label}_full_stack"
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


def max_observed_sp(ids, phase_dat, stations, network="7U"):
    """Max observed S-P interval (seconds) across all --ids events and `stations`, from the
    same phase.dat -- used to size the fetch window so the window default isn't silently too
    short to contain the S arrival at more distant stations (S-P scales with distance, and
    varies a lot within a single cluster: e.g. cluster3's DEEJ medians 0.77s but OTIS's max is
    2.26s)."""
    sta_tags = {f"{network}.{s}": s for s in stations}
    picks = {}
    cur = None
    with open(phase_dat) as f:
        for line in f:
            if line.startswith("#"):
                p = line.split()
                eid = int(p[-1])
                cur = eid if eid in ids else None
                if cur is not None:
                    picks[cur] = {}
            elif cur is not None:
                parts = line.split()
                if len(parts) != 4:
                    continue
                sta, tt, wt, ph = parts
                if sta in sta_tags and ph in ("P", "S"):
                    picks[cur].setdefault(sta_tags[sta], {})[ph] = float(tt)
    sp = [d["S"] - d["P"] for ev in picks.values() for d in ev.values() if "P" in d and "S" in d]
    return max(sp) if sp else 0.0


def fetch_traces(cache, pick_time, sr, window_pre, window_post):
    """Returns {"Z": arr, "N": arr, "E": arr}, each cleaned and independently
    noise-RMS-normalized (sign preserved), plus the shared relative time axis -- or None if
    the window isn't fully available."""
    d = date(pick_time.year, pick_time.month, pick_time.day)
    row = {"primary_date": d.isoformat()}
    window_start = pick_time - window_pre
    window_end = pick_time + window_post
    data, status = extract_window(cache, row, window_start, window_end)
    cache.evict_before(row["primary_date"])
    if data is None or not data["complete"]:
        return None
    t = np.arange(len(data["Z"])) / sr - window_pre
    noise_mask = (t >= NOISE_WINDOW[0]) & (t <= NOISE_WINDOW[1])
    if not noise_mask.any():
        return None
    out = {}
    for comp, key in (("Z", "Z"), ("N", "N"), ("E", "E")):
        x = clean(data[key])
        noise_rms = np.sqrt(np.mean(x[noise_mask] ** 2))
        if noise_rms == 0:
            return None
        out[comp] = x / noise_rms
    return out, t


def peak_snr(stack, t, noise_window=NOISE_WINDOW):
    noise_mask = (t >= noise_window[0]) & (t <= noise_window[1])
    noise_amp = np.std(stack[noise_mask]) if noise_mask.any() else 0.0
    signal_mask = t >= 0
    if noise_amp == 0 or not signal_mask.any():
        return np.nan
    return np.max(np.abs(stack[signal_mask])) / noise_amp


def stack_station(station, events, sr, window_pre, window_post):
    day_index = load_day_file_index_csv(config.DAY_FILE_INDEX_CSV)
    cache = RollingDayCache(day_index, station)

    per_comp_traces = {c: [] for c in COMPONENTS}
    common_t = None
    for eid, ev in sorted(events.items(), key=lambda kv: kv[1]["origin"]):
        if station not in ev["picks"]:
            continue
        pick_time = ev["origin"] + ev["picks"][station]
        result = fetch_traces(cache, pick_time, sr, window_pre, window_post)
        if result is None:
            continue
        traces, t = result
        if common_t is None:
            common_t = t
        elif len(traces["Z"]) != len(common_t):
            continue
        for c in COMPONENTS:
            per_comp_traces[c].append(traces[c])

    n = len(per_comp_traces["Z"])
    if n < 3:
        return None

    mat = {c: np.vstack(per_comp_traces[c]) for c in COMPONENTS}
    # pass 0: catalog-P-pick-aligned stack, no CC yet
    stack_z = mat["Z"].mean(axis=0)
    for _ in range(N_CC_ITERS):
        lags = [cc_lag_signed(tr, stack_z, sr, MAX_LAG_S) for tr in mat["Z"]]
        # one lag per event, applied identically across Z/N/E (shared digitizer clock)
        mat = {c: np.vstack([shift(tr, -lag) for tr, lag in zip(mat[c], lags)]) for c in COMPONENTS}
        stack_z = mat["Z"].mean(axis=0)

    stack = {c: mat[c].mean(axis=0) for c in COMPONENTS}
    snr = {c: peak_snr(stack[c], common_t) for c in COMPONENTS}
    return dict(station=station, n=n, t=common_t, mat=mat, stack=stack, snr=snr)


def plot_station(result, out_path, label):
    t = result["t"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for ax, comp in zip(axes, COMPONENTS):
        for tr in result["mat"][comp]:
            ax.plot(t, tr, color="0.75", lw=0.4, alpha=0.5)
        ax.plot(t, result["stack"][comp], color="black", lw=1.5,
                label=f"stack (n={result['n']})")
        ax.axvline(0, color="steelblue", ls="--", lw=1)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_ylabel(f"{comp}\n(noise-RMS-norm.)")
        ax.set_title(f"{comp}: SNR={result['snr'][comp]:.1f}", fontsize=9, loc="right")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"{label} / {result['station']}: n={result['n']} events, "
                       f"CC-aligned raw (no envelope) stack", fontsize=11, loc="left")
    axes[-1].set_xlabel("seconds relative to catalog P pick")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_npz(result, out_path):
    np.savez(out_path, t=result["t"], n=result["n"],
             stack_z=result["stack"]["Z"], stack_n=result["stack"]["N"], stack_e=result["stack"]["E"],
             mat_z=result["mat"]["Z"], mat_n=result["mat"]["N"], mat_e=result["mat"]["E"])


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

    if args.window_post is None:
        max_sp = max_observed_sp(ids, PHASE_DAT, stations, network=NETWORK)
        args.window_post = max(MIN_WINDOW_POST, max_sp + SP_CODA_BUFFER)
        print(f"[{args.label}] max observed S-P across used stations: {max_sp:.3f}s "
              f"-> window_post={args.window_post:.3f}s")

    sr = config.SAMPLE_RATE_HZ
    rows = []
    for sta in stations:
        result = stack_station(sta, all_events, sr, args.window_pre, args.window_post)
        if result is None:
            print(f"  {sta}: too few usable traces, skipped")
            continue
        snr = result["snr"]
        print(f"  {sta}: n={result['n']}, SNR Z={snr['Z']:.1f} N={snr['N']:.1f} E={snr['E']:.1f}")
        out_png = f"{args.out_dir}/{args.label}_{sta.lower()}_full_stack.png"
        out_npz = f"{args.out_dir}/{args.label}_{sta.lower()}_full_stack.npz"
        plot_station(result, out_png, args.label)
        save_npz(result, out_npz)
        rows.append(dict(station=sta, n=result["n"],
                          snr_z=snr["Z"], snr_n=snr["N"], snr_e=snr["E"]))

    out_csv = f"{args.out_dir}/{args.label}_full_stack_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
