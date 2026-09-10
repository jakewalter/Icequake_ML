#!/usr/bin/env python3
"""RPNet-based P first-motion polarity picking for a T1 hypoDD cluster,
replacing the naive amplitude-sign heuristic in
focal_mech_cluster3_polarities.py with the pretrained RPNet deep-learning
polarity classifier (Han et al., 2025, SRL, doi:10.1785/0220240384;
https://github.com/jongwon-han/RPNet). Originally built for cluster3 (see
t1_composite_focal_mech_result / rpnet_polarity_integration_result memories);
generalized 2026-07-29 to run on any cluster via --label, following this
pipeline's established build_common_argparser convention (see
lib/deej_waveform_common.py) -- defaults reproduce the original cluster3 run
exactly.

Same output schema (id, station, polarity, polarity_raw, prob, std, snr,
reliable) regardless of cluster, so focal_mech_cluster3_fit_rpnet.py /
focal_mech_cluster3_snr_sweep.py can consume any cluster's output the same
way.

Station coverage is auto-detected per cluster from phase.dat (a station is
included if it has at least --min-picks P picks among the cluster's events)
rather than hardcoded, since different clusters see different subsets of
this deployment's 7 stations.

Preprocessing mirrors RPNet's own wf2matrix() exactly (see rpnet/predict.py):
raw Z trace -> interpolate to 100 Hz -> highpass 1 Hz -> trim to
ptime-2.5s:ptime+2.5s -> normalize (max-abs) -> first 500 samples. RPNet's
prep_input() then slices the centered 400-sample window itself. Our own
extract_window/RollingDayCache infra loads the raw 200 Hz window with extra
padding (HALF_WIN_PAD) so the highpass filter and interpolation don't leave
edge transients inside the final trimmed window.

Must run in the `rpnet` conda env (tensorflow==2.7.0, keras-self-attention,
obspy==1.3.1 -- see RPNet's README for the pinned dependency set):

    conda activate rpnet
    python full_catalog_pipeline/rpnet_cluster3_polarities.py --label cluster8
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
import sys
from datetime import date

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # this host's CPU is far more reliable than its GPU driver

import numpy as np
import pandas as pd
from obspy import Stream, Trace, UTCDateTime

sys.path.insert(0, ".")
import config
from lib.day_volume_index import load_day_file_index_csv
from lib.windowing import RollingDayCache, extract_window

from rpnet.predict import pred_rpnet

import catalog_paths

HYPODD_DIR = catalog_paths.work_dir("T1")
PHASE_DAT = f"{HYPODD_DIR}/input_files/phase.dat"
MODEL_PATH = "full_catalog_pipeline/rpnet_model/RPNet_v1.h5"

ALL_STATIONS = ["DEEJ", "ELZA", "LILA", "TJTJ", "OTIS", "LOUS", "SQIG"]  # full 7U deployment
NETWORK = "7U"

TARGET_SR = 100.0
HALF_WIN_PAD = 3.5    # raw-window half-width fetched (200 Hz), padding for interpolation/filter edge effects
HALF_WIN_FINAL = 2.5  # RPNet's own trimmed window half-width
N_SAMPLES = 500        # 5 s @ 100 Hz, matches RPNet's wf2matrix()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="cluster3",
                    help="Cluster label; defaults derive --ids-file/--out-csv from it (e.g. 'cluster8').")
    ap.add_argument("--ids-file", default=None,
                    help=f"Defaults to {HYPODD_DIR}/<label>_event_ids.txt")
    ap.add_argument("--phase-dat", default=PHASE_DAT)
    ap.add_argument("--out-csv", default=None,
                    help=f"Defaults to {HYPODD_DIR}/<label>_polarities_rpnet.csv")
    ap.add_argument("--model-path", default=MODEL_PATH)
    ap.add_argument("--min-picks", type=int, default=10,
                    help="Minimum P picks within this cluster for a station to be included.")
    ap.add_argument("--std-threshold", type=float, default=0.2,
                    help="RPNet repo default (hyperparams.py) for flagging MC-dropout-unstable picks as unknown.")
    ap.add_argument("--iterations", type=int, default=100, help="MC-dropout iterations for mean/std.")
    args = ap.parse_args()
    if args.ids_file is None:
        args.ids_file = f"{HYPODD_DIR}/{args.label}_event_ids.txt"
    if args.out_csv is None:
        args.out_csv = f"{HYPODD_DIR}/{args.label}_polarities_rpnet.csv"
    return args


def load_events(ids, phase_dat, stations, network="7U"):
    """Identical to focal_mech_cluster3_polarities.load_events."""
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


def rpnet_window(cache, pick_time):
    """Returns a 500-sample float64 array (or None) following RPNet's own
    wf2matrix() preprocessing exactly."""
    d = date(pick_time.year, pick_time.month, pick_time.day)
    row = {"primary_date": d.isoformat()}
    window_start = pick_time - HALF_WIN_PAD
    window_end = pick_time + HALF_WIN_PAD
    data, status = extract_window(cache, row, window_start, window_end)
    cache.evict_before(row["primary_date"])
    if data is None or not data["complete"]:
        return None

    z = data["Z"].astype(np.float64)
    tr = Trace(data=z, header={"sampling_rate": config.SAMPLE_RATE_HZ, "starttime": window_start})
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


def main():
    args = parse_args()

    ids = set(int(x) for x in open(args.ids_file))
    all_events = load_events(ids, args.phase_dat, ALL_STATIONS, network=NETWORK)
    print(f"[{args.label}] events with any P pick: {len(all_events)}")

    pick_counts = {sta: 0 for sta in ALL_STATIONS}
    for ev in all_events.values():
        for sta in ev["picks"]:
            pick_counts[sta] += 1
    stations = [sta for sta in ALL_STATIONS if pick_counts[sta] >= args.min_picks]
    print(f"[{args.label}] station P-pick counts: {pick_counts}")
    print(f"[{args.label}] stations used (>={args.min_picks} picks): {stations}")
    if not stations:
        print(f"[{args.label}] no station meets --min-picks, aborting")
        return

    events = {eid: {**ev, "picks": {s: t for s, t in ev["picks"].items() if s in stations}}
              for eid, ev in all_events.items()}
    events = {eid: ev for eid, ev in events.items() if ev["picks"]}
    print(f"[{args.label}] events with >=1 usable-station P pick: {len(events)}")

    day_index = load_day_file_index_csv(config.DAY_FILE_INDEX_CSV)
    caches = {sta: RollingDayCache(day_index, sta) for sta in stations}

    ordered_events = sorted(events.items(), key=lambda kv: kv[1]["origin"])

    rows_meta = []
    mats = []
    n_missing = 0
    for n_done, (eid, ev) in enumerate(ordered_events):
        if n_done % 25 == 0:
            print(f"  progress: {n_done}/{len(ordered_events)} events", flush=True)
        for sta, p_offset in ev["picks"].items():
            pick_time = ev["origin"] + p_offset
            arr = rpnet_window(caches[sta], pick_time)
            if arr is None:
                n_missing += 1
                continue
            mats.append(arr[np.newaxis, :])
            rows_meta.append({"id": eid, "station": sta})

    print(f"total event-station P picks attempted: "
          f"{sum(len(e['picks']) for e in events.values())}")
    print(f"missing/out-of-coverage or filter-failed windows: {n_missing}")
    print(f"usable windows for RPNet: {len(rows_meta)}")
    if not rows_meta:
        print(f"[{args.label}] no usable windows, aborting")
        return

    in_mat = np.vstack(mats)
    meta_df = pd.DataFrame(rows_meta)

    r_df = pred_rpnet(args.model_path, in_mat, meta_df, batch_size=2 ** 13,
                       iteration=args.iterations, gpu_num=-1, time_shift=0.0, mid_point=250)

    r_df["polarity_raw"] = r_df["predict"].map({"U": 1, "D": -1, "K": 0})  # pre-std-threshold, for sweeps
    r_df.loc[r_df["std"] > args.std_threshold, "predict"] = "K"
    r_df["polarity"] = r_df["predict"].map({"U": 1, "D": -1, "K": 0})
    r_df["snr"] = r_df["prob"]  # keep column name compatible with the heuristic CSV's schema
    r_df["reliable"] = r_df["polarity"] != 0

    out = r_df[["id", "station", "polarity", "polarity_raw", "prob", "std", "snr", "reliable"]]
    print(out.groupby("station")["polarity"].apply(lambda s: (s != 0).sum()).rename("n_reliable"))
    print(out.groupby("station")["reliable"].mean().rename("frac_reliable"))
    print(f"\noverall polarity counts: up={(out['polarity'] == 1).sum()}, "
          f"down={(out['polarity'] == -1).sum()}, indeterminate={(out['polarity'] == 0).sum()}")

    out.to_csv(args.out_csv, index=False)
    print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
