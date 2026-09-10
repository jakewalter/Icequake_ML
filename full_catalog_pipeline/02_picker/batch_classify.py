#!/usr/bin/env python3
"""Batch continuous PhaseNet inference over a station list and date range,
loading the model once and reusing it across all station-days (avoids the
~10s per-invocation reload overhead of classify_continuous.py). Writes all
picks to a single parquet file in pyocto's expected pick format
(station, time, phase, probability), with station id as "<NET>.<STA>."
matching pyocto's inventory_to_df convention.

Usage:
    python full_catalog_pipeline/batch_classify.py \
        --stations BAUM,DRSC,EPJZ \
        --start-date 2020-07-01 --end-date 2020-07-31 \
        --checkpoint full_catalog_pipeline/checkpoints/best_model_state_only.pth \
        --out-parquet full_catalog_pipeline/artifacts/pilot_picks_batch1.parquet
"""
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import argparse
import glob
from datetime import date, timedelta

import pandas as pd
import torch
from obspy import Stream, read

import seisbench.models as sbm

NETWORK = "7U"
DAY_VOLUMES_ROOT = "/data/time/day_volumes"


def daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def load_day_stream(station, d):
    yyyymmdd = d.strftime("%Y%m%d")
    doy = d.strftime("%j")
    src_dir = os.path.join(DAY_VOLUMES_ROOT, yyyymmdd)
    st = Stream()
    for chan in ("HHZ", "HH1", "HH2"):
        pattern = os.path.join(src_dir, f"{station}.{NETWORK}..{chan}.{d.year}.{doy}")
        matches = glob.glob(pattern)
        if not matches:
            return None
        st += read(matches[0])
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", required=True, help="comma-separated station codes")
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--sampling-rate", type=float, default=200.0)
    ap.add_argument("--p-threshold", type=float, default=0.5)
    ap.add_argument("--s-threshold", type=float, default=0.5)
    args = ap.parse_args()

    stations = args.stations.split(",")
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = sbm.PhaseNet(phases="PSN", norm="peak")
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.sampling_rate = args.sampling_rate
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}. Stations: {stations}. Range: {start}..{end}")

    rows = []
    n_days = (end - start).days + 1
    for station in stations:
        n_ok, n_missing = 0, 0
        for i, d in enumerate(daterange(start, end)):
            st = load_day_stream(station, d)
            if st is None:
                n_missing += 1
                continue
            st.detrend(type="linear")
            try:
                output = model.classify(
                    st, P_threshold=args.p_threshold, S_threshold=args.s_threshold, batch_size=256
                )
            except Exception as e:
                print(f"  {station} {d}: classify failed: {e}")
                continue
            for pick in output.picks:
                rows.append({
                    "station": f"{NETWORK}.{station}.",
                    "time": pick.peak_time.timestamp,
                    "phase": pick.phase,
                    "probability": pick.peak_value,
                })
            n_ok += 1
            if (i + 1) % 5 == 0:
                print(f"  {station}: {i+1}/{n_days} days done, {len(rows)} picks so far")
        print(f"{station}: {n_ok} days processed, {n_missing} missing")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df)} picks to {args.out_csv}")


if __name__ == "__main__":
    main()
