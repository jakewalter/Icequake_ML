#!/usr/bin/env python3
"""Compare a pyocto-derived event catalog against the QuakeMigrate T1/T2
catalog for the same time window, matching events by origin-time proximity
and reporting location differences for matches.

Usage:
    python full_catalog_pipeline/compare_catalogs.py \
        --pyocto-events full_catalog_pipeline/artifacts/pilot/pyocto_events.csv \
        --qm-events-json /scratch2/qm/t2/quakeml/output/working_files/events.json \
        --start-date 2020-07-01 --end-date 2020-07-31 \
        --tol-sec 5.0
"""
import argparse
import json

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth


def match_events(pred_times, truth_times, tol_sec):
    truth_times = list(truth_times)
    used = [False] * len(truth_times)
    matches = []  # (pred_idx, truth_idx)
    for pi, pt in enumerate(pred_times):
        best_j, best_d = None, tol_sec
        for j, tt in enumerate(truth_times):
            if used[j]:
                continue
            d = abs(pt - tt)
            if d <= best_d:
                best_d, best_j = d, j
        if best_j is not None:
            used[best_j] = True
            matches.append((pi, best_j))
    return matches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyocto-events", required=True)
    ap.add_argument("--qm-events-json", required=True, help="comma-separated list of QuakeMigrate events.json paths")
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--tol-sec", type=float, default=5.0)
    args = ap.parse_args()

    pred = pd.read_csv(args.pyocto_events)
    print(f"pyocto events: {len(pred)}")

    qm = []
    for path in args.qm_events_json.split(","):
        with open(path) as f:
            qm.extend(json.load(f))
    start = UTCDateTime(args.start_date)
    end = UTCDateTime(args.end_date) + 86400
    qm_window = [e for e in qm if start <= UTCDateTime(e["origin_time"]) < end]
    print(f"QuakeMigrate events in window: {len(qm_window)}")

    pred_times = pred["time"].tolist() if len(pred) else []
    truth_times = [UTCDateTime(e["origin_time"]).timestamp for e in qm_window]

    matches = match_events(pred_times, truth_times, args.tol_sec)
    n_matched = len(matches)
    precision = n_matched / len(pred_times) if pred_times else float("nan")
    recall = n_matched / len(truth_times) if truth_times else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision or 0) + (recall or 0) > 0 else float("nan")

    print(f"=== Event-level comparison (tol={args.tol_sec}s) ===")
    print(f"pyocto events: {len(pred_times)}  QuakeMigrate events: {len(truth_times)}  matched: {n_matched}")
    print(f"precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}")

    if n_matched > 0 and "latitude" in pred.columns:
        dists_km, depth_diffs_m, time_diffs_s = [], [], []
        for pi, ti in matches:
            p = pred.iloc[pi]
            t = qm_window[ti]
            if pd.notna(p.get("latitude")) and t.get("origin_latitude") is not None:
                dist_m, _, _ = gps2dist_azimuth(p["latitude"], p["longitude"], t["origin_latitude"], t["origin_longitude"])
                dists_km.append(dist_m / 1000.0)
            if pd.notna(p.get("depth")) and t.get("origin_depth") is not None:
                depth_diffs_m.append(abs(p["depth"] * 1000.0 - t["origin_depth"]))
            time_diffs_s.append(abs(p["time"] - UTCDateTime(t["origin_time"]).timestamp))
        if dists_km:
            print(f"epicentral distance (matched events): mean={np.mean(dists_km):.2f} km  median={np.median(dists_km):.2f} km")
        if depth_diffs_m:
            print(f"depth difference (matched events): mean={np.mean(depth_diffs_m):.0f} m  median={np.median(depth_diffs_m):.0f} m")
        print(f"origin time difference (matched events): mean={np.mean(time_diffs_s):.2f} s  median={np.median(time_diffs_s):.2f} s")


if __name__ == "__main__":
    main()
