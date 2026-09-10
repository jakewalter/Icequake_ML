#!/usr/bin/env python3
"""Score easyQuake seisbench_picks.out continuous-detection output against
the real QuakeMigrate pick catalog for one station-day.

Usage:
    python full_catalog_pipeline/score_continuous_picks.py \
        --picks-out easyquake_project/20200713_drsc_new/seisbench_picks.out \
        --truth-csv artifacts/picks_deduped.csv \
        --station DRSC --date 2020-07-13 --label new_model
"""
import argparse

import pandas as pd
from obspy import UTCDateTime


def load_picks_utc(filepath):
    p_picks, s_picks = [], []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            phase, time_str = parts[3], parts[4]
            try:
                dt = UTCDateTime(time_str)
            except Exception:
                continue
            (p_picks if phase == "P" else s_picks if phase == "S" else []).append(dt)
    p_picks.sort()
    s_picks.sort()
    return p_picks, s_picks


def filter_consecutive(picks, min_gap=0.5):
    if not picks:
        return []
    out = [picks[0]]
    for p in picks[1:]:
        if (p - out[-1]) >= min_gap:
            out.append(p)
    return out


def match(pred, truth, tol):
    truth = list(truth)
    used = [False] * len(truth)
    matched = 0
    for p in pred:
        best_j, best_d = None, tol
        for j, t in enumerate(truth):
            if used[j]:
                continue
            d = abs(p - t)
            if d <= best_d:
                best_d, best_j = d, j
        if best_j is not None:
            used[best_j] = True
            matched += 1
    return matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks-out", required=True)
    ap.add_argument("--truth-csv", required=True)
    ap.add_argument("--station", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--label", default="model")
    ap.add_argument("--tol-sec", type=float, default=1.0)
    args = ap.parse_args()

    pred_p, pred_s = load_picks_utc(args.picks_out)
    pred_p = filter_consecutive(pred_p, 0.5)
    pred_s = filter_consecutive(pred_s, 0.5)

    df = pd.read_csv(args.truth_csv)
    sub = df[(df["station"] == args.station) & (df["pick_time"].str[:10] == args.date)]
    truth_p = sorted(UTCDateTime(t) for t in sub[sub["phase"] == "P"]["pick_time"])
    truth_s = sorted(UTCDateTime(t) for t in sub[sub["phase"] == "S"]["pick_time"])

    mp = match(pred_p, truth_p, args.tol_sec)
    ms = match(pred_s, truth_s, args.tol_sec)

    def stats(matched, n_pred, n_true):
        precision = matched / n_pred if n_pred else float("nan")
        recall = matched / n_true if n_true else float("nan")
        f1 = 2 * precision * recall / (precision + recall) if (precision or 0) + (recall or 0) > 0 else float("nan")
        return precision, recall, f1

    pp, pr, pf = stats(mp, len(pred_p), len(truth_p))
    sp, sr, sf = stats(ms, len(pred_s), len(truth_s))

    print(f"=== {args.label}: {args.station} {args.date} (tol={args.tol_sec}s) ===")
    print(f"P: true={len(truth_p):4d} pred={len(pred_p):4d} matched={mp:4d}  precision={pp:.3f} recall={pr:.3f} f1={pf:.3f}")
    print(f"S: true={len(truth_s):4d} pred={len(pred_s):4d} matched={ms:4d}  precision={sp:.3f} recall={sr:.3f} f1={sf:.3f}")


if __name__ == "__main__":
    main()
