#!/usr/bin/env python3
"""Comparison plots for a pyocto pilot catalog (T2 array, July 2020) vs the
QuakeMigrate T2 catalog for the same window. Saves PNGs (and a metrics JSON)
to --out-dir. Parametrized so the same script runs for any of the A/B/C
SNR/min-stations ablation variants -- see full_catalog_pipeline/artifacts/pilot_*/.

Usage:
    python full_catalog_pipeline/plot_pilot_comparison.py \
        --pyocto-events full_catalog_pipeline/artifacts/pilot_B/pyocto_events.csv \
        --out-dir full_catalog_pipeline/artifacts/pilot_B \
        --label "Model B (SNR>=5dB, >=3 stations)"
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
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from obspy import UTCDateTime
from obspy.geodetics import gps2dist_azimuth

sys.path.insert(0, "full_catalog_pipeline")
from associate_pyocto import build_station_df
from compare_catalogs import match_events

sns.set(style="whitegrid", context="talk")

QM_EVENTS_JSON = "/scratch2/qm/t2/quakeml/output/working_files/events.json"
START, END = "2020-07-01", "2020-07-31"
TOL_SEC = 5.0
T2_STATIONS = {"BAUM", "DRSC", "EPJZ", "FRST", "JULA", "OKGS", "WICH"}

# Categorical colors, fixed assignment (not cycled): QuakeMigrate = slate gray
# (background/reference), pyocto matched = teal, pyocto unmatched = amber.
COLOR_QM = "#8c96a6"
COLOR_MATCHED = "#15616d"
COLOR_UNMATCHED = "#ff7d00"
COLOR_STATION = "#2b2b2b"


def load_data(pyocto_events_path):
    pred = pd.read_csv(pyocto_events_path)
    with open(QM_EVENTS_JSON) as f:
        qm = json.load(f)
    start, end = UTCDateTime(START), UTCDateTime(END) + 86400
    qm_window = [e for e in qm if start <= UTCDateTime(e["origin_time"]) < end]
    stations = build_station_df()
    stations = stations[stations["id"].apply(lambda x: x.split(".")[1] in T2_STATIONS)].reset_index(drop=True)
    return pred, qm_window, stations


def compute_matches(pred, qm_window):
    pred_times = pred["time"].tolist() if len(pred) else []
    truth_times = [UTCDateTime(e["origin_time"]).timestamp for e in qm_window]
    matches = match_events(pred_times, truth_times, TOL_SEC)
    matched_pred_idx = {pi for pi, _ in matches}
    matched_truth_idx = {ti for _, ti in matches}
    return matches, matched_pred_idx, matched_truth_idx


def plot_map(pred, qm_window, stations, matched_pred_idx, out_dir, label):
    fig, ax = plt.subplots(figsize=(9, 8))
    qm_lat = [e["origin_latitude"] for e in qm_window]
    qm_lon = [e["origin_longitude"] for e in qm_window]
    ax.scatter(qm_lon, qm_lat, s=10, color=COLOR_QM, alpha=0.4, label=f"QuakeMigrate (n={len(qm_window)})", zorder=1)

    is_matched = pred.index.isin(matched_pred_idx)
    ax.scatter(pred.loc[~is_matched, "longitude"], pred.loc[~is_matched, "latitude"],
               s=28, color=COLOR_UNMATCHED, marker="o", label=f"pyocto unmatched (n={(~is_matched).sum()})", zorder=2)
    ax.scatter(pred.loc[is_matched, "longitude"], pred.loc[is_matched, "latitude"],
               s=28, color=COLOR_MATCHED, marker="o", label=f"pyocto matched (n={is_matched.sum()})", zorder=3)

    ax.scatter(stations["longitude"], stations["latitude"], s=140, color=COLOR_STATION, marker="^",
               edgecolor="white", linewidth=1, label="T2 stations", zorder=4)
    for _, s in stations.iterrows():
        ax.annotate(s["id"].split(".")[1], (s["longitude"], s["latitude"]), fontsize=9,
                    xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"T2 array: {label} vs QuakeMigrate\n(July 2020)", fontsize=15)
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/pilot_map_comparison.png", dpi=150)
    plt.close(fig)


def plot_daily_counts(pred, qm_window, out_dir, label):
    days = pd.date_range(START, END, freq="D")
    pred_dates = pd.to_datetime(pred["time"], unit="s").dt.floor("D") if len(pred) else pd.Series([], dtype="datetime64[ns]")
    qm_dates = pd.to_datetime([e["origin_time"] for e in qm_window]).tz_localize(None).floor("D")

    pred_counts = pred_dates.value_counts().reindex(days, fill_value=0)
    qm_counts = pd.Series(qm_dates).value_counts().reindex(days, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(days, qm_counts.values, color=COLOR_QM, linewidth=2, marker="o", markersize=4, label="QuakeMigrate")
    ax.plot(days, pred_counts.values, color=COLOR_MATCHED, linewidth=2, marker="o", markersize=4, label=label)
    ax.set_ylabel("Events per day")
    ax.set_title(f"Daily event count: {label} vs QuakeMigrate (T2, July 2020)", fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/pilot_daily_event_counts.png", dpi=150)
    plt.close(fig)


def plot_matched_residuals(pred, qm_window, matches, out_dir):
    time_diffs, dists_km, depth_diffs_m = [], [], []
    for pi, ti in matches:
        p = pred.iloc[pi]
        t = qm_window[ti]
        time_diffs.append(p["time"] - UTCDateTime(t["origin_time"]).timestamp)
        dist_m, _, _ = gps2dist_azimuth(p["latitude"], p["longitude"], t["origin_latitude"], t["origin_longitude"])
        dists_km.append(dist_m / 1000.0)
        depth_diffs_m.append(p["depth"] * 1000.0 - t["origin_depth"])

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    sns.histplot(time_diffs, bins=30, color=COLOR_MATCHED, ax=axs[0], edgecolor="black")
    axs[0].set_title(f"Origin time diff\nmean={np.mean(time_diffs):.2f}s  median={np.median(time_diffs):.2f}s")
    axs[0].set_xlabel("pyocto - QuakeMigrate (s)")

    sns.histplot(dists_km, bins=30, color=COLOR_MATCHED, ax=axs[1], edgecolor="black")
    axs[1].set_title(f"Epicentral distance\nmean={np.mean(dists_km):.2f} km  median={np.median(dists_km):.2f} km")
    axs[1].set_xlabel("distance (km)")

    sns.histplot(depth_diffs_m, bins=30, color=COLOR_MATCHED, ax=axs[2], edgecolor="black")
    axs[2].set_title(f"Depth diff\nmean={np.mean(depth_diffs_m):.0f} m  median={np.median(depth_diffs_m):.0f} m")
    axs[2].set_xlabel("pyocto - QuakeMigrate (m)")

    fig.suptitle(f"Matched-event agreement (n={len(matches)}, tol={TOL_SEC}s)", fontsize=15)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/pilot_matched_residuals.png", dpi=150)
    plt.close(fig)

    return time_diffs, dists_km, depth_diffs_m


def plot_recall_gap_by_pick_count(qm_window, matched_truth_idx, out_dir):
    n_picks_matched = [len(qm_window[ti]["picks"]) for ti in matched_truth_idx]
    n_picks_unmatched = [len(e["picks"]) for i, e in enumerate(qm_window) if i not in matched_truth_idx]

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.arange(0, max(n_picks_matched + n_picks_unmatched, default=1) + 2) - 0.5
    ax.hist(n_picks_unmatched, bins=bins, color=COLOR_UNMATCHED, alpha=0.7, label=f"missed by pyocto (n={len(n_picks_unmatched)})")
    ax.hist(n_picks_matched, bins=bins, color=COLOR_MATCHED, alpha=0.7, label=f"recovered by pyocto (n={len(n_picks_matched)})")
    ax.set_xlabel("QuakeMigrate picks per event")
    ax.set_ylabel("Count")
    ax.set_title("Why pyocto misses events: QuakeMigrate pick count\n(recovered vs missed, T2 July 2020)", fontsize=14)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/pilot_recall_gap_by_pick_count.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyocto-events", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    pred, qm_window, stations = load_data(args.pyocto_events)
    matches, matched_pred_idx, matched_truth_idx = compute_matches(pred, qm_window)
    print(f"[{args.label}] pyocto events: {len(pred)}  QM events: {len(qm_window)}  matched: {len(matches)}")

    plot_map(pred, qm_window, stations, matched_pred_idx, args.out_dir, args.label)
    plot_daily_counts(pred, qm_window, args.out_dir, args.label)
    time_diffs, dists_km, depth_diffs_m = plot_matched_residuals(pred, qm_window, matches, args.out_dir)
    plot_recall_gap_by_pick_count(qm_window, matched_truth_idx, args.out_dir)

    n_pred, n_true, n_matched = len(pred), len(qm_window), len(matches)
    precision = n_matched / n_pred if n_pred else float("nan")
    recall = n_matched / n_true if n_true else float("nan")
    metrics = {
        "label": args.label,
        "n_pyocto_events": n_pred,
        "n_qm_events": n_true,
        "n_matched": n_matched,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan"),
        "matched_time_diff_median_s": float(np.median(time_diffs)) if time_diffs else None,
        "matched_dist_median_km": float(np.median(dists_km)) if dists_km else None,
        "matched_depth_diff_median_m": float(np.median(depth_diffs_m)) if depth_diffs_m else None,
    }
    with open(f"{args.out_dir}/pilot_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{args.label}] precision={precision:.3f} recall={recall:.3f}")
    print(f"Wrote 4 plots + pilot_metrics.json to {args.out_dir}/")


if __name__ == "__main__":
    main()
