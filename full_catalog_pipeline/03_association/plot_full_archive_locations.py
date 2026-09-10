#!/usr/bin/env python3
"""Location plots for the full 2-year pyocto catalog (model E) vs QuakeMigrate,
for T1 and T2 separately. Produces, per array:
  - a map view (lon/lat) with station positions
  - a depth cross-section (distance from array centroid vs depth)

Usage:
    python full_catalog_pipeline/plot_full_archive_locations.py [--suffix v2]
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

START, END = "2019-12-29", "2021-12-26"
TOL_SEC = 5.0

COLOR_QM = "#8c96a6"
COLOR_MATCHED = "#15616d"
COLOR_UNMATCHED = "#ff7d00"
COLOR_STATION = "#2b2b2b"

ARRAYS = {
    "T1": {
        "stations": {"DEEJ", "ELZA", "LILA", "LOUS", "OTIS", "SQIG", "TJTJ"},
        "qm_events_json": "/scratch2/qm/t1/output/working_files/events.json",
        "pyocto_events_csv": "full_catalog_pipeline/artifacts/full_run/T1/pyocto_events.csv",
        "out_dir": "full_catalog_pipeline/artifacts/full_run/T1",
    },
    "T2": {
        "stations": {"BAUM", "DRSC", "EPJZ", "FRST", "JULA", "OKGS", "WICH"},
        "qm_events_json": "/scratch2/qm/t2/quakeml/output/working_files/events.json",
        "pyocto_events_csv": "full_catalog_pipeline/artifacts/full_run/T2/pyocto_events.csv",
        "out_dir": "full_catalog_pipeline/artifacts/full_run/T2",
    },
}


def load_data(cfg):
    pred = pd.read_csv(cfg["pyocto_events_csv"])
    with open(cfg["qm_events_json"]) as f:
        qm = json.load(f)
    start, end = UTCDateTime(START), UTCDateTime(END) + 86400
    qm_window = [e for e in qm if start <= UTCDateTime(e["origin_time"]) < end]
    stations = build_station_df()
    stations = stations[stations["id"].apply(lambda x: x.split(".")[1] in cfg["stations"])].reset_index(drop=True)
    return pred, qm_window, stations


def compute_matches(pred, qm_window):
    pred_times = pred["time"].tolist() if len(pred) else []
    truth_times = [UTCDateTime(e["origin_time"]).timestamp for e in qm_window]
    matches = match_events(pred_times, truth_times, TOL_SEC)
    matched_pred_idx = {pi for pi, _ in matches}
    return matches, matched_pred_idx


def plot_map(array, pred, qm_window, stations, matched_pred_idx, out_dir, tag):
    fig, ax = plt.subplots(figsize=(9, 8))
    qm_lat = [e["origin_latitude"] for e in qm_window]
    qm_lon = [e["origin_longitude"] for e in qm_window]
    ax.scatter(qm_lon, qm_lat, s=6, color=COLOR_QM, alpha=0.25, label=f"QuakeMigrate (n={len(qm_window)})", zorder=1)

    is_matched = pred.index.isin(matched_pred_idx)
    ax.scatter(pred.loc[~is_matched, "longitude"], pred.loc[~is_matched, "latitude"],
               s=16, color=COLOR_UNMATCHED, marker="o", alpha=0.6,
               label=f"pyocto unmatched (n={(~is_matched).sum()})", zorder=2)
    ax.scatter(pred.loc[is_matched, "longitude"], pred.loc[is_matched, "latitude"],
               s=16, color=COLOR_MATCHED, marker="o", alpha=0.6,
               label=f"pyocto matched (n={is_matched.sum()})", zorder=3)

    ax.scatter(stations["longitude"], stations["latitude"], s=160, color=COLOR_STATION, marker="^",
               edgecolor="white", linewidth=1, label=f"{array} stations", zorder=4)
    for _, s in stations.iterrows():
        ax.annotate(s["id"].split(".")[1], (s["longitude"], s["latitude"]), fontsize=9,
                    xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{array} array: full-archive pyocto (model E){tag} vs QuakeMigrate\n({START} to {END})", fontsize=14)
    ax.legend(fontsize=10, loc="best")
    fig.tight_layout()
    fig.savefig(f"{out_dir}/full_archive_map_comparison{tag}.png", dpi=150)
    plt.close(fig)


def plot_depth_cross_section(array, pred, qm_window, stations, matched_pred_idx, out_dir, tag):
    centroid_lat = stations["latitude"].mean()
    centroid_lon = stations["longitude"].mean()

    def dist_from_centroid(lat, lon):
        d_m, _, _ = gps2dist_azimuth(centroid_lat, centroid_lon, lat, lon)
        return d_m / 1000.0

    qm_dist = [dist_from_centroid(e["origin_latitude"], e["origin_longitude"]) for e in qm_window]
    qm_depth = [e["origin_depth"] / 1000.0 for e in qm_window]

    is_matched = pred.index.isin(matched_pred_idx)
    pred_dist = [dist_from_centroid(lat, lon) for lat, lon in zip(pred["latitude"], pred["longitude"])]
    pred_depth_km = pred["depth"].tolist()

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(qm_dist, qm_depth, s=6, color=COLOR_QM, alpha=0.25, label=f"QuakeMigrate (n={len(qm_window)})", zorder=1)
    ax.scatter(np.array(pred_dist)[~is_matched], np.array(pred_depth_km)[~is_matched],
               s=16, color=COLOR_UNMATCHED, alpha=0.6, label=f"pyocto unmatched (n={(~is_matched).sum()})", zorder=2)
    ax.scatter(np.array(pred_dist)[is_matched], np.array(pred_depth_km)[is_matched],
               s=16, color=COLOR_MATCHED, alpha=0.6, label=f"pyocto matched (n={is_matched.sum()})", zorder=3)

    station_dists = [dist_from_centroid(lat, lon) for lat, lon in zip(stations["latitude"], stations["longitude"])]
    ax.scatter(station_dists, [0] * len(station_dists), s=160, color=COLOR_STATION, marker="^",
               edgecolor="white", linewidth=1, label=f"{array} stations (depth=0)", zorder=4)

    ax.invert_yaxis()
    ax.set_xlabel("Distance from array centroid (km)")
    ax.set_ylabel("Depth (km)")
    ax.set_title(f"{array} array: depth cross-section relative to stations{tag}\n({START} to {END})", fontsize=14)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/full_archive_depth_cross_section{tag}.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="", help="e.g. 'v2' to read/write full_run/{T1,T2}_v2 without clobbering originals")
    args = ap.parse_args()
    tag = f"_{args.suffix}" if args.suffix else ""

    for array, cfg in ARRAYS.items():
        out_dir = f"{cfg['out_dir']}{tag}" if args.suffix else cfg["out_dir"]
        events_csv = f"{cfg['out_dir']}{tag}/pyocto_events.csv" if args.suffix else cfg["pyocto_events_csv"]
        cfg_run = {**cfg, "pyocto_events_csv": events_csv}

        pred, qm_window, stations = load_data(cfg_run)
        matches, matched_pred_idx = compute_matches(pred, qm_window)
        print(f"[{array}] pyocto events: {len(pred)}  QM events: {len(qm_window)}  matched: {len(matches)}")

        plot_map(array, pred, qm_window, stations, matched_pred_idx, out_dir, tag)
        plot_depth_cross_section(array, pred, qm_window, stations, matched_pred_idx, out_dir, tag)
        print(f"[{array}] wrote full_archive_map_comparison{tag}.png, full_archive_depth_cross_section{tag}.png to {out_dir}/")


if __name__ == "__main__":
    main()
