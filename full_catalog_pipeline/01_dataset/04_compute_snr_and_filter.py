#!/usr/bin/env python3
"""Stage 4: SNR computation + diagnostic figures + threshold-based filtering.

Reads: artifacts/windows.h5, artifacts/windows_metadata.csv (complete=True rows)
Writes: artifacts/windows_with_snr.csv (always), several PNG figures (always),
        artifacts/windows_filtered.csv (only if --snr-threshold-db is given)
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from lib.snr import compute_snr_db

THRESHOLD_STEPS_DB = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20]


def compute_all_snr():
    meta = pd.read_csv(config.WINDOWS_METADATA_CSV)
    meta = meta[meta["complete"] == True].copy()  # noqa: E712
    print(f"{len(meta)} complete windows to score")

    snr_z, snr_n, snr_e, snr_scalar, snr_status = [], [], [], [], []
    with h5py.File(config.WINDOWS_H5, "r") as h5f:
        for i, row in enumerate(meta.itertuples(index=False)):
            key = f"{row.event_id}__{row.station}"
            grp = h5f[key]

            p_sample = row.p_arrival_sample if pd.notna(row.p_arrival_sample) else None
            s_sample = row.s_arrival_sample if pd.notna(row.s_arrival_sample) else None
            candidates = [v for v in (p_sample, s_sample) if v is not None]
            ref_offset_sec = min(candidates) / config.SAMPLE_RATE_HZ

            z = compute_snr_db(grp["Z"][:], config.SAMPLE_RATE_HZ, ref_offset_sec)
            n = compute_snr_db(grp["N"][:], config.SAMPLE_RATE_HZ, ref_offset_sec)
            e = compute_snr_db(grp["E"][:], config.SAMPLE_RATE_HZ, ref_offset_sec)

            finite = [v for v in (z, n, e) if np.isfinite(v)]
            scalar = max(finite) if finite else float("nan")
            status = "ok" if finite else "degenerate"

            snr_z.append(z); snr_n.append(n); snr_e.append(e)
            snr_scalar.append(scalar); snr_status.append(status)

            if (i + 1) % 10000 == 0:
                print(f"  scored {i + 1}/{len(meta)}")

    meta["snr_db_z"] = snr_z
    meta["snr_db_n"] = snr_n
    meta["snr_db_e"] = snr_e
    meta["trace_snr_db"] = snr_scalar
    meta["snr_status"] = snr_status
    return meta


def survival_table(df):
    ok = df[df["snr_status"] == "ok"]
    total = len(df)
    rows = []
    for t in THRESHOLD_STEPS_DB:
        n = int((ok["trace_snr_db"] >= t).sum())
        rows.append((t, n, 100.0 * n / total if total else 0.0))
    return pd.DataFrame(rows, columns=["threshold_db", "n_survive", "pct_of_total"])


def make_figures(df):
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    ok = df[df["snr_status"] == "ok"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, log_y in zip(axes, [False, True]):
        ax.hist(ok["trace_snr_db"], bins=80, color="steelblue")
        if log_y:
            ax.set_yscale("log")
        for t in [5, 8, 10, 12]:
            ax.axvline(t, color="crimson", linestyle="--", linewidth=0.8)
        ax.set_xlabel("trace_snr_db (max of Z/N/E)")
        ax.set_ylabel("count" + (" (log)" if log_y else ""))
    fig.suptitle("Overall SNR distribution")
    fig.tight_layout()
    fig.savefig(os.path.join(config.ARTIFACTS_DIR, "snr_distribution_overall.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for array, color in [("T1", "tab:blue"), ("T2", "tab:orange")]:
        sub = ok[ok["array"] == array]
        ax.hist(sub["trace_snr_db"], bins=80, alpha=0.55, label=f"{array} (n={len(sub)})", color=color)
    ax.set_yscale("log")
    ax.set_xlabel("trace_snr_db")
    ax.set_ylabel("count (log)")
    ax.legend()
    ax.set_title("SNR distribution by array")
    fig.tight_layout()
    fig.savefig(os.path.join(config.ARTIFACTS_DIR, "snr_distribution_by_array.png"), dpi=130)
    plt.close(fig)

    stations = sorted(ok["station"].unique())
    ncols = 5
    nrows = int(np.ceil(len(stations) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.6 * nrows), sharex=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, station in zip(axes, stations):
        sub = ok[ok["station"] == station]
        ax.hist(sub["trace_snr_db"], bins=40, color="seagreen")
        ax.set_title(f"{station} (n={len(sub)})", fontsize=9)
        ax.set_yscale("log")
    for ax in axes[len(stations):]:
        ax.axis("off")
    fig.suptitle("SNR distribution by station")
    fig.tight_layout()
    fig.savefig(os.path.join(config.ARTIFACTS_DIR, "snr_distribution_by_station.png"), dpi=130)
    plt.close(fig)

    thresholds = np.arange(0, 22, 0.5)
    survive = [(ok["trace_snr_db"] >= t).sum() for t in thresholds]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(thresholds, survive, color="black")
    ax.axhspan(10000, 15000, color="crimson", alpha=0.15, label="10-15k target")
    ax.set_xlabel("SNR threshold (dB)")
    ax.set_ylabel("surviving windows")
    ax.legend()
    ax.set_title("SNR survival curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(config.ARTIFACTS_DIR, "snr_survival_curve.png"), dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--snr-threshold-db", type=float, default=None)
    args = parser.parse_args()

    if args.force or not os.path.exists(config.WINDOWS_WITH_SNR_CSV):
        df = compute_all_snr()
        df.to_csv(config.WINDOWS_WITH_SNR_CSV, index=False)
        print(f"\nWrote {config.WINDOWS_WITH_SNR_CSV}")
    else:
        print(f"{config.WINDOWS_WITH_SNR_CSV} already exists, reusing (use --force to recompute).")
        df = pd.read_csv(config.WINDOWS_WITH_SNR_CSV)

    print("\nMaking diagnostic figures...")
    make_figures(df)
    print(f"Figures written to {config.ARTIFACTS_DIR}")

    table = survival_table(df)
    print("\nSurvival table:")
    print(table.to_string(index=False))

    print("\nPer-station/array survive vs reject at a few candidate thresholds:")
    for t in [5, 8, 10]:
        ok_at_t = df[(df["snr_status"] == "ok") & (df["trace_snr_db"] >= t)]
        print(f"\n  threshold={t} dB, total survive={len(ok_at_t)}")
        print(ok_at_t.groupby(["array", "station"]).size().to_string())

    if args.snr_threshold_db is not None:
        filtered = df[(df["snr_status"] == "ok") & (df["trace_snr_db"] >= args.snr_threshold_db)]
        filtered.to_csv(config.WINDOWS_FILTERED_CSV, index=False)
        print(f"\nApplied threshold {args.snr_threshold_db} dB -> {len(filtered)} rows")
        print(f"Wrote {config.WINDOWS_FILTERED_CSV}")


if __name__ == "__main__":
    main()
