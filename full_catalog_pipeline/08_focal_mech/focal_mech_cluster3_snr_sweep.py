#!/usr/bin/env python3
"""Does thresholding RPNet's per-pick confidence (its softmax 'prob', stored
as 'snr' in cluster3_polarities_rpnet.csv, plus its MC-dropout 'std') sharpen
the T1 cluster3 composite focal-mechanism fit (see
rpnet_polarity_integration_result / t1_composite_focal_mech_result)? If the
~31% misfit is mostly noisy/low-confidence picks dragging it down, tightening
the threshold should push misfit toward the ~15-20% "resolvable" range and
stabilize strike/dip/rake. If misfit stays high regardless of threshold, that
confirms the mixed-mechanism explanation rather than a pick-quality one.

Generalized 2026-07-29 to accept any cluster via --label (defaults reproduce
the original cluster3 run exactly).

Usage:
    python full_catalog_pipeline/focal_mech_cluster3_snr_sweep.py --label cluster8
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

import numpy as np
import pandas as pd

import catalog_paths

HYPODD_DIR = catalog_paths.work_dir("T1")

SRC_COLS = ["evid", "lat", "lon", "sta", "elv", "dist", "az", "ainp", "ains",
            "ttp", "tts", "xp", "yp", "zp", "xs", "ys", "zs"]

COARSE_STEP_DEG = 5
FINE_STEP_DEG = 1
FINE_HALF_RANGE_DEG = 8

PROB_THRESHOLDS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
STD_THRESHOLDS = [0.20, 0.10, 0.05, 0.02]


def radiation_pattern_p(strike_deg, dip_deg, rake_deg, az_deg, ih_deg):
    strike, dip, rake = np.radians(strike_deg), np.radians(dip_deg), np.radians(rake_deg)
    phi = np.radians(az_deg - strike_deg)
    ih = np.radians(ih_deg)
    return (
        np.cos(rake) * np.sin(dip) * np.sin(ih) ** 2 * np.sin(2 * phi)
        - np.cos(rake) * np.cos(dip) * np.sin(2 * ih) * np.cos(phi)
        + np.sin(rake) * np.sin(2 * dip) * (np.cos(ih) ** 2 - np.sin(ih) ** 2 * np.sin(phi) ** 2)
        + np.sin(rake) * np.cos(2 * dip) * np.sin(2 * ih) * np.sin(phi)
    )


def grid_search(az, ih, obs, strikes, dips, rakes):
    best = None
    for strike in strikes:
        for dip in dips:
            for rake in rakes:
                pred = np.sign(radiation_pattern_p(strike, dip, rake, az, ih))
                n_mismatch = int(np.sum(pred != obs))
                if best is None or n_mismatch < best[3]:
                    best = (strike, dip, rake, n_mismatch)
    return best


def fit(df):
    az, ih, obs = df["az"].to_numpy(), df["ainp"].to_numpy(), df["polarity"].to_numpy()
    n = len(df)
    strikes = np.arange(0, 360, COARSE_STEP_DEG)
    dips = np.arange(5, 91, COARSE_STEP_DEG)
    rakes = np.arange(-180, 180, COARSE_STEP_DEG)
    best = grid_search(az, ih, obs, strikes, dips, rakes)
    fs = np.unique(np.arange(best[0] - FINE_HALF_RANGE_DEG, best[0] + FINE_HALF_RANGE_DEG + 1, FINE_STEP_DEG) % 360)
    fd = np.unique(np.clip(np.arange(best[1] - FINE_HALF_RANGE_DEG, best[1] + FINE_HALF_RANGE_DEG + 1, FINE_STEP_DEG), 0, 90))
    fr = np.arange(best[2] - FINE_HALF_RANGE_DEG, best[2] + FINE_HALF_RANGE_DEG + 1, FINE_STEP_DEG)
    fr = np.unique(((fr + 180) % 360) - 180)
    best_fine = grid_search(az, ih, obs, fs, fd, fr)
    return n, best_fine


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="cluster3")
    ap.add_argument("--polarities-csv", default=None,
                    help=f"Defaults to {HYPODD_DIR}/<label>_polarities_rpnet.csv")
    ap.add_argument("--src-file", default=None,
                    help=f"Defaults to {HYPODD_DIR}/<label>_svd_hypoDD.src")
    args = ap.parse_args()
    if args.polarities_csv is None:
        args.polarities_csv = f"{HYPODD_DIR}/{args.label}_polarities_rpnet.csv"
    if args.src_file is None:
        args.src_file = f"{HYPODD_DIR}/{args.label}_svd_hypoDD.src"
    return args


def main():
    args = parse_args()
    pol = pd.read_csv(args.polarities_csv)
    src = pd.read_csv(args.src_file, sep=r"\s+", header=None, names=SRC_COLS)
    src["sta"] = src["sta"].str.split(".").str[-1]
    src = src[["evid", "sta", "az", "ainp"]].rename(columns={"evid": "id"})

    base = pol.merge(src, left_on=["id", "station"], right_on=["id", "sta"], how="inner")
    base = base.drop_duplicates(subset=["id", "station"])
    base = base[base["polarity"] != 0]  # already std<=0.20-filtered upstream

    print(f"{'cut':>18}  {'n':>5}  {'misfit%':>8}  strike  dip  rake")
    n, best = fit(base)
    print(f"{'std<=0.20 (base)':>18}  {n:>5}  {100 * best[3] / n:>7.1f}%  {best[0]:>6.0f}  {best[1]:>3.0f}  {best[2]:>4.0f}")

    print()
    print("-- sweeping RPNet softmax confidence (prob) threshold, on top of the base filter --")
    for thr in PROB_THRESHOLDS:
        sub = base[base["prob"] >= thr]
        if len(sub) < 30:
            print(f"{'prob>=' + str(thr):>18}  {len(sub):>5}  (too few obs, skipped)")
            continue
        n, best = fit(sub)
        print(f"{'prob>=' + str(thr):>18}  {n:>5}  {100 * best[3] / n:>7.1f}%  {best[0]:>6.0f}  {best[1]:>3.0f}  {best[2]:>4.0f}")

    print()
    print("-- sweeping MC-dropout std threshold directly (tighter than the 0.20 default) --")
    pol_all = pol.merge(src, left_on=["id", "station"], right_on=["id", "sta"], how="inner")
    pol_all = pol_all.drop_duplicates(subset=["id", "station"])
    pol_all = pol_all[pol_all["polarity_raw"] != 0]  # RPNet's raw U/D call, before any std filtering
    for thr in STD_THRESHOLDS:
        sub = pol_all[pol_all["std"] <= thr].copy()
        sub["polarity"] = sub["polarity_raw"]
        if len(sub) < 30:
            print(f"{'std<=' + str(thr):>18}  {len(sub):>5}  (too few obs, skipped)")
            continue
        n, best = fit(sub)
        print(f"{'std<=' + str(thr):>18}  {n:>5}  {100 * best[3] / n:>7.1f}%  {best[0]:>6.0f}  {best[1]:>3.0f}  {best[2]:>4.0f}")


if __name__ == "__main__":
    main()
