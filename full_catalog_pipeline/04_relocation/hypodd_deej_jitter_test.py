#!/usr/bin/env python3
"""Closing-the-loop test for [[t1-cluster3-wrongpulse-refuted-deej-onset-finding]]: does
uncorrelated S-pick jitter of DEEJ's own observed magnitude (~20ms, from
cluster3_direct_s_identification.csv's early_lag_ms std) get amplified by hypoDD's
ill-conditioned depth inversion for cluster3 into a depth spread comparable to the real
~942m range / 82m IQR?

Adds independent Gaussian jitter (mean 0, --sigma-ms std) to every DEEJ S-phase
differential-time observation (dt.ct absolute times, dt.cc differential values) for the
cluster3 subset (n=222 ids), keeping every other station/phase and the CC values themselves
untouched, then reruns the exact same SVD (ISOLV=1) hypoDD config as the accepted baseline
(hypodd_svd_cluster_errors.py's DAMP 40/30/15/8/5 schedule).

Two runs with different --seed are independent noise realizations. If the resulting depth
solutions are highly correlated with each other despite independent noise, the observed
depth spread is NOT explained by noise of this magnitude. If they decorrelate and/or the
added variance is comparable to the real spread, that's direct quantitative evidence the
real ~942m/82m-IQR spread is consistent with being noise-injected, not real depth structure.

Usage:
    python full_catalog_pipeline/hypodd_deej_jitter_test.py --sigma-ms 20 --seed 1 --label jitter_s1
    python full_catalog_pipeline/hypodd_deej_jitter_test.py --sigma-ms 20 --seed 2 --label jitter_s2
"""
import os as _os
import sys as _sys

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
import subprocess

import numpy as np
import pandas as pd

import catalog_paths

FULL_DIR = "full_catalog_pipeline/artifacts/full_run/T1_v5/hypodd/input_files"
IDS_FILE = "full_catalog_pipeline/artifacts/full_run/T1_v5/hypodd/cluster3_event_ids.txt"
OUT_DIR = "full_catalog_pipeline/artifacts/full_run/T1_v5/hypodd/deej_jitter_test"
WORK_ROOT = catalog_paths.scratch_dir("hypodd_deej_jitter")

RELOC_COLS = [
    "id", "lat", "lon", "depth", "x", "y", "z", "ex", "ey", "ez",
    "yr", "mo", "dy", "hr", "mi", "sc", "mag",
    "nccp", "nccs", "nctp", "ncts", "rcc", "rct", "cid",
]

# Same schedule as the accepted baseline (hypodd_svd_cluster_errors.py write_inp).
INP_TEMPLATE = """hypoDD_2
dt.cc
dt.ct
event.sel
station.sel


hypoDD.sta
hypoDD.res
hypoDD.src
3 3 150
3 4 1 -999 -999
1 1 1 5
6 0.99 0.99 -999 -999  0.60 0.60 10 100  40
6 0.33 0.33 -999 -999  0.40 0.40 10 100  30
6 0.10 0.10 0.06   50  0.20 0.20  3  20  15
6 0.03 0.03 0.04   20  0.05 0.05 -999 -999   8
8 0.02 0.02 0.02   10  0.05 0.05 -999 -999   5
1
0.0 0.1 3.1 3.8 9.5 14.0 25.0 52.0
2.5 3.85 5.1 5.8 6.1 6.5 7.5 8.05
1.84 2.22 2.95 3.35 3.53 3.76 4.34 4.65
0
"""


def filter_event_sel(ids, out_path):
    cols = ["date", "time", "lat", "lon", "depth", "mag", "eh", "ez", "rms", "id"]
    ev = pd.read_csv(f"{FULL_DIR}/event.sel", sep=r"\s+", header=None, names=cols)
    ev = ev[ev["id"].isin(ids)]
    ev.to_csv(out_path, sep=" ", header=False, index=False, float_format="%.5f")
    return len(ev)


def filter_and_jitter_ct(src_path, ids, jitter, out_path, target_sta="7U.DEEJ"):
    ids = set(ids)
    n_pairs = n_obs = n_jittered = 0
    write_pair = False
    id1 = id2 = None
    with open(src_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            if line.startswith("#"):
                parts = line[1:].split()
                id1, id2 = int(parts[0]), int(parts[1])
                write_pair = id1 in ids and id2 in ids
                if write_pair:
                    fout.write(line)
                    n_pairs += 1
            elif write_pair:
                p = line.split()
                sta, t1, t2, wt, pha = p[0], float(p[1]), float(p[2]), p[3], p[4]
                if sta == target_sta and pha == "S":
                    t1 += jitter.get(id1, 0.0)
                    t2 += jitter.get(id2, 0.0)
                    n_jittered += 1
                fout.write(f"{sta}  {t1:.4f}  {t2:.4f} {wt} {pha}\n")
                n_obs += 1
    return n_pairs, n_obs, n_jittered


def filter_and_jitter_cc(src_path, ids, jitter, out_path, target_sta="7U.DEEJ"):
    ids = set(ids)
    n_pairs = n_obs = n_jittered = 0
    write_pair = False
    id1 = id2 = None
    with open(src_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            if line.startswith("#"):
                parts = line[1:].split()
                id1, id2 = int(parts[0]), int(parts[1])
                write_pair = id1 in ids and id2 in ids
                if write_pair:
                    fout.write(line)
                    n_pairs += 1
            elif write_pair:
                p = line.split()
                sta, dt, cc, pha = p[0], float(p[1]), p[2], p[3]
                if sta == target_sta and pha == "S":
                    dt += jitter.get(id1, 0.0) - jitter.get(id2, 0.0)
                    n_jittered += 1
                fout.write(f"{sta} {dt:.6f} {cc} {pha}\n")
                n_obs += 1
    return n_pairs, n_obs, n_jittered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-ms", type=float, default=20.0)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--ids-file", default=IDS_FILE)
    args = ap.parse_args()

    with open(args.ids_file) as f:
        ids = [int(x) for x in f.read().split()]

    rng = np.random.default_rng(args.seed)
    jitter = {eid: rng.normal(0.0, args.sigma_ms / 1000.0) for eid in ids}

    work_dir = f"{WORK_ROOT}/{args.label}"
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    n_ev = filter_event_sel(ids, f"{work_dir}/event.sel")
    n_ct_pairs, n_ct_obs, n_ct_jit = filter_and_jitter_ct(
        f"{FULL_DIR}/dt.ct", ids, jitter, f"{work_dir}/dt.ct")
    n_cc_pairs, n_cc_obs, n_cc_jit = filter_and_jitter_cc(
        f"{FULL_DIR}/dt.cc", ids, jitter, f"{work_dir}/dt.cc")
    os.system(f"cp {FULL_DIR}/station.sel {work_dir}/station.sel")
    with open(f"{work_dir}/hypoDD.inp", "w") as f:
        f.write(INP_TEMPLATE)

    pd.Series(jitter).to_csv(f"{OUT_DIR}/{args.label}_jitter_ms.csv", header=["jitter_ms"])
    print(f"[{args.label}] sigma={args.sigma_ms}ms seed={args.seed}: {n_ev} events, "
          f"ct {n_ct_pairs} pairs/{n_ct_jit} DEEJ-S obs jittered, "
          f"cc {n_cc_pairs} pairs/{n_cc_jit} DEEJ-S obs jittered")

    result = subprocess.run(["/home/jwalter/bin/hypoDD_svd", "hypoDD.inp"], cwd=work_dir,
                             capture_output=True, timeout=14400)
    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    with open(f"{work_dir}/run.log", "w") as f:
        f.write(stdout + stderr)
    print(f"[{args.label}] hypoDD exit code: {result.returncode}")

    reloc_path = f"{work_dir}/hypoDD.reloc"
    if os.path.exists(reloc_path):
        dst = f"{OUT_DIR}/{args.label}_hypoDD.reloc"
        os.system(f"cp {reloc_path} {dst}")
        reloc = pd.read_csv(reloc_path, sep=r"\s+", header=None, names=RELOC_COLS)
        print(f"[{args.label}] relocated {len(reloc)}/{n_ev} events")
        print(f"[{args.label}] depth median={reloc['depth'].median():.3f}km "
              f"std={reloc['depth'].std()*1000:.0f}m "
              f"IQR={(reloc['depth'].quantile(.75)-reloc['depth'].quantile(.25))*1000:.0f}m "
              f"range={(reloc['depth'].max()-reloc['depth'].min())*1000:.0f}m")
    else:
        print(f"[{args.label}] no hypoDD.reloc produced, check {work_dir}/run.log")


if __name__ == "__main__":
    main()
