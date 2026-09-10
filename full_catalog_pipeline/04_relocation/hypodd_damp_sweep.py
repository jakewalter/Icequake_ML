#!/usr/bin/env python3
"""Rerun hypoDD SVD (ISOLV=1) on the T1 cluster3 basal subset (n=222, same
input as hypodd_svd_cluster_errors.py) with alternate DAMP schedules, to test
whether cluster3's ~940m apparent depth spread is a damping-driven inversion
artifact rather than resolved structure.

Motivation: a prior partial rerun of this exact cluster (scratch dir
68466f68.../hypodd_svd/cluster3_n222, 2026-07-27) showed a mean DZ shift of
~600m in the single iteration where DAMP dropped from 40 (stage 1) to 30
(stage 2), producing a negative-depth "air quake" -- the classic signature of
an ill-conditioned direction being kicked around by reduced regularization,
not a real signal converging smoothly. This script tests that directly by
holding/tapering DAMP differently while keeping the same WT_CC/WT_CT/MAXR
reweighting schedule (so any change in behavior is attributable to DAMP, not
the data-selection schedule).

Usage:
    python full_catalog_pipeline/hypodd_damp_sweep.py --damp 40 40 40 40 40 --label damp_const40
    python full_catalog_pipeline/hypodd_damp_sweep.py --damp 40 35 30 20 10 --label damp_gentle
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

import pandas as pd

import catalog_paths

FULL_DIR = "full_catalog_pipeline/artifacts/full_run/T1_v5/hypodd/input_files"
IDS_FILE = "full_catalog_pipeline/artifacts/full_run/T1_v5/hypodd/cluster3_event_ids.txt"
OUT_DIR = "full_catalog_pipeline/artifacts/full_run/T1_v5/hypodd/damp_sweep"
WORK_ROOT = catalog_paths.scratch_dir("hypodd_damp_sweep")

RELOC_COLS = [
    "id", "lat", "lon", "depth", "x", "y", "z", "ex", "ey", "ez",
    "yr", "mo", "dy", "hr", "mi", "sc", "mag",
    "nccp", "nccs", "nctp", "ncts", "rcc", "rct", "cid",
]

# Same NITER / WT_CC / WT_CT / MAXR schedule as the validated T1 config
# (~/dr/time/input_files/sweep_d150_hcc.inp, hypodd_relocate.py), only DAMP varies.
NITER = [6, 6, 6, 6, 8]
WT_CC = [0.99, 0.33, 0.10, 0.03, 0.02]
WT_CT = [0.60, 0.40, 0.20, 0.05, 0.05]
MAXR_CC = [-999, -999, 0.06, 0.04, 0.02]
MAXD_CC = [-999, -999, 50, 20, 10]
MAXR_CT = [10, 10, 3, -999, -999]
MAXD_CT = [100, 100, 20, -999, -999]


def filter_event_sel(ids, out_path):
    cols = ["date", "time", "lat", "lon", "depth", "mag", "eh", "ez", "rms", "id"]
    ev = pd.read_csv(f"{FULL_DIR}/event.sel", sep=r"\s+", header=None, names=cols)
    ev = ev[ev["id"].isin(ids)]
    ev.to_csv(out_path, sep=" ", header=False, index=False, float_format="%.5f")
    return len(ev)


def filter_dt(src_path, ids, out_path):
    ids = set(ids)
    n_pairs = 0
    n_obs = 0
    write_pair = False
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
                fout.write(line)
                n_obs += 1
    return n_pairs, n_obs


def fmt(v):
    return "-999" if v == -999 else str(v)


def write_inp(work_dir, damp):
    assert len(damp) == 5
    lines = [
        "hypoDD_2", "dt.cc", "dt.ct", "event.sel", "station.sel", "", "",
        "hypoDD.sta", "hypoDD.res", "hypoDD.src",
        "3 3 150", "3 4 1 -999 -999", "1 1 1 5",
    ]
    for i in range(5):
        lines.append(
            f"{NITER[i]} {WT_CC[i]} {WT_CC[i]} {fmt(MAXR_CC[i])} {fmt(MAXD_CC[i])} "
            f"{WT_CT[i]} {WT_CT[i]} {fmt(MAXR_CT[i])} {fmt(MAXD_CT[i])}  {damp[i]}"
        )
    lines += [
        "1",
        "0.0 0.1 3.1 3.8 9.5 14.0 25.0 52.0",
        "2.5 3.85 5.1 5.8 6.1 6.5 7.5 8.05",
        "1.84 2.22 2.95 3.35 3.53 3.76 4.34 4.65",
        "0", "",
    ]
    with open(f"{work_dir}/hypoDD.inp", "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--damp", type=float, nargs=5, required=True,
                     help="5 DAMP values, one per reweighting stage")
    ap.add_argument("--freeze-weights", action="store_true",
                     help="Hold WT_CC/WT_CT/MAXR/MAXD at stage-1 values across all 5 stages "
                          "(isolates whether the reweighting-schedule JUMP itself, not just "
                          "DAMP, destabilizes depth -- NITER per stage still applies, so this "
                          "is just more iterations at stage-1 settings, no schedule change).")
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    if args.freeze_weights:
        global WT_CC, WT_CT, MAXR_CC, MAXD_CC, MAXR_CT, MAXD_CT
        WT_CC = [WT_CC[0]] * 5
        WT_CT = [WT_CT[0]] * 5
        MAXR_CC = [MAXR_CC[0]] * 5
        MAXD_CC = [MAXD_CC[0]] * 5
        MAXR_CT = [MAXR_CT[0]] * 5
        MAXD_CT = [MAXD_CT[0]] * 5

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(IDS_FILE) as f:
        ids = [int(x) for x in f.read().split()]

    work_dir = f"{WORK_ROOT}/{args.label}"
    os.makedirs(work_dir, exist_ok=True)

    n_ev = filter_event_sel(ids, f"{work_dir}/event.sel")
    n_cc_pairs, n_cc_obs = filter_dt(f"{FULL_DIR}/dt.cc", ids, f"{work_dir}/dt.cc")
    n_ct_pairs, n_ct_obs = filter_dt(f"{FULL_DIR}/dt.ct", ids, f"{work_dir}/dt.ct")
    os.system(f"cp {FULL_DIR}/station.sel {work_dir}/station.sel")
    write_inp(work_dir, args.damp)

    print(f"[{args.label}] damp={args.damp} -- {n_ev} events (requested {len(ids)}), "
          f"{n_cc_pairs} cc pairs / {n_cc_obs} obs, {n_ct_pairs} ct pairs / {n_ct_obs} obs")

    result = subprocess.run(["/home/jwalter/bin/hypoDD_svd", "hypoDD.inp"], cwd=work_dir,
                             capture_output=True, timeout=10800)
    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    with open(f"{work_dir}/run.log", "w") as f:
        f.write(stdout + stderr)
    print(f"[{args.label}] hypoDD exit code: {result.returncode}")

    reloc_path = f"{work_dir}/hypoDD.reloc"
    src_path = f"{work_dir}/hypoDD.src"
    if os.path.exists(src_path) and os.path.getsize(src_path) > 0:
        os.system(f"cp {src_path} {OUT_DIR}/{args.label}_hypoDD.src")
    if os.path.exists(reloc_path) and os.path.getsize(reloc_path) > 0:
        os.system(f"cp {reloc_path} {OUT_DIR}/{args.label}_hypoDD.reloc")
        reloc = pd.read_csv(reloc_path, sep=r"\s+", header=None, names=RELOC_COLS)
        print(f"[{args.label}] relocated {len(reloc)}/{n_ev} events")
        print(f"[{args.label}] depth (km): median={reloc['depth'].median():.3f}, "
              f"std={reloc['depth'].std()*1000:.0f}m, "
              f"min={reloc['depth'].min():.3f}, max={reloc['depth'].max():.3f}, "
              f"range={( reloc['depth'].max()-reloc['depth'].min())*1000:.0f}m")
        print(f"[{args.label}] EZ (m): median={reloc['ez'].median():.1f}, "
              f"fraction zero={100*(reloc['ez']==0).mean():.1f}%")
    else:
        print(f"[{args.label}] no hypoDD.reloc produced (see {work_dir}/run.log)")

    # Per-iteration stability trace: mean |DZ| shift and any air-quake warnings
    import re
    dz_trace = re.findall(r"^\s*\d+\s+\S+\s+\d+\s+\d+\s+\d+\s+\d+\s+\S+\s+\d+\s+\S+\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)",
                            stdout, flags=re.M)
    n_air = stdout.count("air quake") + stdout.count("Number of air quakes")
    neg_depth = stdout.count("negative depth")
    print(f"[{args.label}] 'negative depth' warnings: {neg_depth}, air-quake mentions: {n_air}")


if __name__ == "__main__":
    main()
