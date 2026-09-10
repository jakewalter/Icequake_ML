#!/usr/bin/env python3
"""T2 port of hypodd_svd_cluster_errors.py: rerun hypoDD in SVD mode (ISOLV=1)
on a T2 basal-cluster event subset (from plot_hypodd_t2_basal_clusters.py) to
get real per-event formal errors (EX/EY/EZ), avoiding the full-catalog LSQR
run's EZ=0 failure mode for poorly-converging clusters. SVD is O(n^3) --
practical for small clusters, not the whole T2 catalog.

Usage:
    python full_catalog_pipeline/hypodd_svd_cluster_errors_t2.py --ids-file <path> --label <name>
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
import subprocess

import pandas as pd

import catalog_paths

# Paths come from catalog_paths so the authoritative relocation is named in ONE place
# (see that module). Set ICEQUAKE_RELOC=hypodd to reproduce a figure against the
# superseded v5 relocation without editing anything.
FULL_DIR = catalog_paths.input_dir("T2")
OUT_DIR = catalog_paths.work_dir("T2")
WORK_ROOT = os.environ.get(
    "HYPODD_SVD_WORK",
    os.path.join(os.environ.get("TMPDIR", "/tmp"), "hypodd_svd_t2"))

RELOC_COLS = [
    "id", "lat", "lon", "depth", "x", "y", "z", "ex", "ey", "ez",
    "yr", "mo", "dy", "hr", "mi", "sc", "mag",
    "nccp", "nccs", "nctp", "ncts", "rcc", "rct", "cid",
]


def filter_event_sel(ids, out_path):
    cols = ["date", "time", "lat", "lon", "depth", "mag", "eh", "ez", "rms", "id"]
    ev = pd.read_csv(f"{FULL_DIR}/event.sel", sep=r"\s+", header=None, names=cols)
    ev = ev[ev["id"].isin(ids)]
    ev.to_csv(out_path, sep=" ", header=False, index=False,
              float_format="%.5f")
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


def write_inp(work_dir):
    # Same ccscale_0.33 reweighting schedule as T1/T2's official relocations
    # (hypodd_relocate.py's REWEIGHTING_SCHEME); velocity model is T2's own
    # BedMachine/Bedmap2-shifted layers (ARRAY_CONFIG["T2"]).
    inp = """hypoDD_2
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
0.0 0.1 2.02 2.72 8.42 12.92 23.92 50.92
2.5 3.841 5.1 5.8 6.1 6.5 7.5 8.05
1.84 2.2202 2.948 3.3526 3.526 3.7572 4.3353 4.6532
0
"""
    with open(f"{work_dir}/hypoDD.inp", "w") as f:
        f.write(inp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--timeout", type=int, default=7200)
    # SVD is O(n^3) and the binary has a compiled MAXEVE0 ceiling, so large clusters must be
    # subsampled. A random subset preserves the cluster's geometry and station coverage; the
    # resulting EZ describes that geometry, which is what the resolvability question asks.
    ap.add_argument("--max-events", type=int, default=150)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    with open(args.ids_file) as f:
        ids = [int(x) for x in f.read().split()]
    n_full = len(ids)
    if args.max_events and len(ids) > args.max_events:
        import random
        ids = sorted(random.Random(args.seed).sample(ids, args.max_events))
        print(f"[{args.label}] subsampled {n_full} -> {len(ids)} events (seed {args.seed})")

    work_dir = f"{WORK_ROOT}/{args.label}"
    os.makedirs(work_dir, exist_ok=True)

    n_ev = filter_event_sel(ids, f"{work_dir}/event.sel")
    n_cc_pairs, n_cc_obs = filter_dt(f"{FULL_DIR}/dt.cc", ids, f"{work_dir}/dt.cc")
    n_ct_pairs, n_ct_obs = filter_dt(f"{FULL_DIR}/dt.ct", ids, f"{work_dir}/dt.ct")
    os.system(f"cp {FULL_DIR}/station.sel {work_dir}/station.sel")
    write_inp(work_dir)

    print(f"[{args.label}] {n_ev} events (requested {len(ids)}), "
          f"{n_cc_pairs} cc pairs / {n_cc_obs} obs, {n_ct_pairs} ct pairs / {n_ct_obs} obs")

    result = subprocess.run(["/home/jwalter/bin/hypoDD_svd", "hypoDD.inp"], cwd=work_dir,
                             capture_output=True, timeout=args.timeout)
    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")
    with open(f"{work_dir}/run.log", "w") as f:
        f.write(stdout + stderr)
    result.stdout, result.stderr = stdout, stderr
    print(f"[{args.label}] hypoDD exit code: {result.returncode}")
    if "Increase MAXEVE0" in result.stdout or "Increase MAXEVE0" in result.stderr:
        print(f"[{args.label}] FAILED: compiled binary's MAXEVE0 SVD limit too small, need to recompile")
        return
    if "FATAL" in result.stdout or "FATAL" in result.stderr:
        print(f"[{args.label}] FAILED: FATAL error, see {work_dir}/run.log")
        return

    reloc_path = f"{work_dir}/hypoDD.reloc"
    src_path = f"{work_dir}/hypoDD.src"
    if os.path.exists(src_path):
        dst_src = f"{OUT_DIR}/{args.label}_svd_hypoDD.src"
        os.system(f"cp {src_path} {dst_src}")
        print(f"[{args.label}] copied {src_path} -> {dst_src}")
    if os.path.exists(reloc_path):
        dst_reloc = f"{OUT_DIR}/{args.label}_svd_hypoDD.reloc"
        os.system(f"cp {reloc_path} {dst_reloc}")
        print(f"[{args.label}] copied {reloc_path} -> {dst_reloc}")

    if not os.path.exists(reloc_path):
        print(f"[{args.label}] no hypoDD.reloc produced, check {work_dir}/run.log")
        return
    reloc = pd.read_csv(reloc_path, sep=r"\s+", header=None, names=RELOC_COLS)
    print(f"[{args.label}] relocated {len(reloc)}/{n_ev} events")
    print(f"[{args.label}] EZ (vertical error, m): median={reloc['ez'].median():.1f}, "
          f"mean={reloc['ez'].mean():.1f}, p90={reloc['ez'].quantile(0.9):.1f}, "
          f"fraction zero={100*(reloc['ez']==0).mean():.1f}%")
    print(f"[{args.label}] EX (east error, m): median={reloc['ex'].median():.1f}")
    print(f"[{args.label}] EY (north error, m): median={reloc['ey'].median():.1f}")
    print(f"[{args.label}] actual depth std within cluster: {reloc['depth'].std()*1000:.0f} m")


if __name__ == "__main__":
    main()
