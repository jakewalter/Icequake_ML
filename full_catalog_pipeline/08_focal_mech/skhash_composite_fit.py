#!/usr/bin/env python3
"""Per user request/correction: fit ONE composite/aggregate focal mechanism per cluster,
using each station's stack-derived dominant polarity (cluster_p_stack_polarity.py's
proven, high-SNR CC-aligned stack -- see [[cluster_p_stack_polarity_result]]) as that
cluster's single set of polarity observations, fed through real SKHASH.

This is NOT per-event SKHASH (skhash_cluster_fit.py, which fits one mechanism per
individual event and already found every computable per-event mechanism graded quality D
-- see [[skhash_cluster_fit_integration]]). Here the whole cluster collapses to ONE
pseudo-"event" located at the cluster centroid, with one polarity reading per station
(from that station's stack, not any single event) -- the same aggregation logic as this
pipeline's existing hand-rolled composite grid search (focal_mech_cluster3_fit_rpnet.py),
but (a) fed stack-derived polarities instead of RPNet/heuristic per-event picks, pooled
into one reading per station rather than kept as separate per-event observations, and
(b) fit via the real HASH algorithm (proper takeoff-angle ray tracing + bootstrap +
standardized quality grade) instead of a hand-rolled grid search.

Because this literally has only as many polarity observations as there are usable
stations (5-7 for this deployment), it is the cleanest possible test of whether THIS
station geometry can ever resolve a mechanism, using the most reliable polarity reading
this pipeline can produce (SNR up to 300+) -- isolating geometry/coverage from picker
noise as the limiting factor.

Usage:
    python full_catalog_pipeline/skhash_composite_fit.py --label cluster3
Must run in the `rpnet` conda env (SKHASH installed there):
    conda activate rpnet
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
import sys

import numpy as np
import pandas as pd
from obspy import UTCDateTime

sys.path.insert(0, ".")
from rpnet.rpnet2skhash import prep_skhash
from skhash_cluster_fit import (
    write_station_file_fixed, load_station_sel, CONTROL_TEMPLATE,
    VMODEL_DEPTH_KM, VMODEL_VP_KMS, STATION_SEL, NETWORK,
)

import catalog_paths

HYPODD_DIR = catalog_paths.work_dir("T1")
RELOC_COLS = ["id", "lat", "lon", "depth", "x", "y", "z", "ex", "ey", "ez",
              "yr", "mo", "dy", "hr", "mi", "sc", "mag",
              "nccp", "nccs", "nctp", "ncts", "rcc", "rct", "cid"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="cluster3")
    ap.add_argument("--stack-polarities-csv", default=None,
                    help=f"Defaults to {HYPODD_DIR}/<label>_stack_polarity/<label>_stack_polarities.csv "
                         "(cluster_p_stack_polarity.py's per-station dominant-polarity output).")
    ap.add_argument("--ids-file", default=None, help=f"Defaults to {HYPODD_DIR}/<label>_event_ids.txt")
    ap.add_argument("--reloc-file", default=f"{HYPODD_DIR}/output_files/hypoDD.reloc",
                    help="Source of event lat/lon/depth/origin-time, used only to compute the "
                         "cluster centroid location for this one composite pseudo-event.")
    ap.add_argument("--out-dir", default=None, help=f"Defaults to {HYPODD_DIR}/<label>_skhash_composite")
    ap.add_argument("--npolmin", type=int, default=3,
                    help="SKHASH default is 8; this deployment has at most 5-7 stations total per cluster.")
    ap.add_argument("--dang", type=int, default=2, help="Grid-search spacing in degrees.")
    ap.add_argument("--delmax", type=float, default=10.0, help="Max source-receiver distance (km).")
    args = ap.parse_args()
    if args.stack_polarities_csv is None:
        args.stack_polarities_csv = f"{HYPODD_DIR}/{args.label}_stack_polarity/{args.label}_stack_polarities.csv"
    if args.ids_file is None:
        args.ids_file = f"{HYPODD_DIR}/{args.label}_event_ids.txt"
    if args.out_dir is None:
        args.out_dir = f"{HYPODD_DIR}/{args.label}_skhash_composite"
    return args


def cluster_centroid(ids_file, reloc_file):
    ids = set(int(x) for x in open(ids_file))
    reloc = pd.read_csv(reloc_file, sep=r"\s+", header=None, names=RELOC_COLS)
    reloc = reloc[reloc["id"].isin(ids)]
    if len(reloc) == 0:
        raise ValueError(f"no reloc rows matched any id in {ids_file}")
    row0 = reloc.iloc[0]
    origin = str(UTCDateTime(int(row0.yr), int(row0.mo), int(row0.dy),
                              int(row0.hr), int(row0.mi), float(row0.sc)))
    return dict(lat=reloc["lat"].mean(), lon=reloc["lon"].mean(), dep=reloc["depth"].mean(),
                n_events=len(reloc), origin=origin)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    stack_pol = pd.read_csv(args.stack_polarities_csv)
    stack_pol = stack_pol[stack_pol["polarity"] != 0].copy()
    print(f"[{args.label}] station-level stack polarities (from {args.stack_polarities_csv}):")
    print(stack_pol[["station", "n", "polarity", "snr"]].to_string(index=False))

    centroid = cluster_centroid(args.ids_file, args.reloc_file)
    print(f"\n[{args.label}] composite pseudo-event: centroid over {centroid['n_events']} events, "
          f"lat={centroid['lat']:.4f} lon={centroid['lon']:.4f} depth={centroid['dep']:.2f}km")

    data_id = f"composite_{args.label}"
    cat_df = pd.DataFrame({"data_id": [data_id], "jst": [centroid["origin"]],
                           "lat": [centroid["lat"]], "lon": [centroid["lon"]],
                           "dep": [centroid["dep"]], "mag": [0.0]})
    pol = pd.DataFrame({"data_id": [data_id] * len(stack_pol),
                        "sta": stack_pol["station"].to_list(),
                        "predict": stack_pol["polarity"].map({1: "U", -1: "D"}).to_list()})
    print(f"\n[{args.label}] composite polarity observations: {len(pol)} "
          f"(up={ (pol['predict']=='U').sum() }, down={ (pol['predict']=='D').sum() })")

    sta_df = load_station_sel(STATION_SEL, network=NETWORK)

    vmodel_path = os.path.abspath(f"{args.out_dir}/vmodel.csv")
    with open(vmodel_path, "w") as f:
        f.write("# Depth (km), Vp (km/s)\n")
        for d, vp in zip(VMODEL_DEPTH_KM, VMODEL_VP_KMS):
            f.write(f"{d}, {vp}\n")

    ctrl0_path = os.path.abspath(f"{args.out_dir}/control_base.txt")
    with open(ctrl0_path, "w") as f:
        f.write(f"$vmodel_paths  # velocity model\n{vmodel_path}\n\n")
        f.write(CONTROL_TEMPLATE.format(npolmin=args.npolmin, dang=args.dang, delmax=args.delmax))

    out_dir_abs = os.path.abspath(args.out_dir)
    prep_skhash(cat_df=cat_df, pol_df=pol, amp=[], sta_df=sta_df,
                ftime="jst", fwfid="data_id", ctrl0=ctrl0_path,
                out_dir=out_dir_abs, hash_version="hash2")
    write_station_file_fixed(sta_df, f"{out_dir_abs}/hash2/IN/station.txt")

    control_file = f"{out_dir_abs}/hash2/control_file.txt"
    print(f"\n[{args.label}] wrote SKHASH input bundle under {out_dir_abs}/hash2, running SKHASH...")
    result = subprocess.run(["SKHASH", control_file], capture_output=True, text=True)
    print(result.stdout[-4000:])
    if result.returncode != 0:
        print("STDERR:", result.stderr[-4000:])
        print(f"[{args.label}] SKHASH exited with code {result.returncode}")
        return

    out1 = f"{out_dir_abs}/hash2/OUT/out.csv"
    if os.path.exists(out1):
        res = pd.read_csv(out1)
        print(f"\n[{args.label}] SKHASH produced {len(res)} composite mechanism(s)")
        if len(res):
            print(res.to_string())
    else:
        print(f"[{args.label}] no {out1} produced -- check SKHASH output above for why.")


if __name__ == "__main__":
    main()
