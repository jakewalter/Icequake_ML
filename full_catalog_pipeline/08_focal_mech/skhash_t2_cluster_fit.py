#!/usr/bin/env python3
"""T2 port of skhash_cluster_fit.py: feed RPNet P-polarity picks
(rpnet_t2_cluster_polarities.py) into SKHASH for real per-event focal
mechanisms on a T2 basal cluster. Solves ONE mechanism PER EVENT (SKHASH/HASH
convention) -- with only 7 total stations, most individual T2 events likely
won't reach --npolmin either (same premise test as T1, see
t1_composite_focal_mech_result), which is itself the point: an independent,
tool-based check of whether per-event mechanisms are recoverable here at all.

Must run in the `rpnet` conda env (SKHASH installed there alongside RPNet):

    conda activate rpnet
    python full_catalog_pipeline/skhash_t2_cluster_fit.py --label t2_cluster0
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

import catalog_paths

HYPODD_DIR = catalog_paths.work_dir("T2")
STATION_SEL = f"{HYPODD_DIR}/input_files/station.sel"
NETWORK = "7U"

RELOC_COLS = ["id", "lat", "lon", "depth", "x", "y", "z", "ex", "ey", "ez",
              "yr", "mo", "dy", "hr", "mi", "sc", "mag",
              "nccp", "nccs", "nctp", "ncts", "rcc", "rct", "cid"]

# T2's own accepted 1-D P velocity model (km, km/s), depth-shifted for T2's
# shallower ~2.02km ice-bed interface (hypodd_relocate.py ARRAY_CONFIG["T2"]).
VMODEL_DEPTH_KM = [0.0, 0.1, 2.02, 2.72, 8.42, 12.92, 23.92, 50.92]
VMODEL_VP_KMS = [2.5, 3.841, 5.1, 5.8, 6.1, 6.5, 7.5, 8.05]

CONTROL_TEMPLATE = """$npolmin       # minimum number of polarity picks per event
{npolmin}

$max_agap      # maximum azimuthal gap
360

$max_pgap      # maximum "plungal" gap
90

$dang          # minimum grid spacing (degrees)
{dang}

$nmc           # number of trials (e.g., 30)
30

$maxout        # max num of acceptable focal mech. outputs (e.g., 500)
500

$badfrac       # fraction polarities assumed bad
0.1

$delmax        # maximum allowed source-receiver distance in km.
{delmax}

$cangle        # angle for computing mechanisms probability
45

$prob_max      # probability threshold for multiples (e.g., 0.1)
0.25

$num_cpus      # number of cores in parallel (0: use all cpu / 1: single core)
1

$use_fortran   # Fortran subroutine for fast grid search
False
"""


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default="t2_cluster0")
    ap.add_argument("--polarities-csv", default=None,
                    help=f"Defaults to {HYPODD_DIR}/<label>_polarities_rpnet.csv")
    ap.add_argument("--reloc-file", default=f"{HYPODD_DIR}/output_files/hypoDD.reloc",
                    help="Event lat/lon/depth/origin-time source -- main full-catalog hypoDD.reloc, "
                         "not a per-cluster SVD reloc: SKHASH computes its own takeoff angles.")
    ap.add_argument("--out-dir", default=None,
                    help=f"Defaults to {HYPODD_DIR}/<label>_skhash")
    ap.add_argument("--npolmin", type=int, default=5,
                    help="SKHASH default is 8; lowered since this deployment has only 7 stations total.")
    ap.add_argument("--dang", type=int, default=2, help="Grid-search spacing in degrees.")
    ap.add_argument("--delmax", type=float, default=10.0, help="Max source-receiver distance (km).")
    args = ap.parse_args()
    if args.polarities_csv is None:
        args.polarities_csv = f"{HYPODD_DIR}/{args.label}_polarities_rpnet.csv"
    if args.out_dir is None:
        args.out_dir = f"{HYPODD_DIR}/{args.label}_skhash"
    return args


def load_station_sel(path, network=NETWORK):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["net_sta", "lat", "lon", "elv"])
    df["sta"] = df["net_sta"].str.split(".").str[-1]
    df["sta0"] = df["sta"]
    df["net"] = network
    df["chan"] = "HHZ"
    return df[["sta", "sta0", "net", "chan", "lat", "lon", "elv"]]


def write_station_file_fixed(sta_df, path):
    """RPNet's own rpnet2skhash.prep_skhash() station.txt writer is
    off-by-one against the installed SKHASH 1.1.5's actual fixed-width
    colspecs -- see skhash_cluster_fit.py's identical fix for T1."""
    lines = []
    for _, row in sta_df.sort_values("sta").iterrows():
        buf = [" "] * 92

        def put(s, start):
            for i, c in enumerate(s):
                buf[start + i] = c

        put(str(row["sta"])[:4].ljust(4), 0)
        put(str(row["chan"])[:3].ljust(3), 5)
        put(f"{row['lat']:9.5f}", 41)
        put(f"{row['lon']:10.5f}", 51)
        put(f"{int(row['elv']):>5d}", 62)
        put("1900/01/01", 68)
        put("3000/01/01", 79)
        put(str(row["net"])[:2].ljust(2), 90)
        lines.append("".join(buf))
    with open(path, "w") as f:
        f.write("\n".join(lines))


def _origin_time(r):
    # hypoDD can round the seconds field up to 60.0 (same overflow hypoddpy's own
    # _create_output_event_file() guards against) -- wrap to the next minute instead
    # of letting UTCDateTime raise on an out-of-range seconds value.
    sec = float(r.sc)
    add_minute = sec >= 60
    if add_minute:
        sec = 0.0
    t = UTCDateTime(int(r.yr), int(r.mo), int(r.dy), int(r.hr), int(r.mi), int(sec),
                    int((sec % 1.0) * 1e6))
    if add_minute:
        t += 60.0
    return str(t)


def load_cat_df(reloc_file):
    reloc = pd.read_csv(reloc_file, sep=r"\s+", header=None, names=RELOC_COLS)
    origins = reloc.apply(_origin_time, axis=1)
    cat_df = pd.DataFrame({
        "data_id": reloc["id"].astype(str),
        "jst": origins,
        "lat": reloc["lat"], "lon": reloc["lon"], "dep": reloc["depth"],
        "mag": reloc["mag"],
    })
    return cat_df


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pol = pd.read_csv(args.polarities_csv)
    pol = pol[pol["polarity"] != 0].copy()
    pol["predict"] = pol["polarity"].map({1: "U", -1: "D"})
    pol["data_id"] = pol["id"].astype(str)
    pol = pol.rename(columns={"station": "sta"})[["data_id", "sta", "predict"]]

    cat_df = load_cat_df(args.reloc_file)
    cat_df = cat_df[cat_df["data_id"].isin(pol["data_id"])].reset_index(drop=True)
    sta_df = load_station_sel(STATION_SEL)

    print(f"[{args.label}] events with a reloc/lat-lon-depth match: {len(cat_df)}")
    print(f"[{args.label}] polarity rows: {len(pol)}")
    n_per_event = pol.groupby("data_id").size()
    print(f"[{args.label}] picks/event: median={n_per_event.median():.0f}, "
          f"max={n_per_event.max()}, events with >={args.npolmin} picks: "
          f"{(n_per_event >= args.npolmin).sum()}/{len(n_per_event)}")

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
    print(f"[{args.label}] wrote SKHASH input bundle under {out_dir_abs}/hash2, running SKHASH...")
    result = subprocess.run(["SKHASH", control_file], capture_output=True, text=True)
    print(result.stdout[-4000:])
    if result.returncode != 0:
        print("STDERR:", result.stderr[-4000:])
        print(f"[{args.label}] SKHASH exited with code {result.returncode}")
        return

    out1 = f"{out_dir_abs}/hash2/OUT/out.csv"
    if os.path.exists(out1):
        res = pd.read_csv(out1)
        print(f"\n[{args.label}] SKHASH produced {len(res)} event mechanism(s)")
        if len(res):
            print(res.to_string())
            if "quality" in res.columns:
                print("\nquality grade counts:")
                print(res["quality"].value_counts())
    else:
        print(f"[{args.label}] no {out1} produced -- check SKHASH output above for why.")


if __name__ == "__main__":
    main()
