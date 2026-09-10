#!/usr/bin/env python3
"""Run easyQuake continuous detection with a given SeisBench PhaseNet checkpoint
on one real day of T1/T2 continuous archive data, for a real in-domain
deployment test (as opposed to the WILZ/O2 demo notebook, which is an
out-of-network sanity check).

Usage:
    python full_catalog_pipeline/deploy_continuous_detection.py \
        --station DRSC --date 2020-07-13 \
        --checkpoint full_catalog_pipeline/checkpoints/best_model_state_only.pth \
        --project-dir easyquake_project/20200713_drsc_new
"""
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import argparse
import glob
import sys
from datetime import datetime

sys.path.insert(0, '/home/jwalter/easyQuake')

NETWORK = "7U"
DAY_VOLUMES_ROOT = "/data/time/day_volumes"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", required=True)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--project-dir", required=True, help="full path to per-run project subdirectory")
    args = ap.parse_args()

    d = datetime.strptime(args.date, "%Y-%m-%d").date()
    yyyymmdd = d.strftime("%Y%m%d")
    doy = d.strftime("%j")
    src_dir = os.path.join(DAY_VOLUMES_ROOT, yyyymmdd)

    os.makedirs(args.project_dir, exist_ok=True)

    from obspy import read
    for chan in ("HHZ", "HH1", "HH2"):
        pattern = os.path.join(src_dir, f"{args.station}.{NETWORK}..{chan}.{d.year}.{doy}")
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(pattern)
        st = read(matches[0])
        out_path = os.path.join(args.project_dir, f"{NETWORK}.{args.station}.{chan}.mseed")
        st.write(out_path, format="MSEED")
        print(f"Wrote {out_path} ({st[0].stats.npts} samples)")

    project_folder = os.path.dirname(args.project_dir)
    dirname = os.path.basename(args.project_dir)

    infile = os.path.join(args.project_dir, "dayfile.in")
    outfile = os.path.join(args.project_dir, "seisbench_picks.out")
    sys.argv = [sys.argv[0], "-I", infile, "-O", outfile, "-M", args.checkpoint]

    from easyQuake import detection_continuous
    print(f"Running easyQuake Seisbench continuous detection: {args.checkpoint}")
    detection_continuous(
        dirname=dirname,
        project_folder=project_folder,
        project_code="icequake",
        single_date=d,
        machine=True,
        machine_picker="Seisbench",
        seisbenchmodel=args.checkpoint,
    )
    print("Detection complete.")
    outfile = os.path.join(args.project_dir, "seisbench_picks.out")
    if os.path.exists(outfile):
        n = sum(1 for _ in open(outfile))
        print(f"{outfile}: {n} lines")
    else:
        print(f"WARNING: {outfile} not created")


if __name__ == "__main__":
    main()
