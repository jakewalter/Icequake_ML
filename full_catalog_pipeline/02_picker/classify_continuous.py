#!/usr/bin/env python3
"""Correct continuous-detection inference, bypassing easyQuake's
run_seisbench.py preprocessing (which force-decimates to 100 Hz and applies
a hardcoded 3-20 Hz bandpass tuned for crustal earthquakes -- both wrong for
a PhaseNet trained on native 200 Hz icequake windows with 20-80 Hz content).

Uses SeisBench's model.classify() directly with sampling_rate set to match
training (200 Hz) and no destructive filtering (matches the raw/peak-normalized
preprocessing used in training and evaluate.py).

Usage:
    python full_catalog_pipeline/classify_continuous.py \
        --station DRSC --date 2020-07-13 \
        --checkpoint full_catalog_pipeline/checkpoints/best_model_state_only.pth \
        --out-picks full_catalog_pipeline/artifacts/classify_new_model_drsc.out
"""
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import argparse
import glob
from datetime import datetime

import torch
from obspy import Stream, read

import seisbench.models as sbm

NETWORK = "7U"
DAY_VOLUMES_ROOT = "/data/time/day_volumes"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", required=True)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-picks", required=True)
    ap.add_argument("--sampling-rate", type=float, default=200.0)
    ap.add_argument("--p-threshold", type=float, default=0.5)
    ap.add_argument("--s-threshold", type=float, default=0.5)
    args = ap.parse_args()

    d = datetime.strptime(args.date, "%Y-%m-%d").date()
    yyyymmdd = d.strftime("%Y%m%d")
    doy = d.strftime("%j")
    src_dir = os.path.join(DAY_VOLUMES_ROOT, yyyymmdd)

    st = Stream()
    for chan in ("HHZ", "HH1", "HH2"):
        pattern = os.path.join(src_dir, f"{args.station}.{NETWORK}..{chan}.{d.year}.{doy}")
        matches = glob.glob(pattern)
        if not matches:
            raise FileNotFoundError(pattern)
        st += read(matches[0])
    st.detrend(type="linear")
    print(st)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = sbm.PhaseNet(phases="PSN", norm="peak")
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.sampling_rate = args.sampling_rate
    model.to(device)
    model.eval()

    output = model.classify(
        st,
        P_threshold=args.p_threshold,
        S_threshold=args.s_threshold,
        batch_size=256,
    )

    os.makedirs(os.path.dirname(args.out_picks), exist_ok=True)
    with open(args.out_picks, "w") as f:
        for pick in output.picks:
            f.write(f"{args.station} {args.station} {NETWORK} {pick.phase} {pick.peak_time}\n")
    print(f"Wrote {len(output.picks)} picks to {args.out_picks}")


if __name__ == "__main__":
    main()
