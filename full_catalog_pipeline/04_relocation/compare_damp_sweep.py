#!/usr/bin/env python3
"""Compare depth distributions across hypodd_damp_sweep.py configs against the
baseline (current, cluster3_svd_hypoDD.reloc) run.

Usage: python full_catalog_pipeline/compare_damp_sweep.py
"""
import glob
import os

import pandas as pd

BASE_DIR = "full_catalog_pipeline/artifacts/full_run/T1_v5/hypodd"
SWEEP_DIR = f"{BASE_DIR}/damp_sweep"
ICE_BED_DEPTH_KM = 3.24

RELOC_COLS = [
    "id", "lat", "lon", "depth", "x", "y", "z", "ex", "ey", "ez",
    "yr", "mo", "dy", "hr", "mi", "sc", "mag",
    "nccp", "nccs", "nctp", "ncts", "rcc", "rct", "cid",
]


def summarize(path, label):
    reloc = pd.read_csv(path, sep=r"\s+", header=None, names=RELOC_COLS)
    above_bed_m = (ICE_BED_DEPTH_KM - reloc["depth"]) * 1000
    print(f"{label:20s} n={len(reloc):4d}  depth_med={reloc['depth'].median():.3f}km  "
          f"std={reloc['depth'].std()*1000:6.0f}m  range={(reloc['depth'].max()-reloc['depth'].min())*1000:6.0f}m  "
          f"above_bed_med={above_bed_m.median():6.0f}m  above_bed_p10={above_bed_m.quantile(0.1):6.0f}m  "
          f"EZ_med={reloc['ez'].median():5.1f}m  EZ_zero%={100*(reloc['ez']==0).mean():4.1f}")


def main():
    summarize(f"{BASE_DIR}/cluster3_svd_hypoDD.reloc", "baseline(40/30/15/8/5)")
    for path in sorted(glob.glob(f"{SWEEP_DIR}/*_hypoDD.reloc")):
        label = os.path.basename(path).replace("_hypoDD.reloc", "")
        summarize(path, label)


if __name__ == "__main__":
    main()
