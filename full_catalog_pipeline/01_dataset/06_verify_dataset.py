#!/usr/bin/env python3
"""Stage 6: read-only verification of the new dataset vs. the old curated one.

Reads: full_catalog_pipeline/output/, final_curated_seisbench_data/
Writes: full_catalog_pipeline/artifacts/sample_*.png
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


import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seisbench.data as sbd

import config

REPO_ROOT = os.path.dirname(config.PIPELINE_ROOT)
OLD_DATASET_PATH = os.path.join(REPO_ROOT, "final_curated_seisbench_data")


def check_split_leakage(df):
    counts = df.groupby("source_id")["split"].nunique()
    leaking = counts[counts > 1]
    if len(leaking) > 0:
        print(f"\nFAIL: {len(leaking)} events appear in more than one split:")
        print(leaking.head(20))
        return False
    print("\nOK: no event appears in more than one split.")
    return True


def compare_p_s_balance(new_df, old_df):
    print("\nP/S balance comparison:")
    for label, df in [("OLD", old_df), ("NEW", new_df)]:
        n_p = df["trace_p_arrival_sample"].notna().sum() if "trace_p_arrival_sample" in df else 0
        n_s = df["trace_s_arrival_sample"].notna().sum() if "trace_s_arrival_sample" in df else 0
        print(f"  {label}: {len(df)} traces, P={n_p} ({100*n_p/len(df):.1f}%), S={n_s} ({100*n_s/len(df):.1f}%)")


def snr_summary(df):
    if "trace_snr_db" not in df.columns or df["trace_snr_db"].isna().all():
        print("\nNo trace_snr_db populated.")
        return
    print("\nSNR summary (new dataset):")
    print(df["trace_snr_db"].describe())


def per_station_counts(df):
    print("\nPer-station-per-split counts (new dataset):")
    print(df.groupby(["station_code", "split"]).size().unstack(fill_value=0))


def save_sample_plots(dataset, n=5):
    colors = ["#344e41", "#588157", "#a3b18a"]
    components = ["Z (Vertical)", "N (North)", "E (East)"]
    idxs = np.random.RandomState(config.RANDOM_SEED).choice(len(dataset), size=min(n, len(dataset)), replace=False)

    for trace_idx in idxs:
        waveforms = dataset.get_waveforms(trace_idx)
        metadata = dataset.metadata.iloc[trace_idx]
        sr = metadata.get("trace_sampling_rate_hz", 200.0)
        npts = metadata.get("trace_npts", waveforms.shape[1])
        time_axis = np.arange(npts) / sr

        p_sample = metadata.get("trace_p_arrival_sample", np.nan)
        s_sample = metadata.get("trace_s_arrival_sample", np.nan)
        p_time = p_sample / sr if pd.notna(p_sample) else np.nan
        s_time = s_sample / sr if pd.notna(s_sample) else np.nan

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
        fig.suptitle(
            f"Event: {metadata.get('source_id')} | Station: {metadata.get('station_code')} | "
            f"SNR: {metadata.get('trace_snr_db', float('nan')):.1f} dB", fontsize=13)

        for i, (ax, comp, color) in enumerate(zip(axes, components, colors)):
            ax.plot(time_axis, waveforms[i], color=color, linewidth=1.2, label=comp)
            if pd.notna(p_time):
                ax.axvline(x=p_time, color="red", linestyle="--", label="P" if i == 0 else "_nolegend_")
            if pd.notna(s_time):
                ax.axvline(x=s_time, color="blue", linestyle="--", label="S" if i == 0 else "_nolegend_")
            ax.set_ylabel(comp, fontsize=9)
            ax.legend(loc="upper right")
            ax.grid(True, linestyle="--", alpha=0.5)
        axes[-1].set_xlabel("Time (s)")

        out_path = os.path.join(config.ARTIFACTS_DIR, f"sample_{trace_idx}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  saved {out_path}")


def main():
    if not os.path.exists(config.OUTPUT_METADATA_CSV):
        print(f"{config.OUTPUT_METADATA_CSV} not found — run 05_pack_seisbench.py first.")
        return

    new_dataset = sbd.WaveformDataset(config.OUTPUT_DIR)
    new_df = new_dataset.metadata
    print(f"New dataset: {len(new_df)} traces")

    ok = check_split_leakage(new_df)

    if os.path.exists(OLD_DATASET_PATH):
        old_dataset = sbd.WaveformDataset(OLD_DATASET_PATH)
        old_df = old_dataset.metadata
        compare_p_s_balance(new_df, old_df)
    else:
        print(f"\nOld dataset not found at {OLD_DATASET_PATH}, skipping comparison.")

    snr_summary(new_df)
    per_station_counts(new_df)

    print("\nSaving sample plots...")
    save_sample_plots(new_dataset)

    print("\n" + ("VERIFICATION PASSED" if ok else "VERIFICATION FAILED (see split leakage above)"))


if __name__ == "__main__":
    main()
