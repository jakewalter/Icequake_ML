#!/usr/bin/env python3
"""Stage 5: pack SNR-filtered windows into SeisBench format.

Reads: artifacts/windows.h5, artifacts/windows_filtered.csv
Writes: output/metadata.csv, output/waveforms.hdf5
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
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h5py
import numpy as np
import pandas as pd
import seisbench.data as sbd

import config
from lib.seisbench_pack import assign_splits_by_event, load_pick_methods


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(config.WINDOWS_FILTERED_CSV):
        print(f"{config.WINDOWS_FILTERED_CSV} not found — run 04_compute_snr_and_filter.py "
              "--snr-threshold-db <X> first.")
        return

    if not args.force and os.path.exists(config.OUTPUT_METADATA_CSV):
        print(f"{config.OUTPUT_METADATA_CSV} already exists, skipping (use --force to rebuild).")
        return

    df = pd.read_csv(config.WINDOWS_FILTERED_CSV)
    print(f"Packing {len(df)} SNR-filtered windows")

    splits = assign_splits_by_event(df["event_id"].tolist())
    methods = load_pick_methods()

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    written, failed = 0, 0
    per_split_counts = defaultdict(int)
    per_split_phase = defaultdict(lambda: defaultdict(int))
    per_split_station = defaultdict(lambda: defaultdict(int))

    with h5py.File(config.WINDOWS_H5, "r") as h5f, \
         sbd.WaveformDataWriter(config.OUTPUT_METADATA_CSV, config.OUTPUT_WAVEFORMS_HDF5) as writer:

        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
            "measurement": "velocity",
            "unit": "counts",
            "instrument_response": "not restituted",
        }

        for i, row in enumerate(df.itertuples(index=False)):
            key = f"{row.event_id}__{row.station}"
            if key not in h5f:
                failed += 1
                continue

            grp = h5f[key]
            # Handle minor length mismatches (e.g., 2001 vs 2002 samples) by trimming to min
            z_data = grp["Z"][:]
            n_data = grp["N"][:]
            e_data = grp["E"][:]
            min_len = min(len(z_data), len(n_data), len(e_data))
            data_3c = np.vstack([z_data[:min_len], n_data[:min_len], e_data[:min_len]])
            split = splits[row.event_id]

            trace_metadata = {
                "trace_name_original": f"{row.network}.{row.station}.{row.window_start}",
                "station_network_code": row.network,
                "station_code": row.station,
                "trace_channel": "HH",
                "trace_sampling_rate_hz": config.SAMPLE_RATE_HZ,
                "trace_npts": data_3c.shape[1],
                "trace_start_time": row.window_start,
                "source_id": row.event_id,
                "split": split,
                "trace_snr_db": row.trace_snr_db,
            }

            p_sample = row.p_arrival_sample if pd.notna(row.p_arrival_sample) else None
            s_sample = row.s_arrival_sample if pd.notna(row.s_arrival_sample) else None

            if p_sample is not None:
                trace_metadata["trace_p_arrival_sample"] = int(p_sample)
                trace_metadata["trace_p_status"] = methods.get((row.event_id, row.station, "P"), "unknown")
                trace_metadata["trace_p_weight"] = 1.0

            if s_sample is not None:
                trace_metadata["trace_s_arrival_sample"] = int(s_sample)
                trace_metadata["trace_s_status"] = methods.get((row.event_id, row.station, "S"), "unknown")
                trace_metadata["trace_s_weight"] = 1.0

            try:
                writer.add_trace(trace_metadata, data_3c)
                written += 1
                per_split_counts[split] += 1
                if p_sample is not None:
                    per_split_phase[split]["P"] += 1
                if s_sample is not None:
                    per_split_phase[split]["S"] += 1
                per_split_station[split][row.station] += 1
            except Exception as e:
                print(f"Failed writing {row.station}/{row.event_id}: {e}")
                failed += 1

            if (i + 1) % 2000 == 0:
                print(f"  processed {i + 1}/{len(df)}")

    print(f"\nWritten: {written}, failed: {failed}")
    print("\nPer-split counts:")
    for split, count in sorted(per_split_counts.items()):
        print(f"  {split}: {count}")

    print("\nPer-split P/S coverage:")
    for split in sorted(per_split_phase.keys()):
        print(f"  {split}: P={per_split_phase[split]['P']} S={per_split_phase[split]['S']}")

    print("\nPer-split per-station counts:")
    for split in sorted(per_split_station.keys()):
        print(f"  {split}: " + ", ".join(f"{s}={c}" for s, c in sorted(per_split_station[split].items())))

    print(f"\nWrote {config.OUTPUT_METADATA_CSV}")
    print(f"Wrote {config.OUTPUT_WAVEFORMS_HDF5}")


if __name__ == "__main__":
    main()
