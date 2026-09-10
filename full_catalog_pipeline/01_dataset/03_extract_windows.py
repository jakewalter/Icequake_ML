#!/usr/bin/env python3
"""Stage 3: window extraction with a rolling day-file cache and midnight-merge
fallback.

Reads: artifacts/day_file_index.csv, artifacts/covered_station_events.csv,
       raw day-volume files
Writes: artifacts/windows.h5, artifacts/windows_metadata.csv
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
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h5py
import numpy as np
from obspy import UTCDateTime

import config
from lib.day_volume_index import load_day_file_index_csv
from lib.windowing import RollingDayCache, extract_window

METADATA_COLUMNS = [
    "event_id", "station", "network", "array", "window_start", "window_end",
    "npts_z", "npts_n", "npts_e", "complete", "p_arrival_sample", "s_arrival_sample",
    "extraction_status",
]


def load_covered_rows():
    by_station = defaultdict(lambda: defaultdict(list))
    with open(config.COVERED_STATION_EVENTS_CSV) as f:
        for row in csv.DictReader(f):
            by_station[row["station"]][row["primary_date"]].append(row)
    return by_station


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit-events", type=int, default=None,
                         help="Process only the first N (event,station) rows per station (for a quick smoke test).")
    args = parser.parse_args()

    if not args.force and os.path.exists(config.WINDOWS_METADATA_CSV):
        print(f"{config.WINDOWS_METADATA_CSV} already exists, skipping (use --force to rebuild).")
        return

    print("Loading day-file index...")
    day_index = load_day_file_index_csv(config.DAY_FILE_INDEX_CSV)

    print("Loading covered station-events, grouped by station/date...")
    by_station = load_covered_rows()
    total_rows = sum(len(rows) for dates in by_station.values() for rows in dates.values())
    print(f"{len(by_station)} stations, {total_rows} candidate windows")

    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    status_counts = defaultdict(int)
    array_counts = defaultdict(lambda: defaultdict(int))

    with h5py.File(config.WINDOWS_H5, "w") as h5f, \
         open(config.WINDOWS_METADATA_CSV, "w", newline="") as meta_f:

        writer = csv.DictWriter(meta_f, fieldnames=METADATA_COLUMNS)
        writer.writeheader()

        for station in sorted(by_station.keys()):
            dates_for_station = by_station[station]
            cache = RollingDayCache(day_index, station)
            n_done = 0

            for date_iso in sorted(dates_for_station.keys()):
                cache.ensure_window(date_iso)
                rows = dates_for_station[date_iso]

                for row in rows:
                    if args.limit_events is not None and n_done >= args.limit_events:
                        break
                    n_done += 1

                    window_start = UTCDateTime(row["window_start"])
                    window_end = UTCDateTime(row["window_end"])
                    result, status = extract_window(cache, row, window_start, window_end)
                    status_counts[status] += 1
                    array_counts[row["array"]][status] += 1

                    p_sample = ""
                    s_sample = ""
                    if row["p_time"]:
                        p_sample = int(round((UTCDateTime(row["p_time"]) - window_start) * config.SAMPLE_RATE_HZ))
                    if row["s_time"]:
                        s_sample = int(round((UTCDateTime(row["s_time"]) - window_start) * config.SAMPLE_RATE_HZ))

                    if result is not None:
                        key = f"{row['event_id']}__{row['station']}"
                        grp = h5f.create_group(key)
                        grp.create_dataset("Z", data=result["Z"], compression="gzip", compression_opts=4)
                        grp.create_dataset("N", data=result["N"], compression="gzip", compression_opts=4)
                        grp.create_dataset("E", data=result["E"], compression="gzip", compression_opts=4)
                        grp.attrs["window_start"] = str(window_start)
                        grp.attrs["sample_rate"] = config.SAMPLE_RATE_HZ
                        grp.attrs["array"] = row["array"]

                        writer.writerow({
                            "event_id": row["event_id"], "station": row["station"],
                            "network": row["network"], "array": row["array"],
                            "window_start": str(window_start), "window_end": str(window_end),
                            "npts_z": result["npts"]["HHZ"], "npts_n": result["npts"]["HH2"],
                            "npts_e": result["npts"]["HH1"], "complete": result["complete"],
                            "p_arrival_sample": p_sample, "s_arrival_sample": s_sample,
                            "extraction_status": status,
                        })

                cache.evict_before(date_iso)

                if args.limit_events is not None and n_done >= args.limit_events:
                    break

            print(f"  {station}: {n_done} candidates processed")

    print("\nExtraction status summary:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    print("\nPer-array status breakdown:")
    for array, counts in sorted(array_counts.items()):
        print(f"  {array}: " + ", ".join(f"{s}={c}" for s, c in sorted(counts.items())))

    print(f"\nWrote {config.WINDOWS_H5}")
    print(f"Wrote {config.WINDOWS_METADATA_CSV}")


if __name__ == "__main__":
    main()
