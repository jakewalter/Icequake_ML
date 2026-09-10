#!/usr/bin/env python3
"""Stage 2: continuous-data coverage index (filesystem-only, no obspy).

Reads: config.DAY_VOLUMES_ROOT, artifacts/picks_deduped.csv
Writes: artifacts/day_file_index.csv, artifacts/covered_station_events.csv,
        artifacts/uncovered_station_events.csv
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

import config
from lib.day_volume_index import build_day_file_index, has_all_channels
from obspy import UTCDateTime

DAY_INDEX_COLUMNS = ["station", "date", "has_hhz", "has_hh1", "has_hh2", "path_hhz", "path_hh1", "path_hh2"]
STATION_EVENT_COLUMNS = [
    "event_id", "station", "network", "array", "ref_time", "p_time", "s_time",
    "window_start", "window_end", "primary_date", "secondary_date",
]


def write_day_index_csv(index, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DAY_INDEX_COLUMNS)
        writer.writeheader()
        for (station, date_iso), paths in sorted(index.items()):
            writer.writerow({
                "station": station,
                "date": date_iso,
                "has_hhz": paths.get("HHZ") is not None,
                "has_hh1": paths.get("HH1") is not None,
                "has_hh2": paths.get("HH2") is not None,
                "path_hhz": paths.get("HHZ") or "",
                "path_hh1": paths.get("HH1") or "",
                "path_hh2": paths.get("HH2") or "",
            })


def load_grouped_picks():
    grouped = defaultdict(dict)
    meta = {}
    with open(config.PICKS_DEDUPED_CSV) as f:
        for row in csv.DictReader(f):
            key = (row["event_id"], row["station"])
            grouped[key][row["phase"]] = row["pick_time"]
            meta[key] = (row["array"], row["network"])
    return grouped, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.force and os.path.exists(config.COVERED_STATION_EVENTS_CSV):
        print(f"{config.COVERED_STATION_EVENTS_CSV} already exists, skipping (use --force to rebuild).")
        return

    print("Building day-file existence index (filesystem only)...")
    index = build_day_file_index()
    write_day_index_csv(index, config.DAY_FILE_INDEX_CSV)
    print(f"Indexed {len(index)} (station, date) entries -> {config.DAY_FILE_INDEX_CSV}")

    print("\nLoading deduped picks and grouping by (event_id, station)...")
    grouped, meta = load_grouped_picks()
    print(f"{len(grouped)} unique (event_id, station) pairs")

    covered_rows = []
    uncovered_rows = []
    reason_counts = defaultdict(int)

    for (event_id, station), phases in grouped.items():
        array, network = meta[(event_id, station)]
        ref_time = min(UTCDateTime(t) for t in phases.values())
        window_start = ref_time - config.WINDOW_HALF_WIDTH_SEC
        window_end = ref_time + config.WINDOW_HALF_WIDTH_SEC

        primary_date = window_start.date.isoformat()
        end_date = window_end.date.isoformat()
        secondary_date = end_date if end_date != primary_date else ""

        row = {
            "event_id": event_id,
            "station": station,
            "network": network,
            "array": array,
            "ref_time": str(ref_time),
            "p_time": phases.get("P", ""),
            "s_time": phases.get("S", ""),
            "window_start": str(window_start),
            "window_end": str(window_end),
            "primary_date": primary_date,
            "secondary_date": secondary_date,
        }

        if (station, primary_date) not in index:
            row["reason"] = "no_data_on_date"
            uncovered_rows.append(row)
            reason_counts["no_data_on_date"] += 1
        elif not has_all_channels(index, station, primary_date):
            row["reason"] = "missing_channel"
            uncovered_rows.append(row)
            reason_counts["missing_channel"] += 1
        else:
            covered_rows.append(row)

    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    with open(config.COVERED_STATION_EVENTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STATION_EVENT_COLUMNS)
        writer.writeheader()
        writer.writerows(covered_rows)

    with open(config.UNCOVERED_STATION_EVENTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STATION_EVENT_COLUMNS + ["reason"])
        writer.writeheader()
        writer.writerows(uncovered_rows)

    print(f"\nCovered: {len(covered_rows)}")
    print(f"Uncovered: {len(uncovered_rows)}")
    for reason, count in sorted(reason_counts.items()):
        print(f"  {reason}: {count}")

    by_array = defaultdict(lambda: [0, 0])
    for r in covered_rows:
        by_array[r["array"]][0] += 1
    for r in uncovered_rows:
        by_array[r["array"]][1] += 1
    print("\nPer-array covered/uncovered:")
    for array, (cov, uncov) in sorted(by_array.items()):
        print(f"  {array}: covered={cov} uncovered={uncov}")

    print(f"\nWrote {config.COVERED_STATION_EVENTS_CSV}")
    print(f"Wrote {config.UNCOVERED_STATION_EVENTS_CSV}")


if __name__ == "__main__":
    main()
