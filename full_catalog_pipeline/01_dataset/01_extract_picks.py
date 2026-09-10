#!/usr/bin/env python3
"""Stage 1: fast QuakeML pick extraction + dedup for the full T1+T2 catalogs.

Reads: config.T1_QUAKEML, config.T2_QUAKEML
Writes: artifacts/picks_raw.csv, artifacts/picks_deduped.csv
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from lib.quakeml_fast import parse_quakeml_picks, dedupe_picks

COLUMNS = [
    "event_id", "array", "origin_time", "latitude", "longitude", "depth_m",
    "station", "network", "phase", "pick_time", "method_id",
]


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, label):
    by_array_phase = {}
    for r in rows:
        key = (r["array"], r["phase"])
        by_array_phase[key] = by_array_phase.get(key, 0) + 1
    print(f"\n{label}: {len(rows)} total rows")
    for (array, phase), count in sorted(by_array_phase.items()):
        print(f"  {array} {phase}: {count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.force and os.path.exists(config.PICKS_DEDUPED_CSV):
        print(f"{config.PICKS_DEDUPED_CSV} already exists, skipping (use --force to rebuild).")
        return

    print("Parsing T1 catalog...")
    t1_rows = parse_quakeml_picks(config.T1_QUAKEML, "T1", config.NETWORK)
    print(f"  {len(t1_rows)} raw picks")

    print("Parsing T2 catalog...")
    t2_rows = parse_quakeml_picks(config.T2_QUAKEML, "T2", config.NETWORK)
    print(f"  {len(t2_rows)} raw picks")

    all_rows = t1_rows + t2_rows
    summarize(all_rows, "Raw picks")
    write_csv(all_rows, config.PICKS_RAW_CSV)
    print(f"\nWrote {config.PICKS_RAW_CSV}")

    deduped = dedupe_picks(all_rows)
    summarize(deduped, "Deduped picks")
    write_csv(deduped, config.PICKS_DEDUPED_CSV)
    print(f"\nWrote {config.PICKS_DEDUPED_CSV}")

    unique_station_events = {(r["event_id"], r["station"]) for r in deduped}
    events = {r["event_id"] for r in deduped}
    print(f"\nUnique events: {len(events)}")
    print(f"Unique (event_id, station) pairs (upper bound on candidate windows): {len(unique_station_events)}")


if __name__ == "__main__":
    main()
