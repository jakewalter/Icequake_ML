#!/usr/bin/env python3
"""
Stage 4a: Two-stage SNR filtering.

Applies both:
1. Individual window threshold: SNR >= threshold_db
2. Event-level requirement: event must have >= min_stations with qualifying windows

Reads: artifacts/windows_with_snr.csv
Writes: artifacts/windows_filtered.csv
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import config


def apply_two_stage_filtering(snr_df, window_threshold_db=5.0, min_stations=4):
    """
    Apply two-stage filtering:
    1. Keep windows where trace_snr_db >= window_threshold_db
    2. Keep only those windows from events that have >= min_stations with qualifying windows
    
    Returns:
        filtered_df: filtered DataFrame
        summary: dict with statistics
    """
    total_windows_orig = len(snr_df)
    total_events_orig = snr_df['event_id'].nunique()
    
    # Stage 1: individual window threshold
    qualified_windows = snr_df[snr_df['trace_snr_db'] >= window_threshold_db].copy()
    windows_after_stage1 = len(qualified_windows)
    
    # Stage 2: per-event station coverage
    # Group qualified windows by event, count unique stations
    event_station_coverage = qualified_windows.groupby('event_id')['station'].nunique().reset_index()
    event_station_coverage.columns = ['event_id', 'n_qualified_stations']
    
    # Keep only events with >= min_stations
    qualifying_events = event_station_coverage[
        event_station_coverage['n_qualified_stations'] >= min_stations
    ]['event_id'].values
    
    # Filter to only windows from qualifying events
    filtered_df = qualified_windows[qualified_windows['event_id'].isin(qualifying_events)].copy()
    
    summary = {
        'total_windows_orig': total_windows_orig,
        'total_events_orig': total_events_orig,
        'windows_after_window_threshold': windows_after_stage1,
        'events_passing_window_threshold': qualified_windows['event_id'].nunique(),
        'events_after_station_filter': len(qualifying_events),
        'windows_after_all_filtering': len(filtered_df),
        'pct_windows_retained': 100.0 * len(filtered_df) / total_windows_orig,
        'pct_events_retained': 100.0 * len(qualifying_events) / total_events_orig,
    }
    
    return filtered_df, summary


def main():
    parser = argparse.ArgumentParser(
        description="Apply two-stage SNR filtering: window threshold + event-level station requirement"
    )
    parser.add_argument('--window-threshold-db', type=float, default=5.0,
                       help='Individual window SNR threshold in dB (default: 5.0)')
    parser.add_argument('--min-stations', type=int, default=4,
                       help='Minimum number of stations per event with qualifying windows (default: 4)')
    parser.add_argument('--force', action='store_true',
                       help='Recompute even if output exists')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Stage 4a: Two-Stage SNR Filtering")
    print("=" * 80)
    print(f"Window SNR threshold: {args.window_threshold_db} dB")
    print(f"Minimum stations per event: {args.min_stations}")
    print()
    
    # Check if output exists
    if os.path.exists(config.WINDOWS_FILTERED_CSV) and not args.force:
        print(f"Output {config.WINDOWS_FILTERED_CSV} already exists.")
        print("Use --force to recompute.")
        return
    
    print(f"Loading {config.WINDOWS_WITH_SNR_CSV}...")
    snr_df = pd.read_csv(config.WINDOWS_WITH_SNR_CSV)
    print(f"Loaded {len(snr_df)} window records from {snr_df['event_id'].nunique()} events")
    print()
    
    print(f"Applying two-stage filtering...")
    filtered_df, summary = apply_two_stage_filtering(
        snr_df, 
        window_threshold_db=args.window_threshold_db,
        min_stations=args.min_stations
    )
    print()
    
    print("FILTERING RESULTS")
    print("-" * 80)
    print(f"Original dataset:")
    print(f"  {summary['total_windows_orig']:>8,} windows from {summary['total_events_orig']:>6,} events")
    print()
    print(f"After window threshold (SNR >= {args.window_threshold_db} dB):")
    print(f"  {summary['windows_after_window_threshold']:>8,} windows from {summary['events_passing_window_threshold']:>6,} events")
    print()
    print(f"After event-level filter (>= {args.min_stations} stations per event):")
    print(f"  {summary['windows_after_all_filtering']:>8,} windows from {summary['events_after_station_filter']:>6,} events")
    print()
    print(f"Retention rates:")
    print(f"  {summary['pct_windows_retained']:>6.2f}% of windows")
    print(f"  {summary['pct_events_retained']:>6.2f}% of events")
    print("-" * 80)
    print()
    
    print(f"Writing {config.WINDOWS_FILTERED_CSV}...")
    filtered_df.to_csv(config.WINDOWS_FILTERED_CSV, index=False)
    print(f"Wrote {len(filtered_df)} records")
    print()
    print("✓ Stage 4a complete")


if __name__ == "__main__":
    main()
