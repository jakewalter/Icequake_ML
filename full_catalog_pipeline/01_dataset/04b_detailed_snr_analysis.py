#!/usr/bin/env python3
"""
Generate detailed SNR diagnostic figures.
Analyzes event-level statistics, station heterogeneity, and high/low SNR cases.
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
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import config
from lib.snr_analysis import (
    analyze_snr_by_event,
    analyze_station_quality,
    plot_event_snr_histogram,
    plot_station_heterogeneity,
    plot_high_heterogeneity_events,
    plot_station_performance,
    plot_high_snr_events,
    plot_low_snr_events,
)


def main():
    parser = argparse.ArgumentParser(description="Generate detailed SNR analysis figures")
    args = parser.parse_args()

    print("Loading SNR data...")
    snr_df = pd.read_csv(config.WINDOWS_WITH_SNR_CSV)
    print(f"Loaded {len(snr_df)} window records")

    print("\nComputing event-level statistics...")
    event_stats = analyze_snr_by_event(snr_df)
    print(f"Analyzed {len(event_stats)} events")

    print("\nComputing station-level statistics...")
    station_stats = analyze_station_quality(snr_df)
    print(f"Analyzed {len(station_stats)} station pairs")

    print("\n=== EVENT-LEVEL STATISTICS ===")
    print(f"Median SNR across all events: {event_stats['median_snr'].median():.2f} dB")
    print(f"Mean SNR across all events: {event_stats['median_snr'].mean():.2f} dB")
    print(f"Std Dev of event median SNRs: {event_stats['median_snr'].std():.2f} dB")

    print("\n=== STATION HETEROGENEITY ===")
    print(f"Median SNR range within events: {event_stats['snr_range'].median():.2f} dB")
    print(f"Max SNR range within any event: {event_stats['snr_range'].max():.2f} dB")
    print(f"\nTop 10 most heterogeneous events (largest station-to-station spread):")
    top_het = event_stats.nlargest(10, 'snr_range')[['event_id', 'n_stations', 'median_snr', 'snr_range']]
    print(top_het.to_string(index=False))

    print("\n=== STATION PERFORMANCE ===")
    print(station_stats.sort_values('median_snr', ascending=False).to_string(index=False))

    print("\n=== HIGH SNR EVENTS (Top 10) ===")
    top_snr = event_stats.nlargest(10, 'median_snr')[['event_id', 'n_stations', 'median_snr', 'std_snr']]
    print(top_snr.to_string(index=False))

    print("\n=== LOW SNR EVENTS (Bottom 10) ===")
    bot_snr = event_stats.nsmallest(10, 'median_snr')[['event_id', 'n_stations', 'median_snr', 'std_snr']]
    print(bot_snr.to_string(index=False))

    # Generate figures
    out_dir = config.ARTIFACTS_DIR
    print(f"\n=== GENERATING FIGURES ===")

    plot_event_snr_histogram(event_stats, os.path.join(out_dir, 'snr_event_median_histogram.png'))
    plot_station_heterogeneity(event_stats, os.path.join(out_dir, 'snr_station_heterogeneity.png'))
    plot_high_heterogeneity_events(snr_df, event_stats, n_events=6, 
                                   output_path=os.path.join(out_dir, 'snr_high_heterogeneity_events.png'))
    plot_station_performance(station_stats, os.path.join(out_dir, 'snr_station_performance.png'))
    plot_high_snr_events(snr_df, event_stats, n_events=6, 
                        output_path=os.path.join(out_dir, 'snr_high_snr_events.png'))
    plot_low_snr_events(snr_df, event_stats, n_events=6, 
                       output_path=os.path.join(out_dir, 'snr_low_snr_events.png'))

    print("\nAll detailed SNR analysis figures generated!")


if __name__ == "__main__":
    main()
