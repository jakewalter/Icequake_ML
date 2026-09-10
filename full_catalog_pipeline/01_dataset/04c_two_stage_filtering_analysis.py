#!/usr/bin/env python3
"""
Analyze filtering with two-stage criteria:
1. Individual window: SNR >= threshold_db
2. Event-level: event must have >= min_stations with at least one qualifying window
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
import numpy as np
import matplotlib.pyplot as plt
import config


def analyze_with_event_criteria(snr_df, window_snr_threshold=5.0, min_stations_range=None):
    """
    Apply two-stage filtering:
    1. Individual windows: SNR >= window_snr_threshold
    2. Events: must have >= N stations with at least one qualifying window
    
    Returns:
        results: DataFrame with columns:
            min_stations, n_events_qualify, n_windows_survive, pct_events, pct_windows
    """
    if min_stations_range is None:
        min_stations_range = [1, 2, 3, 4, 5, 6, 7]
    
    results = []
    
    # First: find all events and their station coverage at the SNR threshold
    qualified_windows = snr_df[snr_df['trace_snr_db'] >= window_snr_threshold].copy()
    
    # Group by event_id to find unique stations per event that have >= 1 qualifying window
    event_station_coverage = qualified_windows.groupby('event_id')['station'].nunique().reset_index()
    event_station_coverage.columns = ['event_id', 'n_qualified_stations']
    
    total_events = snr_df['event_id'].nunique()
    total_windows = len(snr_df)
    
    for min_stations in min_stations_range:
        # Find events that meet the minimum station criterion
        qualifying_events = event_station_coverage[
            event_station_coverage['n_qualified_stations'] >= min_stations
        ]['event_id'].values
        
        # Count windows from qualifying events
        surviving_windows = qualified_windows[
            qualified_windows['event_id'].isin(qualifying_events)
        ]
        
        n_events = len(qualifying_events)
        n_windows = len(surviving_windows)
        pct_events = 100.0 * n_events / total_events if total_events > 0 else 0
        pct_windows = 100.0 * n_windows / total_windows if total_windows > 0 else 0
        
        results.append({
            'min_stations': min_stations,
            'n_events_qualify': n_events,
            'n_windows_survive': n_windows,
            'pct_events': pct_events,
            'pct_windows': pct_windows,
        })
    
    return pd.DataFrame(results)


def plot_two_stage_filtering(results, window_snr_db=5.0, output_path=None):
    """
    Plot how many events and windows survive at different min_stations thresholds.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Number of qualifying events
    ax = axes[0, 0]
    ax.plot(results['min_stations'], results['n_events_qualify'], 'o-', linewidth=2, markersize=8, color='steelblue')
    ax.set_xlabel('Minimum Stations per Event', fontsize=11)
    ax.set_ylabel('Number of Qualifying Events', fontsize=11)
    ax.set_title('Events Surviving Two-Stage Filter (SNR ≥ 5 dB + Min Stations)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xticks(results['min_stations'])
    for i, row in results.iterrows():
        ax.text(row['min_stations'], row['n_events_qualify'] + 200, 
               f"{row['n_events_qualify']}", ha='center', fontsize=9)
    
    # Plot 2: Percentage of events
    ax = axes[0, 1]
    ax.plot(results['min_stations'], results['pct_events'], 'o-', linewidth=2, markersize=8, color='coral')
    ax.set_xlabel('Minimum Stations per Event', fontsize=11)
    ax.set_ylabel('% of Total Events', fontsize=11)
    ax.set_title('Percentage of Events Surviving', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xticks(results['min_stations'])
    for i, row in results.iterrows():
        ax.text(row['min_stations'], row['pct_events'] + 1, 
               f"{row['pct_events']:.1f}%", ha='center', fontsize=9)
    
    # Plot 3: Number of surviving windows
    ax = axes[1, 0]
    ax.plot(results['min_stations'], results['n_windows_survive'], 'o-', linewidth=2, markersize=8, color='darkgreen')
    ax.set_xlabel('Minimum Stations per Event', fontsize=11)
    ax.set_ylabel('Number of Windows', fontsize=11)
    ax.set_title('Windows Surviving Two-Stage Filter', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xticks(results['min_stations'])
    for i, row in results.iterrows():
        ax.text(row['min_stations'], row['n_windows_survive'] + 500, 
               f"{row['n_windows_survive']}", ha='center', fontsize=9)
    
    # Plot 4: Percentage of windows
    ax = axes[1, 1]
    ax.plot(results['min_stations'], results['pct_windows'], 'o-', linewidth=2, markersize=8, color='darkviolet')
    ax.set_xlabel('Minimum Stations per Event', fontsize=11)
    ax.set_ylabel('% of Total Windows', fontsize=11)
    ax.set_title('Percentage of Windows Surviving', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.set_xticks(results['min_stations'])
    for i, row in results.iterrows():
        ax.text(row['min_stations'], row['pct_windows'] + 0.5, 
               f"{row['pct_windows']:.1f}%", ha='center', fontsize=9)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved: {output_path}")
    else:
        return fig


def main():
    parser = argparse.ArgumentParser(description="Analyze two-stage SNR filtering criteria")
    parser.add_argument('--window-threshold', type=float, default=5.0,
                       help='Individual window SNR threshold in dB (default: 5.0)')
    args = parser.parse_args()
    
    print("Loading SNR data...")
    snr_df = pd.read_csv(config.WINDOWS_WITH_SNR_CSV)
    print(f"Loaded {len(snr_df)} window records across {snr_df['event_id'].nunique()} events")
    
    print(f"\nAnalyzing two-stage filtering (window threshold: {args.window_threshold} dB)...")
    results = analyze_with_event_criteria(snr_df, window_snr_threshold=args.window_threshold)
    
    print("\n" + "=" * 80)
    print("TWO-STAGE FILTERING RESULTS (SNR >= {:.1f} dB + Min Stations per Event)".format(args.window_threshold))
    print("=" * 80)
    print(results.to_string(index=False))
    print("=" * 80)
    
    print("\nKey observations:")
    print(f"  - Total events in dataset: {snr_df['event_id'].nunique()}")
    print(f"  - Total windows in dataset: {len(snr_df)}")
    
    print(f"\n  - With SNR >= {args.window_threshold} dB only (no min-stations requirement):")
    first = results.iloc[0]
    print(f"    * {first['n_events_qualify']:,.0f} events ({first['pct_events']:.1f}%)")
    print(f"    * {first['n_windows_survive']:,.0f} windows ({first['pct_windows']:.1f}%)")
    
    last = results.iloc[-1]
    print(f"\n  - With SNR >= {args.window_threshold} dB AND >= 7 stations per event:")
    print(f"    * {last['n_events_qualify']:,.0f} events ({last['pct_events']:.1f}%)")
    print(f"    * {last['n_windows_survive']:,.0f} windows ({last['pct_windows']:.1f}%)")
    
    # Find threshold that gives ~10-15k windows
    target_min = 10000
    target_max = 15000
    in_range = results[(results['n_windows_survive'] >= target_min) & 
                       (results['n_windows_survive'] <= target_max)]
    if len(in_range) > 0:
        print(f"\n  - To land in 10-15k window target:")
        for _, row in in_range.iterrows():
            print(f"    * min_stations={row['min_stations']:.0f}: {row['n_windows_survive']:,.0f} windows "
                  f"({row['pct_windows']:.1f}%) from {row['n_events_qualify']:,.0f} events")
    
    # Generate figure
    out_path = os.path.join(config.ARTIFACTS_DIR, f'snr_two_stage_filtering_{args.window_threshold:.1f}db.png')
    plot_two_stage_filtering(results, window_snr_db=args.window_threshold, output_path=out_path)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
