"""Detailed SNR analysis and visualization."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec


def analyze_snr_by_event(snr_df):
    """
    Compute per-event SNR statistics to identify high/low SNR events
    and event-to-event heterogeneity.
    
    Returns:
        event_stats: DataFrame with columns:
            event_id, n_stations, median_snr, mean_snr, std_snr, 
            min_snr, max_snr, snr_range, cv (coefficient of variation)
    """
    event_stats = snr_df.groupby('event_id').agg({
        'trace_snr_db': [
            'count',
            'median',
            'mean',
            'std',
            'min',
            'max',
            lambda x: x.max() - x.min(),  # range
            lambda x: x.std() / (x.mean() + 1e-6) if x.mean() > 0 else np.nan,  # CV
        ]
    }).reset_index()
    
    event_stats.columns = [
        'event_id', 'n_stations', 'median_snr', 'mean_snr', 
        'std_snr', 'min_snr', 'max_snr', 'snr_range', 'cv'
    ]
    
    return event_stats


def analyze_station_heterogeneity(snr_df, event_stats):
    """
    Find events where station-to-station SNR varies widely.
    Returns the events with highest SNR_range (max - min within event).
    """
    heterogeneous = event_stats.nlargest(20, 'snr_range')
    return heterogeneous


def analyze_station_quality(snr_df):
    """
    Compute per-station statistics across all events.
    
    Returns:
        station_stats: DataFrame with columns:
            station, n_windows, median_snr, mean_snr, std_snr, etc.
    """
    station_stats = snr_df.groupby(['array', 'station']).agg({
        'trace_snr_db': [
            'count',
            'median',
            'mean',
            'std',
            'min',
            'max',
            lambda x: (x > 10).sum(),  # count above 10 dB
            lambda x: (x > 5).sum(),   # count above 5 dB
        ]
    }).reset_index()
    
    station_stats.columns = [
        'array', 'station', 'n_windows', 'median_snr', 'mean_snr',
        'std_snr', 'min_snr', 'max_snr', 'n_above_10db', 'n_above_5db'
    ]
    station_stats['above_10db_pct'] = 100.0 * station_stats['n_above_10db'] / station_stats['n_windows']
    station_stats['above_5db_pct'] = 100.0 * station_stats['n_above_5db'] / station_stats['n_windows']
    
    return station_stats


def plot_event_snr_histogram(event_stats, output_path):
    """Plot histogram of median SNR per event."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(event_stats['median_snr'].dropna(), bins=40, edgecolor='black', alpha=0.7)
    ax.axvline(event_stats['median_snr'].median(), color='red', linestyle='--', 
               linewidth=2, label=f"Median: {event_stats['median_snr'].median():.2f} dB")
    ax.axvline(event_stats['median_snr'].mean(), color='orange', linestyle='--', 
               linewidth=2, label=f"Mean: {event_stats['median_snr'].mean():.2f} dB")
    
    ax.set_xlabel('Median SNR per Event (dB)', fontsize=12)
    ax.set_ylabel('Number of Events', fontsize=12)
    ax.set_title('Distribution of Event-Level Median SNR', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_station_heterogeneity(event_stats, output_path):
    """Plot distribution of SNR heterogeneity (range) within events."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(event_stats['snr_range'].dropna(), bins=40, edgecolor='black', alpha=0.7, color='coral')
    ax.axvline(event_stats['snr_range'].median(), color='red', linestyle='--', 
               linewidth=2, label=f"Median range: {event_stats['snr_range'].median():.2f} dB")
    
    ax.set_xlabel('SNR Range within Event (max - min, dB)', fontsize=12)
    ax.set_ylabel('Number of Events', fontsize=12)
    ax.set_title('Station-to-Station SNR Variability Within Events', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_high_heterogeneity_events(snr_df, event_stats, n_events=6, output_path=None):
    """
    Plot the top N events with highest station-to-station SNR heterogeneity.
    Show per-station SNR for those events.
    """
    top_heterogeneous = event_stats.nlargest(n_events, 'snr_range')
    event_ids = top_heterogeneous['event_id'].values
    
    fig = plt.figure(figsize=(14, 3 * n_events))
    
    for idx, event_id in enumerate(event_ids):
        ax = plt.subplot(n_events, 1, idx + 1)
        
        event_data = snr_df[snr_df['event_id'] == event_id].sort_values('trace_snr_db', ascending=False)
        stations = [f"{row['array']}/{row['station']}" for _, row in event_data.iterrows()]
        snrs = event_data['trace_snr_db'].values
        
        colors = ['green' if s > 10 else 'orange' if s > 5 else 'red' for s in snrs]
        ax.barh(range(len(stations)), snrs, color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(stations)))
        ax.set_yticklabels(stations, fontsize=9)
        ax.set_xlabel('SNR (dB)', fontsize=10)
        ax.set_title(f"Event {event_id} - Range: {event_data['trace_snr_db'].max() - event_data['trace_snr_db'].min():.2f} dB "
                    f"(Median: {event_data['trace_snr_db'].median():.2f} dB)", fontsize=11)
        ax.axvline(5, color='orange', linestyle=':', alpha=0.5, linewidth=1)
        ax.axvline(10, color='green', linestyle=':', alpha=0.5, linewidth=1)
        ax.grid(alpha=0.2, axis='x')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved: {output_path}")
    else:
        return fig


def plot_station_performance(station_stats, output_path):
    """Plot per-station SNR statistics and pass rates."""
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Sort by median SNR descending
    station_stats_sorted = station_stats.sort_values('median_snr', ascending=False)
    station_names = [f"{row['array']}/{row['station']}" for _, row in station_stats_sorted.iterrows()]
    
    # 1. Median SNR by station
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.barh(range(len(station_names)), station_stats_sorted['median_snr'], color='steelblue', alpha=0.7)
    ax1.set_yticks(range(len(station_names)))
    ax1.set_yticklabels(station_names, fontsize=9)
    ax1.set_xlabel('Median SNR (dB)', fontsize=10)
    ax1.set_title('Median SNR by Station', fontsize=11, fontweight='bold')
    ax1.axvline(5, color='orange', linestyle=':', alpha=0.5)
    ax1.axvline(10, color='green', linestyle=':', alpha=0.5)
    ax1.grid(alpha=0.2, axis='x')
    
    # 2. Pass rate at 5 dB
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(range(len(station_names)), station_stats_sorted['above_5db_pct'], 
            color='darkorange', alpha=0.7)
    ax2.set_yticks(range(len(station_names)))
    ax2.set_yticklabels(station_names, fontsize=9)
    ax2.set_xlabel('% Windows Above 5 dB', fontsize=10)
    ax2.set_title('Pass Rate at 5 dB Threshold', fontsize=11, fontweight='bold')
    ax2.set_xlim(0, 100)
    ax2.grid(alpha=0.2, axis='x')
    
    # 3. Pass rate at 10 dB
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.barh(range(len(station_names)), station_stats_sorted['above_10db_pct'], 
            color='darkgreen', alpha=0.7)
    ax3.set_yticks(range(len(station_names)))
    ax3.set_yticklabels(station_names, fontsize=9)
    ax3.set_xlabel('% Windows Above 10 dB', fontsize=10)
    ax3.set_title('Pass Rate at 10 dB Threshold', fontsize=11, fontweight='bold')
    ax3.set_xlim(0, 100)
    ax3.grid(alpha=0.2, axis='x')
    
    # 4. Mean ± std per station
    ax4 = fig.add_subplot(gs[1, 1])
    x_pos = np.arange(len(station_names))
    ax4.errorbar(x_pos, station_stats_sorted['mean_snr'], 
                yerr=station_stats_sorted['std_snr'], 
                fmt='o', markersize=6, capsize=3, capthick=1, color='purple', alpha=0.7)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(station_names, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('SNR (dB)', fontsize=10)
    ax4.set_title('Mean ± Std SNR by Station', fontsize=11, fontweight='bold')
    ax4.axhline(5, color='orange', linestyle=':', alpha=0.5)
    ax4.axhline(10, color='green', linestyle=':', alpha=0.5)
    ax4.grid(alpha=0.2, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_high_snr_events(snr_df, event_stats, n_events=6, output_path=None):
    """Plot the top N events with highest median SNR."""
    top_high = event_stats.nlargest(n_events, 'median_snr')
    event_ids = top_high['event_id'].values
    
    fig = plt.figure(figsize=(14, 3 * n_events))
    
    for idx, event_id in enumerate(event_ids):
        ax = plt.subplot(n_events, 1, idx + 1)
        
        event_data = snr_df[snr_df['event_id'] == event_id].sort_values('trace_snr_db', ascending=False)
        stations = [f"{row['array']}/{row['station']}" for _, row in event_data.iterrows()]
        snrs = event_data['trace_snr_db'].values
        
        colors = ['green' if s > 10 else 'orange' if s > 5 else 'red' for s in snrs]
        ax.barh(range(len(stations)), snrs, color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(stations)))
        ax.set_yticklabels(stations, fontsize=9)
        ax.set_xlabel('SNR (dB)', fontsize=10)
        ax.set_title(f"Event {event_id} - Median: {event_data['trace_snr_db'].median():.2f} dB "
                    f"(Range: {event_data['trace_snr_db'].max() - event_data['trace_snr_db'].min():.2f} dB)", fontsize=11)
        ax.axvline(5, color='orange', linestyle=':', alpha=0.5, linewidth=1)
        ax.axvline(10, color='green', linestyle=':', alpha=0.5, linewidth=1)
        ax.grid(alpha=0.2, axis='x')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved: {output_path}")
    else:
        return fig


def plot_low_snr_events(snr_df, event_stats, n_events=6, output_path=None):
    """Plot the bottom N events with lowest median SNR."""
    bottom_low = event_stats.nsmallest(n_events, 'median_snr')
    event_ids = bottom_low['event_id'].values
    
    fig = plt.figure(figsize=(14, 3 * n_events))
    
    for idx, event_id in enumerate(event_ids):
        ax = plt.subplot(n_events, 1, idx + 1)
        
        event_data = snr_df[snr_df['event_id'] == event_id].sort_values('trace_snr_db', ascending=False)
        stations = [f"{row['array']}/{row['station']}" for _, row in event_data.iterrows()]
        snrs = event_data['trace_snr_db'].values
        
        colors = ['green' if s > 10 else 'orange' if s > 5 else 'red' for s in snrs]
        ax.barh(range(len(stations)), snrs, color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(range(len(stations)))
        ax.set_yticklabels(stations, fontsize=9)
        ax.set_xlabel('SNR (dB)', fontsize=10)
        ax.set_title(f"Event {event_id} - Median: {event_data['trace_snr_db'].median():.2f} dB "
                    f"(Range: {event_data['trace_snr_db'].max() - event_data['trace_snr_db'].min():.2f} dB)", fontsize=11)
        ax.axvline(5, color='orange', linestyle=':', alpha=0.5, linewidth=1)
        ax.axvline(10, color='green', linestyle=':', alpha=0.5, linewidth=1)
        ax.grid(alpha=0.2, axis='x')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"Saved: {output_path}")
    else:
        return fig
