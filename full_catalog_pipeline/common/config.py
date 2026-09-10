"""Shared constants for the full-catalog dataset build pipeline."""

import os

# This module lives in common/ after the 2026-09 reorganization, so the pipeline root is
# one level up. artifacts/ and output*/ are resolved from it.
PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(PIPELINE_ROOT, "artifacts")

# Set FCP_RUN_SUFFIX (e.g. "_B", "_C") to isolate the filtered-windows list and
# packed dataset for an SNR/min-stations ablation variant without touching the
# baseline output/ dir or the shared windows_with_snr.csv / windows.h5.
_RUN_SUFFIX = os.environ.get("FCP_RUN_SUFFIX", "")
OUTPUT_DIR = os.path.join(PIPELINE_ROOT, f"output{_RUN_SUFFIX}")

T1_QUAKEML = "/scratch2/qm/t1/t1_all_events.quakeml"
T2_QUAKEML = "/scratch2/qm/t2/t2_all_events.quakeml"
DAY_VOLUMES_ROOT = "/data/time/day_volumes"

NETWORK = "7U"
STATIONS_T1 = ["DEEJ", "ELZA", "LILA", "LOUS", "OTIS", "SQIG", "TJTJ"]
STATIONS_T2 = ["BAUM", "DRSC", "EPJZ", "FRST", "JULA", "OKGS", "WICH"]
ARRAY_STATIONS = {"T1": STATIONS_T1, "T2": STATIONS_T2}
ALL_STATIONS = STATIONS_T1 + STATIONS_T2
CHANNELS = ["HHZ", "HH1", "HH2"]

SAMPLE_RATE_HZ = 200.0
WINDOW_HALF_WIDTH_SEC = 5.0  # window = [ref_time - 5s, ref_time + 5s], matches 07's convention

# SNR windows/filter, ported from ref_read_m_file.py (line 80 + its filter chain)
SNR_HIGHPASS_FREQ = 5.0
SNR_HIGHPASS_CORNERS = 4
SNR_BANDPASS_FREQMIN = 20.0
SNR_BANDPASS_FREQMAX = 80.0
SNR_BANDPASS_CORNERS = 2
SNR_SIGNAL_WINDOW = (-0.25, 0.25)   # seconds relative to ref_time
SNR_NOISE_WINDOW = (-2.25, -1.75)   # seconds relative to ref_time

SPLIT_RATIOS = {"train": 0.70, "dev": 0.15, "test": 0.15}
RANDOM_SEED = 42

# Artifact paths
PICKS_RAW_CSV = os.path.join(ARTIFACTS_DIR, "picks_raw.csv")
PICKS_DEDUPED_CSV = os.path.join(ARTIFACTS_DIR, "picks_deduped.csv")
DAY_FILE_INDEX_CSV = os.path.join(ARTIFACTS_DIR, "day_file_index.csv")
COVERED_STATION_EVENTS_CSV = os.path.join(ARTIFACTS_DIR, "covered_station_events.csv")
UNCOVERED_STATION_EVENTS_CSV = os.path.join(ARTIFACTS_DIR, "uncovered_station_events.csv")
WINDOWS_H5 = os.path.join(ARTIFACTS_DIR, "windows.h5")
WINDOWS_METADATA_CSV = os.path.join(ARTIFACTS_DIR, "windows_metadata.csv")
WINDOWS_WITH_SNR_CSV = os.path.join(ARTIFACTS_DIR, "windows_with_snr.csv")
WINDOWS_FILTERED_CSV = os.path.join(ARTIFACTS_DIR, f"windows_filtered{_RUN_SUFFIX}.csv")

OUTPUT_METADATA_CSV = os.path.join(OUTPUT_DIR, "metadata.csv")
OUTPUT_WAVEFORMS_HDF5 = os.path.join(OUTPUT_DIR, "waveforms.hdf5")
