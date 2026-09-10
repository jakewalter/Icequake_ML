#!/usr/bin/env python3
"""Particle-motion test: does the S-window's 2nd/larger pulse (the fixed-delay
firn/site-effect burst from [[t1-sp-vs-depth-vpvs-check-result]] Results 2/4/5/6)
share the SAME linear polarization as the 1st/smaller pulse (consistent with a
simple body-wave reverberation -- literally the same wave, partially reflected),
or does it show a DIFFERENT polarization / more elliptical (Z-R quadrature)
motion (consistent with a distinct converted phase or a guided/Scholte-like
interface mode, as Result 6's forward model suggested once simple reverberation
amplitude was ruled out)?

Requires station-frame orientation offsets from
`cluster3_orientation_calibration.py` (HH1/HH2 have no verified geographic
azimuth in this deployment's metadata) to rotate to radial/transverse per event
using each event's own hypoDD-independent source azimuth (.src "az" field).

For each event at each station:
  - rotate raw HH1/HH2 -> true N/E (station offset) -> R/T (event az)
  - window 1 ("1st pulse"): catalus S pick +/- a short pad
  - window 2 ("2nd pulse"): catalog S pick + this station's per-event peak_lag_ms
    (from cluster3_firn_crossstation_peaklag.csv) +/- a short pad
  - compute, per window: 3-D (Z,R,T) rectilinearity (planarity via eigenvalues),
    dominant polarization azimuth within the R-T plane (0=radial,90=transverse),
    dip of the dominant eigenvector off the horizontal plane, and a Z-vs-R
    quadrature score (zero-lag |corr(Z,R)| vs |corr(Z, Hilbert(R))| -- linear
    body-wave motion should peak at zero lag; retrograde/prograde elliptical
    (Rayleigh/Scholte-like) motion should peak in quadrature).

Usage:
    python full_catalog_pipeline/cluster3_sp_pulse_polarization.py
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

import sys
from datetime import date

import numpy as np
import pandas as pd
from scipy.signal import hilbert

sys.path.insert(0, ".")
import config
from lib.day_volume_index import load_day_file_index_csv
from lib.deej_waveform_common import DEFAULTS, clean, load_merged
from lib.windowing import RollingDayCache, extract_window
from cluster3_orientation_calibration import _origin_and_az_for_station

import catalog_paths

_HYPODD = catalog_paths.work_dir("T1")
PEAKLAG_CSV = f"{_HYPODD}/cluster3_firn_crossstation_peaklag.csv"
OFFSETS_CSV = f"{_HYPODD}/cluster3_orientation_offsets.csv"

STATIONS = ["DEEJ", "LILA", "TJTJ"]
PULSE_HALF_WIDTH_S = 0.045  # +/- window around each pulse center
WINDOW_PRE = 0.2
WINDOW_POST = 2.4


def rotate_to_rt(n, e, station_offset_deg, event_az_deg):
    """raw (n=HH2, e=HH1) sensor-frame components -> (radial, transverse) using
    a station orientation offset (phi_raw - az, solved from P polarization) then
    the event's own source azimuth. true_n = n*cos(off) - e*sin(off) is NOT what
    we need directly: we only need the net rotation from sensor frame to R/T,
    which is (event_az - station_offset) applied to (n,e) as a 2D rotation
    (station_offset already expresses sensor-frame-phi relative to true az, so
    subtracting it converts a sensor-frame reading directly into an az-relative
    (radial) reading without a separate explicit true-N/E intermediate)."""
    theta = np.radians(event_az_deg - station_offset_deg)
    radial = n * np.cos(theta) + e * np.sin(theta)
    transverse = -n * np.sin(theta) + e * np.cos(theta)
    return radial, transverse


def window_stats(z, r, t):
    """3-component polarization stats for one short window."""
    z_c, r_c, t_c = z - z.mean(), r - r.mean(), t - t.mean()
    cov = np.cov(np.vstack([z_c, r_c, t_c]))
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    l3, l2, l1 = eigvals  # l1 = largest
    principal = eigvecs[:, 2]  # [z, r, t] of dominant eigenvector
    rectilinearity = 1.0 - (l2 + l3) / (2.0 * l1) if l1 > 0 else 0.0
    horiz_azimuth = np.degrees(np.arctan2(principal[2], principal[1])) % 180.0  # 0=R,90=T
    dip = np.degrees(np.arcsin(np.clip(abs(principal[0]), 0, 1)))

    # Z-vs-R quadrature: does Z correlate better with R at zero lag (linear body
    # wave) or with R in quadrature (analytic-signal 90 deg phase shift, the
    # Rayleigh/Scholte-like retrograde-ellipse signature)?
    if np.std(z_c) > 0 and np.std(r_c) > 0:
        corr0 = abs(np.corrcoef(z_c, r_c)[0, 1])
        r_analytic = hilbert(r_c)
        r_quad = np.imag(r_analytic)
        corr90 = abs(np.corrcoef(z_c, r_quad)[0, 1])
    else:
        corr0, corr90 = np.nan, np.nan

    return dict(rectilinearity=rectilinearity, horiz_azimuth=horiz_azimuth, dip=dip,
                corr0=corr0, corr90=corr90, quad_dominant=(corr90 > corr0) if not np.isnan(corr0) else np.nan)


def process_station(station, offsets_df, peaklag_df):
    station_offset = offsets_df.loc[station, "offset_deg"]
    conc = offsets_df.loc[station, "concentration"]
    print(f"\n=== {station} (orientation offset={station_offset:.1f} deg, R={conc:.2f}) ===")

    df = _origin_and_az_for_station(station)
    lag_map = peaklag_df[peaklag_df["station"] == station].set_index("id")["peak_lag_ms"]
    df = df[df["id"].isin(lag_map.index)].copy()
    df["peak_lag_ms"] = df["id"].map(lag_map)
    print(f"  {len(df)} events with P/S picks + az + peak-lag")

    day_index = load_day_file_index_csv(config.DAY_FILE_INDEX_CSV)
    cache = RollingDayCache(day_index, station)
    sr = config.SAMPLE_RATE_HZ

    rows = []
    for _, ev in df.sort_values("depth").iterrows():
        eid = int(ev["id"])
        p_pick = ev["origin"] + ev["p_offset"]
        s_offset_rel_p = ev["s_offset"] - ev["p_offset"]
        row = {"primary_date": date(p_pick.year, p_pick.month, p_pick.day).isoformat()}
        window_start = p_pick - WINDOW_PRE
        window_end = p_pick + WINDOW_POST
        data, status = extract_window(cache, row, window_start, window_end)
        cache.evict_before(row["primary_date"])
        if data is None or not data["complete"]:
            continue

        z_c = clean(data["Z"])
        n_c, e_c = clean(data["N"]), clean(data["E"])
        radial, transverse = rotate_to_rt(n_c, e_c, station_offset, ev["az"])
        t = np.arange(len(z_c)) / sr - WINDOW_PRE

        def window_mask(center_s):
            return (t >= center_s - PULSE_HALF_WIDTH_S) & (t <= center_s + PULSE_HALF_WIDTH_S)

        m1 = window_mask(s_offset_rel_p)
        m2 = window_mask(s_offset_rel_p + ev["peak_lag_ms"] / 1000.0)
        if m1.sum() < 10 or m2.sum() < 10:
            continue

        stats1 = window_stats(z_c[m1], radial[m1], transverse[m1])
        stats2 = window_stats(z_c[m2], radial[m2], transverse[m2])
        row_out = {"id": eid, "depth": ev["depth"]}
        row_out.update({f"p1_{k}": v for k, v in stats1.items()})
        row_out.update({f"p2_{k}": v for k, v in stats2.items()})
        rows.append(row_out)

    out = pd.DataFrame(rows)
    print(f"  usable event windows: {len(out)} / {len(df)}")
    return out


def summarize(out, station):
    if len(out) == 0:
        print(f"  {station}: no usable events")
        return
    print(f"  {station} (n={len(out)}):")
    print(f"    1st pulse: median rectilin={out['p1_rectilinearity'].median():.2f}, "
          f"median dip={out['p1_dip'].median():.0f} deg, "
          f"quad-dominant frac={out['p1_quad_dominant'].mean():.2f}")
    print(f"    2nd pulse: median rectilin={out['p2_rectilinearity'].median():.2f}, "
          f"median dip={out['p2_dip'].median():.0f} deg, "
          f"quad-dominant frac={out['p2_quad_dominant'].mean():.2f}")
    # circular comparison of horizontal azimuth (mod 180) between pulses
    d = np.radians(2 * (out["p2_horiz_azimuth"] - out["p1_horiz_azimuth"]))
    conc = np.abs(np.mean(np.exp(1j * d)))
    mean_shift = (np.degrees(np.angle(np.mean(np.exp(1j * d)))) / 2.0) % 180.0
    print(f"    azimuth shift (2nd rel. 1st, mod 180): {mean_shift:.0f} deg, "
          f"consistency R={conc:.2f} (R~1 = same azimuth every event, R~0 = unrelated)")


def main():
    offsets_df = pd.read_csv(OFFSETS_CSV, index_col=0)
    peaklag_df = pd.read_csv(PEAKLAG_CSV)

    all_out = {}
    for station in STATIONS:
        out = process_station(station, offsets_df, peaklag_df)
        all_out[station] = out
        out_csv = f"{_HYPODD}/cluster3_sp_pulse_polarization_{station.lower()}.csv"
        out.to_csv(out_csv, index=False)

    print("\n=== Summary: 1st vs 2nd S-window pulse polarization ===")
    for station, out in all_out.items():
        summarize(out, station)


if __name__ == "__main__":
    main()
