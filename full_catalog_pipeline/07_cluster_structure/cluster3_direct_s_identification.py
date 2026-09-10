#!/usr/bin/env python3
"""Direct test of which phase the catalog S pick actually sits on (see
[[t1-sp-vs-depth-vpvs-check-result]] Result 2's qualitative claim: "the smaller, earlier
bump...is more likely the genuine, depth-sensitive S arrival, and the picks may not be
'wrong' so much as small and hard to see next to a much louder site artifact"). That claim
was never directly quantified -- cluster3_firn_crossstation_check.py only measured the lag
of the GLOBAL envelope peak (the loud Love-wave burst) relative to the catalog pick.

This script instead prominence-detects ALL local peaks in the S window per event, then asks:
is there a distinct local peak close to the catalog pick (the "early" peak), separate from
the global (loudest) peak? If yes, its lag distribution tells us how tightly the catalog pick
tracks a real local maximum (a proxy for genuine phase-pick precision at the actual arrival,
not the contaminant) -- and by extension, how much the Love-wave burst may be degrading pick
precision at DEEJ/LILA relative to a station without it (TJTJ, negative control).

Usage:
    python full_catalog_pipeline/cluster3_direct_s_identification.py
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
from scipy.signal import find_peaks

sys.path.insert(0, ".")
import config
from lib.day_volume_index import load_day_file_index_csv
from lib.deej_waveform_common import DEFAULTS, clean, load_merged
from lib.windowing import RollingDayCache, extract_window

import catalog_paths

STATIONS = {
    "DEEJ": dict(window_post=1.4),
    "LILA": dict(window_post=2.4),
    "TJTJ": dict(window_post=2.4),
}
WINDOW_PRE = 0.2
SEARCH_START_OFFSET = -0.05  # start peak-search slightly before the catalog S pick
MIN_GAP_S = 0.04  # early peak must lead the global peak by at least this much to count as "distinct"
PROMINENCE = 0.08  # in units of envelope peak-normalized amplitude (0-1)

_ORIGIN_CACHE = {}


def _origin_for(eid):
    if not _ORIGIN_CACHE:
        from obspy import UTCDateTime
        cur = None
        with open(DEFAULTS["phase_dat"]) as f:
            for line in f:
                if line.startswith("#"):
                    p = line.split()
                    cur = int(p[-1])
                    yr, mo, dy, hr, mi, sc = p[1:7]
                    _ORIGIN_CACHE[cur] = UTCDateTime(int(yr), int(mo), int(dy), int(hr), int(mi), float(sc))
    return _ORIGIN_CACHE.get(eid)


def analyze_station(station, window_post):
    ids = set(int(x) for x in open(DEFAULTS["ids_file"]))
    df = load_merged(ids, DEFAULTS["phase_dat"], DEFAULTS["reloc"], station,
                      network="7U", src_file=DEFAULTS["src"])
    print(f"{station}: {len(df)} events with P+S picks")

    day_index = load_day_file_index_csv(config.DAY_FILE_INDEX_CSV)
    cache = RollingDayCache(day_index, station)
    sr = config.SAMPLE_RATE_HZ

    rows = []
    for _, ev in df.iterrows():
        eid = int(ev["id"])
        origin = _origin_for(eid)
        if origin is None:
            continue
        p_pick = origin + ev["p_offset"]
        s_offset_rel_p = ev["s_offset"] - ev["p_offset"]
        row = {"primary_date": date(p_pick.year, p_pick.month, p_pick.day).isoformat()}
        window_start = p_pick - WINDOW_PRE
        window_end = p_pick + window_post
        data, status = extract_window(cache, row, window_start, window_end)
        cache.evict_before(row["primary_date"])
        if data is None or not data["complete"]:
            continue
        n_clean, e_clean = clean(data["N"]), clean(data["E"])
        h = np.sqrt(n_clean ** 2 + e_clean ** 2)
        if h.max() == 0:
            continue
        h = h / h.max()
        t = np.arange(len(h)) / sr - WINDOW_PRE  # seconds relative to P pick

        search_mask = t >= (s_offset_rel_p + SEARCH_START_OFFSET)
        if not search_mask.any():
            continue
        idx0 = np.argmax(search_mask)
        seg = h[idx0:]
        seg_t = t[idx0:] - s_offset_rel_p  # seconds relative to the CATALOG S PICK now

        peaks, props = find_peaks(seg, prominence=PROMINENCE)
        if len(peaks) == 0:
            continue
        peak_times = seg_t[peaks]
        peak_amps = seg[peaks]

        gi = np.argmax(peak_amps)
        global_lag_ms = peak_times[gi] * 1000
        global_amp = peak_amps[gi]

        candidates = [(pt, pa) for pt, pa in zip(peak_times, peak_amps)
                      if pt < peak_times[gi] - MIN_GAP_S]
        if candidates:
            early_time, early_amp = min(candidates, key=lambda x: abs(x[0]))
            has_early = True
        else:
            early_time, early_amp = np.nan, np.nan
            has_early = False

        rows.append({
            "id": eid, "depth": ev["depth"],
            "global_lag_ms": global_lag_ms, "global_amp": global_amp,
            "has_early_peak": has_early,
            "early_lag_ms": early_time * 1000 if has_early else np.nan,
            "early_amp": early_amp if has_early else np.nan,
            "amp_ratio_early_over_global": (early_amp / global_amp) if has_early else np.nan,
        })

    out = pd.DataFrame(rows)
    print(f"  usable envelope windows: {len(out)} / {len(df)}")
    if len(out):
        frac_early = out["has_early_peak"].mean()
        print(f"  fraction with a DISTINCT early peak (separate from global/loudest): {frac_early:.2f}")
        early = out[out["has_early_peak"]]
        if len(early):
            print(f"  early-peak lag rel. catalog pick (ms): median={early['early_lag_ms'].median():.1f}, "
                  f"std={early['early_lag_ms'].std():.1f}, IQR=[{early['early_lag_ms'].quantile(.25):.1f}, "
                  f"{early['early_lag_ms'].quantile(.75):.1f}]")
            print(f"  fraction of early-peak lags within +/-40ms of catalog pick: "
                  f"{((early['early_lag_ms'] >= -40) & (early['early_lag_ms'] <= 40)).mean():.2f}")
            print(f"  early/global amplitude ratio: median={early['amp_ratio_early_over_global'].median():.2f}")
        print(f"  global (loudest) peak lag rel. catalog pick (ms): "
              f"median={out['global_lag_ms'].median():.1f}, std={out['global_lag_ms'].std():.1f}")
    return out


def main():
    results = {}
    for station, kwargs in STATIONS.items():
        print(f"\n=== {station} ===")
        results[station] = analyze_station(station, **kwargs)

    out_csv = catalog_paths.work_dir("T1") + "/cluster3_direct_s_identification.csv"
    combined = pd.concat(
        [df.assign(station=sta) for sta, df in results.items() if len(df)], ignore_index=True)
    combined.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
