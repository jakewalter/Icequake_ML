#!/usr/bin/env python3
"""Cross-station check on the firn-reverberation hypothesis (see
[[t1-sp-vs-depth-vpvs-check-result]]): DEEJ's cluster3 S-window shows a smaller bump near
the catalog S pick followed by a MUCH LARGER burst at a roughly fixed ~100-150ms delay,
across nearly the whole depth range. If that fixed-delay secondary lobe is a firn
reverberation under DEEJ specifically (receiver-side), it should shift or disappear at a
different station (TJTJ, LILA) with different near-surface structure. If it persists at the
same delay everywhere, that argues against a DEEJ-local site effect.

Quantifies "time of peak envelope amplitude relative to the catalog S pick" per event, for
each station, as a numeric alternative to eyeballing cascades/stacks (useful if figures can't
be viewed). A tight cluster near a positive delay (matching DEEJ's ~100-150ms) that repeats
at TJTJ/LILA would argue against the firn hypothesis; a station-dependent (or absent) delay
supports it.

Usage:
    python full_catalog_pipeline/cluster3_firn_crossstation_check.py
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

sys.path.insert(0, ".")
import config
from lib.day_volume_index import load_day_file_index_csv
from lib.deej_waveform_common import DEFAULTS, clean, load_merged
from lib.windowing import RollingDayCache, extract_window

import catalog_paths

STATIONS = {
    "DEEJ": dict(window_post=1.4),
    "TJTJ": dict(window_post=2.4),
    "LILA": dict(window_post=2.4),
    "ELZA": dict(window_post=2.4),
    "OTIS": dict(window_post=2.4),
}
WINDOW_PRE = 0.2
SEARCH_START_OFFSET = -0.05  # start peak-search slightly before the catalog S pick


def peak_lag_for_station(station, window_post):
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
        h_search = h.copy()
        h_search[~search_mask] = -1
        peak_idx = np.argmax(h_search)
        peak_lag_s = t[peak_idx] - s_offset_rel_p
        rows.append({"id": eid, "depth": ev["depth"], "peak_lag_ms": peak_lag_s * 1000})

    out = pd.DataFrame(rows)
    print(f"  usable envelope windows: {len(out)} / {len(df)}")
    if len(out):
        print(f"  peak lag relative to S pick (ms): median={out['peak_lag_ms'].median():.0f}, "
              f"mean={out['peak_lag_ms'].mean():.0f}, std={out['peak_lag_ms'].std():.0f}")
        print(f"  fraction with peak lag in [50,250]ms "
              f"(DEEJ's reverberation band): {((out['peak_lag_ms']>=50)&(out['peak_lag_ms']<=250)).mean():.2f}")
        print(f"  fraction with peak lag in [-30,30]ms (peak essentially ON the S pick): "
              f"{((out['peak_lag_ms']>=-30)&(out['peak_lag_ms']<=30)).mean():.2f}")
    return out


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


def main():
    results = {}
    for station, kwargs in STATIONS.items():
        print(f"\n=== {station} ===")
        results[station] = peak_lag_for_station(station, **kwargs)

    out_csv = catalog_paths.work_dir("T1") + "/cluster3_firn_crossstation_peaklag.csv"
    combined = pd.concat(
        [df.assign(station=sta) for sta, df in results.items() if len(df)], ignore_index=True)
    combined.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
