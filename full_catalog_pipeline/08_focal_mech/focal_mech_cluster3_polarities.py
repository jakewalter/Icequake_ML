#!/usr/bin/env python3
"""Investigation 1 (see t1_composite_focal_mech_and_vpvs_consistency_plan memory), step 1:
automated P first-motion polarity picking for every event x station pair in T1 cluster3
(n=222, the already-vetted well-conditioned pyocto/hypoDD basal cluster).

No polarity picks exist anywhere in this pipeline -- phase.dat/pyocto only carry arrival
time + phase label + probability. This pulls a short raw Z-component snippet around each
catalog P pick (reusing the existing day_file_index.csv + RollingDayCache/extract_window
infra built for Stage 3 windowing, config.py's DAY_VOLUMES_ROOT convention) and determines
up/down polarity from the sign of the first significant excursion after the pick, with an
SNR-based reliability flag (not a substitute for the hand-reviewed subset check the plan
calls for -- see focal_mech_cluster3_qc_panel.py).

Usage:
    python full_catalog_pipeline/focal_mech_cluster3_polarities.py
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
from obspy import UTCDateTime

sys.path.insert(0, ".")
import config
from lib.day_volume_index import load_day_file_index_csv
from lib.windowing import RollingDayCache, load_day_traces

import catalog_paths

IDS_FILE = catalog_paths.work_dir("T1") + "/cluster3_event_ids.txt"
PHASE_DAT = catalog_paths.phase_dat("T1")
OUT_CSV = catalog_paths.work_dir("T1") + "/cluster3_polarities.csv"

STATIONS = ["DEEJ", "ELZA", "TJTJ", "OTIS", "LILA"]  # SQIG/LOUS have no P picks in cluster3
NETWORK = "7U"

NOISE_WINDOW = (-0.30, -0.05)   # seconds relative to P pick, for noise amplitude
SIGNAL_LOOKAHEAD = 0.04         # seconds after the pick to search for the first swing
MIN_SNR = 3.0                   # below this, polarity is flagged unreliable


def load_events(ids, phase_dat, stations, network="7U"):
    sta_tags = {f"{network}.{s}": s for s in stations}
    events = {}
    cur = None
    with open(phase_dat) as f:
        for line in f:
            if line.startswith("#"):
                p = line.split()
                eid = int(p[-1])
                cur = eid if eid in ids else None
                if cur is not None:
                    yr, mo, dy, hr, mi, sc = p[1:7]
                    origin = UTCDateTime(int(yr), int(mo), int(dy), int(hr), int(mi), float(sc))
                    events[cur] = {"id": cur, "origin": origin, "picks": {}}
            elif cur is not None:
                parts = line.split()
                if len(parts) != 4:
                    continue
                sta, tt, wt, ph = parts
                if sta in sta_tags and ph == "P":
                    events[cur]["picks"][sta_tags[sta]] = float(tt)
    return events


def first_motion_polarity(z, pick_idx, sr, lookahead_sec):
    """Sign of the largest-magnitude excursion within `lookahead_sec` after the pick,
    relative to the pre-pick noise level. Returns (polarity, snr) where polarity is
    +1 (up/compressional), -1 (down/dilatational), or 0 (indeterminate)."""
    noise_i0 = pick_idx + int(round(NOISE_WINDOW[0] * sr))
    noise_i1 = pick_idx + int(round(NOISE_WINDOW[1] * sr))
    if noise_i0 < 0 or noise_i1 <= noise_i0:
        return 0, np.nan
    noise = z[noise_i0:noise_i1]
    noise_amp = np.std(noise) if len(noise) else 0.0
    if noise_amp == 0:
        return 0, np.nan

    sig_i1 = pick_idx + max(1, int(round(lookahead_sec * sr)))
    window = z[pick_idx:sig_i1] - z[max(0, pick_idx - 3):pick_idx].mean()
    if len(window) == 0:
        return 0, np.nan
    peak_i = np.argmax(np.abs(window))
    peak_val = window[peak_i]
    snr = abs(peak_val) / noise_amp
    if snr < MIN_SNR:
        return 0, snr
    return (1 if peak_val > 0 else -1), snr


def main():
    ids = set(int(x) for x in open(IDS_FILE))
    events = load_events(ids, PHASE_DAT, STATIONS, network=NETWORK)
    print(f"cluster3 events: {len(events)}")

    day_index = load_day_file_index_csv(config.DAY_FILE_INDEX_CSV)
    caches = {sta: RollingDayCache(day_index, sta) for sta in STATIONS}

    # Process events in chronological order and evict each station's cache down to just
    # the current date after every event -- each event needs at most one calendar day per
    # station, and RollingDayCache never evicts on its own, so without this a ~900-day-file
    # sweep (222 events x up to 5 stations, almost every one a distinct day) accumulates
    # full-day traces without bound (observed: >20GB RSS within one minute before this fix).
    ordered_events = sorted(events.items(), key=lambda kv: kv[1]["origin"])

    rows = []
    n_missing_data = 0
    for n_done, (eid, ev) in enumerate(ordered_events):
        if n_done % 25 == 0:
            print(f"  progress: {n_done}/{len(ordered_events)} events", flush=True)
        for sta, p_offset in ev["picks"].items():
            pick_time = ev["origin"] + p_offset
            d = date(pick_time.year, pick_time.month, pick_time.day)
            date_iso = d.isoformat()
            traces = caches[sta].get(date_iso)
            if traces == "MISSING" or traces is None or traces.get("HHZ") is None:
                n_missing_data += 1
                continue
            tr = traces["HHZ"]
            sr = tr.stats.sampling_rate
            if pick_time < tr.stats.starttime or pick_time > tr.stats.endtime:
                n_missing_data += 1
                continue
            pick_idx = int(round((pick_time - tr.stats.starttime) * sr))
            z = tr.data.astype(np.float64)
            # detrend locally (large slow drift dominates otherwise, per deej_waveform_common)
            lo = max(0, pick_idx + int(round(NOISE_WINDOW[0] * sr)))
            hi = min(len(z), pick_idx + int(round(SIGNAL_LOOKAHEAD * sr)) + 1)
            if hi - lo < 5:
                n_missing_data += 1
                continue
            local = z[lo:hi]
            trend = np.polyfit(np.arange(len(local)), local, 1)
            local_detrended = local - np.polyval(trend, np.arange(len(local)))
            local_pick_idx = pick_idx - lo

            polarity, snr = first_motion_polarity(
                local_detrended, local_pick_idx, sr, SIGNAL_LOOKAHEAD)
            rows.append({
                "id": eid, "station": sta, "polarity": polarity, "snr": snr,
                "reliable": bool(polarity != 0),
            })
            caches[sta].evict_before(date_iso)

    df = pd.DataFrame(rows)
    print(f"total event-station P picks attempted: {len(events) * len(STATIONS)} "
          f"(union across events' actual station coverage: {sum(len(e['picks']) for e in events.values())})")
    print(f"missing/out-of-coverage waveform data: {n_missing_data}")
    print(f"polarity rows written: {len(df)}")
    print(df.groupby("station")["polarity"].apply(lambda s: (s != 0).sum()).rename("n_reliable"))
    print(df.groupby("station")["reliable"].mean().rename("frac_reliable"))
    print(f"\noverall polarity counts: up={ (df['polarity']==1).sum() }, "
          f"down={ (df['polarity']==-1).sum() }, indeterminate={ (df['polarity']==0).sum() }")

    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
