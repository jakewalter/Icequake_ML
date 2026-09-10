"""Filesystem-only coverage index over the continuous day-volume archive.

Deliberately avoids obspy here: an obspy headonly read costs ~0.4s/file and
there are ~784 days x 14 stations x 3 channels =~ 33k candidate files, which
would take hours just to check existence. Existence is checked purely via
os.path.exists on the exact expected filename; real start/end/gap handling is
deferred to Stage 3, which needs to load the data anyway.
"""

import os
from datetime import date, timedelta

import config


def _doy_string(d):
    return f"{d.timetuple().tm_yday:03d}"


def _day_dirs():
    for name in sorted(os.listdir(config.DAY_VOLUMES_ROOT)):
        if len(name) == 8 and name.isdigit():
            yield name


def _date_from_dirname(name):
    return date(int(name[0:4]), int(name[4:6]), int(name[6:8]))


def channel_path(day_dir, station, channel):
    d = _date_from_dirname(day_dir)
    filename = f"{station}.{config.NETWORK}..{channel}.{d.year}.{_doy_string(d)}"
    return os.path.join(config.DAY_VOLUMES_ROOT, day_dir, filename)


def build_day_file_index():
    """Returns a dict keyed by (station, date_iso) -> {channel: path_or_None}."""
    index = {}
    for day_dir in _day_dirs():
        d = _date_from_dirname(day_dir)
        date_iso = d.isoformat()
        for station in config.ALL_STATIONS:
            paths = {}
            for channel in config.CHANNELS:
                p = channel_path(day_dir, station, channel)
                paths[channel] = p if os.path.exists(p) else None
            index[(station, date_iso)] = paths
    return index


def has_all_channels(index, station, date_iso):
    entry = index.get((station, date_iso))
    if entry is None:
        return False
    return all(entry.get(ch) is not None for ch in config.CHANNELS)


def adjacent_date_iso(date_iso, delta_days):
    d = date.fromisoformat(date_iso)
    return (d + timedelta(days=delta_days)).isoformat()


def load_day_file_index_csv(path):
    """Reload the CSV written by 02_index_continuous_data.py back into the
    same {(station, date_iso): {channel: path_or_None}} shape build_day_file_index
    returns, so Stage 3 doesn't need to re-scan the filesystem."""
    import csv as _csv

    index = {}
    with open(path) as f:
        for row in _csv.DictReader(f):
            index[(row["station"], row["date"])] = {
                "HHZ": row["path_hhz"] or None,
                "HH1": row["path_hh1"] or None,
                "HH2": row["path_hh2"] or None,
            }
    return index
