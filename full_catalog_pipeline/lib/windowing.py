"""Window extraction with a rolling per-station day-file cache and a
2-day-merge fallback for events near UTC midnight.

Each day-channel file is read from disk at most once per station: dates are
processed in ascending order and a dict cache holds only the [prev, current,
next] days needed at any point, evicted as the cursor advances.
"""

from datetime import date, timedelta

import numpy as np
from obspy import Stream, read

import config

SENTINEL_MISSING = "MISSING"


def _channel_paths(day_index, station, date_iso):
    entry = day_index.get((station, date_iso))
    if entry is None:
        return None
    return entry


def load_day_traces(day_index, station, date_iso):
    """Returns {"HHZ": Trace|None, "HH1": Trace|None, "HH2": Trace|None} or
    SENTINEL_MISSING if there's no index entry at all for this station/date."""
    paths = _channel_paths(day_index, station, date_iso)
    if paths is None:
        return SENTINEL_MISSING

    traces = {}
    for channel in config.CHANNELS:
        path = paths.get(channel)
        if not path:
            traces[channel] = None
            continue
        try:
            st = read(path)
            st.merge(method=1, fill_value=None)
            traces[channel] = st[0] if len(st) == 1 else None
        except Exception:
            traces[channel] = None
    return traces


class RollingDayCache:
    """Per-station cache of loaded day traces, bounded to a sliding window."""

    def __init__(self, day_index, station):
        self.day_index = day_index
        self.station = station
        self.cache = {}

    def get(self, date_iso):
        if date_iso not in self.cache:
            self.cache[date_iso] = load_day_traces(self.day_index, self.station, date_iso)
        return self.cache[date_iso]

    def ensure_window(self, date_iso):
        d = date.fromisoformat(date_iso)
        prev_iso = (d - timedelta(days=1)).isoformat()
        next_iso = (d + timedelta(days=1)).isoformat()
        self.get(prev_iso)
        self.get(date_iso)
        self.get(next_iso)
        return prev_iso, next_iso

    def evict_before(self, date_iso):
        cutoff = date.fromisoformat(date_iso)
        for key in list(self.cache.keys()):
            if date.fromisoformat(key) < cutoff:
                del self.cache[key]


def _get_channel_trace(day_traces, channel):
    if day_traces is None or day_traces == SENTINEL_MISSING:
        return None
    return day_traces.get(channel)


def _slice_with_fallback(cache, primary_date, prev_date, next_date, channel, window_start, window_end):
    """Returns (data: np.ndarray|None, status: str)."""
    primary = _get_channel_trace(cache.get(primary_date), channel)
    if primary is None:
        return None, "missing_channel_at_extract"

    if primary.stats.starttime <= window_start and primary.stats.endtime >= window_end:
        sliced = primary.slice(window_start, window_end)
        return sliced.data, "ok"

    need_prev = window_start < primary.stats.starttime
    need_next = window_end > primary.stats.endtime
    neighbor_date = prev_date if need_prev else (next_date if need_next else None)
    if neighbor_date is None:
        return None, "insufficient_coverage"

    neighbor = _get_channel_trace(cache.get(neighbor_date), channel)
    if neighbor is None:
        return None, "insufficient_coverage"

    ordered = [neighbor, primary] if need_prev else [primary, neighbor]
    merged_stream = Stream([t.copy() for t in ordered])
    try:
        merged_stream.merge(method=1, fill_value=None)
    except Exception:
        return None, "merge_error"
    if len(merged_stream) != 1:
        return None, "insufficient_coverage"

    merged = merged_stream[0]
    if merged.stats.starttime > window_start or merged.stats.endtime < window_end:
        return None, "insufficient_coverage"

    sliced = merged.slice(window_start, window_end)
    return sliced.data, "ok"


def extract_window(cache, row, window_start, window_end):
    """row needs 'primary_date'; returns (channel_data: dict|None, status: str)."""
    d = date.fromisoformat(row["primary_date"])
    prev_date = (d - timedelta(days=1)).isoformat()
    next_date = (d + timedelta(days=1)).isoformat()

    channel_data = {}
    statuses = []
    for channel in config.CHANNELS:
        data, status = _slice_with_fallback(
            cache, row["primary_date"], prev_date, next_date, channel, window_start, window_end
        )
        channel_data[channel] = data
        statuses.append(status)

    if any(s != "ok" for s in statuses):
        first_bad = next(s for s in statuses if s != "ok")
        return None, first_bad

    npts = {ch: len(channel_data[ch]) for ch in config.CHANNELS}
    masked = any(np.ma.is_masked(channel_data[ch]) for ch in config.CHANNELS)
    npts_values = list(npts.values())
    npts_consistent = (max(npts_values) - min(npts_values)) <= 1

    complete = (not masked) and npts_consistent

    def _clean(data):
        # Masked (gappy) windows are never used downstream (complete=False),
        # but must still be filled to a real value before storage.
        filled = np.ma.filled(data, 0.0) if np.ma.is_masked(data) else data
        return np.asarray(filled, dtype=np.float32)

    return {
        "Z": _clean(channel_data["HHZ"]),
        "N": _clean(channel_data["HH2"]),
        "E": _clean(channel_data["HH1"]),
        "npts": npts,
        "complete": complete,
        "masked": masked,
    }, "ok"
