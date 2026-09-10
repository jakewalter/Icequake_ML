"""Shared loading/cleaning/CC-solve helpers for the single-station S-P/waveform diagnostics
(plot_cluster_sp_cascade.py, plot_cluster_deej_stack_check.py, plot_cluster_deej_examples.py,
plot_cluster_deej_cc_alignment.py). Originally built for T1 cluster3/DEEJ (see
[[t1-sp-vs-depth-vpvs-check-result]]) but parameterized so any event-id list / station /
hypoDD run directory can be checked the same way -- nothing here is cluster3- or
DEEJ-specific.

All functions take explicit file paths / station / network rather than reading module-level
constants, so callers (typically an argparse CLI in each plotting script) decide which run,
cluster, and station to inspect.
"""
import argparse
import os

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from scipy.signal import butter, detrend, sosfiltfilt

import config

_T1_HYPODD = "full_catalog_pipeline/artifacts/full_run/T1_v5/hypodd"

# Defaults reproduce the original T1 cluster3/DEEJ investigation
# (see [[t1-sp-vs-depth-vpvs-check-result]]); every default is overridable via CLI flags
# so the same scripts work for any other cluster/station/hypoDD run.
DEFAULTS = dict(
    ids_file=f"{_T1_HYPODD}/cluster3_event_ids.txt",
    phase_dat=f"{_T1_HYPODD}/input_files/phase.dat",
    dtcc=f"{_T1_HYPODD}/input_files/dt.cc",
    reloc=f"{_T1_HYPODD}/cluster3_svd_hypoDD.reloc",
    src=f"{_T1_HYPODD}/cluster3_svd_hypoDD.src",
    out_dir=_T1_HYPODD,
    tag="cluster3_deej",
    station="DEEJ",
    network="7U",
    vp=3.85,
    vs=2.22,
    window_pre=0.2,
    window_post=1.4,
)


def build_common_argparser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--ids-file", default=DEFAULTS["ids_file"],
                   help="Text file of event ids (one per line) defining the cluster.")
    p.add_argument("--phase-dat", default=DEFAULTS["phase_dat"], help="hypoDD phase.dat path.")
    p.add_argument("--dtcc", default=DEFAULTS["dtcc"], help="hypoDD dt.cc path.")
    p.add_argument("--reloc", default=DEFAULTS["reloc"],
                   help="hypoDD.reloc path providing each event's depth.")
    p.add_argument("--src", default=DEFAULTS["src"],
                   help="hypoDD.src path providing each event's epicentral distance to --station.")
    p.add_argument("--out-dir", default=DEFAULTS["out_dir"], help="Directory to write figures/CSVs.")
    p.add_argument("--tag", default=DEFAULTS["tag"],
                   help="Label used in output filenames and figure titles (e.g. 'cluster8_TJTJ').")
    p.add_argument("--station-sel", default=None,
                   help="hypoDD station.sel; used to compute epicentral distance geodetically "
                        "when --src does not cover these events (a full run's hypoDD.src holds "
                        "only the last cluster hypoDD processed).")
    p.add_argument("--station", default=DEFAULTS["station"], help="Station code (no network prefix).")
    p.add_argument("--network", default=DEFAULTS["network"], help="Network code.")
    p.add_argument("--vp", type=float, default=DEFAULTS["vp"],
                   help="Accepted P velocity (km/s) for this cluster's depth range (not tested).")
    p.add_argument("--vs", type=float, default=DEFAULTS["vs"],
                   help="Accepted S velocity (km/s) for this cluster's depth range (not tested).")
    p.add_argument("--window-pre", type=float, default=DEFAULTS["window_pre"],
                   help="Seconds before the P pick to include in each extracted window.")
    p.add_argument("--window-post", type=float, default=DEFAULTS["window_post"],
                   help="Seconds after the P pick to include in each extracted window.")
    return p


def load_ids(ids_file):
    return set(int(x) for x in open(ids_file))


def load_events(ids, phase_dat, station, network="7U"):
    """Per event: origin time + <station> P/S offsets (seconds relative to origin), read
    from a hypoDD-format phase.dat. Only events with BOTH a P and an S pick at `station`
    are returned."""
    sta_tag = f"{network}.{station}"
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
                    events[cur] = {"id": cur, "origin": origin}
            elif cur is not None:
                parts = line.split()
                if len(parts) != 4:
                    continue
                sta, tt, wt, ph = parts
                if sta == sta_tag:
                    if ph == "P":
                        events[cur]["p_offset"] = float(tt)
                    elif ph == "S":
                        events[cur]["s_offset"] = float(tt)
    rows = [v for v in events.values() if "p_offset" in v and "s_offset" in v]
    return pd.DataFrame(rows)


def load_dist(ids, src_file, station, network="7U"):
    """Epicentral distance (km) from each event to `station`, from hypoDD's own
    source-parameter file (hypoDD.src, written by partials_1dsr.f) -- purely geodetic,
    independent of hypoDD's depth (unlike the take-off-angle field in the same file, which
    IS a function of assumed depth and would be circular for a real-vs-artifact depth
    check)."""
    sta_tag = f"{network}.{station}"
    rows = []
    with open(src_file) as f:
        for line in f:
            parts = line.split()
            if parts[3] != sta_tag or int(parts[0]) not in ids:
                continue
            rows.append({"id": int(parts[0]), "dist_km": float(parts[5])})
    return pd.DataFrame(rows).drop_duplicates("id")


RELOC_COLS = [
    "id", "lat", "lon", "depth", "x", "y", "z", "ex", "ey", "ez",
    "yr", "mo", "dy", "hr", "mi", "sc", "mag", "cid",
]


def load_reloc(reloc_file):
    """hypoDD.reloc reader. Handles both the 18-field SVD-rerun format (no
    NCCP/NCCS/NCTP/NCTS/RCC/RCT -- see [[t1-sp-vs-depth-vpvs-check-result]]) and the
    standard 24-field format; depth (field 4) is in the same place either way."""
    with open(reloc_file) as f:
        n_fields = len(f.readline().split())
    cols = RELOC_COLS if n_fields == 18 else RELOC_COLS[:17] + [
        "nccp", "nccs", "nctp", "ncts", "rcc", "rct", "cid"]
    return pd.read_csv(reloc_file, sep=r"\s+", header=None, names=cols)[["id", "depth"]]


def load_dist_geodetic(reloc_file, station_sel, station, network="7U"):
    """Epicentral distance (km) computed from the relocation's own lat/lon against the
    station's, in Antarctic Polar Stereographic.

    Same quantity load_dist() reads out of hypoDD.src, and equally independent of hypoDD's
    DEPTH (which is what matters for a real-vs-artifact depth check -- the take-off-angle
    field in hypoDD.src is a function of assumed depth and would be circular). Needed because
    a full run's hypoDD.src is NOT a catalog-wide file: hypoDD rewrites it per cluster, so the
    copy left in output_files/ holds only the last cluster it processed (2 events, for T2).
    Only the per-cluster SVD reruns leave a usable hypoDD.src behind.
    """
    import pyproj
    cols = RELOC_COLS if len(open(reloc_file).readline().split()) == 18 else \
        RELOC_COLS[:17] + ["nccp", "nccs", "nctp", "ncts", "rcc", "rct", "cid"]
    rel = pd.read_csv(reloc_file, sep=r"\s+", header=None, names=cols)[["id", "lat", "lon"]]
    sta = pd.read_csv(station_sel, sep=r"\s+", header=None,
                      names=["sta", "lat", "lon", "elev"])
    sta["code"] = sta["sta"].str.split(".").str[-1]
    row = sta[sta["code"] == station]
    if row.empty:
        raise SystemExit(f"station {station} not found in {station_sel}")
    row = row.iloc[0]
    to_ps = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3031", always_xy=True)
    sx, sy = to_ps.transform(row["lon"], row["lat"])
    ex, ey = to_ps.transform(rel["lon"].values, rel["lat"].values)
    rel["dist_km"] = np.hypot(ex - sx, ey - sy) / 1000.0
    return rel[["id", "dist_km"]]


def load_merged(ids, phase_dat, reloc_file, station, network="7U", src_file=None,
                station_sel=None):
    events = load_events(ids, phase_dat, station, network=network)
    reloc = load_reloc(reloc_file)
    df = events.merge(reloc, on="id")
    dist = None
    if src_file is not None and os.path.exists(src_file):
        dist = load_dist(ids, src_file, station, network=network)
        if dist.empty or not dist["id"].isin(df["id"]).any():
            dist = None       # a per-cluster src that does not cover these events
    if dist is None and station_sel is not None:
        dist = load_dist_geodetic(reloc_file, station_sel, station, network=network)
    if dist is not None:
        df = df.merge(dist, on="id")
    return df.sort_values("depth").reset_index(drop=True)


_HIGHPASS_SOS = butter(
    config.SNR_HIGHPASS_CORNERS, config.SNR_HIGHPASS_FREQ,
    btype="highpass", fs=config.SAMPLE_RATE_HZ, output="sos")


def clean(x):
    """Linear detrend + demean (removes DC offset), then a zero-phase highpass at this
    pipeline's existing SNR-window corner/order (config.SNR_HIGHPASS_FREQ/CORNERS) to remove
    long-period drift a short detrend window doesn't fully capture -- these traces commonly
    carry a large slow drift that otherwise swamps per-trace amplitude normalization, the P/S
    wiggle amplitude on top of it is tiny by comparison. No true instrument-response removal
    is possible here -- this deployment's StationXML (see hypodd_relocate.py's
    build_station_xml) carries only lat/lon/elevation, no poles-and-zeros -- so this highpass
    is the practical substitute already established elsewhere in the pipeline.
    sosfiltfilt is zero-phase (forward+backward): a phase-induced time shift would corrupt
    the P/S pick-timing comparisons these plots exist to make."""
    x = detrend(np.asarray(x, dtype=np.float64), type="linear")
    x = x - x.mean()
    if len(x) <= 3 * len(_HIGHPASS_SOS):
        return x  # too short for filtfilt's default padding; detrended is the best available
    return sosfiltfilt(_HIGHPASS_SOS, x)


def norm_signed(x):
    x = clean(x)
    peak = np.max(np.abs(x)) if np.max(np.abs(x)) > 0 else 1.0
    return x / peak


def horizontal_envelope(n, e):
    n_clean, e_clean = clean(n), clean(e)
    h = np.sqrt(n_clean ** 2 + e_clean ** 2)
    return h / (h.max() if h.max() > 0 else 1.0)


def cc_lag_signed(trace, stack, sr, max_lag_s):
    """Lag (in samples) of maximum SIGNED cross-correlation of `trace` against `stack`,
    searched only within +/-max_lag_s -- never picks the lag that best-fits after a sign
    flip, by construction (np.dot does not flip sign; we simply never consider abs())."""
    max_lag = int(round(max_lag_s * sr))
    best_lag, best_val = 0, -np.inf
    n = len(stack)
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = trace[lag:], stack[:n - lag]
        else:
            a, b = trace[:n + lag], stack[-lag:]
        if len(a) < n * 0.5:
            continue
        val = np.dot(a, b)
        if val > best_val:
            best_val, best_lag = val, lag
    return best_lag


def shift(x, lag):
    out = np.zeros_like(x)
    if lag > 0:
        out[lag:] = x[:len(x) - lag]
    elif lag < 0:
        out[:len(x) + lag] = x[-lag:]
    else:
        out[:] = x
    return out


def predicted_sp(depth_km, dist_km, vp, vs):
    """S-P predicted from a hypoDD depth + epicentral distance via a straight-ray,
    single-layer-velocity model (matches hypoDD's own partials_1dsr.f convention).
    vp/vs are ACCEPTED inputs (the layer velocity for this depth range), not solved for."""
    slope_vertical = 1 / vs - 1 / vp
    return slope_vertical * np.sqrt(np.asarray(depth_km) ** 2 + np.asarray(dist_km) ** 2)


def load_dtcc_pairs(ids, dtcc_file, station, phase, network="7U"):
    """All dt.cc pairs for `station`/`phase` where both events are in `ids`. Returns
    (id_a, id_b, dt, weight) rows -- dt is the CC-refined differential *travel time*
    (i.e. pick-relative-to-own-origin difference, same convention as phase.dat), not an
    absolute-epoch-time difference (events can be months apart)."""
    sta_tag_variants = (station, f"{network}.{station}")
    rows = []
    cur = None
    with open(dtcc_file) as f:
        for line in f:
            if line.startswith("#"):
                p = line.split()
                e1, e2 = int(p[1]), int(p[2])
                cur = (e1, e2) if (e1 in ids and e2 in ids) else None
            elif cur is not None:
                p = line.split()
                if len(p) < 4:
                    continue
                sta, dt, wt, ph = p[0], float(p[1]), float(p[2]), p[3]
                if sta in sta_tag_variants and ph == phase:
                    rows.append((cur[0], cur[1], dt, wt))
    return rows


def solve_cc_consistent_soffset(events_df, dtcc_pairs, anchor_weight=0.05):
    """Self-consistent per-event S-offset (S pick time relative to the event's own origin)
    from a graph of pairwise CC-refined differential times, via weighted least squares:
    minimize sum_pairs wt*(c_i - c_j - dt_ij)^2 + sum_events anchor_weight^2*(c_i - s_offset_i)^2

    The per-event anchor term both fixes the gauge (the pairwise-only system is invariant to
    adding a constant to every c_i) and keeps the solution close to the original catalog pick
    where CC links are sparse/absent. Note: because every event has its OWN anchor term, this
    cannot detect or correct a bias shared by ALL picks alike -- it refines relative
    (event-to-event) timing using the CC graph, not a systematic catalog-wide offset.

    Returns a pandas Series of solved S-offsets indexed by event id, restricted to events
    present in `events_df`.
    """
    events = events_df.set_index("id")
    event_ids = sorted(events.index)
    idx = {eid: i for i, eid in enumerate(event_ids)}
    n = len(event_ids)

    rows_pair = [(idx[e1], idx[e2], dt, wt) for e1, e2, dt, wt in dtcc_pairs
                 if e1 in idx and e2 in idx]

    A = np.zeros((len(rows_pair) + n, n))
    b = np.zeros(len(rows_pair) + n)
    for k, (i, j, dt, wt) in enumerate(rows_pair):
        w = np.sqrt(wt)
        A[k, i], A[k, j], b[k] = w, -w, w * dt
    for i, eid in enumerate(event_ids):
        A[len(rows_pair) + i, i] = anchor_weight
        b[len(rows_pair) + i] = anchor_weight * events.loc[eid, "s_offset"]

    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return pd.Series(sol, index=event_ids, name="s_offset_cc")
