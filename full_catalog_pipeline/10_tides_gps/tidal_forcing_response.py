#!/usr/bin/env python3
"""Does icequake rate track the MODELLED ocean tide, constituent by constituent?

Uses CATS2008_v2023 (2 km circum-Antarctic, includes sub-ice-shelf cavities and a
grounding-line flexure field). Supersedes the astronomy-only test in
tidal_modulation_test.py by supplying the actual local forcing SPECTRUM and a real tide
height time series.

THE ARRAYS ARE NOT IN THE MODEL DOMAIN. Both sit on mask = 0 (grounded ice): the nearest wet
cell is 209 km away for T1 and 140 km for T2. CATS2008 therefore returns nothing at the array
coordinates, and direct flexure is irrelevant at that range (elastic decay length ~10-20 km;
the model's own flexure field is 0 everywhere near the arrays). What is computed here is the
forcing at the nearest wet cell -- i.e. the grounding-zone tide -- which reaches the arrays
only through an ice-stream-scale response with unknown gain and lag. Amplitudes and phases
below are the FORCING, not the local stress.

THE KEY MEASUREMENT the model unlocks: the local tide is mixed-mainly-diurnal (form factor
F = (K1+O1)/(M2+S2) = 2.7), and M2 is the WEAKEST semidiurnal constituent (~5 cm against K1's
34 cm). So the seismicity's diurnal-yes / M2-no pattern can be tested against forcing
amplitude rather than attributed to an ice-stream filter.

CAVEAT ON S2 -- READ BEFORE INTERPRETING IT. S2 is at exactly 12.0000 h, identical to the
second harmonic of the solar thermal cycle. tidal_modulation_test.py regresses out 24/12/8/6 h
to kill the (large, real) 24 h thermal signal, which unavoidably removes S2 as well. S2's
post-removal p-value is therefore meaningless. The pre-removal values are reported here
instead, and S2 is excluded from the amplitude-vs-response fit.

Usage:
    python full_catalog_pipeline/tidal_forcing_response.py
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

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from tidal_modulation_test import (load_times, hourly_series, within_day_shuffle,
                                   style, INK, MUTED, GRID, SURF, C1, C2)

CATS = "CATS2008_v2023.nc"
CON_ORDER = "m2 s2 n2 k2 k1 o1 p1 q1 mf mm".split()
SEMI, DIUR = ["m2", "s2", "n2", "k2"], ["k1", "o1", "p1", "q1"]
ARRAYS = {"T1": (-77.299, -100.476), "T2": (-76.570, -103.284)}
N_NULL = 500


def nearest_wet(ds, lat0, lon0):
    lat, lon, mask = ds["lat"][:], ds["lon"][:], ds["mask"][:]
    dist = np.hypot((lat - lat0) * 111.0,
                    (lon - lon0) * 111.0 * np.cos(np.radians(lat0)))
    dry = dist[np.unravel_index(np.argmin(dist), dist.shape)]
    dw = np.where(mask > 0, dist, np.inf)
    k = np.unravel_index(np.argmin(dw), dw.shape)
    return k, float(dw[k]), float(lat[k]), float(lon[k]), float(dist[np.unravel_index(np.argmin(dist), dist.shape)])


def harmonics(ds, k):
    """Complex height coefficients (m) for all 10 constituents at grid cell k."""
    hRe = np.asarray(ds["hRe"][:, k[0], k[1]], float)
    hIm = np.asarray(ds["hIm"][:, k[0], k[1]], float)
    omega = np.asarray(ds["omega"][:], float)          # rad/s
    return hRe + 1j * hIm, omega


def predict_tide(times, h, omega, epoch=pd.Timestamp("1992-01-01")):
    """Harmonic synthesis. Nodal corrections from pyTMD where available.

    Nodal modulation matters over a 2-year record (the 18.6 yr cycle changes K1/O1 amplitude
    by up to ~11%/19%), so it is applied rather than ignored.
    """
    t = (times - epoch).total_seconds().values if hasattr(times, "total_seconds") else \
        (pd.DatetimeIndex(times) - epoch).total_seconds().values
    try:
        import pyTMD.arguments as PA
        mjd = (pd.DatetimeIndex(times) - pd.Timestamp("1858-11-17")).total_seconds() / 86400.0
        pu, pf, G = PA.arguments(np.asarray(mjd), CON_ORDER, corrections="OTIS")
        z = np.zeros(len(t))
        for i in range(len(h)):
            th = omega[i] * t + np.radians(G[:, i]) + pu[:, i]
            z += pf[:, i] * (h[i].real * np.cos(th) - h[i].imag * np.sin(th))
        return z, True
    except Exception:
        z = np.zeros(len(t))
        for i in range(len(h)):
            th = omega[i] * t
            z += h[i].real * np.cos(th) - h[i].imag * np.sin(th)
        return z, False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    R = "full_catalog_pipeline/artifacts/full_run"
    ap.add_argument("--t1", default=f"{R}/T1_v5/hypodd_vels1d_t1ccstrong/output_files/hypoDD.reloc")
    ap.add_argument("--t2", default=f"{R}/T2_v5/hypodd_vels1d_ccstrong/output_files/hypoDD.reloc")
    ap.add_argument("--out-dir", default=f"{R}/T2_v5")
    args = ap.parse_args()

    ds = nc.Dataset(CATS)
    cats = {}
    for name, (la, lo) in ARRAYS.items():
        k, dkm, wlat, wlon, dry = nearest_wet(ds, la, lo)
        h, omega = harmonics(ds, k)
        cats[name] = dict(k=k, dist_km=dkm, wlat=wlat, wlon=wlon, h=h, omega=omega,
                          amp_cm=np.abs(h) * 100)
        print(f"{name}: on mask=0 (grounded). Nearest WET cell {dkm:.0f} km away "
              f"at ({wlat:.3f}, {wlon:.3f})")
    print()

    t1, t2 = load_times(args.t1), load_times(args.t2)
    lo_t = max(t1.min(), t2.min()); hi_t = min(t1.max(), t2.max())
    t1 = t1[(t1 >= lo_t) & (t1 <= hi_t)].reset_index(drop=True)
    t2 = t2[(t2 >= lo_t) & (t2 <= hi_t)].reset_index(drop=True)
    rng = np.random.default_rng(0)

    out = []
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for col, (name, t) in enumerate((("T1", t1), ("T2", t2))):
        c = C1 if name == "T1" else C2
        info = cats[name]
        hours, counts = hourly_series(t, lo_t, hi_t)
        z, nodal = predict_tide(hours, info["h"], info["omega"])
        print(f"=== {name}: {len(t)} events, tide synthesised at the grounding-zone cell "
              f"(nodal corrections: {'yes' if nodal else 'NO'}) ===")
        print(f"    tide range {z.min()*100:.0f} .. {z.max()*100:.0f} cm, "
              f"std {z.std()*100:.1f} cm")

        # rate vs tide height, and vs rate-of-change of tide
        dz = np.gradient(z) * 100.0        # cm/hr
        for lbl, drv in (("tide height", z * 100), ("d(tide)/dt", dz)):
            r_obs = pearsonr(counts, drv)[0]
            nulls = np.empty(N_NULL)
            for i in range(N_NULL):
                ts = within_day_shuffle(t, rng).sort_values().reset_index(drop=True)
                nulls[i] = pearsonr(hourly_series(ts, lo_t, hi_t)[1], drv)[0]
            p = float((np.abs(nulls) >= abs(r_obs)).mean())
            print(f"    hourly rate vs {lbl:12s}: r = {r_obs:+.4f}   shuffle p = {p:.3f}")
            out.append(dict(array=name, test=lbl, r=r_obs, p=p))

        # amplitude vs response: does the seismic response track forcing amplitude?
        ax = axes[0, col]
        ax.bar(range(8), info["amp_cm"][:8], color=[c if n in DIUR else MUTED
                                                    for n in CON_ORDER[:8]], zorder=3)
        ax.set_xticks(range(8))
        ax.set_xticklabels([n.upper() for n in CON_ORDER[:8]], fontsize=9)
        ax.set_ylabel("CATS2008 amplitude (cm)", fontsize=9.5, color=MUTED)
        ax.set_title(f"{name}: local tidal forcing spectrum\n"
                     f"grounding-zone cell {info['dist_km']:.0f} km from the array · "
                     f"coloured = diurnal", fontsize=11.5, color=INK, loc="left")
        style(ax)

        # tide time series sample + event rate
        ax = axes[1, col]
        m = slice(0, 24 * 30)
        ax.plot(hours[m], z[m] * 100, color=MUTED, lw=1.2, zorder=3, label="modelled tide")
        ax2 = ax.twinx()
        print("    (the tide/rate overlay below uses a second axis for VISUAL alignment only;"
              " every number reported is from the correlation above)")
        ax2.bar(hours[m], counts[m], width=0.04, color=c, alpha=0.65, zorder=2)
        ax2.set_ylabel("events/hour", fontsize=9.5, color=c)
        ax.set_ylabel("tide (cm)", fontsize=9.5, color=MUTED)
        ax.set_title(f"{name}: first 30 days — modelled tide and event rate",
                     fontsize=11.5, color=INK, loc="left")
        style(ax)
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        print()

    fig.suptitle("Modelled ocean tide (CATS2008_v2023) vs icequake rate — "
                 "forcing sampled 140–209 km from the arrays",
                 fontsize=13, color=INK, y=0.975)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(args.out_dir, "tidal_forcing_response.png")
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=SURF)
    print(f"wrote {p}")

    amp = pd.DataFrame({n: cats[n]["amp_cm"] for n in cats}, index=CON_ORDER)
    amp.to_csv(os.path.join(args.out_dir, "cats2008_local_amplitudes.csv"))
    pd.DataFrame(out).to_csv(os.path.join(args.out_dir, "tidal_forcing_response.csv"),
                             index=False)
    print(f"wrote {os.path.join(args.out_dir, 'cats2008_local_amplitudes.csv')}")


if __name__ == "__main__":
    main()
